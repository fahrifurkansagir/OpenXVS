"""Zero-dependency local RFC 6455 WebSocket <-> GDL-90 / MAVLink v2 UDP bridge.

Pure Python 3.14 standard library only (zero third-party dependencies).

Security & Safety Locks:
  - Binds strictly to `127.0.0.1` (default port 8765).
  - Rejects non-loopback `Origin` headers with HTTP 403 Forbidden.
  - Enforces RFC 6455 client frame masking (closes unmasked client frames with WS 1002).
  - Handles fragmented binary frames, interleaved Ping/Pong, and abrupt TCP disconnects cleanly.
  - Binary protocol:
      0x01 + <GDL-90 bytes>     -> UDP 127.0.0.1:<gdl90_port> (default 4000)
      0x02 + <MAVLink v2 bytes> -> UDP 127.0.0.1:<mavlink_port> (default 14550)
  - Optional TLS via user-supplied PEM files (`--tls-cert` + `--tls-key`).
  - Opt-in LAN ForeFlight discovery (`--foreflight-discover`, OFF by default) listens on UDP 63093
    and unconditionally enforces a `"SIM"` callsign prefix + recalculated CRC-CCITT 0x1021 on
    every GDL-90 #20 (0x14) Traffic Report before any LAN send.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import socket
import ssl
import struct
import time
from typing import Any
import urllib.parse

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
MAX_HTTP_HEADER_BYTES = 16 * 1024
MAX_WS_MESSAGE_BYTES = 1024 * 1024

GDL90_FLAG_BYTE = 0x7E
GDL90_ESCAPE_BYTE = 0x7D
GDL90_STUFF_XOR = 0x20
GDL90_CRC_POLY = 0x1021
GDL90_MSG_ID_TRAFFIC_REPORT = 0x14

LOGGER = logging.getLogger("openxvs.livedeck_bridge")


def _build_gdl90_crc16_table() -> tuple[int, ...]:
    table: list[int] = []
    for i in range(256):
        crc = (i << 8) & 0xFFFF
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ GDL90_CRC_POLY) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
        table.append(crc)
    return tuple(table)


_GDL90_CRC16_TABLE = _build_gdl90_crc16_table()


def compute_gdl90_crc16(data: bytes | bytearray) -> int:
    """Compute FAA GDL-90 ICD §2.2.2 CRC-16-CCITT (poly 0x1021, init 0x0000)."""
    crc = 0x0000
    for b in data:
        crc = _GDL90_CRC16_TABLE[(crc >> 8) & 0xFF] ^ ((crc << 8) & 0xFFFF) ^ int(b)
    return crc & 0xFFFF


def _unstuff_gdl90(stuffed: bytes) -> bytes:
    out = bytearray()
    i = 0
    n = len(stuffed)
    while i < n:
        b = stuffed[i]
        if b == GDL90_ESCAPE_BYTE:
            if i + 1 >= n:
                raise ValueError("Truncated GDL-90 escape sequence")
            out.append(stuffed[i + 1] ^ GDL90_STUFF_XOR)
            i += 2
        elif b == GDL90_FLAG_BYTE:
            raise ValueError("Unexpected unescaped 0x7E inside GDL-90 frame")
        else:
            out.append(b)
            i += 1
    return bytes(out)


def _stuff_gdl90(raw: bytes) -> bytes:
    out = bytearray()
    for b in raw:
        if b in (GDL90_FLAG_BYTE, GDL90_ESCAPE_BYTE):
            out.append(GDL90_ESCAPE_BYTE)
            out.append(b ^ GDL90_STUFF_XOR)
        else:
            out.append(b)
    return bytes(out)


def enforce_gdl90_sim_callsign(frame: bytes) -> bytes:
    """Ensure any GDL-90 #20 (0x14) Traffic Report carries a 'SIM' callsign prefix + valid CRC-16.

    Non-Traffic GDL-90 frames (#0 Heartbeat, #10 Ownship, #11 Geo Alt) pass through unchanged
    after framing and CRC validation. Raises ValueError on corrupt frames so bad bytes are never
    broadcast onto a LAN.
    """
    if len(frame) < 5 or frame[0] != GDL90_FLAG_BYTE or frame[-1] != GDL90_FLAG_BYTE:
        raise ValueError("Invalid GDL-90 frame delimiters")

    unstuffed = _unstuff_gdl90(frame[1:-1])
    if len(unstuffed) < 3:
        raise ValueError("GDL-90 unstuffed payload too short")

    payload = unstuffed[:-2]
    rx_crc = int(unstuffed[-2]) | (int(unstuffed[-1]) << 8)
    if rx_crc != compute_gdl90_crc16(payload):
        raise ValueError("GDL-90 CRC-16 mismatch before ForeFlight LAN send")

    if payload[0] != GDL90_MSG_ID_TRAFFIC_REPORT or len(payload) != 28:
        return frame

    existing_cs = bytes(payload[19:27]).decode("ascii", errors="ignore").strip().upper()
    if existing_cs.startswith("SIM"):
        new_cs = existing_cs[:8].ljust(8, " ")
    else:
        clean_suffix = "".join(
            ch for ch in existing_cs if ("A" <= ch <= "Z") or ("0" <= ch <= "9")
        )
        new_cs = ("SIM" + clean_suffix)[:8].ljust(8, " ")

    mutated = bytearray(payload)
    mutated[19:27] = new_cs.encode("ascii")
    crc = compute_gdl90_crc16(mutated)
    fcs = bytes([crc & 0xFF, (crc >> 8) & 0xFF])
    return bytes([GDL90_FLAG_BYTE]) + _stuff_gdl90(bytes(mutated) + fcs) + bytes([GDL90_FLAG_BYTE])


def _is_loopback_origin(origin: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(origin.strip())
    except ValueError:
        return False
    host = (parsed.hostname or "").lower().strip()
    return host in LOOPBACK_HOSTS


def _ws_pack_server_frame(opcode: int, payload: bytes) -> bytes:
    b0 = 0x80 | (opcode & 0x0F)
    n = len(payload)
    if n < 126:
        hdr = bytes([b0, n])
    elif n < 65536:
        hdr = struct.pack("!BBH", b0, 126, n)
    else:
        hdr = struct.pack("!BBQ", b0, 127, n)
    return hdr + payload


def _ws_close_frame(code: int, reason: str = "") -> bytes:
    payload = struct.pack("!H", int(code)) + reason.encode("utf-8")[:123]
    return _ws_pack_server_frame(0x8, payload)


class _ForeFlightDiscoveryProtocol(asyncio.DatagramProtocol):
    """Opt-in UDP listener for ForeFlight GDL-90 discovery beacons (default port 63093)."""

    def __init__(self, peers: dict[tuple[str, int], float], peer_ttl_s: float = 30.0) -> None:
        self._peers = peers
        self._peer_ttl_s = float(peer_ttl_s)

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        try:
            doc = json.loads(data.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(doc, dict):
            return
        gdl_info = doc.get("GDL90")
        if not isinstance(gdl_info, dict):
            return
        port_raw = gdl_info.get("port")
        if not isinstance(port_raw, int) or not (1 <= port_raw <= 65535):
            return
        peer_addr = (str(addr[0]), int(port_raw))
        self._peers[peer_addr] = time.monotonic() + self._peer_ttl_s
        LOGGER.info("Registered ForeFlight peer %s:%d (SIM guard active)", peer_addr[0], peer_addr[1])


class LiveDeckBridge:
    """Asyncio RFC 6455 binary WebSocket server forwarding GDL-90 & MAVLink v2 over UDP."""

    def __init__(
        self,
        *,
        ws_port: int = 8765,
        gdl90_port: int = 4000,
        mavlink_port: int = 14550,
        tls_cert: str | None = None,
        tls_key: str | None = None,
        foreflight_discover: bool = False,
        foreflight_discover_port: int = 63093,
        foreflight_bind_host: str = "0.0.0.0",
        allowed_origins: tuple[str, ...] = (),
    ) -> None:
        self._requested_ws_port = int(ws_port)
        self.ws_port: int = int(ws_port)
        self.gdl90_target: tuple[str, int] = ("127.0.0.1", int(gdl90_port))
        self.mavlink_target: tuple[str, int] = ("127.0.0.1", int(mavlink_port))
        self.tls_cert = tls_cert
        self.tls_key = tls_key
        self.foreflight_discover = bool(foreflight_discover)
        self._requested_ff_port = int(foreflight_discover_port)
        self.foreflight_discover_port: int = int(foreflight_discover_port)
        self._ff_bind_host = str(foreflight_bind_host)
        self._extra_allowed_origins = frozenset(o.strip().lower() for o in allowed_origins if o.strip())

        self._server: asyncio.AbstractServer | None = None
        self._udp_tx: socket.socket | None = None
        self._ff_transport: asyncio.DatagramTransport | None = None
        self._ff_peers: dict[tuple[str, int], float] = {}
        self._client_tasks: set[asyncio.Task[Any]] = set()
        self.packets_gdl90: int = 0
        self.packets_mavlink: int = 0
        self.packets_dropped: int = 0

    def _build_ssl_context(self) -> ssl.SSLContext | None:
        if not self.tls_cert and not self.tls_key:
            return None
        if not (self.tls_cert and self.tls_key):
            raise ValueError("Both --tls-cert and --tls-key must be supplied together")
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.load_cert_chain(certfile=self.tls_cert, keyfile=self.tls_key)
        return ctx

    async def start(self) -> None:
        loop = asyncio.get_running_loop()

        self._udp_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_tx.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)
        self._udp_tx.setblocking(False)

        if self.foreflight_discover:
            transport, _ = await loop.create_datagram_endpoint(
                lambda: _ForeFlightDiscoveryProtocol(self._ff_peers),
                local_addr=(self._ff_bind_host, self._requested_ff_port),
            )
            self._ff_transport = transport
            sockname = transport.get_extra_info("sockname")
            if sockname:
                self.foreflight_discover_port = int(sockname[1])

        ssl_ctx = self._build_ssl_context()
        self._server = await asyncio.start_server(
            self._on_client_connected,
            host="127.0.0.1",
            port=self._requested_ws_port,
            ssl=ssl_ctx,
        )
        if self._server.sockets:
            self.ws_port = int(self._server.sockets[0].getsockname()[1])

        scheme = "wss" if ssl_ctx is not None else "ws"
        LOGGER.info(
            "OpenXVS Live Deck bridge listening on %s://127.0.0.1:%d (GDL-90->127.0.0.1:%d, MAVLink->127.0.0.1:%d, ForeFlight=%s)",
            scheme,
            self.ws_port,
            self.gdl90_target[1],
            self.mavlink_target[1],
            "ON" if self.foreflight_discover else "OFF",
        )

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        for task in list(self._client_tasks):
            task.cancel()
        if self._client_tasks:
            await asyncio.gather(*self._client_tasks, return_exceptions=True)
            self._client_tasks.clear()

        if self._ff_transport is not None:
            self._ff_transport.close()
            self._ff_transport = None

        if self._udp_tx is not None:
            self._udp_tx.close()
            self._udp_tx = None

    def _on_client_connected(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.create_task(self._handle_client(reader, writer))
        self._client_tasks.add(task)
        task.add_done_callback(self._client_tasks.discard)

    def _is_origin_allowed(self, origin: str | None) -> bool:
        if origin is None:
            return True
        if _is_loopback_origin(origin):
            return True
        if origin.strip().lower() in self._extra_allowed_origins:
            return True
        return False

    async def _perform_handshake(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> bool:
        try:
            raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10.0)
        except (
            asyncio.TimeoutError,
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
            ConnectionError,
            OSError,
        ):
            return False

        if len(raw) > MAX_HTTP_HEADER_BYTES:
            writer.write(b"HTTP/1.1 431 Request Header Fields Too Large\r\nConnection: close\r\n\r\n")
            await writer.drain()
            return False

        lines = raw.decode("latin1").split("\r\n")
        if not lines or not lines[0].startswith("GET "):
            writer.write(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n")
            await writer.drain()
            return False

        headers: dict[str, str] = {}
        for line in lines[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()

        origin = headers.get("origin")
        if not self._is_origin_allowed(origin):
            writer.write(
                b"HTTP/1.1 403 Forbidden\r\n"
                b"Content-Type: text/plain\r\n"
                b"Connection: close\r\n\r\n"
                b"Forbidden: non-loopback Origin rejected by OpenXVS Live Deck bridge\n"
            )
            await writer.drain()
            return False

        upgrade = headers.get("upgrade", "").lower()
        ws_key = headers.get("sec-websocket-key", "")
        ws_ver = headers.get("sec-websocket-version", "")
        if upgrade != "websocket" or not ws_key or ws_ver != "13":
            writer.write(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n")
            await writer.drain()
            return False

        accept = base64.b64encode(
            hashlib.sha1((ws_key + WS_GUID).encode("ascii")).digest()
        ).decode("ascii")
        resp = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
        ).encode("ascii")
        writer.write(resp)
        await writer.drain()
        return True

    async def _read_ws_frame(
        self, reader: asyncio.StreamReader
    ) -> tuple[bool, int, bool, bytes] | None:
        try:
            hdr = await reader.readexactly(2)
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            return None

        b0, b1 = hdr[0], hdr[1]
        fin = bool(b0 & 0x80)
        rsv = b0 & 0x70
        opcode = b0 & 0x0F
        masked = bool(b1 & 0x80)
        length = b1 & 0x7F

        if rsv != 0:
            raise ValueError("Non-zero RSV bits")

        if length == 126:
            length = struct.unpack("!H", await reader.readexactly(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", await reader.readexactly(8))[0]

        if length > MAX_WS_MESSAGE_BYTES:
            raise OverflowError("WS frame exceeds maximum allowed size")

        if not masked:
            # RFC 6455 §5.1: client-to-server frames MUST be masked
            return fin, opcode, False, b""

        mask_key = await reader.readexactly(4)
        payload = await reader.readexactly(length) if length else b""
        if length:
            unmasked = bytes(b ^ mask_key[i & 3] for i, b in enumerate(payload))
        else:
            unmasked = b""
        return fin, opcode, True, unmasked

    def _forward_binary_message(self, msg: bytes) -> None:
        if len(msg) < 2 or self._udp_tx is None:
            self.packets_dropped += 1
            return

        prefix = msg[0]
        payload = msg[1:]
        if prefix == 0x01:
            self._udp_tx.sendto(payload, self.gdl90_target)
            self.packets_gdl90 += 1

            if self.foreflight_discover and self._ff_peers:
                now = time.monotonic()
                expired = [p for p, exp in self._ff_peers.items() if exp < now]
                for p in expired:
                    del self._ff_peers[p]
                if self._ff_peers:
                    try:
                        safe_lan_frame = enforce_gdl90_sim_callsign(payload)
                    except ValueError:
                        self.packets_dropped += 1
                        return
                    for peer in list(self._ff_peers.keys()):
                        try:
                            self._udp_tx.sendto(safe_lan_frame, peer)
                        except OSError:
                            pass
        elif prefix == 0x02:
            self._udp_tx.sendto(payload, self.mavlink_target)
            self.packets_mavlink += 1
        else:
            self.packets_dropped += 1

    async def _send_close_and_finish(
        self, writer: asyncio.StreamWriter, code: int, reason: str = ""
    ) -> None:
        try:
            writer.write(_ws_close_frame(code, reason))
            await writer.drain()
        except (ConnectionError, OSError):
            pass

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            if not await self._perform_handshake(reader, writer):
                return

            frag_buf: bytearray | None = None
            while True:
                try:
                    frame = await self._read_ws_frame(reader)
                except (asyncio.IncompleteReadError, ConnectionError, OSError):
                    break
                except OverflowError:
                    await self._send_close_and_finish(writer, 1009, "Message too big")
                    break
                except ValueError:
                    await self._send_close_and_finish(writer, 1002, "Protocol error")
                    break

                if frame is None:
                    break

                fin, opcode, masked, data = frame
                if not masked:
                    await self._send_close_and_finish(writer, 1002, "Client frame must be masked")
                    break

                # Control frames (0x8..0xF) MUST NOT be fragmented and MUST be <= 125 B
                if opcode >= 0x8:
                    if not fin or len(data) > 125:
                        await self._send_close_and_finish(writer, 1002, "Invalid control frame")
                        break
                    if opcode == 0x8:  # Close
                        await self._send_close_and_finish(writer, 1000)
                        break
                    if opcode == 0x9:  # Ping -> Pong
                        writer.write(_ws_pack_server_frame(0xA, data))
                        await writer.drain()
                        continue
                    if opcode == 0xA:  # Pong
                        continue
                    await self._send_close_and_finish(writer, 1002, "Unknown control opcode")
                    break

                # Data frames
                if opcode == 0x2:  # Binary start
                    if frag_buf is not None:
                        await self._send_close_and_finish(writer, 1002, "Expected continuation")
                        break
                    if fin:
                        self._forward_binary_message(data)
                    else:
                        frag_buf = bytearray(data)
                elif opcode == 0x0:  # Continuation
                    if frag_buf is None:
                        await self._send_close_and_finish(writer, 1002, "Unexpected continuation")
                        break
                    frag_buf.extend(data)
                    if len(frag_buf) > MAX_WS_MESSAGE_BYTES:
                        await self._send_close_and_finish(writer, 1009, "Message too big")
                        break
                    if fin:
                        self._forward_binary_message(bytes(frag_buf))
                        frag_buf = None
                elif opcode == 0x1:  # Text frame not supported on binary avionics wire
                    await self._send_close_and_finish(writer, 1003, "Binary frames only")
                    break
                else:
                    await self._send_close_and_finish(writer, 1002, "Unsupported opcode")
                    break
        except asyncio.CancelledError:
            raise
        except (ConnectionError, OSError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenXVS Live Deck zero-dependency local WebSocket <-> GDL-90 / MAVLink v2 UDP bridge"
    )
    parser.add_argument("--port", type=int, default=8765, help="Localhost WS bind port (default: 8765)")
    parser.add_argument(
        "--gdl90-port",
        type=int,
        default=4000,
        help="Localhost UDP target port for GDL-90 frames (default: 4000)",
    )
    parser.add_argument(
        "--mavlink-port",
        type=int,
        default=14550,
        help="Localhost UDP target port for MAVLink v2 frames (default: 14550)",
    )
    parser.add_argument(
        "--tls-cert",
        type=str,
        default=None,
        help="Optional path to PEM certificate for wss://127.0.0.1 (e.g. generated with mkcert)",
    )
    parser.add_argument(
        "--tls-key",
        type=str,
        default=None,
        help="Optional path to PEM private key for wss://127.0.0.1",
    )
    parser.add_argument(
        "--foreflight-discover",
        action="store_true",
        default=False,
        help="Opt-in LAN ForeFlight UDP 63093 discovery (OFF by default; enforces SIM callsign prefix on GDL-90 #20)",
    )
    parser.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        help="Optional extra browser Origin to allow (e.g. hosted HTTPS Live Deck URL when paired with --tls-cert)",
    )
    return parser


async def _serve_forever(args: argparse.Namespace) -> None:
    bridge = LiveDeckBridge(
        ws_port=args.port,
        gdl90_port=args.gdl90_port,
        mavlink_port=args.mavlink_port,
        tls_cert=args.tls_cert,
        tls_key=args.tls_key,
        foreflight_discover=args.foreflight_discover,
        allowed_origins=tuple(args.allow_origin),
    )
    await bridge.start()
    try:
        await asyncio.Future()
    finally:
        await bridge.stop()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    try:
        asyncio.run(_serve_forever(args))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
