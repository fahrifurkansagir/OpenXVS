"""TDD tests for Milestone M1 Task 5: Zero-dependency stdlib asyncio GDL-90/MAVLink local bridge."""

from __future__ import annotations

from datetime import datetime, timezone
import ast
import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import sys
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_EVIDENCE_PATH = REPO_ROOT / "output" / "xvs" / "v2_m1" / "v6_bridge_evidence.json"

_BRIDGE_EVIDENCE: dict[str, Any] = {
    "schema": "xvs.v2_m1.v6_bridge_evidence/1",
    "verified_utc": "",
    "total_packets_sent": 0,
    "total_packets_received": 0,
    "gdl90_packets_sent": 0,
    "gdl90_packets_received": 0,
    "mavlink_packets_sent": 0,
    "mavlink_packets_received": 0,
    "drops": -1,
    "crc_errors": -1,
    "sim_prefix_enforced": False,
    "non_loopback_rejected_403": False,
    "unmasked_closed_1002": False,
    "fragmented_reassembly_ok": False,
    "pure_stdlib_verified": False,
    "pass": False,
}


def _persist_bridge_evidence() -> None:
    _BRIDGE_EVIDENCE["verified_utc"] = datetime.now(timezone.utc).isoformat()
    _BRIDGE_EVIDENCE["pass"] = bool(
        _BRIDGE_EVIDENCE["total_packets_sent"] == 500
        and _BRIDGE_EVIDENCE["total_packets_received"] == 500
        and _BRIDGE_EVIDENCE["drops"] == 0
        and _BRIDGE_EVIDENCE["crc_errors"] == 0
        and _BRIDGE_EVIDENCE["sim_prefix_enforced"]
        and _BRIDGE_EVIDENCE["non_loopback_rejected_403"]
        and _BRIDGE_EVIDENCE["unmasked_closed_1002"]
        and _BRIDGE_EVIDENCE["fragmented_reassembly_ok"]
        and _BRIDGE_EVIDENCE["pure_stdlib_verified"]
    )
    BRIDGE_EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BRIDGE_EVIDENCE_PATH.write_text(
        json.dumps(_BRIDGE_EVIDENCE, indent=2) + "\n", encoding="utf-8"
    )
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xvs.avionics.gdl90 import (
    GDL90_MSG_ID_TRAFFIC_REPORT,
    Gdl90TargetReport,
    decode_gdl90_frame,
    decode_gdl90_message,
    encode_heartbeat,
    encode_ownship_geo_altitude,
    encode_target_report,
)
from xvs.avionics.mavlink_adsb import (
    DEFAULT_PASSIVE_ADSB_FLAGS,
    MavlinkAdsbVehicle,
    pack_adsb_vehicle,
    unpack_adsb_vehicle,
)

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _ws_build_frame(
    opcode: int,
    payload: bytes,
    *,
    fin: bool = True,
    masked: bool = True,
    mask_key: bytes = b"\x37\xfa\x21\x3d",
) -> bytes:
    b0 = (0x80 if fin else 0x00) | (opcode & 0x0F)
    n = len(payload)
    mask_bit = 0x80 if masked else 0x00
    header = bytearray([b0])
    if n < 126:
        header.append(mask_bit | n)
    elif n < 65536:
        header.append(mask_bit | 126)
        header.extend(struct.pack("!H", n))
    else:
        header.append(mask_bit | 127)
        header.extend(struct.pack("!Q", n))
    if masked:
        assert len(mask_key) == 4
        header.extend(mask_key)
        masked_payload = bytes(b ^ mask_key[i & 3] for i, b in enumerate(payload))
        return bytes(header) + masked_payload
    return bytes(header) + payload


async def _ws_read_frame(reader: asyncio.StreamReader) -> tuple[bool, int, bytes]:
    hdr = await asyncio.wait_for(reader.readexactly(2), timeout=5.0)
    fin = bool(hdr[0] & 0x80)
    opcode = hdr[0] & 0x0F
    masked = bool(hdr[1] & 0x80)
    length = hdr[1] & 0x7F
    if length == 126:
        length = struct.unpack("!H", await asyncio.wait_for(reader.readexactly(2), timeout=5.0))[0]
    elif length == 127:
        length = struct.unpack("!Q", await asyncio.wait_for(reader.readexactly(8), timeout=5.0))[0]
    mask = await asyncio.wait_for(reader.readexactly(4), timeout=5.0) if masked else None
    data = await asyncio.wait_for(reader.readexactly(length), timeout=5.0) if length else b""
    if mask is not None:
        data = bytes(b ^ mask[i & 3] for i, b in enumerate(data))
    return fin, opcode, data


async def _ws_handshake(
    host: str,
    port: int,
    *,
    origin: str | None = "http://127.0.0.1:5173",
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, str]:
    reader, writer = await asyncio.open_connection(host, port)
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    req_lines = [
        "GET / HTTP/1.1",
        f"Host: {host}:{port}",
        "Upgrade: websocket",
        "Connection: Upgrade",
        f"Sec-WebSocket-Key: {key}",
        "Sec-WebSocket-Version: 13",
    ]
    if origin is not None:
        req_lines.append(f"Origin: {origin}")
    req = ("\r\n".join(req_lines) + "\r\n\r\n").encode("ascii")
    writer.write(req)
    await writer.drain()
    resp = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5.0)
    resp_text = resp.decode("latin1")
    if "101 " in resp_text.splitlines()[0]:
        expected = base64.b64encode(
            hashlib.sha1((key + WS_GUID).encode("ascii")).digest()
        ).decode("ascii")
        assert expected in resp_text
    return reader, writer, resp_text


def _make_udp_listener() -> tuple[socket.socket, int]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
    sock.bind(("127.0.0.1", 0))
    sock.setblocking(False)
    return sock, int(sock.getsockname()[1])


def test_500_gdl90_and_mavlink_loopback_packets() -> None:
    """Streams 250 GDL-90 + 250 MAVLink v2 packets over WS -> 500/500 UDP loopback with 0 drops & 0 CRC errors."""
    from tools.livedeck_bridge.bridge import LiveDeckBridge

    async def _run() -> None:
        gdl_sock, gdl_port = _make_udp_listener()
        mav_sock, mav_port = _make_udp_listener()
        bridge = LiveDeckBridge(
            ws_port=0,
            gdl90_port=gdl_port,
            mavlink_port=mav_port,
        )
        await bridge.start()
        try:
            reader, writer, resp = await _ws_handshake("127.0.0.1", bridge.ws_port)
            assert resp.startswith("HTTP/1.1 101")

            # Build 250 real GDL-90 frames and 250 real MAVLink v2 frames
            gdl_frames: list[bytes] = []
            mav_frames: list[bytes] = []
            for i in range(250):
                if i % 3 == 0:
                    gdl_frames.append(encode_heartbeat(utc_time_s=1000 + i))
                elif i % 3 == 1:
                    gdl_frames.append(encode_ownship_geo_altitude(alt_ft=1500.0 + i * 5.0))
                else:
                    gdl_frames.append(
                        encode_target_report(
                            Gdl90TargetReport(
                                msg_id=GDL90_MSG_ID_TRAFFIC_REPORT,
                                alert_status=1 if (i % 2 == 0) else 0,
                                address_type=0,
                                participant_address=0xF00000 + i,
                                lat_deg=37.6117 + i * 1e-4,
                                lon_deg=-122.3583 - i * 1e-4,
                                altitude_ft=1800.0 + i * 25.0,
                                misc_indicators=0x09,
                                nic=8,
                                nacp=8,
                                horiz_vel_kt=140.0,
                                vert_vel_fpm=-256.0,
                                track_heading_deg=(i * 3.0) % 360.0,
                                emitter_category=14,
                                callsign=f"XVS{i:04d}",
                            )
                        )
                    )
                mav_frames.append(
                    pack_adsb_vehicle(
                        MavlinkAdsbVehicle(
                            icao_address=0xF00100 + i,
                            lat_deg_e7=int((37.61 + i * 1e-4) * 1e7),
                            lon_deg_e7=int((-122.36 - i * 1e-4) * 1e7),
                            altitude_mm=500_000 + i * 1000,
                            heading_cdeg=(i * 100) % 36000,
                            hor_velocity_cms=7500,
                            ver_velocity_cms=-120,
                            flags=DEFAULT_PASSIVE_ADSB_FLAGS,
                            squawk=7700 if (i % 5 == 0) else 1200,
                            altitude_type=1,
                            callsign=f"XVS{i:04d}",
                            emitter_type=14,
                            tslc=0,
                            seq=i & 0xFF,
                        )
                    )
                )

            for i in range(250):
                writer.write(_ws_build_frame(0x2, b"\x01" + gdl_frames[i]))
                writer.write(_ws_build_frame(0x2, b"\x02" + mav_frames[i]))
                if i % 25 == 24:
                    await writer.drain()
            await writer.drain()

            # Collect all 500 packets with bounded timeout
            rx_gdl: list[bytes] = []
            rx_mav: list[bytes] = []
            deadline = asyncio.get_running_loop().time() + 5.0
            while (len(rx_gdl) < 250 or len(rx_mav) < 250) and (
                asyncio.get_running_loop().time() < deadline
            ):
                while True:
                    try:
                        data, _ = gdl_sock.recvfrom(4096)
                        rx_gdl.append(data)
                    except BlockingIOError:
                        break
                while True:
                    try:
                        data, _ = mav_sock.recvfrom(4096)
                        rx_mav.append(data)
                    except BlockingIOError:
                        break
                if len(rx_gdl) < 250 or len(rx_mav) < 250:
                    await asyncio.sleep(0.01)

            assert len(rx_gdl) == 250, f"Expected 250 GDL-90 packets, got {len(rx_gdl)}"
            assert len(rx_mav) == 250, f"Expected 250 MAVLink packets, got {len(rx_mav)}"

            crc_errors = 0
            for idx, pkt in enumerate(rx_gdl):
                assert pkt == gdl_frames[idx]
                decode_gdl90_message(pkt)  # Verifies CRC-16-CCITT & unstuffing

            for idx, pkt in enumerate(rx_mav):
                assert pkt == mav_frames[idx]
                decoded = unpack_adsb_vehicle(pkt)  # Verifies X.25 CRC & payload
                assert decoded.seq == (idx & 0xFF)

            _BRIDGE_EVIDENCE["gdl90_packets_sent"] = len(gdl_frames)
            _BRIDGE_EVIDENCE["gdl90_packets_received"] = len(rx_gdl)
            _BRIDGE_EVIDENCE["mavlink_packets_sent"] = len(mav_frames)
            _BRIDGE_EVIDENCE["mavlink_packets_received"] = len(rx_mav)
            _BRIDGE_EVIDENCE["total_packets_sent"] = len(gdl_frames) + len(mav_frames)
            _BRIDGE_EVIDENCE["total_packets_received"] = len(rx_gdl) + len(rx_mav)
            _BRIDGE_EVIDENCE["drops"] = (len(gdl_frames) + len(mav_frames)) - (
                len(rx_gdl) + len(rx_mav)
            )
            _BRIDGE_EVIDENCE["crc_errors"] = crc_errors
            _persist_bridge_evidence()

            writer.write(_ws_build_frame(0x8, struct.pack("!H", 1000)))
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        finally:
            await bridge.stop()
            gdl_sock.close()
            mav_sock.close()

    asyncio.run(_run())


def test_rejects_non_loopback_origin_and_unmasked_frames() -> None:
    """Rejects Origin: https://evil.example with HTTP 403 and closes unmasked frames with WS 1002."""
    from tools.livedeck_bridge.bridge import LiveDeckBridge

    async def _run() -> None:
        bridge = LiveDeckBridge(ws_port=0, gdl90_port=4000, mavlink_port=14550)
        await bridge.start()
        try:
            # 1. Non-loopback origin -> HTTP 403 Forbidden
            r_bad, w_bad, resp_bad = await _ws_handshake(
                "127.0.0.1", bridge.ws_port, origin="https://evil.example"
            )
            assert resp_bad.startswith("HTTP/1.1 403"), f"Expected 403, got: {resp_bad}"
            w_bad.close()
            await w_bad.wait_closed()

            # 2. Unmasked client frame -> WS close 1002 (Protocol Error)
            r_ok, w_ok, resp_ok = await _ws_handshake(
                "127.0.0.1", bridge.ws_port, origin="http://localhost:5173"
            )
            assert resp_ok.startswith("HTTP/1.1 101")
            w_ok.write(_ws_build_frame(0x2, b"\x01" + encode_heartbeat(1), masked=False))
            await w_ok.drain()
            fin, opcode, payload = await _ws_read_frame(r_ok)
            assert fin is True
            assert opcode == 0x8
            assert len(payload) >= 2
            close_code = struct.unpack("!H", payload[:2])[0]
            assert close_code == 1002, f"Expected WS close 1002, got {close_code}"
            _BRIDGE_EVIDENCE["non_loopback_rejected_403"] = True
            _BRIDGE_EVIDENCE["unmasked_closed_1002"] = True
            _persist_bridge_evidence()
            w_ok.close()
            await w_ok.wait_closed()
        finally:
            await bridge.stop()

    asyncio.run(_run())


def test_handles_fragmented_binary_frames_and_abrupt_disconnect() -> None:
    """Reassembles multi-frame RFC 6455 binary messages (with interleaved ping) and survives abrupt TCP close."""
    from tools.livedeck_bridge.bridge import LiveDeckBridge

    async def _run() -> None:
        gdl_sock, gdl_port = _make_udp_listener()
        bridge = LiveDeckBridge(ws_port=0, gdl90_port=gdl_port, mavlink_port=14550)
        await bridge.start()
        try:
            # 1. Fragmented binary message + interleaved Ping
            reader, writer, resp = await _ws_handshake("127.0.0.1", bridge.ws_port)
            assert resp.startswith("HTTP/1.1 101")

            raw_gdl = encode_target_report(
                Gdl90TargetReport(
                    msg_id=GDL90_MSG_ID_TRAFFIC_REPORT,
                    alert_status=1,
                    address_type=0,
                    participant_address=0xABCDEF,
                    lat_deg=37.62,
                    lon_deg=-122.37,
                    altitude_ft=2500.0,
                    misc_indicators=0x09,
                    nic=9,
                    nacp=9,
                    horiz_vel_kt=120.0,
                    vert_vel_fpm=0.0,
                    track_heading_deg=280.0,
                    emitter_category=1,
                    callsign="XVS0077",
                )
            )
            full_msg = b"\x01" + raw_gdl
            c1 = full_msg[:7]
            c2 = full_msg[7:19]
            c3 = full_msg[19:]

            writer.write(_ws_build_frame(0x2, c1, fin=False))
            writer.write(_ws_build_frame(0x9, b"ping42", fin=True))
            writer.write(_ws_build_frame(0x0, c2, fin=False))
            writer.write(_ws_build_frame(0x0, c3, fin=True))
            await writer.drain()

            # Server should reply Pong for interleaved Ping
            fin, op, pong_data = await _ws_read_frame(reader)
            assert fin is True and op == 0xA and pong_data == b"ping42"

            # Reassembled GDL-90 packet arrives intact on UDP
            deadline = asyncio.get_running_loop().time() + 3.0
            rx = None
            while rx is None and asyncio.get_running_loop().time() < deadline:
                try:
                    rx, _ = gdl_sock.recvfrom(4096)
                except BlockingIOError:
                    await asyncio.sleep(0.01)
            assert rx == raw_gdl
            decoded = decode_gdl90_message(rx)
            assert isinstance(decoded, Gdl90TargetReport) and decoded.callsign == "XVS0077"

            # 2. Abrupt disconnect mid-frame must not crash server or leak active clients
            writer.write(b"\x82\xfe\x00\x20\x11\x22")  # Truncated frame
            await writer.drain()
            writer.transport.abort()
            await asyncio.sleep(0.05)

            # 3. Subsequent client works cleanly
            r2, w2, resp2 = await _ws_handshake("127.0.0.1", bridge.ws_port)
            assert resp2.startswith("HTTP/1.1 101")
            hb = encode_heartbeat(utc_time_s=42)
            w2.write(_ws_build_frame(0x2, b"\x01" + hb))
            await w2.drain()

            deadline2 = asyncio.get_running_loop().time() + 3.0
            rx2 = None
            while rx2 is None and asyncio.get_running_loop().time() < deadline2:
                try:
                    rx2, _ = gdl_sock.recvfrom(4096)
                except BlockingIOError:
                    await asyncio.sleep(0.01)
            assert rx2 == hb
            _BRIDGE_EVIDENCE["fragmented_reassembly_ok"] = True
            _persist_bridge_evidence()
            w2.close()
            await w2.wait_closed()
        finally:
            await bridge.stop()
            gdl_sock.close()

    asyncio.run(_run())


def test_foreflight_lan_guard_enforces_sim_callsign_prefix() -> None:
    """Opt-in ForeFlight LAN forwarding rewrites GDL-90 #20 callsigns to 'SIM*' with valid CRC-16."""
    from tools.livedeck_bridge.bridge import LiveDeckBridge, enforce_gdl90_sim_callsign

    # Direct unit test of pure-stdlib SIM callsign rewriter + CRC-CCITT recalculation
    orig_frame = encode_target_report(
        Gdl90TargetReport(
            msg_id=GDL90_MSG_ID_TRAFFIC_REPORT,
            alert_status=1,
            address_type=0,
            participant_address=0x123456,
            lat_deg=37.6117,
            lon_deg=-122.3583,
            altitude_ft=1500.0,
            misc_indicators=0x09,
            nic=8,
            nacp=8,
            horiz_vel_kt=110.0,
            vert_vel_fpm=0.0,
            track_heading_deg=298.0,
            emitter_category=1,
            callsign="N12345",
        )
    )
    rewritten = enforce_gdl90_sim_callsign(orig_frame)
    decoded = decode_gdl90_message(rewritten)
    assert isinstance(decoded, Gdl90TargetReport)
    assert decoded.callsign.startswith("SIM")
    assert decoded.participant_address == 0x123456

    # Heartbeat (#0) passes unchanged with valid CRC
    hb = encode_heartbeat(utc_time_s=1234)
    assert enforce_gdl90_sim_callsign(hb) == hb

    # End-to-end ForeFlight discovery beacon -> LAN peer receives SIM-prefixed GDL-90 #20
    async def _run() -> None:
        local_gdl_sock, local_gdl_port = _make_udp_listener()
        ff_peer_sock, ff_peer_port = _make_udp_listener()
        bridge = LiveDeckBridge(
            ws_port=0,
            gdl90_port=local_gdl_port,
            mavlink_port=14550,
            foreflight_discover=True,
            foreflight_discover_port=0,
            foreflight_bind_host="127.0.0.1",
        )
        await bridge.start()
        try:
            assert bridge.foreflight_discover_port > 0
            # Send ForeFlight discovery JSON beacon
            beacon_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            beacon = json.dumps({"App": "ForeFlight", "GDL90": {"port": ff_peer_port}}).encode(
                "utf-8"
            )
            beacon_sock.sendto(beacon, ("127.0.0.1", bridge.foreflight_discover_port))
            beacon_sock.close()
            await asyncio.sleep(0.05)

            reader, writer, resp = await _ws_handshake("127.0.0.1", bridge.ws_port)
            assert resp.startswith("HTTP/1.1 101")
            writer.write(_ws_build_frame(0x2, b"\x01" + orig_frame))
            await writer.drain()

            deadline = asyncio.get_running_loop().time() + 3.0
            rx_ff = None
            while rx_ff is None and asyncio.get_running_loop().time() < deadline:
                try:
                    rx_ff, _ = ff_peer_sock.recvfrom(4096)
                except BlockingIOError:
                    await asyncio.sleep(0.01)
            assert rx_ff is not None, "ForeFlight peer did not receive forwarded GDL-90 frame"
            ff_decoded = decode_gdl90_message(rx_ff)
            assert isinstance(ff_decoded, Gdl90TargetReport)
            assert ff_decoded.callsign.startswith("SIM")
            # Confirm CRC is valid via decode_gdl90_frame as well
            assert decode_gdl90_frame(rx_ff)[0] == GDL90_MSG_ID_TRAFFIC_REPORT
            _BRIDGE_EVIDENCE["sim_prefix_enforced"] = True
            _persist_bridge_evidence()

            writer.close()
            await writer.wait_closed()
        finally:
            await bridge.stop()
            local_gdl_sock.close()
            ff_peer_sock.close()

    asyncio.run(_run())


def test_bridge_is_pure_stdlib_and_readme_documents_caveats() -> None:
    """bridge.py imports ONLY Python stdlib modules; CLI defaults & README caveats are explicit."""
    bridge_path = REPO_ROOT / "tools" / "livedeck_bridge" / "bridge.py"
    readme_path = REPO_ROOT / "tools" / "livedeck_bridge" / "README.md"
    assert bridge_path.is_file()
    assert readme_path.is_file()

    tree = ast.parse(bridge_path.read_text(encoding="utf-8"))
    stdlib = sys.stdlib_module_names
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0]
                assert root_mod in stdlib, f"Non-stdlib import in bridge.py: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_mod = node.module.split(".")[0]
                assert root_mod in stdlib, f"Non-stdlib from-import in bridge.py: {node.module}"

    from tools.livedeck_bridge.bridge import build_arg_parser

    args = build_arg_parser().parse_args([])
    assert args.port == 8765
    assert args.gdl90_port == 4000
    assert args.mavlink_port == 14550
    assert args.foreflight_discover is False
    assert args.tls_cert is None
    assert args.tls_key is None

    readme = readme_path.read_text(encoding="utf-8")
    assert "mkcert" in readme
    assert "Local-Network-Access" in readme or "Local Network Access" in readme
    assert "Safari" in readme
    assert "--foreflight-discover" in readme
    assert "SIM" in readme
    _BRIDGE_EVIDENCE["pure_stdlib_verified"] = True
    _persist_bridge_evidence()
