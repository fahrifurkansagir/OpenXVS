# OpenXVS Live Deck — Zero-Dependency Local Avionics Bridge

`tools/livedeck_bridge/bridge.py` is a pure Python 3.14 standard-library RFC 6455 binary WebSocket server (`asyncio`, `socket`, `hashlib`, `base64`, `ssl`) that bridges live FAA GDL-90 and MAVLink v2 packets emitted by `OpenXVS Live Deck` (`xvs-wasm`) to local avionics / SITL UDP ports.

## Quick Start (Local Development)

```powershell
python tools/livedeck_bridge/bridge.py
```

By default:
- WebSocket binds **strictly to `127.0.0.1:8765`** (`ws://127.0.0.1:8765`).
- Non-loopback browser `Origin` headers are rejected with **HTTP `403 Forbidden`**.
- Unmasked client frames are rejected with RFC 6455 close code **`1002`**.
- Binary frame protocol:
  - Prefix `0x01` + framed GDL-90 bytes → UDP `127.0.0.1:4000`
  - Prefix `0x02` + MAVLink v2 bytes → UDP `127.0.0.1:14550`

## TLS (`wss://127.0.0.1`) & Browser Security Caveats

Python's standard library does **not** generate X.509 certificates. If you run `OpenXVS Live Deck` from a **hosted HTTPS origin** (e.g. Cloudflare Pages or Hugging Face Spaces) rather than `http://127.0.0.1`, browsers enforce strict mixed-content and private-network rules:

- **Chrome / Edge (Local-Network-Access / Private Network Access):** Chrome allows `ws://127.0.0.1` from some secure contexts only after prompting for **Local-Network-Access** permission; using `wss://127.0.0.1` with a locally trusted certificate avoids mixed-content warnings.
- **Safari / WebKit:** Safari **blocks** `ws://127.0.0.1` from `https://` pages as mixed content. You MUST either run Live Deck locally on `http://127.0.0.1` OR supply a locally trusted TLS certificate (for example via [`mkcert`](https://github.com/FiloSottile/mkcert)) and pass `--tls-cert`, `--tls-key`, and `--allow-origin`:

```powershell
mkcert -install
mkcert 127.0.0.1 localhost
python tools/livedeck_bridge/bridge.py `
  --tls-cert 127.0.0.1+1.pem `
  --tls-key 127.0.0.1+1-key.pem `
  --allow-origin https://your-hosted-livedeck.example
```

## Opt-In ForeFlight LAN Discovery & Mandatory `SIM` Safety Guard

LAN ForeFlight discovery (UDP `63093`) is **OFF by default** so packets never leave `127.0.0.1` unless explicitly enabled:

```powershell
python tools/livedeck_bridge/bridge.py --foreflight-discover
```

When `--foreflight-discover` is active, **every GDL-90 `#20` (`0x14`) Traffic Report** is validated, forced to carry a **`"SIM"`** callsign prefix (`SIM...`, 8 ASCII chars), and re-framed with a freshly computed FAA GDL-90 CRC-16-CCITT (`0x1021`) before any datagram is sent to a discovered LAN peer — ensuring simulated OpenXVS traffic can never be mistaken for live airspace.
