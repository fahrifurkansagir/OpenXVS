"""CLI launcher for OpenXVS Studio v4.0 live 5/5 real-model inference server (`http://127.0.0.1:8765`).

Usage:
  python scripts/xvs/studio_live_server.py --host 127.0.0.1 --port 8765 --device auto
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xvs.v2.studio_bridge import (  # noqa: E402
    DEFAULT_VRAM_CAP_MB,
    StudioInferenceEngine,
    create_studio_http_server,
    load_studio_models,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenXVS Studio v4.0 Live 5/5 Model Inference Server")
    p.add_argument("--host", type=str, default="127.0.0.1", help="Bind address (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=8765, help="Bind TCP port (default 8765)")
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Compute device preference (default auto)",
    )
    p.add_argument(
        "--vram-cap-mb",
        type=float,
        default=DEFAULT_VRAM_CAP_MB,
        help="Strict peak VRAM cap in MiB on 4 GB GPUs (default 2800.0)",
    )
    p.add_argument(
        "--smoke-fast",
        action="store_true",
        help="Skip extracting/loading heavy Colab L4 checkpoints (for CI smoke tests)",
    )
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Load all models, print JSON health summary, and exit without binding a socket",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bundle = load_studio_models(
        device_pref=args.device,
        vram_cap_mb=float(args.vram_cap_mb),
        smoke_fast=bool(args.smoke_fast),
    )
    engine = StudioInferenceEngine(bundle)
    summary = bundle.health_summary()
    summary["media_pack_ready"] = engine.media_pack_ready()
    print(json.dumps(summary, indent=2), flush=True)
    if args.check_only:
        return 0

    server = create_studio_http_server(engine, host=args.host, port=int(args.port))
    print(
        f"[OpenXVS Studio Bridge] Listening on http://{args.host}:{args.port} "
        f"(compute={bundle.compute_device}, gpu={bundle.host_gpu_name}, vram={summary['vram_mb']} MiB)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
