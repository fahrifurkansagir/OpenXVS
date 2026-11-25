"""Benchmark transfer methods between CesiumJS page and Python (N11).

Measures:
1. Base64 RGBA (G3 baseline)
2. In-page RGB-only packing (Alternative 1)
3. In-page lossless PNG (Alternative 2)

Tests bit-exactness on 10 rgb and 10 mask_earth scenes and records timings.
"""

from __future__ import annotations
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import socket
import sys
import threading
import time
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.rig.rig import Rig
from xvs.sim.server import XVSServer
from xvs.sim.renderer import XVSRenderer


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_benchmark():
    port = get_free_port()
    cesium_build = REPO_ROOT / "sim" / "cesium" / "node_modules" / "cesium" / "Build" / "Cesium"
    page_dir = REPO_ROOT / "sim" / "cesium" / "page"
    cache_dir = REPO_ROOT / "data" / "cache" / "usgs_imagery"
    log_dir = REPO_ROOT / "output" / "xvs" / "test_logs"

    server = XVSServer(("127.0.0.1", port), cesium_build, page_dir, cache_dir, log_dir, offline=True)
    t_server = threading.Thread(target=server.serve_forever, daemon=True)
    t_server.start()

    rig = Rig.load_nominal().make_true(20260922)
    renderer = XVSRenderer(port=port, backend="ref", server=server).start()

    out_dir = REPO_ROOT / "output" / "xvs" / "wp01" / "g4"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 10 poses across different altitudes and attitudes
    poses = []
    for i in range(10):
        lat = 37.0 + i * 0.05
        lon = -122.0 - i * 0.05
        h = 500.0 + i * 500.0
        yaw = math.radians(i * 36.0)
        pitch = math.radians(-10.0 + i * 2.0)
        roll = math.radians(-5.0 + i * 1.0)
        poses.append((lat, lon, h, yaw, pitch, roll))

    results = {
        "rgba_base64": {"rgb_ms": [], "mask_ms": [], "bit_exact": True},
        "rgb_packed": {"rgb_ms": [], "mask_ms": [], "bit_exact": True},
        "png_in_page": {"rgb_ms": [], "mask_ms": [], "bit_exact": True},
    }

    try:
        for profile in ("rgb", "mask_earth"):
            print(f"Benchmarking profile '{profile}' (10 images)...")
            renderer.set_profile(profile)
            page = renderer.get_page()

            for idx, (lat, lon, h, yaw, pitch, roll) in enumerate(poses):
                cams = rig.compute_poses(lat, lon, h, yaw, pitch, roll)
                cam_f = cams["F"]
                renderer.set_camera(
                    position=cam_f["position"],
                    direction=cam_f["direction"],
                    up=cam_f["up"],
                    fov_rad=cam_f["fov_rad"],
                    aspect_ratio=cam_f["aspect_ratio"],
                )
                # Render to stability
                renderer.render_frame()

                # Method 1: RGBA Base64 (Baseline)
                t0 = time.perf_counter()
                b64_rgba = page.evaluate("window.readXVSPixelsBase64()")
                raw_rgba = base64.b64decode(b64_rgba)
                arr_rgba = np.frombuffer(raw_rgba, dtype=np.uint8).reshape(1024, 1024, 4)
                rgb_base = np.flipud(arr_rgba[:, :, :3]).copy()
                ms_rgba = (time.perf_counter() - t0) * 1000.0
                sha_base = hashlib.sha256(np.ascontiguousarray(rgb_base).tobytes()).hexdigest()

                if profile == "rgb":
                    results["rgba_base64"]["rgb_ms"].append(ms_rgba)
                else:
                    results["rgba_base64"]["mask_ms"].append(ms_rgba)

                # Method 2: RGB Packed Base64 (Alternative 1)
                t0 = time.perf_counter()
                b64_rgb = page.evaluate("window.readXVSPixelsRGB()")
                raw_rgb = base64.b64decode(b64_rgb)
                arr_rgb = np.frombuffer(raw_rgb, dtype=np.uint8).reshape(1024, 1024, 3)
                rgb_alt1 = np.flipud(arr_rgb).copy()
                ms_rgb = (time.perf_counter() - t0) * 1000.0
                sha_alt1 = hashlib.sha256(np.ascontiguousarray(rgb_alt1).tobytes()).hexdigest()

                if profile == "rgb":
                    results["rgb_packed"]["rgb_ms"].append(ms_rgb)
                else:
                    results["rgb_packed"]["mask_ms"].append(ms_rgb)

                if sha_alt1 != sha_base:
                    results["rgb_packed"]["bit_exact"] = False
                    print(f"  [MISMATCH] Alt1 mismatch on {profile} {idx}: {sha_alt1} vs {sha_base}")

                # Method 3: PNG in Page (Alternative 2)
                t0 = time.perf_counter()
                b64_png = page.evaluate("window.readXVSPixelsPNG()")
                raw_png = base64.b64decode(b64_png)
                im = Image.open(io.BytesIO(raw_png)).convert("RGB")
                rgb_alt2 = np.array(im)
                ms_png = (time.perf_counter() - t0) * 1000.0
                sha_alt2 = hashlib.sha256(np.ascontiguousarray(rgb_alt2).tobytes()).hexdigest()

                if profile == "rgb":
                    results["png_in_page"]["rgb_ms"].append(ms_png)
                else:
                    results["png_in_page"]["mask_ms"].append(ms_png)

                if sha_alt2 != sha_base:
                    results["png_in_page"]["bit_exact"] = False
                    print(f"  [MISMATCH] Alt2 mismatch on {profile} {idx}: {sha_alt2} vs {sha_base}")

    finally:
        renderer.close()
        server.shutdown()

    # Calculate summary metrics
    summary = {}
    for m, data in results.items():
        all_ms = data["rgb_ms"] + data["mask_ms"]
        summary[m] = {
            "bit_exact": data["bit_exact"],
            "median_ms": float(np.median(all_ms)),
            "p95_ms": float(np.percentile(all_ms, 95)),
            "mean_ms": float(np.mean(all_ms)),
            "min_ms": float(np.min(all_ms)),
            "max_ms": float(np.max(all_ms)),
            "rgb_median_ms": float(np.median(data["rgb_ms"])),
            "mask_median_ms": float(np.median(data["mask_ms"])),
        }

    benchmark_path = out_dir / "transfer_benchmark.json"
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "raw_timings": results}, f, indent=2)

    print(f"\nSaved transfer benchmark to {benchmark_path}")
    print("\nBenchmark Summary:")
    print(f"{'Method':<16} | {'Bit-Exact':<10} | {'Median ms':<10} | {'p95 ms':<10} | {'Mean ms':<10}")
    print("-" * 65)
    for m, s in summary.items():
        print(f"{m:<16} | {str(s['bit_exact']):<10} | {s['median_ms']:<10.1f} | {s['p95_ms']:<10.1f} | {s['mean_ms']:<10.1f}")

    return summary


if __name__ == "__main__":
    run_benchmark()
