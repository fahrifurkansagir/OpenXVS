"""Generate OpenXVS procedural traffic and calibration models (WP02 §2.1, §3 G1).

Exports .glb binary models and .json metadata sidecars to assets/traffic/.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.traffic import models

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def generate_all_models(out_dir: str | Path | None = None, seeds: list[int] | None = None) -> list[dict[str, Any]]:
    base_dir = Path(out_dir) if out_dir else (REPO_ROOT / "assets" / "traffic")
    base_dir.mkdir(parents=True, exist_ok=True)

    seeds_to_run = seeds or [1, 2, 3]

    items: list[tuple[str, str, Any]] = [
        # Calibration models
        ("CALIBRATION", "CAL_SPHERES", models.build_cal_spheres()),
        ("CALIBRATION", "CAL", models.build_cal()),
        ("CALIBRATION", "SPHERE_R2", models.build_sphere_r2()),
        ("CALIBRATION", "PANEL", models.build_panel()),
    ]

    for s in seeds_to_run:
        items.extend([
            ("C1_UAS", f"C1_UAS_{s:04d}", models.build_c1_uas(seed=s)),
            ("C2_SEP", f"C2_SEP_{s:04d}", models.build_c2_sep(seed=s)),
            ("C3_TWIN", f"C3_TWIN_{s:04d}", models.build_c3_twin(seed=s)),
            ("C4_BIZJET", f"C4_BIZJET_{s:04d}", models.build_c4_bizjet(seed=s)),
            ("C5_TRANSPORT", f"C5_TRANSPORT_{s:04d}", models.build_c5_transport(seed=s)),
            ("C6_HELI", f"C6_HELI_{s:04d}", models.build_c6_heli(seed=s)),
        ])

    manifest = []
    print(f"{'Model ID':<18} {'Class':<14} {'Triangles':<10} {'Span (m)':<10} {'Len (m)':<10} {'GLB SHA-256':<64}")
    print("-" * 130)

    for cls_name, model_id, model_res in items:
        glb_p = base_dir / cls_name / f"{model_id}.glb"
        sha = models.export_glb(model_res, glb_p)

        meta = {
            "schema": "xvs.traffic_model/1",
            "model_id": model_id,
            "class": cls_name,
            "seed": model_res.seed,
            "dimensions": model_res.dimensions_m,
            "bbox_min": model_res.bbox_min,
            "bbox_max": model_res.bbox_max,
            "origin": [0.0, 0.0, 0.0],
            "triangle_count": model_res.triangle_count,
            "alpha_trim_deg": getattr(model_res, "alpha_trim_deg", 0.0),
            "glb_sha256": sha,
            "path": str(glb_p.relative_to(REPO_ROOT)).replace("\\", "/"),
        }
        json_p = base_dir / cls_name / f"{model_id}.json"
        with open(json_p, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        manifest.append(meta)

        span = model_res.dimensions_m.get("span_m", 0.0)  # default-ok: optional fallback default
        length = model_res.dimensions_m.get("length_m", 0.0)  # default-ok: optional fallback default
        print(f"{model_id:<18} {cls_name:<14} {model_res.triangle_count:<10} {span:<10.2f} {length:<10.2f} {sha:<64}")

    # Write manifest summary
    manifest_p = base_dir / "models_manifest.json"
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSuccessfully generated and exported {len(items)} models to {base_dir}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate OpenXVS procedural models")
    parser.add_argument("--out-dir", default=None, help="Output directory for assets")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="Seeds for class models")
    args = parser.parse_args()

    generate_all_models(args.out_dir, args.seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
