"""Offline/dry-run manifest & deterministic Colab G4 gsplat 3DGUT+2DGS job builder (`XVS-World`)."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .config import (
    REQUIRED_GT_TIER,
    WorldTwinConfig,
    estimate_gsplat_vram_gb,
)


def build_gsplat_command(
    cfg: WorldTwinConfig,
    data_dir: str | Path,
    result_dir: str | Path,
    max_steps: int = 30_000,
) -> list[str]:
    """Build deterministic gsplat v1.5+ 3DGUT+2DGS training CLI invocation for Colab G4."""
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    return [
        "python",
        "-m",
        "gsplat.examples.simple_trainer",
        "mcmc",
        "--data_dir",
        str(data_dir),
        "--result_dir",
        str(result_dir),
        "--max_steps",
        str(int(max_steps)),
        "--cap_max",
        str(cfg.max_splats),
        "--sh_degree",
        str(cfg.sh_degree),
        "--packed",
        "--with_ut",
        "--with_eval3d",
        "--depth_loss",
        "--normal_loss",
        "--seed",
        str(cfg.seed),
    ]


def validate_frame_metadata_visual_stress_only(meta: dict[str, Any]) -> None:
    """CI isolation guard: every XVS-World rendered frame MUST carry VISUAL_STRESS_ONLY stamp."""
    if "gt_tier" not in meta:
        raise ValueError("Missing required 'gt_tier' stamp on XVS-World frame metadata")
    if meta["gt_tier"] != REQUIRED_GT_TIER:
        raise ValueError(
            f"Frame metadata gt_tier={meta['gt_tier']!r} rejected; must be {REQUIRED_GT_TIER!r}"
        )
    if "wp01_certification_eligible" not in meta or meta["wp01_certification_eligible"] is not False:
        raise ValueError("XVS-World frame metadata must explicitly set wp01_certification_eligible=False")
    if "gcp_reproj_uncertainty_m" not in meta:
        raise ValueError("XVS-World frame metadata must record 'gcp_reproj_uncertainty_m'")
    unc = float(meta["gcp_reproj_uncertainty_m"])
    if not (unc >= 0.0 and unc == unc):
        raise ValueError(f"Invalid gcp_reproj_uncertainty_m: {unc}")


def build_colab_g4_job_spec(
    cfg: WorldTwinConfig,
    data_dir: str | Path = "/content/xvs_world_ksfo_28l/colmap",
    result_dir: str | Path = "/content/xvs_world_ksfo_28l/out",
    max_steps: int = 30_000,
) -> dict[str, Any]:
    """Build deterministic offline Colab G4 job specification (zero local GPU execution required)."""
    est_vram_gb = estimate_gsplat_vram_gb(cfg.max_splats)
    cmd = build_gsplat_command(cfg, data_dir=data_dir, result_dir=result_dir, max_steps=max_steps)
    spec: dict[str, Any] = {
        "schema": "xvs.v2_m2.world_twin_job_spec/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DRY_RUN_PENDING_COLAB_G4",
        "target_hardware": "Colab G4 NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM)",
        "estimated_vram_gb": round(est_vram_gb, 3),
        "fits_single_g4_96gb": bool(est_vram_gb < 90.0),
        "gt_tier": REQUIRED_GT_TIER,
        "wp01_certification_eligible": False,
        "config_sha256": cfg.config_sha256(),
        "config": cfg.to_dict(),
        "gsplat_command": cmd,
        "export_contract": {
            "format": "3D_Tiles_1.1_KHR_gaussian_splatting_compression_spz_2",
            "max_spz_gb": cfg.spz_max_gb,
            "target_webgpu_fps_min": 30,
            "required_gcp_rmse_lt_m": 0.30,
            "sample_frame_metadata_template": {
                "site_icao": cfg.site_icao,
                "runway_id": cfg.runway_id,
                "gt_tier": REQUIRED_GT_TIER,
                "wp01_certification_eligible": False,
                "gcp_reproj_uncertainty_m": 0.25,
            },
        },
    }
    validate_frame_metadata_visual_stress_only(
        spec["export_contract"]["sample_frame_metadata_template"]
    )
    return spec


def write_dry_run_manifest(
    cfg: WorldTwinConfig,
    output_path: str | Path,
    *,
    data_dir: str | Path = "/content/xvs_world_ksfo_28l/colmap",
    result_dir: str | Path = "/content/xvs_world_ksfo_28l/out",
    max_steps: int = 30_000,
) -> dict[str, Any]:
    """Write deterministic offline dry-run job manifest JSON to disk."""
    spec = build_colab_g4_job_spec(
        cfg, data_dir=data_dir, result_dir=result_dir, max_steps=max_steps
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return spec
