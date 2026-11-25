"""Scoped 1x1 km runway-threshold 3DGS visual-stress twin contract (`XVS-World`, DR-04 §5 / SP2 §3.3)."""

from .config import (
    ALLOWED_WORLD_BACKEND,
    MAX_EXTENT_M,
    MAX_SPLATS,
    MAX_SPZ_GB,
    REQUIRED_GT_TIER,
    WorldTwinConfig,
    estimate_gsplat_vram_gb,
    ksfo_28l_default_config,
)
from .runner import (
    build_colab_g4_job_spec,
    build_gsplat_command,
    validate_frame_metadata_visual_stress_only,
    write_dry_run_manifest,
)

__all__ = [
    "ALLOWED_WORLD_BACKEND",
    "MAX_EXTENT_M",
    "MAX_SPLATS",
    "MAX_SPZ_GB",
    "REQUIRED_GT_TIER",
    "WorldTwinConfig",
    "build_colab_g4_job_spec",
    "build_gsplat_command",
    "estimate_gsplat_vram_gb",
    "ksfo_28l_default_config",
    "validate_frame_metadata_visual_stress_only",
    "write_dry_run_manifest",
]
