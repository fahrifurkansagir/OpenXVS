"""Strict scope & safety contract for 1x1 km runway-threshold 3DGS (`XVS-World`, DR-04 §5 / SP2 §3.3).

Hard-blocks 5x5 km full-corridor OOM scenes and forbids any use of 3DGS depth buffers
as WP-01 <= 0.05 px geometric ground truth (`gt_tier` MUST be `"VISUAL_STRESS_ONLY"`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

MAX_EXTENT_M: float = 1000.0
MAX_SPLATS: int = 15_000_000
MAX_SPZ_GB: float = 0.35
ALLOWED_WORLD_BACKEND: str = "gsplat_3dgut_2dgs"
REQUIRED_GT_TIER: str = "VISUAL_STRESS_ONLY"
MIN_FAA_ADIP_GCPS: int = 8
MAX_SH_DEGREE: int = 2
BYTES_PER_SPLAT_ADAM_TRAIN: int = 944  # 59 floats * 4 B * 4x Adam/grad state (DR-04 §5.3)
RASTER_OVERHEAD_GB: float = 8.0


def estimate_gsplat_vram_gb(num_splats: int) -> float:
    """First-principles single-GPU training VRAM estimate (GB) from DR-04 §5.3."""
    if num_splats <= 0:
        raise ValueError(f"num_splats must be positive, got {num_splats}")
    return (float(num_splats) * BYTES_PER_SPLAT_ADAM_TRAIN) / 1e9 + RASTER_OVERHEAD_GB


@dataclass(frozen=True)
class WorldTwinConfig:
    """Hard-gated configuration for scoped 1x1 km runway-threshold visual-stress 3DGS twins."""

    site_icao: str = "KSFO"
    runway_id: str = "28L"
    anchor_lat_deg: float = 37.6117
    anchor_lon_deg: float = -122.3583
    anchor_h_wgs84_m: float = -28.6714
    extent_x_m: float = 1000.0
    extent_y_m: float = 1000.0
    max_splats: int = 15_000_000
    spz_max_gb: float = 0.35
    backend: str = ALLOWED_WORLD_BACKEND
    gt_tier: str = REQUIRED_GT_TIER
    sh_degree: int = 2
    min_faa_adip_gcps: int = 8
    packed_memory_mode: bool = True
    seed: int = 20261121

    def __post_init__(self) -> None:
        if not self.site_icao or len(self.site_icao.strip()) < 3:
            raise ValueError(f"Invalid site_icao: {self.site_icao!r}")
        if not math.isfinite(self.extent_x_m) or not (0.0 < self.extent_x_m <= MAX_EXTENT_M):
            raise ValueError(
                f"extent_x_m={self.extent_x_m} violates scoped 1x1 km cap (0 < extent <= {MAX_EXTENT_M} m)"
            )
        if not math.isfinite(self.extent_y_m) or not (0.0 < self.extent_y_m <= MAX_EXTENT_M):
            raise ValueError(
                f"extent_y_m={self.extent_y_m} violates scoped 1x1 km cap (0 < extent <= {MAX_EXTENT_M} m)"
            )
        if not isinstance(self.max_splats, int) or not (1 <= self.max_splats <= MAX_SPLATS):
            raise ValueError(
                f"max_splats={self.max_splats} violates budget (1 <= max_splats <= {MAX_SPLATS})"
            )
        if not math.isfinite(self.spz_max_gb) or not (0.0 < self.spz_max_gb <= MAX_SPZ_GB):
            raise ValueError(
                f"spz_max_gb={self.spz_max_gb} violates browser stream cap (0 < spz_max_gb <= {MAX_SPZ_GB} GB)"
            )
        if self.backend != ALLOWED_WORLD_BACKEND:
            raise ValueError(
                f"backend={self.backend!r} forbidden; must be {ALLOWED_WORLD_BACKEND!r}"
            )
        if self.gt_tier != REQUIRED_GT_TIER:
            raise ValueError(
                f"gt_tier={self.gt_tier!r} forbidden; 3DGS twins MUST be stamped {REQUIRED_GT_TIER!r} "
                f"and NEVER used as WP-01 <= 0.05 px certification ground truth."
            )
        if not (0 <= self.sh_degree <= MAX_SH_DEGREE):
            raise ValueError(f"sh_degree={self.sh_degree} must be in [0, {MAX_SH_DEGREE}]")
        if self.min_faa_adip_gcps < MIN_FAA_ADIP_GCPS:
            raise ValueError(
                f"min_faa_adip_gcps={self.min_faa_adip_gcps} < {MIN_FAA_ADIP_GCPS} required for Sim(3) WGS-84 anchoring"
            )
        if not self.packed_memory_mode:
            raise ValueError("packed_memory_mode must be True for single-GPU Colab G4 memory safety")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_sha256(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


def ksfo_28l_default_config() -> WorldTwinConfig:
    """Canonical KSFO 28L 1x1 km runway-threshold visual-stress preset (GEOID18 h=-28.6714 m)."""
    return WorldTwinConfig(
        site_icao="KSFO",
        runway_id="28L",
        anchor_lat_deg=37.6117,
        anchor_lon_deg=-122.3583,
        anchor_h_wgs84_m=-28.6714,
        extent_x_m=1000.0,
        extent_y_m=1000.0,
        max_splats=15_000_000,
        spz_max_gb=0.35,
        backend=ALLOWED_WORLD_BACKEND,
        gt_tier=REQUIRED_GT_TIER,
        sh_degree=2,
        min_faa_adip_gcps=8,
        packed_memory_mode=True,
        seed=20261121,
    )
