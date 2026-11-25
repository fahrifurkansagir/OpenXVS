"""WP-01b: Min-Max Mipmap DEM pyramid, curved-Earth WGS-84 ray-DEM, and TSO-C151d TAWS FLTA."""

from xvs.v2.wp01b.mipmap import (
    DEFAULT_KSFO_PACK_PATH,
    MinMaxMipmapPyramid,
    build_minmax_mipmap,
    load_ksfo_3dep_mipmap,
    sample_height_bilinear_m,
    u16_to_meters,
    verify_mipmap_invariant,
)
from xvs.v2.wp01b.ray_dem import (
    EARTH_MEAN_RADIUS_M,
    RayDemHit,
    earth_curvature_drop_m,
    intersect_ray_dem_dense_ref,
    intersect_ray_dem_ecef,
    intersect_ray_dem_ned,
)
from xvs.v2.wp01b.taws_flta import (
    CLASS_B_DESENS_INNER_AGL_M,
    CLASS_B_DESENS_INNER_DIST_M,
    CLASS_B_DESENS_OUTER_AGL_M,
    CLASS_B_DESENS_OUTER_DIST_M,
    CLASS_B_RTC_SI_M,
    DESCENT_RATE_THRESHOLD_MPS,
    FltaResult,
    FlightPhase,
    evaluate_taws_flta,
    is_approach_desensitized,
    required_terrain_clearance_m,
)

__all__ = [
    "CLASS_B_DESENS_INNER_AGL_M",
    "CLASS_B_DESENS_INNER_DIST_M",
    "CLASS_B_DESENS_OUTER_AGL_M",
    "CLASS_B_DESENS_OUTER_DIST_M",
    "CLASS_B_RTC_SI_M",
    "DEFAULT_KSFO_PACK_PATH",
    "DESCENT_RATE_THRESHOLD_MPS",
    "EARTH_MEAN_RADIUS_M",
    "FltaResult",
    "FlightPhase",
    "MinMaxMipmapPyramid",
    "RayDemHit",
    "build_minmax_mipmap",
    "earth_curvature_drop_m",
    "evaluate_taws_flta",
    "intersect_ray_dem_dense_ref",
    "intersect_ray_dem_ecef",
    "intersect_ray_dem_ned",
    "is_approach_desensitized",
    "load_ksfo_3dep_mipmap",
    "required_terrain_clearance_m",
    "sample_height_bilinear_m",
    "u16_to_meters",
    "verify_mipmap_invariant",
]
