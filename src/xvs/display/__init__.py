"""OpenXVS WP-08 Crew Display package (G0 & G1)."""

from __future__ import annotations

from xvs.display.symbology import (
    TRAFFIC_TIER_CORRECTIVE,
    TRAFFIC_TIER_NAMES,
    TRAFFIC_TIER_NONE,
    TRAFFIC_TIER_PREVENTIVE,
    TRAFFIC_TIER_WARNING,
    classify_traffic_symbology_tier,
    compute_conformal_horizon_with_pl_band,
    compute_p_wcv_belief,
    compute_rect_intersection_area,
    deconflict_hud_labels,
    draw_g1_symbology,
    draw_panorama_g1_symbology,
    project_passive_track_to_panorama,
    project_passive_track_to_viewport,
)
from xvs.display.viewport import (
    compute_viewport_pose_from_estimated_attitude,
    extract_viewport,
    render_viewport_frame,
)
from xvs.display.latency import (
    compute_coasting_pl_deg,
    evaluate_integrity_display_blanking_latency,
)

__all__ = [
    "TRAFFIC_TIER_NONE",
    "TRAFFIC_TIER_PREVENTIVE",
    "TRAFFIC_TIER_CORRECTIVE",
    "TRAFFIC_TIER_WARNING",
    "TRAFFIC_TIER_NAMES",
    "compute_p_wcv_belief",
    "classify_traffic_symbology_tier",
    "project_passive_track_to_viewport",
    "project_passive_track_to_panorama",
    "compute_conformal_horizon_with_pl_band",
    "compute_rect_intersection_area",
    "deconflict_hud_labels",
    "draw_g1_symbology",
    "draw_panorama_g1_symbology",
    "compute_viewport_pose_from_estimated_attitude",
    "extract_viewport",
    "render_viewport_frame",
    "compute_coasting_pl_deg",
    "evaluate_integrity_display_blanking_latency",
]
