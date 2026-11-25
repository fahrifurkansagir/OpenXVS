"""Traffic perception metrics package for OpenXVS."""

from .aot import (
    AOTConfig,
    AOTMetricsSummary,
    Detection,
    DetectionFrame,
    GTFrame,
    GTObject,
    clopper_pearson_interval,
    compute_eiou,
    compute_iou,
    dilate_box,
    evaluate_aot,
    load_detections_from_jsonl,
    load_gt_from_g2,
    load_gt_from_g3a,
    poisson_rate_upper_bound,
)

__all__ = [
    "AOTConfig",
    "AOTMetricsSummary",
    "Detection",
    "DetectionFrame",
    "GTFrame",
    "GTObject",
    "clopper_pearson_interval",
    "compute_eiou",
    "compute_iou",
    "dilate_box",
    "evaluate_aot",
    "load_detections_from_jsonl",
    "load_gt_from_g2",
    "load_gt_from_g3a",
    "poisson_rate_upper_bound",
]
