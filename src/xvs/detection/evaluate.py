"""Evaluation engine for traffic detection benchmarks (WP-04 §3).

Computes matched pairs, missed GT, false positives, stratified P_d with Clopper-Pearson CIs,
encounter-cluster bootstrap, localization errors, and false positive rates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np

from xvs.metrics.aot import (
    GTObject,
    GTFrame,
    Detection,
    DetectionFrame,
    AOTConfig,
    match_single_frame,
    clopper_pearson_interval,
    poisson_rate_upper_bound,
    compute_eiou,
)

RANGE_BINS_KM = [
    (0.0, 0.5),
    (0.5, 1.0),
    (1.0, 2.0),
    (2.0, 4.0),
    (4.0, 6.0),
    (6.0, 10.0),
]

S_PX_BINS = [
    (0.0, 1.0),
    (1.0, 2.0),
    (2.0, 3.0),
    (3.0, 4.0),
    (4.0, 6.0),
    (6.0, 8.0),
    (8.0, 16.0),
    (16.0, float("inf")),
]


def get_range_bin_label(range_m: float) -> str:
    range_km = range_m / 1000.0
    for r_min, r_max in RANGE_BINS_KM:
        if r_min <= range_km < r_max:
            return f"[{r_min:.1f}, {r_max:.1f}) km"
    return ">= 10.0 km"


def get_spx_bin_label(s_px: float) -> str:
    for s_min, s_max in S_PX_BINS:
        if s_min <= s_px < s_max:
            if math.isinf(s_max):
                return f">= {s_min:.0f} px"
            return f"[{s_min:.0f}, {s_max:.0f}) px"
    return "unknown"


@dataclass
class StratumStats:
    n_instances: int = 0
    n_detected: int = 0
    n_encounters: int = 0
    encounters: set[str] = field(default_factory=set)

    @property
    def pd(self) -> float:
        return float(self.n_detected / self.n_instances) if self.n_instances > 0 else 0.0

    @property
    def ci95(self) -> tuple[float, float]:
        if self.n_instances == 0:
            return (0.0, 0.0)
        return clopper_pearson_interval(self.n_detected, self.n_instances, confidence=0.95)

    def to_dict(self) -> dict[str, Any]:
        ci = self.ci95
        return {
            "n_instances": self.n_instances,
            "n_detected": self.n_detected,
            "n_encounters": len(self.encounters),
            "pd": float(self.pd),
            "ci95_low": float(ci[0]),
            "ci95_high": float(ci[1]),
        }


def evaluate_detections(
    gt_frames: Sequence[GTFrame],
    det_frames: Sequence[DetectionFrame],
    profile: str = "xvs",
    min_eiou: float = 0.2,
    min_area: float = 100.0,
) -> dict[str, Any]:
    """Evaluate candidate detections against ground-truth frames.

    Args:
        gt_frames: sequence of GTFrame objects
        det_frames: sequence of DetectionFrame objects
        profile: 'xvs' (max_eval_range_m = inf) or 'aot' (max_eval_range_m = 700m)
        min_eiou: match threshold (default 0.2)
        min_area: dilation area threshold (default 100 px²)
    """
    max_range = float("inf") if profile == "xvs" else 700.0
    config = AOTConfig(
        min_object_area=min_area,
        match_eiou_threshold=min_eiou,
        fp_eiou_threshold=0.02,
        max_eval_range_m=max_range,
    )

    # Index detection frames by (frame_id, camera_id)
    det_map: dict[tuple[str | int, str], list[Detection]] = {}
    for df in det_frames:
        det_map[(df.frame_id, df.camera_id)] = df.detections

    total_gt = 0
    total_matched = 0
    total_fp = 0
    total_frames = len(gt_frames)

    # Strata
    by_class: dict[str, StratumStats] = {}
    by_range: dict[str, StratumStats] = {}
    by_spx: dict[str, StratumStats] = {}
    by_horizon: dict[str, StratumStats] = {}

    matched_pairs: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []

    for gtf in gt_frames:
        key = (gtf.frame_id, gtf.camera_id)
        dets = det_map.get(key, [])
        matched, missed, dc_gt, fp, dc_det = match_single_frame(gtf.objects, dets, config)

        total_gt += len(gtf.objects)
        total_matched += len(matched)
        total_fp += len(fp)

        # Record false positives
        for fpd in fp:
            false_positives.append({
                "frame_id": gtf.frame_id,
                "camera_id": gtf.camera_id,
                "sequence_id": gtf.sequence_id,
                "box": fpd.box,
                "score": fpd.score,
            })

        # Process matched
        matched_gt_ids = set()
        for gt_obj, det_obj, eiou in matched:
            matched_gt_ids.add(gt_obj.object_id)
            w_gt = gt_obj.box_edges[2] - gt_obj.box_edges[0]
            h_gt = gt_obj.box_edges[3] - gt_obj.box_edges[1]
            s_px = max(w_gt, h_gt)

            matched_pairs.append({
                "frame_id": gtf.frame_id,
                "camera_id": gtf.camera_id,
                "sequence_id": gtf.sequence_id,
                "object_id": gt_obj.object_id,
                "class_id": gt_obj.class_id,
                "range_m": gt_obj.range_m,
                "s_px": s_px,
                "box_gt": gt_obj.box_edges,
                "box_det": det_obj.box,
                "eiou": eiou,
            })

        # Update strata for all gt objects in this frame
        for gt_obj in gtf.objects:
            is_det = (gt_obj.object_id in matched_gt_ids)
            enc_id = gtf.sequence_id

            w_gt = gt_obj.box_edges[2] - gt_obj.box_edges[0]
            h_gt = gt_obj.box_edges[3] - gt_obj.box_edges[1]
            s_px = max(w_gt, h_gt)

            # Class stratum
            cls_stat = by_class.setdefault(gt_obj.class_id, StratumStats())
            cls_stat.n_instances += 1
            if is_det:
                cls_stat.n_detected += 1
            cls_stat.encounters.add(enc_id)

            # Range stratum
            rng_lbl = get_range_bin_label(gt_obj.range_m)
            rng_stat = by_range.setdefault(rng_lbl, StratumStats())
            rng_stat.n_instances += 1
            if is_det:
                rng_stat.n_detected += 1
            rng_stat.encounters.add(enc_id)

            # S_px stratum
            spx_lbl = get_spx_bin_label(s_px)
            spx_stat = by_spx.setdefault(spx_lbl, StratumStats())
            spx_stat.n_instances += 1
            if is_det:
                spx_stat.n_detected += 1
            spx_stat.encounters.add(enc_id)

            # Horizon stratum
            hz_stat = by_horizon.setdefault(gt_obj.horizon, StratumStats())
            hz_stat.n_instances += 1
            if is_det:
                hz_stat.n_detected += 1
            hz_stat.encounters.add(enc_id)

    overall_pd = float(total_matched / total_gt) if total_gt > 0 else 0.0
    overall_ci = clopper_pearson_interval(total_matched, total_gt) if total_gt > 0 else (0.0, 0.0)
    fppi = float(total_fp / total_frames) if total_frames > 0 else 0.0
    fppi_upper = poisson_rate_upper_bound(total_fp, float(total_frames)) if total_frames > 0 else 0.0

    return {
        "profile": profile,
        "total_frames": total_frames,
        "total_gt": total_gt,
        "total_matched": total_matched,
        "total_fp": total_fp,
        "overall_pd": overall_pd,
        "overall_ci95": [float(overall_ci[0]), float(overall_ci[1])],
        "fppi": fppi,
        "fppi_upper_95": fppi_upper,
        "by_class": {k: v.to_dict() for k, v in sorted(by_class.items())},
        "by_range": {k: v.to_dict() for k, v in sorted(by_range.items())},
        "by_spx": {k: v.to_dict() for k, v in sorted(by_spx.items())},
        "by_horizon": {k: v.to_dict() for k, v in sorted(by_horizon.items())},
        "matched_pairs": matched_pairs,
        "false_positives": false_positives,
    }


def encounter_cluster_bootstrap(
    encounters_data: dict[str, list[dict[str, Any]]],
    metric_fn: Any,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 20260925,
) -> tuple[float, float, float]:
    """Encounter-cluster bootstrap confidence interval for arbitrary metric function.

    Args:
        encounters_data: dict mapping encounter_id -> list of instance records
        metric_fn: function taking flat list of instances and returning float metric
        n_resamples: number of bootstrap cluster resamples (default 1000)
        confidence: confidence level (default 0.95)
        seed: random seed for reproducibility

    Returns:
        (nominal_estimate, lower_ci, upper_ci)
    """
    rng = np.random.RandomState(seed)
    encounter_keys = list(encounters_data.keys())
    n_enc = len(encounter_keys)
    if n_enc == 0:
        return (0.0, 0.0, 0.0)

    # Flat nominal data
    all_instances = [inst for inst_list in encounters_data.values() for inst in inst_list]
    nominal = float(metric_fn(all_instances))

    estimates = []
    for _ in range(n_resamples):
        sampled_keys = rng.choice(encounter_keys, size=n_enc, replace=True)
        resampled_instances = [inst for k in sampled_keys for inst in encounters_data[k]]
        val = metric_fn(resampled_instances)
        if not math.isnan(val):
            estimates.append(val)

    if not estimates:
        return (nominal, nominal, nominal)

    alpha = 1.0 - confidence
    low_pct = 100.0 * (alpha / 2.0)
    high_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_low = float(np.percentile(estimates, low_pct))
    ci_high = float(np.percentile(estimates, high_pct))
    return (nominal, ci_low, ci_high)
