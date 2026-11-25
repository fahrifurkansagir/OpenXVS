"""AOT benchmark traffic detection and encounter tracking metrics library.

Conforms to docs/xvs/batches/GEMINI_G-B1.md §1 and docs/xvs/research/R1_traffic_detection.md §3.
All formulas, statistical confidence intervals, eIoU dilation, encounter tracking,
and metric splits are implemented natively without external dependencies beyond
numpy, scipy, and the Python standard library.
"""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any, Iterable, Sequence
import numpy as np
import scipy.stats as stats


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class GTObject:
    """Canonical ground-truth intruder object in a frame."""
    object_id: str
    class_id: str
    range_m: float
    box_edges: list[float]  # [u_min, v_min, u_max, v_max]
    horizon: str  # 'above' | 'below' | 'unknown'

    def __post_init__(self) -> None:
        if self.horizon not in ("above", "below", "unknown"):
            raise ValueError(f"horizon must be 'above'|'below'|'unknown', got {self.horizon!r}")
        if len(self.box_edges) != 4:
            raise ValueError(f"box_edges must have 4 elements [u_min, v_min, u_max, v_max], got {self.box_edges!r}")


@dataclasses.dataclass(frozen=True)
class GTFrame:
    """Canonical ground-truth frame."""
    frame_id: str | int
    t_s: float
    sequence_id: str
    camera_id: str
    image_wh: tuple[int, int]
    objects: list[GTObject]


@dataclasses.dataclass(frozen=True)
class Detection:
    """Candidate detection prediction."""
    box: list[float]  # [u_min, v_min, u_max, v_max]
    score: float
    track_id: str | int | None = None

    def __post_init__(self) -> None:
        if len(self.box) != 4:
            raise ValueError(f"box must have 4 elements [u_min, v_min, u_max, v_max], got {self.box!r}")


@dataclasses.dataclass(frozen=True)
class DetectionFrame:
    """Candidate detections for a specific frame and camera."""
    frame_id: str | int
    camera_id: str
    detections: list[Detection]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class AOTConfig:
    """Configurable thresholds for AOT benchmark evaluation."""
    min_object_area: float = 100.0  # px²
    match_eiou_threshold: float = 0.2
    fp_eiou_threshold: float = 0.02
    max_eval_range_m: float = 700.0
    min_secs: float = 3.0  # s
    max_gap_factor: float = 0.1  # max gap = max_gap_factor * min_secs (0.3 s)
    upper_bound_min_dist: float = 330.0  # m
    upper_bound_max_dist: float = 700.0  # m
    tracking_duration_s: float = 3.0  # s
    tracking_deadline_range_m: float = 300.0  # m
    range_bins: tuple[tuple[float, float], ...] = (
        (0.0, 300.0),
        (300.0, 700.0),
        (700.0, 1500.0),
        (1500.0, 3000.0),
        (3000.0, 5000.0),
        (5000.0, 10000.0),
        (10000.0, float("inf")),
    )
    alpha: float = 0.05  # 95% confidence level
    flight_hours: float | None = None  # Optional flight hours exposure override



# ---------------------------------------------------------------------------
# Extended IoU (eIoU) and Geometry
# ---------------------------------------------------------------------------

def dilate_box(box: Sequence[float], min_area: float = 100.0) -> list[float]:
    """Dilate a box about its centre to min_area px², keeping its aspect ratio.

    If box area >= min_area, the box is returned unchanged (as a list of floats).
    Non-positive dimensions are guarded to preserve centre and produce min_area.
    Coordinates are normalized so that min <= max.
    """
    u1, v1, u2, v2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    u_min, u_max = min(u1, u2), max(u1, u2)
    v_min, v_max = min(v1, v2), max(v1, v2)
    w = u_max - u_min
    h = v_max - v_min
    area = w * h
    if area >= min_area and w > 0 and h > 0:
        return [u_min, v_min, u_max, v_max]

    u_c = (u_min + u_max) / 2.0
    v_c = (v_min + v_max) / 2.0

    if w <= 0.0 and h <= 0.0:
        side = math.sqrt(min_area)
        w_prime = side
        h_prime = side
    elif w <= 0.0:
        h_pos = max(h, 1e-6)
        w_prime = min_area / h_pos
        h_prime = h_pos
    elif h <= 0.0:
        w_pos = max(w, 1e-6)
        w_prime = w_pos
        h_prime = min_area / w_pos
    else:
        scale = math.sqrt(min_area / area)
        w_prime = w * scale
        h_prime = h * scale

    return [
        u_c - w_prime / 2.0,
        v_c - h_prime / 2.0,
        u_c + w_prime / 2.0,
        v_c + h_prime / 2.0,
    ]


def compute_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Compute standard Intersection over Union (IoU) of two boxes [u_min, v_min, u_max, v_max]."""
    u1_1, v1_1, u2_1, v2_1 = float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3])
    u1_2, v1_2, u2_2, v2_2 = float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3])

    u_min1, u_max1 = min(u1_1, u2_1), max(u1_1, u2_1)
    v_min1, v_max1 = min(v1_1, v2_1), max(v1_1, v2_1)
    u_min2, u_max2 = min(u1_2, u2_2), max(u1_2, u2_2)
    v_min2, v_max2 = min(v1_2, v2_2), max(v1_2, v2_2)

    inter_u_min = max(u_min1, u_min2)
    inter_v_min = max(v_min1, v_min2)
    inter_u_max = min(u_max1, u_max2)
    inter_v_max = min(v_max1, v_max2)

    inter_w = max(0.0, inter_u_max - inter_u_min)
    inter_h = max(0.0, inter_v_max - inter_v_min)
    inter_area = inter_w * inter_h

    area1 = max(0.0, u_max1 - u_min1) * max(0.0, v_max1 - v_min1)
    area2 = max(0.0, u_max2 - u_min2) * max(0.0, v_max2 - v_min2)
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def compute_eiou(
    box_gt: Sequence[float],
    box_det: Sequence[float],
    min_area: float = 100.0,
) -> float:
    """Compute extended IoU (eIoU) according to the AOT benchmark definition.

    - A GT box smaller than min_area (100 px²) is dilated about its centre to 100 px²,
      keeping its aspect ratio.
    - The detection matched against it is dilated the same way (if < min_area).
    - If GT box >= min_area and detection < min_area, detection is also dilated to min_area.
    """
    area_gt = max(0.0, box_gt[2] - box_gt[0]) * max(0.0, box_gt[3] - box_gt[1])
    area_det = max(0.0, box_det[2] - box_det[0]) * max(0.0, box_det[3] - box_det[1])

    b_gt = dilate_box(box_gt, min_area) if area_gt < min_area else list(box_gt)
    b_det = dilate_box(box_det, min_area) if area_det < min_area else list(box_det)

    return compute_iou(b_gt, b_det)


# ---------------------------------------------------------------------------
# Statistical Confidence Intervals
# ---------------------------------------------------------------------------

def clopper_pearson_interval(
    k: int,
    n: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Exact two-sided Clopper-Pearson confidence interval for binomial proportion k / n.

    Args:
        k: number of successes (0 <= k <= n)
        n: number of trials (n >= 0)
        confidence: confidence level (default 0.95)

    Returns:
        (lower, upper) confidence interval bounds in [0.0, 1.0].
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    if k < 0 or k > n:
        raise ValueError(f"k must be in [0, n], got k={k}, n={n}")

    alpha = 1.0 - confidence

    # Lower bound: beta.ppf(alpha / 2, k, n - k + 1)
    if k == 0:
        lower = 0.0
    else:
        lower = float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))

    # Upper bound: beta.ppf(1 - alpha / 2, k + 1, n - k)
    if k == n:
        upper = 1.0
    else:
        upper = float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))

    return (max(0.0, lower), min(1.0, upper))


def poisson_rate_upper_bound(
    k: int,
    exposure: float,
    confidence: float = 0.95,
) -> float:
    """Exact one-sided Poisson rate upper bound via chi-square distribution.

    For k observed events in exposure T (flight hours or images):
      mu_upper = 0.5 * chi2.ppf(confidence, 2 * (k + 1))
      rate_upper = mu_upper / exposure

    Args:
        k: number of observed events (k >= 0)
        exposure: total exposure (T > 0)
        confidence: one-sided confidence level (default 0.95)

    Returns:
        One-sided upper bound on the Poisson event rate.
    """
    if exposure <= 0.0:
        return float("inf")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")

    alpha = 1.0 - confidence
    # 1 - alpha = confidence
    mu_upper = 0.5 * float(stats.chi2.ppf(confidence, 2 * (k + 1)))
    return float(mu_upper / exposure)


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

def load_gt_from_g2(
    data: dict[str, Any],
    use_mask_box: bool = False,
) -> list[GTFrame]:
    """Convert G2 `results_*.json` records into canonical GTFrame objects.

    Args:
        data: loaded G2 results JSON (contains 'records')
        use_mask_box: If True, uses renders.A.box_edges when present; defaults to box_analytic.
    """
    frames: list[GTFrame] = []
    records = data.get("records", [])

    for r in records:
        instant_id = r.get("instant_id", "")
        t_s = float(r.get("t_s", 0.0))  # default-ok: optional fallback default
        sequence_id = r.get("encounter_id", "default_seq")
        class_id = r.get("class_id", "unknown_class")
        model_id = r.get("model_id", instant_id)
        range_m = float(r.get("intruder", {}).get("range_m", r.get("target_range_m", 0.0)))  # default-ok: optional fallback default

        cams = r.get("cams", {})
        for cam_id, cam in cams.items():
            win = cam.get("window", [-0.5, -0.5, 1023.5, 1023.5])
            w = int(round(win[2] - win[0]))
            h = int(round(win[3] - win[1]))
            image_wh = (w, h)

            gt_info = cam.get("gt", {})
            in_image = bool(gt_info.get("in_image", False))
            objects: list[GTObject] = []

            if in_image:
                box: list[float] | None = None
                if use_mask_box:
                    renders_a = cam.get("renders", {}).get("A", {})
                    box = renders_a.get("box_edges")
                if box is None:
                    box = gt_info.get("box_analytic")

                if box is not None:
                    limb_dv = gt_info.get("limb_dv_px")
                    if limb_dv is None:
                        horizon = "unknown"
                    elif limb_dv > 0:
                        horizon = "below"
                    elif limb_dv < 0:
                        horizon = "above"
                    else:
                        horizon = "unknown"

                    objects.append(
                        GTObject(
                            object_id=model_id,
                            class_id=class_id,
                            range_m=range_m,
                            box_edges=[float(x) for x in box],
                            horizon=horizon,
                        )
                    )

            frames.append(
                GTFrame(
                    frame_id=instant_id,
                    t_s=t_s,
                    sequence_id=sequence_id,
                    camera_id=cam_id,
                    image_wh=image_wh,
                    objects=objects,
                )
            )

    return frames


def load_gt_from_g3a(
    frames_iterable: Iterable[dict[str, Any]],
    use_mask_box: bool = False,
    cameras: Sequence[str] | None = None,
) -> list[GTFrame]:
    """Convert WP02 §5 `frames.jsonl` objects into canonical GTFrame objects.

    WP02 §5 schema:
      {"frame_id": ..., "t_s": ..., "sequence_id": ...,
       "traffic": [{"id": ..., "class": ..., "range_m": ...,
                    "cams": {"F": {"u": ..., "v": ..., "in_image": true,
                                   "box_analytic": [...], "box_amodal_px": [...],
                                   "limb_dv_px": ...}}}]}
    """
    frames: list[GTFrame] = []

    for item in frames_iterable:
        frame_id = item.get("frame_id", "")
        t_s = float(item.get("t_s", 0.0))  # default-ok: optional fallback default
        sequence_id = item.get("sequence_id", "g3a_seq")
        traffic_list = item.get("traffic", [])

        # Collect cameras: use provided camera list, or unique cameras mentioned in this frame
        if cameras is not None:
            active_cams = list(cameras)
        else:
            cam_ids: set[str] = set()
            for intruder in traffic_list:
                cam_ids.update(intruder.get("cams", {}).keys())
            active_cams = sorted(cam_ids) if cam_ids else ["F"]

        for cam_id in active_cams:
            objects: list[GTObject] = []
            image_wh = (1024, 1024)

            for intruder in traffic_list:
                obj_id = str(intruder.get("id", intruder.get("model_id", "obj")))
                class_id = str(intruder.get("class", intruder.get("class_id", "unknown")))
                range_m = float(intruder.get("range_m", 0.0))  # default-ok: optional fallback default

                cam_data = intruder.get("cams", {}).get(cam_id)
                if not cam_data or not cam_data.get("in_image", False):
                    continue

                box: list[float] | None = None
                if use_mask_box:
                    box_amodal = cam_data.get("box_amodal_px") or cam_data.get("box_edges")
                    if box_amodal is not None and len(box_amodal) == 4:
                        box = [float(x) for x in box_amodal]
                if box is None:
                    box_an = cam_data.get("box_analytic")
                    if box_an is not None and len(box_an) == 4:
                        box = [float(x) for x in box_an]

                if box is not None:
                    limb_dv = cam_data.get("limb_dv_px")
                    if limb_dv is None:
                        horizon = "unknown"
                    elif limb_dv > 0:
                        horizon = "below"
                    elif limb_dv < 0:
                        horizon = "above"
                    else:
                        horizon = "unknown"

                    objects.append(
                        GTObject(
                            object_id=obj_id,
                            class_id=class_id,
                            range_m=range_m,
                            box_edges=box,
                            horizon=horizon,
                        )
                    )

            frames.append(
                GTFrame(
                    frame_id=frame_id,
                    t_s=t_s,
                    sequence_id=sequence_id,
                    camera_id=cam_id,
                    image_wh=image_wh,
                    objects=objects,
                )
            )

    return frames


def load_detections_from_jsonl(lines: Iterable[str | dict[str, Any]]) -> list[DetectionFrame]:
    """Parse JSONL lines into DetectionFrame instances."""
    det_frames: list[DetectionFrame] = []
    for item in lines:
        if isinstance(item, str):
            item = item.strip()
            if not item:
                continue
            data = json.loads(item)
        else:
            data = item

        frame_id = data["frame_id"]
        camera_id = data["camera_id"]
        dets: list[Detection] = []
        for d in data.get("detections", []):
            dets.append(
                Detection(
                    box=[float(x) for x in d["box"]],
                    score=float(d.get("score", 1.0)),  # default-ok: optional fallback default
                    track_id=d.get("track_id"),
                )
            )
        det_frames.append(
            DetectionFrame(
                frame_id=frame_id,
                camera_id=camera_id,
                detections=dets,
            )
        )
    return det_frames


# ---------------------------------------------------------------------------
# Evaluation Results & Metrics
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FrameMatchResult:
    """Matching result for a single camera frame."""
    frame_id: str | int
    camera_id: str
    sequence_id: str
    t_s: float
    matched_gt: list[tuple[GTObject, Detection, float]]  # (gt, det, eiou)
    missed_gt: list[GTObject]  # GT within 700 m with no match
    dont_care_gt: list[GTObject]  # GT > 700 m
    false_positives: list[Detection]  # Detections with eIoU < 0.02 against every GT
    dont_care_detections: list[Detection]  # Detections in [0.02, 0.2] or matching > 700 m GT


@dataclasses.dataclass
class EncounterResult:
    """Encounter outcome."""
    encounter_id: str
    sequence_id: str
    object_id: str
    class_id: str
    t_start: float
    t_end: float
    duration_s: float
    min_range_m: float
    max_range_m: float
    is_valid: bool
    is_detected: bool
    detect_t_s: float | None
    track_id: str | int | None


@dataclasses.dataclass
class MetricsReport:
    """Aggregated metric numbers with statistical bounds."""
    count_total: int = 0
    count_matched: int = 0
    rate: float = float("nan")
    rate_ci_95: tuple[float, float] = (float("nan"), float("nan"))
    fp_count: int = 0
    exposure: float = 0.0
    rate_per_exposure: float = float("nan")
    rate_upper_95: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "count_total": self.count_total,
            "count_matched": self.count_matched,
            "rate": round(self.rate, 5) if not math.isnan(self.rate) else None,
            "rate_ci_95": [round(x, 5) for x in self.rate_ci_95] if not math.isnan(self.rate_ci_95[0]) else None,
            "fp_count": self.fp_count,
            "exposure": round(self.exposure, 4),
            "rate_per_exposure": round(self.rate_per_exposure, 5) if not math.isnan(self.rate_per_exposure) else None,
            "rate_upper_95": round(self.rate_upper_95, 5) if not math.isnan(self.rate_upper_95) else None,
        }


@dataclasses.dataclass
class AOTMetricsSummary:
    """Complete AOT metrics summary including splits and encounter tracking."""
    # Frame level
    total_images: int
    gt_eval_objects: int  # within 700 m
    matched_gt_objects: int
    afdr: float
    afdr_ci_95: tuple[float, float]
    false_positives: int
    fppi: float
    fppi_upper_95: float

    # Encounter level
    total_encounters: int
    valid_encounters: int
    detected_valid_encounters: int
    edr: float
    edr_ci_95: tuple[float, float]

    # False alarm rate (HFAR)
    total_flight_hours: float
    false_alarms: int
    hfar: float
    hfar_upper_95: float

    # Splits
    by_class: dict[str, dict[str, Any]]
    by_range: dict[str, dict[str, Any]]
    by_horizon: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_level": {
                "total_images": self.total_images,
                "gt_eval_objects": self.gt_eval_objects,
                "matched_gt_objects": self.matched_gt_objects,
                "afdr": round(self.afdr, 5) if not math.isnan(self.afdr) else None,
                "afdr_ci_95": [round(x, 5) for x in self.afdr_ci_95] if not math.isnan(self.afdr_ci_95[0]) else None,
                "false_positives": self.false_positives,
                "fppi": round(self.fppi, 6) if not math.isnan(self.fppi) else None,
                "fppi_upper_95": round(self.fppi_upper_95, 6) if not math.isnan(self.fppi_upper_95) else None,
            },
            "encounter_level": {
                "total_encounters": self.total_encounters,
                "valid_encounters": self.valid_encounters,
                "detected_valid_encounters": self.detected_valid_encounters,
                "edr": round(self.edr, 5) if not math.isnan(self.edr) else None,
                "edr_ci_95": [round(x, 5) for x in self.edr_ci_95] if not math.isnan(self.edr_ci_95[0]) else None,
            },
            "hfar_level": {
                "total_flight_hours": round(self.total_flight_hours, 6),
                "false_alarms": self.false_alarms,
                "hfar": round(self.hfar, 5) if not math.isnan(self.hfar) else None,
                "hfar_upper_95": round(self.hfar_upper_95, 5) if not math.isnan(self.hfar_upper_95) else None,
            },
            "splits": {
                "by_class": self.by_class,
                "by_range": self.by_range,
                "by_horizon": self.by_horizon,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Core Matching & Evaluation Engine
# ---------------------------------------------------------------------------

def match_single_frame(
    gt_objects: list[GTObject],
    detections: list[Detection],
    config: AOTConfig,
) -> tuple[
    list[tuple[GTObject, Detection, float]],
    list[GTObject],
    list[GTObject],
    list[Detection],
    list[Detection],
]:
    """Greedy one-to-one matching by descending score using eIoU.

    Returns:
        (matched_pairs, missed_gt_within_700m, dont_care_gt, false_positives, dont_care_detections)
    """
    sorted_dets = sorted(detections, key=lambda d: d.score, reverse=True)
    available_gt_indices = set(range(len(gt_objects)))

    matched_pairs: list[tuple[GTObject, Detection, float]] = []
    unmatched_dets: list[Detection] = []

    # Greedy 1-to-1 matching
    for det in sorted_dets:
        best_eiou = -1.0
        best_gt_idx = -1
        for gt_idx in available_gt_indices:
            eiou = compute_eiou(gt_objects[gt_idx].box_edges, det.box, config.min_object_area)
            if eiou > best_eiou:
                best_eiou = eiou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0 and best_eiou > config.match_eiou_threshold:
            matched_pairs.append((gt_objects[best_gt_idx], det, best_eiou))
            available_gt_indices.remove(best_gt_idx)
        else:
            unmatched_dets.append(det)

    # Classify unmatched GT
    missed_gt: list[GTObject] = []
    dont_care_gt: list[GTObject] = []
    for gt_idx in available_gt_indices:
        gt = gt_objects[gt_idx]
        if gt.range_m <= config.max_eval_range_m:
            missed_gt.append(gt)
        else:
            dont_care_gt.append(gt)

    # Classify unmatched detections
    false_positives: list[Detection] = []
    dont_care_detections: list[Detection] = []

    for det in unmatched_dets:
        # Check against ALL GT objects in the frame
        if len(gt_objects) == 0:
            false_positives.append(det)
        else:
            max_eiou = max(
                compute_eiou(gt.box_edges, det.box, config.min_object_area)
                for gt in gt_objects
            )
            if max_eiou < config.fp_eiou_threshold:
                false_positives.append(det)
            else:
                dont_care_detections.append(det)

    return matched_pairs, missed_gt, dont_care_gt, false_positives, dont_care_detections


def evaluate_encounters(
    frame_results: list[FrameMatchResult],
    config: AOTConfig,
) -> list[EncounterResult]:
    """Segment GT tracks into encounters and evaluate detection requirements.

    Definition:
      - Encounter: An interval of >= MIN_SECS (3.0 s) in which one object is present,
        allowing gaps of up to 0.1 * MIN_SECS (0.3 s).
      - Valid encounter: min distance <= 330 m and max distance <= 700 m.
        For continuous flights starting beyond 700 m, the encounter evaluation
        window is the contiguous sub-interval where the intruder is within 700 m.
      - Detected: tracked (matched with same track_id) for >= 3.0 s, and that 3.0 s track
        happens before object comes within 300 m or within the first 3.0 s of the encounter.
    """
    max_gap = config.max_gap_factor * config.min_secs

    # Group observations by (sequence_id, object_id) and unique t_s
    # to handle multi-camera observations at the same timestamp cleanly
    obj_time_map: dict[tuple[str, str], dict[float, dict[str, Any]]] = {}

    for fr in frame_results:
        all_gt_in_frame = [p[0] for p in fr.matched_gt] + fr.missed_gt + fr.dont_care_gt
        matched_map: dict[str, set[str | int]] = {}
        for gt, det, _ in fr.matched_gt:
            if det.track_id is not None:
                matched_map.setdefault(gt.object_id, set()).add(det.track_id)

        for gt in all_gt_in_frame:
            key = (fr.sequence_id, gt.object_id)
            seq_obj_dict = obj_time_map.setdefault(key, {})
            if fr.t_s not in seq_obj_dict:
                seq_obj_dict[fr.t_s] = {
                    "t_s": fr.t_s,
                    "range_m": gt.range_m,
                    "class_id": gt.class_id,
                    "matched_track_ids": set(matched_map.get(gt.object_id, set())),
                }
            else:
                seq_obj_dict[fr.t_s]["matched_track_ids"].update(matched_map.get(gt.object_id, set()))
                seq_obj_dict[fr.t_s]["range_m"] = min(seq_obj_dict[fr.t_s]["range_m"], gt.range_m)

    encounter_results: list[EncounterResult] = []
    enc_counter = 0

    for (seq_id, obj_id), t_dict in obj_time_map.items():
        records = [t_dict[t] for t in sorted(t_dict.keys())]
        if not records:
            continue

        # Split full presence history into continuous intervals (gap <= max_gap)
        presence_intervals: list[list[dict[str, Any]]] = []
        cur_pres: list[dict[str, Any]] = [records[0]]
        for r in records[1:]:
            if r["t_s"] - cur_pres[-1]["t_s"] <= max_gap:
                cur_pres.append(r)
            else:
                presence_intervals.append(cur_pres)
                cur_pres = [r]
        presence_intervals.append(cur_pres)

        for pres in presence_intervals:
            # Check if this presence interval penetrates <= 700 m
            # If so, extract the sub-segment(s) where range_m <= upper_bound_max_dist (allowing gaps <= max_gap)
            pts_within_700 = [p for p in pres if p["range_m"] <= config.upper_bound_max_dist]

            candidate_intervals: list[list[dict[str, Any]]] = []
            if pts_within_700:
                cur_cand: list[dict[str, Any]] = [pts_within_700[0]]
                for p in pts_within_700[1:]:
                    if p["t_s"] - cur_cand[-1]["t_s"] <= max_gap:
                        cur_cand.append(p)
                    else:
                        candidate_intervals.append(cur_cand)
                        cur_cand = [p]
                candidate_intervals.append(cur_cand)
            else:
                candidate_intervals = [pres]

            for interval in candidate_intervals:
                t_start = interval[0]["t_s"]
                t_end = interval[-1]["t_s"]
                duration = t_end - t_start

                if duration < config.min_secs:
                    continue

                enc_counter += 1
                enc_id = f"ENC_{enc_counter:04d}_{seq_id}_{obj_id}"
                class_id = interval[0]["class_id"]
                min_r = min(x["range_m"] for x in interval)
                max_r = max(x["range_m"] for x in interval)

                is_valid = (min_r <= config.upper_bound_min_dist and max_r <= config.upper_bound_max_dist)

                is_detected = False
                detect_t: float | None = None
                detected_track_id: str | int | None = None

                # Find first time object comes within 300 m
                t_under_300 = float("inf")
                for x in interval:
                    if x["range_m"] <= config.tracking_deadline_range_m:
                        t_under_300 = x["t_s"]
                        break

                t_deadline_early = t_start + config.tracking_duration_s

                # Find all track IDs appearing in this presence interval (including prior tracking)
                all_track_ids: set[str | int] = set()
                for x in pres:
                    all_track_ids.update(x["matched_track_ids"])

                for trk in sorted(all_track_ids, key=str):
                    trk_times = sorted([x["t_s"] for x in pres if trk in x["matched_track_ids"]])
                    if not trk_times:
                        continue

                    # Find continuous sub-segments of trk with gap <= max_gap
                    sub_start = trk_times[0]
                    prev_t = trk_times[0]
                    for cur_t in trk_times[1:]:
                        if cur_t - prev_t <= max_gap:
                            prev_t = cur_t
                            if prev_t - sub_start >= config.tracking_duration_s:
                                achieved_t = prev_t
                                if (achieved_t <= t_under_300 + 1e-6) or (achieved_t <= t_deadline_early + 1e-6):
                                    is_detected = True
                                    detect_t = achieved_t
                                    detected_track_id = trk
                                    break
                        else:
                            sub_start = cur_t
                            prev_t = cur_t
                    if is_detected:
                        break

                encounter_results.append(
                    EncounterResult(
                        encounter_id=enc_id,
                        sequence_id=seq_id,
                        object_id=obj_id,
                        class_id=class_id,
                        t_start=t_start,
                        t_end=t_end,
                        duration_s=duration,
                        min_range_m=min_r,
                        max_range_m=max_r,
                        is_valid=is_valid,
                        is_detected=is_detected,
                        detect_t_s=detect_t,
                        track_id=detected_track_id,
                    )
                )

    return encounter_results


def evaluate_aot(
    gt_frames: list[GTFrame],
    det_frames: list[DetectionFrame],
    config: AOTConfig | None = None,
) -> AOTMetricsSummary:
    """Run full AOT benchmark evaluation across frames and sequences.

    Produces AFDR, FPPI, EDR, HFAR, confidence intervals, and breakdowns.
    """
    if config is None:
        config = AOTConfig()

    # Index detection frames by (frame_id, camera_id)
    det_map: dict[tuple[str | int, str], list[Detection]] = {}
    for df in det_frames:
        det_map[(df.frame_id, df.camera_id)] = df.detections

    # Map GT frames by (frame_id, camera_id)
    gt_map: dict[tuple[str | int, str], GTFrame] = {
        (gf.frame_id, gf.camera_id): gf for gf in gt_frames
    }

    # All unique camera frames across GT and Detections
    all_keys = list(gt_map.keys())
    for df in det_frames:
        k = (df.frame_id, df.camera_id)
        if k not in gt_map:
            all_keys.append(k)

    # Process each frame
    frame_results: list[FrameMatchResult] = []
    total_images = len(all_keys)
    seq_timestamps: dict[str, list[float]] = {}

    for k in all_keys:
        gf = gt_map.get(k)
        dets = det_map.get(k, [])
        if gf is not None:
            matched_pairs, missed_gt, dont_care_gt, fps, dont_care_dets = match_single_frame(
                gf.objects, dets, config
            )
            frame_results.append(
                FrameMatchResult(
                    frame_id=gf.frame_id,
                    camera_id=gf.camera_id,
                    sequence_id=gf.sequence_id,
                    t_s=gf.t_s,
                    matched_gt=matched_pairs,
                    missed_gt=missed_gt,
                    dont_care_gt=dont_care_gt,
                    false_positives=fps,
                    dont_care_detections=dont_care_dets,
                )
            )
            seq_timestamps.setdefault(gf.sequence_id, []).append(gf.t_s)
        else:
            # Frame present in detections but not in GT -> negative image with 0 GT objects
            matched_pairs, missed_gt, dont_care_gt, fps, dont_care_dets = match_single_frame(
                [], dets, config
            )
            frame_results.append(
                FrameMatchResult(
                    frame_id=k[0],
                    camera_id=k[1],
                    sequence_id="unknown_seq",
                    t_s=0.0,
                    matched_gt=[],
                    missed_gt=[],
                    dont_care_gt=[],
                    false_positives=fps,
                    dont_care_detections=dont_care_dets,
                )
            )

    # Calculate flight hours: use override if specified, otherwise sum of sequence durations
    if config.flight_hours is not None:
        flight_hours = float(config.flight_hours)
    else:
        total_duration_s = 0.0
        for seq_id, t_list in seq_timestamps.items():
            if t_list:
                dur = max(t_list) - min(t_list)
                total_duration_s += dur
        flight_hours = total_duration_s / 3600.0 if total_duration_s > 0.0 else 0.0

    # Frame level metrics: AFDR & FPPI
    gt_eval_objects = 0
    matched_gt_objects = 0
    total_fps = 0

    # For HFAR: Count false alarms once per track_id (per sequence), once per frame if track_id is None
    fa_tracks: set[tuple[str, str | int]] = set()  # (sequence_id, track_id)
    fa_untracked_count = 0

    for fr in frame_results:
        # AFDR considers GT objects within 700 m
        for gt, _, _ in fr.matched_gt:
            if gt.range_m <= config.max_eval_range_m:
                gt_eval_objects += 1
                matched_gt_objects += 1
        gt_eval_objects += len(fr.missed_gt)

        total_fps += len(fr.false_positives)

        for det in fr.false_positives:
            if det.track_id is None:
                fa_untracked_count += 1
            else:
                fa_tracks.add((fr.sequence_id, det.track_id))

    afdr = (matched_gt_objects / gt_eval_objects) if gt_eval_objects > 0 else float("nan")
    afdr_ci = clopper_pearson_interval(matched_gt_objects, gt_eval_objects, 1.0 - config.alpha)

    fppi = (total_fps / total_images) if total_images > 0 else float("nan")
    fppi_upper = poisson_rate_upper_bound(total_fps, float(total_images), 1.0 - config.alpha)

    total_false_alarms = len(fa_tracks) + fa_untracked_count
    hfar = (total_false_alarms / flight_hours) if flight_hours > 0.0 else float("nan")
    hfar_upper = poisson_rate_upper_bound(total_false_alarms, flight_hours, 1.0 - config.alpha) if flight_hours > 0.0 else float("nan")

    # Encounter level metrics: EDR
    encounters = evaluate_encounters(frame_results, config)
    valid_encounters = [e for e in encounters if e.is_valid]
    detected_valid = [e for e in valid_encounters if e.is_detected]

    n_valid = len(valid_encounters)
    n_detected = len(detected_valid)
    edr = (n_detected / n_valid) if n_valid > 0 else float("nan")
    edr_ci = clopper_pearson_interval(n_detected, n_valid, 1.0 - config.alpha)

    # Splits
    # 1. By class
    class_splits: dict[str, dict[str, Any]] = {}
    classes = sorted(list({gt.class_id for fr in frame_results for gt in ([p[0] for p in fr.matched_gt] + fr.missed_gt + fr.dont_care_gt)}))
    for c_id in classes:
        c_gt_total = 0
        c_gt_matched = 0
        for fr in frame_results:
            for gt, _, _ in fr.matched_gt:
                if gt.class_id == c_id and gt.range_m <= config.max_eval_range_m:
                    c_gt_total += 1
                    c_gt_matched += 1
            for gt in fr.missed_gt:
                if gt.class_id == c_id:
                    c_gt_total += 1

        c_enc_valid = [e for e in valid_encounters if e.class_id == c_id]
        c_enc_det = [e for e in c_enc_valid if e.is_detected]
        c_afdr = (c_gt_matched / c_gt_total) if c_gt_total > 0 else None
        c_afdr_ci = list(clopper_pearson_interval(c_gt_matched, c_gt_total, 1.0 - config.alpha)) if c_gt_total > 0 else None
        c_edr = (len(c_enc_det) / len(c_enc_valid)) if len(c_enc_valid) > 0 else None

        class_splits[c_id] = {
            "gt_objects_within_700m": c_gt_total,
            "matched_gt_objects": c_gt_matched,
            "afdr": round(c_afdr, 5) if c_afdr is not None else None,
            "afdr_ci_95": [round(x, 5) for x in c_afdr_ci] if c_afdr_ci else None,
            "valid_encounters": len(c_enc_valid),
            "detected_encounters": len(c_enc_det),
            "edr": round(c_edr, 5) if c_edr is not None else None,
        }

    # 2. By range bin
    range_splits: dict[str, dict[str, Any]] = {}
    for r_low, r_high in config.range_bins:
        label = f"[{int(r_low)}, {int(r_high) if r_high != float('inf') else 'inf'})"
        r_total = 0
        r_matched = 0
        for fr in frame_results:
            for gt, _, _ in fr.matched_gt:
                if r_low <= gt.range_m < r_high:
                    r_total += 1
                    r_matched += 1
            for gt in (fr.missed_gt + fr.dont_care_gt):
                if r_low <= gt.range_m < r_high:
                    r_total += 1

        r_recall = (r_matched / r_total) if r_total > 0 else None
        r_ci = list(clopper_pearson_interval(r_matched, r_total, 1.0 - config.alpha)) if r_total > 0 else None
        range_splits[label] = {
            "total_objects": r_total,
            "matched_objects": r_matched,
            "recall": round(r_recall, 5) if r_recall is not None else None,
            "recall_ci_95": [round(x, 5) for x in r_ci] if r_ci else None,
        }

    # 3. By horizon
    horizon_splits: dict[str, dict[str, Any]] = {}
    for h_label in ("above", "below", "unknown"):
        h_total = 0
        h_matched = 0
        for fr in frame_results:
            for gt, _, _ in fr.matched_gt:
                if gt.horizon == h_label and gt.range_m <= config.max_eval_range_m:
                    h_total += 1
                    h_matched += 1
            for gt in fr.missed_gt:
                if gt.horizon == h_label:
                    h_total += 1

        h_afdr = (h_matched / h_total) if h_total > 0 else None
        h_ci = list(clopper_pearson_interval(h_matched, h_total, 1.0 - config.alpha)) if h_total > 0 else None
        horizon_splits[h_label] = {
            "gt_objects_within_700m": h_total,
            "matched_gt_objects": h_matched,
            "afdr": round(h_afdr, 5) if h_afdr is not None else None,
            "afdr_ci_95": [round(x, 5) for x in h_ci] if h_ci else None,
        }

    return AOTMetricsSummary(
        total_images=total_images,
        gt_eval_objects=gt_eval_objects,
        matched_gt_objects=matched_gt_objects,
        afdr=afdr,
        afdr_ci_95=afdr_ci,
        false_positives=total_fps,
        fppi=fppi,
        fppi_upper_95=fppi_upper,
        total_encounters=len(encounters),
        valid_encounters=n_valid,
        detected_valid_encounters=n_detected,
        edr=edr,
        edr_ci_95=edr_ci,
        total_flight_hours=flight_hours,
        false_alarms=total_false_alarms,
        hfar=hfar,
        hfar_upper_95=hfar_upper,
        by_class=class_splits,
        by_range=range_splits,
        by_horizon=horizon_splits,
    )
