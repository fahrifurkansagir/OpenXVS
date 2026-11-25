"""Monitor M0 for OpenXVS WP-07 Integrity Layer.

Implements feature extraction, leave-one-camera-out consistency,
and threshold-based alarm rules for horizon attitude estimation.
Conforms strictly to docs/xvs/WP07_INTEGRITY.md §2 and §5 (V2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Sequence
import numpy as np

from xvs.rig.rig import Rig
from xvs.horizon_att.limb_model import (
    LimbRay,
    LimbRayBatch,
    solve_attitude_gn,
)


@dataclass
class CameraConsistencyResult:
    """Result of camera-consistency evaluation (D27, spec v0.1.3 §2, §5)."""
    max_angle_deg: float
    arg_max_camera: str | None = None
    diffs_deg: dict[str, float] = field(default_factory=dict)
    cam_diffs_deg: dict[str, float] = field(default_factory=dict)
    max_cam_diff_deg: float = float("nan")
    consistency_statistic: float = float("nan")

    def __post_init__(self) -> None:
        if math.isnan(self.consistency_statistic):
            self.consistency_statistic = self.max_angle_deg
        if not self.cam_diffs_deg:
            self.cam_diffs_deg = dict(self.diffs_deg)
        if math.isnan(self.max_cam_diff_deg):
            self.max_cam_diff_deg = self.max_angle_deg

    def __float__(self) -> float:
        return float(self.max_angle_deg)


LeaveOneOutResult = CameraConsistencyResult


def camera_consistency(
    full_fit: dict[str, Any],
    inliers_or_batch: LimbRayBatch,
    rig: Rig,
    lat_deg: float,
    lon_deg: float,
    h_meas_m: float,
    yaw_rad: float,
    max_iter: int = 25,
    tol_rad: float = 1e-10,
    damping: float = 1e-6,
) -> CameraConsistencyResult:
    """Compute camera-consistency attitude statistic across contributing cameras (D27).

    Spec v0.1.3 §2: camera c alone against the rest.
    - Partition the full fit's RANSAC inliers by camera.
    - Fit camera c's inliers, and separately the other cameras' inliers, with the WP-06
      refinement solver, seeded from the full fit. No new RANSAC and no raw edges.
    - The value exists only when c has >= 10 inliers and the rest still holds a camera
      with >= 10 inliers. Otherwise it is null (NaN / None), never a number.
    - The feature is the maximum over c, plus the arg-max camera.

    Args:
        full_fit: Full multi-camera fit dict (with pitch_deg, roll_deg, and optional inliers).
        inliers_or_batch: LimbRayBatch of RANSAC inliers (or full LimbRayBatch sliced by full_fit['inliers']).
        rig: Rig sensor model.
        lat_deg, lon_deg: Aircraft position.
        h_meas_m: Altitude above WGS84 ellipsoid.
        yaw_rad: Aircraft heading.
        max_iter: Maximum GN iterations.
        tol_rad: GN convergence tolerance.
        damping: LM diagonal damping.

    Returns:
        CameraConsistencyResult containing max_angle_deg, arg_max_camera, and per-camera diffs.
    """
    if not isinstance(inliers_or_batch, LimbRayBatch):
        raise TypeError(
            "camera_consistency requires a LimbRayBatch of inliers (D27: no raw edge dicts allowed)."
        )

    batch_all = inliers_or_batch
    if "inliers" in full_fit and len(full_fit["inliers"]) < batch_all.n_rays:
        idx = full_fit["inliers"]
        batch_all = LimbRayBatch(
            batch_all.d_body[idx],
            batch_all.lever_arm_body[idx],
            [batch_all.cam_ids[i] for i in idx],
        )

    inlier_counts: dict[str, int] = {}
    for cid in batch_all.cam_ids:
        inlier_counts[cid] = inlier_counts.get(cid, 0) + 1  # default-ok: frequency counter accumulator

    # D27: Value exists only when c has >= 10 inliers and the rest still holds a camera with >= 10 inliers.
    # This requires at least 2 distinct cameras each with >= 10 inliers.
    valid_cams = sorted([c for c, cnt in inlier_counts.items() if cnt >= 10])

    if len(valid_cams) < 2:
        return CameraConsistencyResult(
            max_angle_deg=float("nan"),
            arg_max_camera=None,
            diffs_deg={},
            cam_diffs_deg={},
            max_cam_diff_deg=float("nan"),
            consistency_statistic=float("nan"),
        )

    p_full = float(full_fit["pitch_deg"])
    r_full = float(full_fit["roll_deg"])
    p_init_rad = full_fit.get("pitch_rad", math.radians(p_full))
    r_init_rad = full_fit.get("roll_rad", math.radians(r_full))

    diffs_cam: dict[str, float] = {}

    for c in valid_cams:
        # Camera c alone
        mask_c = [cid == c for cid in batch_all.cam_ids]
        batch_c = LimbRayBatch(
            batch_all.d_body[mask_c],
            batch_all.lever_arm_body[mask_c],
            [cid for cid, m in zip(batch_all.cam_ids, mask_c) if m],
        )

        # The rest of the cameras (excluding c)
        mask_rest = [cid != c for cid in batch_all.cam_ids]
        batch_rest = LimbRayBatch(
            batch_all.d_body[mask_rest],
            batch_all.lever_arm_body[mask_rest],
            [cid for cid, m in zip(batch_all.cam_ids, mask_rest) if m],
        )

        if batch_c.n_rays >= 3 and batch_rest.n_rays >= 3:
            fit_c = solve_attitude_gn(
                lat_deg, lon_deg, h_meas_m, yaw_rad,
                p_init_rad, r_init_rad,
                batch_c, max_iter=max_iter, tol_rad=tol_rad, damping=damping,
            )
            fit_rest = solve_attitude_gn(
                lat_deg, lon_deg, h_meas_m, yaw_rad,
                p_init_rad, r_init_rad,
                batch_rest, max_iter=max_iter, tol_rad=tol_rad, damping=damping,
            )

            if fit_c.get("status") == "CONVERGED" and fit_rest.get("status") == "CONVERGED":
                dp = fit_c["pitch_deg"] - fit_rest["pitch_deg"]
                dr = fit_c["roll_deg"] - fit_rest["roll_deg"]
                diffs_cam[c] = float(math.hypot(dp, dr))
            else:
                diffs_cam[c] = float("nan")
        else:
            diffs_cam[c] = float("nan")

    valid_diffs = {c: v for c, v in diffs_cam.items() if not math.isnan(v)}
    if not valid_diffs:
        return CameraConsistencyResult(
            max_angle_deg=float("nan"),
            arg_max_camera=None,
            diffs_deg=diffs_cam,
            cam_diffs_deg=diffs_cam,
            max_cam_diff_deg=float("nan"),
            consistency_statistic=float("nan"),
        )

    arg_max_cam = max(valid_diffs, key=valid_diffs.get)
    max_angle = valid_diffs[arg_max_cam]

    return CameraConsistencyResult(
        max_angle_deg=float(max_angle),
        arg_max_camera=arg_max_cam,
        diffs_deg=diffs_cam,
        cam_diffs_deg=diffs_cam,
        max_cam_diff_deg=float(max_angle),
        consistency_statistic=float(max_angle),
    )


compute_leave_one_out_consistency = camera_consistency


@dataclass
class MonitorThresholds:
    """Threshold configuration for Monitor M0 alarm evaluator."""
    far_target: float
    max_rms_residual_urad: float | None = None
    max_condition_number: float | None = None
    min_inlier_fraction: float | None = None
    min_inlier_count: int | None = None
    max_leave_one_out_deg: float | None = None
    by_cam_set: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "far_target": self.far_target,
            "max_rms_residual_urad": self.max_rms_residual_urad,
            "max_condition_number": self.max_condition_number,
            "min_inlier_fraction": self.min_inlier_fraction,
            "min_inlier_count": self.min_inlier_count,
            "max_leave_one_out_deg": self.max_leave_one_out_deg,
            "by_cam_set": self.by_cam_set,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MonitorThresholds:
        return cls(
            far_target=float(data.get("far_target", 0.01)),  # default-ok: optional fallback default
            max_rms_residual_urad=float(data["max_rms_residual_urad"]) if data.get("max_rms_residual_urad") is not None else None,
            max_condition_number=float(data["max_condition_number"]) if data.get("max_condition_number") is not None else None,
            min_inlier_fraction=float(data["min_inlier_fraction"]) if data.get("min_inlier_fraction") is not None else None,
            min_inlier_count=int(data["min_inlier_count"]) if data.get("min_inlier_count") is not None else None,
            max_leave_one_out_deg=float(data["max_leave_one_out_deg"]) if data.get("max_leave_one_out_deg") is not None else None,
            by_cam_set=data.get("by_cam_set", {}),
        )


def evaluate_monitor_alarm(
    record: dict[str, Any],
    thresholds: MonitorThresholds | dict[str, Any],
) -> dict[str, Any]:
    """Evaluate Monitor M0 alarm rule on a single attitude estimate record.

    An alarm is raised if fit status is not CONVERGED, or if any monitor feature
    exceeds its configured threshold.

    Args:
        record: Horizon attitude estimation record containing monitor features.
        thresholds: MonitorThresholds instance or dict.

    Returns:
        dict containing:
            alarm: bool
            reasons: list of triggered failure codes
    """
    if isinstance(thresholds, dict):
        th = MonitorThresholds.from_dict(thresholds)
    else:
        th = thresholds

    reasons: list[str] = []

    # 1. Status check
    status = record.get("status", "")
    if status != "CONVERGED":
        reasons.append(f"STATUS_{status or 'NOT_CONVERGED'}")

    cam_set = record.get("cam_set", "")
    th_cam = th.by_cam_set.get(cam_set, {}) if th.by_cam_set else {}

    max_rms = th_cam.get("max_rms_residual_urad", th.max_rms_residual_urad)
    max_cond = th_cam.get("max_condition_number", th.max_condition_number)
    min_frac = th_cam.get("min_inlier_fraction", th.min_inlier_fraction)
    min_cnt = th_cam.get("min_inlier_count", th.min_inlier_count)
    if min_cnt is not None:
        min_cnt = int(min_cnt)
    max_loo = th_cam.get("max_leave_one_out_deg", th.max_leave_one_out_deg)

    # 2. RMS residual check
    if max_rms is not None:
        rms = record.get("rms_residual_urad", float("nan"))
        if math.isnan(rms) or rms > max_rms:
            reasons.append(f"RMS_HIGH_{rms:.1f}urad")

    # 3. Condition number check
    if max_cond is not None:
        cond = record.get("condition_number", float("nan"))
        if math.isnan(cond) or cond > max_cond:
            reasons.append(f"COND_HIGH_{cond:.1e}")

    # 4. Inlier fraction check
    if min_frac is not None:
        inlier_frac = record.get("inlier_fraction", float("nan"))
        if math.isnan(inlier_frac) or inlier_frac < min_frac:
            reasons.append(f"INLIER_FRAC_LOW_{inlier_frac:.2f}")

    # 5. Inlier count check
    if min_cnt is not None:
        inlier_cnt = record.get("inlier_count", 0)  # default-ok: optional fallback default
        if inlier_cnt < min_cnt:
            reasons.append(f"INLIER_COUNT_LOW_{inlier_cnt}")

    # 6. Camera consistency check (FRBL/horizontal and all sets only, D27)
    if cam_set in ("FRBL", "horizontal", "all") and max_loo is not None:
        loo = record.get("camera_consistency_deg")
        if loo is None or (isinstance(loo, float) and math.isnan(loo)):
            loo = record.get("leave_one_out_consistency_deg", float("nan"))
        # D27 / F-B12-2: If camera consistency is not available (null/NaN), it does not trigger an alarm.
        if (
            loo is not None
            and not math.isnan(loo)
            and not math.isnan(max_loo)
            and loo > max_loo
        ):
            reasons.append(f"LOO_HIGH_{loo:.4f}deg")

    is_alarmed = len(reasons) > 0
    return {
        "alarm": is_alarmed,
        "reasons": reasons,
    }


def tune_monitor_thresholds(
    nominal_val_records: list[dict[str, Any]],
    target_far: float = 0.01,
) -> MonitorThresholds:
    """Tune Monitor M0 thresholds on nominal validation records for a target FAR.

    Per D28 / W2:
    - Per camera set, each feature's threshold is the empirical FAR/m tail quantile of fault-free validation frames,
      upper or lower according to the feature's alarm side.
    - m is the number of features available for that camera set (4 for F, 5 for FRBL/all).
    - Remove the x1.05, x1.10 and x0.90 margins and the floors.

    Args:
        nominal_val_records: List of fault-free records from validation split.
        target_far: Target false alarm rate (e.g. 0.01 or 0.001).

    Returns:
        Tuned MonitorThresholds object.
    """
    if not nominal_val_records:
        raise ValueError("nominal_val_records is empty; cannot tune monitor thresholds.")

    converged_recs = [r for r in nominal_val_records if r.get("status") == "CONVERGED"]
    if not converged_recs:
        raise ValueError("No CONVERGED records in nominal_val_records; cannot tune monitor thresholds.")

    def _get_cc(r: dict[str, Any]) -> float:
        v = r.get("camera_consistency_deg")
        if v is None or (isinstance(v, float) and math.isnan(v)):
            v = r.get("leave_one_out_consistency_deg", float("nan"))
        return float(v) if v is not None else float("nan")

    # Validate feature presence (F-B12-3: missing features must raise)
    for f in ("rms_residual_urad", "condition_number", "inlier_fraction", "inlier_count"):
        if any(f not in r for r in converged_recs):
            raise ValueError(f"Missing feature in validation records: {f} list is empty or missing.")

    multi_cam_recs = [r for r in converged_recs if r.get("cam_set") in ("FRBL", "horizontal", "all")]
    if multi_cam_recs:
        if any("camera_consistency_deg" not in r and "leave_one_out_consistency_deg" not in r for r in multi_cam_recs):
            raise ValueError("Missing feature in validation records: leave_one_out_consistency_deg list is empty.")

    # 1. Compute per-camera-set thresholds
    by_cam_set: dict[str, dict[str, float]] = {}
    cam_sets = set(r.get("cam_set", "all") for r in converged_recs)

    for cset in sorted(cam_sets):
        c_recs = [r for r in converged_recs if r.get("cam_set", "all") == cset]
        if not c_recs:
            continue
        m = 4 if cset == "F" else 5
        tail_prob = max(1e-5, target_far / m)
        q_upper = 100.0 * (1.0 - tail_prob)
        q_lower = 100.0 * tail_prob

        rms_vals = [r["rms_residual_urad"] for r in c_recs if not math.isnan(r.get("rms_residual_urad", float("nan")))]
        cond_vals = [r["condition_number"] for r in c_recs if not math.isnan(r.get("condition_number", float("nan")))]
        frac_vals = [r["inlier_fraction"] for r in c_recs if not math.isnan(r.get("inlier_fraction", float("nan")))]
        cnt_vals = [r["inlier_count"] for r in c_recs if "inlier_count" in r]

        th_c_rms = float(np.percentile(rms_vals, q_upper)) if rms_vals else float("nan")
        th_c_cond = float(np.percentile(cond_vals, q_upper)) if cond_vals else float("nan")
        th_c_frac = float(np.percentile(frac_vals, q_lower)) if frac_vals else float("nan")
        th_c_cnt = int(np.percentile(cnt_vals, q_lower)) if cnt_vals else 0

        th_c_loo: float | None = None
        loo_reason: str | None = None
        if m == 5:
            loo_vals = [_get_cc(r) for r in c_recs if not math.isnan(_get_cc(r))]
            if loo_vals:
                th_c_loo = float(np.percentile(loo_vals, q_upper))
        else:
            loo_reason = "Single-camera set F (c=1): leave-one-camera-out consistency requires c>=2 (D27)"

        cam_entry: dict[str, Any] = {
            "max_rms_residual_urad": th_c_rms,
            "max_condition_number": th_c_cond,
            "min_inlier_fraction": th_c_frac,
            "min_inlier_count": th_c_cnt,
            "max_leave_one_out_deg": th_c_loo,
        }
        if loo_reason is not None:
            cam_entry["max_leave_one_out_deg_reason"] = loo_reason
        by_cam_set[cset] = cam_entry

    # 2. Overall pooled thresholds across all converged records
    tail_prob_overall = max(1e-5, target_far / 5)
    q_upper_all = 100.0 * (1.0 - tail_prob_overall)
    q_lower_all = 100.0 * tail_prob_overall

    all_rms = [r["rms_residual_urad"] for r in converged_recs if not math.isnan(r.get("rms_residual_urad", float("nan")))]
    all_cond = [r["condition_number"] for r in converged_recs if not math.isnan(r.get("condition_number", float("nan")))]
    all_frac = [r["inlier_fraction"] for r in converged_recs if not math.isnan(r.get("inlier_fraction", float("nan")))]
    all_cnt = [r["inlier_count"] for r in converged_recs if "inlier_count" in r]
    all_loo = [_get_cc(r) for r in converged_recs if r.get("cam_set") in ("FRBL", "horizontal", "all") and not math.isnan(_get_cc(r))]

    th_rms = float(np.percentile(all_rms, q_upper_all))
    th_cond = float(np.percentile(all_cond, q_upper_all))
    th_frac = float(np.percentile(all_frac, q_lower_all))
    th_cnt = int(np.percentile(all_cnt, q_lower_all))
    th_loo = float(np.percentile(all_loo, q_upper_all)) if all_loo else float("nan")

    return MonitorThresholds(
        far_target=target_far,
        max_rms_residual_urad=th_rms,
        max_condition_number=th_cond,
        min_inlier_fraction=th_frac,
        min_inlier_count=th_cnt,
        max_leave_one_out_deg=th_loo,
        by_cam_set=by_cam_set,
    )


def __getattr__(name: str) -> Any:
    if name in (
        "ALTITUDE_BANDS",
        "MonitorThresholdsG1",
        "classify_altitude_band",
        "compute_horizon_strip_ood_features",
        "evaluate_monitor_alarm_g1",
        "isolate_and_exclude_faulty_camera",
        "tune_monitor_thresholds_g1",
    ):
        from xvs.integrity import g1_extensions
        return getattr(g1_extensions, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

