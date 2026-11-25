"""Evaluation and metric computation for WP-05 G0 multi-camera tracking.

Conforms strictly to docs/xvs/WP05_TRACKING.md §4 and GEMINI_G-B3.md §1 items 5 and 6.
Evaluates:
- Tracking acquisition delay, fragmentations, ID switches, seam crossings, and false tracks;
- Earth-fixed LOS rate accuracy (R7-P2) by range bin, intruder class, and track age;
- Time-to-contact (TTC) accuracy against true CPA time;
- Alert lead times (R7-P1) for warning (25s) and corrective (55s) alerts;
- False alert rates for non-threat encounters;
- 95% Clopper-Pearson confidence intervals for all proportions.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Sequence
import numpy as np

from xvs.geo import wgs84
from xvs.metrics.aot import clopper_pearson_interval
from xvs.sim.trajectory import t1_cruise_state, AircraftState
from xvs.traffic.encounter import (
    EncounterParams,
    construct_encounter,
    evaluate_events,
    evaluate_actual_cpa,
)
from xvs.traffic.ring import TrafficRing
from xvs.tracking.measurements import SyntheticMeasurementGenerator, DetectionRecord
from xvs.tracking.tracker import MultiCameraTracker, Track, TrackStatus
from xvs.tracking.risk import (
    evaluate_track_risk,
    evaluate_wc1_for_beta,
    RiskEstimate,
    LOWC_HMD_LIMIT_M,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "xvs" / "tracking" / "g0.json"
BETA_WC_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

RANGE_BINS_M = [
    ("0_1000m", 0.0, 1000.0),
    ("1000_2000m", 1000.0, 2000.0),
    ("2000_3000m", 2000.0, 3000.0),
    ("3000_5000m", 3000.0, 5000.0),
    ("5000m_plus", 5000.0, float("inf")),
]

TRACK_AGE_BINS_S = [
    ("age_0_3s", 0.0, 3.0),
    ("age_3_10s", 3.0, 10.0),
    ("age_10s_plus", 10.0, float("inf")),
]


def get_range_bin(r_m: float) -> str:
    for name, r_min, r_max in RANGE_BINS_M:
        if r_min <= r_m < r_max:
            return name
    return "5000m_plus"


def get_age_bin(age_s: float) -> str:
    for name, a_min, a_max in TRACK_AGE_BINS_S:
        if a_min <= age_s < a_max:
            return name
    return "age_10s_plus"


# Module-level cache for precomputed ownship states (601 frames at 10 Hz)
_CACHED_OWNSHIP_STATES: list[AircraftState] | None = None
TRUTH_CACHE_VERSION = "1"
OWNSHIP_SCENARIO_ID = "t1_cruise"
OWNSHIP_CONFIG_PATH = REPO_ROOT / "configs" / "xvs" / "scenarios" / "t1_cruise.json"
_OVERRIDE_CACHE_DIR: Path | None = None


def set_truth_cache_dir(path: Path | str | None) -> None:
    """Override truth cache directory (primarily for tests)."""
    global _OVERRIDE_CACHE_DIR
    _OVERRIDE_CACHE_DIR = Path(path) if path is not None else None


def get_truth_cache_dir() -> Path:
    """Return truth cache directory under data root or override."""
    if _OVERRIDE_CACHE_DIR is not None:
        return _OVERRIDE_CACHE_DIR
    from xvs.paths import get_data_root
    return get_data_root() / "xvs" / "wp05" / "g0" / "cache"


def get_ownship_config_sha256() -> str:
    """Return LF SHA-256 of the ownship scenario config."""
    if OWNSHIP_CONFIG_PATH.is_file():
        raw = OWNSHIP_CONFIG_PATH.read_bytes().replace(b"\r\n", b"\n")
        return hashlib.sha256(raw).hexdigest()
    raise FileNotFoundError(f"Ownship scenario config not found at {OWNSHIP_CONFIG_PATH}")


def compute_truth_cache_key(
    encounters: Sequence[Any],
    ownship_scenario_id: str = OWNSHIP_SCENARIO_ID,
    ownship_config_sha256: str | None = None,
    t_start: float = 0.0,
    t_end: float = 60.0,
    dt: float = 0.01,
    version: str = TRUTH_CACHE_VERSION,
) -> str:
    """Compute SHA-256 over:
    - LF bytes of the encounter list;
    - ownship scenario id and its config SHA-256;
    - t_start, t_end and dt;
    - TRUTH_CACHE_VERSION.
    """
    if ownship_config_sha256 is None:
        ownship_config_sha256 = get_ownship_config_sha256()

    hasher = hashlib.sha256()
    for e in encounters:
        d = asdict(e) if hasattr(e, "__dataclass_fields__") else (asdict(e) if hasattr(e, "id") and hasattr(e, "class_id") else dict(e))
        line = json.dumps(d, sort_keys=True) + "\n"
        hasher.update(line.encode("utf-8"))

    hasher.update(f"{ownship_scenario_id}\n{ownship_config_sha256}\n".encode("utf-8"))
    hasher.update(f"{t_start:.6f}:{t_end:.6f}:{dt:.6f}\n".encode("utf-8"))
    hasher.update(f"{version}\n".encode("utf-8"))
    return hasher.hexdigest()


def load_or_compute_truth_cache(
    encounters: Sequence[Any],
    max_workers: int = 4,
    cache_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Load or recompute truth cache.
    Returns (truth_cache_map, {"status": status, "reason": reason}).
    No bare except.
    """
    if cache_dir is None:
        cache_dir = get_truth_cache_dir()

    key = compute_truth_cache_key(encounters)
    key16 = key[:16]
    cache_file = cache_dir / f"truth_{key16}.json"

    status = "miss"
    reason = "cache_not_found"

    if cache_file.is_file():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                raise ValueError("Cache payload is not a JSON dictionary")
            cached_key = payload.get("key")
            if cached_key != key:
                status = "recomputed"
                reason = f"key_mismatch: got {cached_key[:16] if cached_key else 'none'}, expected {key16}"
            else:
                truth_cache_map = payload.get("encounters", {})
                missing_eids = [
                    (e.id if hasattr(e, "id") else e["id"])
                    for e in encounters
                    if (e.id if hasattr(e, "id") else e["id"]) not in truth_cache_map
                ]
                if missing_eids:
                    status = "recomputed"
                    reason = f"incomplete_cache: {len(missing_eids)} encounters missing"
                else:
                    return truth_cache_map, {"status": "hit", "reason": "valid_cache_key_matched"}
        except (json.JSONDecodeError, KeyError, ValueError, OSError, TypeError) as exc:
            status = "recomputed"
            reason = f"corrupt_cache: {type(exc).__name__}: {exc}"

    # Recompute missing or all
    tasks = [
        asdict(e) if hasattr(e, "__dataclass_fields__") else (asdict(e) if hasattr(e, "id") else dict(e))
        for e in encounters
    ]
    if max_workers > 1:
        from multiprocessing import Pool
        with Pool(processes=max_workers) as pool:
            computed = pool.map(compute_encounter_truth, tasks)
    else:
        computed = [compute_encounter_truth(m) for m in tasks]

    truth_cache_map = {eid: tdict for eid, tdict in computed}

    # Save to cache
    save_failed = False
    save_error = None
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": TRUTH_CACHE_VERSION,
            "key": key,
            "ownship_scenario_id": OWNSHIP_SCENARIO_ID,
            "t_start": 0.0,
            "t_end": 60.0,
            "dt": 0.01,
            "encounters": truth_cache_map,
        }
        tmp_file = cache_dir / f"truth_{key16}.tmp_{os.getpid()}"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        tmp_file.replace(cache_file)
    except OSError as exc:
        save_failed = True
        save_error = str(exc)

    info = {"status": status, "reason": reason, "save_failed": save_failed}
    if save_error is not None:
        info["save_error"] = save_error
    return truth_cache_map, info


def generate_truth_labels(
    encounters: Sequence[Any],
    output_path: Path | str | None = None,
    max_workers: int = 4,
) -> Path:
    """Generate truth_labels.json containing exact truth events and CPA for all encounters.
    Holds per encounter: id, threat, lowc_t, nmac_t, t_cpa, hmd, along with content key.
    """
    if output_path is None:
        from xvs.paths import get_data_root
        output_path = get_data_root() / "xvs" / "wp05" / "g0" / "truth_labels.json"
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    key = compute_truth_cache_key(encounters)
    truth_map, _ = load_or_compute_truth_cache(encounters, max_workers=max_workers)

    records = {}
    for e in encounters:
        eid = e.id if hasattr(e, "id") else e["id"]
        tdict = truth_map.get(eid, {})
        ev = tdict.get("events", {})
        cpa = tdict.get("cpa", {})
        is_threat = (ev.get("lowc_t") is not None)
        records[eid] = {
            "id": eid,
            "threat": is_threat,
            "lowc_t": ev.get("lowc_t"),
            "nmac_t": ev.get("nmac_t"),
            "t_cpa": cpa["t_cpa_s"],
            "hmd": cpa["hmd_m"],
        }

    payload = {
        "key": key,
        "encounters_total": len(records),
        "threats_count": sum(1 for r in records.values() if r["threat"]),
        "encounters": records,
    }

    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return p


def get_ownship_states() -> list[AircraftState]:
    global _CACHED_OWNSHIP_STATES
    if _CACHED_OWNSHIP_STATES is None:
        times = [k * 0.1 for k in range(601)]
        _CACHED_OWNSHIP_STATES = [t1_cruise_state(t) for t in times]
    return _CACHED_OWNSHIP_STATES


def compute_encounter_truth(params_dict: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Compute exact ground truth events and CPA for an encounter once."""
    p = EncounterParams(**params_dict)
    enc = construct_encounter(t1_cruise_state, p)
    ev = evaluate_events(t1_cruise_state, enc, t_start=0.0, t_end=60.0, dt=0.01)
    cpa = evaluate_actual_cpa(t1_cruise_state, enc, t_start=0.0, t_end=60.0, dt=0.01)
    return p.id, {"events": ev, "cpa": cpa}


def compute_encounter_label(params_dict: dict[str, Any]) -> tuple[str, bool]:
    """Compute only the event label (is_threat) for an encounter."""
    p = EncounterParams(**params_dict)
    enc = construct_encounter(t1_cruise_state, p)
    ev = evaluate_events(t1_cruise_state, enc, t_start=0.0, t_end=60.0, dt=0.01)
    return p.id, (ev["lowc_t"] is not None)


def simulate_single_encounter(
    params_dict: dict[str, Any],
    config: dict[str, Any],
    seed: int,
    truth_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Top-level worker function to simulate and score a single encounter."""
    params = EncounterParams(**params_dict)
    enc = construct_encounter(t1_cruise_state, params)

    # 1. Ground truth events and CPA (cached or evaluated)
    if truth_cache is not None:
        ev = truth_cache["events"]
        cpa = truth_cache["cpa"]
    else:
        ev = evaluate_events(t1_cruise_state, enc, t_start=0.0, t_end=60.0, dt=0.01)
        cpa = evaluate_actual_cpa(t1_cruise_state, enc, t_start=0.0, t_end=60.0, dt=0.01)

    lowc_t = ev["lowc_t"]
    nmac_t = ev["nmac_t"]
    is_threat = (lowc_t is not None)
    t_cpa_s = cpa["t_cpa_s"]
    hmd_m = cpa["hmd_m"]

    # 2. Setup generator and tracker
    ring = TrafficRing()
    gen = SyntheticMeasurementGenerator(config=config, seed=seed, ring=ring)
    tracker = MultiCameraTracker(config=config, ring=ring)

    ownship_states = get_ownship_states()
    latency_ms = float(config["latency_ms"])
    latency_frames = int(round(latency_ms / (1000.0 * tracker.dt)))

    # Delayed detection queue for latency simulation
    delayed_queue: list[tuple[float, list[DetectionRecord], np.ndarray, np.ndarray, np.ndarray]] = []

    first_detectable_t: float | None = None
    first_confirmed_t: float | None = None

    los_rate_errors: list[dict[str, Any]] = []
    ttc_errors: list[dict[str, Any]] = []

    first_alert_a1_warn_t: float | None = None
    first_alert_a1_corr_t: float | None = None
    first_alert_a2_warn_t: float | None = None
    first_alert_a2_corr_t: float | None = None
    first_alert_a2_warn_ck_t: float | None = None
    first_alert_a2_corr_ck_t: float | None = None

    first_alert_wc1_warn_t: float | None = None
    first_alert_wc1_corr_t: float | None = None
    first_alert_wc1_warn_ck_t: float | None = None
    first_alert_wc1_corr_ck_t: float | None = None
    first_alert_wc1_warn_by_beta: dict[str, float | None] = {f"{b:.2f}": None for b in BETA_WC_GRID}
    first_alert_wc1_corr_by_beta: dict[str, float | None] = {f"{b:.2f}": None for b in BETA_WC_GRID}

    has_fa_a1_warn = False
    has_fa_a1_corr = False
    has_fa_a2_warn = False
    has_fa_a2_corr = False
    has_fa_a2_warn_ck = False
    has_fa_a2_corr_ck = False

    has_fa_wc1_warn = False
    has_fa_wc1_corr = False
    has_fa_wc1_warn_ck = False
    has_fa_wc1_corr_ck = False
    has_fa_wc1_warn_by_beta: dict[str, bool] = {f"{b:.2f}": False for b in BETA_WC_GRID}
    has_fa_wc1_corr_by_beta: dict[str, bool] = {f"{b:.2f}": False for b in BETA_WC_GRID}

    fa_a1_warn_events = 0
    fa_a1_corr_events = 0
    fa_a2_warn_events = 0
    fa_a2_corr_events = 0
    fa_a2_warn_ck_events = 0
    fa_a2_corr_ck_events = 0
    fa_wc1_warn_events = 0
    fa_wc1_corr_events = 0
    fa_wc1_warn_ck_events = 0
    fa_wc1_corr_ck_events = 0
    fa_wc1_warn_events_by_beta: dict[str, int] = {f"{b:.2f}": 0 for b in BETA_WC_GRID}
    fa_wc1_corr_events_by_beta: dict[str, int] = {f"{b:.2f}": 0 for b in BETA_WC_GRID}
    prev_fa_by_track: dict[str, dict[str, bool]] = {}

    # Track primary confirmed track for seam analysis
    prev_cam: str | None = None
    seam_crossings = 0
    seam_breaks = 0
    unique_confirmed_ids: set[str] = set()
    detectable_frames = 0
    confirmed_on_detectable_frames = 0

    omega_thr_radps = float(config["alert_omega_thr_radps"])
    k_sigma = float(config["ttc_k_sigma"])
    hmd_limit_m = float(config["hmd_lowc_limit_m"])
    t_warning_s = float(config["alert_t_warning_s"])
    t_corrective_s = float(config["alert_t_corrective_s"])
    beta_wc = float(config.get("beta_wc", 0.5))  # default-ok: G1 default conservatism weight
    k_theta = float(config.get("k_theta", 1.0))  # default-ok: G1 range interval k_theta
    k_omega = float(config.get("k_omega", 1.0))  # default-ok: G1 HMD interval k_omega
    alpha_r = float(config.get("alpha_r", 0.0))  # default-ok: G1 prior quantile alpha_r

    k_check = max(0, min(len(ownship_states) - 1, int(round((t_cpa_s - 12.5) / tracker.dt))))
    miss_check_at_tcpa_minus_12_5: dict[str, Any] = {}

    for k, st_o in enumerate(ownship_states):
        t_s = k * tracker.dt

        # Generate frame
        dets, meta = gen.generate_frame(t_s, st_o, [enc])
        r_ned_ins = meta["r_ned_body_ins"]
        p_own_ecef = meta["p_own_ecef"]
        r_ecef_ned = meta["r_ecef_ned"]

        # Check detectable condition (P_d > 0.5)
        gt_info = meta["gt_by_intruder"][params.id]
        is_detectable = False
        for cam_id, cam_gt in gt_info["cams"].items():
            s_px = cam_gt["s_px"]
            if gen.detection_probability(s_px) > 0.5:
                is_detectable = True
                if first_detectable_t is None:
                    first_detectable_t = t_s
                break
        if is_detectable:
            detectable_frames += 1

        # Queue detections for latency
        delayed_queue.append((t_s, dets, r_ned_ins, p_own_ecef, r_ecef_ned))

        if k >= latency_frames:
            pop_t, pop_dets, pop_r_ins, pop_p_own, pop_r_ecef_ned = delayed_queue[k - latency_frames]
            confirmed_tracks = tracker.step(
                t_s, pop_dets, pop_r_ins, p_own_ecef=pop_p_own, r_ecef_ned=pop_r_ecef_ned
            )
        else:
            confirmed_tracks = tracker.step(
                t_s, [], r_ned_ins, p_own_ecef=p_own_ecef, r_ecef_ned=r_ecef_ned
            )

        # Match confirmed tracks associated with this intruder
        target_tracks = [
            trk for trk in confirmed_tracks
            if trk.last_associated_truth_id == params.id
        ]

        if target_tracks and is_detectable:
            confirmed_on_detectable_frames += 1

        if k == k_check:
            p_i_chk = enc.position_at(t_s)
            p_arp_chk = st_o.p_ecef_m
            r_ecef_ned_chk = wgs84.r_ecef_ned(st_o.lat_deg, st_o.lon_deg)
            r_ecef_chk = p_i_chk - p_arp_chk
            v_rel_ecef_chk = enc.velocity_at(t_s) - (r_ecef_ned_chk @ st_o.v_ned_mps)
            r_ned_chk = r_ecef_ned_chk.T @ r_ecef_chk
            v_rel_ned_chk = r_ecef_ned_chk.T @ v_rel_ecef_chk
            r_N_c, r_E_c, r_D_c = float(r_ned_chk[0]), float(r_ned_chk[1]), float(r_ned_chk[2])
            v_N_c, v_E_c, v_D_c = float(v_rel_ned_chk[0]), float(v_rel_ned_chk[1]), float(v_rel_ned_chk[2])
            r_xy_c = math.hypot(r_N_c, r_E_c)
            range_c = float(np.linalg.norm(r_ned_chk))
            if r_xy_c > 1e-6 and range_c > 1e-6:
                az_r_gt_c = (r_N_c * v_E_c - r_E_c * v_N_c) / (r_xy_c * r_xy_c)
                el_r_gt_c = (-r_xy_c * v_D_c + r_D_c * (r_N_c * v_N_c + r_E_c * v_E_c) / r_xy_c) / (range_c * range_c)
            else:
                az_r_gt_c = 0.0
                el_r_gt_c = 0.0
            omega_true_chk = math.hypot(az_r_gt_c, el_r_gt_c)
            tau_true_chk = max(0.0, t_cpa_s - t_s)
            tau_true_ge_T = bool(tau_true_chk >= t_warning_s)
            omega_true_ge_thr = bool(omega_true_chk >= omega_thr_radps)

            # Exact ground-truth passive WC1 observables at t_check
            r_dot_true_chk = float(np.dot(r_ned_chk, v_rel_ned_chk)) / max(1e-6, range_c)
            eta_true_chk = max(1e-6, -r_dot_true_chk / max(1e-6, range_c))
            hmd_true_passive_chk = range_c * omega_true_chk / math.hypot(eta_true_chk, omega_true_chk)
            if range_c <= hmd_limit_m:
                tau_mod_true_chk = 0.0
            else:
                tau_mod_true_chk = max(0.0, (1.0 / eta_true_chk) * (1.0 - (hmd_limit_m * hmd_limit_m) / (range_c * range_c)))
            wc1_tau_mod_true_gt_T = bool(tau_mod_true_chk > t_warning_s)
            wc1_hmd_true_gt_d_thr = bool(hmd_true_passive_chk > hmd_limit_m)

            if not target_tracks:
                no_confirmed_track = True
                tau_s_unavailable = False
                tau_s_ge_T = False
                omega_ge_thr = False
                tau_est_chk = None
                omega_est_chk = None
                wc1_ttc_unavailable = False
                wc1_tau_mod_hat_gt_T = False
                wc1_hmd_hat_gt_d_thr = False
                wc1_tau_mod_hat_chk = None
                wc1_hmd_hat_chk = None
            else:
                pt_chk = target_tracks[0]
                no_confirmed_track = False
                risk_chk = evaluate_track_risk(
                    pt_chk,
                    t_s=t_s,
                    omega_thr_radps=omega_thr_radps,
                    k_sigma=k_sigma,
                    hmd_limit_m=hmd_limit_m,
                    t_warning_s=t_warning_s,
                    t_corrective_s=t_corrective_s,
                    class_id=None,
                    beta_wc=beta_wc,
                    k_theta=k_theta,
                    k_omega=k_omega,
                    alpha_r=alpha_r,
                )
                tau_est_chk = risk_chk.tau_s
                omega_est_chk = risk_chk.omega_los_radps
                tau_s_unavailable = (tau_est_chk is None)
                tau_s_ge_T = (tau_est_chk is not None and tau_est_chk >= t_warning_s)
                omega_ge_thr = (omega_est_chk >= omega_thr_radps)

                wc1_tau_mod_hat_chk = risk_chk.tau_mod_hat_s
                wc1_hmd_hat_chk = risk_chk.hmd_hat_m
                wc1_ttc_unavailable = (wc1_tau_mod_hat_chk is None or wc1_hmd_hat_chk is None)
                wc1_tau_mod_hat_gt_T = (wc1_tau_mod_hat_chk is not None and wc1_tau_mod_hat_chk > t_warning_s)
                wc1_hmd_hat_gt_d_thr = (wc1_hmd_hat_chk is not None and wc1_hmd_hat_chk > hmd_limit_m)

            failed_conds = []
            if no_confirmed_track:
                failed_conds.append("no_confirmed_track")
            if tau_s_unavailable:
                failed_conds.append("tau_s_unavailable")
            if tau_s_ge_T:
                failed_conds.append("tau_s_ge_T")
            if omega_ge_thr:
                failed_conds.append("omega_ge_thr")

            joint_key = "+".join(failed_conds) if failed_conds else "none_failed_at_t_check"
            tau_failed = (tau_s_unavailable or tau_s_ge_T)
            omega_failed = omega_ge_thr
            tau_class = ("rule_limited" if tau_true_ge_T else "estimation_limited") if tau_failed else "passed_or_no_track"
            omega_class = ("rule_limited" if omega_true_ge_thr else "estimation_limited") if omega_failed else "passed_or_no_track"
            if (tau_failed and tau_true_ge_T) or (omega_failed and omega_true_ge_thr):
                overall_limit_class = "rule_limited"
            elif no_confirmed_track:
                overall_limit_class = "no_confirmed_track_rule_limited" if (tau_true_ge_T or omega_true_ge_thr) else "no_confirmed_track"
            elif tau_failed or omega_failed:
                overall_limit_class = "estimation_limited"
            else:
                overall_limit_class = "passed_at_t_check"

            # G1 WC1 miss decomposition classification
            wc1_failed_conds = []
            if no_confirmed_track:
                wc1_failed_conds.append("no_confirmed_track")
            if wc1_ttc_unavailable:
                wc1_failed_conds.append("ttc_unavailable")
            if wc1_tau_mod_hat_gt_T:
                wc1_failed_conds.append("tau_mod_hat_gt_T")
            if wc1_hmd_hat_gt_d_thr:
                wc1_failed_conds.append("hmd_hat_gt_d_thr")

            wc1_joint_key = "+".join(wc1_failed_conds) if wc1_failed_conds else "none_failed_at_t_check"
            wc1_tau_failed = (wc1_ttc_unavailable or wc1_tau_mod_hat_gt_T)
            wc1_hmd_failed = wc1_hmd_hat_gt_d_thr
            if (wc1_tau_failed and wc1_tau_mod_true_gt_T) or (wc1_hmd_failed and wc1_hmd_true_gt_d_thr):
                wc1_overall_limit_class = "rule_limited"
            elif no_confirmed_track:
                wc1_overall_limit_class = (
                    "no_confirmed_track_rule_limited"
                    if (wc1_tau_mod_true_gt_T or wc1_hmd_true_gt_d_thr)
                    else "no_confirmed_track"
                )
            elif wc1_tau_failed or wc1_hmd_failed:
                wc1_overall_limit_class = "estimation_limited"
            else:
                wc1_overall_limit_class = "passed_at_t_check"

            miss_check_at_tcpa_minus_12_5 = {
                "t_check_s": float(t_s),
                "no_confirmed_track": bool(no_confirmed_track),
                "tau_s_unavailable": bool(tau_s_unavailable),
                "tau_s_ge_T": bool(tau_s_ge_T),
                "omega_ge_thr": bool(omega_ge_thr),
                "joint_condition": joint_key,
                "tau_true_s": float(tau_true_chk),
                "omega_true_radps": float(omega_true_chk),
                "tau_est_s": float(tau_est_chk) if tau_est_chk is not None else None,
                "omega_est_radps": float(omega_est_chk) if omega_est_chk is not None else None,
                "tau_true_ge_T": bool(tau_true_ge_T),
                "omega_true_ge_thr": bool(omega_true_ge_thr),
                "tau_limit_type": tau_class,
                "omega_limit_type": omega_class,
                "overall_limit_type": overall_limit_class,
                "wc1_ttc_unavailable": bool(wc1_ttc_unavailable),
                "wc1_tau_mod_hat_gt_T": bool(wc1_tau_mod_hat_gt_T),
                "wc1_hmd_hat_gt_d_thr": bool(wc1_hmd_hat_gt_d_thr),
                "wc1_joint_condition": wc1_joint_key,
                "wc1_tau_mod_true_s": float(tau_mod_true_chk),
                "wc1_hmd_true_passive_m": float(hmd_true_passive_chk),
                "wc1_tau_mod_hat_s": float(wc1_tau_mod_hat_chk) if wc1_tau_mod_hat_chk is not None else None,
                "wc1_hmd_hat_m": float(wc1_hmd_hat_chk) if wc1_hmd_hat_chk is not None else None,
                "wc1_tau_mod_true_gt_T": bool(wc1_tau_mod_true_gt_T),
                "wc1_hmd_true_gt_d_thr": bool(wc1_hmd_true_gt_d_thr),
                "wc1_overall_limit_type": wc1_overall_limit_class,
            }

        if target_tracks:
            primary_trk = target_tracks[0]
            unique_confirmed_ids.add(primary_trk.track_id)

            if first_confirmed_t is None:
                first_confirmed_t = t_s

            # Seam tracking
            curr_cam = primary_trk.history_cams[-1] if primary_trk.history_cams else None
            if prev_cam is not None and curr_cam is not None and curr_cam != prev_cam:
                seam_crossings += 1
            prev_cam = curr_cam

            # Ground truth LOS rate in NED (WP-02 gt.py lines 118-124)
            p_i = enc.position_at(t_s)
            p_arp_ecef = st_o.p_ecef_m
            r_ecef_ned_mat = wgs84.r_ecef_ned(st_o.lat_deg, st_o.lon_deg)
            r_ecef = p_i - p_arp_ecef
            v_rel_ecef = enc.velocity_at(t_s) - (r_ecef_ned_mat @ st_o.v_ned_mps)
            r_ned_3d = r_ecef_ned_mat.T @ r_ecef
            v_rel_ned = r_ecef_ned_mat.T @ v_rel_ecef
            r_N, r_E, r_D = float(r_ned_3d[0]), float(r_ned_3d[1]), float(r_ned_3d[2])
            v_N, v_E, v_D = float(v_rel_ned[0]), float(v_rel_ned[1]), float(v_rel_ned[2])
            r_xy = math.hypot(r_N, r_E)
            range_m = float(np.linalg.norm(r_ned_3d))

            if r_xy > 1e-6 and range_m > 1e-6:
                az_rate_gt = (r_N * v_E - r_E * v_N) / (r_xy * r_xy)
                el_rate_gt = (-r_xy * v_D + r_D * (r_N * v_N + r_E * v_E) / r_xy) / (range_m * range_m)
            else:
                az_rate_gt = 0.0
                el_rate_gt = 0.0

            err_az_rate = primary_trk.az_rate - az_rate_gt
            err_el_rate = primary_trk.el_rate - el_rate_gt

            track_age_s = t_s - (primary_trk.confirmed_time or primary_trk.t_init)
            los_rate_errors.append({
                "err_az_rate": float(err_az_rate),
                "err_el_rate": float(err_el_rate),
                "range_bin": get_range_bin(range_m),
                "age_bin": get_age_bin(track_age_s),
                "class_id": params.class_id,
            })

            # TTC error
            tau_true = max(0.0, t_cpa_s - t_s)
            if primary_trk.tau_s is not None:
                delta_tau = abs(primary_trk.tau_s - tau_true)
                ttc_errors.append({
                    "delta_tau": float(delta_tau),
                    "tau_est": float(primary_trk.tau_s),
                    "tau_true": float(tau_true),
                    "range_bin": get_range_bin(range_m),
                    "class_id": params.class_id,
                })

        else:
            if prev_cam is not None and first_confirmed_t is not None and t_s < t_cpa_s:
                # Track lost prematurely before CPA
                seam_breaks += 1
                prev_cam = None

        # Risk evaluation and alerts across all confirmed tracks (WP-05 §3, R-GB-20, B7 §1)
        for trk in confirmed_tracks:
            is_threat_track = (is_threat and trk.last_associated_truth_id == params.id)
            risk = evaluate_track_risk(
                trk,
                t_s=t_s,
                omega_thr_radps=omega_thr_radps,
                k_sigma=k_sigma,
                hmd_limit_m=hmd_limit_m,
                t_warning_s=t_warning_s,
                t_corrective_s=t_corrective_s,
                class_id=None,
                beta_wc=beta_wc,
                k_theta=k_theta,
                k_omega=k_omega,
                alpha_r=alpha_r,
            )
            risk_ck = evaluate_track_risk(
                trk,
                t_s=t_s,
                omega_thr_radps=omega_thr_radps,
                k_sigma=k_sigma,
                hmd_limit_m=hmd_limit_m,
                t_warning_s=t_warning_s,
                t_corrective_s=t_corrective_s,
                class_id=params.class_id if trk.last_associated_truth_id == params.id else None,
                beta_wc=beta_wc,
                k_theta=k_theta,
                k_omega=k_omega,
                alpha_r=alpha_r,
            )

            hmd_iv = (risk.hmd_min_m, risk.hmd_max_m) if (risk.hmd_min_m is not None and risk.hmd_max_m is not None) else None
            tau_mod_iv = (risk.tau_mod_min_s, risk.tau_mod_max_s) if (risk.tau_mod_min_s is not None and risk.tau_mod_max_s is not None) else None
            closing_ok = (hmd_iv is not None and tau_mod_iv is not None)

            if is_threat_track:
                if risk.alert_a1_warning and first_alert_a1_warn_t is None:
                    first_alert_a1_warn_t = t_s
                if risk.alert_a1_corrective and first_alert_a1_corr_t is None:
                    first_alert_a1_corr_t = t_s
                if risk.alert_a2_warning and first_alert_a2_warn_t is None:
                    first_alert_a2_warn_t = t_s
                if risk.alert_a2_corrective and first_alert_a2_corr_t is None:
                    first_alert_a2_corr_t = t_s

                if risk_ck.alert_a2_warning and first_alert_a2_warn_ck_t is None:
                    first_alert_a2_warn_ck_t = t_s
                if risk_ck.alert_a2_corrective and first_alert_a2_corr_ck_t is None:
                    first_alert_a2_corr_ck_t = t_s

                if risk.wc1_warning and first_alert_wc1_warn_t is None:
                    first_alert_wc1_warn_t = t_s
                if risk.wc1_corrective and first_alert_wc1_corr_t is None:
                    first_alert_wc1_corr_t = t_s
                if risk_ck.wc1_warning and first_alert_wc1_warn_ck_t is None:
                    first_alert_wc1_warn_ck_t = t_s
                if risk_ck.wc1_corrective and first_alert_wc1_corr_ck_t is None:
                    first_alert_wc1_corr_ck_t = t_s

                if closing_ok:
                    for b_val in BETA_WC_GRID:
                        b_key = f"{b_val:.2f}"
                        bw, bc, _, _ = evaluate_wc1_for_beta(
                            hmd_iv, tau_mod_iv, beta_wc=b_val, d_thr_m=hmd_limit_m,
                            t_warning_s=t_warning_s, t_corrective_s=t_corrective_s, closing_valid=True,
                        )
                        if bw and first_alert_wc1_warn_by_beta[b_key] is None:
                            first_alert_wc1_warn_by_beta[b_key] = t_s
                        if bc and first_alert_wc1_corr_by_beta[b_key] is None:
                            first_alert_wc1_corr_by_beta[b_key] = t_s
            else:
                prev_st = prev_fa_by_track.get(trk.track_id, {})  # default-ok: previous alert state per track
                if risk.alert_a1_warning:
                    has_fa_a1_warn = True
                    if not prev_st.get("a1_warn", False):  # default-ok: rising-edge check
                        fa_a1_warn_events += 1
                if risk.alert_a1_corrective:
                    has_fa_a1_corr = True
                    if not prev_st.get("a1_corr", False):  # default-ok: rising-edge check
                        fa_a1_corr_events += 1
                if risk.alert_a2_warning:
                    has_fa_a2_warn = True
                    if not prev_st.get("a2_warn", False):  # default-ok: rising-edge check
                        fa_a2_warn_events += 1
                if risk.alert_a2_corrective:
                    has_fa_a2_corr = True
                    if not prev_st.get("a2_corr", False):  # default-ok: rising-edge check
                        fa_a2_corr_events += 1
                if risk_ck.alert_a2_warning:
                    has_fa_a2_warn_ck = True
                    if not prev_st.get("a2_warn_ck", False):  # default-ok: rising-edge check
                        fa_a2_warn_ck_events += 1
                if risk_ck.alert_a2_corrective:
                    has_fa_a2_corr_ck = True
                    if not prev_st.get("a2_corr_ck", False):  # default-ok: rising-edge check
                        fa_a2_corr_ck_events += 1
                if risk.wc1_warning:
                    has_fa_wc1_warn = True
                    if not prev_st.get("wc1_warn", False):  # default-ok: rising-edge check
                        fa_wc1_warn_events += 1
                if risk.wc1_corrective:
                    has_fa_wc1_corr = True
                    if not prev_st.get("wc1_corr", False):  # default-ok: rising-edge check
                        fa_wc1_corr_events += 1
                if risk_ck.wc1_warning:
                    has_fa_wc1_warn_ck = True
                    if not prev_st.get("wc1_warn_ck", False):  # default-ok: rising-edge check
                        fa_wc1_warn_ck_events += 1
                if risk_ck.wc1_corrective:
                    has_fa_wc1_corr_ck = True
                    if not prev_st.get("wc1_corr_ck", False):  # default-ok: rising-edge check
                        fa_wc1_corr_ck_events += 1

                next_st = {
                    "a1_warn": bool(risk.alert_a1_warning),
                    "a1_corr": bool(risk.alert_a1_corrective),
                    "a2_warn": bool(risk.alert_a2_warning),
                    "a2_corr": bool(risk.alert_a2_corrective),
                    "a2_warn_ck": bool(risk_ck.alert_a2_warning),
                    "a2_corr_ck": bool(risk_ck.alert_a2_corrective),
                    "wc1_warn": bool(risk.wc1_warning),
                    "wc1_corr": bool(risk.wc1_corrective),
                    "wc1_warn_ck": bool(risk_ck.wc1_warning),
                    "wc1_corr_ck": bool(risk_ck.wc1_corrective),
                }
                if closing_ok:
                    for b_val in BETA_WC_GRID:
                        b_key = f"{b_val:.2f}"
                        bw, bc, _, _ = evaluate_wc1_for_beta(
                            hmd_iv, tau_mod_iv, beta_wc=b_val, d_thr_m=hmd_limit_m,
                            t_warning_s=t_warning_s, t_corrective_s=t_corrective_s, closing_valid=True,
                        )
                        if bw:
                            has_fa_wc1_warn_by_beta[b_key] = True
                            if not prev_st.get(f"wc1_warn_{b_key}", False):  # default-ok: rising-edge check
                                fa_wc1_warn_events_by_beta[b_key] += 1
                        if bc:
                            has_fa_wc1_corr_by_beta[b_key] = True
                            if not prev_st.get(f"wc1_corr_{b_key}", False):  # default-ok: rising-edge check
                                fa_wc1_corr_events_by_beta[b_key] += 1
                        next_st[f"wc1_warn_{b_key}"] = bool(bw)
                        next_st[f"wc1_corr_{b_key}"] = bool(bc)
                prev_fa_by_track[trk.track_id] = next_st

    # Track management metrics
    confirmed_count = len(unique_confirmed_ids)
    fragmentations = max(0, confirmed_count - 1)
    acq_delay_s = max(0.0, first_confirmed_t - first_detectable_t) if (first_confirmed_t is not None and first_detectable_t is not None) else None

    # Count false tracks (D32: primary count is ever-confirmed tracks not associated with intruder;
    # secondary count is tracks still CONFIRMED at end of run).
    ever_confirmed = [
        t for t in tracker.tracks
        if t.confirmed_time is not None or t.status == TrackStatus.CONFIRMED
    ]
    end_confirmed = [t for t in tracker.tracks if t.status == TrackStatus.CONFIRMED]
    false_tracks = sum(1 for t in ever_confirmed if t.last_associated_truth_id != params.id)
    false_tracks_end_of_run = sum(1 for t in end_confirmed if t.last_associated_truth_id != params.id)

    # Lead times for threat encounters
    lead_a1_warn_s = (lowc_t - first_alert_a1_warn_t) if (is_threat and lowc_t is not None and first_alert_a1_warn_t is not None) else None
    lead_a1_corr_s = (lowc_t - first_alert_a1_corr_t) if (is_threat and lowc_t is not None and first_alert_a1_corr_t is not None) else None
    lead_a2_warn_s = (lowc_t - first_alert_a2_warn_t) if (is_threat and lowc_t is not None and first_alert_a2_warn_t is not None) else None
    lead_a2_corr_s = (lowc_t - first_alert_a2_corr_t) if (is_threat and lowc_t is not None and first_alert_a2_corr_t is not None) else None
    lead_a2_warn_ck_s = (lowc_t - first_alert_a2_warn_ck_t) if (is_threat and lowc_t is not None and first_alert_a2_warn_ck_t is not None) else None
    lead_a2_corr_ck_s = (lowc_t - first_alert_a2_corr_ck_t) if (is_threat and lowc_t is not None and first_alert_a2_corr_ck_t is not None) else None

    lead_wc1_warn_s = (lowc_t - first_alert_wc1_warn_t) if (is_threat and lowc_t is not None and first_alert_wc1_warn_t is not None) else None
    lead_wc1_corr_s = (lowc_t - first_alert_wc1_corr_t) if (is_threat and lowc_t is not None and first_alert_wc1_corr_t is not None) else None
    lead_wc1_warn_ck_s = (lowc_t - first_alert_wc1_warn_ck_t) if (is_threat and lowc_t is not None and first_alert_wc1_warn_ck_t is not None) else None
    lead_wc1_corr_ck_s = (lowc_t - first_alert_wc1_corr_ck_t) if (is_threat and lowc_t is not None and first_alert_wc1_corr_ck_t is not None) else None

    out_dict: dict[str, Any] = {
        "id": params.id,
        "class_id": params.class_id,
        "is_threat": is_threat,
        "lowc_t": lowc_t,
        "nmac_t": nmac_t,
        "t_cpa_s": t_cpa_s,
        "hmd_m": hmd_m,
        "first_detectable_t": first_detectable_t,
        "first_confirmed_t": first_confirmed_t,
        "acq_delay_s": acq_delay_s,
        "fragmentations": fragmentations,
        "seam_crossings": seam_crossings,
        "seam_breaks": seam_breaks,
        "false_tracks": false_tracks,
        "false_tracks_ever_confirmed": false_tracks,
        "false_tracks_end_of_run": false_tracks_end_of_run,
        "ground_clutter_rejected_count": int(tracker.ground_clutter_rejected_count),
        "detectable_frames": detectable_frames,
        "confirmed_on_detectable_frames": confirmed_on_detectable_frames,
        "true_track_coverage": (confirmed_on_detectable_frames / detectable_frames) if detectable_frames > 0 else float("nan"),
        "miss_check_at_tcpa_minus_12_5": miss_check_at_tcpa_minus_12_5,
        "los_rate_errors": los_rate_errors,
        "ttc_errors": ttc_errors,
        "has_alert_a1_warn": (first_alert_a1_warn_t is not None) if is_threat else has_fa_a1_warn,
        "has_alert_a1_corr": (first_alert_a1_corr_t is not None) if is_threat else has_fa_a1_corr,
        "has_alert_a2_warn": (first_alert_a2_warn_t is not None) if is_threat else has_fa_a2_warn,
        "has_alert_a2_corr": (first_alert_a2_corr_t is not None) if is_threat else has_fa_a2_corr,
        "has_alert_a2_warn_ck": (first_alert_a2_warn_ck_t is not None) if is_threat else has_fa_a2_warn_ck,
        "has_alert_a2_corr_ck": (first_alert_a2_corr_ck_t is not None) if is_threat else has_fa_a2_corr_ck,
        "has_alert_wc1_warn": (first_alert_wc1_warn_t is not None) if is_threat else has_fa_wc1_warn,
        "has_alert_wc1_corr": (first_alert_wc1_corr_t is not None) if is_threat else has_fa_wc1_corr,
        "has_alert_wc1_warn_ck": (first_alert_wc1_warn_ck_t is not None) if is_threat else has_fa_wc1_warn_ck,
        "has_alert_wc1_corr_ck": (first_alert_wc1_corr_ck_t is not None) if is_threat else has_fa_wc1_corr_ck,
        "has_fa_a1_warn": has_fa_a1_warn,
        "has_fa_a1_corr": has_fa_a1_corr,
        "has_fa_a2_warn": has_fa_a2_warn,
        "has_fa_a2_corr": has_fa_a2_corr,
        "has_fa_a2_warn_ck": has_fa_a2_warn_ck,
        "has_fa_a2_corr_ck": has_fa_a2_corr_ck,
        "has_fa_wc1_warn": has_fa_wc1_warn,
        "has_fa_wc1_corr": has_fa_wc1_corr,
        "has_fa_wc1_warn_ck": has_fa_wc1_warn_ck,
        "has_fa_wc1_corr_ck": has_fa_wc1_corr_ck,
        "fa_a1_warn_events": fa_a1_warn_events,
        "fa_a1_corr_events": fa_a1_corr_events,
        "fa_a2_warn_events": fa_a2_warn_events,
        "fa_a2_corr_events": fa_a2_corr_events,
        "fa_a2_warn_ck_events": fa_a2_warn_ck_events,
        "fa_a2_corr_ck_events": fa_a2_corr_ck_events,
        "fa_wc1_warn_events": fa_wc1_warn_events,
        "fa_wc1_corr_events": fa_wc1_corr_events,
        "fa_wc1_warn_ck_events": fa_wc1_warn_ck_events,
        "fa_wc1_corr_ck_events": fa_wc1_corr_ck_events,
        "lead_a1_warn_s": lead_a1_warn_s,
        "lead_a1_corr_s": lead_a1_corr_s,
        "lead_a2_warn_s": lead_a2_warn_s,
        "lead_a2_corr_s": lead_a2_corr_s,
        "lead_a2_warn_ck_s": lead_a2_warn_ck_s,
        "lead_a2_corr_ck_s": lead_a2_corr_ck_s,
        "lead_wc1_warn_s": lead_wc1_warn_s,
        "lead_wc1_corr_s": lead_wc1_corr_s,
        "lead_wc1_warn_ck_s": lead_wc1_warn_ck_s,
        "lead_wc1_corr_ck_s": lead_wc1_corr_ck_s,
    }
    for b_val in BETA_WC_GRID:
        b_key = f"{b_val:.2f}"
        fw = first_alert_wc1_warn_by_beta[b_key]
        fc = first_alert_wc1_corr_by_beta[b_key]
        out_dict[f"lead_wc1_warn_b{b_key}_s"] = (lowc_t - fw) if (is_threat and lowc_t is not None and fw is not None) else None
        out_dict[f"lead_wc1_corr_b{b_key}_s"] = (lowc_t - fc) if (is_threat and lowc_t is not None and fc is not None) else None
        out_dict[f"has_alert_wc1_warn_b{b_key}"] = (fw is not None) if is_threat else has_fa_wc1_warn_by_beta[b_key]
        out_dict[f"has_alert_wc1_corr_b{b_key}"] = (fc is not None) if is_threat else has_fa_wc1_corr_by_beta[b_key]
        out_dict[f"fa_wc1_warn_b{b_key}_events"] = fa_wc1_warn_events_by_beta[b_key]
        out_dict[f"fa_wc1_corr_b{b_key}_events"] = fa_wc1_corr_events_by_beta[b_key]

    return out_dict


def _worker_wrapper(args: tuple[dict[str, Any], dict[str, Any], int, dict[str, Any] | None]) -> dict[str, Any]:
    params_dict, config, seed, truth_cache = args
    return simulate_single_encounter(params_dict, config, seed, truth_cache=truth_cache)


def aggregate_dataset_metrics(
    encounter_results: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate per-encounter results into complete R7 and WP-05 metrics."""
    n_encounters = len(encounter_results)
    total_flight_hours = (n_encounters * 60.0) / 3600.0

    # 1. Tracking acquisition and stability
    delays = [r["acq_delay_s"] for r in encounter_results if r["acq_delay_s"] is not None]
    acq_stats = {
        "count": len(delays),
        "mean_s": float(np.mean(delays)) if delays else float("nan"),
        "median_s": float(np.median(delays)) if delays else float("nan"),
        "p95_s": float(np.percentile(delays, 95)) if delays else float("nan"),
    }

    total_frags = sum(r["fragmentations"] for r in encounter_results)
    total_seam_crossings = sum(r["seam_crossings"] for r in encounter_results)
    total_seam_breaks = sum(r["seam_breaks"] for r in encounter_results)
    total_false_tracks = sum(r["false_tracks"] for r in encounter_results)
    false_tracks_per_hour = total_false_tracks / total_flight_hours if total_flight_hours > 0 else 0.0
    total_false_tracks_end_of_run = sum(
        r.get("false_tracks_end_of_run", r["false_tracks"]) for r in encounter_results  # default-ok: backward compatibility
    )
    false_tracks_end_of_run_per_hour = (
        total_false_tracks_end_of_run / total_flight_hours if total_flight_hours > 0 else 0.0
    )
    total_ground_rejected = sum(
        int(r.get("ground_clutter_rejected_count", 0)) for r in encounter_results  # default-ok: backward compatibility
    )
    ground_rejected_per_hour = (
        total_ground_rejected / total_flight_hours if total_flight_hours > 0 else 0.0
    )
    total_detectable_frames = sum(r.get("detectable_frames", 0) for r in encounter_results)  # default-ok: backward compatibility
    total_confirmed_detectable_frames = sum(r.get("confirmed_on_detectable_frames", 0) for r in encounter_results)  # default-ok: backward compatibility
    true_track_coverage = (
        total_confirmed_detectable_frames / total_detectable_frames
        if total_detectable_frames > 0 else float("nan")
    )

    # 2. LOS Rate Errors (R7-P2)
    all_az_errs = []
    all_el_errs = []
    errs_by_range: dict[str, list[tuple[float, float]]] = {name: [] for name, _, _ in RANGE_BINS_M}
    errs_by_class: dict[str, list[tuple[float, float]]] = {}
    errs_by_age: dict[str, list[tuple[float, float]]] = {name: [] for name, _, _ in TRACK_AGE_BINS_S}

    for r in encounter_results:
        for err_dict in r["los_rate_errors"]:
            d_az = err_dict["err_az_rate"]
            d_el = err_dict["err_el_rate"]
            all_az_errs.append(d_az)
            all_el_errs.append(d_el)

            pair = (d_az, d_el)
            errs_by_range[err_dict["range_bin"]].append(pair)
            errs_by_age[err_dict["age_bin"]].append(pair)
            cls_id = err_dict["class_id"]
            if cls_id not in errs_by_class:
                errs_by_class[cls_id] = []
            errs_by_class[cls_id].append(pair)

    sigma_az_rate = float(np.std(all_az_errs)) if all_az_errs else 0.0
    sigma_el_rate = float(np.std(all_el_errs)) if all_el_errs else 0.0
    sigma_omega_total = math.hypot(sigma_az_rate, sigma_el_rate)

    def _calc_bin_sigma_omega(pairs: list[tuple[float, float]]) -> float:
        if len(pairs) < 2:
            return 0.0
        s_az = float(np.std([p[0] for p in pairs]))
        s_el = float(np.std([p[1] for p in pairs]))
        return math.hypot(s_az, s_el)

    los_rate_summary = {
        "sigma_az_rate_radps": sigma_az_rate,
        "sigma_el_rate_radps": sigma_el_rate,
        "sigma_omega_total_radps": sigma_omega_total,
        "nasa_bound_radps": 0.0021,
        "meets_nasa_bound": bool(sigma_omega_total < 0.0021),
        "by_range_bin": {k: _calc_bin_sigma_omega(v) for k, v in errs_by_range.items()},
        "by_class": {k: _calc_bin_sigma_omega(v) for k, v in errs_by_class.items()},
        "by_age_bin": {k: _calc_bin_sigma_omega(v) for k, v in errs_by_age.items()},
    }

    # 3. TTC Errors
    all_ttc_deltas = []
    ttc_by_range: dict[str, list[float]] = {name: [] for name, _, _ in RANGE_BINS_M}
    ttc_by_class: dict[str, list[float]] = {}

    for r in encounter_results:
        for item in r["ttc_errors"]:
            d_tau = item["delta_tau"]
            all_ttc_deltas.append(d_tau)
            ttc_by_range[item["range_bin"]].append(d_tau)
            cls_id = item["class_id"]
            if cls_id not in ttc_by_class:
                ttc_by_class[cls_id] = []
            ttc_by_class[cls_id].append(d_tau)

    ttc_summary = {
        "mean_error_s": float(np.mean(all_ttc_deltas)) if all_ttc_deltas else float("nan"),
        "median_error_s": float(np.median(all_ttc_deltas)) if all_ttc_deltas else float("nan"),
        "std_error_s": float(np.std(all_ttc_deltas)) if all_ttc_deltas else float("nan"),
        "by_range_bin": {k: float(np.mean(v)) if v else float("nan") for k, v in ttc_by_range.items()},
        "by_class": {k: float(np.mean(v)) if v else float("nan") for k, v in ttc_by_class.items()},
    }

    # 4. Alerting Performance (R7-P1)
    threats = [r for r in encounter_results if r["is_threat"]]
    non_threats = [r for r in encounter_results if not r["is_threat"]]

    n_threats = len(threats)
    n_non_threats = len(non_threats)
    non_threat_hours = (n_non_threats * 60.0) / 3600.0

    def compute_alert_stats(rule_key: str) -> dict[str, Any]:
        leads = [r.get(f"lead_{rule_key}_s") for r in threats if r.get(f"lead_{rule_key}_s") is not None]
        k_12_5 = sum(1 for L in leads if L >= 12.5)
        k_25_0 = sum(1 for L in leads if L >= 25.0)
        k_55_0 = sum(1 for L in leads if L >= 55.0)
        k_missed = n_threats - sum(1 for L in leads if L > 0.0)

        frac_12_5 = (k_12_5 / n_threats) if n_threats > 0 else 0.0
        frac_25_0 = (k_25_0 / n_threats) if n_threats > 0 else 0.0
        frac_55_0 = (k_55_0 / n_threats) if n_threats > 0 else 0.0
        frac_missed = (k_missed / n_threats) if n_threats > 0 else 0.0

        ci_12_5 = list(clopper_pearson_interval(k_12_5, n_threats, 0.95)) if n_threats > 0 else [0.0, 0.0]
        ci_25_0 = list(clopper_pearson_interval(k_25_0, n_threats, 0.95)) if n_threats > 0 else [0.0, 0.0]
        ci_55_0 = list(clopper_pearson_interval(k_55_0, n_threats, 0.95)) if n_threats > 0 else [0.0, 0.0]
        ci_missed = list(clopper_pearson_interval(k_missed, n_threats, 0.95)) if n_threats > 0 else [0.0, 0.0]

        # False alerts on non-threats
        k_false_alert = sum(1 for r in non_threats if r.get(f"has_alert_{rule_key}", False))  # default-ok: backward compatibility
        k_false_alert_events = sum(
            int(r.get(f"fa_{rule_key}_events", 1 if r.get(f"has_alert_{rule_key}", False) else 0))  # default-ok: backward compatibility
            for r in non_threats
        )
        frac_false_alert = (k_false_alert / n_non_threats) if n_non_threats > 0 else 0.0
        ci_false_alert = list(clopper_pearson_interval(k_false_alert, n_non_threats, 0.95)) if n_non_threats > 0 else [0.0, 0.0]
        fa_per_hour = (k_false_alert / non_threat_hours) if non_threat_hours > 0 else 0.0
        fa_events_per_hour = (k_false_alert_events / non_threat_hours) if non_threat_hours > 0 else 0.0

        return {
            "threats_count": n_threats,
            "non_threats_count": n_non_threats,
            "lead_time_mean_s": float(np.mean(leads)) if leads else float("nan"),
            "lead_time_median_s": float(np.median(leads)) if leads else float("nan"),
            "frac_ge_12_5s": frac_12_5,
            "ci_ge_12_5s": ci_12_5,
            "frac_ge_25_0s": frac_25_0,
            "ci_ge_25_0s": ci_25_0,
            "frac_ge_55_0s": frac_55_0,
            "ci_ge_55_0s": ci_55_0,
            "frac_missed": frac_missed,
            "ci_missed": ci_missed,
            "false_alert_fraction": frac_false_alert,
            "false_alert_ci": ci_false_alert,
            "false_alerts_per_hour": fa_per_hour,
            "runs_with_false_alert_count": k_false_alert,
            "runs_with_false_alert_per_h": fa_per_hour,
            "false_alert_events_count": k_false_alert_events,
            "false_alert_events_per_h": fa_events_per_hour,
        }

    wc1_by_beta: dict[str, Any] = {}
    for b_val in BETA_WC_GRID:
        b_key = f"{b_val:.2f}"
        wc1_by_beta[b_key] = {
            "beta_wc": float(b_val),
            "warning": compute_alert_stats(f"wc1_warn_b{b_key}"),
            "corrective": compute_alert_stats(f"wc1_corr_b{b_key}"),
        }

    alerting_summary = {
        "A1_warning": compute_alert_stats("a1_warn"),
        "A1_corrective": compute_alert_stats("a1_corr"),
        "A2_warning": compute_alert_stats("a2_warn"),
        "A2_corrective": compute_alert_stats("a2_corr"),
        "A2_warning_class_known": compute_alert_stats("a2_warn_ck"),
        "A2_corrective_class_known": compute_alert_stats("a2_corr_ck"),
        "WC1_warning": compute_alert_stats("wc1_warn"),
        "WC1_corrective": compute_alert_stats("wc1_corr"),
        "WC1_warning_class_known": compute_alert_stats("wc1_warn_ck"),
        "WC1_corrective_class_known": compute_alert_stats("wc1_corr_ck"),
        "WC1_by_beta": wc1_by_beta,
    }

    return {
        "encounters_count": n_encounters,
        "flight_hours_total": total_flight_hours,
        "threats_count": n_threats,
        "non_threats_count": n_non_threats,
        "tracking": {
            "acquisition_delay": acq_stats,
            "fragmentations_total": total_frags,
            "seam_crossings_total": total_seam_crossings,
            "seam_breaks_total": total_seam_breaks,
            "false_tracks_total": total_false_tracks,
            "false_tracks_per_hour": false_tracks_per_hour,
            "false_tracks_ever_confirmed_total": total_false_tracks,
            "false_tracks_end_of_run_total": total_false_tracks_end_of_run,
            "false_tracks_end_of_run_per_hour": false_tracks_end_of_run_per_hour,
            "ground_clutter_rejected_total": total_ground_rejected,
            "ground_clutter_rejected_per_hour": ground_rejected_per_hour,
            "detectable_frames_total": total_detectable_frames,
            "confirmed_on_detectable_frames_total": total_confirmed_detectable_frames,
            "true_track_coverage": float(true_track_coverage),
        },
        "los_rate": los_rate_summary,
        "ttc": ttc_summary,
        "alerting": alerting_summary,
    }


def run_evaluation(
    encounters: Sequence[EncounterParams],
    config: dict[str, Any] | None = None,
    seed: int = 101,
    max_workers: int = 8,
) -> dict[str, Any]:
    """Execute evaluation over all encounters with multiprocessing."""
    from multiprocessing import Pool

    cfg = dict(config) if config is not None else {}
    if not cfg:
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    # Warmup ownship states cache before multiprocessing
    get_ownship_states()

    # Load or compute truth cache
    truth_cache_map, truth_cache_info = load_or_compute_truth_cache(
        encounters, max_workers=max_workers
    )

    tasks = [
        (asdict(e), cfg, seed + i, truth_cache_map.get(e.id))
        for i, e in enumerate(encounters)
    ]

    t0 = time.perf_counter()
    if max_workers > 1:
        with Pool(processes=max_workers) as pool:
            results = pool.map(_worker_wrapper, tasks)
    else:
        results = [_worker_wrapper(task) for task in tasks]
    t1 = time.perf_counter()

    metrics = aggregate_dataset_metrics(results, cfg)
    metrics["runtime_s"] = t1 - t0
    metrics["frames_per_second"] = (len(encounters) * 601) / (t1 - t0)
    metrics["truth_cache"] = truth_cache_info
    return metrics
