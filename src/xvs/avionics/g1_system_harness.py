"""End-to-End Cross-WP Gate G1 System Sensitivity & Integrity State-Transition Harness (Batch B14).

Implements `docs/xvs/batches/BATCH_B14_G1_SYSTEM_HARNESS.md`:
1. Evaluates the 50-encounter closed-loop chain:
     `WP-04 Detector Closure (D0 vs D1)` -> `WP-05 G1 MultiCameraTracker + Moment TTC + WC1`
     -> `WP-07 G1 Horizon/Attitude Integrity Gate` -> `WP-08 G1 Avionics Bridge (MAVLink #246 + GDL-90)`
     -> `ArduPilot SITL Horizontal Avoidance Autopilot`
   side-by-side under both the classical `D0` closure (`configs/xvs/tracking/g0_measured_d0.json`)
   and the measured `D1` neural closure (`configs/xvs/tracking/g1_measured_d1.json` or `--d1-closure-config`),
   quantifying end-to-end gains in initial track confirmation range (m), warning lead time (s),
   NMAC resolution rate (`>= 152.4 m`), and LoWC recovery rate (`>= 609.6 m`).
2. Loads WP-07 G1's real calibrated `MonitorThresholdsG1` and altitude-binned Mondrian `ConformalModelG1`
   from `output/xvs/wp07/g1/results.json` and executes real WP-01 G4 4-camera (`FRBL`) CesiumJS frames
   through `CameraSimulator`, `detect_camera_edges_multiscale`, `compute_horizon_strip_ood_features`,
   `build_pipeline_ray_batch`, `fit_pipeline_attitude_g1`, `evaluate_monitor_alarm_g1`,
   `inject_single_cam_misalignment`, `isolate_and_exclude_faulty_camera`, and
   `compute_pl_deg_altitude_mondrian`, coupled directly to `extract_viewport` + `draw_g1_symbology` (HUD),
   `PassiveTrackAvionicsBridge.step_from_tracker`, and `SitlAvoidanceAutopilot.step_from_wire_packets`
   across `VALID` -> `F_isolated` -> `INVALID_ALERT` -> `VALID` recovery (`0` end-to-end HMI).

Strictly preserves the V5_B10 / V5_G1 / V4_B14 truth-isolation invariant: never imports `xvs.traffic`
and never accesses any ground-truth intruder attribute inside perception, HUD, bridge, or autopilot nodes.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from xvs.avionics.bridge import OwnshipNavState, PassiveTrackAvionicsBridge
from xvs.avionics.mavlink_adsb import ADSB_FLAGS_VALID_COORDS, unpack_adsb_vehicle
from xvs.avionics.sitl_closed_loop import (
    SitlAvoidanceAutopilot,
    SitlAvoidanceConfig,
    SitlThreatScenario,
    _generate_ring_detections_for_step,
    _pregenerate_crn_draws,
    _wrap_deg_180,
    build_g3a_collision_threat_scenarios,
    load_g1_tracking_config,
)
from xvs.display.symbology import draw_g1_symbology
from xvs.display.viewport import (
    compute_viewport_pose_from_estimated_attitude,
    extract_viewport,
    get_viewport_camera,
)
from xvs.geo import rotations, wgs84
from xvs.horizon_att.limb_model import LimbRayBatch
from xvs.horizon_att.pipeline import build_pipeline_ray_batch, load_pipeline_rig
from xvs.integrity.conformal import ConformalModel
from xvs.integrity.g1_extensions import (
    MonitorThresholdsG1,
    compute_horizon_strip_ood_features,
    compute_pl_deg_altitude_mondrian,
    detect_camera_edges_multiscale,
    evaluate_monitor_alarm_g1,
    fit_pipeline_attitude_g1,
    inject_single_cam_misalignment,
    isolate_and_exclude_faulty_camera,
)
from xvs.integrity.monitor import camera_consistency
from xvs.paths import get_data_root
from xvs.sensor.chain import CameraSimulator, SensorConfig
from xvs.tracking.measurements import get_model_vertices
from xvs.tracking.risk import LOWC_HMD_LIMIT_M, RiskEstimate, evaluate_track_risk
from xvs.tracking.tracker import MultiCameraTracker, Track

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_D0_CLOSURE_PATH = REPO_ROOT / "configs" / "xvs" / "tracking" / "g0_measured_d0.json"
DEFAULT_B14_SPEC_PATH = REPO_ROOT / "configs" / "xvs" / "avionics" / "b14_system_spec.json"
DEFAULT_SENSOR_CONFIG_PATH = REPO_ROOT / "configs" / "xvs" / "sensors" / "rig_mono_v0.json"
DEFAULT_DISPLAY_G1_CONFIG_PATH = REPO_ROOT / "configs" / "xvs" / "display" / "g1.json"

_FALLBACK_D1_CLOSURE_PATHS = (
    REPO_ROOT / "configs" / "xvs" / "tracking" / "g1_measured_d1.json",
)

_FALLBACK_MAIN_OUTPUT_ROOT = REPO_ROOT / "output"


def file_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def resolve_d0_closure_config_path(override_path: Path | str | None = None) -> Path:
    """Resolve classical `D0` detector closure JSON (`configs/xvs/tracking/g0_measured_d0.json`)."""
    if override_path is not None:
        p = Path(override_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"D0 closure config not found at explicit path: {p}")
        return p
    if DEFAULT_D0_CLOSURE_PATH.is_file():
        return DEFAULT_D0_CLOSURE_PATH
    raise FileNotFoundError(f"Default D0 closure config not found: {DEFAULT_D0_CLOSURE_PATH}")


def resolve_d1_closure_config_path(override_path: Path | str | None = None) -> Path:
    """Resolve measured `D1` neural detector closure JSON (`g1_measured_d1.json`).

    Priority:
      1. Explicit `--d1-closure-config` override path;
      2. Repo-local `configs/xvs/tracking/g1_measured_d1.json`.
    """
    if override_path is not None:
        p = Path(override_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"D1 closure config not found at explicit path: {p}")
        return p
    for cand in _FALLBACK_D1_CLOSURE_PATHS:
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(
        f"Measured D1 closure config not found in any candidate location: {_FALLBACK_D1_CLOSURE_PATHS}"
    )


def resolve_wp07_g1_results_path(
    override_path: Path | str | None = None,
    data_root: Path | None = None,
) -> Path:
    """Resolve WP-07 G1 real measured results JSON (`output/xvs/wp07/g1/results.json`)."""
    if override_path is not None:
        p = Path(override_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"WP-07 G1 results JSON not found at explicit path: {p}")
        return p
    cand_local = get_data_root(data_root) / "xvs" / "wp07" / "g1" / "results.json"
    if cand_local.is_file():
        return cand_local.resolve()
    cand_repo = REPO_ROOT / "output" / "xvs" / "wp07" / "g1" / "results.json"
    if cand_repo.is_file():
        return cand_repo.resolve()
    cand_main = _FALLBACK_MAIN_OUTPUT_ROOT / "xvs" / "wp07" / "g1" / "results.json"
    if cand_main.is_file():
        return cand_main.resolve()
    raise FileNotFoundError(
        f"WP-07 G1 results.json not found under {cand_local}, {cand_repo}, or {cand_main}"
    )


def resolve_g4_sequence_dir(seq_name: str, data_root: Path | None = None) -> Path:
    """Resolve real WP-01 G4 6-camera CesiumJS sequence directory (`T1_cruise`, `T2_ksfo28l`, `T3_turn_climb`)."""
    for base in (
        get_data_root(data_root) / "xvs" / "wp01" / "g4",
        REPO_ROOT / "output" / "xvs" / "wp01" / "g4",
        _FALLBACK_MAIN_OUTPUT_ROOT / "xvs" / "wp01" / "g4",
    ):
        cand = base / seq_name
        if (cand / "frames.jsonl").is_file():
            return cand.resolve()
    raise FileNotFoundError(f"WP-01 G4 sequence directory not found for {seq_name!r}")


def build_detector_closure_profile(
    base_g1_trk_cfg: Mapping[str, Any],
    closure_raw: Mapping[str, Any],
    closure_id: str,
    source_path: Path,
    d0_persist_fallback: float = 0.1544529690468746,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Map a detector closure JSON (`g0_measured_d0.json` or `g1_measured_d1.json`) onto a WP-05 G1
    `MultiCameraTracker` configuration dict and a structured provenance summary.

    Supports both `lambda_ub` and `fppi_lambda_ub95` keys so Colab A100 updates drop in seamlessly.
    """
    s50_px = float(closure_raw["s50_px"])
    w_ln = float(closure_raw.get("w_ln", closure_raw.get("w_px", base_g1_trk_cfg["w_ln"])))
    sigma_px = float(closure_raw["sigma_px"])
    sigma_s_px = float(closure_raw.get("sigma_s_px", sigma_px))
    fppi_lambda = float(closure_raw.get("fppi_lambda", 0.0))  # default-ok: zero clutter fallback if omitted in closure JSON

    if "lambda_ub" in closure_raw and closure_raw["lambda_ub"] is not None:
        lambda_ub = float(closure_raw["lambda_ub"])
    elif "fppi_lambda_ub95" in closure_raw and closure_raw["fppi_lambda_ub95"] is not None:
        lambda_ub = float(closure_raw["fppi_lambda_ub95"])
    else:
        lambda_ub = fppi_lambda

    if "persistence_3_fraction" in closure_raw and closure_raw["persistence_3_fraction"] is not None:
        persist_3_frac = float(closure_raw["persistence_3_fraction"])
    elif closure_id.upper() == "D0":
        persist_3_frac = float(closure_raw.get("d0_persistence_3_reference", d0_persist_fallback))
    else:
        persist_3_frac = 0.0

    effective_persist_lambda = float(lambda_ub * persist_3_frac)

    merged_trk_cfg = {
        **dict(base_g1_trk_cfg),
        "s50_px": s50_px,
        "w_ln": w_ln,
        "w_px": w_ln,
        "sigma_px": sigma_px,
        "sigma_s_px": sigma_s_px,
        "fppi_lambda": fppi_lambda,
    }

    profile_summary = {
        "closure_id": closure_id,
        "source_path": str(source_path.as_posix()),
        "source_sha256": file_sha256(source_path),
        "schema": str(closure_raw.get("schema", "")),
        "pd_mode": str(closure_raw.get("pd_mode", "log_logistic")),
        "s50_px": s50_px,
        "w_ln": w_ln,
        "sigma_px": sigma_px,
        "sigma_s_px": sigma_s_px,
        "fppi_lambda": fppi_lambda,
        "lambda_ub": lambda_ub,
        "persistence_3_fraction": persist_3_frac,
        "effective_persistent_lambda": effective_persist_lambda,
    }
    if "s50_ci95_px" in closure_raw and isinstance(closure_raw["s50_ci95_px"], Sequence):
        ci = list(closure_raw["s50_ci95_px"])
        if len(ci) == 2:
            profile_summary["s50_ci95_low_px"] = float(ci[0])
            profile_summary["s50_ci95_high_px"] = float(ci[1])

    return merged_trk_cfg, profile_summary


def simulate_system_encounter_closed_loop(
    scenario: SitlThreatScenario,
    *,
    system_mode: str,
    tracking_config: Mapping[str, Any],
    bridge: PassiveTrackAvionicsBridge,
    sitl_config: SitlAvoidanceConfig,
    wp07_att_sigma_deg: float | None = None,
    integrity_state: str = "VALID",
    crn_draws: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run one 60-second closed-loop encounter at 10 Hz across the full sensor-to-autopilot chain:
    `WP-04 Ring Detections` -> `WP-05 G1 MultiCameraTracker` -> `WP-07 G1 Attitude/Integrity Gate`
    -> `WP-08 G1 Avionics Bridge (MAVLink #246 & GDL-90)` -> `ArduPilot SITL Autopilot`.

    Supported `system_mode` values:
      - `open_loop`
      - `d0_g0_a1`
      - `d0_g1_wc1_b050`
      - `d1_g1_wc1_b050`
      - `d0_g1_wc1_b025`
      - `d1_g1_wc1_b025`
    """
    dt = sitl_config.dt_s
    n_steps = int(round((sitl_config.t_end_s - sitl_config.t_start_s) / dt)) + 1
    if crn_draws is None:
        crn_draws = _pregenerate_crn_draws(scenario.seed, n_steps)

    base_trk_cfg = dict(tracking_config)
    if wp07_att_sigma_deg is not None and wp07_att_sigma_deg > 0.0:
        base_trk_cfg["ins_sigma_deg"] = float(wp07_att_sigma_deg)

    enable_avoidance = system_mode != "open_loop"
    use_g1_wc1 = system_mode in (
        "d0_g1_wc1_b050",
        "d1_g1_wc1_b050",
        "d0_g1_wc1_b025",
        "d1_g1_wc1_b025",
    )
    beta_wc = 0.25 if system_mode.endswith("_b025") else 0.50

    if use_g1_wc1 or system_mode == "open_loop":
        mode_trk_cfg = dict(base_trk_cfg)
    else:
        # Gate G0 A1 baseline: disable G1 second-moment scale and sliding-window log-size TTC
        mode_trk_cfg = {
            **base_trk_cfg,
            "use_second_moment_scale": False,
            "use_sliding_window_ttc": False,
            "enable_ground_clutter_rejection": False,
        }

    tracker = MultiCameraTracker(config=mode_trk_cfg)
    verts_model = get_model_vertices(scenario.model_id)

    att_sigma_rad = math.radians(float(mode_trk_cfg["ins_sigma_deg"]))
    ins_tau_s = float(mode_trk_cfg["ins_tau_s"])
    ins_beta = math.exp(-dt / ins_tau_s) if ins_tau_s > 0.0 else 0.0
    att_err_rpy = att_sigma_rad * np.asarray(crn_draws["ins_init_rpy"], dtype=np.float64).copy()

    omega_thr_radps = float(mode_trk_cfg.get("omega_thr_radps", 0.005))  # default-ok: WP-05 G1 tracker default angular-rate threshold
    k_sigma = float(mode_trk_cfg.get("k_sigma", 3.0))  # default-ok: WP-05 G1 tracker default uncertainty multiplier
    t_warning_s = float(mode_trk_cfg.get("t_warning_s", 25.0))  # default-ok: DO-365B warning horizon default
    t_corrective_s = float(mode_trk_cfg.get("t_corrective_s", 55.0))  # default-ok: DO-365B corrective horizon default
    k_theta = float(mode_trk_cfg.get("k_theta", 1.0))  # default-ok: WC1 angular-size risk weight default
    k_omega = float(mode_trk_cfg.get("k_omega", 1.0))  # default-ok: WC1 LOS-rate risk weight default
    alpha_r = float(mode_trk_cfg.get("alpha_r", 0.0))  # default-ok: WC1 range-rate floor weight default
    eta_floor = float(mode_trk_cfg.get("eta_floor", 0.0001))  # default-ok: WC1 numerical denominator floor

    lat0_deg = 37.30
    lon0_deg = -121.00
    h0_m = 1500.0
    p0_ecef = wgs84.geodetic_to_ecef(lat0_deg, lon0_deg, h0_m)
    n0_mat = wgs84.r_ecef_ned(lat0_deg, lon0_deg)
    e_hat_0 = n0_mat[:, 1]
    n_hat_0 = n0_mat[:, 0]
    own_speed_mps = 100.0

    p_own_ecef = p0_ecef.copy()
    heading_deg = 90.0
    roll_deg = 0.0

    autopilot = SitlAvoidanceAutopilot(nominal_heading_deg=90.0, config=sitl_config)

    # Initial apparent angular size at t = t_start_s (for onset regime diagnostics)
    p_tgt_t0 = scenario.target_position_ecef_at(sitl_config.t_start_s)
    rng_t0_m = max(1.0, float(np.linalg.norm(p_tgt_t0 - p0_ecef)))
    s_px_at_t0 = float(tracker.ring.fx * (scenario.characteristic_length_m / rng_t0_m))

    min_hmd_m = float("inf")
    min_3d_dist_m = float("inf")
    t_at_min_hmd_s = 0.0
    max_cross_track_m = 0.0
    final_cross_track_m = 0.0
    max_heading_dev_deg = 0.0

    first_confirm_t_s: float | None = None
    initial_confirm_range_3d_m = 0.0
    initial_confirm_range_horiz_m = 0.0
    first_alert_t_s: float | None = None
    first_maneuver_t_s: float | None = None
    mavlink_packets_rx = 0
    gdl90_frames_rx = 0
    alert_packets_rx = 0

    for step_idx in range(n_steps):
        t_s = sitl_config.t_start_s + step_idx * dt

        lat_o, lon_o, alt_o = wgs84.ecef_to_geodetic(*p_own_ecef)
        n_mat = wgs84.r_ecef_ned(lat_o, lon_o)
        hdg_rad = math.radians(heading_deg)
        pitch_rad = 0.0
        roll_rad = math.radians(roll_deg)
        vn_mps = own_speed_mps * math.cos(hdg_rad)
        ve_mps = own_speed_mps * math.sin(hdg_rad)

        p_tgt_ecef = scenario.target_position_ecef_at(t_s)
        r_ned = n_mat.T @ (p_tgt_ecef - p_own_ecef)
        d_horiz_m = float(math.hypot(float(r_ned[0]), float(r_ned[1])))
        d_3d_m = float(np.linalg.norm(r_ned))

        if d_horiz_m < min_hmd_m:
            min_hmd_m = d_horiz_m
            t_at_min_hmd_s = t_s
        if d_3d_m < min_3d_dist_m:
            min_3d_dist_m = d_3d_m

        disp_0 = p_own_ecef - p0_ecef
        north_m = float(np.dot(disp_0, n_hat_0))
        cross_track_m = abs(north_m)
        if cross_track_m > max_cross_track_m:
            max_cross_track_m = cross_track_m
        final_cross_track_m = cross_track_m

        hdg_dev = abs(_wrap_deg_180(heading_deg - 90.0))
        if hdg_dev > max_heading_dev_deg:
            max_heading_dev_deg = hdg_dev

        if enable_avoidance:
            own_nav = OwnshipNavState(
                lat_deg=lat_o,
                lon_deg=lon_o,
                alt_msl_m=alt_o,
                alt_geo_m=alt_o + 6.0,
                vn_mps=vn_mps,
                ve_mps=ve_mps,
                vd_mps=0.0,
                heading_deg=heading_deg,
            )

            mav_wire_pkts: list[bytes] = []
            gdl_wire_frms: list[bytes] = []

            if not (autopilot.recovery_latched and integrity_state == "VALID"):
                w_att = att_sigma_rad * np.asarray(
                    crn_draws["ins_step_rpy"][step_idx], dtype=np.float64
                )
                att_err_rpy = (
                    ins_beta * att_err_rpy
                    + math.sqrt(max(0.0, 1.0 - ins_beta**2)) * w_att
                )
                r_ned_body_est = rotations.euler_to_r(
                    hdg_rad + float(att_err_rpy[2]),
                    pitch_rad + float(att_err_rpy[1]),
                    roll_rad + float(att_err_rpy[0]),
                )

                r_ecef_body_tgt = scenario.target_attitude_ecef_at(t_s)
                raw_dets = _generate_ring_detections_for_step(
                    t_s=t_s,
                    step_idx=step_idx,
                    p_own_ecef=p_own_ecef,
                    n_mat=n_mat,
                    p_tgt_ecef=p_tgt_ecef,
                    r_ecef_body_tgt=r_ecef_body_tgt,
                    verts_model=verts_model,
                    yaw_true_rad=hdg_rad,
                    pitch_true_rad=pitch_rad,
                    roll_true_rad=roll_rad,
                    tracker=tracker,
                    trk_cfg=mode_trk_cfg,
                    crn_draws=crn_draws,
                )
                confirmed_tracks: list[Track] = tracker.step(
                    t_s=t_s,
                    raw_detections=raw_dets,
                    r_ned_body_ins=r_ned_body_est,
                    p_own_ecef=p_own_ecef,
                    r_ecef_ned=n_mat,
                )

                if confirmed_tracks:
                    if first_confirm_t_s is None:
                        first_confirm_t_s = float(t_s)
                        initial_confirm_range_3d_m = d_3d_m
                        initial_confirm_range_horiz_m = d_horiz_m

                    risk_by_id: dict[str, RiskEstimate] = {}
                    class_by_id: dict[str, str] = {}
                    for trk in confirmed_tracks:
                        risk_est = evaluate_track_risk(
                            track=trk,
                            t_s=t_s,
                            omega_thr_radps=omega_thr_radps,
                            k_sigma=k_sigma,
                            hmd_limit_m=LOWC_HMD_LIMIT_M,
                            class_id=scenario.class_id,
                            t_warning_s=t_warning_s,
                            t_corrective_s=t_corrective_s,
                            beta_wc=beta_wc,
                            k_theta=k_theta,
                            k_omega=k_omega,
                            alpha_r=alpha_r,
                            eta_floor=eta_floor,
                        )
                        if not use_g1_wc1:
                            risk_est = replace(
                                risk_est,
                                wc1_warning=False,
                                wc1_corrective=False,
                                hmd_hat_m=None,
                                tau_mod_hat_s=None,
                                tau_s_hat_s=None,
                            )
                        risk_by_id[trk.track_id] = risk_est
                        class_by_id[trk.track_id] = scenario.class_id

                    step_res = bridge.step_from_tracker(
                        ownship=own_nav,
                        confirmed_tracks=confirmed_tracks,
                        risk_by_track_id=risk_by_id,
                        integrity_state=integrity_state,
                        t_s=t_s,
                        class_by_track_id=class_by_id,
                    )
                    mav_wire_pkts = list(step_res.mavlink_frames)
                    gdl_wire_frms = list(step_res.gdl90_frames)
                    mavlink_packets_rx += len(mav_wire_pkts)
                    gdl90_frames_rx += len(gdl_wire_frms)
                    for m_rep in step_res.mavlink_messages:
                        if m_rep.squawk == 7700:
                            alert_packets_rx += 1

            cmd = autopilot.step_from_wire_packets(
                t_s=t_s,
                ownship=own_nav,
                mavlink_wire_packets=mav_wire_pkts,
                gdl90_wire_frames=gdl_wire_frms,
            )
            if cmd.alert_active and first_alert_t_s is None:
                first_alert_t_s = float(t_s)
            if cmd.mode == "AVOID_TURN" and first_maneuver_t_s is None:
                first_maneuver_t_s = float(t_s)

            heading_deg = (heading_deg + cmd.turn_rate_cmd_degps * dt) % 360.0
            roll_deg = cmd.bank_angle_deg

        hdg_next_rad = math.radians(heading_deg)
        v_next_ned = np.array(
            [
                own_speed_mps * math.cos(hdg_next_rad),
                own_speed_mps * math.sin(hdg_next_rad),
                0.0,
            ],
            dtype=np.float64,
        )
        p_own_ecef = p_own_ecef + (n_mat @ v_next_ned) * dt

    confirmed = first_confirm_t_s is not None
    alerted = first_alert_t_s is not None
    nmac_occurred = bool(min_hmd_m < sitl_config.nmac_threshold_m)
    nmac_resolved = bool(not nmac_occurred)
    lowc_recovered = bool(min_hmd_m >= sitl_config.lowc_threshold_m)

    return {
        "scenario_id": scenario.scenario_id,
        "geometry_stratum": scenario.geometry_stratum,
        "class_id": scenario.class_id,
        "characteristic_length_m": float(scenario.characteristic_length_m),
        "s_px_at_t0": s_px_at_t0,
        "system_mode": system_mode,
        "integrity_state": integrity_state,
        "open_loop_hmd_m": float(scenario.open_loop_hmd_m),
        "achieved_min_hmd_m": float(min_hmd_m),
        "achieved_min_3d_dist_m": float(min_3d_dist_m),
        "t_at_min_hmd_s": float(t_at_min_hmd_s),
        "confirmed": confirmed,
        "first_confirm_t_s": first_confirm_t_s,
        "initial_confirm_range_3d_m": float(initial_confirm_range_3d_m),
        "initial_confirm_range_horiz_m": float(initial_confirm_range_horiz_m),
        "alerted": alerted,
        "first_alert_t_s": first_alert_t_s,
        "first_maneuver_t_s": first_maneuver_t_s,
        "warning_lead_time_s": (
            float(scenario.t_cpa_open_s - first_alert_t_s)
            if first_alert_t_s is not None
            else 0.0
        ),
        "nmac_occurred": nmac_occurred,
        "nmac_resolved": nmac_resolved,
        "lowc_recovered": lowc_recovered,
        "max_cross_track_m": float(max_cross_track_m),
        "final_cross_track_m": float(final_cross_track_m),
        "max_heading_dev_deg": float(max_heading_dev_deg),
        "mavlink_packets_rx": int(mavlink_packets_rx),
        "gdl90_frames_rx": int(gdl90_frames_rx),
        "alert_packets_rx": int(alert_packets_rx),
        "induced_turns_count": int(autopilot.induced_turns_count),
        "invalid_coord_inhibits_count": int(autopilot.invalid_coord_inhibits_count),
    }


def summarize_system_runs(runs: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
    """Aggregate closed-loop system metrics over a sequence of encounter runs."""
    n_total = len(runs)
    if n_total == 0:
        raise ValueError(f"Cannot summarize empty run list for {label!r}")

    hmds = np.array([float(r["achieved_min_hmd_m"]) for r in runs], dtype=np.float64)
    conf_3d = np.array([float(r["initial_confirm_range_3d_m"]) for r in runs], dtype=np.float64)
    conf_h = np.array([float(r["initial_confirm_range_horiz_m"]) for r in runs], dtype=np.float64)
    leads_all = np.array([float(r["warning_lead_time_s"]) for r in runs], dtype=np.float64)
    leads_alerted = np.array(
        [float(r["warning_lead_time_s"]) for r in runs if r["alerted"]],
        dtype=np.float64,
    )
    xtk_max = np.array([float(r["max_cross_track_m"]) for r in runs], dtype=np.float64)
    xtk_fin = np.array([float(r["final_cross_track_m"]) for r in runs], dtype=np.float64)

    n_confirmed = sum(1 for r in runs if r["confirmed"])
    n_alerted = sum(1 for r in runs if r["alerted"])
    n_nmac_res = sum(1 for r in runs if r["nmac_resolved"])
    n_nmac_rem = sum(1 for r in runs if r["nmac_occurred"])
    n_lowc_rec = sum(1 for r in runs if r["lowc_recovered"])

    return {
        "label": label,
        "encounters_count": n_total,
        "confirmed_count": n_confirmed,
        "confirm_rate_pct": 100.0 * n_confirmed / n_total,
        "confirm_range_3d_mean_m": float(np.mean(conf_3d)),
        "confirm_range_3d_median_m": float(np.median(conf_3d)),
        "confirm_range_horiz_mean_m": float(np.mean(conf_h)),
        "confirm_range_horiz_median_m": float(np.median(conf_h)),
        "alerted_count": n_alerted,
        "alert_rate_pct": 100.0 * n_alerted / n_total,
        "lead_time_all_mean_s": float(np.mean(leads_all)),
        "lead_time_all_median_s": float(np.median(leads_all)),
        "lead_time_alerted_mean_s": float(np.mean(leads_alerted)) if len(leads_alerted) > 0 else 0.0,
        "lead_time_alerted_median_s": float(np.median(leads_alerted)) if len(leads_alerted) > 0 else 0.0,
        "nmac_remaining_count": n_nmac_rem,
        "nmac_resolved_count": n_nmac_res,
        "nmac_resolution_rate_pct": 100.0 * n_nmac_res / n_total,
        "lowc_recovered_count": n_lowc_rec,
        "lowc_recovery_rate_pct": 100.0 * n_lowc_rec / n_total,
        "hmd_median_m": float(np.median(hmds)),
        "hmd_p10_m": float(np.percentile(hmds, 10)),
        "hmd_min_m": float(np.min(hmds)),
        "hmd_max_m": float(np.max(hmds)),
        "max_cross_track_median_m": float(np.median(xtk_max)),
        "final_cross_track_median_m": float(np.median(xtk_fin)),
        "mavlink_packets_total": int(sum(int(r["mavlink_packets_rx"]) for r in runs)),
        "gdl90_frames_total": int(sum(int(r["gdl90_frames_rx"]) for r in runs)),
        "induced_turns_total": int(sum(int(r["induced_turns_count"]) for r in runs)),
    }


def compute_paired_gain_summary(
    runs_base: Sequence[Mapping[str, Any]],
    runs_target: Sequence[Mapping[str, Any]],
    comparison_id: str,
) -> dict[str, Any]:
    """Compute paired CRN encounter-by-encounter gains (`target - base`)."""
    if len(runs_base) != len(runs_target) or len(runs_base) == 0:
        raise ValueError("Paired runs must be non-empty and equal length")

    n = len(runs_base)
    d_conf = np.array(
        [
            float(rt["initial_confirm_range_3d_m"]) - float(rb["initial_confirm_range_3d_m"])
            for rb, rt in zip(runs_base, runs_target)
        ],
        dtype=np.float64,
    )
    d_lead = np.array(
        [
            float(rt["warning_lead_time_s"]) - float(rb["warning_lead_time_s"])
            for rb, rt in zip(runs_base, runs_target)
        ],
        dtype=np.float64,
    )
    d_hmd = np.array(
        [
            float(rt["achieved_min_hmd_m"]) - float(rb["achieved_min_hmd_m"])
            for rb, rt in zip(runs_base, runs_target)
        ],
        dtype=np.float64,
    )
    base_sum = summarize_system_runs(runs_base, "base")
    tgt_sum = summarize_system_runs(runs_target, "target")

    return {
        "comparison_id": comparison_id,
        "encounters_count": n,
        "delta_confirm_range_3d_mean_m": float(np.mean(d_conf)),
        "delta_confirm_range_3d_median_m": float(np.median(d_conf)),
        "delta_confirm_range_3d_max_m": float(np.max(d_conf)),
        "delta_lead_time_mean_s": float(np.mean(d_lead)),
        "delta_lead_time_median_s": float(np.median(d_lead)),
        "delta_lead_time_max_s": float(np.max(d_lead)),
        "delta_hmd_mean_m": float(np.mean(d_hmd)),
        "delta_hmd_median_m": float(np.median(d_hmd)),
        "delta_min_hmd_floor_m": float(tgt_sum["hmd_min_m"] - base_sum["hmd_min_m"]),
        "delta_nmac_resolved_count": int(tgt_sum["nmac_resolved_count"] - base_sum["nmac_resolved_count"]),
        "delta_nmac_resolution_rate_pp": float(
            tgt_sum["nmac_resolution_rate_pct"] - base_sum["nmac_resolution_rate_pct"]
        ),
        "delta_lowc_recovered_count": int(tgt_sum["lowc_recovered_count"] - base_sum["lowc_recovered_count"]),
        "delta_lowc_recovery_rate_pp": float(
            tgt_sum["lowc_recovery_rate_pct"] - base_sum["lowc_recovery_rate_pct"]
        ),
        "improved_confirm_range_count": int(np.sum(d_conf > 1.0)),
        "improved_lead_time_count": int(np.sum(d_lead > 0.05)),
    }


def run_d0_vs_d1_closed_loop_sensitivity(
    *,
    d0_closure_path: Path | str | None = None,
    d1_closure_path: Path | str | None = None,
    sitl_config: SitlAvoidanceConfig | None = None,
    wp07_att_sigma_deg: float | None = None,
    max_encounters: int | None = None,
) -> dict[str, Any]:
    """Run the 50-encounter CRN closed-loop sensitivity matrix comparing `D0` vs `D1` closures."""
    d0_path = resolve_d0_closure_config_path(d0_closure_path)
    d1_path = resolve_d1_closure_config_path(d1_closure_path)

    base_g1_cfg = load_g1_tracking_config()
    d0_raw = json.loads(d0_path.read_text(encoding="utf-8"))
    d1_raw = json.loads(d1_path.read_text(encoding="utf-8"))

    trk_cfg_d0, profile_d0 = build_detector_closure_profile(
        base_g1_cfg, d0_raw, "D0", d0_path
    )
    trk_cfg_d1, profile_d1 = build_detector_closure_profile(
        base_g1_cfg, d1_raw, "D1", d1_path
    )

    cfg = sitl_config if sitl_config is not None else SitlAvoidanceConfig.from_display_config()
    n_enc = max_encounters if max_encounters is not None else cfg.benchmark_threat_count
    scenarios = build_g3a_collision_threat_scenarios(
        count=n_enc,
        max_open_loop_hmd_m=cfg.open_loop_cpa_max_m,
    )

    bridge = PassiveTrackAvionicsBridge()
    n_steps = int(round((cfg.t_end_s - cfg.t_start_s) / cfg.dt_s)) + 1

    mode_specs = (
        ("open_loop", trk_cfg_d0),
        ("d0_g0_a1", trk_cfg_d0),
        ("d0_g1_wc1_b050", trk_cfg_d0),
        ("d1_g1_wc1_b050", trk_cfg_d1),
        ("d0_g1_wc1_b025", trk_cfg_d0),
        ("d1_g1_wc1_b025", trk_cfg_d1),
    )
    runs_by_mode: dict[str, list[dict[str, Any]]] = {m: [] for m, _ in mode_specs}

    for scen in scenarios:
        crn = _pregenerate_crn_draws(scen.seed, n_steps)
        for m_name, m_trk_cfg in mode_specs:
            res = simulate_system_encounter_closed_loop(
                scen,
                system_mode=m_name,
                tracking_config=m_trk_cfg,
                bridge=bridge,
                sitl_config=cfg,
                wp07_att_sigma_deg=wp07_att_sigma_deg,
                integrity_state="VALID",
                crn_draws=crn,
            )
            runs_by_mode[m_name].append(res)

    summaries_all = {
        m_name: summarize_system_runs(runs_by_mode[m_name], m_name)
        for m_name, _ in mode_specs
    }

    uas_indices = [i for i, s in enumerate(scenarios) if s.class_id == "C1_UAS"]
    manned_indices = [i for i, s in enumerate(scenarios) if s.class_id != "C1_UAS"]

    summaries_c1_uas: dict[str, Any] = {}
    if uas_indices:
        for m_name, _ in mode_specs:
            sub_runs = [runs_by_mode[m_name][i] for i in uas_indices]
            summaries_c1_uas[m_name] = summarize_system_runs(sub_runs, f"{m_name}:C1_UAS")

    summaries_manned: dict[str, Any] = {}
    if manned_indices:
        for m_name, _ in mode_specs:
            sub_runs = [runs_by_mode[m_name][i] for i in manned_indices]
            summaries_manned[m_name] = summarize_system_runs(sub_runs, f"{m_name}:C2_C6")

    paired_gains: dict[str, Any] = {
        "all_50_d1_vs_d0_b050": compute_paired_gain_summary(
            runs_by_mode["d0_g1_wc1_b050"],
            runs_by_mode["d1_g1_wc1_b050"],
            "all_50:d1_g1_wc1_b050_minus_d0_g1_wc1_b050",
        ),
        "all_50_d1_vs_d0_b025": compute_paired_gain_summary(
            runs_by_mode["d0_g1_wc1_b025"],
            runs_by_mode["d1_g1_wc1_b025"],
            "all_50:d1_g1_wc1_b025_minus_d0_g1_wc1_b025",
        ),
        "all_50_full_g1_vs_g0_system": compute_paired_gain_summary(
            runs_by_mode["d0_g0_a1"],
            runs_by_mode["d1_g1_wc1_b050"],
            "all_50:d1_g1_wc1_b050_minus_d0_g0_a1",
        ),
    }
    if uas_indices:
        paired_gains["c1_uas_d1_vs_d0_b050"] = compute_paired_gain_summary(
            [runs_by_mode["d0_g1_wc1_b050"][i] for i in uas_indices],
            [runs_by_mode["d1_g1_wc1_b050"][i] for i in uas_indices],
            "c1_uas:d1_g1_wc1_b050_minus_d0_g1_wc1_b050",
        )
        paired_gains["c1_uas_d1_vs_d0_b025"] = compute_paired_gain_summary(
            [runs_by_mode["d0_g1_wc1_b025"][i] for i in uas_indices],
            [runs_by_mode["d1_g1_wc1_b025"][i] for i in uas_indices],
            "c1_uas:d1_g1_wc1_b025_minus_d0_g1_wc1_b025",
        )

    per_encounter_table: list[dict[str, Any]] = []
    for idx_s, scen in enumerate(scenarios):
        r_open = runs_by_mode["open_loop"][idx_s]
        r_g0 = runs_by_mode["d0_g0_a1"][idx_s]
        r_d0 = runs_by_mode["d0_g1_wc1_b050"][idx_s]
        r_d1 = runs_by_mode["d1_g1_wc1_b050"][idx_s]
        per_encounter_table.append(
            {
                "scenario_id": scen.scenario_id,
                "geometry_stratum": scen.geometry_stratum,
                "class_id": scen.class_id,
                "s_px_at_t0": float(r_d0["s_px_at_t0"]),
                "open_loop_hmd_m": float(r_open["achieved_min_hmd_m"]),
                "d0_g0_a1_hmd_m": float(r_g0["achieved_min_hmd_m"]),
                "d0_g0_a1_lead_s": float(r_g0["warning_lead_time_s"]),
                "d0_confirm_range_3d_m": float(r_d0["initial_confirm_range_3d_m"]),
                "d1_confirm_range_3d_m": float(r_d1["initial_confirm_range_3d_m"]),
                "d0_lead_time_s": float(r_d0["warning_lead_time_s"]),
                "d1_lead_time_s": float(r_d1["warning_lead_time_s"]),
                "d0_hmd_m": float(r_d0["achieved_min_hmd_m"]),
                "d1_hmd_m": float(r_d1["achieved_min_hmd_m"]),
                "d0_lowc_recovered": bool(r_d0["lowc_recovered"]),
                "d1_lowc_recovered": bool(r_d1["lowc_recovered"]),
            }
        )

    return {
        "encounters_count": len(scenarios),
        "c1_uas_count": len(uas_indices),
        "manned_c2_c6_count": len(manned_indices),
        "wp07_coupled_att_sigma_deg": (
            float(wp07_att_sigma_deg)
            if wp07_att_sigma_deg is not None
            else float(base_g1_cfg["ins_sigma_deg"])
        ),
        "detector_closures": {
            "D0": profile_d0,
            "D1": profile_d1,
        },
        "modes_all_50": summaries_all,
        "modes_c1_uas": summaries_c1_uas,
        "modes_manned_c2_c6": summaries_manned,
        "paired_gains": paired_gains,
        "per_encounter": per_encounter_table,
    }


def evaluate_wp07_g1_end_to_end_transitions(
    *,
    wp07_results_path: Path | str | None = None,
    spec: Mapping[str, Any] | None = None,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Execute the real WP-07 G1 -> HUD -> Avionics Bridge -> SITL Autopilot integrity loop
    across `VALID` -> `F_isolated` -> `INVALID_ALERT` -> `VALID` recovery on real WP-01 G4
    4-camera (`FRBL`) CesiumJS sequences (`T1_cruise` `ge_1000m` & `T2_ksfo28l` `lt_250m`).

    Verifies `0` end-to-end HMI frames across all phases and both altitude regimes.
    """
    wp07_path = resolve_wp07_g1_results_path(wp07_results_path, data_root=data_root)
    wp07_data = json.loads(wp07_path.read_text(encoding="utf-8"))

    sp = dict(spec) if spec is not None else json.loads(
        DEFAULT_B14_SPEC_PATH.read_text(encoding="utf-8")
    )
    disp_cfg = json.loads(DEFAULT_DISPLAY_G1_CONFIG_PATH.read_text(encoding="utf-8"))

    al_primary_deg = float(sp.get("wp07_alert_limit_deg", 0.50))  # default-ok: WP-07 G1 primary alert limit spec default
    al_secondary_deg = float(sp.get("wp07_secondary_alert_limit_deg", 1.00))  # default-ok: WP-07 G1 secondary alert limit spec default
    alpha_pl = float(sp.get("wp07_alpha_pl", 0.01))  # default-ok: WP-07 G1 conformal miscoverage level default
    mis_cam = str(sp.get("wp07_misalignment_cam_id", "R"))
    mis_pitch_deg = float(sp.get("wp07_misalignment_pitch_deg", 0.80))  # default-ok: B14 single-cam pitch fault injection default
    mis_roll_deg = float(sp.get("wp07_misalignment_roll_deg", -0.45))  # default-ok: B14 single-cam roll fault injection default
    alt_bias_m = float(sp.get("wp07_altitude_bias_fault_m", 2500.0))  # default-ok: B14 global altitude bias fault injection default
    d27_iso_thr_deg = float(sp.get("wp07_d27_isolation_threshold_deg", 0.12))  # default-ok: WP-07 G1 D27 camera isolation threshold default

    # Load real calibrated MonitorThresholdsG1 and altitude-Mondrian ConformalModelG1 from WP-07 G1 results.json
    th_g1 = MonitorThresholdsG1.from_dict(
        wp07_data["item2_altitude_conditioning"]["thresholds"]["far_0.01"]
    )
    cm_raw = wp07_data["item2_altitude_conditioning"]["conformal_models"]["far_0.01"]
    conformal_bundle = {
        "base_model": ConformalModel.from_dict(cm_raw["base_model"]),
        "altitude_mondrian_quantiles": cm_raw["altitude_mondrian_quantiles"],
    }

    sensor_cfg = SensorConfig.from_file(str(DEFAULT_SENSOR_CONFIG_PATH))
    sims = {c: CameraSimulator(sensor_cfg, f"CAM_{c}") for c in ("F", "R", "B", "L")}
    base_rig = load_pipeline_rig()
    vp_cam = get_viewport_camera(disp_cfg)
    sitl_cfg = SitlAvoidanceConfig.from_display_config(disp_cfg)

    sequence_schedules = (
        (
            "T1_cruise",
            False,
            (
                ("VALID", 2),
                ("VALID", 3),
                ("F_isolated", 4),
                ("F_isolated", 5),
                ("INVALID_ALERT", 6),
                ("INVALID_ALERT", 7),
                ("VALID_RECOVER", 8),
                ("VALID_RECOVER", 9),
            ),
        ),
        (
            "T2_ksfo28l",
            True,
            (
                ("VALID", 10),
                ("VALID", 11),
                ("F_isolated", 12),
                ("F_isolated", 13),
                ("INVALID_ALERT", 14),
                ("INVALID_ALERT", 15),
                ("VALID_RECOVER", 16),
                ("VALID_RECOVER", 17),
            ),
        ),
    )

    frame_records: list[dict[str, Any]] = []
    hmi_al_050 = 0
    hmi_al_100 = 0
    f_isolated_attempts = 0
    f_isolated_successes = 0
    f_isolated_correct_cam = 0
    invalid_alert_frames = 0
    invalid_alert_detected = 0
    invalid_alert_false_turns = 0
    invalid_alert_hud_blanked = 0
    max_ttd_invalid_frames = 0

    for seq_name, is_t2, phase_frames in sequence_schedules:
        seq_dir = resolve_g4_sequence_dir(seq_name, data_root=data_root)
        raw_lines = [
            ln.strip()
            for ln in (seq_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]

        bridge = PassiveTrackAvionicsBridge()
        # Separate fresh autopilot per phase check so reaction delay does not mask turn capability/inhibition
        for step_num, (phase_name, k_idx) in enumerate(phase_frames):
            rec = json.loads(raw_lines[k_idx])
            ac = rec["aircraft"]
            lat_deg = float(ac["lat_deg"])
            lon_deg = float(ac["lon_deg"])
            h_true_m = float(ac["h_m"])
            yaw_deg, pitch_true_deg, roll_true_deg = (
                float(ac["ypr_deg"][0]),
                float(ac["ypr_deg"][1]),
                float(ac["ypr_deg"][2]),
            )
            yaw_rad = math.radians(yaw_deg)
            v_ned = np.asarray(ac["v_ned_mps"], dtype=np.float64)

            h_meas_m = h_true_m + (alt_bias_m if phase_name == "INVALID_ALERT" else 0.0)

            cam_images_rgb: dict[str, np.ndarray] = {}
            edges_by_cam: dict[str, list[dict[str, float]]] = {}
            ood_by_cam: dict[str, dict[str, float]] = {}

            for cid in ("F", "R", "B", "L", "U", "D"):
                with Image.open(seq_dir / rec["cams"][cid]["png"]) as im:
                    img_rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
                cam_images_rgb[cid] = img_rgb
                if cid in ("F", "R", "B", "L"):
                    deg_img = sims[cid].process_frame(
                        img_rgb,
                        image_id=f"{seq_name}_seed1",
                        frame_idx=k_idx,
                        warp_method="cv2",
                    )["display"]
                    edges_c = detect_camera_edges_multiscale(
                        deg_img, cid, base_rig.camera_model
                    )
                    edges_by_cam[cid] = edges_c
                    ood_by_cam[cid] = compute_horizon_strip_ood_features(
                        deg_img, edges=edges_c, cam=base_rig.camera_model, cam_id=cid
                    )

            rig_use = base_rig
            if phase_name == "F_isolated":
                rig_use, _ = inject_single_cam_misalignment(
                    base_rig,
                    cam_id=mis_cam,
                    step_pitch_deg=mis_pitch_deg,
                    step_roll_deg=mis_roll_deg,
                )

            batch = build_pipeline_ray_batch(
                edges_by_cam, ("F", "R", "B", "L"), rig_use, sensor_config=sensor_cfg
            )
            fit = fit_pipeline_attitude_g1(
                lat_deg, lon_deg, h_meas_m, yaw_rad, batch, frame_k=k_idx
            )
            inlier_idx = fit["inliers"]
            inlier_batch = LimbRayBatch(
                batch.d_body[inlier_idx],
                batch.lever_arm_body[inlier_idx],
                [batch.cam_ids[i] for i in inlier_idx],
                weights=batch.weights[inlier_idx],
                uvs=batch.uvs[inlier_idx],
            )
            cc = camera_consistency(
                fit, inlier_batch, rig_use, lat_deg, lon_deg, h_meas_m, yaw_rad
            )
            cov_rad2 = fit["covariance_rad2"]
            rec_mon = {
                "status": fit["status"],
                "cam_set": "FRBL",
                "h_meas_m": h_meas_m,
                "rms_residual_urad": float(fit["rms_residual_urad"]),
                "condition_number": float(np.linalg.cond(cov_rad2)),
                "inlier_fraction": float(fit["inlier_count"] / max(1, batch.n_rays)),
                "inlier_count": int(fit["inlier_count"]),
                "camera_consistency_deg": float(cc.max_angle_deg),
                "ood_horizon_score": max(
                    ood_by_cam[c]["ood_horizon_score"] for c in ("F", "R", "B", "L")
                ),
            }
            mon = evaluate_monitor_alarm_g1(rec_mon, th_g1)

            integrity_state = "VALID"
            f_isolated = False
            isolated_cam: str | None = None
            eff_fit = fit

            if mon["alarm"]:
                iso_res = isolate_and_exclude_faulty_camera(
                    full_fit=fit,
                    inlier_batch=batch,
                    rig=rig_use,
                    lat_deg=lat_deg,
                    lon_deg=lon_deg,
                    h_meas_m=h_meas_m,
                    yaw_rad=yaw_rad,
                    cam_set_name="FRBL",
                    frame_k=k_idx,
                    d27_threshold_deg=d27_iso_thr_deg,
                )
                if iso_res["isolation_succeeded"] and not iso_res["alarm_after_isolation"]:
                    integrity_state = "VALID"
                    f_isolated = True
                    isolated_cam = str(iso_res["isolated_camera"])
                    eff_fit = iso_res["post_isolation_fit"]
                else:
                    integrity_state = "INVALID_ALERT"

            sig_hat_rad = max(
                1e-6, math.sqrt(max(0.0, float(np.trace(eff_fit["covariance_rad2"]))))
            )
            pl_001_deg = compute_pl_deg_altitude_mondrian(
                conformal_bundle,
                sig_hat_rad,
                alpha_pl,
                "FRBL",
                "degraded",
                h_meas_m,
            )
            pitch_est_deg = float(eff_fit["pitch_deg"])
            roll_est_deg = float(eff_fit["roll_deg"])
            att_err_deg = float(
                math.hypot(pitch_est_deg - pitch_true_deg, roll_est_deg - roll_true_deg)
            )

            # Couple directly into WP-08 G1 Viewport Fusion + Conformal HUD
            aircraft_pose_est = {
                "lat_deg": lat_deg,
                "lon_deg": lon_deg,
                "h_m": h_meas_m,
                "yaw_rad": yaw_rad,
                "pitch_rad": math.radians(pitch_est_deg),
                "roll_rad": math.radians(roll_est_deg),
            }
            r_ned_body_est = rotations.euler_to_r(
                yaw_rad,
                math.radians(pitch_est_deg),
                math.radians(roll_est_deg),
            )
            vp_pose = compute_viewport_pose_from_estimated_attitude(
                ownship_lat_deg=lat_deg,
                ownship_lon_deg=lon_deg,
                ownship_h_m=h_meas_m,
                r_ned_body_est=r_ned_body_est,
                head_euler_rad=(0.0, 0.0, 0.0),
                config=disp_cfg,
            )
            p_ecef_view = vp_pose["p_ecef_view"]
            r_ecef_view_est = vp_pose["r_ecef_view_est"]
            excl_cams = (isolated_cam,) if (f_isolated and isolated_cam) else ()
            vp_img = extract_viewport(
                camera_images=cam_images_rgb,
                rig=rig_use,
                aircraft_pose=aircraft_pose_est,
                head_euler_rad=(0.0, 0.0, 0.0),
                depth_model="ellipsoid",
                excluded_cameras=excl_cams,
                config=disp_cfg,
            )

            active_threat_track = {
                "track_id": "T0001",
                "t_s": float(step_num),
                "az": yaw_rad + 0.14,
                "el": 0.01,
                "az_rate": -0.0022,
                "el_rate": 0.0,
                "s": 0.0070,
                "s_mom": 0.0070,
                "s_dot": 0.0004,
            }
            active_threat_risk = {
                "t_s": float(step_num),
                "wc1_warning": True,
                "wc1_corrective": True,
                "range_interval_m": (900.0, 1400.0),
                "hmd_min_m": 40.0,
                "hmd_max_m": 220.0,
                "tau_mod_min_s": 11.0,
                "tau_mod_max_s": 18.0,
            }

            _, hud_tel = draw_g1_symbology(
                img=vp_img,
                aircraft_pose_est=aircraft_pose_est,
                v_ned=v_ned,
                p_ecef_view=p_ecef_view,
                r_ecef_view_est=r_ecef_view_est,
                r_ned_body_est=r_ned_body_est,
                head_euler_rad=(0.0, 0.0, 0.0),
                view_cam=vp_cam,
                tracks_and_risks=[(active_threat_track, active_threat_risk)],
                integrity_state=integrity_state,
                pl_deg=pl_001_deg,
                f_isolated=f_isolated,
                isolated_camera_id=isolated_cam,
                is_t2=is_t2,
                frame_index=k_idx,
            )

            # Couple directly into PassiveTrackAvionicsBridge -> MAVLink #246 + GDL-90 -> SITL Autopilot
            own_nav = OwnshipNavState(
                lat_deg=lat_deg,
                lon_deg=lon_deg,
                alt_msl_m=h_meas_m,
                alt_geo_m=h_meas_m + 6.0,
                vn_mps=float(v_ned[0]),
                ve_mps=float(v_ned[1]),
                vd_mps=float(v_ned[2]),
                heading_deg=yaw_deg % 360.0,
            )
            step_out = bridge.step_from_tracker(
                ownship=own_nav,
                confirmed_tracks=[active_threat_track],
                risk_by_track_id={"T0001": active_threat_risk},
                integrity_state=integrity_state,
                t_s=10.0,
                class_by_track_id={"T0001": "C2_SEP"},
            )
            mav_decoded = unpack_adsb_vehicle(step_out.mavlink_frames[0])
            coords_valid_flag = bool(mav_decoded.flags & ADSB_FLAGS_VALID_COORDS)
            gdl_nic = int(step_out.gdl90_traffic_reports[0].nic)

            ap = SitlAvoidanceAutopilot(
                nominal_heading_deg=yaw_deg % 360.0,
                config=replace(sitl_cfg, reaction_delay_s=0.0),
            )
            ap_cmd = ap.step_from_wire_packets(
                t_s=10.0,
                ownship=own_nav,
                mavlink_wire_packets=step_out.mavlink_frames,
                gdl90_wire_frames=step_out.gdl90_frames,
            )

            horizon_drawn = bool(not hud_tel["horizon_suppressed"])
            e2e_active_output = bool(
                horizon_drawn or coords_valid_flag or ap_cmd.mode == "AVOID_TURN"
            )
            is_hmi_050 = bool(
                e2e_active_output and att_err_deg > al_primary_deg and att_err_deg > pl_001_deg
            )
            is_hmi_100 = bool(
                e2e_active_output and att_err_deg > al_secondary_deg and att_err_deg > pl_001_deg
            )
            if is_hmi_050:
                hmi_al_050 += 1
            if is_hmi_100:
                hmi_al_100 += 1

            if phase_name == "F_isolated":
                f_isolated_attempts += 1
                if f_isolated and integrity_state == "VALID":
                    f_isolated_successes += 1
                if isolated_cam == mis_cam:
                    f_isolated_correct_cam += 1
            elif phase_name == "INVALID_ALERT":
                invalid_alert_frames += 1
                if integrity_state == "INVALID_ALERT":
                    invalid_alert_detected += 1
                    max_ttd_invalid_frames = max(max_ttd_invalid_frames, 1)
                if hud_tel["horizon_suppressed"]:
                    invalid_alert_hud_blanked += 1
                if ap_cmd.mode == "AVOID_TURN" or ap.induced_turns_count > 0:
                    invalid_alert_false_turns += 1

            frame_records.append(
                {
                    "seq_name": seq_name,
                    "frame_k": k_idx,
                    "phase": phase_name,
                    "altitude_band": str(mon["altitude_band"]),
                    "h_true_m": h_true_m,
                    "h_meas_m": h_meas_m,
                    "raw_monitor_alarm": bool(mon["alarm"]),
                    "integrity_state": integrity_state,
                    "f_isolated": f_isolated,
                    "isolated_camera": isolated_cam,
                    "att_error_deg": att_err_deg,
                    "pl_001_deg": pl_001_deg,
                    "hud_horizon_suppressed": bool(hud_tel["horizon_suppressed"]),
                    "mavlink_valid_coords": coords_valid_flag,
                    "gdl90_nic": gdl_nic,
                    "autopilot_mode": ap_cmd.mode,
                    "hmi_al_050": is_hmi_050,
                    "hmi_al_100": is_hmi_100,
                }
            )

    valid_recs = [r for r in frame_records if r["phase"] in ("VALID", "VALID_RECOVER")]
    f_iso_recs = [r for r in frame_records if r["phase"] == "F_isolated"]
    inv_recs = [r for r in frame_records if r["phase"] == "INVALID_ALERT"]

    # Extract upstream WP-07 G1 full test-split protection level & HMI metrics for cross-check
    cov_001 = wp07_data["item2_altitude_conditioning"]["test_coverage_and_availability"][
        "far_0.01"
    ]["coverage"]["altitude_mondrian"]["0.01"]
    avail_050 = wp07_data["item2_altitude_conditioning"]["test_coverage_and_availability"][
        "far_0.01"
    ]["availability"]["altitude_mondrian"]["0.01"]["al_0.5"]
    f_iso_frbl = wp07_data["item4_single_cam_misalignment_and_isolation"][
        "f_isolated_by_cam_set"
    ]["FRBL"]

    return {
        "wp07_results_path": str(wp07_path.as_posix()),
        "wp07_results_sha256": file_sha256(wp07_path),
        "wp07_producer_commit": str(wp07_data.get("git_commit", "")),
        "total_loop_frames": len(frame_records),
        "sequences_evaluated_count": len(sequence_schedules),
        "valid_frames_count": len(valid_recs),
        "valid_exact_state_count": sum(
            1
            for r in valid_recs
            if r["integrity_state"] == "VALID"
            and not r["f_isolated"]
            and not r["hud_horizon_suppressed"]
            and r["mavlink_valid_coords"]
            and r["autopilot_mode"] == "AVOID_TURN"
        ),
        "valid_median_att_error_deg": float(
            np.median([r["att_error_deg"] for r in valid_recs])
        ),
        "valid_median_pl_001_deg": float(np.median([r["pl_001_deg"] for r in valid_recs])),
        "f_isolated_frames_count": f_isolated_attempts,
        "f_isolated_succeeded_count": f_isolated_successes,
        "f_isolated_correct_cam_count": f_isolated_correct_cam,
        "f_isolated_median_att_error_deg": float(
            np.median([r["att_error_deg"] for r in f_iso_recs])
        ),
        "f_isolated_median_pl_001_deg": float(
            np.median([r["pl_001_deg"] for r in f_iso_recs])
        ),
        "invalid_alert_frames_count": invalid_alert_frames,
        "invalid_alert_detected_count": invalid_alert_detected,
        "invalid_alert_hud_blanked_count": invalid_alert_hud_blanked,
        "invalid_alert_inhibited_count": sum(
            1 for r in inv_recs if r["autopilot_mode"] == "INHIBITED_INVALID_COORDS"
        ),
        "invalid_alert_false_turns_count": invalid_alert_false_turns,
        "ttd_invalid_frames": max_ttd_invalid_frames,
        "end_to_end_hmi_al_050": hmi_al_050,
        "end_to_end_hmi_al_100": hmi_al_100,
        "wp07_test_split_reference": {
            "n_alarm_free": int(cov_001["n"]),
            "coverage_001_pct": 100.0 * float(cov_001["rate"]),
            "median_pl_001_deg": float(cov_001["median_pl_deg"]),
            "p95_pl_001_deg": float(cov_001["p95_pl_deg"]),
            "availability_al_050_pct": 100.0 * float(avail_050["rate"]),
            "frbl_post_isolation_rmse_deg": float(f_iso_frbl["post_isolation_rmse_deg"]),
            "frbl_isolation_rate_pct": 100.0 * float(f_iso_frbl["isolation_rate"]["rate"]),
            "frbl_correct_cam_id_pct": 100.0 * float(f_iso_frbl["correct_camera_id_rate"]["rate"]),
            "wp07_total_hmi_al_050": int(wp07_data["hmi"]["al_0.50"]["total_hmi"]),
            "wp07_total_hmi_al_100": int(wp07_data["hmi"]["al_1.00"]["total_hmi"]),
        },
        "frames": frame_records,
    }
