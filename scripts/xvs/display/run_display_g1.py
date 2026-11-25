"""Single-process WP-08 Gate G1 Producer: Live Track/Risk Symbology, Integrity HUD & Closed-Loop SITL Avoidance (Batch B12).

Executes deterministic verification and rendering for WP-08 Gate G1 (`docs/xvs/WP08_DISPLAY.md` §8
and `docs/xvs/batches/BATCH_B12_WP08_G1_ARGON.md`):
  1. V1_G1: Conformal passive track reticle projection accuracy (`<= 0.05 px`), 4-tier RTCA DO-365B /
     DO-317C symbology (`NONE`, `PREVENTIVE`, `CORRECTIVE`, `WARNING`), 5.0 s angular-rate leader line,
     3-sigma covariance halo, and exact `P_hat(WCV) >= beta_WC <=> wc1_corrective(beta_WC)` duality.
  2. V2_G1: WP-07 Horizon Integrity HUD coupling (`VALID`, `VALID` + `F_isolated`, `DEGRADED_COAST`
     coasting PL widening, and 1-frame `INVALID_ALERT` conformal horizon blanking with 0 HUD HMI frames).
  3. V3_G1: 50-encounter CRN closed-loop MAVLink `ADSB_VEHICLE` (#246) -> ArduPilot SITL horizontal
     avoidance benchmark (`Open-Loop` vs `G0 A1` vs `G1 WC1 beta_WC = 0.50` vs `G1 WC1 beta_WC = 0.25`).
  4. V4_G1: `INVALID_ALERT` (`VALID_COORDS == 0`, `NIC == 0`) SITL autonomous turn inhibition verification.
  5. V5_G1: Static AST + runtime `PoisonedTruthTrack` sentinel truth-isolation audit across
     `src/xvs/display/symbology.py` and `src/xvs/avionics/sitl_closed_loop.py`.
  6. V6_G1: Repeated-run byte-identity verification on JSON and rendered PNG stills (`workers = 1`).

Writes:
  - `output/xvs/wp08/g1/results.json`
  - `output/xvs/wp08/g1/run_manifest.json`
  - `output/xvs/wp08/g1/stills/*.png` and `output/xvs/wp08/g1/stills/g1_hud_sequence_preview.webp`
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.avionics.bridge import PassiveTrackInput
from xvs.avionics.sitl_closed_loop import (
    SitlAvoidanceConfig,
    run_closed_loop_sitl_benchmark,
)
from xvs.paths import get_data_root
from xvs.display.fusion import render_display_panorama
from xvs.display.latency import (
    compute_coasting_pl_deg,
    evaluate_integrity_display_blanking_latency,
)
from xvs.display.symbology import (
    classify_traffic_symbology_tier,
    compute_conformal_horizon_with_pl_band,
    compute_p_wcv_belief,
    compute_rect_intersection_area,
    draw_g1_symbology,
    draw_panorama_g1_symbology,
    passive_bearing_ned_and_jacobian,
    project_passive_track_to_panorama,
    project_passive_track_to_viewport,
)
from xvs.display.viewport import (
    compute_viewport_pose_from_estimated_attitude,
    extract_viewport,
    get_viewport_camera,
)
from xvs.geo import rotations, wgs84
from xvs.rig.camera import PinholeCamera
from xvs.rig.rig import Rig
from xvs.sim.horizon import compute_horizon_gt
from xvs.tracking.risk import LOWC_HMD_LIMIT_M, RiskEstimate

DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "xvs" / "display" / "g1.json"
DEFAULT_SPEC_PATH = REPO_ROOT / "configs" / "xvs" / "display" / "g1_spec.json"
DEFAULT_TIER0_JSON_PATH = (
    REPO_ROOT / "output" / "xvs" / "wp08" / "tier0_b10" / "tier0_verification.json"
)

TRACKED_CODE_FILES = (
    "src/xvs/display/__init__.py",
    "src/xvs/display/symbology.py",
    "src/xvs/display/viewport.py",
    "src/xvs/display/latency.py",
    "src/xvs/avionics/__init__.py",
    "src/xvs/avionics/sitl_closed_loop.py",
    "scripts/xvs/display/run_display_g1.py",
    "scripts/xvs/display/report_tables_wp08_g1.py",
    "tests/xvs/test_display_g1.py",
)

FORBIDDEN_TRUTH_IDENTIFIERS = frozenset(
    {
        "truth_id",
        "true_range",
        "true_range_m",
        "range_true",
        "range_true_m",
        "p_int_ned_true",
        "intruder_pos_true",
        "tau_true_s",
        "omega_true_radps",
    }
)

_R_BODY_CAM_FORWARD = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def _get_git_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=10,
        check=True,
    )
    return proc.stdout.strip()


def _get_git_status_porcelain() -> str:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=10,
        check=True,
    )
    lines = [
        ln
        for ln in proc.stdout.splitlines()
        if ln.strip()
        and "output/xvs/wp08/g1/" not in ln.replace("\\", "/")
        and "docs/xvs/reports/WP08_G1_report.md" not in ln.replace("\\", "/")
    ]
    return "\n".join(lines).strip()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_code_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for rel in TRACKED_CODE_FILES:
        p = REPO_ROOT / rel
        if p.is_file():
            hashes[rel] = _sha256_file(p)
    return hashes


class PoisonedTruthTrack:
    """Runtime sentinel object that raises RuntimeError if any ground-truth attribute is accessed."""

    def __init__(
        self,
        track_id: str = "T0099",
        az: float = 0.08,
        el: float = 0.02,
        az_rate: float = -0.002,
        el_rate: float = 0.001,
        s_mom: float = 0.006,
        s_dot: float = 0.0004,
    ) -> None:
        self.track_id = track_id
        self.az = az
        self.el = el
        self.az_rate = az_rate
        self.el_rate = el_rate
        self.s = s_mom
        self.s_mom = s_mom
        self.s_dot = s_dot
        self.omega_los = math.hypot(az_rate, el_rate)
        self.P = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
        self.time_since_update = 0.0

    @property
    def truth_id(self) -> str:
        raise RuntimeError("V5_G1 VIOLATION: accessed forbidden attribute truth_id")

    @property
    def true_range(self) -> float:
        raise RuntimeError("V5_G1 VIOLATION: accessed forbidden attribute true_range")

    @property
    def true_range_m(self) -> float:
        raise RuntimeError("V5_G1 VIOLATION: accessed forbidden attribute true_range_m")

    @property
    def range_true_m(self) -> float:
        raise RuntimeError("V5_G1 VIOLATION: accessed forbidden attribute range_true_m")

    @property
    def p_int_ned_true(self) -> Any:
        raise RuntimeError("V5_G1 VIOLATION: accessed forbidden attribute p_int_ned_true")

    @property
    def tau_true_s(self) -> float:
        raise RuntimeError("V5_G1 VIOLATION: accessed forbidden attribute tau_true_s")


def evaluate_v1_conformal_traffic_reticle(cfg: dict[str, Any]) -> dict[str, Any]:
    """Evaluate V1_G1 conformal passive track reticle projection and 4-tier classification."""
    view_cam = get_viewport_camera(cfg)
    pano_w = int(cfg["panorama"]["width"])
    pano_h = int(cfg["panorama"]["height"])
    trf_cfg = cfg["traffic_symbology"]
    leader_s = float(trf_cfg["leader_horizon_s"])
    sigma_mult = float(trf_cfg["covariance_halo_sigma_mult"])

    rng = np.random.default_rng(20261108)
    n_samples = 200
    vp_errors_px: list[float] = []
    q8_errors_px: list[float] = []
    pano_errors_px: list[float] = []
    leader_errors_px: list[float] = []

    for _ in range(n_samples):
        yaw_deg = float(rng.uniform(-30.0, 30.0))
        pitch_deg = float(rng.uniform(-10.0, 10.0))
        roll_deg = float(rng.uniform(-15.0, 15.0))
        head_yaw_deg = float(rng.uniform(-10.0, 10.0))
        head_pitch_deg = float(rng.uniform(-5.0, 5.0))
        head_euler_rad = (
            math.radians(head_yaw_deg),
            math.radians(head_pitch_deg),
            0.0,
        )

        r_ned_b = rotations.euler_to_r(
            math.radians(yaw_deg), math.radians(pitch_deg), math.radians(roll_deg)
        )

        # Sample a passive track bearing inside the viewport FOV
        total_yaw_rad = math.radians(yaw_deg + head_yaw_deg)
        total_pitch_rad = math.radians(pitch_deg + head_pitch_deg)
        az_rad = float(total_yaw_rad + rng.uniform(-0.35, 0.35))
        el_rad = float(total_pitch_rad + rng.uniform(-0.20, 0.20))
        az_rate = float(rng.uniform(-0.015, 0.015))
        el_rate = float(rng.uniform(-0.010, 0.010))

        trk_dict = {
            "track_id": "T0001",
            "az": az_rad,
            "el": el_rad,
            "az_rate": az_rate,
            "el_rate": el_rate,
            "s": 0.005,
            "s_mom": 0.005,
            "s_dot": 0.0002,
            "wc1_corrective": True,
            "hmd_min_m": 120.0,
            "hmd_max_m": 480.0,
            "hmd_hat_m": 300.0,
            "tau_mod_min_s": 14.0,
            "tau_mod_max_s": 28.0,
            "tau_mod_hat_s": 21.0,
        }

        proj = project_passive_track_to_viewport(
            track=trk_dict,
            risk=trk_dict,
            r_ned_body_est=r_ned_b,
            head_euler_rad=head_euler_rad,
            view_cam=view_cam,
            integrity_state="VALID",
            beta_wc=0.50,
            leader_horizon_s=leader_s,
            sigma_mult=sigma_mult,
        )
        assert proj["is_in_front"] is True

        # Independent analytic pinhole projection of unit bearing vector u_ned
        u_ned, du_daz, du_del = passive_bearing_ned_and_jacobian(az_rad, el_rad)
        r_body_head = rotations.euler_to_r(*head_euler_rad)
        r_ned_view = r_ned_b @ (r_body_head @ _R_BODY_CAM_FORWARD)
        d_cam = r_ned_view.T @ u_ned
        u_exact = view_cam.cx + view_cam.fx * (float(d_cam[0]) / float(d_cam[2]))
        v_exact = view_cam.cy + view_cam.fy * (float(d_cam[1]) / float(d_cam[2]))

        err_px = math.hypot(proj["u_vp"] - u_exact, proj["v_vp"] - v_exact)
        q8_err_px = math.hypot(proj["u_q8_vp"] - u_exact, proj["v_q8_vp"] - v_exact)
        vp_errors_px.append(err_px)
        q8_errors_px.append(q8_err_px)

        # Independent Jacobian leader endpoint check
        u_dot_ned = du_daz * az_rate + du_del * el_rate
        d_dot_cam = r_ned_view.T @ u_dot_ned
        dz = float(d_cam[2])
        u_dot_ex = view_cam.fx * (float(d_dot_cam[0]) * dz - float(d_cam[0]) * float(d_dot_cam[2])) / (dz * dz)
        v_dot_ex = view_cam.fy * (float(d_dot_cam[1]) * dz - float(d_cam[1]) * float(d_dot_cam[2])) / (dz * dz)
        lead_err_px = math.hypot(
            proj["u_leader_vp"] - (u_exact + leader_s * u_dot_ex),
            proj["v_leader_vp"] - (v_exact + leader_s * v_dot_ex),
        )
        leader_errors_px.append(lead_err_px)

        # Panorama projection check (NED frame)
        pano_ret = project_passive_track_to_panorama(
            track=trk_dict,
            risk=trk_dict,
            r_ned_body_est=r_ned_b,
            width=pano_w,
            height=pano_h,
            frame="ned",
            beta_wc=0.50,
        )
        psi_ex = math.atan2(float(u_ned[1]), float(u_ned[0]))
        el_ex = math.asin(max(-1.0, min(1.0, -float(u_ned[2]))))
        u_pano_exact = (psi_ex / (2.0 * math.pi) + 0.5) * pano_w - 0.5
        v_pano_exact = (0.5 - el_ex / math.pi) * pano_h - 0.5
        pano_err_px = math.hypot(
            pano_ret["u_pan"] - u_pano_exact, pano_ret["v_pan"] - v_pano_exact
        )
        pano_errors_px.append(pano_err_px)

    # Verify exact mathematical duality: P_hat(WCV) >= beta_WC <=> wc1_corrective(beta_WC)
    duality_checks = 500
    duality_mismatches = 0
    for _ in range(duality_checks):
        h_min = float(rng.uniform(20.0, 900.0))
        h_max = float(h_min + rng.uniform(50.0, 1200.0))
        t_min = float(rng.uniform(2.0, 80.0))
        t_max = float(t_min + rng.uniform(5.0, 60.0))
        beta_val = float(rng.choice([0.25, 0.50, 0.75]))

        h_hat = (1.0 - beta_val) * h_min + beta_val * h_max
        t_hat = (1.0 - beta_val) * t_min + beta_val * t_max
        wc1_corr = bool(h_hat < LOWC_HMD_LIMIT_M and t_hat <= 55.0)
        p_wcv = compute_p_wcv_belief(
            {
                "hmd_min_m": h_min,
                "hmd_max_m": h_max,
                "tau_mod_min_s": t_min,
                "tau_mod_max_s": t_max,
            },
            d_thr_m=LOWC_HMD_LIMIT_M,
            t_corr_s=55.0,
        )
        if bool(p_wcv >= beta_val - 1e-12) != wc1_corr:
            duality_mismatches += 1

    # Verify the 4 canonical symbology tiers
    tier_samples = {
        "NONE": classify_traffic_symbology_tier(
            {"wc1_warning": False, "wc1_corrective": False, "hmd_min_m": 900.0, "hmd_max_m": 1500.0, "tau_mod_min_s": 85.0, "tau_mod_max_s": 110.0, "s_dot": 0.0}
        )["tier_name"],
        "PREVENTIVE": classify_traffic_symbology_tier(
            {"wc1_warning": False, "wc1_corrective": False, "hmd_min_m": 300.0, "hmd_max_m": 1200.0, "tau_mod_min_s": 25.0, "tau_mod_max_s": 70.0, "s_dot": 0.0002}
        )["tier_name"],
        "CORRECTIVE": classify_traffic_symbology_tier(
            {"wc1_warning": False, "wc1_corrective": True, "hmd_min_m": 150.0, "hmd_max_m": 550.0, "tau_mod_min_s": 18.0, "tau_mod_max_s": 40.0, "s_dot": 0.0003}
        )["tier_name"],
        "WARNING": classify_traffic_symbology_tier(
            {"wc1_warning": True, "wc1_corrective": True, "hmd_min_m": 80.0, "hmd_max_m": 320.0, "tau_mod_min_s": 8.0, "tau_mod_max_s": 20.0, "s_dot": 0.0005}
        )["tier_name"],
    }

    vp_arr = np.asarray(vp_errors_px, dtype=np.float64)
    q8_arr = np.asarray(q8_errors_px, dtype=np.float64)
    pano_arr = np.asarray(pano_errors_px, dtype=np.float64)
    lead_arr = np.asarray(leader_errors_px, dtype=np.float64)

    return {
        "samples_count": n_samples,
        "max_error_px": float(np.max(q8_arr)),
        "median_error_px": float(np.median(q8_arr)),
        "p95_error_px": float(np.percentile(q8_arr, 95)),
        "max_continuous_error_px": float(np.max(vp_arr)),
        "max_panorama_error_px": float(np.max(pano_arr)),
        "max_leader_endpoint_error_px": float(np.max(lead_arr)),
        "wcv_duality_checks_count": duality_checks,
        "wcv_duality_mismatches_count": duality_mismatches,
        "tier_classification_exact": bool(
            tier_samples["NONE"] == "NONE"
            and tier_samples["PREVENTIVE"] == "PREVENTIVE"
            and tier_samples["CORRECTIVE"] == "CORRECTIVE"
            and tier_samples["WARNING"] == "WARNING"
        ),
        "tiers_verified_count": 4,
    }


def evaluate_v2_integrity_hud_coupling(cfg: dict[str, Any]) -> dict[str, Any]:
    """Evaluate V2_G1 conformal horizon +/- PL_0.01 band accuracy, coasting widening, and 1-frame INVALID_ALERT blanking."""
    view_cam = get_viewport_camera(cfg)
    int_cfg = cfg["integrity_hud"]
    pl0_deg = float(int_cfg["nominal_pl_001_deg"])
    pl_iso_deg = float(int_cfg["isolated_pl_001_deg"])
    drift_degps = float(int_cfg["coasting_drift_deg_per_s"])

    lat_deg, lon_deg, alt_m = 37.30, -121.00, 1500.0
    rng = np.random.default_rng(20261109)
    n_poses = 30
    pl_band_errors_px: list[float] = []

    for _ in range(n_poses):
        yaw_deg = float(rng.uniform(-45.0, 45.0))
        pitch_deg = float(rng.uniform(-10.0, 10.0))
        roll_deg = float(rng.uniform(-20.0, 20.0))
        pl_deg = float(rng.uniform(0.15, 0.45))

        r_ned_b = rotations.euler_to_r(
            math.radians(yaw_deg), math.radians(pitch_deg), math.radians(roll_deg)
        )
        vp_pose = compute_viewport_pose_from_estimated_attitude(
            ownship_lat_deg=lat_deg,
            ownship_lon_deg=lon_deg,
            ownship_h_m=alt_m,
            r_ned_body_est=r_ned_b,
            head_euler_rad=(0.0, 0.0, 0.0),
            config=cfg,
        )

        hb = compute_conformal_horizon_with_pl_band(
            p_ecef_view=vp_pose["p_ecef_view"],
            r_ecef_view_est=vp_pose["r_ecef_view_est"],
            view_cam=view_cam,
            pl_deg=pl_deg,
            integrity_state="VALID",
            f_isolated=True,
            isolated_camera_id="F",
            num_samples=512,
        )
        assert hb["horizon_visible"] is True

        # Independent analytic check of +/- PL_0.01 band points against q8 coordinates
        hz_gt = compute_horizon_gt(
            vp_pose["p_ecef_view"], vp_pose["r_ecef_view_est"], view_cam, num_samples=512
        )
        hz_uv = np.asarray(hz_gt["horizon_uv"], dtype=np.float64)
        alpha_v = np.arctan((hz_uv[:, 1] - view_cam.cy) / view_cam.fy)
        pl_rad = math.radians(pl_deg)
        v_up_ex = view_cam.cy + view_cam.fy * np.tan(alpha_v - pl_rad)
        pts_up_ex = np.column_stack([hz_uv[:, 0], v_up_ex])
        err_up = float(np.max(np.linalg.norm(hb["upper_pl_q8_uv"] - pts_up_ex, axis=1)))
        pl_band_errors_px.append(err_up)

    # Check DEGRADED_COAST PL widening
    pl_coast_1s = compute_coasting_pl_deg(
        pl_base_deg=pl0_deg, coast_time_s=1.0, drift_rate_degps=drift_degps
    )
    pl_coast_5s = compute_coasting_pl_deg(
        pl_base_deg=pl0_deg, coast_time_s=5.0, drift_rate_degps=drift_degps
    )

    # Evaluate 30-frame integrity state sequence (`VALID` -> `DEGRADED_COAST` -> `INVALID_ALERT`)
    seq_states = ["VALID"] * 20 + ["DEGRADED_COAST"] * 5 + ["INVALID_ALERT"] * 5
    seq_errors = [0.08] * 20 + [0.32] * 5 + [1.25] * 5
    blank_metrics = evaluate_integrity_display_blanking_latency(
        integrity_state_trace=seq_states,
        att_error_deg_trace=seq_errors,
        alert_limit_deg=float(int_cfg["alert_limit_deg"]),
        pl_base_deg=pl0_deg,
        dt_s=0.1,
        drift_rate_degps=drift_degps,
    )

    return {
        "poses_evaluated": n_poses,
        "max_pl_band_error_px": float(max(pl_band_errors_px)),
        "nominal_pl_4cam_deg": pl0_deg,
        "nominal_pl_3cam_isolated_deg": pl_iso_deg,
        "coasting_pl_1s_deg": float(pl_coast_1s),
        "coasting_pl_5s_deg": float(pl_coast_5s),
        "coasting_pl_monotonic": bool(pl_coast_5s > pl_coast_1s > pl0_deg),
        "ttd_display_frames": int(blank_metrics["ttd_display_frames"]),
        "ttd_display_s": float(blank_metrics["ttd_display_frames"] * 0.1),
        "hud_hmi_frames": int(blank_metrics["hud_hmi_frames"]),
        "invalid_alert_epochs": int(blank_metrics["invalid_alert_frames"]),
        "blanked_within_one_frame": bool(blank_metrics["ttd_display_frames"] == 1),
    }


def evaluate_v5_truth_isolation(cfg: dict[str, Any]) -> dict[str, Any]:
    """Static AST + runtime sentinel audit on `src/xvs/display/symbology.py` and `src/xvs/avionics/sitl_closed_loop.py`."""
    audit_files = (
        REPO_ROOT / "src" / "xvs" / "display" / "symbology.py",
        REPO_ROOT / "src" / "xvs" / "avionics" / "sitl_closed_loop.py",
    )
    forbidden_imports_count = 0
    forbidden_attr_refs_count = 0
    violations: list[str] = []

    for fpath in audit_files:
        src_text = fpath.read_text(encoding="utf-8")
        tree = ast.parse(src_text, filename=str(fpath))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "xvs.traffic" in alias.name or alias.name.endswith(".truth"):
                        forbidden_imports_count += 1
                        violations.append(f"{fpath.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if "xvs.traffic" in mod or mod.endswith(".truth"):
                    forbidden_imports_count += 1
                    violations.append(f"{fpath.name}: from {mod} import ...")
            elif isinstance(node, ast.Attribute):
                if node.attr in FORBIDDEN_TRUTH_IDENTIFIERS:
                    forbidden_attr_refs_count += 1
                    violations.append(f"{fpath.name}:{node.lineno}: attr .{node.attr}")

    # Runtime sentinel trap check
    runtime_violations = 0
    sentinel_trk = PoisonedTruthTrack()
    sentinel_risk = RiskEstimate(
        track_id="T0099",
        t_s=10.0,
        s_rad=0.006,
        s_dot_radps=0.0004,
        sigma_s_dot=1e-5,
        tau_s=15.0,
        omega_los_radps=0.0022,
        range_interval_m=(1200.0, 2200.0),
        hmd_interval_m=(80.0, 240.0),
        alert_a1_warning=True,
        alert_a1_corrective=True,
        alert_a2_warning=True,
        alert_a2_corrective=True,
        wc1_warning=True,
        wc1_corrective=True,
        hmd_min_m=80.0,
        hmd_max_m=240.0,
        hmd_hat_m=160.0,
        tau_mod_min_s=11.0,
        tau_mod_max_s=19.0,
        tau_mod_hat_s=15.0,
        tau_s_hat_s=15.0,
        beta_wc=0.50,
    )
    view_cam = get_viewport_camera(cfg)
    r_ned_b = rotations.euler_to_r(0.0, 0.0, 0.0)
    vp_pose = compute_viewport_pose_from_estimated_attitude(
        ownship_lat_deg=37.30,
        ownship_lon_deg=-121.00,
        ownship_h_m=1500.0,
        r_ned_body_est=r_ned_b,
        head_euler_rad=(0.0, 0.0, 0.0),
        config=cfg,
    )
    canvas = np.zeros((view_cam.height, view_cam.width, 3), dtype=np.uint8)
    try:
        _ = project_passive_track_to_viewport(
            track=sentinel_trk,
            risk=sentinel_risk,
            r_ned_body_est=r_ned_b,
            head_euler_rad=(0.0, 0.0, 0.0),
            view_cam=view_cam,
        )
        _ = draw_g1_symbology(
            img=canvas,
            aircraft_pose_est={"lat_deg": 37.30, "lon_deg": -121.00, "h_m": 1500.0, "yaw_rad": 0.0, "pitch_rad": 0.0, "roll_rad": 0.0},
            v_ned=np.array([100.0, 0.0, 0.0], dtype=np.float64),
            p_ecef_view=vp_pose["p_ecef_view"],
            r_ecef_view_est=vp_pose["r_ecef_view_est"],
            r_ned_body_est=r_ned_b,
            head_euler_rad=(0.0, 0.0, 0.0),
            view_cam=view_cam,
            tracks_and_risks=[(sentinel_trk, sentinel_risk)],
            integrity_state="VALID",
        )
        _ = PassiveTrackInput.from_track_and_risk(sentinel_trk, sentinel_risk, class_id="C2_SEP")
    except RuntimeError:
        runtime_violations += 1

    return {
        "files_audited": len(audit_files),
        "forbidden_imports_count": forbidden_imports_count,
        "forbidden_attribute_refs_count": forbidden_attr_refs_count,
        "runtime_sentinel_violations": runtime_violations,
        "total_violations": forbidden_imports_count + forbidden_attr_refs_count + runtime_violations,
        "isolated": (
            forbidden_imports_count == 0
            and forbidden_attr_refs_count == 0
            and runtime_violations == 0
        ),
        "violations": violations,
    }


_FALLBACK_G4_ROOTS = (
    REPO_ROOT / "output" / "xvs" / "wp01" / "g4",
)


def _resolve_g4_seq_dir(seq_name: str, data_root: Path | None = None) -> Path:
    """Resolve WP-01 G4 sequence directory (`T1_cruise`, `T2_ksfo28l`, `T3_turn_climb`)."""
    cand0 = get_data_root(data_root) / "xvs" / "wp01" / "g4" / seq_name
    if (cand0 / "frames.jsonl").is_file():
        return cand0
    for root in _FALLBACK_G4_ROOTS:
        cand = root / seq_name
        if (cand / "frames.jsonl").is_file():
            return cand
    raise FileNotFoundError(
        f"WP-01 G4 sequence {seq_name!r} not found under {cand0} or {_FALLBACK_G4_ROOTS}"
    )


def _load_g4_mid_frame(seq_name: str, data_root: Path | None = None) -> dict[str, Any]:
    """Load the middle frame of a real WP-01 G4 6-camera CesiumJS sequence (`F, R, B, L, U, D`)."""
    seq_dir = _resolve_g4_seq_dir(seq_name, data_root=data_root)
    lines = [
        ln.strip()
        for ln in (seq_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    mid_idx = len(lines) // 2
    rec = json.loads(lines[mid_idx])

    cam_images_rgb: dict[str, np.ndarray] = {}
    for cid, cam_info in rec["cams"].items():
        img_path = seq_dir / cam_info["png"]
        with Image.open(img_path) as im:
            cam_images_rgb[cid] = np.asarray(im.convert("RGB"), dtype=np.uint8)

    ac = rec["aircraft"]
    aircraft_pose = {
        "lat_deg": float(ac["lat_deg"]),
        "lon_deg": float(ac["lon_deg"]),
        "h_m": float(ac["h_m"]),
        "yaw_rad": math.radians(float(ac["ypr_deg"][0])),
        "pitch_rad": math.radians(float(ac["ypr_deg"][1])),
        "roll_rad": math.radians(float(ac["ypr_deg"][2])),
    }
    v_ned = np.asarray(ac["v_ned_mps"], dtype=np.float64)
    return {
        "seq_name": seq_name,
        "seq_dir": str(seq_dir.as_posix()),
        "frame_index": int(rec.get("frame", mid_idx)),
        "t_s": float(rec.get("t_s", 0.0)),  # default-ok: optional record/config fallback
        "camera_images_rgb": cam_images_rgb,
        "aircraft_pose": aircraft_pose,
        "v_ned": v_ned,
    }


def _compute_max_hud_bbox_overlap_px2(telemetry: dict[str, Any]) -> float:
    """Compute maximum pairwise intersection area (px^2) across all HUD label and reticle bounding boxes."""
    tracks = telemetry.get("projected_tracks", [])
    max_ov = 0.0
    n_trks = len(tracks)
    for i in range(n_trks):
        lbl_i = tuple(tracks[i]["label_bbox"])
        ret_i = tuple(tracks[i]["reticle_bbox"])
        max_ov = max(max_ov, compute_rect_intersection_area(lbl_i, ret_i))
        for j in range(i + 1, n_trks):
            lbl_j = tuple(tracks[j]["label_bbox"])
            ret_j = tuple(tracks[j]["reticle_bbox"])
            max_ov = max(
                max_ov,
                compute_rect_intersection_area(lbl_i, lbl_j),
                compute_rect_intersection_area(lbl_i, ret_j),
                compute_rect_intersection_area(ret_i, lbl_j),
            )
    return float(max_ov)


def _build_demo_four_tier_tracks(yaw_center_rad: float) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Construct 4 representative (track, risk) pairs spanning all 4 symbology tiers inside the viewport."""
    items = [
        {
            "track_id": "T0001",
            "az": yaw_center_rad - 0.45,
            "el": 0.08,
            "az_rate": 0.001,
            "el_rate": 0.0,
            "s": 0.003,
            "s_mom": 0.003,
            "s_dot": 0.0,
            "wc1_warning": False,
            "wc1_corrective": False,
            "hmd_min_m": 820.0,
            "hmd_max_m": 1450.0,
            "tau_mod_min_s": 85.0,
            "tau_mod_max_s": 110.0,
        },
        {
            "track_id": "T0002",
            "az": yaw_center_rad - 0.16,
            "el": 0.05,
            "az_rate": 0.0045,
            "el_rate": -0.0015,
            "s": 0.0045,
            "s_mom": 0.0045,
            "s_dot": 0.0002,
            "wc1_warning": False,
            "wc1_corrective": False,
            "hmd_min_m": 340.0,
            "hmd_max_m": 980.0,
            "tau_mod_min_s": 28.0,
            "tau_mod_max_s": 64.0,
        },
        {
            "track_id": "T0003",
            "az": yaw_center_rad + 0.12,
            "el": -0.02,
            "az_rate": -0.0025,
            "el_rate": 0.0012,
            "s": 0.0065,
            "s_mom": 0.0065,
            "s_dot": 0.00035,
            "wc1_warning": False,
            "wc1_corrective": True,
            "hmd_min_m": 140.0,
            "hmd_max_m": 490.0,
            "tau_mod_min_s": 19.0,
            "tau_mod_max_s": 37.0,
        },
        {
            "track_id": "T0004",
            "az": yaw_center_rad + 0.38,
            "el": 0.02,
            "az_rate": -0.0008,
            "el_rate": -0.0004,
            "s": 0.0095,
            "s_mom": 0.0095,
            "s_dot": 0.0006,
            "wc1_warning": True,
            "wc1_corrective": True,
            "hmd_min_m": 45.0,
            "hmd_max_m": 210.0,
            "tau_mod_min_s": 8.5,
            "tau_mod_max_s": 16.5,
        },
    ]
    return [(d, d) for d in items]


def _render_sitl_trajectory_plot(rep_trace: dict[str, Any]) -> Image.Image:
    """Render a 1280x720 top-down East-North plot comparing Open-Loop vs G0 A1 vs G1 WC1 trajectories."""
    w, h = 1280, 720
    img = Image.new("RGB", (w, h), (16, 22, 32))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    margin_l, margin_r, margin_t, margin_b = 90, 60, 80, 70
    pw = w - margin_l - margin_r
    ph = h - margin_t - margin_b
    e_min, e_max = -200.0, 6200.0

    # Auto-scale North axis symmetrically around nominal Eastbound airway (0m N) to include all traces
    all_north_vals: list[float] = [0.0]
    for m_data in rep_trace["modes"].values():
        for _, n_val in m_data["own_trace_en_m"]:
            all_north_vals.append(float(n_val))
    for _, n_val in rep_trace["modes"]["open_loop"]["tgt_trace_en_m"]:
        if -500.0 <= n_val <= 1500.0:
            all_north_vals.append(float(n_val))
    max_abs_n = max(800.0, max(abs(v) for v in all_north_vals) + 150.0)
    n_bound = float(int(math.ceil(max_abs_n / 200.0)) * 200)
    n_min, n_max = -n_bound, n_bound

    def to_px(east_m: float, north_m: float) -> tuple[int, int]:
        u = margin_l + int(round(((east_m - e_min) / (e_max - e_min)) * pw))
        v = margin_t + int(round(((n_max - north_m) / (n_max - n_min)) * ph))
        return u, v

    for e_grid in range(0, 6001, 1000):
        u0, v0 = to_px(float(e_grid), n_min)
        u1, v1 = to_px(float(e_grid), n_max)
        draw.line([(u0, v0), (u1, v1)], fill=(38, 48, 65), width=1)
        draw.text((u0 - 18, h - margin_b + 10), f"{e_grid}m E", fill=(160, 175, 195), font=font)

    grid_step = 200 if n_bound <= 1000.0 else 400
    for n_grid in range(-int(n_bound) + grid_step, int(n_bound), grid_step):
        u0, v0 = to_px(e_min, float(n_grid))
        u1, v1 = to_px(e_max, float(n_grid))
        col = (85, 110, 145) if n_grid == 0 else (38, 48, 65)
        draw.line([(u0, v0), (u1, v1)], fill=col, width=(2 if n_grid == 0 else 1))
        draw.text((15, v0 - 6), f"{n_grid:+d}m N", fill=(160, 175, 195), font=font)

    tgt_pts = [to_px(p[0], p[1]) for p in rep_trace["modes"]["open_loop"]["tgt_trace_en_m"]]
    if len(tgt_pts) >= 2:
        draw.line(tgt_pts, fill=(255, 95, 95), width=2)
        draw.ellipse(
            [tgt_pts[0][0] - 4, tgt_pts[0][1] - 4, tgt_pts[0][0] + 4, tgt_pts[0][1] + 4],
            outline=(255, 95, 95),
            width=2,
        )

    mode_styles = [
        ("open_loop", (170, 170, 170), 5, "Open-Loop (No Avoidance)"),
        ("closed_loop_g0_a1", (255, 185, 50), 2, "Closed-Loop G0 A1"),
        ("closed_loop_g1_wc1_b050", (60, 235, 140), 6, "Closed-Loop G1 WC1 (beta_WC=0.50)"),
        ("closed_loop_g1_wc1_b025", (80, 195, 255), 2, "Closed-Loop G1 WC1 (beta_WC=0.25)"),
    ]

    leg_y = margin_t + 14
    for m_key, col, line_w, label in mode_styles:
        m_data = rep_trace["modes"][m_key]
        pts = [to_px(p[0], p[1]) for p in m_data["own_trace_en_m"]]
        if len(pts) >= 2:
            draw.line(pts, fill=col, width=line_w)
        hmd_val = m_data["achieved_min_hmd_m"]
        xtk_val = m_data["max_cross_track_m"]
        fin_xtk_val = m_data.get("final_cross_track_m", xtk_val)
        draw.rectangle([margin_l + 15, leg_y, margin_l + 35, leg_y + 10], fill=col)
        draw.text(
            (margin_l + 44, leg_y - 2),
            f"{label}: min HMD = {hmd_val:.1f} m | max XTK = {xtk_val:.1f} m | final XTK = {fin_xtk_val:.1f} m",
            fill=(235, 240, 248),
            font=font,
        )
        leg_y += 20

    draw.rectangle([margin_l + 15, leg_y, margin_l + 35, leg_y + 10], fill=(255, 95, 95))
    draw.text(
        (margin_l + 44, leg_y - 2),
        f"Intruder Trajectory ({rep_trace['scenario_id']}, {rep_trace['geometry_stratum']}, {rep_trace['class_id']}) -- Stern-Pass & Airway Recovery",
        fill=(255, 150, 150),
        font=font,
    )

    draw.text(
        (margin_l, 24),
        "WP-08 Gate G1 Closed-Loop MAVLink #246 -> ArduPilot SITL Horizontal Avoidance (Top-Down East-North)",
        fill=(255, 255, 255),
        font=font,
    )
    draw.text(
        (margin_l, 44),
        f"Encounter {rep_trace['scenario_id']} | NMAC Threshold = 152.4 m (500 ft) | LoWC D_THR = 609.6 m (2000 ft)",
        fill=(190, 210, 235),
        font=font,
    )
    return img


_G4_MID_FRAME_CACHE: dict[str, dict[str, Any]] = {}


def _get_cached_g4_mid_frame(seq_name: str) -> dict[str, Any]:
    if seq_name not in _G4_MID_FRAME_CACHE:
        _G4_MID_FRAME_CACHE[seq_name] = _load_g4_mid_frame(seq_name)
    return _G4_MID_FRAME_CACHE[seq_name]


def render_g1_stills_and_preview(
    cfg: dict[str, Any],
    rep_trace: dict[str, Any],
    stills_dir: Path | None = None,
) -> dict[str, Any]:
    """Render deterministic G1 HUD stills fused from real WP-01 G4 (`T1_cruise`, `T2_ksfo28l`,
    `T3_turn_climb`) 6-camera CesiumJS frames via `src/xvs/display/fusion.py` (`ellipsoid` depth model)
    and `src/xvs/display/viewport.py` (`B12-F1`), plus NED panorama overlay, SITL trajectory plot,
    and WebP preview.
    """
    view_cam = get_viewport_camera(cfg)
    int_cfg = cfg["integrity_hud"]
    rig = Rig.load_nominal().make_true(seed=int(cfg.get("rig_seed", 20260922)))  # default-ok: optional record/config fallback
    head_euler = (0.0, 0.0, 0.0)

    t1_frame = _get_cached_g4_mid_frame("T1_cruise")
    t2_frame = _get_cached_g4_mid_frame("T2_ksfo28l")
    t3_frame = _get_cached_g4_mid_frame("T3_turn_climb")

    # 1. VALID + F_isolated=True (5-camera see-through fusion on real T2_ksfo28l with Camera R isolated)
    t2_pose = t2_frame["aircraft_pose"]
    r_ned_b_t2 = rotations.euler_to_r(t2_pose["yaw_rad"], t2_pose["pitch_rad"], t2_pose["roll_rad"])
    vp_pose_t2 = compute_viewport_pose_from_estimated_attitude(
        ownship_lat_deg=t2_pose["lat_deg"],
        ownship_lon_deg=t2_pose["lon_deg"],
        ownship_h_m=t2_pose["h_m"],
        r_ned_body_est=r_ned_b_t2,
        head_euler_rad=head_euler,
        config=cfg,
    )
    vp_t2_iso_rgb = extract_viewport(
        camera_images=t2_frame["camera_images_rgb"],
        rig=rig,
        aircraft_pose=t2_pose,
        head_euler=head_euler,
        depth_model="ellipsoid",
        config=cfg,
        excluded_cameras=("R",),
    )
    vp_t2_iso_bgr = vp_t2_iso_rgb[:, :, ::-1].copy()
    demo_tracks_t2 = _build_demo_four_tier_tracks(t2_pose["yaw_rad"])

    hud_valid_bgr, tel_valid = draw_g1_symbology(
        img=vp_t2_iso_bgr,
        aircraft_pose_est=t2_pose,
        v_ned=t2_frame["v_ned"],
        p_ecef_view=vp_pose_t2["p_ecef_view"],
        r_ecef_view_est=vp_pose_t2["r_ecef_view_est"],
        r_ned_body_est=r_ned_b_t2,
        head_euler_rad=head_euler,
        view_cam=view_cam,
        tracks_and_risks=demo_tracks_t2,
        integrity_state="VALID",
        pl_deg=float(int_cfg["isolated_pl_001_deg"]),
        f_isolated=True,
        isolated_camera_id="R",
        coast_time_s=0.0,
        coast_drift_deg_per_s=float(int_cfg["coasting_drift_deg_per_s"]),
        beta_wc=0.50,
        burn_in_text="WP-08 G1 | T2_ksfo28l (5-CAM FUSED, CAM R ISOLATED) | 4 DO-365B TRAFFIC TIERS",
    )
    img_valid = Image.fromarray(hud_valid_bgr[:, :, ::-1])

    # 2. DEGRADED_COAST (6-camera see-through fusion on real T1_cruise, coasting 3.5 s, widened PL band)
    t1_pose = t1_frame["aircraft_pose"]
    r_ned_b_t1 = rotations.euler_to_r(t1_pose["yaw_rad"], t1_pose["pitch_rad"], t1_pose["roll_rad"])
    vp_pose_t1 = compute_viewport_pose_from_estimated_attitude(
        ownship_lat_deg=t1_pose["lat_deg"],
        ownship_lon_deg=t1_pose["lon_deg"],
        ownship_h_m=t1_pose["h_m"],
        r_ned_body_est=r_ned_b_t1,
        head_euler_rad=head_euler,
        config=cfg,
    )
    vp_t1_rgb = extract_viewport(
        camera_images=t1_frame["camera_images_rgb"],
        rig=rig,
        aircraft_pose=t1_pose,
        head_euler=head_euler,
        depth_model="ellipsoid",
        config=cfg,
    )
    vp_t1_bgr = vp_t1_rgb[:, :, ::-1].copy()
    demo_tracks_t1 = _build_demo_four_tier_tracks(t1_pose["yaw_rad"])

    hud_coast_bgr, tel_coast = draw_g1_symbology(
        img=vp_t1_bgr,
        aircraft_pose_est=t1_pose,
        v_ned=t1_frame["v_ned"],
        p_ecef_view=vp_pose_t1["p_ecef_view"],
        r_ecef_view_est=vp_pose_t1["r_ecef_view_est"],
        r_ned_body_est=r_ned_b_t1,
        head_euler_rad=head_euler,
        view_cam=view_cam,
        tracks_and_risks=demo_tracks_t1,
        integrity_state="DEGRADED_COAST",
        pl_deg=float(int_cfg["nominal_pl_001_deg"]),
        f_isolated=False,
        coast_time_s=3.5,
        coast_drift_deg_per_s=float(int_cfg["coasting_drift_deg_per_s"]),
        beta_wc=0.50,
        burn_in_text="WP-08 G1 | T1_cruise (6-CAM FUSED) | STATE: DEGRADED_COAST (dt=3.5s) [BRG ONLY]",
    )
    img_coast = Image.fromarray(hud_coast_bgr[:, :, ::-1])

    # 3. INVALID_ALERT (6-camera see-through fusion on real T3_turn_climb roll=25 deg, blanked conformal horizon)
    t3_pose = t3_frame["aircraft_pose"]
    r_ned_b_t3 = rotations.euler_to_r(t3_pose["yaw_rad"], t3_pose["pitch_rad"], t3_pose["roll_rad"])
    vp_pose_t3 = compute_viewport_pose_from_estimated_attitude(
        ownship_lat_deg=t3_pose["lat_deg"],
        ownship_lon_deg=t3_pose["lon_deg"],
        ownship_h_m=t3_pose["h_m"],
        r_ned_body_est=r_ned_b_t3,
        head_euler_rad=head_euler,
        config=cfg,
    )
    vp_t3_rgb = extract_viewport(
        camera_images=t3_frame["camera_images_rgb"],
        rig=rig,
        aircraft_pose=t3_pose,
        head_euler=head_euler,
        depth_model="ellipsoid",
        config=cfg,
    )
    vp_t3_bgr = vp_t3_rgb[:, :, ::-1].copy()
    demo_tracks_t3 = _build_demo_four_tier_tracks(t3_pose["yaw_rad"])

    hud_inv_bgr, tel_inv = draw_g1_symbology(
        img=vp_t3_bgr,
        aircraft_pose_est=t3_pose,
        v_ned=t3_frame["v_ned"],
        p_ecef_view=vp_pose_t3["p_ecef_view"],
        r_ecef_view_est=vp_pose_t3["r_ecef_view_est"],
        r_ned_body_est=r_ned_b_t3,
        head_euler_rad=head_euler,
        view_cam=view_cam,
        tracks_and_risks=demo_tracks_t3,
        integrity_state="INVALID_ALERT",
        pl_deg=float(int_cfg["nominal_pl_001_deg"]),
        f_isolated=False,
        coast_time_s=6.0,
        coast_drift_deg_per_s=float(int_cfg["coasting_drift_deg_per_s"]),
        beta_wc=0.50,
        burn_in_text="WP-08 G1 | T3_turn_climb (6-CAM FUSED) | STATE: INVALID_ALERT (TTD=1 FRAME, 0 HMI)",
    )
    img_inv = Image.fromarray(hud_inv_bgr[:, :, ::-1])

    # 4. 2048x1024 NED Panorama fused from real T2_ksfo28l 6-camera CesiumJS imagery + G1 traffic overlay
    pano_w = int(cfg["panorama"]["width"])
    pano_h = int(cfg["panorama"]["height"])
    pano_rgb, pano_stats = render_display_panorama(
        camera_images=t2_frame["camera_images_rgb"],
        rig=rig,
        aircraft_pose=t2_pose,
        depth_model="ellipsoid",
        frame="ned",
        width=pano_w,
        height=pano_h,
        config=cfg,
    )
    pano_bgr = pano_rgb[:, :, ::-1].copy()
    pano_out_bgr, pano_projs = draw_panorama_g1_symbology(
        panorama_img=pano_bgr,
        r_ned_body_est=r_ned_b_t2,
        tracks_and_risks=demo_tracks_t2,
        frame="ned",
        beta_wc=0.50,
        burn_in_text="WP-08 G1 | T2_ksfo28l 2048x1024 6-CAM FUSED NED PANORAMA | PASSIVE WC1 TRAFFIC OVERLAY",
    )
    img_pano = Image.fromarray(pano_out_bgr[:, :, ::-1])

    # 5. SITL closed-loop trajectory comparison plot
    img_sitl = _render_sitl_trajectory_plot(rep_trace)

    # Verify zero callout/reticle bounding-box overlap across all rendered HUD stills (B12-F1)
    max_callout_overlap_px2 = max(
        _compute_max_hud_bbox_overlap_px2(tel_valid),
        _compute_max_hud_bbox_overlap_px2(tel_coast),
        _compute_max_hud_bbox_overlap_px2(tel_inv),
        _compute_max_hud_bbox_overlap_px2({"projected_tracks": pano_projs}),
    )

    images_map = {
        "g1_hud_valid_isolated_traffic_tiers.png": img_valid,
        "g1_hud_degraded_coast.png": img_coast,
        "g1_hud_invalid_alert_blanked.png": img_inv,
        "g1_panorama_ned_traffic.png": img_pano,
        "g1_sitl_closed_loop_trajectories.png": img_sitl,
    }

    png_hashes: dict[str, str] = {}
    if stills_dir is not None:
        stills_dir.mkdir(parents=True, exist_ok=True)
        for fname, im in images_map.items():
            out_p = stills_dir / fname
            im.save(out_p, format="PNG")
            png_hashes[fname] = _sha256_file(out_p)

        webp_path = stills_dir / "g1_hud_sequence_preview.webp"
        preview_frames = [
            im.resize((640, 360), Image.Resampling.BILINEAR)
            for im in (img_valid, img_coast, img_inv)
        ]
        preview_frames[0].save(
            webp_path,
            format="WEBP",
            save_all=True,
            append_images=preview_frames[1:],
            duration=800,
            loop=0,
        )
        png_hashes["g1_hud_sequence_preview.webp"] = _sha256_file(webp_path)
    else:
        for fname, im in images_map.items():
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            png_hashes[fname] = _sha256_bytes(buf.getvalue())

    return {
        "stills_count": len(images_map),
        "png_hashes": png_hashes,
        "max_callout_overlap_px2": float(max_callout_overlap_px2),
        "g4_sequences_fused": ["T1_cruise", "T2_ksfo28l", "T3_turn_climb"],
        "t2_panorama_valid_coverage": float(
            1.0 - float(pano_stats["coverage"]["unseen_fraction"])
        ),
    }


def evaluate_v6_determinism(
    cfg: dict[str, Any],
    sitl_cfg: SitlAvoidanceConfig,
) -> dict[str, Any]:
    """Verify byte-identical JSON and PNG hashes across two repeated single-process runs."""
    v1_a = evaluate_v1_conformal_traffic_reticle(cfg)
    v2_a = evaluate_v2_integrity_hud_coupling(cfg)
    sitl_a = run_closed_loop_sitl_benchmark(config=sitl_cfg, max_encounters=1)
    stills_a = render_g1_stills_and_preview(cfg, sitl_a["representative_encounter_trace"], stills_dir=None)

    v1_b = evaluate_v1_conformal_traffic_reticle(cfg)
    v2_b = evaluate_v2_integrity_hud_coupling(cfg)
    sitl_b = run_closed_loop_sitl_benchmark(config=sitl_cfg, max_encounters=1)
    stills_b = render_g1_stills_and_preview(cfg, sitl_b["representative_encounter_trace"], stills_dir=None)

    json_a = json.dumps({"v1": v1_a, "v2": v2_a, "sitl": sitl_a}, sort_keys=True)
    json_b = json.dumps({"v1": v1_b, "v2": v2_b, "sitl": sitl_b}, sort_keys=True)
    json_identical = (json_a == json_b)
    png_identical = (stills_a["png_hashes"] == stills_b["png_hashes"])

    return {
        "workers": 1,
        "repeated_runs": 2,
        "json_byte_identical": bool(json_identical),
        "png_hashes_identical": bool(png_identical),
        "json_sha256": _sha256_bytes(json_a.encode("utf-8")),
    }


def run_display_g1_pipeline(
    out_dir: Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
    spec_path: Path = DEFAULT_SPEC_PATH,
    max_encounters: int | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    commit = _get_git_commit()
    git_status_start = _get_git_status_porcelain()
    code_hashes_start = _compute_code_hashes()

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    sitl_cfg = SitlAvoidanceConfig.from_display_config(config_path)

    v1_res = evaluate_v1_conformal_traffic_reticle(cfg)
    v2_res = evaluate_v2_integrity_hud_coupling(cfg)
    sitl_res = run_closed_loop_sitl_benchmark(config=sitl_cfg, max_encounters=max_encounters)
    v5_res = evaluate_v5_truth_isolation(cfg)
    v6_res = evaluate_v6_determinism(cfg, sitl_cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    stills_dir = out_dir / "stills"
    stills_res = render_g1_stills_and_preview(
        cfg=cfg,
        rep_trace=sitl_res["representative_encounter_trace"],
        stills_dir=stills_dir,
    )
    v1_res["max_callout_overlap_px2"] = float(stills_res["max_callout_overlap_px2"])

    modes = sitl_res["modes"]
    g0_a1 = modes["closed_loop_g0_a1"]
    g1_b050 = modes["closed_loop_g1_wc1_b050"]
    g1_b025 = modes["closed_loop_g1_wc1_b025"]
    open_l = modes["open_loop"]
    rep_modes = sitl_res["representative_encounter_trace"]["modes"]

    v3_summary = {
        "benchmark_threat_count": sitl_res["benchmark_threat_count"],
        "max_turn_rate_degps": sitl_res["max_turn_rate_degps"],
        "max_bank_angle_deg": sitl_res["max_bank_angle_deg"],
        "reaction_delay_s": sitl_res["reaction_delay_s"],
        "nmac_threshold_m": sitl_res["nmac_threshold_m"],
        "lowc_threshold_m": sitl_res["lowc_threshold_m"],
        "modes": modes,
        "g1_b050_vs_g0_a1_nmac_resolved_gain_pct": float(
            g1_b050["nmac_resolution_rate_all_pct"] - g0_a1["nmac_resolution_rate_all_pct"]
        ),
        "g1_b025_vs_g0_a1_nmac_resolved_gain_pct": float(
            g1_b025["nmac_resolution_rate_all_pct"] - g0_a1["nmac_resolution_rate_all_pct"]
        ),
        "g1_b050_vs_open_loop_hmd_median_gain_m": float(
            g1_b050["hmd_median_m"] - open_l["hmd_median_m"]
        ),
        "g1_b025_vs_open_loop_hmd_median_gain_m": float(
            g1_b025["hmd_median_m"] - open_l["hmd_median_m"]
        ),
        "representative_encounter_trace": {
            "scenario_id": sitl_res["representative_encounter_trace"]["scenario_id"],
            "geometry_stratum": sitl_res["representative_encounter_trace"]["geometry_stratum"],
            "class_id": sitl_res["representative_encounter_trace"]["class_id"],
            "open_loop_hmd_m": sitl_res["representative_encounter_trace"]["open_loop_hmd_m"],
            "g0_a1_min_hmd_m": rep_modes["closed_loop_g0_a1"]["achieved_min_hmd_m"],
            "g0_a1_max_xtk_m": rep_modes["closed_loop_g0_a1"]["max_cross_track_m"],
            "g0_a1_final_xtk_m": rep_modes["closed_loop_g0_a1"]["final_cross_track_m"],
            "g1_wc1_b050_min_hmd_m": rep_modes["closed_loop_g1_wc1_b050"]["achieved_min_hmd_m"],
            "g1_wc1_b050_max_xtk_m": rep_modes["closed_loop_g1_wc1_b050"]["max_cross_track_m"],
            "g1_wc1_b050_final_xtk_m": rep_modes["closed_loop_g1_wc1_b050"]["final_cross_track_m"],
            "g1_wc1_b025_min_hmd_m": rep_modes["closed_loop_g1_wc1_b025"]["achieved_min_hmd_m"],
            "g1_wc1_b025_max_xtk_m": rep_modes["closed_loop_g1_wc1_b025"]["max_cross_track_m"],
            "g1_wc1_b025_final_xtk_m": rep_modes["closed_loop_g1_wc1_b025"]["final_cross_track_m"],
        },
        "per_encounter": sitl_res["per_encounter"],
    }

    v4_summary = sitl_res["invalid_alert_injection"]

    elapsed_s = time.perf_counter() - t0
    code_hashes_end = _compute_code_hashes()
    git_status_end = _get_git_status_porcelain()

    inputs_sha256 = {
        "configs/xvs/display/g1.json": _sha256_file(config_path),
        "configs/xvs/display/g1_spec.json": _sha256_file(spec_path),
    }
    if DEFAULT_TIER0_JSON_PATH.is_file():
        inputs_sha256["output/xvs/wp08/tier0_b10/tier0_verification.json"] = _sha256_file(
            DEFAULT_TIER0_JSON_PATH
        )

    results_payload: dict[str, Any] = {
        "schema": "xvs.display.g1_results/1",
        "work_package": "WP-08",
        "gate": "G1",
        "batch_id": "B12",
        "commit": commit,
        "git_status": git_status_start,
        "argv": [str(Path(sys.argv[0]).as_posix())] + sys.argv[1:],
        "workers": 1,
        "elapsed_wall_s": elapsed_s,
        "code_hashes": code_hashes_start,
        "inputs_sha256": inputs_sha256,
        "v1_conformal_traffic_reticle": v1_res,
        "v2_integrity_hud_coupling": v2_res,
        "v3_closed_loop_sitl_avoidance": v3_summary,
        "v4_invalid_alert_inhibition": v4_summary,
        "v5_truth_isolation": v5_res,
        "v6_determinism": v6_res,
        "stills": stills_res,
    }

    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results_payload, indent=2) + "\n", encoding="utf-8")

    manifest_payload: dict[str, Any] = {
        "schema": "xvs.display.g1_run_manifest/1",
        "work_package": "WP-08",
        "gate": "G1",
        "batch_id": "B12",
        "commit": commit,
        "git_status_start": git_status_start,
        "git_status_end": git_status_end,
        "argv": [str(Path(sys.argv[0]).as_posix())] + sys.argv[1:],
        "workers": 1,
        "elapsed_wall_s": elapsed_s,
        "code_hashes_start": code_hashes_start,
        "code_hashes_end": code_hashes_end,
        "code_hashes_match": (code_hashes_start == code_hashes_end),
        "inputs_sha256": inputs_sha256,
        "outputs_sha256": {
            "output/xvs/wp08/g1/results.json": _sha256_file(results_path),
            **{f"output/xvs/wp08/g1/stills/{k}": v for k, v in stills_res["png_hashes"].items()},
        },
    }
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {results_path} and {manifest_path} in {elapsed_s:.2f}s.")
    return results_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="WP-08 Gate G1 Display & SITL Producer")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to <repo>/output/xvs/wp08/g1)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to configs/xvs/display/g1.json",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_SPEC_PATH,
        help="Path to configs/xvs/display/g1_spec.json",
    )
    parser.add_argument(
        "--max-encounters",
        type=int,
        default=None,
        help="Optional override for number of SITL encounters (for fast smoke tests)",
    )
    args = parser.parse_args()

    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else (REPO_ROOT / "output" / "xvs" / "wp08" / "g1")
    )
    run_display_g1_pipeline(
        out_dir=out_dir,
        config_path=args.config,
        spec_path=args.spec,
        max_encounters=args.max_encounters,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
