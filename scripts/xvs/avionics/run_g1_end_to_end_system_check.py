"""Single-process Batch B14 Producer: End-to-End Detector-to-Avionics Sensitivity & Integrity Harness.

Executes `docs/xvs/batches/BATCH_B14_G1_SYSTEM_HARNESS.md`:
  1. V2_B14 & V3_B14: Evaluates real WP-07 G1 calibrated `MonitorThresholdsG1` and altitude-binned
     Mondrian `ConformalModelG1` (`output/xvs/wp07/g1/results.json`) on real WP-01 G4 4-camera (`FRBL`)
     CesiumJS frames (`T1_cruise` & `T2_ksfo28l`), coupled directly to `extract_viewport` + `draw_g1_symbology`,
     `PassiveTrackAvionicsBridge.step_from_tracker`, and `SitlAvoidanceAutopilot.step_from_wire_packets`
     across `VALID` -> `F_isolated` -> `INVALID_ALERT` -> `VALID` recovery (`0` end-to-end HMI).
  2. V1_B14: Evaluates the 50-encounter CRN closed-loop chain (`WP-04 -> WP-05 G1 MultiCameraTracker
     -> WP-07 G1 Integrity Gate -> WP-08 G1 Avionics Bridge + SITL Autopilot`) side-by-side under both
     classical `D0` (`configs/xvs/tracking/g0_measured_d0.json`) and measured `D1`
     (`configs/xvs/tracking/g1_measured_d1.json` or `--d1-closure-config`).
  3. V4_B14: Static AST + runtime `PoisonedTruthTrack` sentinel truth-isolation audit across
     `src/xvs/avionics/g1_system_harness.py`, `src/xvs/avionics/bridge.py`,
     `src/xvs/avionics/sitl_closed_loop.py`, and `src/xvs/display/symbology.py`.
  4. V5_B14: Repeated-run byte-identity determinism check (`workers = 1`) and 3-panel comparative PNG
     rendering (`g1_d0_vs_d1_system_leadtime_hmd.png`).

Writes:
  - `output/xvs/wp08/b14_system_harness/results.json`
  - `output/xvs/wp08/b14_system_harness/run_manifest.json`
  - `output/xvs/wp08/b14_system_harness/g1_d0_vs_d1_system_leadtime_hmd.png`
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
from typing import Any, Mapping, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.avionics.bridge import OwnshipNavState, PassiveTrackAvionicsBridge
from xvs.avionics.g1_system_harness import (
    DEFAULT_B14_SPEC_PATH,
    evaluate_wp07_g1_end_to_end_transitions,
    file_sha256,
    resolve_d0_closure_config_path,
    resolve_d1_closure_config_path,
    resolve_wp07_g1_results_path,
    run_d0_vs_d1_closed_loop_sensitivity,
)
from xvs.avionics.sitl_closed_loop import SitlAvoidanceAutopilot, SitlAvoidanceConfig
from xvs.display.symbology import draw_g1_symbology
from xvs.display.viewport import (
    compute_viewport_pose_from_estimated_attitude,
    get_viewport_camera,
)
from xvs.geo import rotations
from xvs.paths import get_data_root
from xvs.tracking.risk import RiskEstimate

TRACKED_CODE_FILES = (
    "src/xvs/avionics/g1_system_harness.py",
    "src/xvs/avionics/bridge.py",
    "src/xvs/avionics/sitl_closed_loop.py",
    "src/xvs/display/symbology.py",
    "scripts/xvs/avionics/run_g1_end_to_end_system_check.py",
    "scripts/xvs/avionics/report_tables_g1_system_summary.py",
    "tests/xvs/test_g1_system_harness.py",
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


class PoisonedTruthTrack:
    """Sentinel passive track raising `AssertionError` if any ground-truth attribute is read."""

    track_id = "T0099"
    t_s = 12.0
    az = 0.12
    el = 0.015
    az_rate = -0.0021
    el_rate = 0.0002
    s = 0.0065
    s_mom = 0.0065
    s_dot = 0.00042

    def __getattr__(self, name: str) -> Any:
        if name in FORBIDDEN_TRUTH_IDENTIFIERS or "true" in name.lower() or "truth" in name.lower():
            raise AssertionError(f"Truth-isolation violation: accessed forbidden attribute {name!r}")
        raise AttributeError(name)


def _git_head_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def _git_status_porcelain() -> str:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return "unknown"
    lines = [
        ln
        for ln in proc.stdout.splitlines()
        if ln.strip() and not ln[3:].strip().replace("\\", "/").startswith("output/")
    ]
    return "\n".join(lines)


def _compute_code_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for rel in TRACKED_CODE_FILES:
        p = REPO_ROOT / rel
        if p.is_file():
            hashes[rel] = file_sha256(p)
    return hashes


def evaluate_b14_truth_isolation() -> dict[str, Any]:
    """Static AST + runtime `PoisonedTruthTrack` audit across B14/B12/B10 production modules (`V4_B14`)."""
    audit_files = (
        REPO_ROOT / "src" / "xvs" / "avionics" / "g1_system_harness.py",
        REPO_ROOT / "src" / "xvs" / "avionics" / "bridge.py",
        REPO_ROOT / "src" / "xvs" / "avionics" / "sitl_closed_loop.py",
        REPO_ROOT / "src" / "xvs" / "display" / "symbology.py",
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

    # Runtime sentinel check across HUD + Bridge + SITL Autopilot
    runtime_violations = 0
    sentinel_trk = PoisonedTruthTrack()
    sentinel_risk = RiskEstimate(
        track_id="T0099",
        t_s=12.0,
        s_rad=0.0065,
        s_dot_radps=0.00042,
        sigma_s_dot=1e-5,
        tau_s=15.5,
        omega_los_radps=0.0021,
        range_interval_m=(1100.0, 2100.0),
        hmd_interval_m=(70.0, 230.0),
        alert_a1_warning=True,
        alert_a1_corrective=True,
        alert_a2_warning=True,
        alert_a2_corrective=True,
        wc1_warning=True,
        wc1_corrective=True,
        hmd_min_m=70.0,
        hmd_max_m=230.0,
        hmd_hat_m=150.0,
        tau_mod_min_s=11.0,
        tau_mod_max_s=19.0,
        tau_mod_hat_s=15.0,
        tau_s_hat_s=15.5,
        beta_wc=0.50,
    )

    try:
        disp_cfg = json.loads(
            (REPO_ROOT / "configs" / "xvs" / "display" / "g1.json").read_text(encoding="utf-8")
        )
        view_cam = get_viewport_camera(disp_cfg)
        r_ned_b = rotations.euler_to_r(0.0, 0.0, 0.0)
        vp_pose = compute_viewport_pose_from_estimated_attitude(
            ownship_lat_deg=37.30,
            ownship_lon_deg=-121.00,
            ownship_h_m=1500.0,
            r_ned_body_est=r_ned_b,
            head_euler_rad=(0.0, 0.0, 0.0),
            config=disp_cfg,
        )
        blank_img = np.zeros((view_cam.height, view_cam.width, 3), dtype=np.uint8)
        draw_g1_symbology(
            img=blank_img,
            aircraft_pose_est={
                "lat_deg": 37.30,
                "lon_deg": -121.00,
                "h_m": 1500.0,
                "yaw_rad": 0.0,
                "pitch_rad": 0.0,
                "roll_rad": 0.0,
            },
            v_ned=np.array([100.0, 0.0, 0.0], dtype=np.float64),
            p_ecef_view=vp_pose["p_ecef_view"],
            r_ecef_view_est=vp_pose["r_ecef_view_est"],
            r_ned_body_est=r_ned_b,
            head_euler_rad=(0.0, 0.0, 0.0),
            view_cam=view_cam,
            tracks_and_risks=[(sentinel_trk, sentinel_risk)],
            integrity_state="VALID",
            pl_deg=0.05,
        )
        bridge = PassiveTrackAvionicsBridge()
        own_nav = OwnshipNavState(
            lat_deg=37.30,
            lon_deg=-121.00,
            alt_msl_m=1500.0,
            alt_geo_m=1506.0,
            vn_mps=100.0,
            ve_mps=0.0,
            vd_mps=0.0,
            heading_deg=0.0,
        )
        step_out = bridge.step_from_tracker(
            ownship=own_nav,
            confirmed_tracks=[sentinel_trk],
            risk_by_track_id={"T0099": sentinel_risk},
            integrity_state="VALID",
            t_s=12.0,
            class_by_track_id={"T0099": "C2_SEP"},
        )
        ap = SitlAvoidanceAutopilot(
            nominal_heading_deg=0.0,
            config=SitlAvoidanceConfig.from_display_config(disp_cfg),
        )
        ap.step_from_wire_packets(
            t_s=12.0,
            ownship=own_nav,
            mavlink_wire_packets=step_out.mavlink_frames,
            gdl90_wire_frames=step_out.gdl90_frames,
        )
    except AssertionError as exc:
        runtime_violations += 1
        violations.append(f"runtime_sentinel: {exc}")

    total_violations = forbidden_imports_count + forbidden_attr_refs_count + runtime_violations
    return {
        "files_audited": len(audit_files),
        "audited_paths": [str(p.relative_to(REPO_ROOT).as_posix()) for p in audit_files],
        "forbidden_imports_count": forbidden_imports_count,
        "forbidden_attr_refs_count": forbidden_attr_refs_count,
        "runtime_sentinel_violations": runtime_violations,
        "total_violations": total_violations,
        "isolated": (total_violations == 0),
        "violations": violations,
    }


def _draw_dashed_line_bgr(
    canvas: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color_bgr: tuple[int, int, int],
    thickness: int = 1,
    dash_px: int = 8,
    gap_px: int = 6,
) -> None:
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    dist = math.hypot(dx, dy)
    if dist < 1.0:
        return
    ux = dx / dist
    uy = dy / dist
    s = 0.0
    step = float(dash_px + gap_px)
    while s < dist:
        e = min(dist, s + float(dash_px))
        x1 = int(round(p1[0] + ux * s))
        y1 = int(round(p1[1] + uy * s))
        x2 = int(round(p1[0] + ux * e))
        y2 = int(round(p1[1] + uy * e))
        cv2.line(canvas, (x1, y1), (x2, y2), color_bgr, thickness, cv2.LINE_AA)
        s += step


def render_d0_vs_d1_comparison_plot(
    sens: Mapping[str, Any],
    wp07: Mapping[str, Any],
) -> bytes:
    """Render deterministic 3-panel comparative PNG (`1680x640` RGB) for `D0` vs `D1` sensitivity.

    Uses OpenCV vector anti-aliasing (`cv2.LINE_AA`) and pure ASCII typography (`--` only, zero
    missing-glyph boxes) and returns compressed PNG bytes for byte-identity verification.
    """
    w_img, h_img = 1680, 640
    # Dark slate background in BGR
    canvas = np.full((h_img, w_img, 3), (26, 22, 18), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Header bar
    cv2.rectangle(canvas, (0, 0), (w_img, 64), (40, 32, 24), -1)
    cv2.line(canvas, (0, 64), (w_img, 64), (110, 95, 75), 1)

    d0_s50 = float(sens["detector_closures"]["D0"]["s50_px"])
    d1_s50 = float(sens["detector_closures"]["D1"]["s50_px"])
    att_sig = float(sens["wp07_coupled_att_sigma_deg"])
    n_enc = int(sens["encounters_count"])
    hmi_cnt = int(wp07["end_to_end_hmi_al_050"])

    title_txt = (
        f"OpenXVS Batch B14 -- End-to-End Gate G1 System Sensitivity "
        f"(D0 vs D1, n={n_enc} CRN Threats, workers=1)"
    )
    sub_txt = (
        f"Classical D0 (s50={d0_s50:.2f} px) vs Neural D1 (s50={d1_s50:.2f} px) | "
        f"WP-07 G1 Coupled sigma_att={att_sig:.4f} deg, End-to-End HMI={hmi_cnt}"
    )
    cv2.putText(canvas, title_txt, (24, 28), font, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(canvas, sub_txt, (24, 52), font, 0.46, (195, 215, 235), 1, cv2.LINE_AA)

    per_enc: Sequence[Mapping[str, Any]] = sens["per_encounter"]
    g_all = sens["paired_gains"]["all_50_d1_vs_d0_b050"]
    g_uas = sens["paired_gains"].get("c1_uas_d1_vs_d0_b050", g_all)
    m_d0_all = sens["modes_all_50"]["d0_g1_wc1_b050"]
    m_d1_all = sens["modes_all_50"]["d1_g1_wc1_b050"]
    m_g0_all = sens["modes_all_50"]["d0_g0_a1"]

    # Colors (BGR)
    col_panel_bg = (34, 29, 24)
    col_grid = (62, 54, 46)
    col_axis = (165, 160, 150)
    col_text = (232, 232, 232)
    col_sub = (180, 185, 190)
    col_uas = (220, 245, 60)      # Bright cyan-green for C1_UAS
    col_manned = (80, 185, 255)   # Warm gold-amber for C2..C6
    col_diag = (150, 150, 150)

    # ------------------------------------------------------------------
    # Panel 1: Initial Track Confirmation Range (km), D0 vs D1
    # ------------------------------------------------------------------
    p1_x0, p1_y0, p1_x1, p1_y1 = 72, 112, 532, 542
    cv2.rectangle(canvas, (p1_x0, p1_y0), (p1_x1, p1_y1), col_panel_bg, -1)
    cv2.putText(
        canvas,
        "1. Initial Track Confirmation Range (km)",
        (p1_x0 - 8, p1_y0 - 14),
        font,
        0.50,
        col_text,
        1,
        cv2.LINE_AA,
    )

    max_km = 10.0
    for tk in (0, 2, 4, 6, 8, 10):
        frac = tk / max_km
        px = int(round(p1_x0 + frac * (p1_x1 - p1_x0)))
        py = int(round(p1_y1 - frac * (p1_y1 - p1_y0)))
        cv2.line(canvas, (px, p1_y0), (px, p1_y1), col_grid, 1)
        cv2.line(canvas, (p1_x0, py), (p1_x1, py), col_grid, 1)
        cv2.putText(canvas, f"{tk}", (px - 6, p1_y1 + 20), font, 0.40, col_sub, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{tk}", (p1_x0 - 26, py + 4), font, 0.40, col_sub, 1, cv2.LINE_AA)

    _draw_dashed_line_bgr(canvas, (p1_x0, p1_y1), (p1_x1, p1_y0), col_diag, 1, 7, 5)
    cv2.rectangle(canvas, (p1_x0, p1_y0), (p1_x1, p1_y1), col_axis, 1)

    for row in per_enc:
        x_km = min(max_km, max(0.0, float(row["d0_confirm_range_3d_m"]) / 1000.0))
        y_km = min(max_km, max(0.0, float(row["d1_confirm_range_3d_m"]) / 1000.0))
        sx = int(round(p1_x0 + (x_km / max_km) * (p1_x1 - p1_x0)))
        sy = int(round(p1_y1 - (y_km / max_km) * (p1_y1 - p1_y0)))
        if row["class_id"] == "C1_UAS":
            pts = np.array([[sx, sy - 6], [sx + 6, sy], [sx, sy + 6], [sx - 6, sy]], dtype=np.int32)
            cv2.fillConvexPoly(canvas, pts, col_uas, cv2.LINE_AA)
            cv2.polylines(canvas, [pts], True, (15, 15, 15), 1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (sx, sy), 4, col_manned, -1, cv2.LINE_AA)
            cv2.circle(canvas, (sx, sy), 4, (20, 20, 20), 1, cv2.LINE_AA)

    # Panel 1 legend & stats box (top-left inside panel)
    cv2.rectangle(canvas, (p1_x0 + 10, p1_y0 + 10), (p1_x0 + 276, p1_y0 + 88), (22, 18, 15), -1)
    cv2.rectangle(canvas, (p1_x0 + 10, p1_y0 + 10), (p1_x0 + 276, p1_y0 + 88), (90, 85, 75), 1)
    pts_lg = np.array(
        [[p1_x0 + 22, p1_y0 + 22], [p1_x0 + 28, p1_y0 + 28], [p1_x0 + 22, p1_y0 + 34], [p1_x0 + 16, p1_y0 + 28]],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(canvas, pts_lg, col_uas, cv2.LINE_AA)
    cv2.putText(canvas, "C1_UAS (s50-limited)", (p1_x0 + 36, p1_y0 + 32), font, 0.39, col_uas, 1, cv2.LINE_AA)
    cv2.circle(canvas, (p1_x0 + 22, p1_y0 + 48), 4, col_manned, -1, cv2.LINE_AA)
    cv2.putText(canvas, "C2..C6 Manned Airframes", (p1_x0 + 36, p1_y0 + 52), font, 0.39, col_manned, 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"Gain All: +{float(g_all['delta_confirm_range_3d_mean_m']):.0f} m | UAS: +{float(g_uas['delta_confirm_range_3d_mean_m']):.0f} m",
        (p1_x0 + 16, p1_y0 + 76),
        font,
        0.37,
        (225, 240, 225),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(canvas, "D0 Confirm Range (km)", (p1_x0 + 145, p1_y1 + 42), font, 0.42, col_text, 1, cv2.LINE_AA)
    cv2.putText(canvas, "D1 (km)", (p1_x0 - 62, p1_y0 + 215), font, 0.40, col_text, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Panel 2: Warning Lead Time Before CPA (s), D0 vs D1
    # ------------------------------------------------------------------
    p2_x0, p2_y0, p2_x1, p2_y1 = 626, 112, 1086, 542
    cv2.rectangle(canvas, (p2_x0, p2_y0), (p2_x1, p2_y1), col_panel_bg, -1)
    cv2.putText(
        canvas,
        "2. Warning Lead Time Before CPA (s)",
        (p2_x0 - 4, p2_y0 - 14),
        font,
        0.50,
        col_text,
        1,
        cv2.LINE_AA,
    )

    max_lead_s = 45.0
    for tk in (0, 10, 20, 30, 40):
        frac = tk / max_lead_s
        px = int(round(p2_x0 + frac * (p2_x1 - p2_x0)))
        py = int(round(p2_y1 - frac * (p2_y1 - p2_y0)))
        cv2.line(canvas, (px, p2_y0), (px, p2_y1), col_grid, 1)
        cv2.line(canvas, (p2_x0, py), (p2_x1, py), col_grid, 1)
        cv2.putText(canvas, f"{tk}", (px - 8, p2_y1 + 20), font, 0.40, col_sub, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{tk}", (p2_x0 - 28, py + 4), font, 0.40, col_sub, 1, cv2.LINE_AA)

    # DO-365B 12.5 s warning lead reference
    ref_frac = 12.5 / max_lead_s
    rx_125 = int(round(p2_x0 + ref_frac * (p2_x1 - p2_x0)))
    ry_125 = int(round(p2_y1 - ref_frac * (p2_y1 - p2_y0)))
    _draw_dashed_line_bgr(canvas, (p2_x0, ry_125), (p2_x1, ry_125), (90, 140, 230), 1, 5, 5)
    _draw_dashed_line_bgr(canvas, (rx_125, p2_y0), (rx_125, p2_y1), (90, 140, 230), 1, 5, 5)
    _draw_dashed_line_bgr(canvas, (p2_x0, p2_y1), (p2_x1, p2_y0), col_diag, 1, 7, 5)
    cv2.rectangle(canvas, (p2_x0, p2_y0), (p2_x1, p2_y1), col_axis, 1)

    for row in per_enc:
        x_s = min(max_lead_s, max(0.0, float(row["d0_lead_time_s"])))
        y_s = min(max_lead_s, max(0.0, float(row["d1_lead_time_s"])))
        sx = int(round(p2_x0 + (x_s / max_lead_s) * (p2_x1 - p2_x0)))
        sy = int(round(p2_y1 - (y_s / max_lead_s) * (p2_y1 - p2_y0)))
        if row["class_id"] == "C1_UAS":
            pts = np.array([[sx, sy - 6], [sx + 6, sy], [sx, sy + 6], [sx - 6, sy]], dtype=np.int32)
            cv2.fillConvexPoly(canvas, pts, col_uas, cv2.LINE_AA)
            cv2.polylines(canvas, [pts], True, (15, 15, 15), 1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (sx, sy), 4, col_manned, -1, cv2.LINE_AA)
            cv2.circle(canvas, (sx, sy), 4, (20, 20, 20), 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (p2_x0 + 10, p2_y0 + 10), (p2_x0 + 295, p2_y0 + 72), (22, 18, 15), -1)
    cv2.rectangle(canvas, (p2_x0 + 10, p2_y0 + 10), (p2_x0 + 295, p2_y0 + 72), (90, 85, 75), 1)
    cv2.putText(
        canvas,
        f"Mean Lead: D0 {float(m_d0_all['lead_time_all_mean_s']):.2f}s -> D1 {float(m_d1_all['lead_time_all_mean_s']):.2f}s",
        (p2_x0 + 18, p2_y0 + 32),
        font,
        0.38,
        col_text,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"C1_UAS Gain: +{float(g_uas['delta_lead_time_mean_s']):.2f}s mean (+{float(g_uas['delta_lead_time_median_s']):.2f}s med)",
        (p2_x0 + 18, p2_y0 + 56),
        font,
        0.37,
        col_uas,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(canvas, "12.5s ref", (p2_x1 - 74, ry_125 - 6), font, 0.35, (120, 175, 245), 1, cv2.LINE_AA)
    cv2.putText(canvas, "D0 Warning Lead Time (s)", (p2_x0 + 135, p2_y1 + 42), font, 0.42, col_text, 1, cv2.LINE_AA)
    cv2.putText(canvas, "D1 (s)", (p2_x0 - 54, p2_y0 + 215), font, 0.40, col_text, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Panel 3: Sorted Closed-Loop Min HMD Curves (m)
    # ------------------------------------------------------------------
    p3_x0, p3_y0, p3_x1, p3_y1 = 1180, 112, 1646, 542
    cv2.rectangle(canvas, (p3_x0, p3_y0), (p3_x1, p3_y1), col_panel_bg, -1)
    cv2.putText(
        canvas,
        "3. Sorted Closed-Loop Min HMD (m)",
        (p3_x0 - 4, p3_y0 - 14),
        font,
        0.50,
        col_text,
        1,
        cv2.LINE_AA,
    )

    max_hmd_plot = 1200.0
    for tk_m in (0, 200, 400, 600, 800, 1000, 1200):
        frac = tk_m / max_hmd_plot
        py = int(round(p3_y1 - frac * (p3_y1 - p3_y0)))
        cv2.line(canvas, (p3_x0, py), (p3_x1, py), col_grid, 1)
        cv2.putText(canvas, f"{tk_m}", (p3_x0 - 40, py + 4), font, 0.39, col_sub, 1, cv2.LINE_AA)

    n_pts = max(1, len(per_enc))
    x_ticks = [1, max(1, n_pts // 4), max(1, n_pts // 2), max(1, (3 * n_pts) // 4), n_pts]
    for rk in sorted(set(x_ticks)):
        frac_x = (rk - 1) / max(1, n_pts - 1) if n_pts > 1 else 0.5
        px = int(round(p3_x0 + frac_x * (p3_x1 - p3_x0)))
        cv2.line(canvas, (px, p3_y0), (px, p3_y1), col_grid, 1)
        cv2.putText(canvas, f"{rk}", (px - 6, p3_y1 + 20), font, 0.39, col_sub, 1, cv2.LINE_AA)

    # NMAC (152.4 m) and LoWC (609.6 m) horizontal limit lines
    y_nmac = int(round(p3_y1 - (152.4 / max_hmd_plot) * (p3_y1 - p3_y0)))
    y_lowc = int(round(p3_y1 - (609.6 / max_hmd_plot) * (p3_y1 - p3_y0)))
    _draw_dashed_line_bgr(canvas, (p3_x0, y_nmac), (p3_x1, y_nmac), (80, 80, 250), 2, 8, 5)
    _draw_dashed_line_bgr(canvas, (p3_x0, y_lowc), (p3_x1, y_lowc), (60, 195, 255), 1, 8, 5)
    cv2.putText(canvas, "NMAC 152.4m", (p3_x1 - 102, y_nmac - 6), font, 0.36, (120, 120, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "LoWC 609.6m", (p3_x1 - 102, y_lowc - 6), font, 0.36, (90, 210, 255), 1, cv2.LINE_AA)

    curves = (
        ("open_loop_hmd_m", (120, 120, 175), 1),
        ("d0_g0_a1_hmd_m", (70, 145, 245), 2),
        ("d0_hmd_m", (245, 190, 75), 2),
        ("d1_hmd_m", (110, 245, 95), 2),
    )
    for key_h, col_c, thk in curves:
        vals = sorted(float(r[key_h]) for r in per_enc)
        poly = []
        for idx_v, val_m in enumerate(vals):
            fx = idx_v / max(1, n_pts - 1) if n_pts > 1 else 0.5
            fy = min(max_hmd_plot, max(0.0, val_m)) / max_hmd_plot
            px = int(round(p3_x0 + fx * (p3_x1 - p3_x0)))
            py = int(round(p3_y1 - fy * (p3_y1 - p3_y0)))
            poly.append([px, py])
        if len(poly) >= 2:
            cv2.polylines(canvas, [np.array(poly, dtype=np.int32)], False, col_c, thk, cv2.LINE_AA)
        elif len(poly) == 1:
            cv2.circle(canvas, (poly[0][0], poly[0][1]), 4, col_c, -1, cv2.LINE_AA)

    cv2.rectangle(canvas, (p3_x0, p3_y0), (p3_x1, p3_y1), col_axis, 1)

    # Panel 3 legend box (top-left inside panel)
    cv2.rectangle(canvas, (p3_x0 + 10, p3_y0 + 10), (p3_x0 + 278, p3_y0 + 108), (22, 18, 15), -1)
    cv2.rectangle(canvas, (p3_x0 + 10, p3_y0 + 10), (p3_x0 + 278, p3_y0 + 108), (90, 85, 75), 1)
    lg_items = (
        ((110, 245, 95), f"G1 D1+WC1: {int(m_d1_all['nmac_resolved_count'])}/{n_enc} NMAC (min {float(m_d1_all['hmd_min_m']):.0f}m)"),
        ((245, 190, 75), f"G1 D0+WC1: {int(m_d0_all['nmac_resolved_count'])}/{n_enc} NMAC (min {float(m_d0_all['hmd_min_m']):.0f}m)"),
        ((70, 145, 245), f"G0 D0+A1:  {int(m_g0_all['nmac_resolved_count'])}/{n_enc} NMAC resolved"),
        ((120, 120, 175), "Open-Loop (< 152 m collision)"),
    )
    for idx_l, (lc, ltxt) in enumerate(lg_items):
        ly = p3_y0 + 28 + idx_l * 21
        cv2.line(canvas, (p3_x0 + 18, ly - 4), (p3_x0 + 38, ly - 4), lc, 2, cv2.LINE_AA)
        cv2.putText(canvas, ltxt, (p3_x0 + 45, ly), font, 0.36, col_text, 1, cv2.LINE_AA)

    cv2.putText(canvas, "Sorted Encounter Rank", (p3_x0 + 145, p3_y1 + 42), font, 0.42, col_text, 1, cv2.LINE_AA)

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def _evaluate_determinism_probe(
    *,
    d0_path: Path,
    d1_path: Path,
    sitl_cfg: SitlAvoidanceConfig,
    wp07_att_sigma_deg: float,
    wp07_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Run a fast 2-encounter repeated-execution check confirming byte-identical JSON & PNG hashes (`V5_B14`)."""
    run_a = run_d0_vs_d1_closed_loop_sensitivity(
        d0_closure_path=d0_path,
        d1_closure_path=d1_path,
        sitl_config=sitl_cfg,
        wp07_att_sigma_deg=wp07_att_sigma_deg,
        max_encounters=2,
    )
    png_a = render_d0_vs_d1_comparison_plot(run_a, wp07_summary)

    run_b = run_d0_vs_d1_closed_loop_sensitivity(
        d0_closure_path=d0_path,
        d1_closure_path=d1_path,
        sitl_config=sitl_cfg,
        wp07_att_sigma_deg=wp07_att_sigma_deg,
        max_encounters=2,
    )
    png_b = render_d0_vs_d1_comparison_plot(run_b, wp07_summary)

    json_a_sha = hashlib.sha256(json.dumps(run_a, sort_keys=True).encode("utf-8")).hexdigest()
    json_b_sha = hashlib.sha256(json.dumps(run_b, sort_keys=True).encode("utf-8")).hexdigest()
    png_a_sha = hashlib.sha256(png_a).hexdigest()
    png_b_sha = hashlib.sha256(png_b).hexdigest()

    return {
        "workers": 1,
        "repeated_runs": 2,
        "probe_encounters_count": 2,
        "json_sha256_run1": json_a_sha,
        "json_sha256_run2": json_b_sha,
        "json_byte_identical": (json_a_sha == json_b_sha),
        "png_sha256_run1": png_a_sha,
        "png_sha256_run2": png_b_sha,
        "png_byte_identical": (png_a_sha == png_b_sha),
    }


def run_b14_system_check(
    *,
    out_dir: Path,
    d1_closure_config: Path | None = None,
    d0_closure_config: Path | None = None,
    wp07_g1_results: Path | None = None,
    spec_path: Path = DEFAULT_B14_SPEC_PATH,
    max_encounters: int | None = None,
) -> dict[str, Any]:
    """Execute the full Batch B14 single-process verification harness and write artifacts."""
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    commit = _git_head_commit()
    git_status_start = _git_status_porcelain()
    code_hashes_start = _compute_code_hashes()

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    d0_path = resolve_d0_closure_config_path(d0_closure_config)
    d1_path = resolve_d1_closure_config_path(d1_closure_config)
    wp07_path = resolve_wp07_g1_results_path(wp07_g1_results)

    # 1. Real WP-07 G1 -> HUD -> Avionics Bridge -> SITL Autopilot integrity state loop
    wp07_e2e = evaluate_wp07_g1_end_to_end_transitions(
        wp07_results_path=wp07_path,
        spec=spec,
    )
    coupled_att_sigma_deg = float(
        wp07_e2e["wp07_test_split_reference"]["frbl_post_isolation_rmse_deg"]
    )

    # 2. 50-encounter CRN D0 vs D1 closed-loop system sensitivity
    disp_g1_path = REPO_ROOT / "configs" / "xvs" / "display" / "g1.json"
    sitl_cfg = SitlAvoidanceConfig.from_display_config(disp_g1_path)
    n_enc = int(max_encounters) if max_encounters is not None else int(spec["benchmark_encounters_count"])

    sens = run_d0_vs_d1_closed_loop_sensitivity(
        d0_closure_path=d0_path,
        d1_closure_path=d1_path,
        sitl_config=sitl_cfg,
        wp07_att_sigma_deg=coupled_att_sigma_deg,
        max_encounters=n_enc,
    )

    # 3. Truth isolation audit (V4_B14)
    truth_iso = evaluate_b14_truth_isolation()

    # 4. Comparative PNG plot
    png_bytes = render_d0_vs_d1_comparison_plot(sens, wp07_e2e)
    png_path = out_dir / "g1_d0_vs_d1_system_leadtime_hmd.png"
    png_path.write_bytes(png_bytes)

    # 5. Repeated-run determinism check (V5_B14)
    det_res = _evaluate_determinism_probe(
        d0_path=d0_path,
        d1_path=d1_path,
        sitl_cfg=sitl_cfg,
        wp07_att_sigma_deg=coupled_att_sigma_deg,
        wp07_summary=wp07_e2e,
    )

    elapsed_s = float(time.perf_counter() - t0)
    code_hashes_end = _compute_code_hashes()
    git_status_end = _git_status_porcelain()

    inputs_sha256 = {
        "configs/xvs/avionics/b14_system_spec.json": file_sha256(spec_path),
        "configs/xvs/display/g1.json": file_sha256(disp_g1_path),
        "configs/xvs/tracking/g1.json": file_sha256(REPO_ROOT / "configs" / "xvs" / "tracking" / "g1.json"),
        "d0_closure_config": file_sha256(d0_path),
        "d1_closure_config": file_sha256(d1_path),
        "wp07_g1_results": file_sha256(wp07_path),
    }

    argv_norm = [str(Path(sys.argv[0]).as_posix())] + [str(a) for a in sys.argv[1:]]
    results_payload: dict[str, Any] = {
        "schema": "xvs.avionics.b14_system_harness/1",
        "work_package": "WP-08",
        "gate": "G1",
        "batch_id": "B14",
        "commit": commit,
        "git_status": git_status_end,
        "argv": argv_norm,
        "workers": 1,
        "elapsed_wall_s": elapsed_s,
        "inputs_sha256": inputs_sha256,
        "v1_d0_vs_d1_sensitivity": sens,
        "v2_v3_wp07_g1_integrity_loop": wp07_e2e,
        "v4_truth_isolation": truth_iso,
        "v5_determinism": det_res,
        "plot": {
            "filename": "g1_d0_vs_d1_system_leadtime_hmd.png",
            "sha256": file_sha256(png_path),
            "size_bytes": int(png_path.stat().st_size),
        },
    }

    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results_payload, indent=2) + "\n", encoding="utf-8")

    manifest_payload: dict[str, Any] = {
        "schema": "xvs.avionics.b14_run_manifest/1",
        "work_package": "WP-08",
        "gate": "G1",
        "batch_id": "B14",
        "commit": commit,
        "git_status": git_status_end,
        "git_status_start": git_status_start,
        "git_status_end": git_status_end,
        "argv": argv_norm,
        "workers": 1,
        "elapsed_wall_s": elapsed_s,
        "code_hashes": code_hashes_end,
        "expected_code_hashes": code_hashes_start,
        "code_hashes_start": code_hashes_start,
        "code_hashes_end": code_hashes_end,
        "code_hashes_match": (code_hashes_start == code_hashes_end),
        "inputs_sha256": inputs_sha256,
        "outputs_sha256": {
            "output/xvs/wp08/b14_system_harness/results.json": file_sha256(results_path),
            "output/xvs/wp08/b14_system_harness/g1_d0_vs_d1_system_leadtime_hmd.png": file_sha256(png_path),
        },
    }
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")

    print(
        f"Batch B14 complete in {elapsed_s:.2f}s (workers=1): "
        f"wrote {results_path}, {manifest_path}, and {png_path.name}."
    )
    return results_payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch B14 End-to-End Cross-WP G1 System Sensitivity & Integrity Check"
    )
    parser.add_argument(
        "--d1-closure-config",
        type=Path,
        default=None,
        help="Optional explicit path to measured D1 closure JSON (`g1_measured_d1.json`)",
    )
    parser.add_argument(
        "--d0-closure-config",
        type=Path,
        default=None,
        help="Optional explicit path to classical D0 closure JSON (`g0_measured_d0.json`)",
    )
    parser.add_argument(
        "--wp07-g1-results",
        type=Path,
        default=None,
        help="Optional explicit path to WP-07 G1 results JSON (`output/xvs/wp07/g1/results.json`)",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_B14_SPEC_PATH,
        help="Path to configs/xvs/avionics/b14_system_spec.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to <data_root>/xvs/wp08/b14_system_harness)",
    )
    parser.add_argument(
        "--max-encounters",
        type=int,
        default=None,
        help="Optional encounter count override (for fast CLI/unit smoke checks)",
    )
    args = parser.parse_args()

    default_out = (
        get_data_root() / "xvs" / "wp08" / "b14_system_harness"
        if "XVS_DATA_ROOT" in os.environ
        else (REPO_ROOT / "output" / "xvs" / "wp08" / "b14_system_harness")
    )
    out_dir = args.out_dir if args.out_dir is not None else default_out

    run_b14_system_check(
        out_dir=out_dir,
        d1_closure_config=args.d1_closure_config,
        d0_closure_config=args.d0_closure_config,
        wp07_g1_results=args.wp07_g1_results,
        spec_path=args.spec,
        max_encounters=args.max_encounters,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
