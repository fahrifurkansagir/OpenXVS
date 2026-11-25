"""WP-02 Gate G2 - rendered encounters and validated ground truth (WP02_TRAFFIC.md §3 G2, §4).

Rewritten on 2026-09-24 by Architecture Lead (Director, acting as implementer after the hand-over).
It replaces the unfinished Gemini draft, whose defects are listed in docs/xvs/reports/WP02_G2_report.md.

Validation set E-VAL-12 (§3 G2): 12 encounters on ownship T1, one per class x {sky (intruder
150 m above), earth (intruder 300 m below)}, head-on (approach 175 or 185 deg), hmd 100 m,
t_cpa 45 s. Each encounter is rendered at the 4 instants where range = 500, 1000, 2000 and 4000 m:
48 instants.

Renders per instant (one Cesium page per pass, all in one browser context):
  A  mask_amodal + aircraft model (UNLIT white)   - the 6 rig cameras and every ring ROI (*)
  V  mask_visible + aircraft model (UNLIT white)  - same cameras
  S  mask_amodal + SPHERE_R2 at the model origin  - rig cameras whose analytic sphere box meets the
                                                    image, and every ring ROI (*)
  E  WP-01 mask_earth, no models (MSAA 4)         - cameras whose projected centre is in the image
  (*) ring cameras whose projected intruder centre lies in the 2448x2048 image; ROI per §2.4.

Checks (thresholds from §4; "ref" must pass, "hw" is reported):
  GT-1  SPHERE_R2 mask centroid vs projected centre, masks >= 50 px and fully inside the window:
        median <= 0.15 px, max <= 0.50 px.
  GT-2  (spec v1.1) every A and S mask equals the point-sampled reference rasterization of the same
        GLB mesh under the analytic projection (xvs.traffic.raster), except "ties": pixels whose centre has
        the boundary of the union of the projected triangles within 1/16 px (independent review R-G2-2; an
        interior edge does not count). Zero non-tie mismatches. Every A-mask pixel centre
        lies inside the analytic vertex box (tolerance 1/16 px). A mesh crossing the camera plane
        (e.g. a rig side camera with the intruder ~90 deg off axis) is clipped at the frustum near
        plane z = 0.5 m before rasterization (raster.clip_near).
        The v1.0 criterion (amodal box_edges vs analytic box within 1.0 px for masks >= 10 px) is
        still computed and reported; it fails for correct renders when thin parts (wings seen
        edge-on, rotor blades) cover no pixel centre.
  GT-3  (a) v1.0 rule (amodal non-empty exactly when the analytic box is >= 1 px in both directions
        and meets the window): every exception reported; the v1.1 rule is covered by GT-2.
        (b) visible subset of amodal: zero exceptions.
  GT-4  mask_earth value at the pixel nearest the projected centre vs the analytic background
        label, excluding |limb_dv| <= 1 px: agreement rate reported (disagreements listed).
  GT-5  5 instants re-rendered in a fresh process (all passes, all cameras): ref identical
        rgb_sha256; hw max_diff <= 1 and n_diff <= 1 %.
  Mask purity (§2.3): A, V and S masks contain only (0,0,0) and (255,255,255).

Usage:
  python scripts/xvs/validate_traffic_g2.py --backend ref|hw [--out output/xvs/wp02/g2]
  (internal) --gt5-replay <results_<backend>.json>   re-render the GT-5 instants (fresh process)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.optimize import brentq

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.geo import wgs84  # noqa: E402
from xvs.rig.rig import Rig  # noqa: E402
from xvs.sim.renderer import XVSRenderer  # noqa: E402
from xvs.sim.server import XVSServer  # noqa: E402
from xvs.sim.trajectory import t1_cruise_state  # noqa: E402
from xvs.traffic.encounter import (  # noqa: E402
    EncounterParams,
    EncounterTrajectory,
    construct_encounter,
    evaluate_actual_cpa,
)
from xvs.traffic import raster  # noqa: E402
from xvs.traffic.gt import compute_intruder_analytic_gt  # noqa: E402
from xvs.traffic.ring import TrafficRing  # noqa: E402

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCHEMA = "xvs.wp02_g2_results/1"

# ---------------------------------------------------------------------------------------------
# E-VAL-12 definition (fixed; recorded verbatim in the results)
# ---------------------------------------------------------------------------------------------
CLASS_CONFIGS = [
    {"class_id": "C1_UAS", "model_id": "C1_UAS_0001", "gs_mps": 20.0, "alpha_trim_deg": 0.0},
    {"class_id": "C2_SEP", "model_id": "C2_SEP_0001", "gs_mps": 60.0, "alpha_trim_deg": 2.0},
    {"class_id": "C3_TWIN", "model_id": "C3_TWIN_0001", "gs_mps": 85.0, "alpha_trim_deg": 2.0},
    {"class_id": "C4_BIZJET", "model_id": "C4_BIZJET_0001", "gs_mps": 170.0, "alpha_trim_deg": 2.0},
    {"class_id": "C5_TRANSPORT", "model_id": "C5_TRANSPORT_0001", "gs_mps": 185.0, "alpha_trim_deg": 2.0},
    {"class_id": "C6_HELI", "model_id": "C6_HELI_0001", "gs_mps": 50.0, "alpha_trim_deg": -3.0},
]
BACKGROUND_CONFIGS = [
    {"bg": "sky", "vmd_m": 150.0, "approach_deg": 175.0},
    {"bg": "earth", "vmd_m": -300.0, "approach_deg": 185.0},
]
T_CPA_S = 45.0
HMD_M = 100.0
SIDE = "right"
VS_MPS = 0.0
TARGET_RANGES_M = [4000.0, 2000.0, 1000.0, 500.0]  # chronological order along each encounter
RIG_CAMS = ("F", "R", "B", "L", "U", "D")
RING_CAMS = ("T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7")
GT5_INSTANTS = (
    "C1_UAS_sky_500m",
    "C2_SEP_earth_1000m",
    "C3_TWIN_sky_2000m",
    "C5_TRANSPORT_earth_4000m",
    "C6_HELI_earth_500m",
)

TH = {
    "gt1_median_px": 0.15, "gt1_max_px": 0.50, "gt1_min_area_px": 50,
    "gt2_edge_px": 1.0, "gt2_min_area_px": 10, "gt2_raster_tie_px": raster.TIE_PX,
    "gt2_raster_max_mismatch_px": 0, "gt2_box_containment_tol_px": raster.TIE_PX,
    "gt4_limb_band_px": 1.0,
    "gt5_hw_max_diff": 1, "gt5_hw_n_diff_frac": 0.01,
}
MODULES = ("src/xvs/traffic/raster.py", "src/xvs/traffic/gt.py", "src/xvs/traffic/encounter.py", "src/xvs/traffic/ring.py",
           "src/xvs/sim/horizon.py", "src/xvs/sim/renderer.py", "src/xvs/sim/server.py",
           "src/xvs/rig/camera.py", "src/xvs/rig/rig.py", "src/xvs/geo/wgs84.py")
PASSES = ("A", "V", "S", "E")
PASS_PROFILE = {"A": "mask_amodal", "V": "mask_visible", "S": "mask_amodal", "E": "mask_earth"}


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_state() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            return f"unavailable: {exc}"
    return {"head": run(["rev-parse", "HEAD"]), "branch": run(["rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty_files": [ln for ln in run(["status", "--porcelain"]).splitlines() if ln][:200]}


# ---------------------------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------------------------
def solve_instant(traj: EncounterTrajectory, target_m: float) -> float:
    def f(t: float) -> float:
        return float(np.linalg.norm(traj.position_at(t) - t1_cruise_state(t).p_ecef_m) - target_m)
    return float(brentq(f, 0.0, T_CPA_S, xtol=1e-12, rtol=4 * np.finfo(float).eps, maxiter=200))


def build_eval12() -> list[dict[str, Any]]:
    encs = []
    for cc in CLASS_CONFIGS:
        for bc in BACKGROUND_CONFIGS:
            enc_id = f"{cc['class_id']}_{bc['bg']}"
            params = EncounterParams(
                id=enc_id, class_id=cc["class_id"], model_id=cc["model_id"], t_cpa=T_CPA_S,
                hmd_m=HMD_M, side=SIDE, vmd_m=bc["vmd_m"], approach_deg=bc["approach_deg"],
                gs_mps=cc["gs_mps"], vs_mps=VS_MPS, alpha_trim_deg=cc["alpha_trim_deg"], weight=1.0,
                stratum={"set": "E-VAL-12", "background": bc["bg"]},
            )
            traj = construct_encounter(t1_cruise_state, params)
            cpa = evaluate_actual_cpa(t1_cruise_state, traj, 0.0, 60.0, 0.01)
            instants = []
            for r in TARGET_RANGES_M:
                t_s = solve_instant(traj, r)
                instants.append({"instant_id": f"{enc_id}_{int(r)}m", "target_range_m": r, "t_s": t_s})
            encs.append({"encounter_id": enc_id, "class_id": cc["class_id"], "model_id": cc["model_id"],
                         "bg": bc["bg"], "params": params, "traj": traj, "achieved_cpa": cpa, "instants": instants})
    return encs


def camera_set(rig: Rig, ring: TrafficRing, st: Any) -> dict[str, dict[str, Any]]:
    rig_poses = rig.compute_poses(st.lat_deg, st.lon_deg, st.h_m, st.yaw_rad, st.pitch_rad, st.roll_rad)
    ring_poses = ring.compute_poses(st.lat_deg, st.lon_deg, st.h_m, st.yaw_rad, st.pitch_rad, st.roll_rad)
    cams: dict[str, dict[str, Any]] = {}
    for cid in RIG_CAMS:
        p = rig_poses[cid]
        cams[cid] = {"kind": "rig", "pose": p, "model": rig.camera_model, "fov_rad": float(p["fov_rad"]),
                     "p": np.array(p["position"], float), "R": np.array(p["r_ecef_cam"], float)}
    for cid in RING_CAMS:
        p = ring_poses[cid]
        cams[cid] = {"kind": "ring", "pose": p, "model": ring.camera_model, "fov_rad": math.radians(ring.fov_h_deg),
                     "p": np.array(p["position"], float), "R": np.array(p["r_ecef_cam"], float)}
    return cams


def window_of(cam: dict[str, Any], roi: tuple[int, int] | None) -> tuple[float, float, float, float]:
    """Rendered window in full-image pixel coordinates [u_lo, v_lo, u_hi, v_hi] (pixel-edge bounds)."""
    if roi is None:
        m = cam["model"]
        return (-0.5, -0.5, m.width - 0.5, m.height - 0.5)
    u0, v0 = roi
    return (u0 - 0.5, v0 - 0.5, u0 + 1023.5, v0 + 1023.5)


def box_meets(box: list[float] | None, win: tuple[float, float, float, float]) -> bool:
    return box is not None and box[2] >= win[0] and box[0] <= win[2] and box[3] >= win[1] and box[1] <= win[3]


def box_inside(box: list[float] | None, win: tuple[float, float, float, float]) -> bool:
    return box is not None and box[0] >= win[0] and box[1] >= win[1] and box[2] <= win[2] and box[3] <= win[3]


def mask_stats(m01: np.ndarray, u0: int, v0: int) -> dict[str, Any]:
    rows, cols = np.nonzero(m01)
    n = int(rows.size)
    if n == 0:
        return {"px": 0, "box_px": None, "box_edges": None, "centroid": None}
    box = [int(cols.min()) + u0, int(rows.min()) + v0, int(cols.max()) + u0, int(rows.max()) + v0]
    return {"px": n, "box_px": box, "box_edges": [box[0] - 0.5, box[1] - 0.5, box[2] + 0.5, box[3] + 0.5],
            "centroid": [float(cols.mean()) + u0, float(rows.mean()) + v0]}


# ---------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------
class G2Renderer:
    def __init__(self, backend: str, out: Path):
        self.backend = backend
        port = get_free_port()
        self.server = XVSServer(
            server_address=("127.0.0.1", port),
            cesium_build_dir=REPO_ROOT / "sim" / "cesium" / "node_modules" / "cesium" / "Build" / "Cesium",
            page_dir=REPO_ROOT / "sim" / "cesium" / "page",
            cache_dir=REPO_ROOT / "data" / "cache" / "usgs_imagery",
            log_dir=out / "logs",
            offline=True,
        )
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.port = port
        self.r = XVSRenderer(port=port, backend=backend, server=self.server).start()
        self.pages = {}
        self.resolved = {}
        for ps in PASSES:
            pg = self.r.new_camera_page(f"g2_{ps}", profile=PASS_PROFILE[ps])
            self.pages[ps] = pg
            self.resolved[ps] = self.r.resolved_settings
        self.loaded: dict[str, str | None] = {ps: None for ps in PASSES}
        self.model_load_ms: list[float] = []

    def webgl(self) -> dict[str, Any]:
        return self.r.get_webgl_diagnostics(page=self.pages["A"])

    def load(self, ps: str, rel_url: str) -> None:
        if self.loaded[ps] == rel_url:
            return
        pg = self.pages[ps]
        t0 = time.perf_counter()
        self.r.clear_models(page=pg)
        self.r.add_model(url=f"http://127.0.0.1:{self.port}/{rel_url}", custom_shader="unlit_white", page=pg)
        self.model_load_ms.append((time.perf_counter() - t0) * 1000.0)
        self.loaded[ps] = rel_url

    def render(self, ps: str, cam: dict[str, Any], roi: tuple[int, int] | None, model_m: list[float] | None) -> Any:
        pg = self.pages[ps]
        if model_m is not None:
            self.r.set_model_pose(0, model_m, page=pg)
        p = cam["pose"]
        readback, ms_set = self.r.set_camera(position=p["position"], direction=p["direction"], up=p["up"],
                                             fov_rad=cam["fov_rad"], aspect_ratio=1.0, page=pg)
        if roi is not None:
            m = cam["model"]
            self.r.set_roi_frustum(u0=roi[0], v0=roi[1], fx=m.fx, fy=m.fy, cx=m.cx, cy=m.cy, page=pg)
        res = self.r.render_frame(page=pg)
        return res, ms_set

    def close(self) -> None:
        try:
            self.r.close()
        finally:
            self.server.shutdown()


def check_rgb(img: np.ndarray, ps: str) -> tuple[np.ndarray, int, list[int]]:
    """Return (grey uint8, impure pixel count, sorted unique grey values). Masks A/V/S must be pure 0/255."""
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    grey_ok = (r == g) & (g == b)
    if ps in ("A", "V", "S"):
        pure = grey_ok & ((r == 0) | (r == 255))
        impure = int(np.count_nonzero(~pure))
    else:
        impure = int(np.count_nonzero(~grey_ok))
    return r.copy(), impure, sorted(int(x) for x in np.unique(r))


# ---------------------------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------------------------
def instant_plan(encs: list[dict[str, Any]], only: set[str] | None = None) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    plan = []
    for enc in encs:
        for inst in enc["instants"]:
            if only is None or inst["instant_id"] in only:
                plan.append((enc, inst))
    return plan


def process_instant(gr: G2Renderer, rig: Rig, ring: TrafficRing, enc: dict[str, Any], inst: dict[str, Any],
                    verts: dict[str, np.ndarray], faces: dict[str, np.ndarray], root: Path,
                    img_prefix: str) -> dict[str, Any]:
    """Render and measure one instant. Images go to root/<img_prefix>/<instant>/<cam>_<pass>.png."""
    t_start = time.perf_counter()
    t_s = inst["t_s"]
    st = t1_cruise_state(t_s)
    traj: EncounterTrajectory = enc["traj"]
    p_i = traj.position_at(t_s)
    r_i, ypr_i = traj.attitude_at(t_s)
    m4 = np.eye(4)
    m4[:3, :3] = r_i
    m4[:3, 3] = p_i
    model_m = m4.T.flatten().tolist()  # Cesium column-major

    cams = camera_set(rig, ring, st)
    cam_tuples = {cid: (c["p"], c["R"], c["model"]) for cid, c in cams.items()}
    v_o_ecef = wgs84.r_ecef_ned(st.lat_deg, st.lon_deg) @ st.v_ned_mps
    gt_ac = compute_intruder_analytic_gt(t_s, traj, st.p_ecef_m, v_o_ecef, st.r_ecef_body, cam_tuples,
                                         model_vertices=verts[enc["model_id"]],
                                         max_body_dim_m=float(np.max(np.ptp(verts[enc["model_id"]], axis=0))))
    gt_sp = compute_intruder_analytic_gt(t_s, traj, st.p_ecef_m, v_o_ecef, st.r_ecef_body, cam_tuples,
                                         model_vertices=verts["SPHERE_R2"], max_body_dim_m=4.0)

    model_rel = f"assets/traffic/{enc['class_id']}/{enc['model_id']}.glb"
    gr.load("A", model_rel)
    gr.load("V", model_rel)
    gr.load("S", "assets/traffic/CALIBRATION/SPHERE_R2.glb")

    rec: dict[str, Any] = {
        "instant_id": inst["instant_id"], "encounter_id": enc["encounter_id"], "class_id": enc["class_id"],
        "model_id": enc["model_id"], "bg": enc["bg"], "target_range_m": inst["target_range_m"], "t_s": t_s,
        "ownship": {"lat_deg": st.lat_deg, "lon_deg": st.lon_deg, "h_m": st.h_m, "p_ecef_m": st.p_ecef_m.tolist(),
                    "ypr_deg": [math.degrees(st.yaw_rad), math.degrees(st.pitch_rad), math.degrees(st.roll_rad)],
                    "R_ecef_body": st.r_ecef_body.tolist()},
        "intruder": {"p_ecef_m": p_i.tolist(), "ypr_deg": list(ypr_i), "R_ecef_body": r_i.tolist(),
                     "model_matrix_colmajor": model_m, "range_m": gt_ac.range_m,
                     "range_rate_mps": gt_ac.range_rate_mps, "az_el_body_deg": gt_ac.az_el_body_deg,
                     "angular_span_rad": gt_ac.angular_span_rad},
        "cams": {},
    }
    for cid, cam in cams.items():
        g = gt_ac.cams[cid]
        gs = gt_sp.cams[cid]
        if cam["kind"] == "ring":
            if not g.in_image:
                continue
            roi = ring.compute_roi(g.u, g.v)
            passes = ["A", "V", "S", "E"]
        else:
            roi = None
            passes = ["A", "V"]
            if box_meets(gs.box_analytic, window_of(cam, None)):
                passes.append("S")
            if g.in_image:
                passes.append("E")
        win = window_of(cam, roi)
        crec: dict[str, Any] = {
            "kind": cam["kind"], "roi": list(roi) if roi else None, "window": list(win),
            "p_ecef_m": cam["p"].tolist(), "R_ecef_cam": cam["R"].tolist(),
            "intrinsics": {"fx": cam["model"].fx, "fy": cam["model"].fy, "cx": cam["model"].cx, "cy": cam["model"].cy,
                           "width": cam["model"].width, "height": cam["model"].height},
            "gt": {"u": g.u, "v": g.v, "z_m": g.z_m, "in_image": g.in_image, "box_analytic": g.box_analytic,
                   "limb_dv_px": g.limb_dv_px, "background": g.background},
            "gt_sphere": {"u": gs.u, "v": gs.v, "box_analytic": gs.box_analytic},
            "renders": {},
        }
        u0, v0 = roi if roi else (0, 0)
        a_grey = None
        for ps in passes:
            res, ms_set = gr.render(ps, cam, roi, None if ps == "E" else model_m)
            grey, impure, values = check_rgb(res.image, ps)
            rel = f"{img_prefix}/{inst['instant_id']}/{cid}_{ps}.png"
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(grey, mode="L").save(root / rel, format="PNG", compress_level=6)
            if ps == "A":
                a_grey = grey
            rr: dict[str, Any] = {
                "png": rel, "rgb_sha256": res.rgb_sha256, "n_renders": res.n_renders, "stable": res.stable,
                "incomplete": res.incomplete, "last_max_diff": res.last_max_diff, "last_n_diff": res.last_n_diff,
                "ms_set": ms_set, "ms_render": res.ms_render_loop, "ms_readback": res.ms_readback,
                "drawing_buffer": list(res.drawing_buffer), "impure_px": impure, "values": values[:16],
                "readback": res.readback,
            }
            if ps in ("A", "V", "S"):
                rr.update(mask_stats(grey > 127, u0, v0))
            if ps in ("A", "S"):
                mid = enc["model_id"] if ps == "A" else "SPHERE_R2"
                pw = p_i[None, :] + verts[mid] @ r_i.T
                pc = (pw - cam["p"][None, :]) @ cam["R"]
                m = cam["model"]
                cmp_ = raster.compare_with_mask(grey > 127, pc, faces[mid], m.fx, m.fy, m.cx, m.cy, u0, v0)
                if cmp_.get("ref_box_px"):
                    b = cmp_["ref_box_px"]
                    cmp_["ref_box_px"] = [b[0] + u0, b[1] + v0, b[2] + u0, b[3] + v0]
                rr["raster"] = cmp_
                abox = g.box_analytic if ps == "A" else gs.box_analytic
                if abox is not None and rr["px"] > 0:
                    ys, xs = np.nonzero(grey > 127)
                    xs = xs + u0
                    ys = ys + v0
                    tol = raster.TIE_PX
                    rr["outside_analytic_box_px"] = int(np.count_nonzero(
                        (xs < abox[0] - tol) | (xs > abox[2] + tol) | (ys < abox[1] - tol) | (ys > abox[3] + tol)))
                else:
                    rr["outside_analytic_box_px"] = 0 if rr["px"] == 0 else None
            if ps == "E":
                iu, iv = int(round(g.u)) - u0, int(round(g.v)) - v0
                rr["centre_px"] = [iu + u0, iv + v0]
                rr["centre_value"] = int(grey[iv, iu]) if (0 <= iu < 1024 and 0 <= iv < 1024) else None
            if ps == "V":
                rr["visible_outside_amodal_px"] = int(np.count_nonzero((grey > 127) & (a_grey <= 127)))
            crec["renders"][ps] = rr
        rec["cams"][cid] = crec
    rec["ms_instant"] = (time.perf_counter() - t_start) * 1000.0
    return rec


def evaluate(records: list[dict[str, Any]]) -> dict[str, Any]:
    gt1, gt2, gt3a, gt3b, gt4, purity, incompl = [], [], [], [], [], [], []
    gt2r, contain = [], []
    gt2_clipped_edges = 0
    for rec in records:
        for cid, c in rec["cams"].items():
            win = tuple(c["window"])
            rs = c["renders"]
            for ps, rr in rs.items():
                if rr["impure_px"]:
                    purity.append({"instant_id": rec["instant_id"], "cam": cid, "pass": ps, "impure_px": rr["impure_px"],
                                   "values": rr["values"]})
                if rr["incomplete"]:
                    incompl.append({"instant_id": rec["instant_id"], "cam": cid, "pass": ps, "n_renders": rr["n_renders"]})
            # GT-1
            if "S" in rs:
                s = rs["S"]
                sb = c["gt_sphere"]["box_analytic"]
                if s["px"] >= TH["gt1_min_area_px"] and box_inside(sb, win):
                    du = s["centroid"][0] - c["gt_sphere"]["u"]
                    dv = s["centroid"][1] - c["gt_sphere"]["v"]
                    gt1.append({"instant_id": rec["instant_id"], "cam": cid, "kind": c["kind"], "range_m": rec["target_range_m"],
                                "area_px": s["px"], "du": du, "dv": dv, "err_px": math.hypot(du, dv)})
            a = rs["A"]
            box = c["gt"]["box_analytic"]
            # GT-2 (spec v1.1): reference rasterization and box containment, A and S passes
            for ps in ("A", "S"):
                if ps not in rs:
                    continue
                rz = rs[ps]["raster"]
                gt2r.append({"instant_id": rec["instant_id"], "cam": cid, "pass": ps, "kind": c["kind"],
                             "class_id": rec["class_id"], "range_m": rec["target_range_m"], "gpu_px": rs[ps]["px"],
                             **rz})
                contain.append({"instant_id": rec["instant_id"], "cam": cid, "pass": ps,
                                "outside_analytic_box_px": rs[ps]["outside_analytic_box_px"]})
            # GT-2 (spec v1.0 criterion, reported)
            if a["px"] >= TH["gt2_min_area_px"] and box is not None:
                errs = []
                for k, (edge_an, edge_mk) in enumerate(zip(box, a["box_edges"])):
                    lo, hi = (win[0], win[2]) if k in (0, 2) else (win[1], win[3])
                    if lo + 1.0 <= edge_an <= hi - 1.0:
                        errs.append(edge_mk - edge_an)
                    else:
                        errs.append(None)
                        gt2_clipped_edges += 1
                valid = [abs(e) for e in errs if e is not None]
                gt2.append({"instant_id": rec["instant_id"], "cam": cid, "kind": c["kind"], "class_id": rec["class_id"],
                            "range_m": rec["target_range_m"], "area_px": a["px"], "box_analytic": box,
                            "box_edges": a["box_edges"], "edge_err_px": errs, "max_abs_err_px": max(valid) if valid else None})
            # GT-3(a)
            expected = (box is not None and (box[2] - box[0]) >= 1.0 and (box[3] - box[1]) >= 1.0 and box_meets(box, win))
            actual = a["px"] > 0
            if expected != actual:
                gt3a.append({"instant_id": rec["instant_id"], "cam": cid, "kind": c["kind"], "class_id": rec["class_id"],
                             "range_m": rec["target_range_m"], "expected_non_empty": expected, "amodal_px": a["px"],
                             "box_analytic": box,
                             "box_size_px": None if box is None else [box[2] - box[0], box[3] - box[1]]})
            # GT-3(b)
            if "V" in rs:
                gt3b.append({"instant_id": rec["instant_id"], "cam": cid, "visible_px": rs["V"]["px"], "amodal_px": a["px"],
                             "visible_outside_amodal_px": rs["V"]["visible_outside_amodal_px"]})
            # GT-4
            if "E" in rs and rs["E"]["centre_value"] is not None:
                val = rs["E"]["centre_value"]
                rendered = "earth" if val >= 128 else "sky"
                dv = c["gt"]["limb_dv_px"]
                near = dv is not None and abs(dv) <= TH["gt4_limb_band_px"]
                gt4.append({"instant_id": rec["instant_id"], "cam": cid, "kind": c["kind"], "value": val,
                            "rendered": rendered, "analytic": c["gt"]["background"], "limb_dv_px": dv,
                            "near_limb": near, "agree": rendered == c["gt"]["background"]})
    comp = [r for r in gt2r if r["computable"]]
    e1 = [r["err_px"] for r in gt1]
    e2 = [r["max_abs_err_px"] for r in gt2 if r["max_abs_err_px"] is not None]
    scored4 = [r for r in gt4 if not r["near_limb"]]
    out = {
        "gt1": {"n": len(e1), "median_px": float(np.median(e1)) if e1 else None, "max_px": float(np.max(e1)) if e1 else None,
                "pass": bool(e1) and float(np.median(e1)) <= TH["gt1_median_px"] and float(np.max(e1)) <= TH["gt1_max_px"],
                "records": gt1},
        "gt2": {"n": len(comp), "n_not_computable": len(gt2r) - len(comp),
                "n_near_clipped": sum(1 for r in comp if r.get("near_clipped")),
                "images_with_differences": sum(1 for r in comp if r["diff_px"] > 0),
                "diff_px": sum(r["diff_px"] for r in comp), "tie_px": sum(r["tie_px"] for r in comp),
                "mismatch_px": sum(r["mismatch_px"] for r in comp),
                "max_diff_edge_distance_px": max((r["max_diff_edge_distance_px"] for r in comp
                                                  if r["max_diff_edge_distance_px"] is not None), default=None),
                "outside_analytic_box_px": sum((r["outside_analytic_box_px"] or 0) for r in contain),
                "containment_unknown": sum(1 for r in contain if r["outside_analytic_box_px"] is None),
                "pass": (bool(comp) and len(comp) == len(gt2r) and all(r["mismatch_px"] == 0 for r in comp)
                         and all(r["outside_analytic_box_px"] == 0 for r in contain)),
                "records": gt2r},
        "gt2_v10_box": {"n": len(e2), "median_px": float(np.median(e2)) if e2 else None,
                        "max_px": float(np.max(e2)) if e2 else None,
                        "n_over_threshold": sum(1 for x in e2 if x > TH["gt2_edge_px"]),
                        "clipped_edges_not_compared": gt2_clipped_edges,
                        "pass_v10": bool(e2) and float(np.max(e2)) <= TH["gt2_edge_px"], "records": gt2},
        "gt3a": {"n_pairs": sum(len(r["cams"]) for r in records), "exceptions": len(gt3a), "records": gt3a},
        "gt3b": {"n": len(gt3b), "violations": sum(1 for r in gt3b if r["visible_outside_amodal_px"] > 0),
                 "violating_px": sum(r["visible_outside_amodal_px"] for r in gt3b),
                 "pass": all(r["visible_outside_amodal_px"] == 0 for r in gt3b) and bool(gt3b), "records": gt3b},
        "gt4": {"n": len(gt4), "n_scored": len(scored4), "n_near_limb": len(gt4) - len(scored4),
                "agree": sum(1 for r in scored4 if r["agree"]),
                "agreement_rate": (sum(1 for r in scored4 if r["agree"]) / len(scored4)) if scored4 else None,
                "disagreements": [r for r in scored4 if not r["agree"]], "records": gt4},
        "purity": {"impure_images": len(purity), "pass": not purity, "records": purity},
        "incomplete": {"n": len(incompl), "records": incompl},
    }
    return out


def perf_summary(records: list[dict[str, Any]], gr: G2Renderer | None, wall_s: float, backend: str) -> dict[str, Any]:
    def stats(v: list[float]) -> dict[str, float] | None:
        if not v:
            return None
        return {"n": len(v), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "mean": float(np.mean(v)),
                "max": float(np.max(v))}
    groups: dict[str, dict[str, list[float]]] = {}
    for rec in records:
        for c in rec["cams"].values():
            for ps, rr in c["renders"].items():
                key = f"{ps}_{c['kind']}"
                g = groups.setdefault(key, {"ms_render": [], "ms_readback": [], "n_renders": []})
                g["ms_render"].append(rr["ms_render"])
                g["ms_readback"].append(rr["ms_readback"])
                g["n_renders"].append(rr["n_renders"])
    return {
        "schema": "xvs.perf/1", "backend": backend, "n_instants": len(records), "wall_time_s": wall_s,
        "ms_per_instant": stats([r["ms_instant"] for r in records]),
        "images_per_instant": stats([float(sum(len(c["renders"]) for c in r["cams"].values())) for r in records]),
        "per_pass": {k: {m: stats(v) for m, v in g.items()} for k, g in sorted(groups.items())},
        "model_load_ms": stats(gr.model_load_ms) if gr else None,
        "passes": {"A": "mask_amodal + model", "V": "mask_visible + model", "S": "mask_amodal + SPHERE_R2",
                   "E": "mask_earth, no models"},
    }


def env_record(gr: G2Renderer, backend: str) -> dict[str, Any]:
    from importlib.metadata import version

    return {
        "backend": backend, "python": sys.version.split()[0], "platform": platform.platform(),
        "packages": {k: version(k) for k in ("numpy", "scipy", "pillow", "playwright", "pygltflib")},
        "browser_args": gr.r.get_browser_args(), "webgl": gr.webgl(), "git": git_state(),
        "resolved_settings": gr.resolved,
        "page_sha256": sha256_file(REPO_ROOT / "sim" / "cesium" / "page" / "xvs_page.js"),
        "profiles_sha256": sha256_file(REPO_ROOT / "configs" / "xvs" / "render_profiles.json"),
        "module_sha256": {m: sha256_file(REPO_ROOT / m) for m in MODULES},  # R-G2-9
    }


def load_vertices() -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, np.ndarray]]:
    """GLB meshes of the E-VAL-12 models and SPHERE_R2: vertices, file SHA-256 and triangles."""
    verts, shas, faces = {}, {}, {}
    items = [(cc["class_id"], cc["model_id"]) for cc in CLASS_CONFIGS] + [("CALIBRATION", "SPHERE_R2")]
    for sub, mid in items:
        p = REPO_ROOT / "assets" / "traffic" / sub / f"{mid}.glb"
        verts[mid], faces[mid] = raster.glb_mesh(p)
        shas[mid] = sha256_file(p)
    return verts, shas, faces


def run_main(backend: str, root: Path, run_gt5: bool, limit: int | None, only: set[str] | None = None) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rig = Rig.load_nominal().make_true(20260922)
    ring = TrafficRing()
    verts, shas, faces = load_vertices()
    encs = build_eval12()
    plan = instant_plan(encs, only=only)
    if limit:
        plan = plan[:limit]
    print(f"WP-02 G2 [{backend}]: {len(plan)} instants -> {root}")
    gr = G2Renderer(backend, root / backend)
    records = []
    try:
        env = env_record(gr, backend)
        for k, (enc, inst) in enumerate(plan):
            rec = process_instant(gr, rig, ring, enc, inst, verts, faces, root, f"{backend}/masks")
            records.append(rec)
            n_img = sum(len(c["renders"]) for c in rec["cams"].values())
            print(f"  [{k + 1:2d}/{len(plan)}] {inst['instant_id']:<26s} t={inst['t_s']:7.3f}s  cams={len(rec['cams']):2d}"
                  f"  images={n_img:2d}  {rec['ms_instant'] / 1000:6.1f}s  (elapsed {time.perf_counter() - t0:7.1f}s)", flush=True)
    finally:
        gr.close()
    wall = time.perf_counter() - t0
    ev = evaluate(records)
    results = {
        "schema": SCHEMA, "backend": backend, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "scripts/xvs/validate_traffic_g2.py", "script_sha256": sha256_file(Path(__file__)),
        "environment": env, "thresholds": TH,
        "eval12": {"t_cpa_s": T_CPA_S, "hmd_m": HMD_M, "side": SIDE, "vs_mps": VS_MPS, "ranges_m": TARGET_RANGES_M,
                   "classes": CLASS_CONFIGS, "backgrounds": BACKGROUND_CONFIGS, "model_sha256": shas,
                   "encounters": [{"encounter_id": e["encounter_id"], "class_id": e["class_id"], "model_id": e["model_id"],
                                   "bg": e["bg"], "p_cpa_ecef_m": e["traj"].p_cpa_ecef.tolist(),
                                   "v_ecef_mps": e["traj"].v_ecef.tolist(), "achieved_cpa": e["achieved_cpa"],
                                   "instants": e["instants"]} for e in encs]},
        "summary": {k: {kk: vv for kk, vv in v.items() if kk not in ("records",)} for k, v in ev.items()},
        "checks": ev,
        "records": records,
    }
    perf = perf_summary(records, gr, wall, backend)
    results["perf"] = perf
    res_path = root / f"results_{backend}.json"
    res_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    (root / f"perf_{backend}.json").write_text(json.dumps(perf, indent=1), encoding="utf-8")

    if run_gt5:
        print("GT-5: re-rendering the determinism instants in a fresh process ...", flush=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--backend", backend, "--out", str(root),
               "--gt5-replay", str(res_path)]
        t5 = time.perf_counter()
        cp = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        gt5_path = root / f"gt5_{backend}.json"
        if cp.returncode != 0 or not gt5_path.exists():
            results["gt5"] = {"pass": False, "error": (cp.stdout[-4000:] + "\n" + cp.stderr[-4000:])}
        else:
            results["gt5"] = json.loads(gt5_path.read_text(encoding="utf-8"))
            results["gt5"]["subprocess_wall_s"] = time.perf_counter() - t5
        results["summary"]["gt5"] = {k: v for k, v in results["gt5"].items() if k != "records"}
        res_path.write_text(json.dumps(results, indent=1), encoding="utf-8")

    s = results["summary"]
    verdict = {
        "GT-1": s["gt1"]["pass"], "GT-2 (v1.1 raster + containment)": s["gt2"]["pass"], "GT-3b": s["gt3b"]["pass"],
        "GT-4": s["gt4"]["agreement_rate"] == 1.0, "purity": s["purity"]["pass"],
        "no_incomplete": s["incomplete"]["n"] == 0,
        "GT-5": results.get("gt5", {}).get("pass") if run_gt5 else None,
    }
    results["verdict"] = verdict
    results["all_hard_checks_pass"] = all(v for v in verdict.values() if v is not None)
    res_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps({"summary": results["summary"], "verdict": verdict}, indent=1)[:6000])
    return results


def run_gt5_replay(backend: str, root: Path, main_results: Path) -> int:
    """Fresh process: re-render every image of the GT-5 instants and compare with the main run."""
    main = json.loads(main_results.read_text(encoding="utf-8"))
    rig = Rig.load_nominal().make_true(20260922)
    ring = TrafficRing()
    verts, _, faces = load_vertices()
    encs = build_eval12()
    wanted = set(GT5_INSTANTS)
    plan = instant_plan(encs, only=wanted)
    main_by_id = {r["instant_id"]: r for r in main["records"]}
    gr = G2Renderer(backend, root / f"{backend}_gt5_replay")
    rows = []
    try:
        for enc, inst in plan:
            if inst["instant_id"] not in main_by_id:
                continue
            rec = process_instant(gr, rig, ring, enc, inst, verts, faces, root, f"{backend}_gt5_replay/masks")
            ref_rec = main_by_id[inst["instant_id"]]
            for cid, c in ref_rec["cams"].items():
                c2 = rec["cams"].get(cid)
                for ps, rr in c["renders"].items():
                    rr2 = c2["renders"].get(ps) if c2 else None
                    row = {"instant_id": inst["instant_id"], "cam": cid, "pass": ps, "present": rr2 is not None}
                    if rr2 is not None:
                        a = np.array(Image.open(root / rr["png"]), dtype=np.int16)
                        b = np.array(Image.open(root / rr2["png"]), dtype=np.int16)
                        d = np.abs(a - b)
                        row.update({"sha_main": rr["rgb_sha256"], "sha_replay": rr2["rgb_sha256"],
                                    "sha_equal": rr["rgb_sha256"] == rr2["rgb_sha256"],
                                    "max_diff": int(d.max()), "n_diff": int(np.count_nonzero(d))})
                        if backend == "ref":
                            row["pass"] = row["sha_equal"]
                        else:
                            row["pass"] = (row["max_diff"] <= TH["gt5_hw_max_diff"]
                                           and row["n_diff"] <= TH["gt5_hw_n_diff_frac"] * d.size)
                    else:
                        row["pass"] = False
                    rows.append(row)
    finally:
        gr.close()
    res = {"backend": backend, "instants": list(GT5_INSTANTS), "n_images": len(rows),
           "n_pass": sum(1 for r in rows if r["pass"]), "n_sha_equal": sum(1 for r in rows if r.get("sha_equal")),
           "max_diff": max((r.get("max_diff", 0) for r in rows), default=None),  # default-ok: optional fallback default
           "pass": bool(rows) and all(r["pass"] for r in rows), "records": rows}
    (root / f"gt5_{backend}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"GT-5 replay [{backend}]: {res['n_pass']}/{res['n_images']} images pass")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="WP-02 G2 validation (rendered encounters, validated GT)")
    ap.add_argument("--backend", choices=["ref", "hw"], required=True)
    ap.add_argument("--out", default=str(REPO_ROOT / "output" / "xvs" / "wp02" / "g2"))
    ap.add_argument("--skip-gt5", action="store_true")
    ap.add_argument("--limit-instants", type=int, default=None, help="debug only: first N instants")
    ap.add_argument("--only", default=None, help="debug only: comma-separated instant ids")
    ap.add_argument("--gt5-replay", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    root = Path(a.out)
    if a.gt5_replay:
        return run_gt5_replay(a.backend, root, Path(a.gt5_replay))
    only = set(a.only.split(",")) if a.only else None
    res = run_main(a.backend, root, run_gt5=not a.skip_gt5, limit=a.limit_instants, only=only)
    return 0 if res["all_hard_checks_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
