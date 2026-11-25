"""Measure temporal persistence of D0 false positives on G3a v0.1 (Item 4a).

Loads the 180-encounter G3a v0.1 per-ROI cache (output/xvs/wp04/g0/raw_eval_cache.pkl.gz),
extracts exact (u, v, score, is_earth) false-positive coordinates for every ROI with fp_count > 0
under the degraded sensor (k=3, tau in {20.0, 30.0, 50.0, 75.0, 100.0}), verifies 100% count
parity against raw_eval_cache.pkl.gz, and measures the fraction of FPs that have a match within
2.0 px in the next 1, 3, and 5 frames after ego-motion compensation (both 6-DoF WGS84 terrain
ray-intersection + rotational ego-motion compensation and pure rotational attitude homography,
as well as uncompensated image coordinates).

Reports both:
  1. Lag-k match fraction at frame k + dk for dk in {1, 3, 5}.
  2. Contiguous persistence fraction across all frames k+1 .. k+dk for dk in {1, 3, 5}.
"""

from __future__ import annotations

import concurrent.futures
import gzip
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.detection.d0 import (
    compute_cmo,
    compute_local_stats,
    compute_z_score,
    extract_components,
    load_d0_config,
)
from xvs.geo import wgs84
from xvs.paths import get_data_root
from xvs.sensor.chain import SensorConfig
from xvs.sensor.noise import apply_sensor_noise, generate_fpn_maps
from xvs.sensor.psf import apply_psf
from xvs.sensor.srgb import REC709_B_WEIGHT, REC709_G_WEIGHT, REC709_R_WEIGHT, srgb_eotf
from xvs.sim.horizon import is_earth_at_pixels
from xvs.traffic.ring import TrafficRing

TAUS_TO_CHECK = (20.0, 30.0, 50.0, 75.0, 100.0)
MATCH_RADIUS_PX = 2.0
LAGS = (1, 3, 5)
TRACKED_FILES_PERSISTENCE = (
    "scripts/xvs/detection/measure_d0_clutter_persistence.py",
    "src/xvs/detection/d0.py",
    "src/xvs/sensor/chain.py",
    "src/xvs/sensor/noise.py",
    "src/xvs/sensor/psf.py",
    "src/xvs/sim/horizon.py",
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
    return proc.stdout.strip()


def _compute_code_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in TRACKED_FILES_PERSISTENCE:
        p = REPO_ROOT / rel
        if p.is_file():
            out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out

_W_RING: TrafficRing | None = None
_W_SCFG: SensorConfig | None = None
_W_KERNEL: np.ndarray | None = None
_W_PRNU: np.ndarray | None = None
_W_DSNU: np.ndarray | None = None
_W_D0_CFG: dict[str, Any] = {}
_W_G3A_ROOT: Path | None = None
_W_EOTF_R: np.ndarray | None = None
_W_EOTF_G: np.ndarray | None = None
_W_EOTF_B: np.ndarray | None = None


def _init_worker(g3a_root_str: str) -> None:
    global _W_RING, _W_SCFG, _W_KERNEL, _W_PRNU, _W_DSNU, _W_D0_CFG, _W_G3A_ROOT
    global _W_EOTF_R, _W_EOTF_G, _W_EOTF_B
    _W_RING = TrafficRing()
    scfg_path = REPO_ROOT / "configs" / "xvs" / "sensors" / "ring_mono_v0_nodist.json"
    _W_SCFG = SensorConfig.from_file(scfg_path)
    kernel, _, _ = _W_SCFG.get_psf_kernel()
    _W_KERNEL = kernel
    prnu, dsnu = generate_fpn_maps("ring_cam", 1024, 1024, _W_SCFG.prnu_rms, _W_SCFG.dsnu_rms_e)
    _W_PRNU = prnu
    _W_DSNU = dsnu
    _W_D0_CFG = load_d0_config()
    _W_G3A_ROOT = Path(g3a_root_str)
    lut = srgb_eotf(np.arange(256, dtype=np.uint8))
    _W_EOTF_R = lut * REC709_R_WEIGHT
    _W_EOTF_G = lut * REC709_G_WEIGHT
    _W_EOTF_B = lut * REC709_B_WEIGHT


def _extract_fps_for_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    g3a_root = _W_G3A_ROOT
    ring = _W_RING
    scfg = _W_SCFG
    kernel = _W_KERNEL
    prnu = _W_PRNU
    dsnu = _W_DSNU
    d0_cfg = _W_D0_CFG

    tile_size = int(d0_cfg["tile_size"])
    tile_stride = int(d0_cfg["tile_stride"])
    mad_scale = float(d0_cfg["mad_scale"])
    dilation_size = int(d0_cfg["dilation_kernel_size"])
    sigma_min_deg = float(d0_cfg["sigma_min_degraded"])

    out_list = []
    for item in batch:
        rgb_full = g3a_root / item["rgb_path"]
        with Image.open(rgb_full) as im:
            rgb_arr = np.array(im.convert("RGB"), dtype=np.uint8)

        lin_mono_clean = (
            _W_EOTF_R[rgb_arr[:, :, 0]]
            + _W_EOTF_G[rgb_arr[:, :, 1]]
            + _W_EOTF_B[rgb_arr[:, :, 2]]
        )
        blurred = apply_psf(lin_mono_clean, kernel)
        raw_dn = apply_sensor_noise(
            linear_radiance=blurred,
            prnu_map=prnu,
            dsnu_map=dsnu,
            E=scfg.full_well_e,
            full_well_e=scfg.full_well_e,
            sigma_r_e=scfg.sigma_r_e,
            dark_current_e_per_s=scfg.dark_current_e_per_s,
            exposure_time_s=scfg.exposure_time_s,
            k_dn_per_e=scfg.k_dn_per_e,
            bits=scfg.bits,
            black_level_dn=scfg.black_level_dn,
            rng_temporal=np.random.Generator(np.random.PCG64(item["k_frame"])),
        )
        lin_mono_deg = (raw_dn.astype(np.float64) - scfg.black_level_dn) / scfg.k_dn_per_e

        cmo_d = compute_cmo(lin_mono_deg, k=3)
        m_d, s_d = compute_local_stats(cmo_d, tile_size=tile_size, tile_stride=tile_stride, mad_scale=mad_scale)
        z_d = compute_z_score(cmo_d, m_d, s_d, sigma_min=sigma_min_deg)

        box_an = item.get("box_analytic")
        p_cam = np.asarray(item["p_cam_ecef"], dtype=np.float64)
        r_cam = np.asarray(item["R_cam_ecef"], dtype=np.float64)

        fps_by_tau: dict[str, list[tuple[float, float, float, bool]]] = {}
        for tau in TAUS_TO_CHECK:
            t_k = f"{tau:.1f}"
            if item["expected_fp_tot"][t_k] == 0:
                fps_by_tau[t_k] = []
                continue
            dets = extract_components(
                z_d,
                cmo_d,
                tau=tau,
                u0=item["roi_u0"],
                v0=item["roi_v0"],
                image_id=item["id"],
                camera=item["cam_id"],
                dilation_size=dilation_size,
            )
            outside = []
            for d in dets:
                du, dv, sc = float(d["u"]), float(d["v"]), float(d["score"])
                if box_an is not None:
                    if (box_an[0] - 16.0 <= du <= box_an[2] + 16.0) and (
                        box_an[1] - 16.0 <= dv <= box_an[3] + 16.0
                    ):
                        continue
                outside.append((du, dv, sc))
            if outside:
                u_arr = np.array([x[0] for x in outside], dtype=np.float64)
                v_arr = np.array([x[1] for x in outside], dtype=np.float64)
                is_e = is_earth_at_pixels(p_cam, r_cam, ring.camera_model, u_arr, v_arr)
                fps_by_tau[t_k] = [
                    (outside[i][0], outside[i][1], outside[i][2], bool(is_e[i]))
                    for i in range(len(outside))
                ]
            else:
                fps_by_tau[t_k] = []

        out_list.append({
            "id": item["id"],
            "fps_by_tau": fps_by_tau,
        })
    return out_list


def project_ego_compensated(
    u_arr: np.ndarray,
    v_arr: np.ndarray,
    p_cam1: np.ndarray,
    r_cam1: np.ndarray,
    p_cam2: np.ndarray,
    r_cam2: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    mode: str = "6dof_wgs84",
) -> tuple[np.ndarray, np.ndarray]:
    """Project pixels (u_arr, v_arr) from frame 1 into frame 2 after ego-motion compensation.

    Modes:
      - '6dof_wgs84': intersects Earth rays with the WGS84 ellipsoid from p_cam1 and reprojects
                      from p_cam2, R_cam2; sky rays use rotational compensation at infinity.
      - 'rotational': pure attitude homography R_cam2^T @ R_cam1 @ d_cam1 (infinity assumption).
      - 'uncompensated': raw pixel coordinates (u_arr, v_arr).
    """
    if len(u_arr) == 0 or mode == "uncompensated":
        return u_arr.copy(), v_arr.copy()

    d_cam1 = np.vstack([(u_arr - cx) / fx, (v_arr - cy) / fy, np.ones_like(u_arr)])  # (3, N)
    d_ecef = r_cam1 @ d_cam1  # (3, N)

    if mode == "rotational":
        d_cam2 = r_cam2.T @ d_ecef
        u2 = fx * (d_cam2[0] / d_cam2[2]) + cx
        v2 = fy * (d_cam2[1] / d_cam2[2]) + cy
        return u2, v2

    # 6dof_wgs84 mode
    a = wgs84.WGS84_A
    b = wgs84.WGS84_B
    s_vec = np.array([1.0 / a, 1.0 / a, 1.0 / b], dtype=np.float64)
    c_prime = p_cam1 * s_vec  # (3,)
    big_k = float(np.dot(c_prime, c_prime)) - 1.0

    d_prime = d_ecef * s_vec[:, None]  # (3, N)
    c_dot_d = c_prime @ d_prime  # (N,)
    d2 = np.sum(d_prime * d_prime, axis=0)  # (N,)
    disc = c_dot_d * c_dot_d - d2 * big_k
    hits_earth = (disc > 0.0) & (c_dot_d < 0.0)

    # Default to rotational vector at infinity
    v_cam2 = r_cam2.T @ d_ecef  # (3, N)

    if np.any(hits_earth):
        idx = np.where(hits_earth)[0]
        t_hit = (-c_dot_d[idx] - np.sqrt(np.maximum(0.0, disc[idx]))) / d2[idx]
        x_ecef = p_cam1[:, None] + d_ecef[:, idx] * t_hit[None, :]  # (3, M)
        rel_ecef = x_ecef - p_cam2[:, None]
        v_cam2[:, idx] = r_cam2.T @ rel_ecef

    u2 = fx * (v_cam2[0] / v_cam2[2]) + cx
    v2 = fy * (v_cam2[1] / v_cam2[2]) + cy
    return u2, v2


def _point_in_valid_roi(
    u: float,
    v: float,
    roi_u0: float,
    roi_v0: float,
    box_analytic: list[float] | None,
    margin: float = 2.0,
) -> bool:
    if not (roi_u0 + margin <= u <= roi_u0 + 1024.0 - margin):
        return False
    if not (roi_v0 + margin <= v <= roi_v0 + 1024.0 - margin):
        return False
    if box_analytic is not None:
        if (box_analytic[0] - 16.0 <= u <= box_analytic[2] + 16.0) and (
            box_analytic[1] - 16.0 <= v <= box_analytic[3] + 16.0
        ):
            return False
    return True


def main() -> int:
    t0 = time.perf_counter()
    git_status_start = _get_git_status_porcelain()
    if git_status_start:
        raise RuntimeError(f"Refusing to run measure_d0_clutter_persistence.py on dirty git working tree:\n{git_status_start}")

    start_commit = _get_git_commit()
    start_hashes = _compute_code_hashes()
    start_time_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    data_root = get_data_root()
    worktree_out = REPO_ROOT / "output"
    cache_cand = worktree_out / "xvs" / "wp04" / "g0" / "raw_eval_cache.pkl.gz"
    cache_path = cache_cand if cache_cand.is_file() else (data_root / "xvs" / "wp04" / "g0" / "raw_eval_cache.pkl.gz")
    g3a_cand = worktree_out / "xvs" / "wp02" / "g3a_v01"
    has_worktree_g3a = (g3a_cand / "blocks").is_dir() and bool(list((g3a_cand / "blocks").glob("b*/frames.jsonl")))
    g3a_root = g3a_cand if has_worktree_g3a else (data_root / "xvs" / "wp02" / "g3a_v01")
    # Write strictly to worktree output (F6)
    out_json = worktree_out / "xvs" / "wp04" / "g0" / "d0_clutter_persistence.json"
    cache_sha256 = hashlib.sha256(cache_path.read_bytes()).hexdigest()

    print(f"Loading per-ROI cache: {cache_path}")
    with gzip.open(cache_path, "rb") as f:
        cache_data = pickle.load(f)

    pos_results = cache_data["results_positive"]
    neg_results = cache_data["results_negative"]
    pos_by_id = {r["instance_id"]: r for r in pos_results}
    neg_by_id = {r["negative_id"]: r for r in neg_results}

    # Load frame metadata from g3a_v01 blocks to get p_cam_ecef, R_cam_ecef, roi_u0, roi_v0, rgb_path
    print(f"Loading ROI metadata from {g3a_root / 'blocks'}...")
    roi_meta: dict[str, dict[str, Any]] = {}
    for fr_file in sorted((g3a_root / "blocks").glob("b*/frames.jsonl")):
        with open(fr_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                obj = json.loads(s)
                enc = obj["encounter_id"]
                kf = int(obj["k"])
                t_s = float(obj["t_s"])

                for tr in obj.get("traffic", []):
                    for cid, cdata in tr.get("cams", {}).items():
                        if cdata.get("kind") == "ring" and cdata.get("in_image") and cdata.get("rgb") and cdata.get("box_analytic") is not None:
                            inst_id = f"{enc}_{kf:06d}_{cid}"
                            if inst_id in pos_by_id:
                                cinfo = obj["cameras"][cid]
                                ev_deg3 = pos_by_id[inst_id]["eval_by_sensor_k"]["degraded"][3]
                                exp_tot = {
                                    f"{t:.1f}": int(ev_deg3["fp_counts_sky"][f"{t:.1f}"] + ev_deg3["fp_counts_earth"][f"{t:.1f}"])
                                    for t in TAUS_TO_CHECK
                                }
                                roi_meta[inst_id] = {
                                    "id": inst_id,
                                    "kind": "positive",
                                    "encounter_id": enc,
                                    "k_frame": kf,
                                    "t_s": t_s,
                                    "cam_id": cid,
                                    "roi_u0": float(cdata["roi"][0]),
                                    "roi_v0": float(cdata["roi"][1]),
                                    "box_analytic": [float(x) for x in cdata["box_analytic"]],
                                    "rgb_path": cdata["rgb"],
                                    "p_cam_ecef": np.array(cinfo["p_ecef_m"], dtype=np.float64),
                                    "R_cam_ecef": np.array(cinfo["R_ecef_cam"], dtype=np.float64),
                                    "expected_fp_sky": {f"{t:.1f}": int(ev_deg3["fp_counts_sky"][f"{t:.1f}"]) for t in TAUS_TO_CHECK},
                                    "expected_fp_earth": {f"{t:.1f}": int(ev_deg3["fp_counts_earth"][f"{t:.1f}"]) for t in TAUS_TO_CHECK},
                                    "expected_fp_tot": exp_tot,
                                }

                neg = obj.get("negative")
                if neg and neg.get("png"):
                    cid = neg["cam"]
                    neg_id = f"{enc}_{kf:06d}_{cid}_neg"
                    if neg_id in neg_by_id:
                        cinfo = obj["cameras"][cid]
                        ev_deg3 = neg_by_id[neg_id]["eval_by_sensor_k"]["degraded"][3]
                        exp_tot = {
                            f"{t:.1f}": int(ev_deg3["fp_counts_sky"][f"{t:.1f}"] + ev_deg3["fp_counts_earth"][f"{t:.1f}"])
                            for t in TAUS_TO_CHECK
                        }
                        roi_meta[neg_id] = {
                            "id": neg_id,
                            "kind": "negative",
                            "encounter_id": enc,
                            "k_frame": kf,
                            "t_s": t_s,
                            "cam_id": cid,
                            "roi_u0": float(neg["roi"][0]),
                            "roi_v0": float(neg["roi"][1]),
                            "box_analytic": None,
                            "rgb_path": neg["png"],
                            "p_cam_ecef": np.array(cinfo["p_ecef_m"], dtype=np.float64),
                            "R_cam_ecef": np.array(cinfo["R_ecef_cam"], dtype=np.float64),
                            "expected_fp_sky": {f"{t:.1f}": int(ev_deg3["fp_counts_sky"][f"{t:.1f}"]) for t in TAUS_TO_CHECK},
                            "expected_fp_earth": {f"{t:.1f}": int(ev_deg3["fp_counts_earth"][f"{t:.1f}"]) for t in TAUS_TO_CHECK},
                            "expected_fp_tot": exp_tot,
                        }

    print(f"Matched {len(roi_meta):,} ROIs from frames.jsonl (expected {len(pos_results) + len(neg_results):,}).")
    assert len(roi_meta) == len(pos_results) + len(neg_results), "ROI count mismatch against cache!"

    # Filter ROIs that have at least 1 FP at tau=20.0
    tasks_to_run = [m for m in roi_meta.values() if m["expected_fp_tot"]["20.0"] > 0]
    for m in roi_meta.values():
        m["fps_by_tau"] = {f"{t:.1f}": [] for t in TAUS_TO_CHECK}

    workers = min(8, max(1, (os.cpu_count() or 4) - 4))
    batch_size = 50
    batches = [tasks_to_run[i : i + batch_size] for i in range(0, len(tasks_to_run), batch_size)]
    print(f"Extracting FP coordinates for {len(tasks_to_run):,} ROIs with FPs across {len(batches)} batches on {workers} workers...")

    t_ext = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(g3a_root),),
    ) as ex:
        futs = [ex.submit(_extract_fps_for_batch, b) for b in batches]
        done_rois = 0
        for fut in concurrent.futures.as_completed(futs):
            res_batch = fut.result()
            for r in res_batch:
                roi_meta[r["id"]]["fps_by_tau"] = r["fps_by_tau"]
            done_rois += len(res_batch)
            if done_rois % 1000 == 0 or done_rois == len(tasks_to_run):
                print(f"  [Progress] {done_rois:,}/{len(tasks_to_run):,} ROIs ({time.perf_counter() - t_ext:.1f}s)")

    # Verify 100% parity with raw_eval_cache.pkl.gz counts
    mismatches = 0
    for m in roi_meta.values():
        for tau in TAUS_TO_CHECK:
            tk = f"{tau:.1f}"
            fps = m["fps_by_tau"][tk]
            n_sky = sum(1 for x in fps if not x[3])
            n_earth = sum(1 for x in fps if x[3])
            if n_sky != m["expected_fp_sky"][tk] or n_earth != m["expected_fp_earth"][tk]:
                mismatches += 1
    print(f"Parity check against raw_eval_cache.pkl.gz: {mismatches} mismatches across {len(roi_meta):,} ROIs.")
    if mismatches != 0:
        raise RuntimeError(f"Cache parity check failed with {mismatches} mismatches!")

    # Index positive ROIs by (encounter_id, cam_id, k_frame) for temporal persistence analysis
    ring = TrafficRing()
    fx, fy, cx, cy = ring.fx, ring.fy, ring.cx, ring.cy

    pos_by_seq: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for m in roi_meta.values():
        if m["kind"] == "positive":
            pos_by_seq[(m["encounter_id"], m["cam_id"])][m["k_frame"]] = m

    def evaluate_persistence(tau_val: float, comp_mode: str) -> dict[str, Any]:
        tk = f"{tau_val:.1f}"
        stats: dict[str, Any] = {"lag_match": {}, "contiguous_persistence": {}}

        for bg_filter in ("all", "earth", "sky"):
            stats["lag_match"][bg_filter] = {}
            stats["contiguous_persistence"][bg_filter] = {}

            for lag in LAGS:
                # 1. Pairwise lag match at k + lag
                n_denom_in_roi = 0
                n_match_in_roi = 0
                n_denom_all_seq = 0
                n_match_all_seq = 0
                du_matches_in_roi: list[float] = []
                dv_matches_in_roi: list[float] = []
                d_matches_in_roi: list[float] = []

                # 2. Contiguous persistence across all frames k+1 .. k+lag
                n_cont_denom_in_roi = 0
                n_cont_match_in_roi = 0

                for (enc, cid), frames_map in pos_by_seq.items():
                    for kf, m_curr in frames_map.items():
                        fps_curr = m_curr["fps_by_tau"][tk]
                        if bg_filter == "earth":
                            fps_curr = [x for x in fps_curr if x[3]]
                        elif bg_filter == "sky":
                            fps_curr = [x for x in fps_curr if not x[3]]
                        if not fps_curr:
                            continue

                        u1 = np.array([x[0] for x in fps_curr], dtype=np.float64)
                        v1 = np.array([x[1] for x in fps_curr], dtype=np.float64)

                        # Check pairwise lag match at kf + lag
                        m_target = frames_map.get(kf + lag)
                        if m_target is not None:
                            u_proj, v_proj = project_ego_compensated(
                                u1, v1,
                                m_curr["p_cam_ecef"], m_curr["R_cam_ecef"],
                                m_target["p_cam_ecef"], m_target["R_cam_ecef"],
                                fx, fy, cx, cy, mode=comp_mode,
                            )
                            fps_next = m_target["fps_by_tau"][tk]
                            u2 = np.array([x[0] for x in fps_next], dtype=np.float64) if fps_next else np.empty(0)
                            v2 = np.array([x[1] for x in fps_next], dtype=np.float64) if fps_next else np.empty(0)

                            for idx_fp in range(len(fps_curr)):
                                up, vp = float(u_proj[idx_fp]), float(v_proj[idx_fp])
                                n_denom_all_seq += 1
                                has_hit = False
                                min_d = float("inf")
                                min_idx = -1
                                if len(u2) > 0:
                                    dists = np.hypot(u2 - up, v2 - vp)
                                    min_idx = int(np.argmin(dists))
                                    min_d = float(dists[min_idx])
                                    if min_d <= MATCH_RADIUS_PX:
                                        has_hit = True
                                if has_hit:
                                    n_match_all_seq += 1

                                if _point_in_valid_roi(up, vp, m_target["roi_u0"], m_target["roi_v0"], m_target["box_analytic"]):
                                    n_denom_in_roi += 1
                                    if has_hit and min_idx >= 0:
                                        n_match_in_roi += 1
                                        du_matches_in_roi.append(float(u2[min_idx] - up))
                                        dv_matches_in_roi.append(float(v2[min_idx] - vp))
                                        d_matches_in_roi.append(min_d)

                        # Check contiguous persistence across kf+1 .. kf+lag
                        if all((kf + s) in frames_map for s in range(1, lag + 1)):
                            # Project into each step 1..lag
                            in_roi_all = np.ones(len(fps_curr), dtype=bool)
                            matched_all = np.ones(len(fps_curr), dtype=bool)
                            for s in range(1, lag + 1):
                                m_step = frames_map[kf + s]
                                up_s, vp_s = project_ego_compensated(
                                    u1, v1,
                                    m_curr["p_cam_ecef"], m_curr["R_cam_ecef"],
                                    m_step["p_cam_ecef"], m_step["R_cam_ecef"],
                                    fx, fy, cx, cy, mode=comp_mode,
                                )
                                fps_step = m_step["fps_by_tau"][tk]
                                u_s = np.array([x[0] for x in fps_step], dtype=np.float64) if fps_step else np.empty(0)
                                v_s = np.array([x[1] for x in fps_step], dtype=np.float64) if fps_step else np.empty(0)

                                for idx_fp in range(len(fps_curr)):
                                    up, vp = float(up_s[idx_fp]), float(vp_s[idx_fp])
                                    if not _point_in_valid_roi(up, vp, m_step["roi_u0"], m_step["roi_v0"], m_step["box_analytic"]):
                                        in_roi_all[idx_fp] = False
                                    hit_s = False
                                    if len(u_s) > 0:
                                        if float(np.min(np.hypot(u_s - up, v_s - vp))) <= MATCH_RADIUS_PX:
                                            hit_s = True
                                    if not hit_s:
                                        matched_all[idx_fp] = False

                            for idx_fp in range(len(fps_curr)):
                                if in_roi_all[idx_fp]:
                                    n_cont_denom_in_roi += 1
                                    if matched_all[idx_fp]:
                                        n_cont_match_in_roi += 1

                disp_1d_std = (
                    float(np.std(np.concatenate([du_matches_in_roi, dv_matches_in_roi])))
                    if du_matches_in_roi else 0.0
                )
                disp_radial_rms = (
                    float(np.sqrt(np.mean(np.square(d_matches_in_roi))))
                    if d_matches_in_roi else 0.0
                )
                stats["lag_match"][bg_filter][f"lag_{lag}"] = {
                    "lag_frames": lag,
                    "k_matched_in_roi": n_match_in_roi,
                    "n_evaluable_in_roi": n_denom_in_roi,
                    "fraction_in_roi": float(n_match_in_roi / n_denom_in_roi) if n_denom_in_roi > 0 else 0.0,
                    "k_matched_all_seq": n_match_all_seq,
                    "n_evaluable_all_seq": n_denom_all_seq,
                    "fraction_all_seq": float(n_match_all_seq / n_denom_all_seq) if n_denom_all_seq > 0 else 0.0,
                    "matched_displacement_std_px": disp_1d_std,
                    "matched_radial_rms_px": disp_radial_rms,
                }
                stats["contiguous_persistence"][bg_filter][f"persist_{lag}"] = {
                    "consecutive_frames": lag,
                    "k_persisted_in_roi": n_cont_match_in_roi,
                    "n_evaluable_in_roi": n_cont_denom_in_roi,
                    "fraction_in_roi": float(n_cont_match_in_roi / n_cont_denom_in_roi) if n_cont_denom_in_roi > 0 else 0.0,
                }

        return stats

    print("\nComputing ego-motion-compensated temporal persistence statistics...")
    results_by_tau: dict[str, Any] = {}
    for tau in (20.0, 30.0, 50.0, 75.0):
        tk = f"{tau:.1f}"
        results_by_tau[tk] = {
            "6dof_wgs84": evaluate_persistence(tau, "6dof_wgs84"),
            "rotational": evaluate_persistence(tau, "rotational"),
            "uncompensated": evaluate_persistence(tau, "uncompensated"),
        }

    # Check the 10% threshold criterion at closure operating point (k=3, tau=20.0)
    p1_lag_6dof = results_by_tau["20.0"]["6dof_wgs84"]["lag_match"]["all"]["lag_1"]["fraction_in_roi"]
    p3_lag_6dof = results_by_tau["20.0"]["6dof_wgs84"]["lag_match"]["all"]["lag_3"]["fraction_in_roi"]
    p3_cont_6dof = results_by_tau["20.0"]["6dof_wgs84"]["contiguous_persistence"]["all"]["persist_3"]["fraction_in_roi"]
    p3_lag_earth = results_by_tau["20.0"]["6dof_wgs84"]["lag_match"]["earth"]["lag_3"]["fraction_in_roi"]
    p3_cont_earth = results_by_tau["20.0"]["6dof_wgs84"]["contiguous_persistence"]["earth"]["persist_3"]["fraction_in_roi"]
    jitter_6dof_px = results_by_tau["20.0"]["6dof_wgs84"]["lag_match"]["all"]["lag_1"]["matched_displacement_std_px"]
    p_stay_6dof = math.sqrt(p3_cont_6dof / p1_lag_6dof) if p1_lag_6dof > 0.0 else 0.0

    exceeds_10pct = max(p3_lag_6dof, p3_cont_6dof, p3_lag_earth, p3_cont_earth) > 0.10

    end_commit = _get_git_commit()
    end_hashes = _compute_code_hashes()
    git_status_end = _get_git_status_porcelain()
    end_time_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    hashes_unchanged = (start_commit == end_commit) and (start_hashes == end_hashes)
    if not hashes_unchanged:
        raise RuntimeError("Code hashes or git commit changed during measure_d0_clutter_persistence.py execution!")

    d0_cfg_path = REPO_ROOT / "configs" / "xvs" / "detection" / "d0.json"
    scfg_path = REPO_ROOT / "configs" / "xvs" / "sensors" / "ring_mono_v0_nodist.json"

    payload = {
        "gate": "WP-04 G0 / WP-05 G0 Item 4a Clutter Persistence",
        "timestamp_utc": end_time_utc,
        "cache_source": str(cache_path),
        "cache_sha256": cache_sha256,
        "g3a_root": str(g3a_root),
        "g3a_frame_rate_hz": 1.0,
        "release_json_sha256": cache_data.get("release_json_sha256"),
        "detector_config": {"sensor": "degraded", "k": 3, "closure_tau": 20.0},
        "match_radius_px": MATCH_RADIUS_PX,
        "cache_parity_mismatches": mismatches,
        "n_positive_rois": len(pos_results),
        "n_negative_rois": len(neg_results),
        "results_by_tau": results_by_tau,
        "markov_parameters_tau_20_6dof_all": {
            "grid_rate_hz": 1.0,
            "clutter_p1": float(p1_lag_6dof),
            "clutter_persist_3": float(p3_cont_6dof),
            "clutter_p_stay": float(p_stay_6dof),
            "clutter_mean_len_1s": float(1.0 + p1_lag_6dof / max(1e-12, 1.0 - p_stay_6dof)),
            "matched_displacement_std_px": float(jitter_6dof_px),
            "sensitivity_10hz_p1_step": float(p1_lag_6dof ** 0.1),
            "sensitivity_10hz_p_stay_step": float(p_stay_6dof ** 0.1),
        },
        "criterion_10pct_3frames": {
            "threshold": 0.10,
            "tau_20_6dof_all_lag3_fraction": p3_lag_6dof,
            "tau_20_6dof_all_contiguous3_fraction": p3_cont_6dof,
            "tau_20_6dof_earth_lag3_fraction": p3_lag_earth,
            "tau_20_6dof_earth_contiguous3_fraction": p3_cont_earth,
            "exceeds_10pct": bool(exceeds_10pct),
            "rerun_3c_required": bool(exceeds_10pct),
        },
        "provenance": {
            "argv": list(sys.argv),
            "start_time_utc": start_time_utc,
            "end_time_utc": end_time_utc,
            "git_commit_start": start_commit,
            "git_commit_end": end_commit,
            "git_status_start": git_status_start,
            "git_status_end": git_status_end,
            "code_hashes_start": start_hashes,
            "code_hashes_end": end_hashes,
            "code_hashes_unchanged_during_run": hashes_unchanged,
            "configs_read": {
                "configs/xvs/detection/d0.json": hashlib.sha256(d0_cfg_path.read_bytes()).hexdigest(),
                "configs/xvs/sensors/ring_mono_v0_nodist.json": hashlib.sha256(scfg_path.read_bytes()).hexdigest(),
            },
        },
        "wall_time_s": time.perf_counter() - t0,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved clutter persistence results to {out_json} in {payload['wall_time_s']:.1f}s.")

    # Print summary table for tau=20.0
    print("\n=== D0 False Positive Temporal Persistence at Closure Point (k=3, tau=20.0, r <= 2.0 px) ===")
    for mode in ("6dof_wgs84", "rotational", "uncompensated"):
        print(f"\nMode: {mode}")
        for bg in ("all", "earth", "sky"):
            lm = results_by_tau["20.0"][mode]["lag_match"][bg]
            cp = results_by_tau["20.0"][mode]["contiguous_persistence"][bg]
            print(
                f"  [{bg:5s}] Lag match (+1, +3, +5): "
                f"+1={lm['lag_1']['fraction_in_roi']*100:.2f}% ({lm['lag_1']['k_matched_in_roi']}/{lm['lag_1']['n_evaluable_in_roi']}), "
                f"+3={lm['lag_3']['fraction_in_roi']*100:.2f}% ({lm['lag_3']['k_matched_in_roi']}/{lm['lag_3']['n_evaluable_in_roi']}), "
                f"+5={lm['lag_5']['fraction_in_roi']*100:.2f}% ({lm['lag_5']['k_matched_in_roi']}/{lm['lag_5']['n_evaluable_in_roi']}) | "
                f"Contiguous (>=1, >=3, >=5): "
                f">=1={cp['persist_1']['fraction_in_roi']*100:.2f}%, "
                f">=3={cp['persist_3']['fraction_in_roi']*100:.2f}% ({cp['persist_3']['k_persisted_in_roi']}/{cp['persist_3']['n_evaluable_in_roi']}), "
                f">=5={cp['persist_5']['fraction_in_roi']*100:.2f}% ({cp['persist_5']['k_persisted_in_roi']}/{cp['persist_5']['n_evaluable_in_roi']})"
            )
    print(f"\nExceeds 10% at >=3 frames? {exceeds_10pct}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
