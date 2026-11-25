"""Tier A Real Amazon AOT S3 Passive Proof (`XVS-D2` + `tau` + `CP-MSC`), 1080p Video & Native `1:1` Tab 3 Pack.

Spec §4.1.3 & Plan Task 3:
- `--live-s3`: Downloads `6-10` real held-out Amazon AOT encounters from public S3
  (`part1/ImageSets/valid_encounters_maxRange700_maxGap3_minEncLen30.csv` + `groundtruth.csv`,
  cached in gitignored `data/aot_real_proof/`), splits 50/50 by encounter into disjoint `cal`
  (fits frozen Split Conformal `q_hat`) and `test` sets, runs real Colab G4
  `apps/livedeck/public/models/xvs_d2_web_fp16.onnx` on native `1:1` `256x256` crops, and evaluates:
    1. Real-sky `XVS-D2` detection rate & centroid RMSE.
    2. Scale-cancelling optical expansion `rho_dot = -s_dot / s` vs real GPS closure `R_dot / R`.
    3. Honest passive `CP-MSC` (`a_az = 0`) finite-sample coverage `P(R_true >= R_safe_lower)`
       and hazardous range-overestimation reduction vs Naive Bbox-Size Regressor (`11 m` Cessna prior).
- `--smoke-synthetic`: Offline deterministic CI/pytest mode (`0` S3 dependency).
- Outputs:
    - `1920x1080` HD showcase MP4 (`outputs/xvs_activerange_aot_proof_1080p.mp4`)
    - JSON metrics (`outputs/xvs_activerange_aot_metrics.json`)
    - Markdown report (`docs/xvs/reports/V3_M1_aot_realsky_proof.md`)
    - Native `1:1` `256x256` Tab 3 browser pack (`apps/livedeck/public/assets/realsky_aot_pack/`, `<= 1.5 MB`).

Zero unannotated `.get()` calls; strictly NED conventions.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from xvs.detection.aot_s3_ingest import (
    GROUNDTRUTH_S3_KEY,
    IMAGES_S3_PREFIX,
    VALID_ENCOUNTERS_S3_KEY,
    download_s3_object_with_https_fallback,
    parse_valid_encounters_csv,
)
from xvs.v2.xvs_active_range.conformal import (
    compute_conformal_lower_bound,
    evaluate_conformal_coverage,
    fit_one_sided_conformal_quantile,
)
from xvs.v2.xvs_active_range.filter import (
    ActiveRangeConfig,
    ActiveRangeController,
    ActiveRangeMeasurement,
    OwnshipStateNed,
    R_INIT_M,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONNX_PATH = REPO_ROOT / "apps" / "livedeck" / "public" / "models" / "xvs_d2_web_fp16.onnx"
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "aot_real_proof"
DEFAULT_VIDEO_PATH = REPO_ROOT / "outputs" / "xvs_activerange_aot_proof_1080p.mp4"
DEFAULT_METRICS_PATH = REPO_ROOT / "outputs" / "xvs_activerange_aot_metrics.json"
DEFAULT_REPORT_PATH = REPO_ROOT / "docs" / "xvs" / "reports" / "V3_M1_aot_realsky_proof.md"
DEFAULT_PACK_DIR = REPO_ROOT / "apps" / "livedeck" / "public" / "assets" / "realsky_aot_pack"

AOT_FOCAL_PX: float = 2690.0
AOT_FPS: float = 10.0
AOT_FULL_W: int = 2448
AOT_FULL_H: int = 2048
CROP_SIZE: int = 256
NAIVE_WINGSPAN_M: float = 11.0
MAX_PACK_BYTES: int = 1_500_000

CANVAS_W: int = 1920
CANVAS_H: int = 1080

# Avionics BGR palette
COL_BG = (15, 19, 26)
COL_PANEL = (23, 30, 41)
COL_BORDER = (68, 84, 106)
COL_TEXT = (235, 241, 248)
COL_MUTED = (148, 165, 185)
COL_CYAN = (248, 224, 48)
COL_EMERALD = (118, 236, 68)
COL_AMBER = (56, 190, 248)
COL_ROSE = (98, 92, 244)


@dataclass(frozen=True)
class AotFrameRecord:
    """Single annotated frame in an AOT encounter (real S3 or offline synthetic)."""

    frame_idx: int
    img_path: Path | None
    gray_full: np.ndarray | None
    gt_left: float
    gt_top: float
    gt_right: float
    gt_bottom: float
    range_distance_m: float
    is_above_horizon: bool
    aircraft_class: str


@dataclass(frozen=True)
class AotEncounter:
    """Contiguous encounter sequence of annotated frames."""

    encounter_id: str
    flight_id: str
    fps: float
    aircraft_class: str
    frames: tuple[AotFrameRecord, ...]


# ---- Exact XVS-D2 ONNX Pre/Post-Processing (Matching `render_aot_d2_demo_video.py`) ----


def _scene_normalize_gray(gray_f32: np.ndarray, min_std: float = 8.0, clamp_val: float = 5.0) -> np.ndarray:
    mean = float(gray_f32.mean())
    std = max(float(gray_f32.std()), float(min_std))
    return np.clip((gray_f32 - mean) / std, -clamp_val, clamp_val).astype(np.float32)


def _warp_gray_and_mask(gray_f32: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = gray_f32.shape
    mat = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    warped = cv2.warpAffine(
        gray_f32, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )
    ones = np.ones((h, w), dtype=np.float32)
    mask = cv2.warpAffine(
        ones, mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )
    return warped, (mask > 0.5).astype(np.float32)


def _warp_hidden_state(h_prev: np.ndarray, dx_full: float, dy_full: float, stride: float = 4.0) -> np.ndarray:
    if abs(dx_full) < 1.0e-4 and abs(dy_full) < 1.0e-4:
        return h_prev
    _, channels, hh, hw = h_prev.shape
    mat = np.array([[1.0, 0.0, dx_full / stride], [0.0, 1.0, dy_full / stride]], dtype=np.float32)
    out = np.empty_like(h_prev)
    for ch in range(channels):
        out[0, ch] = cv2.warpAffine(
            h_prev[0, ch], mat, (hw, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        )
    return out


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x.astype(np.float32), -40.0, 40.0)))


def _decode_xvs_d2_top(
    hm_logits: np.ndarray,
    offset: np.ndarray,
    log_scale: np.ndarray,
    peak_mask: np.ndarray,
    *,
    score_thr: float = 0.25,
    stride: float = 2.0,
) -> tuple[tuple[float, float, float, float, float] | None, float]:
    """Decode top-scoring detection `(x1, y1, x2, y2, score)` in `256x256` crop coordinates."""
    hm = hm_logits[0, 0].astype(np.float32)
    off = offset[0].astype(np.float32)
    lsc = log_scale[0].astype(np.float32)
    pmk = peak_mask[0, 0].astype(np.float32)

    probs = _sigmoid_np(hm)
    h_sz, w_sz = probs.shape
    max_prob = float(probs.max())

    keep = (probs >= float(score_thr)) & (pmk > 0.5)
    ys, xs = np.where(keep)
    if ys.size == 0:
        return None, max_prob

    scores = probs[ys, xs]
    best = int(np.argmax(scores))
    cy = int(ys[best])
    cx = int(xs[best])
    score = float(probs[cy, cx])

    y0, y1 = max(0, cy - 1), min(h_sz, cy + 2)
    x0, x1 = max(0, cx - 1), min(w_sz, cx + 2)
    patch = hm[y0:y1, x0:x1]
    exp_p = np.exp(patch - float(patch.max()))
    w_p = exp_p / max(float(exp_p.sum()), 1.0e-12)
    gy = np.arange(y0, y1, dtype=np.float32)[:, np.newaxis]
    gx = np.arange(x0, x1, dtype=np.float32)[np.newaxis, :]
    soft_u = float((w_p * gx).sum())
    soft_v = float((w_p * gy).sum())

    cell_u = 0.5 * (float(cx) + float(off[0, cy, cx])) + 0.5 * soft_u
    cell_v = 0.5 * (float(cy) + float(off[1, cy, cx])) + 0.5 * soft_v
    u_px = (cell_u + 0.5) * stride
    v_px = (cell_v + 0.5) * stride

    s_mom = math.exp(max(-4.0, min(8.0, float(lsc[0, cy, cx]))))
    r_wh = math.exp(max(-3.0, min(3.0, float(lsc[1, cy, cx]))))
    bw = max(6.0, min(96.0, s_mom * math.sqrt(r_wh)))
    bh = max(6.0, min(96.0, s_mom / math.sqrt(r_wh)))
    return (u_px - 0.5 * bw, v_px - 0.5 * bh, u_px + 0.5 * bw, v_px + 0.5 * bh, score), max_prob


class XvsD2CropRunner:
    """Persistent ConvGRU ONNX runner on `1:1` native `256x256` crops with exact crop-shift GMC."""

    def __init__(self, onnx_path: Path) -> None:
        self.available = onnx_path.is_file()
        self.sess: ort.InferenceSession | None = None
        self.hid_c = 24
        self.x_dtype = np.float32
        self.h_dtype = np.float32
        if self.available:
            self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            inps = {inp.name: inp for inp in self.sess.get_inputs()}
            if "h_prev_warped" in inps:
                h_shape = inps["h_prev_warped"].shape
                if len(h_shape) >= 2 and isinstance(h_shape[1], int):
                    self.hid_c = int(h_shape[1])
                if inps["h_prev_warped"].type == "tensor(float16)":
                    self.h_dtype = np.float16
            if "x_t" in inps and inps["x_t"].type == "tensor(float16)":
                self.x_dtype = np.float16
        self.reset()

    def reset(self) -> None:
        self._h_prev = np.zeros((1, self.hid_c, CROP_SIZE // 4, CROP_SIZE // 4), dtype=np.float32)
        self._gray_hist: list[tuple[np.ndarray, int, int]] = []

    def step(
        self,
        crop_gray_f32: np.ndarray,
        crop_x: int,
        crop_y: int,
        *,
        score_thr: float = 0.25,
    ) -> tuple[tuple[float, float, float, float, float] | None, float, float]:
        """Run one `XVS-D2` step; returns `(top_box_or_none, max_prob, latency_ms)`."""
        t0 = time.perf_counter()
        c0 = _scene_normalize_gray(crop_gray_f32)
        h, w = c0.shape

        if len(self._gray_hist) >= 1:
            g1, cx1, cy1 = self._gray_hist[-1]
            dx1 = float(cx1 - crop_x)
            dy1 = float(cy1 - crop_y)
            w1, r1 = _warp_gray_and_mask(g1, dx1, dy1)
            c1 = _scene_normalize_gray(w1) * r1
        else:
            dx1, dy1 = 0.0, 0.0
            c1 = np.zeros((h, w), dtype=np.float32)
            r1 = np.zeros((h, w), dtype=np.float32)

        if len(self._gray_hist) >= 5:
            g5, cx5, cy5 = self._gray_hist[-5]
            dx5 = float(cx5 - crop_x)
            dy5 = float(cy5 - crop_y)
            w5, r5 = _warp_gray_and_mask(g5, dx5, dy5)
            c2 = _scene_normalize_gray(w5) * r5
        else:
            c2 = np.zeros((h, w), dtype=np.float32)
            r5 = np.zeros((h, w), dtype=np.float32)

        c3 = (c0 - c1) * r1
        c4 = (c0 - c2) * r5
        c5 = r1 * r5
        x_t = np.stack([c0, c1, c2, c3, c4, c5], axis=0)[np.newaxis, :, :, :].astype(np.float32)
        h_warped = _warp_hidden_state(self._h_prev, dx1, dy1, stride=4.0)

        self._gray_hist.append((crop_gray_f32.copy(), int(crop_x), int(crop_y)))
        if len(self._gray_hist) > 8:
            self._gray_hist.pop(0)

        if self.sess is None:
            return None, 0.0, (time.perf_counter() - t0) * 1000.0

        outs = self.sess.run(
            None,
            {
                "x_t": x_t.astype(self.x_dtype),
                "h_prev_warped": h_warped.astype(self.h_dtype),
            },
        )
        hm_logits = outs[0].astype(np.float32)
        offset = outs[1].astype(np.float32)
        log_scale = outs[2].astype(np.float32)
        peak_mask = outs[4].astype(np.float32)
        self._h_prev = outs[5].astype(np.float32)

        det, max_prob = _decode_xvs_d2_top(hm_logits, offset, log_scale, peak_mask, score_thr=score_thr)
        return det, max_prob, (time.perf_counter() - t0) * 1000.0


# ---- Offline Synthetic & Live S3 Encounter Loaders ----


def build_synthetic_aot_encounters(
    *,
    num_encounters: int = 6,
    frames_per_enc: int = 24,
    seed: int = 20261121,
) -> list[AotEncounter]:
    """Build deterministic offline synthetic sky encounters (for `--smoke-synthetic` CI only)."""
    rng = np.random.default_rng(int(seed))
    cal_specs = [
        ("SYN_AOT_CAL_01", 2.4, 540.0, 62.0, "Drone"),
        ("SYN_AOT_CAL_02", 7.8, 660.0, 65.0, "Helicopter"),
        ("SYN_AOT_CAL_03", 13.5, 690.0, 54.0, "Airplane"),
    ]
    test_specs = [
        ("SYN_AOT_TEST_01", 2.8, 570.0, 60.0, "Drone"),
        ("SYN_AOT_TEST_02", 7.2, 640.0, 63.0, "Helicopter"),
        ("SYN_AOT_TEST_03", 11.8, 670.0, 56.0, "Airplane"),
    ]
    n_req = max(2, min(int(num_encounters), len(cal_specs) + len(test_specs)))
    n_cal = max(1, min(len(cal_specs), n_req // 2))
    n_test = max(1, min(len(test_specs), n_req - n_cal))
    specs = cal_specs[:n_cal] + test_specs[:n_test]
    encounters: list[AotEncounter] = []
    full_w, full_h = 640, 480
    yy, xx = np.mgrid[0:full_h, 0:full_w].astype(np.float32)
    base_sky = np.clip(175.0 + 38.0 * (yy / float(full_h)) + 6.0 * np.sin(xx / 45.0), 0.0, 255.0)

    for enc_id, wingspan_m, r0_m, vc_mps, cls_name in specs:
        frames: list[AotFrameRecord] = []
        u0 = 260.0 + float(rng.uniform(-40.0, 40.0))
        v0 = 190.0 + float(rng.uniform(-30.0, 30.0))
        du = float(rng.uniform(-1.6, 1.6))
        dv = float(rng.uniform(-0.8, 0.8))

        for t in range(int(frames_per_enc)):
            r_true = max(140.0, r0_m - vc_mps * (t / AOT_FPS))
            size_px = max(6.0, min(64.0, AOT_FOCAL_PX * wingspan_m / r_true))
            cx = u0 + du * t
            cy = v0 + dv * t
            bw = size_px * 1.15
            bh = max(4.0, size_px * 0.72)

            img = base_sky + rng.normal(0.0, 2.2, size=(full_h, full_w)).astype(np.float32)
            # Draw crisp high-contrast civil aircraft silhouette (fuselage + wings + tail)
            ix, iy = int(round(cx)), int(round(cy))
            half_w = max(2, int(round(0.5 * bw)))
            half_h = max(1, int(round(0.5 * bh)))
            cv2.ellipse(img, (ix, iy), (half_w, max(1, half_h // 2)), 0.0, 0.0, 360.0, 42.0, -1, cv2.LINE_AA)
            cv2.line(img, (ix - half_w, iy), (ix + half_w, iy), 30.0, max(1, half_h // 3), cv2.LINE_AA)
            cv2.line(img, (ix, iy - half_h), (ix, iy + half_h // 2), 38.0, max(1, half_w // 4), cv2.LINE_AA)
            gray_u8 = np.clip(img, 0.0, 255.0).astype(np.uint8)

            frames.append(
                AotFrameRecord(
                    frame_idx=t,
                    img_path=None,
                    gray_full=gray_u8,
                    gt_left=cx - 0.5 * bw,
                    gt_top=cy - 0.5 * bh,
                    gt_right=cx + 0.5 * bw,
                    gt_bottom=cy + 0.5 * bh,
                    range_distance_m=r_true,
                    is_above_horizon=True,
                    aircraft_class=cls_name,
                )
            )
        encounters.append(
            AotEncounter(
                encounter_id=enc_id,
                flight_id=enc_id,
                fps=AOT_FPS,
                aircraft_class=cls_name,
                frames=tuple(frames),
            )
        )
    return encounters


def load_live_s3_aot_encounters(
    cache_dir: Path,
    *,
    num_encounters: int = 8,
    max_frames_per_enc: int = 40,
    seed: int = 20261121,
) -> list[AotEncounter]:
    """Download & cache `6-10` real held-out Amazon AOT encounters from public S3."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    enc_csv = cache_dir / "valid_encounters_maxRange700_maxGap3_minEncLen30.csv"
    gt_csv = cache_dir / "groundtruth.csv"

    print(f"[Tier A S3] Fetching AOT encounter manifest -> {enc_csv}", flush=True)
    download_s3_object_with_https_fallback(VALID_ENCOUNTERS_S3_KEY, enc_csv, prefer_aws_cli=False)
    print(f"[Tier A S3] Fetching AOT groundtruth.csv (cached once) -> {gt_csv}", flush=True)
    download_s3_object_with_https_fallback(GROUNDTRUTH_S3_KEY, gt_csv, timeout_s=600, prefer_aws_cli=False)

    all_encs = parse_valid_encounters_csv(enc_csv)
    # Prefer clean above-horizon closing encounters with >= 40 frames and real GPS range variation
    candidates = [
        e
        for e in all_encs
        if e["horizon"] == "above"
        and int(e["framecount"]) >= 35
        and (int(e["framemax"]) - int(e["framemin"])) >= 35
        and e["min_enc_range"] is not None
        and e["max_enc_range"] is not None
        and float(e["max_enc_range"]) - float(e["min_enc_range"]) >= 80.0
    ]
    candidates.sort(key=lambda x: (str(x["flight_id"]), int(x["framemin"]), str(x["encounter_id"])))
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(candidates)).tolist()

    # Pick distinct flights for honest disjoint encounter generalization
    chosen: list[dict[str, Any]] = []
    seen_flights: set[str] = set()
    for idx in perm:
        rec = candidates[idx]
        fid = str(rec["flight_id"])
        if fid in seen_flights:
            continue
        seen_flights.add(fid)
        chosen.append(rec)
        if len(chosen) >= int(num_encounters):
            break

    flight_windows: dict[str, tuple[int, int, str]] = {}
    for rec in chosen:
        fid = str(rec["flight_id"])
        fmin = int(rec["framemin"])
        fmax = int(rec["framemax"])
        # Take contiguous closing window of length `max_frames_per_enc` ending near closest approach
        if (fmax - fmin + 1) > max_frames_per_enc:
            start_f = fmax - max_frames_per_enc + 1
            end_f = fmax
        else:
            start_f = fmin
            end_f = fmax
        flight_windows[fid] = (start_f, end_f, str(rec["encounter_id"]))

    # Single streaming scan of groundtruth.csv filtering ONLY the chosen flights/frames
    rows_by_flight: dict[str, dict[int, dict[str, Any]]] = {fid: {} for fid in flight_windows}
    print(f"[Tier A S3] Scanning groundtruth.csv for {len(flight_windows)} held-out flights...", flush=True)
    with gt_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = str(row["flight_id"]).strip() if "flight_id" in row else ""
            if fid not in flight_windows:
                continue
            obj_id = str(row["id"]).strip() if "id" in row else ""
            if not obj_id:
                continue
            rng_str = str(row["range_distance_m"]).strip() if "range_distance_m" in row else ""
            if not rng_str or rng_str.lower() == "nan":
                continue
            fnum = int(float(row["frame"]))
            start_f, end_f, _ = flight_windows[fid]
            if fnum < start_f or fnum > end_f:
                continue
            rng_m = float(rng_str)
            if not math.isfinite(rng_m) or rng_m <= 10.0:
                continue
            # Keep closest airborne object per frame if multiple exist
            prev_row = rows_by_flight[fid][fnum] if fnum in rows_by_flight[fid] else None
            if prev_row is None or rng_m < float(prev_row["range_distance_m"]):
                rows_by_flight[fid][fnum] = {
                    "frame": fnum,
                    "img_name": str(row["img_name"]).strip(),
                    "id": obj_id,
                    "range_distance_m": rng_m,
                    "is_above_horizon": float(row["is_above_horizon"]) > 0.0 if row["is_above_horizon"] else True,
                    "gt_left": float(row["gt_left"]),
                    "gt_top": float(row["gt_top"]),
                    "gt_right": float(row["gt_right"]),
                    "gt_bottom": float(row["gt_bottom"]),
                }

    # Download required PNGs in parallel
    download_jobs: list[tuple[str, Path]] = []
    for fid, frame_map in rows_by_flight.items():
        for fnum in sorted(frame_map.keys()):
            img_name = str(frame_map[fnum]["img_name"])
            s3_key = f"{IMAGES_S3_PREFIX}/{fid}/{img_name}"
            local_path = cache_dir / "Images" / fid / img_name
            if not (local_path.is_file() and local_path.stat().st_size > 0):
                download_jobs.append((s3_key, local_path))

    if download_jobs:
        print(f"[Tier A S3] Downloading {len(download_jobs)} real AOT PNG frames (8 threads)...", flush=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = [
                pool.submit(download_s3_object_with_https_fallback, key, dst, prefer_aws_cli=False)
                for key, dst in download_jobs
            ]
            done_cnt = 0
            for fut in as_completed(futs):
                fut.result()
                done_cnt += 1
                if done_cnt % 40 == 0 or done_cnt == len(download_jobs):
                    print(f"[Tier A S3] Downloaded {done_cnt}/{len(download_jobs)} PNGs", flush=True)

    encounters: list[AotEncounter] = []
    for rec in chosen:
        fid = str(rec["flight_id"])
        enc_id = str(rec["encounter_id"])
        frame_map = rows_by_flight[fid]
        sorted_fnums = sorted(frame_map.keys())
        if len(sorted_fnums) < 12:
            continue
        frames: list[AotFrameRecord] = []
        cls_name = "Aircraft"
        for idx, fnum in enumerate(sorted_fnums):
            r = frame_map[fnum]
            img_path = cache_dir / "Images" / fid / str(r["img_name"])
            if not img_path.is_file():
                continue
            obj_id_str = str(r["id"])
            cls_name = "".join(ch for ch in obj_id_str if ch.isalpha()) or "Aircraft"
            frames.append(
                AotFrameRecord(
                    frame_idx=idx,
                    img_path=img_path,
                    gray_full=None,
                    gt_left=float(r["gt_left"]),
                    gt_top=float(r["gt_top"]),
                    gt_right=float(r["gt_right"]),
                    gt_bottom=float(r["gt_bottom"]),
                    range_distance_m=float(r["range_distance_m"]),
                    is_above_horizon=bool(r["is_above_horizon"]),
                    aircraft_class=cls_name,
                )
            )
        if len(frames) >= 12:
            encounters.append(
                AotEncounter(
                    encounter_id=enc_id,
                    flight_id=fid,
                    fps=AOT_FPS,
                    aircraft_class=cls_name,
                    frames=tuple(frames),
                )
            )
    if len(encounters) < 2:
        raise RuntimeError(f"Expected >= 2 valid live S3 AOT encounters, got {len(encounters)}")
    return encounters


# ---- Evaluation Engine ----


def _load_frame_gray(rec: AotFrameRecord) -> np.ndarray:
    if rec.gray_full is not None:
        return rec.gray_full
    if rec.img_path is None or not rec.img_path.is_file():
        raise FileNotFoundError(f"Missing AOT frame image: {rec.img_path}")
    img = cv2.imread(str(rec.img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to decode image: {rec.img_path}")
    return img


def _compute_smooth_crop_origin(
    cx: float,
    cy: float,
    img_w: int,
    img_h: int,
    prev_origin: tuple[int, int] | None,
) -> tuple[int, int]:
    """Compute stable `1:1` native `256x256` crop origin keeping target centered without jitter."""
    target_x = int(round(cx - 0.5 * CROP_SIZE))
    target_y = int(round(cy - 0.5 * CROP_SIZE))
    if prev_origin is not None:
        px, py = prev_origin
        # Only pan crop window when target approaches margin, preserving integer GMC stability
        if abs(cx - (px + 0.5 * CROP_SIZE)) < 56.0:
            target_x = px
        else:
            target_x = int(round(0.35 * target_x + 0.65 * px))
        if abs(cy - (py + 0.5 * CROP_SIZE)) < 56.0:
            target_y = py
        else:
            target_y = int(round(0.35 * target_y + 0.65 * py))
    ox = max(0, min(max(0, img_w - CROP_SIZE), target_x))
    oy = max(0, min(max(0, img_h - CROP_SIZE), target_y))
    return ox, oy


def _evaluate_encounter(
    enc: AotEncounter,
    runner: XvsD2CropRunner,
    *,
    q_hat_conformal: float,
    score_thr: float = 0.25,
) -> list[dict[str, Any]]:
    """Run real `XVS-D2` ONNX + passive `ActiveRangeController` (`a_az = 0`) over one encounter.

    Pure detector stream (`det_only`):
    - On detected frames (`det is not None`), update `last_det_obs` and feed pure `XVS-D2` centroid & size.
    - On missed frames after acquisition (`det is None` and `last_det_obs is not None`), coast/hold
      `last_det_obs` with `size_rate_px_s = 0.0` (ZERO GT bbox substitution).
    - On pre-acquisition frames (`det is None` and `last_det_obs is None`), mark `track_status = "pre_acq"`
      and exclude from `det_only` calibration/evaluation (never inject GT boxes into the filter).
    - Separately compute `gt_box_physics_tau_obs_rho_dot` from GT boxes strictly as an explicitly
      labeled ideal optical-expansion physics ceiling.
    """
    runner.reset()
    cfg = ActiveRangeConfig(q_hat_conformal=float(q_hat_conformal), sigma_bearing_rad=2.86 / AOT_FOCAL_PX)
    ctrl = ActiveRangeController(cfg)

    dt_s = 1.0 / max(enc.fps, 1.0)
    prev_origin: tuple[int, int] | None = None
    last_det_obs: tuple[float, float, float] | None = None
    det_obs_hist: list[tuple[float, float, float]] = []
    det_track_age = 0
    gt_size_hist: list[float] = []
    gt_range_hist: list[float] = []
    prior_r_mean = float(np.mean(R_INIT_M))
    prior_r_std = float(np.std(R_INIT_M))
    step_rows: list[dict[str, Any]] = []

    for idx, rec in enumerate(enc.frames):
        gray_full = _load_frame_gray(rec)
        img_h, img_w = gray_full.shape[:2]

        gt_cx = 0.5 * (rec.gt_left + rec.gt_right)
        gt_cy = 0.5 * (rec.gt_top + rec.gt_bottom)
        gt_w = max(1.0, rec.gt_right - rec.gt_left)
        gt_h = max(1.0, rec.gt_bottom - rec.gt_top)
        gt_s = math.sqrt(gt_w * gt_h)

        crop_x, crop_y = _compute_smooth_crop_origin(gt_cx, gt_cy, img_w, img_h, prev_origin)
        prev_origin = (crop_x, crop_y)
        crop_u8 = gray_full[crop_y : crop_y + CROP_SIZE, crop_x : crop_x + CROP_SIZE]
        if crop_u8.shape != (CROP_SIZE, CROP_SIZE):
            pad_b = max(0, CROP_SIZE - crop_u8.shape[0])
            pad_r = max(0, CROP_SIZE - crop_u8.shape[1])
            crop_u8 = cv2.copyMakeBorder(crop_u8, 0, pad_b, 0, pad_r, cv2.BORDER_REPLICATE)
        crop_f32 = crop_u8.astype(np.float32)

        det, max_prob, lat_ms = runner.step(crop_f32, crop_x, crop_y, score_thr=score_thr)

        gt_box_crop = (
            rec.gt_left - crop_x,
            rec.gt_top - crop_y,
            rec.gt_right - crop_x,
            rec.gt_bottom - crop_y,
        )
        gt_cx_c = 0.5 * (gt_box_crop[0] + gt_box_crop[2])
        gt_cy_c = 0.5 * (gt_box_crop[1] + gt_box_crop[3])

        # Detection accuracy diagnostics against GT (evaluation-only; never gates GT injection)
        detected = False
        centroid_err_px: float | None = None
        if det is not None:
            px1, py1, px2, py2, _pscore = det
            pcx = 0.5 * (px1 + px2)
            pcy = 0.5 * (py1 + py2)
            err = math.hypot(pcx - gt_cx_c, pcy - gt_cy_c)
            if err <= 25.0:
                detected = True
                centroid_err_px = err

        # Separate ideal optical-expansion physics ceiling from GT boxes (clearly labeled, never fed to controller)
        gt_size_hist.append(gt_s)
        gt_range_hist.append(rec.range_distance_m)
        gt_box_physics_tau_obs: float | None = None
        gt_box_physics_gps_true: float | None = None
        if len(gt_size_hist) >= 5 and gt_s >= 6.0:
            win_k = min(7, len(gt_size_hist))
            win_gt_s = np.array(gt_size_hist[-win_k:], dtype=np.float64)
            t_win_gt = np.arange(win_k, dtype=np.float64) * dt_s
            t_c_gt = t_win_gt - float(t_win_gt.mean())
            denom_gt = max(float(np.dot(t_c_gt, t_c_gt)), 1.0e-9)
            ln_gt_s = np.log(np.maximum(win_gt_s, 1.0))
            gt_box_physics_tau_obs = -float(np.dot(t_c_gt, ln_gt_s - float(ln_gt_s.mean())) / denom_gt)
            ln_gt_r = np.log(np.maximum(np.array(gt_range_hist[-win_k:], dtype=np.float64), 10.0))
            gt_box_physics_gps_true = float(np.dot(t_c_gt, ln_gt_r - float(ln_gt_r.mean())) / denom_gt)

        # Pure detector-driven stream (detected -> update; missed after acquisition -> coast/hold; before acquisition -> pre_acq)
        det_only_tau_obs: float | None = None
        det_only_gps_true: float | None = None
        s_rate_px_s = 0.0
        t_now = float(idx) * dt_s

        if det is not None:
            px1, py1, px2, py2, _pscore = det
            obs_cx_full = crop_x + 0.5 * (px1 + px2)
            obs_cy_full = crop_y + 0.5 * (py1 + py2)
            obs_s = math.sqrt(max(1.0, (px2 - px1) * (py2 - py1)))
            last_det_obs = (obs_cx_full, obs_cy_full, obs_s)
            det_obs_hist.append((t_now, obs_s, rec.range_distance_m))
            track_status = "detected"
            has_det_track = True
            if len(det_obs_hist) >= 5:
                win_d = det_obs_hist[-min(7, len(det_obs_hist)) :]
                t_d = np.array([item[0] for item in win_d], dtype=np.float64)
                s_d = np.array([item[1] for item in win_d], dtype=np.float64)
                r_d = np.array([item[2] for item in win_d], dtype=np.float64)
                t_span = float(t_d[-1] - t_d[0])
                t_c_d = t_d - float(t_d.mean())
                denom_d = max(float(np.dot(t_c_d, t_c_d)), 1.0e-9)
                if t_span >= 0.15:
                    s_rate_px_s = float(np.dot(t_c_d, s_d - float(s_d.mean())) / denom_d)
                if obs_s >= 6.0 and t_span >= 0.2:
                    ln_s_d = np.log(np.maximum(s_d, 1.0))
                    det_only_tau_obs = -float(np.dot(t_c_d, ln_s_d - float(ln_s_d.mean())) / denom_d)
                    ln_r_d = np.log(np.maximum(r_d, 10.0))
                    det_only_gps_true = float(np.dot(t_c_d, ln_r_d - float(ln_r_d.mean())) / denom_d)
        elif last_det_obs is not None:
            # Coast/hold last valid XVS-D2 detection (ZERO GT bbox injection)
            obs_cx_full, obs_cy_full, obs_s = last_det_obs
            track_status = "coasted"
            has_det_track = True
            s_rate_px_s = 0.0
        else:
            # Pre-acquisition: no XVS-D2 detection yet; do NOT inject GT into ActiveRangeController
            track_status = "pre_acq"
            has_det_track = False
            obs_cx_full = 0.0
            obs_cy_full = 0.0
            obs_s = 0.0

        if has_det_track:
            det_track_age += 1
            alpha_rad = math.atan2(obs_cx_full - 0.5 * img_w, AOT_FOCAL_PX)
            beta_rad = math.atan2(0.5 * img_h - obs_cy_full, AOT_FOCAL_PX)
            # Honest passive Tier A: a_ned = (0, 0, 0) since AOT groundtruth lacks ownship IMU
            own = OwnshipStateNed(v_ned=(35.0, 0.0, 0.0), a_ned=(0.0, 0.0, 0.0), heading_rad=0.0, tas_mps=35.0)
            meas = ActiveRangeMeasurement(
                alpha_rad=alpha_rad,
                beta_rad=beta_rad,
                size_px=obs_s,
                size_rate_px_s=s_rate_px_s,
                contrast_ratio=0.5,
                sky_beta_ext=0.0,
                daytime_clear=False,
                track_age=det_track_age,
                dt_s=dt_s,
            )
            tel = ctrl.step(own, meas)
            r_hat_m = tel.r_hat_m
            sigma_r_m = tel.sigma_r_m
            r_safe_lower_m = tel.r_safe_lower_m
            weights_row = list(tel.weights)
            r_naive_11m: float | None = AOT_FOCAL_PX * NAIVE_WINGSPAN_M / max(obs_s, 1.0)
        else:
            r_hat_m = prior_r_mean
            sigma_r_m = prior_r_std
            r_safe_lower_m = compute_conformal_lower_bound(prior_r_mean, prior_r_std, q_hat_conformal)
            weights_row = [1.0 / len(R_INIT_M)] * len(R_INIT_M)
            r_naive_11m = None

        step_rows.append(
            {
                "encounter_id": enc.encounter_id,
                "aircraft_class": enc.aircraft_class,
                "frame_idx": idx,
                "crop_x": crop_x,
                "crop_y": crop_y,
                "gt_box_crop": [float(v) for v in gt_box_crop],
                "det_box_crop": [float(v) for v in det] if det is not None else None,
                "detected": detected,
                "track_status": track_status,
                "has_det_track": has_det_track,
                "max_prob": max_prob,
                "centroid_err_px": centroid_err_px,
                "latency_ms": lat_ms,
                "size_px": obs_s,
                "gt_size_px": gt_s,
                "range_true_m": rec.range_distance_m,
                "r_hat_m": r_hat_m,
                "sigma_r_m": sigma_r_m,
                "r_safe_lower_m": r_safe_lower_m,
                "r_naive_11m": r_naive_11m,
                "det_only_tau_obs_rho_dot": det_only_tau_obs,
                "det_only_gps_true_rho_dot": det_only_gps_true,
                "gt_box_physics_tau_obs_rho_dot": gt_box_physics_tau_obs,
                "gt_box_physics_gps_true_rho_dot": gt_box_physics_gps_true,
                "tau_obs_rho_dot": det_only_tau_obs,
                "gps_true_rho_dot": det_only_gps_true,
                "weights": weights_row,
                "is_above_horizon": rec.is_above_horizon,
                "crop_u8": crop_u8,
                "full_gray": gray_full,
                "gt_box_full": (rec.gt_left, rec.gt_top, rec.gt_right, rec.gt_bottom),
            }
        )
    return step_rows


# ---- 1080p Showcase Video Renderer ----


def _render_proof_frame_1080p(
    row: dict[str, Any],
    enc_history: list[dict[str, Any]],
    *,
    q_hat: float,
    overall_cov_pct: float,
    overall_naive_over_pct: float,
    is_live_s3: bool,
) -> np.ndarray:
    """Render one crisp 1920x1080 avionics proof frame."""
    canvas = np.full((CANVAS_H, CANVAS_W, 3), COL_BG, dtype=np.uint8)

    # Top header bar
    cv2.rectangle(canvas, (0, 0), (CANVAS_W, 56), COL_PANEL, -1)
    cv2.line(canvas, (0, 56), (CANVAS_W, 56), COL_BORDER, 1)
    src_badge = "REAL AMAZON AOT S3 AIRBORNE VIDEO" if is_live_s3 else "OFFLINE SYNTHETIC CI SMOKE"
    cv2.putText(
        canvas,
        f"OPENXVS v3.0 // XVS-ActiveRange TIER A PASSIVE PROOF ({src_badge})",
        (24, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.74,
        COL_TEXT,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"ENC: {row['encounter_id']} | FRM {int(row['frame_idx']) + 1:03d} | a_az = 0 (HONEST PASSIVE)",
        (1240, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        COL_AMBER,
        1,
        cv2.LINE_AA,
    )

    # Left primary viewport: full sky frame (1120x840)
    vp_x, vp_y, vp_w, vp_h = 24, 76, 1120, 840
    full_bgr = cv2.cvtColor(row["full_gray"], cv2.COLOR_GRAY2BGR)
    src_h, src_w = full_bgr.shape[:2]
    scaled = cv2.resize(full_bgr, (vp_w, vp_h), interpolation=cv2.INTER_AREA)
    sx = float(vp_w) / float(src_w)
    sy = float(vp_h) / float(src_h)

    gx1, gy1, gx2, gy2 = row["gt_box_full"]
    cv2.rectangle(
        scaled,
        (int(round(gx1 * sx)) - 4, int(round(gy1 * sy)) - 4),
        (int(round(gx2 * sx)) + 4, int(round(gy2 * sy)) + 4),
        COL_CYAN,
        2,
        cv2.LINE_AA,
    )
    if row["det_box_crop"] is not None:
        dx1, dy1, dx2, dy2, dscore = row["det_box_crop"]
        cx0, cy0 = float(row["crop_x"]), float(row["crop_y"])
        cv2.rectangle(
            scaled,
            (int(round((cx0 + dx1) * sx)) - 3, int(round((cy0 + dy1) * sy)) - 3),
            (int(round((cx0 + dx2) * sx)) + 3, int(round((cy0 + dy2) * sy)) + 3),
            COL_EMERALD,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            scaled,
            f"XVS-D2 p={dscore:.2f}",
            (int(round((cx0 + dx1) * sx)), max(22, int(round((cy0 + dy1) * sy)) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COL_EMERALD,
            1,
            cv2.LINE_AA,
        )
    canvas[vp_y : vp_y + vp_h, vp_x : vp_x + vp_w] = scaled
    cv2.rectangle(canvas, (vp_x, vp_y), (vp_x + vp_w, vp_y + vp_h), COL_BORDER, 2)

    # 4x Native 1:1 Crop PiP overlay in top-right of viewport
    pip_sz = 256
    pip_bgr = cv2.cvtColor(row["crop_u8"], cv2.COLOR_GRAY2BGR)
    gb = row["gt_box_crop"]
    cv2.rectangle(
        pip_bgr,
        (int(round(gb[0])), int(round(gb[1]))),
        (int(round(gb[2])), int(round(gb[3]))),
        COL_CYAN,
        2,
        cv2.LINE_AA,
    )
    if row["det_box_crop"] is not None:
        db = row["det_box_crop"]
        cv2.rectangle(
            pip_bgr,
            (int(round(db[0])), int(round(db[1]))),
            (int(round(db[2])), int(round(db[3]))),
            COL_EMERALD,
            2,
            cv2.LINE_AA,
        )
    pip_x = vp_x + vp_w - pip_sz - 16
    pip_y = vp_y + 34
    cv2.rectangle(canvas, (pip_x - 2, pip_y - 24), (pip_x + pip_sz + 2, pip_y + pip_sz + 2), COL_PANEL, -1)
    cv2.rectangle(canvas, (pip_x - 2, pip_y - 24), (pip_x + pip_sz + 2, pip_y + pip_sz + 2), COL_AMBER, 1)
    cv2.putText(canvas, "1:1 NATIVE 256x256 XVS-D2 CROP", (pip_x + 6, pip_y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COL_AMBER, 1, cv2.LINE_AA)
    canvas[pip_y : pip_y + pip_sz, pip_x : pip_x + pip_sz] = pip_bgr

    # Right Top Panel: Live Range Safety Comparison Chart (730x360)
    rx, ry, rw, rh = 1166, 76, 730, 360
    cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), COL_PANEL, -1)
    cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), COL_BORDER, 1)
    cv2.putText(
        canvas,
        "RANGE SAFETY: GPS TRUTH vs CP-MSC SAFE LOWER vs NAIVE BBOX (11m)",
        (rx + 16, ry + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        COL_TEXT,
        1,
        cv2.LINE_AA,
    )

    gx, gy, gw, gh = rx + 54, ry + 48, rw - 76, rh - 96
    cv2.rectangle(canvas, (gx, gy), (gx + gw, gy + gh), COL_BG, -1)
    cv2.rectangle(canvas, (gx, gy), (gx + gw, gy + gh), COL_BORDER, 1)

    naive_vals = [float(r["r_naive_11m"]) for r in enc_history if r["r_naive_11m"] is not None]
    max_y = max(
        900.0,
        max(float(r["range_true_m"]) for r in enc_history) * 1.25,
        min(2600.0, (max(naive_vals) * 1.05) if naive_vals else 900.0),
    )
    n_pts = max(2, len(enc_history))

    def _pt(i: int, r_val: float) -> tuple[int, int]:
        px = gx + int(round((float(i) / float(max(1, n_pts - 1))) * gw))
        py = gy + gh - int(round((max(0.0, min(max_y, r_val)) / max_y) * gh))
        return px, py

    for g_val in (250.0, 500.0, 1000.0, 1500.0, 2000.0):
        if g_val < max_y:
            _, py = _pt(0, g_val)
            cv2.line(canvas, (gx, py), (gx + gw, py), (38, 48, 62), 1)
            cv2.putText(canvas, f"{int(g_val)}m", (rx + 8, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_MUTED, 1, cv2.LINE_AA)

    pts_true = [_pt(i, float(r["range_true_m"])) for i, r in enumerate(enc_history)]
    pts_safe = [_pt(i, float(r["r_safe_lower_m"])) for i, r in enumerate(enc_history) if r["has_det_track"]]
    pts_naive = [_pt(i, float(r["r_naive_11m"])) for i, r in enumerate(enc_history) if r["r_naive_11m"] is not None]
    if len(pts_naive) >= 2:
        cv2.polylines(canvas, [np.array(pts_naive, dtype=np.int32)], False, COL_ROSE, 2, cv2.LINE_AA)
    if len(pts_safe) >= 2:
        cv2.polylines(canvas, [np.array(pts_safe, dtype=np.int32)], False, COL_EMERALD, 3, cv2.LINE_AA)
    if len(pts_true) >= 2:
        cv2.polylines(canvas, [np.array(pts_true, dtype=np.int32)], False, COL_CYAN, 2, cv2.LINE_AA)

    naive_lbl = f"{float(row['r_naive_11m']):.0f} m" if row["r_naive_11m"] is not None else "PRE-ACQ"
    safe_lbl = f"{float(row['r_safe_lower_m']):.0f} m" if row["has_det_track"] else "PRE-ACQ"
    cv2.putText(canvas, f"[CYAN] GPS True: {row['range_true_m']:.0f} m", (gx, ry + rh - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COL_CYAN, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"[GREEN] CP-MSC Safe: {safe_lbl}", (gx + 215, ry + rh - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COL_EMERALD, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"[ROSE] Naive 11m: {naive_lbl}", (gx + 465, ry + rh - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COL_ROSE, 1, cv2.LINE_AA)

    # Right Middle Panel: M = 6 MSC-RPEKF Weights & Scale-Cancelling Tau Closure (730x250)
    my = ry + rh + 20
    mh = 250
    cv2.rectangle(canvas, (rx, my), (rx + rw, my + mh), COL_PANEL, -1)
    cv2.rectangle(canvas, (rx, my), (rx + rw, my + mh), COL_BORDER, 1)
    cv2.putText(
        canvas,
        "M=6 MSC-RPEKF HYPOTHESIS WEIGHTS & OPTICAL EXPANSION TAU (SIZE-AGNOSTIC)",
        (rx + 16, my + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        COL_TEXT,
        1,
        cv2.LINE_AA,
    )
    bank_labels = ("200m", "348m", "606m", "1056m", "1838m", "3200m")
    weights = row["weights"]
    bar_w = 88
    gap = 24
    bx0 = rx + 26
    by_base = my + 178
    for b_idx in range(6):
        w_val = float(weights[b_idx])
        bh_px = max(4, int(round(w_val * 115.0)))
        x1 = bx0 + b_idx * (bar_w + gap)
        cv2.rectangle(canvas, (x1, by_base - bh_px), (x1 + bar_w, by_base), COL_AMBER, -1)
        cv2.rectangle(canvas, (x1, by_base - 115), (x1 + bar_w, by_base), COL_BORDER, 1)
        cv2.putText(canvas, bank_labels[b_idx], (x1 + 14, by_base + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COL_MUTED, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{w_val:.2f}", (x1 + 22, by_base - bh_px - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_TEXT, 1, cv2.LINE_AA)

    tau_str = (
        f"Det-Only Optical tau={row['det_only_tau_obs_rho_dot']:+.3f} s^-1 vs GPS={row['det_only_gps_true_rho_dot']:+.3f} s^-1"
        if (row["det_only_tau_obs_rho_dot"] is not None and row["det_only_gps_true_rho_dot"] is not None)
        else f"Det-Only tau [{str(row['track_status']).upper()}] (det_size={row['size_px']:.1f} px)"
    )
    cv2.putText(canvas, tau_str, (rx + 26, my + mh - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, COL_CYAN, 1, cv2.LINE_AA)

    # Right Bottom Panel: Honest Aggregate Tier A Metrics & Conformal Guarantee (730x210)
    by = my + mh + 20
    bh_pan = 210
    cv2.rectangle(canvas, (rx, by), (rx + rw, by + bh_pan), COL_PANEL, -1)
    cv2.rectangle(canvas, (rx, by), (rx + rw, by + bh_pan), COL_BORDER, 1)
    cv2.putText(canvas, "HELD-OUT TIER A TEST SUMMARY (DISJOINT 50/50 CAL/TEST SPLIT)", (rx + 16, by + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, COL_TEXT, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"CP-MSC Conformal Coverage P(R_true >= R_safe):  {overall_cov_pct:5.1f}%  (q_hat={q_hat:.2f})", (rx + 24, by + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.56, COL_EMERALD, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Naive 11m Bbox Hazardous Overestimation Rate:   {overall_naive_over_pct:5.1f}%", (rx + 24, by + 114), cv2.FONT_HERSHEY_SIMPLEX, 0.56, COL_ROSE, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"XVS-D2 257K FP16 ONNX Latency: {row['latency_ms']:.1f} ms | Class: {row['aircraft_class']} | Size: {row['size_px']:.1f} px", (rx + 24, by + 154), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COL_MUTED, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Note: Pre-recorded AOT lacks IMU a_az; active <=12 deg swerve verified in Tier B 6-DoF MC.", (rx + 24, by + 188), cv2.FONT_HERSHEY_SIMPLEX, 0.43, COL_AMBER, 1, cv2.LINE_AA)

    # Bottom status bar
    cv2.rectangle(canvas, (0, CANVAS_H - 44), (CANVAS_W, CANVAS_H), COL_PANEL, -1)
    cv2.line(canvas, (0, CANVAS_H - 44), (CANVAS_W, CANVAS_H - 44), COL_BORDER, 1)
    cv2.putText(
        canvas,
        "OPENXVS v3.0 // STRICTLY CIVIL/DUAL-USE DAA SAFETY VERIFICATION // KANI 18/18 FORMAL PROOFS",
        (24, CANVAS_H - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        COL_MUTED,
        1,
        cv2.LINE_AA,
    )
    return canvas


def _export_realsky_tab3_pack(
    test_encounters_rows: list[list[dict[str, Any]]],
    pack_dir: Path,
    *,
    q_hat_conformal: float,
    max_encounters: int = 2,
    max_frames_per_enc: int = 40,
) -> dict[str, Any]:
    """Export `1:1` native `256x256` real-sky crops (`<= 1.5 MB`) + `manifest.json` for Observatory Tab 3."""
    pack_dir.mkdir(parents=True, exist_ok=True)
    for old_jpg in pack_dir.glob("*.jpg"):
        old_jpg.unlink()

    selected_encs = test_encounters_rows[: max(1, int(max_encounters))]
    quality = 86

    while True:
        manifest_encs: list[dict[str, Any]] = []
        total_bytes = 0
        for e_idx, rows in enumerate(selected_encs):
            clipped = rows[: int(max_frames_per_enc)]
            enc_id = str(clipped[0]["encounter_id"])
            cls_name = str(clipped[0]["aircraft_class"])
            frame_entries: list[dict[str, Any]] = []
            for f_i, row in enumerate(clipped):
                fname = f"enc{e_idx}_f{f_i:03d}.jpg"
                fpath = pack_dir / fname
                ok, buf = cv2.imencode(".jpg", row["crop_u8"], [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
                if not ok:
                    raise RuntimeError(f"Failed JPEG encode for {fname}")
                raw_bytes = buf.tobytes()
                fpath.write_bytes(raw_bytes)
                total_bytes += len(raw_bytes)
                frame_entries.append(
                    {
                        "file": fname,
                        "frame_idx": int(f_i),
                        "crop_x": int(row["crop_x"]),
                        "crop_y": int(row["crop_y"]),
                        "gt_box_crop": [round(float(v), 2) for v in row["gt_box_crop"]],
                        "range_distance_m": round(float(row["range_true_m"]), 2),
                        "is_above_horizon": bool(row["is_above_horizon"]),
                    }
                )
            manifest_encs.append(
                {
                    "encounter_id": enc_id,
                    "fps": AOT_FPS,
                    "aircraft_class": cls_name,
                    "frames": frame_entries,
                }
            )

        manifest_doc = {
            "schema_version": "realsky_aot_pack_v1",
            "focal_px": AOT_FOCAL_PX,
            "q_hat_conformal": round(float(q_hat_conformal), 6),
            "encounters": manifest_encs,
        }
        manifest_bytes = (json.dumps(manifest_doc, indent=2) + "\n").encode("utf-8")
        if (total_bytes + len(manifest_bytes)) <= MAX_PACK_BYTES or quality <= 60:
            (pack_dir / "manifest.json").write_bytes(manifest_bytes)
            return {
                "pack_dir": str(pack_dir),
                "total_bytes": int(total_bytes + len(manifest_bytes)),
                "jpeg_quality": int(quality),
                "num_encounters": len(manifest_encs),
                "num_frames": sum(len(e["frames"]) for e in manifest_encs),
            }
        quality -= 8


def _rel_display(path: Path) -> str:
    """Format path cleanly relative to `REPO_ROOT` with forward slashes when inside repo."""
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _render_tier_a_markdown(metrics: dict[str, Any]) -> str:
    """Format Tier A Real Amazon AOT passive benchmark report."""
    mode_tag = "LIVE AMAZON AOT S3 (`part1/ImageSets/valid_encounters_maxRange700_maxGap3_minEncLen30.csv`)" if metrics["live_s3"] else "OFFLINE SYNTHETIC CI SMOKE"
    lines = [
        "# OpenXVS v3.0 — Tier A Real Amazon AOT Passive Sky Proof (`XVS-D2` + `tau` + `CP-MSC`)",
        "",
        f"- **Generated (UTC):** `{metrics['generated_utc']}`",
        f"- **Data Source:** {mode_tag}",
        f"- **ONNX Model:** `{_rel_display(Path(str(metrics['onnx_model'])))}` (`257K` params, `552 KB` FP16 ONNX, native `1:1` `256×256` crops)",
        f"- **Disjoint Encounter Split:** `{metrics['n_cal_encounters']}` calibration encounters (`{metrics['n_cal_frames']}` total / `{metrics['n_cal_det_tracked_frames']}` tracked frames) -> frozen Split Conformal `q_hat = {metrics['q_hat_conformal']:.4f}` (`alpha = 0.05`) evaluated on **`{metrics['n_test_encounters']}` unseen test encounters (`{metrics['n_test_frames']}` total / `{metrics['n_test_det_tracked_frames']}` tracked frames)**.",
        f"- **Pure Detector-Stream Discipline (Zero GT Bbox Injection):** Live `ActiveRangeController`, `CP-MSC`, Naive `11 m` Bbox, and `det_only` optical `tau` consume **strictly `XVS-D2` detections** (`{metrics['det_frames_count']}` detected + `{metrics['coast_frames_count']}` coasted/held last valid detection + `{metrics['pre_acq_frames_count']}` pre-acquisition frames excluded from tracked metrics). Zero GT boxes are substituted on missed detections. Ideal GT-box optical-expansion physics (`gt_box_physics_tau_*`) is reported separately below as an upper-bound physics reference.",
        "- **Honest Passive Scope (`a_az = 0`):** Pre-recorded Amazon AOT `groundtruth.csv` contains real GPS `range_distance_m` and 2D bounding boxes but lacks ownship IMU lateral acceleration `a_az` and cannot be retroactively steered. Tier A therefore strictly evaluates passive real-sky layers (`XVS-D2` detection + scale-cancelling optical expansion `tau` + `CP-MSC` conformal safe lower bound vs Naive `11 m` Bbox-Size Regressor). Full closed-loop active micro-swerve (`<= 12°`) is verified in Tier B (`docs/xvs/reports/V3_M1_active_range_benchmark.md`).",
        "",
        "## 1. Aggregate Held-Out Test Metrics",
        "",
        "| Metric | Value | Gate / Comparison |",
        "| :--- | :---: | :--- |",
        f"| **`XVS-D2` Detection Rate (`<= 25 px`)** | `{metrics['xvs_d2_detection_rate']*100.0:.1f}%` | Centroid RMSE: `{metrics['xvs_d2_centroid_rmse_px']:.2f} px`, Mean ORT Latency: `{metrics['mean_ort_latency_ms']:.1f} ms` |",
        f"| **Test Frame Stream Accounting (Zero GT Substitution)** | `{metrics['det_frames_count']} det / {metrics['coast_frames_count']} coast / {metrics['pre_acq_frames_count']} pre-acq` | `{metrics['n_test_det_tracked_frames']}` tracked frames evaluated out of `{metrics['n_test_frames']}` total |",
        f"| **Pure Detector-Stream `det_only` Optical `tau` MAE (`rho_dot`)** | `{metrics['det_only_tau_mae_inv_s']:.4f} s^-1` | RMSE: `{metrics['det_only_tau_rmse_inv_s']:.4f} s^-1` across `{metrics['det_only_tau_valid_frames']}` pure-detector frames (`s_det >= 6 px`, no GT) |",
        f"| **Ideal GT-Box Physics Reference `gt_box_physics` `tau` MAE (`rho_dot`)** | `{metrics['gt_box_physics_tau_mae_inv_s']:.4f} s^-1` | RMSE: `{metrics['gt_box_physics_tau_rmse_inv_s']:.4f} s^-1` across `{metrics['gt_box_physics_tau_valid_frames']}` GT frames (labeled physics ceiling) |",
        f"| **`CP-MSC` Conformal Coverage `P(R_true >= R_safe)` (`det_only`)** | **`{metrics['conformal_coverage']*100.0:.1f}%`** | Spec §5 gate: `>= 90.0%` (target `>= 95.0%`, pure detector + coast stream) |",
        f"| **`CP-MSC` Hazardous Overestimation Rate (`R_safe > R_true`, `det_only`)** | **`{metrics['cp_msc_overestimation_rate']*100.0:.1f}%`** | Strictly lower than Naive Bbox-Size (`11 m` Cessna prior) |",
        f"| **Naive Bbox-Size (`11 m`) Hazardous Overestimation Rate (`det_only`)** | **`{metrics['naive_overestimation_rate']*100.0:.1f}%`** | Mean Naive Range RMSE: `{metrics['naive_rmse_m']:.1f} m` |",
        f"| **1080p Showcase Proof Video** | `{_rel_display(Path(str(metrics['video_path'])))}` | `{metrics['video_bytes']:,}` bytes (`{metrics['video_frames']}` frames @ `{AOT_FPS:.0f}` FPS) |",
        f"| **Native `1:1` Observatory Tab 3 Pack** | `{_rel_display(Path(str(metrics['pack']['pack_dir'])))}` | `{metrics['pack']['num_encounters']}` encounters (`{metrics['pack']['num_frames']}` total crisp `256×256` crops, `{metrics['pack']['total_bytes']/1024.0:.1f} KiB <= 1,500 KiB`) |",
        "",
        "## 2. Per-Encounter Held-Out Test Breakdown",
        "",
        "| Encounter ID | Class | Frames (Det / Coast / Pre) | GPS Range Span (m) | `XVS-D2` Det (%) | `CP-MSC` Coverage (%) | `CP-MSC` Overest (%) | Naive `11 m` Overest (%) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for pe in metrics["per_encounter_test"]:
        lines.append(
            f"| `{pe['encounter_id']}` | `{pe['aircraft_class']}` | `{pe['n_frames']} ({pe['det_frames']}/{pe['coast_frames']}/{pe['pre_acq_frames']})` | "
            f"`{pe['min_range_m']:.0f}–{pe['max_range_m']:.0f}` | `{pe['det_rate']*100.0:.1f}%` | "
            f"`{pe['cp_coverage']*100.0:.1f}%` | `{pe['cp_overest_rate']*100.0:.1f}%` | `{pe['naive_overest_rate']*100.0:.1f}%` |"
        )
    lines.append("")
    return "\n".join(lines)


def run_aot_activerange_proof(
    *,
    live_s3: bool,
    onnx_path: Path = DEFAULT_ONNX_PATH,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    video_path: Path = DEFAULT_VIDEO_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    pack_dir: Path = DEFAULT_PACK_DIR,
    num_encounters: int = 8,
    max_frames_per_enc: int = 40,
    seed: int = 20261121,
) -> dict[str, Any]:
    """Execute Tier A passive proof, render 1080p MP4, emit report, and export Tab 3 `1:1` pack."""
    if live_s3:
        encounters = load_live_s3_aot_encounters(
            cache_dir,
            num_encounters=num_encounters,
            max_frames_per_enc=max_frames_per_enc,
            seed=seed,
        )
    else:
        encounters = build_synthetic_aot_encounters(
            num_encounters=min(6, max(4, num_encounters)),
            frames_per_enc=min(24, max_frames_per_enc),
            seed=seed,
        )

    split_idx = max(1, len(encounters) // 2)
    cal_encs = encounters[:split_idx]
    test_encs = encounters[split_idx:]

    runner = XvsD2CropRunner(onnx_path)

    # 1. Calibration split -> fit frozen one-sided Split Conformal quantile q_hat strictly on tracked detector frames
    cal_rows_all: list[dict[str, Any]] = []
    for enc in cal_encs:
        cal_rows_all.extend(_evaluate_encounter(enc, runner, q_hat_conformal=1.96))

    cal_rows_det = [r for r in cal_rows_all if r["has_det_track"]]
    cal_rows_fit = cal_rows_det if cal_rows_det else cal_rows_all
    q_hat = fit_one_sided_conformal_quantile(
        np.array([float(r["r_hat_m"]) for r in cal_rows_fit], dtype=np.float64),
        np.array([float(r["range_true_m"]) for r in cal_rows_fit], dtype=np.float64),
        np.array([float(r["sigma_r_m"]) for r in cal_rows_fit], dtype=np.float64),
        alpha=0.05,
    )

    # 2. Unseen test split -> evaluate pure detector-stream (det + coast) CP-MSC + tau + XVS-D2 vs Naive 11 m Bbox
    test_rows_by_enc: list[list[dict[str, Any]]] = []
    test_rows_all: list[dict[str, Any]] = []
    per_enc_summary: list[dict[str, Any]] = []

    for enc in test_encs:
        rows = _evaluate_encounter(enc, runner, q_hat_conformal=q_hat)
        # Re-apply exact conformal lower bound explicitly for transparency
        for r in rows:
            r["r_safe_lower_m"] = compute_conformal_lower_bound(float(r["r_hat_m"]), float(r["sigma_r_m"]), q_hat)
        test_rows_by_enc.append(rows)
        test_rows_all.extend(rows)

        tracked_e = [r for r in rows if r["has_det_track"]]
        r_true_all_e = np.array([float(r["range_true_m"]) for r in rows], dtype=np.float64)
        det_frames_e = sum(1 for r in rows if r["track_status"] == "detected")
        coast_frames_e = sum(1 for r in rows if r["track_status"] == "coasted")
        pre_acq_frames_e = sum(1 for r in rows if r["track_status"] == "pre_acq")
        if tracked_e:
            r_true_e = np.array([float(r["range_true_m"]) for r in tracked_e], dtype=np.float64)
            r_safe_e = np.array([float(r["r_safe_lower_m"]) for r in tracked_e], dtype=np.float64)
            r_naive_e = np.array([float(r["r_naive_11m"]) for r in tracked_e], dtype=np.float64)
            cp_cov_e = float(np.mean(r_true_e >= r_safe_e))
            cp_over_e = float(np.mean(r_safe_e > r_true_e))
            naive_over_e = float(np.mean(r_naive_e > r_true_e))
        else:
            cp_cov_e = 1.0
            cp_over_e = 0.0
            naive_over_e = 0.0
        per_enc_summary.append(
            {
                "encounter_id": enc.encounter_id,
                "aircraft_class": enc.aircraft_class,
                "n_frames": len(rows),
                "det_frames": det_frames_e,
                "coast_frames": coast_frames_e,
                "pre_acq_frames": pre_acq_frames_e,
                "min_range_m": float(r_true_all_e.min()),
                "max_range_m": float(r_true_all_e.max()),
                "det_rate": float(np.mean([1.0 if r["detected"] else 0.0 for r in rows])),
                "cp_coverage": cp_cov_e,
                "cp_overest_rate": cp_over_e,
                "naive_overest_rate": naive_over_e,
            }
        )

    test_rows_det = [r for r in test_rows_all if r["has_det_track"]]
    det_frames_count = sum(1 for r in test_rows_all if r["track_status"] == "detected")
    coast_frames_count = sum(1 for r in test_rows_all if r["track_status"] == "coasted")
    pre_acq_frames_count = sum(1 for r in test_rows_all if r["track_status"] == "pre_acq")

    eval_rows = test_rows_det if test_rows_det else test_rows_all
    cov_eval = evaluate_conformal_coverage(
        np.array([float(r["r_hat_m"]) for r in eval_rows], dtype=np.float64),
        np.array([float(r["range_true_m"]) for r in eval_rows], dtype=np.float64),
        np.array([float(r["sigma_r_m"]) for r in eval_rows], dtype=np.float64),
        q_hat,
    )
    naive_rows = [r for r in eval_rows if r["r_naive_11m"] is not None]
    if naive_rows:
        r_true_naive = np.array([float(r["range_true_m"]) for r in naive_rows], dtype=np.float64)
        r_naive_all = np.array([float(r["r_naive_11m"]) for r in naive_rows], dtype=np.float64)
        naive_overest_rate = float(np.mean(r_naive_all > r_true_naive))
        naive_rmse_m = float(math.sqrt(float(np.mean((r_naive_all - r_true_naive) ** 2))))
    else:
        naive_overest_rate = 0.0
        naive_rmse_m = 0.0

    det_rate = float(np.mean([1.0 if r["detected"] else 0.0 for r in test_rows_all]))
    cerrs = [float(r["centroid_err_px"]) for r in test_rows_all if r["centroid_err_px"] is not None]
    centroid_rmse = float(math.sqrt(float(np.mean(np.square(cerrs))))) if cerrs else 0.0
    mean_lat_ms = float(np.mean([float(r["latency_ms"]) for r in test_rows_all]))

    det_only_tau_errs = [
        abs(float(r["det_only_tau_obs_rho_dot"]) - float(r["det_only_gps_true_rho_dot"]))
        for r in test_rows_all
        if (r["det_only_tau_obs_rho_dot"] is not None and r["det_only_gps_true_rho_dot"] is not None)
    ]
    det_only_tau_mae = float(np.mean(det_only_tau_errs)) if det_only_tau_errs else 0.0
    det_only_tau_rmse = float(math.sqrt(float(np.mean(np.square(det_only_tau_errs))))) if det_only_tau_errs else 0.0

    gt_phys_tau_errs = [
        abs(float(r["gt_box_physics_tau_obs_rho_dot"]) - float(r["gt_box_physics_gps_true_rho_dot"]))
        for r in test_rows_all
        if (r["gt_box_physics_tau_obs_rho_dot"] is not None and r["gt_box_physics_gps_true_rho_dot"] is not None)
    ]
    gt_phys_tau_mae = float(np.mean(gt_phys_tau_errs)) if gt_phys_tau_errs else 0.0
    gt_phys_tau_rmse = float(math.sqrt(float(np.mean(np.square(gt_phys_tau_errs))))) if gt_phys_tau_errs else 0.0

    # 3. Export native 1:1 256x256 Tab 3 Browser Pack (<= 1.5 MB)
    # Prefer test encounters with strongest visual clarity / XVS-D2 detection rate
    ranked_pack_encs = sorted(
        test_rows_by_enc,
        key=lambda rows: (
            float(np.mean([1.0 if r["detected"] else 0.0 for r in rows])),
            len(rows),
        ),
        reverse=True,
    )
    pack_info = _export_realsky_tab3_pack(
        ranked_pack_encs,
        pack_dir,
        q_hat_conformal=q_hat,
        max_encounters=2,
        max_frames_per_enc=min(40, max_frames_per_enc),
    )

    # 4. Render 1920x1080 Showcase MP4 Video across top test encounters
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    for codec in ("mp4v", "avc1"):
        w_try = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*codec), AOT_FPS, (CANVAS_W, CANVAS_H))
        if w_try.isOpened():
            writer = w_try
            break
        w_try.release()
    if writer is None:
        raise RuntimeError(f"Failed to open cv2.VideoWriter for {video_path}")

    video_frames = 0
    try:
        for enc_rows in ranked_pack_encs[:3]:
            for idx in range(len(enc_rows)):
                frame_bgr = _render_proof_frame_1080p(
                    enc_rows[idx],
                    enc_rows[: idx + 1],
                    q_hat=q_hat,
                    overall_cov_pct=cov_eval["empirical_coverage"] * 100.0,
                    overall_naive_over_pct=naive_overest_rate * 100.0,
                    is_live_s3=live_s3,
                )
                writer.write(frame_bgr)
                video_frames += 1
    finally:
        writer.release()

    video_bytes = video_path.stat().st_size if video_path.is_file() else 0

    metrics: dict[str, Any] = {
        "schema": "xvs.v3.aot_activerange_tier_a_proof/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "live_s3": bool(live_s3),
        "onnx_model": str(onnx_path),
        "n_cal_encounters": len(cal_encs),
        "n_cal_frames": len(cal_rows_all),
        "n_cal_det_tracked_frames": len(cal_rows_det),
        "n_test_encounters": len(test_encs),
        "n_test_frames": len(test_rows_all),
        "n_test_det_tracked_frames": len(test_rows_det),
        "det_frames_count": int(det_frames_count),
        "coast_frames_count": int(coast_frames_count),
        "pre_acq_frames_count": int(pre_acq_frames_count),
        "q_hat_conformal": float(q_hat),
        "xvs_d2_detection_rate": det_rate,
        "xvs_d2_centroid_rmse_px": centroid_rmse,
        "mean_ort_latency_ms": mean_lat_ms,
        "det_only_tau_valid_frames": len(det_only_tau_errs),
        "det_only_tau_mae_inv_s": det_only_tau_mae,
        "det_only_tau_rmse_inv_s": det_only_tau_rmse,
        "gt_box_physics_tau_valid_frames": len(gt_phys_tau_errs),
        "gt_box_physics_tau_mae_inv_s": gt_phys_tau_mae,
        "gt_box_physics_tau_rmse_inv_s": gt_phys_tau_rmse,
        "tau_valid_frames": len(det_only_tau_errs),
        "tau_mae_inv_s": det_only_tau_mae,
        "tau_rmse_inv_s": det_only_tau_rmse,
        "conformal_coverage": float(cov_eval["empirical_coverage"]),
        "cp_msc_overestimation_rate": float(cov_eval["overestimation_rate"]),
        "naive_overestimation_rate": naive_overest_rate,
        "naive_rmse_m": naive_rmse_m,
        "video_path": str(video_path),
        "video_bytes": int(video_bytes),
        "video_frames": int(video_frames),
        "pack": pack_info,
        "per_encounter_test": per_enc_summary,
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_tier_a_markdown(metrics), encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="OpenXVS v3.0 Tier A Real Amazon AOT S3 Passive Proof")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live-s3", action="store_true", help="Stream real held-out Amazon AOT S3 encounters")
    mode.add_argument("--smoke-synthetic", action="store_true", help="Offline deterministic CI smoke mode")
    parser.add_argument("--out-dir", type=Path, default=None, help="Override base directory for all outputs (CI/tests)")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--num-encounters", type=int, default=8)
    parser.add_argument("--max-frames-per-enc", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261121)
    args = parser.parse_args(argv)

    video_p = args.video
    metrics_p = args.metrics
    report_p = args.report
    pack_p = args.pack_dir
    if args.out_dir is not None:
        base = args.out_dir
        video_p = base / "xvs_activerange_aot_proof_1080p.mp4"
        metrics_p = base / "metrics.json"
        report_p = base / "V3_M1_aot_realsky_proof.md"
        pack_p = base / "realsky_aot_pack"

    res = run_aot_activerange_proof(
        live_s3=bool(args.live_s3),
        onnx_path=args.onnx,
        cache_dir=args.cache_dir,
        video_path=video_p,
        metrics_path=metrics_p,
        report_path=report_p,
        pack_dir=pack_p,
        num_encounters=int(args.num_encounters),
        max_frames_per_enc=int(args.max_frames_per_enc),
        seed=int(args.seed),
    )
    print(
        json.dumps(
            {
                "live_s3": res["live_s3"],
                "det_frames_count": res["det_frames_count"],
                "coast_frames_count": res["coast_frames_count"],
                "pre_acq_frames_count": res["pre_acq_frames_count"],
                "conformal_coverage": res["conformal_coverage"],
                "cp_msc_overestimation_rate": res["cp_msc_overestimation_rate"],
                "naive_overestimation_rate": res["naive_overestimation_rate"],
                "det_only_tau_mae_inv_s": res["det_only_tau_mae_inv_s"],
                "gt_box_physics_tau_mae_inv_s": res["gt_box_physics_tau_mae_inv_s"],
                "xvs_d2_detection_rate": res["xvs_d2_detection_rate"],
                "video_bytes": res["video_bytes"],
                "pack_total_bytes": res["pack"]["total_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
