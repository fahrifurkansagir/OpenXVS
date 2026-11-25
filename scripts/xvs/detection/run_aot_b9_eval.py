"""Batch B9 — Real AOT Background-Generalisation Evaluator (`WP04_DETECTION.md` §9.6, `D34-S1`, `D34-S6`, `B8-R3-F3`, `B8-Colab-F2..F3`).

Evaluates `D0`, `D1-SF`, and `D1-KF` on Airborne Object Tracking (AOT) flight sequences
using batched full-frame sliding-window tiled inference (`tile_size=256`, `overlap=32`,
`tile_batch_size=64`) on `cuda` / `cpu` with cross-tile NMS and classical 2D homography
background registration (`align_aot_kframe_stack_2d` at `dt = 1.0 s` / 10-frame step):
- Primary AOT profile (`evaluate_detections(..., profile="aot")`, `range_m <= 700.0`,
  `compute_eiou >= 0.15`, `min_area_px2 = 20.0`), including separate above-horizon vs below-horizon `Pd`.
- Secondary XVS profile (`evaluate_detections(..., profile="xvs")`, all ranges, `min_area_px2 = 0.0`),
  including separate above-horizon vs below-horizon `Pd`.
- False-alarm density in `FP / Mpx` and per-frame $\\lambda$ scaled to the 1.483 Mpx T1 ring.

Modes (`B8-R3-F3` — Zero Silent Fallback):
- `--mode strict` (default): Validates that `--aot-dir` exists and contains genuine AOT flight
  sequence directories + ground-truth metadata. Raises `FileNotFoundError` / `RuntimeError` if
  real AOT files are missing. Never falls back to G3a synthetic/pack data.
- `--mode check-readiness`: Checks whether `--aot-dir` is mounted with real AOT sequences. When
  not mounted locally, writes `output/xvs/wp04/g1/aot_b9_results.json` with
  `"status": "UNMEASURED_AWAITING_COLAB_S3"`, `"scratch_mounted": false`,
  `"real_aot_sequences_evaluated": 0`, and `null` metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.xvs.detection.run_g1_eval import load_d1_checkpoint_cpu_fp32
from xvs.detection.d0 import detect_d0, load_d0_config
from xvs.detection.d1_model import D1HeatmapDetector, extract_d1_detections
from xvs.detection.evaluate import (
    Detection,
    DetectionFrame,
    GTFrame,
    GTObject,
    evaluate_detections,
)
from xvs.detection.g3a_eval import T1_RING_EARTH_MPX, T1_RING_SKY_MPX
from xvs.detection.warp import (
    align_aot_kframe_stack_2d,
    build_prealigned_kframe_tensor,
    register_kframe_stack,
    validate_pose_dict,
)
from xvs.metrics.aot import clopper_pearson_interval, poisson_rate_upper_bound


def _tile_starts(full_dim: int, tile_size: int, overlap: int) -> list[int]:
    if full_dim <= tile_size:
        return [0]
    step = max(1, int(tile_size) - int(overlap))
    starts = list(range(0, full_dim - tile_size + 1, step))
    if starts[-1] != full_dim - tile_size:
        starts.append(full_dim - tile_size)
    return starts


def cross_tile_nms(
    detections: Sequence[Mapping[str, Any]],
    nms_radius_px: float = 4.0,
) -> list[dict[str, Any]]:
    """Suppress duplicate detections in overlapping sliding-window tile borders (`B8-R3-F3`)."""
    sorted_dets = sorted(detections, key=lambda d: float(d["score"]), reverse=True)
    kept: list[dict[str, Any]] = []
    for d in sorted_dets:
        du = float(d["u"])
        dv = float(d["v"])
        if any(math.hypot(du - float(k["u"]), dv - float(k["v"])) <= float(nms_radius_px) for k in kept):
            continue
        kept.append(dict(d))
    return kept


@torch.no_grad()
def infer_d1_full_frame_tiled(
    model: D1HeatmapDetector,
    frames: Sequence[np.ndarray],
    poses: Sequence[Mapping[str, Any] | None] | None = None,
    score_threshold: float = 0.35,
    image_id: str = "aot_0",
    camera: str = "T0",
    tile_size: int = 256,
    overlap: int = 32,
    nms_radius_px: float = 4.0,
    device: str | torch.device = "cpu",
    tile_batch_size: int = 64,
    prealigned_stack_tensor: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    """Run batched GPU/CPU sliding-window tiled `D1` inference across a full-frame AOT stack (`B8-Colab-F2`, `B8-Colab-F3`).

    - When `model.num_frames_k > 1` and `poses` is `None` (or contains `None` entries, as on real single-camera
      AOT sequences), aligns `{I_{t-K+1}, ..., I_t}` onto `I_t` via `align_aot_kframe_stack_2d` (`B8-Colab-F2`).
    - Stacks tiles into batches of `tile_batch_size` (`64` default) on `device` in `torch.no_grad()` (`B8-Colab-F3`).
    """
    dev = torch.device(device)
    eff_ablation = model.ablation_mode
    first_param = next(model.parameters(), None)
    if first_param is None or first_param.device != dev or first_param.dtype != torch.float32:
        model_eval = model.to(device=dev, dtype=torch.float32)
    else:
        model_eval = model
    model_eval.eval()

    h_full, w_full = frames[-1].shape
    xs = _tile_starts(w_full, tile_size, overlap)
    ys = _tile_starts(h_full, tile_size, overlap)

    has_explicit_6dof_poses = (
        poses is not None
        and len(poses) > 0
        and all(p is not None for p in poses)
    )

    full_c_tensor: torch.Tensor | None = prealigned_stack_tensor
    if full_c_tensor is None:
        if model.num_frames_k == 1:
            if has_explicit_6dof_poses and poses is not None:
                for p in poses:
                    if p is not None:
                        validate_pose_dict(p, ablation_mode=eff_ablation)
            full_c_tensor = torch.from_numpy(np.asarray(frames[-1], dtype=np.float32)).unsqueeze(0)
        elif not has_explicit_6dof_poses:
            if len(frames) != model.num_frames_k:
                raise ValueError(f"D1-KF requires {model.num_frames_k} frames, got {len(frames)}")
            aligned_frames, valid_masks = align_aot_kframe_stack_2d(frames)
            full_c_tensor = build_prealigned_kframe_tensor(aligned_frames, valid_masks)

    tile_specs: list[tuple[int, int, torch.Tensor]] = []
    for y0 in ys:
        for x0 in xs:
            y1 = min(h_full, y0 + tile_size)
            x1 = min(w_full, x0 + tile_size)
            if full_c_tensor is not None:
                t_tile = full_c_tensor[:, y0:y1, x0:x1]
            else:
                assert poses is not None
                valid_poses = [p for p in poses if p is not None]
                tile_frames = [np.asarray(f[y0:y1, x0:x1], dtype=np.float32) for f in frames]
                tile_origins = [(float(x0), float(y0)) for _ in frames]
                reg = register_kframe_stack(
                    frames=tile_frames,
                    poses=valid_poses,
                    roi_origins=tile_origins,
                    ablation_mode=eff_ablation,
                )
                t_tile = reg["input_tensor"]
            tile_specs.append((x0, y0, t_tile))

    b_sz = max(1, int(tile_batch_size))
    all_dets: list[dict[str, Any]] = []
    for b_start in range(0, len(tile_specs), b_sz):
        chunk = tile_specs[b_start : b_start + b_sz]
        batch_in = torch.stack([item[2] for item in chunk], dim=0).to(device=dev, dtype=torch.float32)
        batch_out = model_eval(batch_in)
        hm_cpu = batch_out["heatmap"].detach().cpu()
        off_cpu = batch_out["offset"].detach().cpu()
        lsc_cpu = batch_out["log_scale"].detach().cpu()

        for idx_in_chunk, (x0, y0, _) in enumerate(chunk):
            single_out = {
                "heatmap": hm_cpu[idx_in_chunk : idx_in_chunk + 1],
                "offset": off_cpu[idx_in_chunk : idx_in_chunk + 1],
                "log_scale": lsc_cpu[idx_in_chunk : idx_in_chunk + 1],
            }
            dets_tile = extract_d1_detections(
                outputs=single_out,
                score_threshold=score_threshold,
                roi_u0=float(x0),
                roi_v0=float(y0),
                image_id=f"{image_id}_{x0}_{y0}",
                camera=camera,
            )
            all_dets.extend(dets_tile)

    return cross_tile_nms(all_dets, nms_radius_px=nms_radius_px)


def validate_real_aot_directory(
    aot_dir: Path,
    allow_unit_test_fixture: bool = False,
) -> dict[str, Any]:
    """Verify that `aot_dir` exists and contains genuine AOT flight sequence directories + ground truth (`B8-R3-F3`)."""
    if not aot_dir.exists() or not aot_dir.is_dir():
        raise FileNotFoundError(
            f"Real AOT directory not found at {aot_dir}. "
            "Silent fallback to G3a pack is prohibited (B8-R3-F3). "
            "Mount /content/aot via AWS S3 on Colab or run with --mode check-readiness."
        )

    gt_manifest_path = aot_dir / "aot_ground_truth.json"
    if not gt_manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing AOT ground-truth manifest at {gt_manifest_path}. "
            "Directory does not contain valid AOT flight sequences."
        )

    gt_doc = json.loads(gt_manifest_path.read_text(encoding="utf-8"))
    dataset_id = str(gt_doc.get("dataset_id", ""))
    is_synthetic = bool(gt_doc.get("synthetic", False))
    is_fixture = bool(gt_doc.get("unit_test_fixture", False))

    if is_synthetic:
        raise RuntimeError(
            f"Rejected synthetic dataset in {aot_dir} (synthetic=True). "
            "run_aot_b9_eval.py --mode strict requires real AOT sequences."
        )
    if is_fixture and not allow_unit_test_fixture:
        raise RuntimeError(
            f"Rejected unit-test fixture in {aot_dir} without --allow-unit-test-fixture."
        )
    if not dataset_id.startswith("AOT"):
        raise RuntimeError(
            f"Invalid dataset_id={dataset_id!r} in {gt_manifest_path}; expected 'AOT-v1'."
        )

    sequences = gt_doc.get("sequences", [])
    if not isinstance(sequences, list) or len(sequences) == 0:
        raise RuntimeError(f"Zero AOT flight sequences listed in {gt_manifest_path}.")

    validated_sequences: list[dict[str, Any]] = []
    for seq in sequences:
        seq_id = str(seq["sequence_id"])
        frames_meta = seq.get("frames", [])
        if not frames_meta:
            raise RuntimeError(f"AOT sequence {seq_id} has zero frames in {gt_manifest_path}.")
        for fm in frames_meta:
            rel_img = str(fm["image_path"])
            img_path = aot_dir / rel_img
            if not img_path.is_file():
                raise FileNotFoundError(f"Missing AOT frame file {img_path} for sequence {seq_id}.")
        validated_sequences.append(seq)

    return {
        "dataset_id": dataset_id,
        "unit_test_fixture": is_fixture,
        "sequences": validated_sequences,
    }


def _load_aot_frame_gray(img_path: Path) -> np.ndarray:
    if img_path.suffix.lower() == ".npy":
        return np.asarray(np.load(img_path), dtype=np.float32)
    with Image.open(img_path) as im:
        arr = np.asarray(im.convert("L"), dtype=np.float32)
    return arr


def _select_dt_1s_stack_indices(
    frames_list: Sequence[Mapping[str, Any]],
    curr_idx: int,
    k_need: int,
    dt_target_s: float = 1.0,
) -> list[int]:
    """Select `k_need` indices ending at `curr_idx` separated by `dt = 1.0 s` (`10`-frame step at 10 Hz)."""
    if k_need <= 1:
        return [curr_idx]
    t_curr = float(frames_list[curr_idx].get("t_s", float(curr_idx)))
    f_curr = frames_list[curr_idx].get("aot_frame")
    selected_indices: list[int] = []
    for lag in range(int(k_need) - 1, 0, -1):
        if f_curr is not None:
            want_f = int(f_curr) - int(round(lag * dt_target_s * 10.0))
            best_i = min(
                range(0, curr_idx + 1),
                key=lambda i: abs(int(frames_list[i].get("aot_frame", i * 10)) - want_f),
            )
        else:
            want_t = t_curr - float(lag) * float(dt_target_s)
            best_i = min(
                range(0, curr_idx + 1),
                key=lambda i: abs(float(frames_list[i].get("t_s", float(i))) - want_t),
            )
        selected_indices.append(int(best_i))
    selected_indices.append(int(curr_idx))
    return selected_indices


def _compute_horizon_breakdown_within_range(
    gt_frames: Sequence[GTFrame],
    matched_pairs: Sequence[Mapping[str, Any]],
    max_range_m: float,
) -> dict[str, dict[str, Any]]:
    """Compute stratified `above` / `below` horizon `Pd` and 95% Clopper-Pearson CI within `max_range_m`."""
    matched_keys = {
        (str(mp["frame_id"]), str(mp["camera_id"]), str(mp["object_id"]))
        for mp in matched_pairs
    }
    stats: dict[str, dict[str, Any]] = {}
    for hz in ("above", "below"):
        stats[hz] = {"n_instances": 0, "n_detected": 0, "encounters": set()}

    for gtf in gt_frames:
        for obj in gtf.objects:
            if float(obj.range_m) > float(max_range_m):
                continue
            hz_key = str(obj.horizon).lower()
            bucket = stats.setdefault(hz_key, {"n_instances": 0, "n_detected": 0, "encounters": set()})
            bucket["n_instances"] += 1
            bucket["encounters"].add(str(gtf.sequence_id))
            if (str(gtf.frame_id), str(gtf.camera_id), str(obj.object_id)) in matched_keys:
                bucket["n_detected"] += 1

    out: dict[str, dict[str, Any]] = {}
    for hz_key, b in sorted(stats.items()):
        n_inst = int(b["n_instances"])
        n_det = int(b["n_detected"])
        if n_inst > 0:
            pd_val = float(n_det / n_inst)
            ci_lo, ci_hi = clopper_pearson_interval(n_det, n_inst, confidence=0.95)
        else:
            pd_val = None
            ci_lo, ci_hi = 0.0, 0.0
        out[hz_key] = {
            "n_instances": n_inst,
            "n_detected": n_det,
            "n_encounters": len(b["encounters"]),
            "pd": pd_val,
            "ci95_low": float(ci_lo),
            "ci95_high": float(ci_hi),
        }
    return out


def evaluate_real_aot_sequences(
    aot_dir: Path,
    aot_meta: Mapping[str, Any],
    sf_model: D1HeatmapDetector,
    sf_tau: float,
    kf_model: D1HeatmapDetector,
    kf_tau: float,
    tile_size: int = 256,
    overlap: int = 32,
    device: str | torch.device = "cpu",
    tile_batch_size: int = 64,
) -> dict[str, Any]:
    """Run batched tiled full-frame inference and AOT/XVS box evaluation across validated AOT sequences (`B8-Colab-F2..F3`)."""
    d0_cfg = load_d0_config()
    ring_area_mpx = float(T1_RING_SKY_MPX + T1_RING_EARTH_MPX)
    dev = torch.device(device)
    sf_model_dev = sf_model.to(device=dev, dtype=torch.float32).eval()
    kf_model_dev = kf_model.to(device=dev, dtype=torch.float32).eval()

    gt_frames: list[GTFrame] = []
    det_frames_by_model: dict[str, list[DetectionFrame]] = {
        "d0": [],
        "d1_sf": [],
        "d1_kf": [],
    }
    fp_counts_by_model: dict[str, int] = {"d0": 0, "d1_sf": 0, "d1_kf": 0}
    total_area_px = 0.0
    seq_ids: list[str] = []

    for seq in aot_meta["sequences"]:
        seq_id = str(seq["sequence_id"])
        seq_ids.append(seq_id)
        frames_list = seq["frames"]

        loaded_imgs: list[np.ndarray] = []
        loaded_poses: list[dict[str, Any] | None] = []
        for fm in frames_list:
            img = _load_aot_frame_gray(aot_dir / str(fm["image_path"]))
            loaded_imgs.append(img)
            raw_pose = fm.get("pose")
            loaded_poses.append(dict(raw_pose) if isinstance(raw_pose, Mapping) else None)

        for idx, fm in enumerate(frames_list):
            frame_id = str(fm.get("frame_id", f"{seq_id}_{idx:04d}"))
            t_s = float(fm.get("t_s", float(idx)))
            img_curr = loaded_imgs[idx]
            h_f, w_f = img_curr.shape
            total_area_px += float(h_f * w_f)

            gt_objs: list[GTObject] = []
            gt_boxes: list[list[float]] = []
            for obj in fm.get("objects", []):
                bx = [float(x) for x in obj["box_edges"]]
                gt_boxes.append(bx)
                gt_objs.append(
                    GTObject(
                        object_id=str(obj.get("object_id", f"{frame_id}_obj")),
                        class_id=str(obj.get("class_id", "airborne")),
                        range_m=float(obj.get("range_m", 500.0)),  # default-ok: nominal AOT range fallback when metadata omits range_m
                        box_edges=bx,
                        horizon=str(obj.get("horizon", "above")),
                    )
                )

            gt_frames.append(
                GTFrame(
                    frame_id=frame_id,
                    t_s=t_s,
                    sequence_id=seq_id,
                    camera_id="T0",
                    image_wh=(int(w_f), int(h_f)),
                    objects=gt_objs,
                )
            )

            # 1. D0 baseline
            dets_d0 = detect_d0(
                np.asarray(img_curr, dtype=np.float64),
                is_clean=False,
                k=3,
                tau=20.0,
                u0=0.0,
                v0=0.0,
                image_id=frame_id,
                camera="T0",
                config=d0_cfg,
            )

            # 2. D1-SF batched tiled inference on `device`
            sf_poses_arg = [loaded_poses[idx]] if loaded_poses[idx] is not None else None
            dets_sf = infer_d1_full_frame_tiled(
                model=sf_model_dev,
                frames=[img_curr],
                poses=sf_poses_arg,
                score_threshold=sf_tau,
                image_id=frame_id,
                tile_size=tile_size,
                overlap=overlap,
                device=dev,
                tile_batch_size=tile_batch_size,
            )

            # 3. D1-KF batched tiled inference with dt = 1.0 s (10-frame step) + 2D homography registration
            k_need = int(kf_model_dev.num_frames_k)
            stack_idxs = _select_dt_1s_stack_indices(frames_list, idx, k_need=k_need, dt_target_s=1.0)
            kf_imgs = [loaded_imgs[b_i] for b_i in stack_idxs]
            kf_poses_raw = [loaded_poses[b_i] for b_i in stack_idxs]
            kf_poses_arg = kf_poses_raw if all(p is not None for p in kf_poses_raw) else None

            dets_kf = infer_d1_full_frame_tiled(
                model=kf_model_dev,
                frames=kf_imgs,
                poses=kf_poses_arg,
                score_threshold=kf_tau,
                image_id=frame_id,
                tile_size=tile_size,
                overlap=overlap,
                device=dev,
                tile_batch_size=tile_batch_size,
            )

            for m_key, raw_dets in (("d0", dets_d0), ("d1_sf", dets_sf), ("d1_kf", dets_kf)):
                det_objs: list[Detection] = []
                for d in raw_dets:
                    du = float(d["u"])
                    dv = float(d["v"])
                    be = d.get("box_edges")
                    if be is None:
                        s_d = max(4.5, float(d.get("s_det", 4.5)))  # default-ok: minimum 4.5 px box side when detector emits point centroid
                        be = [du - 0.5 * s_d, dv - 0.5 * s_d, du + 0.5 * s_d, dv + 0.5 * s_d]
                    else:
                        be = [float(x) for x in be]
                        if (be[2] - be[0]) * (be[3] - be[1]) < 20.25:
                            cx = 0.5 * (be[0] + be[2])
                            cy = 0.5 * (be[1] + be[3])
                            be = [cx - 2.25, cy - 2.25, cx + 2.25, cy + 2.25]
                    det_objs.append(
                        Detection(
                            box=be,
                            score=float(d.get("score", d.get("snr", 1.0))),  # default-ok: unit fallback score when neither score nor snr key exists
                        )
                    )
                    in_any_gt = any(
                        bx[0] - 16.0 <= du <= bx[2] + 16.0 and bx[1] - 16.0 <= dv <= bx[3] + 16.0
                        for bx in gt_boxes
                    )
                    if not in_any_gt:
                        fp_counts_by_model[m_key] += 1

                det_frames_by_model[m_key].append(
                    DetectionFrame(frame_id=frame_id, camera_id="T0", detections=det_objs)
                )

    area_mpx = max(1e-6, total_area_px / 1.0e6)
    by_model: dict[str, dict[str, Any]] = {}
    for m_key in ("d0", "d1_sf", "d1_kf"):
        ev_aot = evaluate_detections(gt_frames=gt_frames, det_frames=det_frames_by_model[m_key], profile="aot")
        ev_xvs = evaluate_detections(gt_frames=gt_frames, det_frames=det_frames_by_model[m_key], profile="xvs")
        hz_aot = _compute_horizon_breakdown_within_range(gt_frames, ev_aot["matched_pairs"], max_range_m=700.0)
        hz_xvs = _compute_horizon_breakdown_within_range(gt_frames, ev_xvs["matched_pairs"], max_range_m=float("inf"))

        fp_cnt = int(fp_counts_by_model[m_key])
        fp_per_mpx = float(fp_cnt) / area_mpx
        lambda_frame = fp_per_mpx * ring_area_mpx
        lambda_ub95 = float(poisson_rate_upper_bound(fp_cnt, exposure=area_mpx / ring_area_mpx))
        by_model[m_key] = {
            "overall_pd_aot_700m": float(ev_aot["overall_pd"]),
            "overall_pd_xvs": float(ev_xvs["overall_pd"]),
            "pd_above_horizon_aot_700m": hz_aot["above"]["pd"],
            "pd_below_horizon_aot_700m": hz_aot["below"]["pd"],
            "pd_above_horizon_xvs": hz_xvs["above"]["pd"],
            "pd_below_horizon_xvs": hz_xvs["below"]["pd"],
            "by_horizon_aot_700m": hz_aot,
            "by_horizon_xvs": hz_xvs,
            "fp_count": fp_cnt,
            "evaluated_area_mpx": float(area_mpx),
            "fp_per_mpx": float(fp_per_mpx),
            "lambda_per_camera_frame": float(lambda_frame),
            "lambda_ub95_per_camera_frame": float(lambda_ub95),
        }

    kf_res = by_model["d1_kf"]
    return {
        "schema": "xvs.detection_aot_b9_eval/1",
        "batch_id": "B9",
        "status": "MEASURED_REAL_AOT",
        "status_summary": f"MEASURED_REAL_AOT ({len(seq_ids)} real AOT sequences evaluated)",
        "scratch_mounted": True,
        "aot_dir": str(aot_dir),
        "dataset_id": str(aot_meta["dataset_id"]),
        "device": str(dev),
        "tile_batch_size": int(tile_batch_size),
        "registration_mode": "homography_2d_ransac_with_phase_fallback",
        "real_aot_sequences_evaluated": len(seq_ids),
        "sequence_ids": seq_ids,
        "tile_size": int(tile_size),
        "tile_overlap": int(overlap),
        "overall_pd_aot_700m": kf_res["overall_pd_aot_700m"],
        "overall_pd_xvs": kf_res["overall_pd_xvs"],
        "pd_above_horizon_aot_700m": kf_res["pd_above_horizon_aot_700m"],
        "pd_below_horizon_aot_700m": kf_res["pd_below_horizon_aot_700m"],
        "pd_above_horizon_xvs": kf_res["pd_above_horizon_xvs"],
        "pd_below_horizon_xvs": kf_res["pd_below_horizon_xvs"],
        "fp_per_mpx": kf_res["fp_per_mpx"],
        "by_model": by_model,
    }


def _unmeasured_readiness_payload(aot_dir: Path) -> dict[str, Any]:
    return {
        "schema": "xvs.detection_aot_b9_eval/1",
        "batch_id": "B9",
        "status": "UNMEASURED_AWAITING_COLAB_S3",
        "status_summary": "UNMEASURED ON LOCAL HOST (0 real AOT sequences; Colab S3 streaming pipeline staged)",
        "scratch_mounted": False,
        "aot_dir": str(aot_dir),
        "real_aot_sequences_evaluated": 0,
        "overall_pd_aot_700m": None,
        "overall_pd_xvs": None,
        "pd_above_horizon_aot_700m": None,
        "pd_below_horizon_aot_700m": None,
        "pd_above_horizon_xvs": None,
        "pd_below_horizon_xvs": None,
        "fp_per_mpx": None,
        "by_model": None,
    }


def main(argv: list[str] | None = None) -> int:
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    torch.set_num_threads(2)

    parser = argparse.ArgumentParser(description="Batch B9 AOT background-generalisation evaluation (B8-R3-F3, B8-Colab-F2..F3).")
    parser.add_argument(
        "--aot-dir",
        "--aot-scratch-dir",
        dest="aot_dir",
        type=Path,
        default=Path("/content/aot"),
        help="Path to mounted real AOT flight sequences directory (default /content/aot).",
    )
    parser.add_argument(
        "--mode",
        choices=("strict", "check-readiness"),
        default="strict",
        help="'strict' requires real AOT sequences in --aot-dir; 'check-readiness' records UNMEASURED_AWAITING_COLAB_S3 if unmounted.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=REPO_ROOT / "output" / "xvs" / "wp04" / "g1" / "checkpoints",
        help="Directory containing trained D1-SF and D1-KF checkpoints.",
    )
    parser.add_argument("--tile-size", type=int, default=256, help="Sliding-window tile size (px).")
    parser.add_argument("--overlap", type=int, default=32, help="Sliding-window tile overlap (px).")
    parser.add_argument(
        "--tile-batch-size",
        type=int,
        default=64,
        help="Number of sliding-window tiles stacked per GPU/CPU forward pass (default 64).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch inference device ('cuda' when available on Colab A100, else 'cpu').",
    )
    parser.add_argument(
        "--allow-unit-test-fixture",
        action="store_true",
        help="Allow unit-test fixture AOT directory (only permitted when --out is outside output/xvs/).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "output" / "xvs" / "wp04" / "g1" / "aot_b9_results.json",
    )
    args = parser.parse_args(argv)

    if args.mode == "check-readiness":
        try:
            aot_meta = validate_real_aot_directory(args.aot_dir, allow_unit_test_fixture=False)
        except (FileNotFoundError, RuntimeError):
            payload = _unmeasured_readiness_payload(args.aot_dir)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(payload, indent=2))
            return 0
    else:
        aot_meta = validate_real_aot_directory(
            args.aot_dir,
            allow_unit_test_fixture=bool(args.allow_unit_test_fixture),
        )

    if aot_meta.get("unit_test_fixture", False):
        out_resolved = args.out.resolve()
        repo_out_resolved = (REPO_ROOT / "output").resolve()
        if str(out_resolved).startswith(str(repo_out_resolved)):
            raise RuntimeError(
                f"Unit-test fixture results may never be written to {repo_out_resolved} (got {out_resolved})."
            )

    sf_ckpt_path = args.checkpoints_dir / "d1_sf_synth.pt"
    if not sf_ckpt_path.is_file():
        sf_ckpt_path = args.checkpoints_dir / "best_model.pt"
    kf_ckpt_path = args.checkpoints_dir / "d1_kf_k3_synth.pt"
    if not kf_ckpt_path.is_file():
        kf_ckpt_path = args.checkpoints_dir / "best_model.pt"

    sf_model, sf_meta = load_d1_checkpoint_cpu_fp32(sf_ckpt_path)
    kf_model, kf_meta = load_d1_checkpoint_cpu_fp32(kf_ckpt_path)
    sf_tau = float(sf_meta.get("tau_star", 0.45))  # default-ok: fallback operating threshold when checkpoint omits tau_star
    kf_tau = float(kf_meta.get("tau_star", 0.45))  # default-ok: fallback operating threshold when checkpoint omits tau_star

    payload = evaluate_real_aot_sequences(
        aot_dir=args.aot_dir,
        aot_meta=aot_meta,
        sf_model=sf_model,
        sf_tau=sf_tau,
        kf_model=kf_model,
        kf_tau=kf_tau,
        tile_size=int(args.tile_size),
        overlap=int(args.overlap),
        device=args.device,
        tile_batch_size=int(args.tile_batch_size),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "out": str(args.out),
                "device": payload["device"],
                "real_aot_sequences_evaluated": payload["real_aot_sequences_evaluated"],
                "d1_kf_pd_aot_700m": payload["overall_pd_aot_700m"],
                "d1_kf_pd_above_aot_700m": payload["pd_above_horizon_aot_700m"],
                "d1_kf_pd_below_aot_700m": payload["pd_below_horizon_aot_700m"],
                "d1_kf_fp_per_mpx": payload["fp_per_mpx"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
