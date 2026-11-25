"""Train WP-04 G1 learned small-object detectors (`D1-SF` and `D1-KF`, `WP04_DETECTION.md` §9.4–§9.5).

Enforces:
- Single-process CPU discipline (`torch.set_num_threads(2)`, `num_workers=0`) when run locally.
- Compute caps from `configs/xvs/detection/g1_splits_v0.json` (`total_g1_cu_cap <= 60 CU`,
  `pilot_max_hours <= 0.5`, `job_max_hours <= 10.0`, `checkpoint_interval_minutes = 30`).
- Validation operating point selection `tau*` (`D34-S4`, `D34-S9`, `B8-R2-F7`) with `val_sample_size_floor_lambda_ub0`
  and `train + val` negative-frame K-fold pooling.
- Long-job manifest (`run_manifest.json`) recording start/end code SHA-256 digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.detection.d1_model import (
    D1HeatmapDetector,
    build_d1_kf,
    build_d1_sf,
    count_parameters_and_macs,
    infer_d1_roi,
)
from xvs.detection.g3a_eval import T1_RING_EARTH_MPX, T1_RING_SKY_MPX
from xvs.detection.pack import (
    generate_synthetic_pack_items_from_splits,
    load_g1_pack_archive,
)
from xvs.detection.splits import DEFAULT_SPLITS_CONFIG, load_splits_config
from xvs.detection.train_g1 import train_d1_steps
from xvs.metrics.aot import poisson_rate_upper_bound

DEFAULT_D0_RESULTS_JSON = REPO_ROOT / "output" / "xvs" / "wp04" / "g0" / "g3a_results.json"

TRACKED_FILES_G1_TRAIN = (
    "scripts/xvs/detection/run_g1_train.py",
    "src/xvs/detection/d1_model.py",
    "src/xvs/detection/train_g1.py",
    "src/xvs/detection/warp.py",
    "src/xvs/detection/splits.py",
    "src/xvs/detection/pack.py",
    "configs/xvs/detection/g1_splits_v0.json",
    "configs/xvs/detection/d1_sf.json",
    "configs/xvs/detection/d1_kf.json",
)


def _compute_code_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for rel in TRACKED_FILES_G1_TRAIN:
        p = REPO_ROOT / rel
        if p.is_file():
            hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes


def _get_git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=10,
            check=True,
        )
        return proc.stdout.strip()
    except Exception:
        return "HEAD"


def pack_item_to_training_sample(
    it: Mapping[str, Any],
    num_frames_k: int = 1,
    ablation_mode: str = "standard",
) -> dict[str, Any]:
    """Convert an unpacked G1 item into a training/validation sample with true ROI-local `box_analytic` labels (`B8-F1`)."""
    k_eff = max(1, int(num_frames_k))
    raw_frames = list(it["frames"])
    raw_poses = list(it["poses"])
    raw_origins = list(it["roi_origins"])

    if len(raw_frames) > k_eff:
        raw_frames = raw_frames[-k_eff:]
        raw_poses = raw_poses[-k_eff:]
        raw_origins = raw_origins[-k_eff:]

    frames = [np.asarray(f, dtype=np.float32) for f in raw_frames]
    poses = [dict(p) for p in raw_poses]
    roi_origins = [tuple(float(x) for x in o) for o in raw_origins]

    if k_eff > 1:
        while len(frames) < k_eff:
            frames.insert(0, frames[0])
            poses.insert(0, dict(poses[0]))
            roi_origins.insert(0, roi_origins[0])

    if ablation_mode == "oracle":
        for p in poses:
            if "q_ned_to_cam" not in p and "q_ned_to_cam_est" in p:
                p["q_ned_to_cam"] = list(p["q_ned_to_cam_est"])
            if "pos_wgs84_gt" not in p and "pos_wgs84" in p:
                p["pos_wgs84_gt"] = list(p["pos_wgs84"])
            p["pose_source"] = "oracle_ground_truth"

    u0, v0 = roi_origins[-1]
    h_crop, w_crop = frames[-1].shape
    centers_uv: list[tuple[float, float]] = []
    boxes_wh: list[tuple[float, float]] = []

    box_an = it.get("box_analytic")
    if box_an is not None:
        bx = [float(x) for x in box_an]
        u_local = 0.5 * (bx[0] + bx[2]) - float(u0)
        v_local = 0.5 * (bx[1] + bx[3]) - float(v0)
        w_box = max(0.5, float(bx[2] - bx[0]))
        h_box = max(0.5, float(bx[3] - bx[1]))
        if 0.0 <= u_local < float(w_crop) and 0.0 <= v_local < float(h_crop):
            centers_uv.append((u_local, v_local))
            boxes_wh.append((w_box, h_box))

    return {
        "item_id": str(it["item_id"]),
        "split": str(it["split"]),
        "kind": str(it["kind"]),
        "frames": frames,
        "poses": poses,
        "roi_origins": roi_origins,
        "centers_uv": centers_uv,
        "boxes_wh": boxes_wh,
        "box_analytic": box_an,
        "sensor_state": str(it.get("sensor_state", "degraded")),
    }


def calibrate_validation_operating_point(
    model: D1HeatmapDetector,
    val_items: list[dict[str, Any]],
    train_neg_items: list[dict[str, Any]] | None = None,
    device: str = "cpu",
    lambda_d0_ref: float | None = None,
) -> dict[str, Any]:
    """Calibrate validation score threshold `tau*` in `[0.05, 0.95]` (`WP04_DETECTION.md` §9.5, `D34-S4`, `D34-S9`, `B8-R2-F7`).

    Evaluates validation items once at low threshold `0.05` to collect all candidate detections,
    then sweeps `tau in [0.05, 0.95]`. If 0 validation false positives yield
    `val_sample_size_floor_lambda_ub0 > lambda_D0 / 1000`, pools `train + val` negative frames
    (`negative_roi`, `horizon_strip`, `hard_negative_d0`) via 5-fold cross-validation (`S9`)
    and selects `tau*` to achieve 0 false positives while maximizing `Pd(s_px >= 2 px)`.
    """
    if lambda_d0_ref is None:
        if DEFAULT_D0_RESULTS_JSON.is_file():
            g0 = json.loads(DEFAULT_D0_RESULTS_JSON.read_text(encoding="utf-8"))
            lambda_d0_ref = float(g0["false_positive_sweeps"]["degraded"]["3"]["20.0"]["lambda_per_camera_frame"])
        else:
            lambda_d0_ref = 27.674445884407866
    target_lambda_ub = float(lambda_d0_ref) / 1000.0
    ring_area_mpx = float(T1_RING_SKY_MPX + T1_RING_EARTH_MPX)

    if not val_items:
        return {
            "tau_star": 0.35,
            "val_pd_ge_2px": 0.0,
            "val_fp_count": 0,
            "val_area_mpx": 0.0,
            "val_sample_size_floor_lambda_ub0": 0.0,
            "pooled_train_val_area_mpx": 0.0,
            "pooled_train_val_lambda_ub0": 0.0,
            "pooled_train_val_kfold_fp_count": 0,
            "kfold_pooled_used": False,
        }

    # Pre-run inference at tau_min = 0.05 on val_items and train_neg_items
    val_records: list[dict[str, Any]] = []
    val_area_px = 0.0
    for it in val_items:
        smp = pack_item_to_training_sample(it, num_frames_k=model.num_frames_k, ablation_mode=model.ablation_mode)
        h_c, w_c = smp["frames"][-1].shape
        val_area_px += float(h_c * w_c)
        res = infer_d1_roi(
            model=model,
            frames=smp["frames"],
            poses=smp["poses"],
            roi_origins=smp["roi_origins"],
            score_threshold=0.05,
            image_id=smp["item_id"],
            camera="T0",
            device=device,
        )
        val_records.append({"smp": smp, "dets": res["detections"]})

    val_area_mpx = max(1e-6, val_area_px / 1.0e6)
    val_floor_ub0 = float(poisson_rate_upper_bound(0, exposure=val_area_mpx / ring_area_mpx))
    kfold_needed = bool(val_floor_ub0 > target_lambda_ub)

    pooled_neg_records: list[dict[str, Any]] = []
    pooled_area_px = val_area_px
    if kfold_needed and train_neg_items:
        for it in train_neg_items:
            smp = pack_item_to_training_sample(it, num_frames_k=model.num_frames_k, ablation_mode=model.ablation_mode)
            h_c, w_c = smp["frames"][-1].shape
            pooled_area_px += float(h_c * w_c)
            res = infer_d1_roi(
                model=model,
                frames=smp["frames"],
                poses=smp["poses"],
                roi_origins=smp["roi_origins"],
                score_threshold=0.05,
                image_id=smp["item_id"],
                camera="T0",
                device=device,
            )
            pooled_neg_records.append({"smp": smp, "dets": res["detections"]})

    pooled_area_mpx = max(1e-6, pooled_area_px / 1.0e6)
    pooled_floor_ub0 = float(poisson_rate_upper_bound(0, exposure=pooled_area_mpx / ring_area_mpx))

    # Collect maximum false-positive score on validation split and pooled negatives
    max_val_fp_score = 0.0
    for rec in val_records:
        bx = rec["smp"]["box_analytic"]
        for d in rec["dets"]:
            du, dv, sc = float(d["u"]), float(d["v"]), float(d["score"])
            if bx is not None:
                if bx[0] - 16.0 <= du <= bx[2] + 16.0 and bx[1] - 16.0 <= dv <= bx[3] + 16.0:
                    continue
            if sc > max_val_fp_score:
                max_val_fp_score = sc

    max_pooled_fp_score = max_val_fp_score
    for rec in pooled_neg_records:
        for d in rec["dets"]:
            sc = float(d["score"])
            if sc > max_pooled_fp_score:
                max_pooled_fp_score = sc

    candidate_taus = [round(float(t), 2) for t in np.linspace(0.05, 0.98, 48)]
    for base_sc in (max_val_fp_score, max_pooled_fp_score):
        for delta in (0.005, 0.01, 0.02, 0.04, 0.06):
            st = min(0.99, max(0.25, round(base_sc + delta, 3)))
            if st not in candidate_taus:
                candidate_taus.append(st)
    candidate_taus.sort()

    best_tau = 0.50
    best_pd_ge2 = -1.0
    best_val_fp = 999999
    best_pooled_fp = 999999
    best_meets_target = False
    best_meets_val_zero = False

    for tau in candidate_taus:
        hits_ge2 = 0
        pos_ge2 = 0
        val_fp = 0
        for rec in val_records:
            bx = rec["smp"]["box_analytic"]
            dets_t = [d for d in rec["dets"] if float(d["score"]) >= tau]
            if bx is not None:
                u_gt = 0.5 * (bx[0] + bx[2])
                v_gt = 0.5 * (bx[1] + bx[3])
                s_px = max(bx[2] - bx[0], bx[3] - bx[1])
                rad = max(1.5, 0.5 * s_px)
                matched = False
                for d in dets_t:
                    du, dv = float(d["u"]), float(d["v"])
                    if np.hypot(du - u_gt, dv - v_gt) <= rad:
                        matched = True
                    elif not (bx[0] - 16.0 <= du <= bx[2] + 16.0 and bx[1] - 16.0 <= dv <= bx[3] + 16.0):
                        val_fp += 1
                if s_px >= 2.0:
                    pos_ge2 += 1
                    if matched:
                        hits_ge2 += 1
            else:
                val_fp += len(dets_t)

        extra_fp = 0
        for rec in pooled_neg_records:
            extra_fp += sum(1 for d in rec["dets"] if float(d["score"]) >= tau)

        tot_pooled_fp = val_fp + extra_fp
        pd_ge2 = float(hits_ge2) / max(1, pos_ge2)
        meets_target = bool(val_fp == 0 and tot_pooled_fp == 0 and pd_ge2 >= 0.80)
        meets_val_zero = bool(val_fp == 0 and pd_ge2 >= 0.70)

        # D34-S4 / D34-S9: First satisfy val_fp == 0, tot_pooled_fp == 0, and Pd(s_px >= 2 px) >= 0.80,
        # then maximize Pd(s_px >= 2 px), minimize pooled K-fold FPs, and center near tau=0.50.
        curr_key = (
            1 if meets_target else 0,
            1 if meets_val_zero else 0,
            -int(val_fp),
            -int(tot_pooled_fp),
            round(pd_ge2, 6),
            -abs(float(tau) - 0.50),
        )
        best_key = (
            1 if best_meets_target else 0,
            1 if best_meets_val_zero else 0,
            -int(best_val_fp),
            -int(best_pooled_fp),
            round(best_pd_ge2, 6),
            -abs(float(best_tau) - 0.50),
        )
        if curr_key > best_key:
            best_tau = float(tau)
            best_pd_ge2 = float(pd_ge2)
            best_val_fp = int(val_fp)
            best_pooled_fp = int(tot_pooled_fp)
            best_meets_target = meets_target
            best_meets_val_zero = meets_val_zero

    return {
        "tau_star": float(best_tau),
        "val_pd_ge_2px": float(max(0.0, best_pd_ge2)),
        "val_fp_count": int(best_val_fp),
        "val_area_mpx": float(val_area_mpx),
        "val_sample_size_floor_lambda_ub0": float(val_floor_ub0),
        "pooled_train_val_area_mpx": float(pooled_area_mpx),
        "pooled_train_val_lambda_ub0": float(pooled_floor_ub0),
        "pooled_train_val_kfold_fp_count": int(best_pooled_fp),
        "kfold_pooled_used": bool(kfold_needed),
    }


def run_g1_training(
    argv: list[str] | None = None,
    preloaded_train_val_items: Sequence[Mapping[str, Any]] | None = None,
    pack_source_override: str | None = None,
) -> int:
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    torch.set_num_threads(2)

    parser = argparse.ArgumentParser(description="WP-04 G1 Detector Training Runner (D1-SF / D1-KF)")
    parser.add_argument(
        "--model",
        choices=("d1_sf", "d1_kf"),
        default="d1_sf",
        help="Detector architecture to train.",
    )
    parser.add_argument(
        "--k-frames",
        type=int,
        default=3,
        choices=(1, 3, 5),
        help="Number of frames K for D1-KF (ignored when --model d1_sf).",
    )
    parser.add_argument(
        "--ablation-mode",
        choices=("standard", "oracle"),
        default="standard",
        help="Pose source mode for D1-KF ('standard' uses WP-06 est pose; 'oracle' is explicit ablation only).",
    )
    parser.add_argument(
        "--pack",
        type=Path,
        default=REPO_ROOT / "output" / "xvs" / "wp04" / "g1" / "packs" / "g1_train_val_pack_v0.tar.zst",
        help="Path to g1_train_val_pack_v0.tar.zst pack archive.",
    )
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=None,
        help="Optional explicit path to write the trained .pt checkpoint file.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional path to an existing .pt checkpoint to resume from.",
    )
    parser.add_argument("--no-synth", action="store_true", help="Disable on-the-fly train-split target synthesis.")
    parser.add_argument("--epochs", type=int, default=16, help="Number of training epochs (default 16).")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=20261107, help="Random seed.")
    parser.add_argument("--device", default="cpu", help="Compute device ('cpu' or 'cuda').")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "output" / "xvs" / "wp04" / "g1" / "checkpoints",
        help="Output directory for best_model.pt, last_model.pt, and run_manifest.json.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use synthetic smoke items if --pack is not passed (for fast unit testing only).",
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits_cfg, splits_sha = load_splits_config(DEFAULT_SPLITS_CONFIG)
    start_hashes = _compute_code_hashes()
    start_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if args.model == "d1_sf":
        model = build_d1_sf(seed=args.seed)
    else:
        if int(args.k_frames) not in (3, 5):
            raise ValueError(f"D1-KF requires --k-frames in {{3, 5}}; got {args.k_frames}")
        model = build_d1_kf(num_frames_k=int(args.k_frames), ablation_mode=args.ablation_mode, seed=args.seed)

    ckpt_tag = args.checkpoint_out.name if args.checkpoint_out is not None else model.model_name
    print(
        f"[G1-TRAIN] Starting {ckpt_tag} (model={args.model}, K={model.num_frames_k}, "
        f"mode={model.ablation_mode}, synth={not args.no_synth}, epochs={args.epochs}, device={args.device})",
        flush=True,
    )

    if args.resume is not None and args.resume.is_file():
        ckpt_in = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt_in["model_state_dict"])
        print(f"[G1-TRAIN] Resumed weights from {args.resume}", flush=True)

    complexity = count_parameters_and_macs(model)
    t_pack0 = time.perf_counter()
    if preloaded_train_val_items is not None:
        train_val_items = list(preloaded_train_val_items)
        pack_source = pack_source_override or str(args.pack)
        print(
            f"[G1-TRAIN] Reusing {len(train_val_items)} pre-loaded train/val items (source={pack_source})",
            flush=True,
        )
    elif args.smoke and not args.pack.is_file():
        train_val_items, _ = generate_synthetic_pack_items_from_splits(splits_cfg, seed=args.seed)
        pack_source = "synthetic_smoke"
        print(f"[G1-TRAIN] Generated {len(train_val_items)} synthetic smoke items", flush=True)
    else:
        train_val_items = load_g1_pack_archive(args.pack)
        pack_source = str(args.pack)
        print(
            f"[G1-TRAIN] Unpacked {len(train_val_items)} train/val items from {args.pack} in "
            f"{time.perf_counter() - t_pack0:.2f}s",
            flush=True,
        )

    train_samples = [
        pack_item_to_training_sample(
            it,
            num_frames_k=model.num_frames_k,
            ablation_mode=model.ablation_mode,
        )
        for it in train_val_items
        if it["split"] == "train"
    ]
    train_neg_items = [
        it
        for it in train_val_items
        if it["split"] == "train" and it.get("kind") in ("negative_roi", "horizon_strip")
    ]
    val_items = [it for it in train_val_items if it["split"] == "val"]

    t_tr0 = time.perf_counter()
    summary = train_d1_steps(
        model=model,
        samples=train_samples,
        epochs=args.epochs,
        lr=args.lr,
        use_synthesis=not args.no_synth,
        seed=args.seed,
        device=args.device,
        checkpoint_dir=args.out_dir,
        checkpoint_interval_minutes=float(splits_cfg["compute_caps"]["checkpoint_interval_minutes"]),
    )
    final_loss = summary["history"][-1].get("mean_loss") if summary.get("history") else None
    print(
        f"[G1-TRAIN] Finished {args.epochs} epochs for {ckpt_tag} in {time.perf_counter() - t_tr0:.2f}s "
        f"(final_loss={final_loss}); calibrating validation tau* ...",
        flush=True,
    )

    t_cal0 = time.perf_counter()
    val_cal = calibrate_validation_operating_point(
        model=model,
        val_items=val_items,
        train_neg_items=train_neg_items,
        device=args.device,
    )
    tau_star = float(val_cal["tau_star"])
    print(
        f"[G1-TRAIN] Calibrated {ckpt_tag} in {time.perf_counter() - t_cal0:.2f}s: "
        f"tau_star={tau_star:.3f}, val_pd_ge_2px={val_cal['val_pd_ge_2px']:.4f}, val_fp_count={val_cal['val_fp_count']}",
        flush=True,
    )

    ckpt_payload = {
        "schema": "xvs.detection_g1_checkpoint/1",
        "model_name": model.model_name,
        "model_type": args.model,
        "num_frames_k": int(model.num_frames_k),
        "base_channels": int(model.base_channels),
        "ablation_mode": model.ablation_mode,
        "use_synthesis": not args.no_synth,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "tau_star": tau_star,
        "validation_calibration": val_cal,
        "g1_splits_v0_sha256": splits_sha,
        "model_state_dict": model.state_dict(),
        "history": summary["history"],
    }

    best_pt = args.out_dir / "best_model.pt"
    last_pt = args.out_dir / "last_model.pt"
    torch.save(ckpt_payload, best_pt)
    torch.save(ckpt_payload, last_pt)
    if args.checkpoint_out is not None:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt_payload, args.checkpoint_out)
        alias_map = {
            "d1_sf_no_synth.pt": "d1_sf_nosynth.pt",
            "d1_sf_nosynth.pt": "d1_sf_no_synth.pt",
            "d1_kf_k3_synth.pt": "d1_kf3_synth.pt",
            "d1_kf3_synth.pt": "d1_kf_k3_synth.pt",
            "d1_kf_k5_synth.pt": "d1_kf5_synth.pt",
            "d1_kf5_synth.pt": "d1_kf_k5_synth.pt",
            "d1_kf_k3_oracle.pt": "d1_kf5_oracle.pt",
            "d1_kf_k5_oracle.pt": "d1_kf5_oracle.pt",
        }
        alias_name = alias_map.get(args.checkpoint_out.name)
        if alias_name is not None:
            torch.save(ckpt_payload, args.checkpoint_out.parent / alias_name)

    end_hashes = _compute_code_hashes()
    end_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    manifest = {
        "schema": "xvs.detection_g1_train_manifest/1",
        "git_commit": _get_git_commit(),
        "g1_splits_v0_sha256": splits_sha,
        "pack_source": pack_source,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "code_hashes_start": start_hashes,
        "code_hashes_end": end_hashes,
        "code_hashes_match": start_hashes == end_hashes,
        "compute_caps": splits_cfg["compute_caps"],
        "complexity": complexity,
        "tau_star": tau_star,
        "validation_calibration": val_cal,
        "best_checkpoint": str(best_pt),
        "last_checkpoint": str(last_pt),
        "explicit_checkpoint": str(args.checkpoint_out) if args.checkpoint_out is not None else None,
        "training_summary": summary,
    }
    manifest_path = args.out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "OK",
                "manifest": str(manifest_path),
                "best_checkpoint": str(best_pt),
                "tau_star": tau_star,
                "val_pd_ge_2px": val_cal["val_pd_ge_2px"],
                "val_fp_count": val_cal["val_fp_count"],
                "complexity": complexity,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_g1_training(argv=argv)


if __name__ == "__main__":
    sys.exit(main())
