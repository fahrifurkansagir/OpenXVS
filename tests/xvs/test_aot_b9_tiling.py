"""Unit tests for Batch B9 real AWS S3 AOT schema ingestion, 2D homography stack registration,
batched CUDA/CPU sliding-window tiling parity, cross-tile NMS, and zero silent fallback (`B8-R3-F3`, `B8-Colab-F1..F3`)."""

from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pytest
import torch

from scripts.xvs.detection.run_aot_b9_eval import (
    cross_tile_nms,
    infer_d1_full_frame_tiled,
    main as run_aot_main,
    validate_real_aot_directory,
)
from xvs.detection.aot_s3_ingest import (
    ingest_stratified_aot_from_s3,
    parse_valid_encounters_csv,
    select_stratified_aot_encounters,
)
from xvs.detection.d1_model import build_d1_kf, build_d1_sf
from xvs.detection.warp import align_aot_kframe_stack_2d


def test_aot_strict_mode_raises_without_real_aot_dir(tmp_path: Path) -> None:
    """`--mode strict` must raise FileNotFoundError when `--aot-dir` is missing and never fall back to G3a packs."""
    missing_dir = tmp_path / "nonexistent_aot"
    out_json = tmp_path / "aot_out.json"
    with pytest.raises(FileNotFoundError, match="Real AOT directory not found"):
        run_aot_main(["--aot-dir", str(missing_dir), "--mode", "strict", "--out", str(out_json)])
    assert not out_json.exists()


def test_aot_check_readiness_writes_unmeasured_status(tmp_path: Path) -> None:
    """`--mode check-readiness` writes honest UNMEASURED_AWAITING_COLAB_S3 JSON with null metrics when unmounted."""
    missing_dir = tmp_path / "unmounted_aot"
    out_json = tmp_path / "aot_b9_results.json"
    rc = run_aot_main(["--aot-dir", str(missing_dir), "--mode", "check-readiness", "--out", str(out_json)])
    assert rc == 0
    assert out_json.is_file()
    doc = json.loads(out_json.read_text(encoding="utf-8"))
    assert doc["status"] == "UNMEASURED_AWAITING_COLAB_S3"
    assert doc["scratch_mounted"] is False
    assert doc["real_aot_sequences_evaluated"] == 0
    assert doc["overall_pd_aot_700m"] is None
    assert doc["overall_pd_xvs"] is None
    assert doc["fp_per_mpx"] is None
    assert doc["by_model"] is None


def test_cross_tile_nms_deduplicates_border_detections() -> None:
    """Overlapping sliding-window tiles producing duplicate detections within `nms_radius_px` are deduplicated."""
    dets = [
        {"u": 120.0, "v": 120.0, "score": 0.91, "s_det": 4.0, "box_edges": [118.0, 118.0, 122.0, 122.0]},
        {"u": 121.2, "v": 119.5, "score": 0.84, "s_det": 4.0, "box_edges": [119.2, 117.5, 123.2, 121.5]},
        {"u": 210.0, "v": 180.0, "score": 0.77, "s_det": 3.5, "box_edges": [208.2, 178.2, 211.8, 181.8]},
    ]
    kept = cross_tile_nms(dets, nms_radius_px=4.0)
    assert len(kept) == 2
    assert kept[0]["score"] == pytest.approx(0.91)
    assert kept[1]["u"] == pytest.approx(210.0)


def test_real_aot_csv_parsing_and_stratified_s3_ingestion(tmp_path: Path) -> None:
    """(a) Verify parsing of real AWS S3 AOT `valid_encounters_...csv` + `groundtruth.csv` and stratified 6+6 ingestion (`B8-Colab-F1`)."""
    src_bucket_dir = tmp_path / "mock_s3_part1"
    (src_bucket_dir / "part1" / "ImageSets").mkdir(parents=True, exist_ok=True)

    enc_lines = [
        ",encounter_id,flight_id,framemin,framemax,framecount,enc_len_with_gaps,is_above_horizon,min_enc_range,max_enc_range,median_enc_range,mean_enc_range"
    ]
    gt_lines = [
        ",time,flight_id,img_name,frame,id,range_distance_m,is_above_horizon,size_width,size_height,gt_left,gt_top,gt_right,gt_bottom"
    ]

    row_idx = 0
    gt_idx = 0
    for i in range(16):
        fid = f"000{i // 4:01d}ba865c8e410e88609541b8f55ffc"
        img_dir = src_bucket_dir / "part1" / "Images" / fid
        img_dir.mkdir(parents=True, exist_ok=True)
        is_above = 1.0 if i < 8 else -1.0
        fmin = 100 + (i % 4) * 80
        fmax = fmin + 30
        enc_id = f"enc_{i:03d}"
        enc_lines.append(
            f"{row_idx},{enc_id},{fid},{fmin},{fmax},31,31,{is_above},280.0,650.0,450.0,455.0"
        )
        row_idx += 1

        # Add sampled encounter frames at step 10 (fmin, fmin+10, fmin+20, fmin+30)
        for fr in range(fmin, fmax + 1, 10):
            ts_ns = 1559069698000000000 + fr * 100_000_000
            img_name = f"{ts_ns}{fid}.png"
            png_p = img_dir / img_name
            if not png_p.is_file():
                arr = np.full((64, 80), 96, dtype=np.uint8)
                Image.fromarray(arr, mode="L").save(png_p)
            gt_lines.append(
                f"{gt_idx},{ts_ns},{fid},{img_name},{fr},Aircraft1,480.0,{int(is_above)},6.0,4.0,30.0,22.0,36.0,26.0"
            )
            gt_idx += 1

        # Add target-free frames (empty id and gt_left..gt_bottom) at frames 10, 20, 30 for each flight
        if i % 4 == 0:
            for neg_fr in (10, 20, 30):
                ts_ns = 1559069698000000000 + neg_fr * 100_000_000
                img_name = f"{ts_ns}{fid}.png"
                png_p = img_dir / img_name
                if not png_p.is_file():
                    arr = np.full((64, 80), 92, dtype=np.uint8)
                    Image.fromarray(arr, mode="L").save(png_p)
                gt_lines.append(f"{gt_idx},{ts_ns},{fid},{img_name},{neg_fr},,,,,,,,,")
                gt_idx += 1

    valid_csv = src_bucket_dir / "part1" / "ImageSets" / "valid_encounters_maxRange700_maxGap3_minEncLen30.csv"
    gt_csv = src_bucket_dir / "part1" / "ImageSets" / "groundtruth.csv"
    valid_csv.write_text("\n".join(enc_lines) + "\n", encoding="utf-8")
    gt_csv.write_text("\n".join(gt_lines) + "\n", encoding="utf-8")

    parsed = parse_valid_encounters_csv(valid_csv)
    assert len(parsed) == 16
    selected = select_stratified_aot_encounters(parsed, n_above=6, n_below=6, seed=20261109)
    assert len(selected) == 12
    assert sum(1 for r in selected if r["horizon"] == "above") == 6
    assert sum(1 for r in selected if r["horizon"] == "below") == 6

    def _mock_downloader(object_key: str, dest_path: Path) -> None:
        src_p = src_bucket_dir / object_key
        if not src_p.is_file():
            raise FileNotFoundError(f"Mock S3 missing object: {object_key}")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(src_p.read_bytes())

    local_aot = tmp_path / "content_aot"
    summary = ingest_stratified_aot_from_s3(
        local_aot_dir=local_aot,
        n_above=6,
        n_below=6,
        seed=20261109,
        dt_frame_step=10,
        max_frames_per_encounter=4,
        target_free_stacks_per_flight=1,
        max_download_threads=1,
        cleanup_groundtruth_csv=True,
        downloader=_mock_downloader,
    )
    assert summary["above_horizon_encounters"] == 6
    assert summary["below_horizon_encounters"] == 6
    assert summary["target_free_sequences"] >= 1
    assert summary["downloaded_png_count"] >= 15
    assert not (local_aot / "ImageSets" / "groundtruth.csv").exists()

    gt_manifest = validate_real_aot_directory(local_aot, allow_unit_test_fixture=False)
    assert gt_manifest["dataset_id"] == "AOT-v1"
    assert len(gt_manifest["sequences"]) >= 12
    raw_gt_doc = json.loads((local_aot / "aot_ground_truth.json").read_text(encoding="utf-8"))
    assert raw_gt_doc["synthetic"] is False


def test_2d_homography_rotated_starfield_alignment_and_sky_fallback() -> None:
    """(b) Verify `align_aot_kframe_stack_2d` achieves < 0.5 px residual on rotated+translated starfield and falls back on pure sky (`B8-Colab-F2`)."""
    rng = np.random.default_rng(20261110)
    h, w = 256, 320
    curr = np.full((h, w), 28.0, dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    star_pts_curr: list[tuple[float, float]] = []
    for _ in range(55):
        su = float(rng.uniform(36.0, w - 36.0))
        sv = float(rng.uniform(36.0, h - 36.0))
        amp = float(rng.uniform(90.0, 210.0))
        sig = float(rng.uniform(1.1, 1.8))
        curr += amp * np.exp(-0.5 * ((xx - su) ** 2 + (yy - sv) ** 2) / (sig**2)).astype(np.float32)
        star_pts_curr.append((su, sv))

    # Synthesize previous frames by applying known 2D rigid/similarity motions (curr -> prev),
    # then verify align_aot_kframe_stack_2d warps prev back onto curr within < 0.5 px RMS!
    m_curr_to_f0 = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), 1.6, 1.0)
    m_curr_to_f0[0, 2] += 4.25
    m_curr_to_f0[1, 2] -= 2.75
    f0 = cv2.warpAffine(curr, m_curr_to_f0, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    m_curr_to_f1 = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), 0.8, 1.0)
    m_curr_to_f1[0, 2] += 2.10
    m_curr_to_f1[1, 2] -= 1.35
    f1 = cv2.warpAffine(curr, m_curr_to_f1, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    aligned, masks = align_aot_kframe_stack_2d([f0, f1, curr])
    assert len(aligned) == 3
    assert len(masks) == 3

    # Measure sub-pixel centroid alignment residual on interior stars in aligned[0] vs curr
    residuals_px: list[float] = []
    for su, sv in star_pts_curr:
        ui, vi = int(round(su)), int(round(sv))
        if masks[0][vi, ui] < 0.99:
            continue
        x0, x1 = max(0, ui - 4), min(w, ui + 5)
        y0, y1 = max(0, vi - 4), min(h, vi + 5)
        patch_ref = np.maximum(0.0, curr[y0:y1, x0:x1] - 28.0)
        patch_reg = np.maximum(0.0, aligned[0][y0:y1, x0:x1] - 28.0)
        if float(patch_ref.sum()) < 20.0 or float(patch_reg.sum()) < 20.0:
            continue
        py, px = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        u_ref = float((px * patch_ref).sum() / patch_ref.sum())
        v_ref = float((py * patch_ref).sum() / patch_ref.sum())
        u_reg = float((px * patch_reg).sum() / patch_reg.sum())
        v_reg = float((py * patch_reg).sum() / patch_reg.sum())
        residuals_px.append(float(np.hypot(u_reg - u_ref, v_reg - v_ref)))

    assert len(residuals_px) >= 25
    rms_residual = float(np.sqrt(np.mean(np.square(residuals_px))))
    assert rms_residual < 0.5, f"Expected 2D homography residual < 0.5 px, got {rms_residual:.4f} px"

    # Pure featureless blue-sky stack (< 8 keypoints) must fall back cleanly without raising
    sky0 = np.full((128, 128), 140.0, dtype=np.float32)
    sky1 = np.full((128, 128), 140.0, dtype=np.float32)
    sky_aligned, sky_masks = align_aot_kframe_stack_2d([sky0, sky1])
    assert len(sky_aligned) == 2
    assert float(sky_masks[0].mean()) == pytest.approx(1.0)


def test_batched_tile_inference_parity_batch64_vs_batch1() -> None:
    """(c) Verify batched tile inference (`tile_batch_size=64`) matches `tile_batch_size=1` within `1e-5` (`B8-Colab-F3`)."""
    rng = np.random.default_rng(20261111)
    h, w = 220, 260
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    frames: list[np.ndarray] = []
    for k in range(3):
        fr = 35.0 + rng.normal(0.0, 1.2, size=(h, w)).astype(np.float32)
        # Add stationary background stars + moving target
        for su, sv in ((55.0, 60.0), (185.0, 72.0), (140.0, 165.0), (75.0, 175.0), (205.0, 155.0)):
            fr += 80.0 * np.exp(-0.5 * ((xx - su) ** 2 + (yy - sv) ** 2) / (1.4**2)).astype(np.float32)
        tu = 112.0 + 2.5 * k
        tv = 104.0 - 1.5 * k
        fr += 160.0 * np.exp(-0.5 * ((xx - tu) ** 2 + (yy - tv) ** 2) / (1.3**2)).astype(np.float32)
        frames.append(fr)

    sf_model = build_d1_sf(seed=20261107)
    kf_model = build_d1_kf(num_frames_k=3, seed=20261107)

    for model_inst, f_in in ((sf_model, [frames[-1]]), (kf_model, frames)):
        dets_b1 = infer_d1_full_frame_tiled(
            model=model_inst,
            frames=f_in,
            poses=None,
            score_threshold=0.25,
            image_id="parity_check",
            tile_size=96,
            overlap=32,
            device="cpu",
            tile_batch_size=1,
        )
        dets_b64 = infer_d1_full_frame_tiled(
            model=model_inst,
            frames=f_in,
            poses=None,
            score_threshold=0.25,
            image_id="parity_check",
            tile_size=96,
            overlap=32,
            device="cpu",
            tile_batch_size=64,
        )
        assert len(dets_b64) == len(dets_b1)
        assert len(dets_b64) >= 1
        for d64, d1 in zip(dets_b64, dets_b1):
            assert float(d64["score"]) == pytest.approx(float(d1["score"]), abs=1e-5)
            assert float(d64["u"]) == pytest.approx(float(d1["u"]), abs=1e-5)
            assert float(d64["v"]) == pytest.approx(float(d1["v"]), abs=1e-5)
            for e64, e1 in zip(d64["box_edges"], d1["box_edges"]):
                assert float(e64) == pytest.approx(float(e1), abs=1e-5)


def test_aot_tiled_evaluator_on_unit_test_fixture_in_tmp_path(tmp_path: Path) -> None:
    """Verify full-frame sliding-window tiled inference + 2D homography AOT evaluation on an isolated `tmp_path` fixture."""
    aot_fixture_dir = tmp_path / "aot_fixture"
    seq_dir = aot_fixture_dir / "sequences" / "flight_0001"
    seq_dir.mkdir(parents=True, exist_ok=True)

    # Create a 3-frame 160x160 sequence at dt=1.0s without pose dicts so real 2D homography registration runs
    frames_meta = []
    for k in range(3):
        img = np.full((160, 160), 40.0, dtype=np.float32)
        u_t = 76 + 3 * k
        v_t = 82
        yy, xx = np.mgrid[0:160, 0:160]
        img += 320.0 * np.exp(-0.5 * ((xx - u_t) ** 2 + (yy - v_t) ** 2) / (1.3**2)).astype(np.float32)
        rel_npy = f"sequences/flight_0001/frame_{k:04d}.npy"
        np.save(aot_fixture_dir / rel_npy, img)
        frames_meta.append(
            {
                "frame_id": f"flight_0001_{k:04d}",
                "t_s": float(k),
                "image_path": rel_npy,
                "objects": [
                    {
                        "object_id": "intruder_1",
                        "class_id": "airborne",
                        "range_m": 550.0,
                        "box_edges": [float(u_t - 3.0), float(v_t - 3.0), float(u_t + 3.0), float(v_t + 3.0)],
                        "horizon": "above" if k < 2 else "below",
                    }
                ],
            }
        )

    gt_manifest = {
        "dataset_id": "AOT-v1-unit-test-fixture",
        "synthetic": False,
        "unit_test_fixture": True,
        "sequences": [
            {
                "sequence_id": "flight_0001",
                "frames": frames_meta,
            }
        ],
    }
    (aot_fixture_dir / "aot_ground_truth.json").write_text(json.dumps(gt_manifest, indent=2), encoding="utf-8")

    # Without --allow-unit-test-fixture, validate_real_aot_directory must reject the fixture
    with pytest.raises(RuntimeError, match="Rejected unit-test fixture"):
        validate_real_aot_directory(aot_fixture_dir, allow_unit_test_fixture=False)

    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sf_model = build_d1_sf(seed=20261107)
    kf_model = build_d1_kf(num_frames_k=3, seed=20261107)

    f0 = np.load(aot_fixture_dir / "sequences/flight_0001/frame_0002.npy")
    tiled_dets = infer_d1_full_frame_tiled(
        model=sf_model,
        frames=[f0],
        poses=None,
        score_threshold=0.35,
        image_id="flight_0001_0002",
        tile_size=96,
        overlap=32,
        tile_batch_size=64,
    )
    assert len(tiled_dets) >= 1

    torch.save(
        {
            "schema": "xvs.detection_g1_checkpoint/1",
            "model_name": sf_model.model_name,
            "model_type": "d1_sf",
            "num_frames_k": 1,
            "ablation_mode": "standard",
            "seed": 20261107,
            "tau_star": 0.45,
            "model_state_dict": sf_model.state_dict(),
        },
        ckpt_dir / "d1_sf_synth.pt",
    )
    torch.save(
        {
            "schema": "xvs.detection_g1_checkpoint/1",
            "model_name": kf_model.model_name,
            "model_type": "d1_kf",
            "num_frames_k": 3,
            "ablation_mode": "standard",
            "seed": 20261107,
            "tau_star": 0.45,
            "model_state_dict": kf_model.state_dict(),
        },
        ckpt_dir / "d1_kf_k3_synth.pt",
    )

    out_json = tmp_path / "fixture_aot_eval.json"
    rc = run_aot_main(
        [
            "--aot-dir",
            str(aot_fixture_dir),
            "--mode",
            "strict",
            "--allow-unit-test-fixture",
            "--checkpoints-dir",
            str(ckpt_dir),
            "--tile-size",
            "96",
            "--overlap",
            "32",
            "--tile-batch-size",
            "64",
            "--out",
            str(out_json),
        ]
    )
    assert rc == 0
    res = json.loads(out_json.read_text(encoding="utf-8"))
    assert res["status"] == "MEASURED_REAL_AOT"
    assert res["real_aot_sequences_evaluated"] == 1
    assert res["by_model"]["d1_kf"]["overall_pd_aot_700m"] > 0.5
    assert "pd_above_horizon_aot_700m" in res["by_model"]["d1_kf"]
    assert "pd_below_horizon_aot_700m" in res["by_model"]["d1_kf"]
