"""V3_G1, V4_G1 & V5_G1 — CPU fp32 evaluation determinism, no-truth import/runtime guard, and report provenance (`WP04_DETECTION.md` §9.7).

Verifies:
1. V3_G1 (Decision D34-S6): Given fixed checkpoint weights and seed, running the G1 evaluator
   twice on CPU in `fp32` produces byte-identical JSON outputs.
2. V4_G1 (Decision D34-S3, S8): AST import-graph and runtime check proves `src/xvs/detection/d1_model.py`,
   `src/xvs/detection/warp.py`, `src/xvs/detection/splits.py`, and `src/xvs/detection/pack.py`
   never import from `xvs.traffic`, `xvs.metrics`, or ground-truth mask I/O, and that non-oracle
   `D1-KF` inference rejects ground-truth pose inputs.
3. On-the-fly target synthesis (`synthesize_target_on_background`) rejects `'val'` and `'test'` splits.
4. V5_G1: `scripts/xvs/detection/report_tables_wp04_g1.py` generates a report into `tmp_path` that
   passes `verify_report_provenance` with `0` unprovenanced numbers.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.xvs.detection.build_g1_packs import main as build_packs_main
from scripts.xvs.detection.report_tables_wp04_g1 import generate_wp04_g1_report
from scripts.xvs.detection.run_aot_b9_eval import main as run_aot_main
from scripts.xvs.detection.run_g1_eval import (
    evaluate_g1_items_cpu_fp32,
    load_d0_reference_metrics,
    main as run_eval_main,
)
from scripts.xvs.detection.run_g1_train import (
    main as run_train_main,
    pack_item_to_training_sample,
)
from xvs.detection.d1_model import (
    build_d1_kf,
    build_d1_sf,
    compute_d1_losses,
    count_parameters_and_macs,
    infer_d1_roi,
    penalty_reduced_focal_loss,
)
from xvs.detection.pack import generate_synthetic_pack_items_from_splits
from xvs.detection.splits import DEFAULT_SPLITS_CONFIG, load_splits_config
from xvs.detection.train_g1 import synthesize_target_on_background
from xvs.report_provenance import find_denylisted_config_keys, verify_report_provenance


def test_v3_g1_cpu_fp32_evaluation_determinism(tmp_path: Path) -> None:
    """Running the G1 evaluator twice on CPU in fp32 with fixed weights and seed yields byte-identical JSON."""
    splits_cfg, _ = load_splits_config(DEFAULT_SPLITS_CONFIG)
    _, test_items = generate_synthetic_pack_items_from_splits(splits_cfg, seed=20261107)

    for model_kind in ("sf", "kf_k3", "kf_k5"):
        if model_kind == "sf":
            m1 = build_d1_sf(seed=20261107)
            m2 = build_d1_sf(seed=20261107)
        elif model_kind == "kf_k3":
            m1 = build_d1_kf(num_frames_k=3, ablation_mode="standard", seed=20261107)
            m2 = build_d1_kf(num_frames_k=3, ablation_mode="standard", seed=20261107)
        else:
            m1 = build_d1_kf(num_frames_k=5, ablation_mode="standard", seed=20261107)
            m2 = build_d1_kf(num_frames_k=5, ablation_mode="standard", seed=20261107)
            # Extend synthetic items to 5 frames at 1.0 s step for K=5 test
            test_items_k5 = []
            for it in test_items:
                it5 = dict(it)
                if it5["num_frames_k"] == 3:
                    f0 = it5["frames"][0]
                    p0 = it5["poses"][0]
                    o0 = it5["roi_origins"][0]
                    t0 = it5["timestamps_s"][0]
                    it5["num_frames_k"] = 5
                    it5["frames"] = [f0, f0] + list(it5["frames"])
                    it5["poses"] = [p0, p0] + list(it5["poses"])
                    it5["roi_origins"] = [o0, o0] + list(it5["roi_origins"])
                    it5["timestamps_s"] = [t0 - 2.0, t0 - 1.0] + list(it5["timestamps_s"])
                test_items_k5.append(it5)
            r1 = evaluate_g1_items_cpu_fp32(m1, test_items_k5, score_threshold=0.35)
            r2 = evaluate_g1_items_cpu_fp32(m2, test_items_k5, score_threshold=0.35)
            j1 = json.dumps(r1, indent=2, sort_keys=True).encode("utf-8")
            j2 = json.dumps(r2, indent=2, sort_keys=True).encode("utf-8")
            assert j1 == j2
            continue

        r1 = evaluate_g1_items_cpu_fp32(m1, test_items, score_threshold=0.35)
        r2 = evaluate_g1_items_cpu_fp32(m2, test_items, score_threshold=0.35)
        out1 = tmp_path / f"eval_{model_kind}_run1.json"
        out2 = tmp_path / f"eval_{model_kind}_run2.json"
        out1.write_text(json.dumps(r1, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        out2.write_text(json.dumps(r2, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        assert out1.read_bytes() == out2.read_bytes()


def test_v4_g1_no_truth_in_detector_inference_and_warp_modules() -> None:
    """AST import-graph and runtime check on G1 detector and warp modules (`V4_G1`, `S3`, `S8`)."""
    target_modules = [
        REPO_ROOT / "src" / "xvs" / "detection" / "d1_model.py",
        REPO_ROOT / "src" / "xvs" / "detection" / "warp.py",
        REPO_ROOT / "src" / "xvs" / "detection" / "splits.py",
        REPO_ROOT / "src" / "xvs" / "detection" / "pack.py",
    ]
    forbidden_prefixes = ("xvs.traffic", "xvs.metrics")

    for mod_path in target_modules:
        tree = ast.parse(mod_path.read_text(encoding="utf-8"), filename=str(mod_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(forbidden_prefixes), (
                        f"Forbidden import {alias.name!r} in {mod_path.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                mod_name = node.module or ""
                assert not mod_name.startswith(forbidden_prefixes), (
                    f"Forbidden from-import {mod_name!r} in {mod_path.name}"
                )

    # Runtime check: non-oracle D1-KF inference rejects ground-truth pose inputs (`S3`, `S8`)
    model_kf = build_d1_kf(num_frames_k=3, ablation_mode="standard")
    dummy_frames = [np.zeros((32, 32), dtype=np.float32) for _ in range(3)]
    gt_poses = [
        {"q_ned_to_cam": [1.0, 0.0, 0.0, 0.0], "pos_wgs84": [37.05, -120.85, 3000.0]}
        for _ in range(3)
    ]
    with pytest.raises(ValueError, match="Forbidden ground-truth pose keys"):
        infer_d1_roi(model=model_kf, frames=dummy_frames, poses=gt_poses)

    # Verify parameter count and MACs counter
    stats_sf = count_parameters_and_macs(build_d1_sf())
    stats_kf = count_parameters_and_macs(model_kf)
    assert stats_sf["trainable_params"] > 0
    assert stats_kf["trainable_params"] > stats_sf["trainable_params"]
    assert stats_sf["macs_per_1024_roi"] > 0
    assert stats_sf["flops_per_1024_roi"] == 2 * stats_sf["macs_per_1024_roi"]


def test_train_split_synthesis_rejects_val_and_test() -> None:
    """Verify on-the-fly target synthesis strictly rejects 'val' and 'test' splits (`WP04_DETECTION.md` §9.3, `B8-R2-F6`)."""
    bg = np.full((64, 64), 50.0, dtype=np.float32)
    rng = np.random.default_rng(42)
    syn = synthesize_target_on_background(
        bg,
        rng=rng,
        split="train",
        sensor_state="degraded",
        center_uv=(32.0, 32.0),
    )
    assert syn["image"].shape == (64, 64)
    # B8-R2-F6: far corner background pixel (0, 0) must NOT receive a second pass of whole-image noise when sensor_state='degraded'
    assert float(syn["image"][0, 0]) == float(bg[0, 0])

    with pytest.raises(ValueError, match="ONLY on the 'train' split"):
        synthesize_target_on_background(bg, rng=rng, split="val")
    with pytest.raises(ValueError, match="ONLY on the 'train' split"):
        synthesize_target_on_background(bg, rng=rng, split="test")


def test_v5_g1_report_provenance_generation(tmp_path: Path) -> None:
    """Verify `report_tables_wp04_g1.py` generates a report with 0 unprovenanced numbers (`V5_G1`)."""
    out_rep = tmp_path / "WP04_G1_report.md"
    generate_wp04_g1_report(out_path=out_rep)
    assert out_rep.is_file()
    assert (tmp_path / "WP04_G1_report.md.provenance.json").is_file()
    assert verify_report_provenance(out_rep) == []


def test_b8_f1_f2_f3_train_checkpoint_eval_and_aot_round_trip(tmp_path: Path) -> None:
    """Verify B8-F1, B8-F2, and B8-F3 end-to-end pack -> train .pt -> eval fp32 -> AOT B9 -> report provenance."""
    packs_dir = tmp_path / "packs"
    ckpt_dir = tmp_path / "checkpoints"
    eval_out = tmp_path / "eval_results.json"
    closure_out = tmp_path / "g1_measured_d1.json"
    aot_out = tmp_path / "aot_b9_results.json"
    rep_out = tmp_path / "WP04_G1_report_full.md"

    # 1. Build smoke packs in tmp_path
    assert build_packs_main(["--smoke", "--out-dir", str(packs_dir)]) == 0
    tv_pack = packs_dir / "g1_train_val_pack_v0.tar.zst"
    te_pack = packs_dir / "g1_test_eval_pack_v0.tar.zst"
    assert tv_pack.is_file() and te_pack.is_file()

    # 2. Verify pack_item_to_training_sample computes true ROI-local coordinates from box_analytic (not hardcoded 30, 30)
    splits_cfg, _ = load_splits_config(DEFAULT_SPLITS_CONFIG)
    tv_items, _ = generate_synthetic_pack_items_from_splits(splits_cfg)
    pos_item = next(it for it in tv_items if it["box_analytic"] is not None)
    smp = pack_item_to_training_sample(pos_item, num_frames_k=3)
    bx = pos_item["box_analytic"]
    u0, v0 = pos_item["roi_origins"][-1]
    exp_u = 0.5 * (bx[0] + bx[2]) - u0
    exp_v = 0.5 * (bx[1] + bx[3]) - v0
    assert smp["centers_uv"] == [(exp_u, exp_v)]
    assert smp["boxes_wh"] == [(bx[2] - bx[0], bx[3] - bx[1])]

    # 3. Train D1-KF (K=3) from the pack archive and save best_model.pt, last_model.pt, and explicit checkpoint
    explicit_pt = ckpt_dir / "d1_kf_k3_synth.pt"
    assert (
        run_train_main(
            [
                "--model",
                "d1_kf",
                "--k-frames",
                "3",
                "--pack",
                str(tv_pack),
                "--out-dir",
                str(ckpt_dir),
                "--checkpoint-out",
                str(explicit_pt),
                "--epochs",
                "1",
            ]
        )
        == 0
    )
    assert (ckpt_dir / "best_model.pt").is_file()
    assert (ckpt_dir / "last_model.pt").is_file()
    assert explicit_pt.is_file()

    # 4. Evaluate trained .pt checkpoint on unpacked test pack in fp32 on CPU
    assert (
        run_eval_main(
            [
                "--model",
                "d1_kf",
                "--k-frames",
                "3",
                "--test-pack",
                str(te_pack),
                "--checkpoint",
                str(explicit_pt),
                "--checkpoints-dir",
                str(ckpt_dir),
                "--export-tracking-closure",
                str(closure_out),
                "--out",
                str(eval_out),
            ]
        )
        == 0
    )
    eval_data = json.loads(eval_out.read_text(encoding="utf-8"))
    d0_refs = load_d0_reference_metrics()
    assert eval_data["checkpoint_loaded"] == str(explicit_pt)
    assert abs(eval_data["persistence_6dof"]["d0_persist_3_reference"] - d0_refs["d0_persist3_ref"]) < 1e-12
    assert "d1_k_persisted_in_roi" in eval_data["persistence_6dof"]
    assert "d1_n_evaluable_in_roi" in eval_data["persistence_6dof"]
    assert "closure_curve_fit" in eval_data
    assert "localization_metrics" in eval_data
    assert "val_sample_size_floor_lambda_ub0" in eval_data["false_alarm_metrics"]
    assert "delta_pd_ge_2px_bootstrap_ci95" in eval_data
    assert len(eval_data["delta_pd_ge_2px_bootstrap_ci95"]) == 2
    assert isinstance(eval_data["meets_delta_pd_ge_2px_lb_ge_neg_0_02"], bool)
    assert isinstance(eval_data["meets_zero_persistent_clutter"], bool)
    perf_out = eval_out.parent / "eval_perf.json"
    assert perf_out.is_file()
    perf_data = json.loads(perf_out.read_text(encoding="utf-8"))
    assert float(perf_data["inference_ms_per_1024_roi"]) > 0.0
    assert find_denylisted_config_keys(json.loads(closure_out.read_text(encoding="utf-8"))) == []

    # 5. Run Batch B9 AOT readiness check (zero silent fallback to G3a packs, B8-R3-F3)
    assert (
        run_aot_main(
            [
                "--aot-dir",
                str(tmp_path / "unmounted_aot"),
                "--mode",
                "check-readiness",
                "--out",
                str(aot_out),
            ]
        )
        == 0
    )
    aot_data = json.loads(aot_out.read_text(encoding="utf-8"))
    assert aot_data["status"] == "UNMEASURED_AWAITING_COLAB_S3"
    assert aot_data["scratch_mounted"] is False
    assert aot_data["real_aot_sequences_evaluated"] == 0
    assert aot_data["overall_pd_aot_700m"] is None
    assert aot_data["overall_pd_xvs"] is None
    assert aot_data["fp_per_mpx"] is None
    assert aot_data["by_model"] is None

    # 6. Render full WP-04 G1 report with eval_results.json and aot_b9_results.json and verify provenance
    generate_wp04_g1_report(out_path=rep_out, g1_eval_path=eval_out, aot_b9_path=aot_out)
    assert verify_report_provenance(rep_out) == []


def test_zero_conv_weights_give_uniform_prior_heatmap() -> None:
    """`B8-R3-F1`: Zeroing the final head conv weights of `D1HeatmapDetector` must yield an identical prior heatmap for a bright 3x3 target vs an all-zero image (confirming zero classical morphology bypass)."""
    for model in (build_d1_sf(seed=20261107), build_d1_kf(num_frames_k=3, seed=20261107)):
        model.eval()
        with torch.no_grad():
            model.head_heatmap[-1].weight.zero_()

        c_in = model.in_channels
        x_blank = torch.zeros(1, c_in, 64, 64, dtype=torch.float32)
        x_bright = torch.zeros(1, c_in, 64, 64, dtype=torch.float32)
        x_bright[:, 0, 30:33, 30:33] = 5000.0

        with torch.no_grad():
            out_blank = model(x_blank)
            out_bright = model(x_bright)

        assert torch.allclose(out_bright["heatmap"], out_blank["heatmap"], atol=1e-6)


def test_sf_vs_kf_distinct_learned_outputs() -> None:
    """`B8-R3-F1`: Verify `D1-SF` and `D1-KF` produce distinct learned outputs and moving-target temporal cues improve `D1-KF` over `D1-SF`."""
    sf = build_d1_sf(seed=20261107)
    kf = build_d1_kf(num_frames_k=3, seed=20261107)
    sf.eval()
    kf.eval()

    rng = np.random.default_rng(123)
    bg = rng.normal(50.0, 2.5, size=(64, 64)).astype(np.float32)
    # Add a stationary clutter spot at (18, 18) and a moving target that arrives at (44, 44) at frame t
    yy, xx = np.mgrid[0:64, 0:64]
    clutter = 7.5 * np.exp(-0.5 * ((xx - 18.0) ** 2 + (yy - 18.0) ** 2) / (1.1**2)).astype(np.float32)

    frames = []
    poses = []
    origins = []
    for k_i, (u_t, v_t) in enumerate(((36.0, 44.0), (40.0, 44.0), (44.0, 44.0))):
        tgt = 8.5 * np.exp(-0.5 * ((xx - u_t) ** 2 + (yy - v_t) ** 2) / (1.1**2)).astype(np.float32)
        frames.append(bg + clutter + tgt)
        poses.append(
            {
                "pose_source": "wp06_nav_attitude",
                "q_ned_to_cam_est": [1.0, 0.0, 0.0, 0.0],
                "pos_wgs84": [37.6188, -122.3754, 1500.0],
            }
        )
        origins.append((0.0, 0.0))

    res_sf = infer_d1_roi(sf, [frames[-1]], [poses[-1]], [origins[-1]], score_threshold=0.05)
    res_kf = infer_d1_roi(kf, frames, poses, origins, score_threshold=0.05)

    hm_sf = res_sf["heatmap"]
    hm_kf = res_kf["heatmap"]
    assert not np.allclose(hm_sf, hm_kf, atol=1e-4)

    # Moving target at (44, 44) vs stationary clutter at (18, 18): D1-KF target-to-clutter margin exceeds D1-SF
    margin_sf = float(hm_sf[44, 44] - hm_sf[18, 18])
    margin_kf = float(hm_kf[44, 44] - hm_kf[18, 18])
    assert margin_kf > margin_sf


def test_b8_d16_m1_bf16_focal_and_d1_losses_finite_loss_and_grads() -> None:
    """`B8-D16-m1`: Passing `bfloat16` `pred_hm` with exact `0.0`, `0.9999`, and `1.0` must produce finite loss and finite gradients."""
    pred_hm = torch.tensor(
        [[[[0.0, 0.9999, 1.0, 0.5], [1.0, 0.0, 0.9999, 0.25]]]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    gt_hm = torch.tensor(
        [[[[0.0, 1.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0]]]],
        dtype=torch.bfloat16,
    )
    loss_focal = penalty_reduced_focal_loss(pred_hm, gt_hm)
    assert loss_focal.dtype == torch.float32
    assert torch.isfinite(loss_focal)
    loss_focal.backward()
    assert pred_hm.grad is not None
    assert torch.isfinite(pred_hm.grad).all()

    # Also verify full compute_d1_losses under bf16 autocast with exact boundary values
    pred_hm2 = torch.tensor(
        [[[[0.0, 0.9999, 1.0, 0.5], [1.0, 0.0, 0.9999, 0.25]]]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    pred_off = torch.zeros((1, 2, 2, 4), dtype=torch.bfloat16, requires_grad=True)
    pred_scale = torch.ones((1, 2, 2, 4), dtype=torch.bfloat16, requires_grad=True)
    targets = {
        "gt_heatmap": gt_hm,
        "gt_offset": torch.zeros((1, 2, 2, 4), dtype=torch.bfloat16),
        "gt_log_scale": torch.ones((1, 2, 2, 4), dtype=torch.bfloat16),
        "center_mask": (gt_hm >= 0.999).to(dtype=torch.bfloat16),
    }
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        losses = compute_d1_losses(
            {"heatmap": pred_hm2, "offset": pred_off, "log_scale": pred_scale},
            targets,
        )
    for key, val in losses.items():
        assert val.dtype == torch.float32, f"{key} should be float32, got {val.dtype}"
        assert torch.isfinite(val), f"{key} is not finite: {val}"
    losses["total_loss"].backward()
    assert pred_hm2.grad is not None and torch.isfinite(pred_hm2.grad).all()
    assert pred_off.grad is not None and torch.isfinite(pred_off.grad).all()
    assert pred_scale.grad is not None and torch.isfinite(pred_scale.grad).all()


def test_b8_d16_m2_closed_empirical_banner_replaces_stale_staged_banner() -> None:
    """`B8-D16-m2` & `B8-D16-R1`: Whenever full 180-encounter GPU training and real AOT are complete, gate_status_banner must NEVER emit 'Staged for Colab A100' (both s50<=1.5 PASS and s50>1.5 FAIL paths)."""
    splits_cfg, _ = load_splits_config(DEFAULT_SPLITS_CONFIG)
    _, test_items = generate_synthetic_pack_items_from_splits(splits_cfg, seed=20261107)
    model = build_d1_kf(num_frames_k=3, ablation_mode="standard", seed=20261107)

    # Path 1: normal threshold — verify never emits 'Staged for Colab A100'
    res_norm = evaluate_g1_items_cpu_fp32(
        model=model,
        items=test_items,
        score_threshold=0.35,
        real_aot_evaluated=True,
        real_aot_sequences_evaluated=24,
        full_180_encounter_gpu_trained=True,
    )
    assert "Staged for Colab A100" not in res_norm["gate_status_banner"]
    assert "24 sequences" in res_norm["gate_status_banner"]
    if res_norm["meets_s50_le_1_5px"]:
        assert res_norm["gate_status"] in ("PASS", "CLOSED_EMPIRICAL")
        if res_norm["gate_status"] == "CLOSED_EMPIRICAL":
            assert "s50 <= 1.50 px PASS" in res_norm["gate_status_banner"]

    # Path 2: force s50 > 1.50 px FAIL via unreachable threshold (tau=1.01 -> 0 hits -> s50 > 1.5)
    res_fail_s50 = evaluate_g1_items_cpu_fp32(
        model=model,
        items=test_items,
        score_threshold=1.01,
        real_aot_evaluated=True,
        real_aot_sequences_evaluated=24,
        full_180_encounter_gpu_trained=True,
    )
    assert res_fail_s50["meets_s50_le_1_5px"] is False
    assert res_fail_s50["gate_status"] == "PARTIAL"
    assert "Staged for Colab A100" not in res_fail_s50["gate_status_banner"]
    assert "24 sequences" in res_fail_s50["gate_status_banner"]
    assert "> 1.50 px FAIL" in res_fail_s50["gate_status_banner"]

    # Path 3: pre-Colab local run (not full GPU trained) still emits Staged banner
    res_local = evaluate_g1_items_cpu_fp32(
        model=model,
        items=test_items,
        score_threshold=0.35,
        real_aot_evaluated=False,
        real_aot_sequences_evaluated=0,
        full_180_encounter_gpu_trained=False,
    )
    assert res_local["gate_status"] == "PARTIAL"
    assert "Staged for Colab A100" in res_local["gate_status_banner"]
