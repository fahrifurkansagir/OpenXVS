"""Gate `V2-M2L-2` verification: `XVS-D2` 6-ch Haar stem, 4 exact tiers, losses, AOT WDS & ONNX export.

Verifies `SP2_M2_MODELS_DESIGN.md` §3.2 (`DR-01`) and emits `output/xvs/v2_m2/v2_xvs_d2_evidence.json`
plus `output/xvs/v2_m2/{xvs_d2_web_fp32.onnx, xvs_d2_web_fp16.onnx}`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from xvs.v2.xvs_d2 import (
    CROP_PRIOR_BIAS,
    FULL_FRAME_PRIOR_BIAS,
    AotG3aBatchMixer,
    FixedHaarWavelet2D,
    XVS_D2_TIER_SPECS,
    build_dual_interval_input,
    build_xvs_d2,
    compute_sky_rotation_homography,
    compute_wgs84_ground_homography,
    compute_xvs_d2_composite_loss,
    count_parameters,
    cross_kd_loss,
    cscwd_distillation_loss,
    decode_peaks_local_softargmax,
    estimate_peak_s2_fp16_mib,
    export_xvs_d2_web_onnx,
    extract_capture_day_utc,
    lla_to_ecef,
    logit_domain_focal_loss,
    normalized_wasserstein_loss,
    parse_aot_bbox_ltwh,
    read_wds_shard,
    split_flights_by_capture_day,
    warp_tensor_with_homography,
    write_wds_shard,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "output" / "xvs" / "v2_m2"
EVIDENCE_PATH = OUT_DIR / "v2_xvs_d2_evidence.json"

_EVIDENCE: dict[str, object] = {
    "schema": "xvs.v2_m2.v2_xvs_d2_evidence/1",
    "verified_utc": "",
    "haar_l2_rel_err": None,
    "haar_frozen": False,
    "tier_params": {},
    "web_peak_s2_fp16_mib": None,
    "bf16_focal_extreme_50_finite": False,
    "aot_m_box_zero_exact": False,
    "aot_capture_day_overlap_count": None,
    "wds_roundtrip_ok": False,
    "onnx_export": {},
    "pass": False,
}


def _persist_evidence() -> None:
    _EVIDENCE["verified_utc"] = datetime.now(timezone.utc).isoformat()
    onnx_rep = _EVIDENCE["onnx_export"]
    tiers = _EVIDENCE["tier_params"]
    _EVIDENCE["pass"] = bool(
        _EVIDENCE["haar_l2_rel_err"] is not None
        and float(_EVIDENCE["haar_l2_rel_err"]) < 1e-5
        and bool(_EVIDENCE["haar_frozen"])
        and isinstance(tiers, dict)
        and set(tiers.keys()) == {"teacher", "edge", "edge_slim", "web"}
        and _EVIDENCE["web_peak_s2_fp16_mib"] is not None
        and float(_EVIDENCE["web_peak_s2_fp16_mib"]) <= 16.0
        and bool(_EVIDENCE["bf16_focal_extreme_50_finite"])
        and bool(_EVIDENCE["aot_m_box_zero_exact"])
        and _EVIDENCE["aot_capture_day_overlap_count"] == 0
        and bool(_EVIDENCE["wds_roundtrip_ok"])
        and isinstance(onnx_rep, dict)
        and bool(onnx_rep.get("all_checks_passed"))
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(_EVIDENCE, indent=2) + "\n", encoding="utf-8")


def test_6ch_input_and_fixed_haar_parseval_norm() -> None:
    torch.manual_seed(20261121)
    i_t = torch.randn(2, 1, 64, 64) * 30.0 + 120.0
    i_tm1 = i_t + torch.randn(2, 1, 64, 64) * 2.0
    i_tm5 = i_t + torch.randn(2, 1, 64, 64) * 4.0
    r1 = torch.ones(2, 1, 64, 64)
    r5 = torch.ones(2, 1, 64, 64)
    r5[:, :, :4, :] = 0.0

    x_t = build_dual_interval_input(i_t, i_tm1, i_tm5, r1, r5)
    assert x_t.shape == (2, 6, 64, 64)
    assert float(x_t[:, :3].abs().max().item()) <= 5.0 + 1e-6
    assert float(x_t[:, 4, :4, :].abs().max().item()) == 0.0
    assert float(x_t[:, 5, :4, :].abs().max().item()) == 0.0

    haar = FixedHaarWavelet2D(in_channels=6)
    assert haar.conv.groups == 6
    assert haar.conv.weight.requires_grad is False
    y = haar(x_t)
    assert y.shape == (2, 24, 32, 32)

    in_norm = float(torch.linalg.vector_norm(x_t).item())
    out_norm = float(torch.linalg.vector_norm(y).item())
    rel_err = abs(out_norm - in_norm) / max(in_norm, 1e-12)
    assert rel_err < 1e-5, f"Haar L2 Parseval error {rel_err:.3e} >= 1e-5"

    _EVIDENCE["haar_l2_rel_err"] = rel_err
    _EVIDENCE["haar_frozen"] = not haar.conv.weight.requires_grad
    _persist_evidence()


def test_all_four_dr01_tiers_forward_heads_and_stopgrad() -> None:
    torch.manual_seed(20261121)
    tier_params: dict[str, int] = {}

    for tier_name, spec in XVS_D2_TIER_SPECS.items():
        model = build_xvs_d2(tier_name)
        n_params = count_parameters(model)
        tier_params[tier_name] = n_params
        assert spec.target_param_min <= n_params <= spec.target_param_max, (
            f"Tier {tier_name} param count {n_params:,} outside [{spec.target_param_min:,}, {spec.target_param_max:,}]"
        )
        assert math.isclose(float(model.heads.hm_out.bias.item()), CROP_PRIOR_BIAS, rel_tol=0.0, abs_tol=1e-4)

        x_t = torch.randn(1, 6, 64, 64)
        h_prev = torch.randn(1, spec.c_gru, 16, 16)
        out = model(x_t, h_prev, return_pyramid=True)

        assert out["hm_logits"].shape == (1, 1, 32, 32)
        assert out["offset"].shape == (1, 2, 32, 32)
        assert out["log_scale"].shape == (1, 2, 32, 32)
        assert out["log_var"].shape == (1, 3, 32, 32)
        assert out["peak_mask"].shape == (1, 1, 32, 32)
        assert out["h_next"].shape == (1, spec.c_gru, 16, 16)

        assert float(out["offset"].abs().max().item()) <= 1.5 + 1e-6
        assert float(out["log_var"].min().item()) >= -6.0 - 1e-6
        assert float(out["log_var"].max().item()) <= 4.0 + 1e-6

    # Check full-frame prior bias option and detached log_var stop-gradient on web tier
    web_ff = build_xvs_d2("web", prior_bias=FULL_FRAME_PRIOR_BIAS)
    assert math.isclose(float(web_ff.heads.hm_out.bias.item()), FULL_FRAME_PRIOR_BIAS, abs_tol=1e-4)

    web = build_xvs_d2("web")
    web.zero_grad()
    x_t = torch.randn(1, 6, 64, 64)
    out = web(x_t)
    out["log_var"].sum().backward()
    backbone_grad_norm = sum(
        float(p.grad.abs().sum().item())
        for p in web.backbone.parameters()
        if p.grad is not None
    )
    assert backbone_grad_norm == 0.0, f"log_var leaked gradient into backbone: {backbone_grad_norm}"

    # Verify decode_peaks_local_softargmax sub-pixel precision (< 0.5 px)
    hm = torch.full((1, 1, 32, 32), -10.0)
    # Place Gaussian peak centered at cell (14.25, 10.75) -> full-res ((14.25+0.5)*2, (10.75+0.5)*2) = (29.5, 22.5)
    yy, xx = torch.meshgrid(torch.arange(32, dtype=torch.float32), torch.arange(32, dtype=torch.float32), indexing="ij")
    hm[0, 0] = 6.0 * torch.exp(-((xx - 14.25) ** 2 + (yy - 10.75) ** 2) / (2.0 * 0.85**2)) - 4.0
    off = torch.zeros(1, 2, 32, 32)
    off[0, 0, 11, 14] = 0.25
    off[0, 1, 11, 14] = -0.25
    lsc = torch.zeros(1, 2, 32, 32)
    lvr = torch.full((1, 3, 32, 32), -2.0)
    pmk = (hm == torch.nn.functional.max_pool2d(hm, 3, 1, 1)).to(torch.float32)
    dets = decode_peaks_local_softargmax(hm, off, lsc, lvr, pmk, score_thr=0.5, max_k=16)
    assert len(dets) == 1
    assert abs(dets[0]["u_px"] - 29.5) < 0.5
    assert abs(dets[0]["v_px"] - 22.5) < 0.5

    web_mib = estimate_peak_s2_fp16_mib("web", height=1024, width=1224)
    assert web_mib <= 16.0, f"Web peak s2 FP16 buffer {web_mib:.2f} MiB > 16 MiB"

    _EVIDENCE["tier_params"] = tier_params
    _EVIDENCE["web_peak_s2_fp16_mib"] = round(web_mib, 4)
    _persist_evidence()


def test_wgs84_dual_interval_homography_and_warp() -> None:
    k_mat = np.array([[2319.0, 0.0, 1224.0], [0.0, 2319.0, 1024.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    r_eye = np.eye(3, dtype=np.float64)
    h_sky_id = compute_sky_rotation_homography(k_mat, r_eye, r_eye)
    assert np.allclose(h_sky_id, np.eye(3), atol=1e-12)

    lat = math.radians(37.6188)
    lon = math.radians(-122.3754)
    p0 = lla_to_ecef(lat, lon, 300.0)
    p1 = lla_to_ecef(lat + 5e-5, lon, 300.0)
    h_gnd = compute_wgs84_ground_homography(k_mat, r_eye, r_eye, p0, p1, lat, lon, alt_agl_prev_m=300.0)
    assert h_gnd.shape == (3, 3) and np.all(np.isfinite(h_gnd))

    src = torch.ones(1, 24, 32, 32)
    warped, mask = warp_tensor_with_homography(src, h_sky_id, feature_scale=0.25)
    assert warped.shape == src.shape and mask.shape == (1, 1, 32, 32)
    assert torch.allclose(warped, src, atol=1e-5)


def test_bf16_safe_focal_loss_nwd_mask_and_kd() -> None:
    # Extreme logits ±50 under bfloat16 autocast MUST yield finite loss & finite gradients
    logits = torch.tensor(
        [[[[50.0, -50.0, 0.0, 48.0], [-48.0, 50.0, -11.09, 12.0]]]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    gt = torch.tensor(
        [[[[1.0, 0.0, 0.5, 0.0], [1.0, 0.25, 0.0, 0.9]]]],
        dtype=torch.bfloat16,
    )
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss_hm = logit_domain_focal_loss(logits, gt)
    assert loss_hm.dtype == torch.float32
    assert torch.isfinite(loss_hm)
    loss_hm.backward()
    assert logits.grad is not None and torch.all(torch.isfinite(logits.grad))

    # m_box = 0 on AOT circle approximations MUST yield exact 0.0 loss and 0.0 grad
    pred_box = torch.tensor([[100.0, 100.0, 12.0, 12.0], [200.0, 200.0, 4.0, 4.0]], requires_grad=True)
    gt_box = torch.tensor([[105.0, 108.0, 6.0, 6.0], [250.0, 250.0, 16.0, 16.0]])
    zero_mask = torch.zeros(2)
    l_nwd_masked = normalized_wasserstein_loss(pred_box, gt_box, box_mask=zero_mask)
    assert float(l_nwd_masked.item()) == 0.0
    l_nwd_masked.backward()
    assert pred_box.grad is not None and float(pred_box.grad.abs().sum().item()) == 0.0

    # Non-overlapping 2x2 boxes with m_box=1 still get smooth non-zero NWD gradient
    pred_box2 = torch.tensor([[10.0, 10.0, 2.0, 2.0]], requires_grad=True)
    gt_box2 = torch.tensor([[18.0, 10.0, 2.0, 2.0]])
    l_nwd_active = normalized_wasserstein_loss(pred_box2, gt_box2, box_mask=torch.ones(1))
    assert float(l_nwd_active.item()) > 0.0
    l_nwd_active.backward()
    assert pred_box2.grad is not None and float(pred_box2.grad.abs().sum().item()) > 0.0

    # CSCWD + CrossKD + Composite loss smoke
    teacher = build_xvs_d2("edge").eval()
    student = build_xvs_d2("web").train()
    x_t = torch.randn(1, 6, 64, 64)
    with torch.no_grad():
        t_out = teacher(x_t, return_pyramid=True)
    s_out = student(x_t, return_pyramid=True)
    adapter = torch.nn.Conv2d(student.spec.c_fpn, teacher.spec.c_fpn, kernel_size=1)
    l_cscwd = cscwd_distillation_loss(s_out["pyramid"], t_out["pyramid"])
    l_cross = cross_kd_loss(s_out["pyramid"]["p2"], teacher.heads, t_out, fp2_adapter=adapter)
    comp = compute_xvs_d2_composite_loss(
        hm_logits=s_out["hm_logits"],
        gt_heatmap=torch.zeros_like(s_out["hm_logits"]),
        pred_boxes_cxcywh=pred_box2.detach().clone().requires_grad_(True),
        gt_boxes_cxcywh=gt_box2,
        pred_mean=torch.zeros(1, 3, requires_grad=True),
        gt_mean=torch.ones(1, 3),
        pred_log_var=torch.zeros(1, 3, requires_grad=True),
        kd_loss=l_cscwd + l_cross,
        lambda_kd=1.0,
    )
    assert torch.isfinite(comp["total"])
    comp["total"].backward()

    _EVIDENCE["bf16_focal_extreme_50_finite"] = True
    _EVIDENCE["aot_m_box_zero_exact"] = True
    _persist_evidence()


def test_aot_ltwh_capture_day_split_and_stdlib_wds_shard(tmp_path: Path) -> None:
    # [left, top, width, height] with left=2200 > 2048 proves x/y order is [left, top, w, h]
    box_ok = parse_aot_bbox_ltwh([2200.0, 500.0, 200.0, 40.0])
    assert box_ok.cx == 2300.0 and box_ok.cy == 520.0
    assert box_ok.is_circle_approx is False and box_ok.m_box == 1.0

    # Fixed 6x6 and 16x16 AOT circle approximations MUST set m_box = 0.0
    assert parse_aot_bbox_ltwh([400.0, 300.0, 6.0, 6.0]).m_box == 0.0
    assert parse_aot_bbox_ltwh([400.0, 300.0, 16.0, 16.0]).m_box == 0.0

    # Swapped (top=2200 > 2048) MUST raise ValueError
    with pytest.raises(ValueError):
        parse_aot_bbox_ltwh([500.0, 2200.0, 40.0, 40.0])

    # Capture-day extraction & 0-overlap atomic split
    base_day_ns = 1_570_000_000_000_000_000  # 2019-10-02 UTC
    flights = []
    for day_idx in range(10):
        ts_ns = base_day_ns + day_idx * 86_400_000_000_000 + 3600_000_000_000
        for fl in range(3):
            fname = f"{ts_ns:019d}{'a' * 32}.png"
            flights.append({"flight_id": f"d{day_idx}_f{fl}", "sample_filename": fname})

    pinned_day = extract_capture_day_utc(flights[0]["sample_filename"])
    splits = split_flights_by_capture_day(flights, pinned_test_days={pinned_day}, seed=2026)
    days_tr = {r["capture_day"] for r in splits["train"]}
    days_va = {r["capture_day"] for r in splits["val"]}
    days_te = {r["capture_day"] for r in splits["test"]}
    overlap = len(days_tr & days_va) + len(days_tr & days_te) + len(days_va & days_te)
    assert overlap == 0
    assert pinned_day in days_te
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == len(flights)

    # Pure-stdlib tarfile WDS shard round-trip + 70% AOT / 30% G3a mixer
    aot_samples = [
        {"key": f"aot_{i:03d}", "x_t": np.full((6, 16, 16), float(i), dtype=np.float32), "meta": {"id": i, "m_box": 0.0}}
        for i in range(10)
    ]
    g3a_samples = [
        {"key": f"g3a_{i:03d}", "x_t": np.full((6, 16, 16), -float(i), dtype=np.float32), "meta": {"id": i, "m_box": 1.0}}
        for i in range(6)
    ]
    shard_p = write_wds_shard(tmp_path / "shard_0000.tar", aot_samples)
    loaded = read_wds_shard(shard_p)
    assert len(loaded) == len(aot_samples)
    assert loaded[3]["key"] == "aot_003"
    assert np.allclose(loaded[3]["x_t"], aot_samples[3]["x_t"])
    assert loaded[3]["meta"]["m_box"] == 0.0

    mixer = AotG3aBatchMixer(loaded, g3a_samples, aot_ratio=0.70, seed=42)
    batch = mixer.sample_batch(20)
    n_aot = sum(1 for item in batch if item["source_domain"] == "aot")
    n_g3a = sum(1 for item in batch if item["source_domain"] == "g3a")
    assert n_aot == 14 and n_g3a == 6

    _EVIDENCE["aot_capture_day_overlap_count"] = overlap
    _EVIDENCE["wds_roundtrip_ok"] = True
    _persist_evidence()


def test_xvs_d2_web_onnx_export_zero_forbidden_ops_and_ort_parity() -> None:
    report = export_xvs_d2_web_onnx(OUT_DIR, height=256, width=256, seed=20261121)
    assert report["forbidden_ops_fp32"] == [], f"Forbidden FP32 ops: {report['forbidden_ops_fp32']}"
    assert report["forbidden_ops_fp16"] == [], f"Forbidden FP16 ops: {report['forbidden_ops_fp16']}"
    assert report["fp16_onnx_bytes"] <= 1_048_576, f"FP16 ONNX > 1 MB: {report['fp16_onnx_bytes']}"
    assert report["fp32_max_abs_diff"] < 1e-5, f"FP32 diff too high: {report['fp32_max_abs_diff']}"
    assert report["fp16_max_abs_diff"] < 2e-3, f"FP16 diff too high: {report['fp16_max_abs_diff']}"
    assert report["fp32_all_finite"] is True and report["fp16_all_finite"] is True
    assert report["all_checks_passed"] is True

    _EVIDENCE["onnx_export"] = report
    _persist_evidence()
