"""Tests for DR-13 `XVS-AST v2` Adverse-ODD Falsifier, Croissant 1.1 RAI & Stage-4 HNM mix."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_XVS = REPO_ROOT / "scripts" / "xvs"
if str(SCRIPTS_XVS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_XVS))

import render_aot_d2_demo_video as video_mod
from xvs.v2.xvs_ast import (
    CROISSANT_1_1_CONFORMS_TO,
    CROISSANT_RAI_CONFORMS_TO,
    DatasetQuarantineError,
    MetarOddScenario,
    XvsAstFalsifier,
    assert_dataset_not_quarantined,
    build_croissant_1_1_rai_metadata,
    evaluate_odd_scenario,
)
from xvs.v2.xvs_d2.aot_wds import AotG3aBatchMixer
from xvs.v2.xvs_d2.train_g4 import SyntheticG3aClipDataset, _sample_stage4_batch
from xvs.v2.xvs_optical_odd import OpticalOddAction, OpticalOddMode
from xvs.v2.xvs_vlm_gate import GateStatus


def test_direct_odd_boundaries_fog_thermal_primary_glare_and_icing() -> None:
    """Verify exact ODD transitions: VFR NOMINAL, LIFR fog -> THERMAL_PRIMARY, glare, icing."""
    vfr = MetarOddScenario(
        visibility_mor_m=12000.0,
        sun_angle_deg=70.0,
        rain_blur_frac=0.0,
        window_ice_delta_c=8.0,
        lwir_ffc_age_s=45.0,
        fog_contrast=0.85,
    )
    res_vfr = evaluate_odd_scenario(vfr)
    assert res_vfr.odd_action == OpticalOddAction.NOMINAL
    assert res_vfr.robustness > 0.0
    assert res_vfr.falsified is False
    assert res_vfr.monotone_safety_ok is True

    # Dense LIFR fog (< 1609 m) with healthy LWIR -> THERMAL_PRIMARY
    lifr_fog = MetarOddScenario(
        visibility_mor_m=650.0,
        sun_angle_deg=75.0,
        rain_blur_frac=0.0,
        window_ice_delta_c=7.5,
        lwir_ffc_age_s=60.0,
        fog_contrast=0.22,
    )
    res_fog = evaluate_odd_scenario(lifr_fog)
    assert OpticalOddMode.FOG_VISIBILITY in res_fog.triggered_modes
    assert res_fog.odd_action == OpticalOddAction.THERMAL_PRIMARY
    assert res_fog.falsified is True
    assert res_fog.monotone_safety_ok is True

    # Direct solar glare inside SPA cone -> EVS_LWIR_ELEVATED + VLM masked
    glare = MetarOddScenario(
        visibility_mor_m=10000.0,
        sun_angle_deg=8.0,
        rain_blur_frac=0.0,
        window_ice_delta_c=9.0,
        lwir_ffc_age_s=40.0,
        fog_contrast=0.85,
    )
    res_glare = evaluate_odd_scenario(glare)
    assert OpticalOddMode.SOLAR_GLARE in res_glare.triggered_modes
    assert res_glare.odd_action == OpticalOddAction.EVS_LWIR_ELEVATED
    assert res_glare.vlm_gate_status == GateStatus.OPTICAL_ODD_MASKED
    assert res_glare.monotone_safety_ok is True

    # Window icing (T_win - T_dew <= 2 C) -> DEGRADED_FALLBACK + VLM masked + AF447 ban intact
    icing = MetarOddScenario(
        visibility_mor_m=8000.0,
        sun_angle_deg=70.0,
        rain_blur_frac=0.0,
        window_ice_delta_c=0.5,
        lwir_ffc_age_s=50.0,
        fog_contrast=0.80,
    )
    res_ice = evaluate_odd_scenario(icing)
    assert OpticalOddMode.WINDOW_ICING in res_ice.triggered_modes
    assert res_ice.odd_action == OpticalOddAction.DEGRADED_FALLBACK
    assert res_ice.vlm_gate_status == GateStatus.OPTICAL_ODD_MASKED
    assert res_ice.monotone_safety_ok is True


def test_xvs_ast_v2_partitioned_search_discovers_fog_and_glare_icing_with_zero_violations() -> None:
    """Multi-round LHS + elite refinement finds fog/glare/icing breaches with 0 safety violations."""
    falsifier = XvsAstFalsifier(seed=20261121)
    report = falsifier.run(
        n_lhs=24,
        refinement_rounds=2,
        elites_per_round=4,
        samples_per_elite=3,
    )

    assert report.rounds == 3
    assert report.total_rollouts == 24 + 2 * 4 * 3
    assert report.monotone_safety_violations == 0
    assert report.min_robustness < 0.0
    assert len(report.falsified_scenarios) >= 6
    assert report.vlm_masked_count >= 1

    # Must discover fog boundary (including THERMAL_PRIMARY) AND glare or icing boundaries
    assert report.triggered_mode_counts["FOG_VISIBILITY"] >= 1
    assert report.action_counts["THERMAL_PRIMARY"] >= 1
    assert (
        report.triggered_mode_counts["SOLAR_GLARE"] >= 1
        or report.triggered_mode_counts["WINDOW_ICING"] >= 1
    )

    # Confirm bit-exact seed determinism
    report2 = XvsAstFalsifier(seed=20261121).run(
        n_lhs=24,
        refinement_rounds=2,
        elites_per_round=4,
        samples_per_elite=3,
    )
    assert report2.min_robustness == pytest.approx(report.min_robustness, abs=1e-12)
    assert report2.worst_case_scenario.scenario == report.worst_case_scenario.scenario


def test_croissant_1_1_rai_schema_30_fields_and_quarantine_guard() -> None:
    """Verify >= 30 Croissant 1.1 + RAI fields and fail-closed exclusion of quarantined datasets."""
    report = XvsAstFalsifier(seed=7).run(n_lhs=8, refinement_rounds=1, elites_per_round=2, samples_per_elite=2)
    meta = build_croissant_1_1_rai_metadata(
        dataset_name="OpenXVS-Bench-v2-KSFO",
        version="2.5.0",
        description="Civil aviation adverse-ODD optical & DAA falsification benchmark.",
        source_datasets=[
            {"name": "Amazon AOT", "url": "https://registry.opendata.aws/airborne-object-tracking", "license": "CC-BY-4.0"},
            {"name": "USGS 3DEP & NAIP", "url": "https://basemap.nationalmap.gov", "license": "U.S. Public Domain"},
        ],
        ast_report=report,
    )

    assert len(meta.keys()) >= 30
    assert CROISSANT_1_1_CONFORMS_TO in meta["conformsTo"]
    assert CROISSANT_RAI_CONFORMS_TO in meta["conformsTo"]
    assert meta["license"] == "Apache-2.0"
    assert meta["rai:astFalsificationSummary"]["monotone_safety_violations"] == 0
    assert "LARDON" in meta["rai:quarantinedSourcesExcluded"]

    # Quarantined datasets MUST raise DatasetQuarantineError
    for bad_name, bad_url in (
        ("LARD 2.0", "https://example.org/lard_2_0"),
        ("lard_ges", "https://example.org/ges"),
        ("ges", ""),
        ("BingMaps Aerial", "https://virtualearth.net/bingmaps"),
        ("ArcGIS World Imagery", "https://server.arcgisonline.com/arcgis"),
        ("LARDON", "https://example.org/lardon"),
        ("PASS", "https://example.org/pass"),
        ("M2E-UAV", "https://example.org/m2e-uav"),
    ):
        with pytest.raises(DatasetQuarantineError):
            assert_dataset_not_quarantined(bad_name, bad_url)
        with pytest.raises(DatasetQuarantineError):
            build_croissant_1_1_rai_metadata(
                dataset_name="bad",
                version="0.1",
                description="bad",
                source_datasets=[{"name": bad_name, "url": bad_url, "license": "unknown"}],
            )


def test_stage4_hnm_negative_mix_and_demo_max_box_frac_filter() -> None:
    """Verify Stage-4 ~15% negative clutter mixing in train_g4.py and optional --max-box-frac filter."""
    ds = SyntheticG3aClipDataset(num_pos=12, num_neg=8, height=64, width=64, clip_len=3, seed=42)
    pos_aot = [{**c, "source_domain": "aot", "is_neg_marker": False} for c in ds.positive_clips]
    pos_g3a = [{**c, "source_domain": "g3a", "is_neg_marker": False} for c in ds.positive_clips]
    neg_aot = [{**c, "source_domain": "aot", "is_neg_marker": True} for c in ds.negative_clips]
    mixer = AotG3aBatchMixer(pos_aot, pos_g3a, aot_ratio=0.70, seed=42)

    # bs=64 -> round(64 * 0.15) = 10 negatives, 54 positives
    b64 = _sample_stage4_batch(mixer, neg_aot, batch_size=64, neg_ratio=0.15, rng_index=3)
    assert len(b64) == 64
    n_neg_64 = sum(1 for item in b64 if bool(item["is_neg_marker"]))
    assert n_neg_64 == 10

    # bs=4 -> at least 1 negative clip, 3 positives
    b4 = _sample_stage4_batch(mixer, neg_aot, batch_size=4, neg_ratio=0.15, rng_index=1)
    assert len(b4) == 4
    n_neg_4 = sum(1 for item in b4 if bool(item["is_neg_marker"]))
    assert n_neg_4 == 1

    # Edge cases: bs=1 or empty neg_clips safely return pure positives without crashing
    b1 = _sample_stage4_batch(mixer, neg_aot, batch_size=1, neg_ratio=0.15, rng_index=0)
    assert len(b1) == 1
    assert b1[0]["is_neg_marker"] is False

    b_empty = _sample_stage4_batch(mixer, [], batch_size=8, neg_ratio=0.15, rng_index=0)
    assert len(b_empty) == 8
    assert all(not bool(item["is_neg_marker"]) for item in b_empty)

    # Optional --max-box-frac spatial clutter filter in render_aot_d2_demo_video.py
    small_acft = video_mod.Box2D(x1=100.0, y1=100.0, x2=114.0, y2=110.0, label="AIRCRAFT", score=0.9)
    huge_clutter = video_mod.Box2D(x1=50.0, y1=200.0, x2=350.0, y2=420.0, label="AIRCRAFT", score=0.4)
    raw_boxes = [small_acft, huge_clutter]

    # Default None leaves boxes unchanged
    assert video_mod.filter_boxes_by_max_frac(raw_boxes, frame_w=960, frame_h=540, max_box_frac=None) == raw_boxes
    # 0.25 drops huge_clutter (300/960=0.31 > 0.25, 220/540=0.41 > 0.25) while keeping small_acft
    filtered = video_mod.filter_boxes_by_max_frac(raw_boxes, frame_w=960, frame_h=540, max_box_frac=0.25)
    assert filtered == [small_acft]
