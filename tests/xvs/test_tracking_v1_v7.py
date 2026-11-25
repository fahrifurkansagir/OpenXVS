"""Unit tests for WP-05 G0 acceptance items V1 (generator exactness) and V7 (labels).

Conforms strictly to docs/xvs/WP05_TRACKING.md §5 and GEMINI_G-B3.md §1 item 1.
"""

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import pytest
import numpy as np
from xvs.sim.trajectory import t1_cruise_state
from xvs.traffic.encounter import (
    EncounterParams,
    EncounterSamplerV1,
    construct_encounter,
    evaluate_events,
    evaluate_actual_cpa,
    TestOwnshipO1,
)
from xvs.traffic.ring import TrafficRing
from xvs.tracking.measurements import (
    SyntheticMeasurementGenerator,
    rebuild_los_ned,
    get_model_vertices,
)

# Prevent pytest from treating TestOwnshipO1 as a test class
TestOwnshipO1.__test__ = False


def test_v1_generator_exactness():
    """V1, generator exactness:
    With zero noise, P_d ≡ 1, no false alarms and no INS error:
    - detections equal the exact projections within 1e-9 px;
    - the Earth-fixed LOS rebuilt from them equals the GT LOS within 1e-9 rad.
    """
    gen = SyntheticMeasurementGenerator(seed=123)
    ring = gen.ring

    # Create test encounter with C2_SEP
    enc_param = EncounterParams(
        id="E_V1_TEST",
        class_id="C2_SEP",
        model_id="C2_SEP_0001",
        t_cpa=45.0,
        hmd_m=500.0,
        side="right",
        vmd_m=100.0,
        approach_deg=175.0,
        gs_mps=60.0,
        vs_mps=0.0,
    )
    enc = construct_encounter(t1_cruise_state, enc_param)

    # Test across multiple time steps
    times_to_test = [10.0, 20.0, 30.0, 40.0, 45.0]
    matched_count = 0

    for t_s in times_to_test:
        st_o = t1_cruise_state(t_s)
        dets, meta = gen.generate_frame(
            t_s=t_s,
            ownship_state=st_o,
            encounters=[enc],
            disable_noise=True,
            pd_override=1.0,
        )

        gt_info = meta["gt_by_intruder"][enc_param.id]
        cams_gt = gt_info["cams"]

        for det in dets:
            assert not det.is_false_alarm
            assert det.truth_id == enc_param.id
            cam_id = det.camera

            # 1. Detection equals exact projection within 1e-9 px
            assert cam_id in cams_gt
            cam_gt = cams_gt[cam_id]
            u_true = cam_gt["u_true"]
            v_true = cam_gt["v_true"]

            err_u = abs(det.u - u_true)
            err_v = abs(det.v - v_true)
            assert err_u < 1e-9, f"u error {err_u:.3e} exceeds 1e-9 px"
            assert err_v < 1e-9, f"v error {err_v:.3e} exceeds 1e-9 px"

            # 2. Earth-fixed LOS rebuilt from detection equals GT LOS within 1e-9 rad
            r_ned_body_ins = meta["r_ned_body_ins"]
            rebuilt_los = rebuild_los_ned(det.u, det.v, cam_id, ring, r_ned_body_ins)

            gt_los = cam_gt["gt_cam_los_ned"]
            cross_norm = float(np.linalg.norm(np.cross(rebuilt_los, gt_los)))
            dot_val = float(np.dot(rebuilt_los, gt_los))
            ang_err_rad = math.atan2(cross_norm, dot_val)

            assert ang_err_rad < 1e-9, f"Angular error {ang_err_rad:.3e} exceeds 1e-9 rad"
            matched_count += 1

    assert matched_count > 0, "No detections tested for V1"


def test_v7_event_labels_and_times_o1_and_boundaries():
    """V7, labels and event times match WP-02 G1 event evaluation.
    Hermetic fast test on O1 boundary cases and a fixed seeded subset of 30 sampler-v1 encounters.
    """
    o1 = TestOwnshipO1()
    test_cases = [
        ("lowc_yes", 500.0, 50.0, True),
        ("lowc_no", 800.0, 100.0, False),
        ("boundary_inside", 609.5, 76.1, True),
        ("boundary_outside", 609.7, 76.3, False),
    ]

    for name, dh, dv, exp_threat in test_cases:
        p = EncounterParams(
            id=name,
            class_id="C2_SEP",
            model_id="C2_SEP_0001",
            t_cpa=35.0,
            hmd_m=dh,
            side="right",
            vmd_m=dv,
            approach_deg=180.0,
            gs_mps=60.0,
            vs_mps=0.0,
        )
        enc = construct_encounter(o1, p)
        ev = evaluate_events(o1, enc, t_start=34.0, t_end=36.0, dt=0.01)
        threat = ev["lowc_t"] is not None
        assert threat == exp_threat

    # Hermetic fast check: fixed seeded subset of 30 sampler-v1 encounters,
    # recomputed and compared against committed fixture tests/fixtures/v7_30_encounters.json (R-GB-9).
    fixture_path = Path(__file__).resolve().parent.parent / "fixtures" / "v7_30_encounters.json"
    assert fixture_path.is_file(), f"Fixture file missing: {fixture_path}"
    with open(fixture_path, "r", encoding="utf-8") as f:
        expected_30 = json.load(f)

    sampler = EncounterSamplerV1()
    encs_30 = sampler.generate(t1_cruise_state)[:30]
    assert len(encs_30) == 30
    for p in encs_30:
        enc = construct_encounter(t1_cruise_state, p)
        ev = evaluate_events(t1_cruise_state, enc, t_start=0.0, t_end=60.0, dt=0.01)
        cpa = evaluate_actual_cpa(t1_cruise_state, enc, t_start=0.0, t_end=60.0, dt=0.01)

        exp = expected_30[p.id]
        # Exact boolean labels
        assert (ev["lowc_t"] is not None) == exp["is_threat"], f"{p.id} threat label mismatch"
        assert (ev["nmac_t"] is not None) == exp["is_nmac"], f"{p.id} nmac label mismatch"

        # Event times within 1e-9 s
        if exp["lowc_t"] is not None:
            assert abs(ev["lowc_t"] - exp["lowc_t"]) < 1e-9, f"{p.id} lowc_t mismatch"
        else:
            assert ev["lowc_t"] is None

        if exp["nmac_t"] is not None:
            assert abs(ev["nmac_t"] - exp["nmac_t"]) < 1e-9, f"{p.id} nmac_t mismatch"
        else:
            assert ev["nmac_t"] is None

        # CPA kinematics within tolerances
        assert abs(cpa["t_cpa_s"] - exp["t_cpa_s"]) < 1e-6, f"{p.id} t_cpa mismatch"
        assert abs(cpa["hmd_m"] - exp["hmd_m"]) < 1e-4, f"{p.id} hmd_m mismatch"
        assert abs(cpa["vmd_m"] - exp["vmd_m"]) < 1e-4, f"{p.id} vmd_m mismatch"


@pytest.mark.data
def test_v7_full_900_labels_against_baseline(check_data):
    """Full-900 V7: compares truth labels in truth_labels.json with a fresh recomputation.
    Per encounter: labels exact, event times within 1e-9 s.
    No cross-root fallback (F-B6-1, F-B6-2).
    """
    from xvs.paths import get_data_root
    from xvs.tracking.evaluate import compute_encounter_truth, compute_truth_cache_key
    from multiprocessing import Pool

    data_root = get_data_root()
    truth_labels_path = data_root / "xvs" / "wp05" / "g0" / "truth_labels.json"
    check_data(truth_labels_path, "WP-05 truth labels JSON")

    with open(truth_labels_path, "r", encoding="utf-8") as f:
        truth_data = json.load(f)

    expected_encounters = truth_data.get("encounters", {})

    sampler = EncounterSamplerV1()
    encs = sampler.generate(t1_cruise_state)
    assert len(encs) == 900

    expected_key = compute_truth_cache_key(encs)
    assert truth_data.get("key") == expected_key, "Truth labels cache key mismatch"

    cores = os.cpu_count() or 4
    workers = max(1, cores - 4)

    tasks = [asdict(e) for e in encs]
    if workers > 1:
        with Pool(processes=workers) as pool:
            computed = pool.map(compute_encounter_truth, tasks)
    else:
        computed = [compute_encounter_truth(t) for t in tasks]

    assert len(computed) == len(expected_encounters), f"Count mismatch: {len(computed)} vs {len(expected_encounters)}"

    for eid, truth_dict in computed:
        assert eid in expected_encounters, f"Missing encounter {eid} in truth_labels.json"
        exp = expected_encounters[eid]

        ev = truth_dict["events"]
        cpa = truth_dict["cpa"]
        comp_threat = (ev["lowc_t"] is not None)

        # 1. Exact label match
        assert comp_threat == exp["threat"], f"Encounter {eid} threat mismatch: {comp_threat} vs {exp['threat']}"

        # 2. Event times within 1e-9 s
        if ev["lowc_t"] is None:
            assert exp["lowc_t"] is None, f"Encounter {eid} lowc_t mismatch: None vs {exp['lowc_t']}"
        else:
            assert exp["lowc_t"] is not None, f"Encounter {eid} lowc_t mismatch: {ev['lowc_t']} vs None"
            assert abs(ev["lowc_t"] - exp["lowc_t"]) <= 1e-9, f"Encounter {eid} lowc_t diff > 1e-9"

        if ev["nmac_t"] is None:
            assert exp["nmac_t"] is None, f"Encounter {eid} nmac_t mismatch: None vs {exp['nmac_t']}"
        else:
            assert exp["nmac_t"] is not None, f"Encounter {eid} nmac_t mismatch: {ev['nmac_t']} vs None"
            assert abs(ev["nmac_t"] - exp["nmac_t"]) <= 1e-9, f"Encounter {eid} nmac_t diff > 1e-9"

        assert abs(cpa["t_cpa_s"] - exp["t_cpa"]) <= 1e-9, f"Encounter {eid} t_cpa diff > 1e-9"
        assert abs(cpa["hmd_m"] - exp["hmd"]) <= 1e-9, f"Encounter {eid} hmd diff > 1e-9"
