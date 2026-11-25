"""Unit tests for WP-05 G0 acceptance items V5 (determinism), V6 (runtime bench), and truth cache handling.

Conforms strictly to docs/xvs/WP05_TRACKING.md §5 and GEMINI_G-B6.md §1 item F4.
"""

from copy import deepcopy
import hashlib
import json
import pytest
import time
from pathlib import Path

from xvs.sim.trajectory import t1_cruise_state
from xvs.traffic.encounter import EncounterSamplerV1
from xvs.tracking.evaluate import (
    run_evaluation,
    set_truth_cache_dir,
    load_or_compute_truth_cache,
    compute_truth_cache_key,
)


@pytest.fixture(autouse=True)
def isolate_cache(tmp_path):
    """Ensure all tests in this module point truth cache to a temporary directory."""
    set_truth_cache_dir(tmp_path)
    yield
    set_truth_cache_dir(None)


def test_v5_determinism(tmp_path):
    """V5, determinism:
    The same seeds give byte-identical JSON outputs.
    Cache is directed to tmp_path so nothing is written into the repo.
    """
    sampler = EncounterSamplerV1()
    encs = sampler.generate(t1_cruise_state)[:15]
    seed = 8888

    # First run
    res1 = run_evaluation(encs, seed=seed, max_workers=2)
    res1_clean = deepcopy(res1)
    res1_clean.pop("runtime_s", None)
    res1_clean.pop("frames_per_second", None)
    res1_clean.pop("truth_cache", None)
    json1 = json.dumps(res1_clean, sort_keys=True, indent=2)
    sha1 = hashlib.sha256(json1.encode("utf-8")).hexdigest()

    # Second run
    res2 = run_evaluation(encs, seed=seed, max_workers=2)
    res2_clean = deepcopy(res2)
    res2_clean.pop("runtime_s", None)
    res2_clean.pop("frames_per_second", None)
    res2_clean.pop("truth_cache", None)
    json2 = json.dumps(res2_clean, sort_keys=True, indent=2)
    sha2 = hashlib.sha256(json2.encode("utf-8")).hexdigest()

    assert sha1 == sha2, f"V5 failed: Run 1 ({sha1}) != Run 2 ({sha2})"
    assert json1 == json2


@pytest.mark.bench
def test_v6_runtime_fast_bench():
    """V6, runtime sanity bench:
    Records runtime for 15 encounters and asserts nothing.
    """
    sampler = EncounterSamplerV1()
    encs = sampler.generate(t1_cruise_state)[:15]

    t0 = time.perf_counter()
    metrics = run_evaluation(encs, seed=1234, max_workers=4)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    fps = metrics.get("frames_per_second", 0.0)
    print(f"\n[V6 bench] 15 encounters: {elapsed:.2f} s ({fps:.1f} fps)")


def test_truth_cache_keying_and_recomputation(tmp_path):
    """Test F4 truth cache behavior:
    - First run generates cache (miss);
    - Second run uses cache (hit);
    - Corrupt cache recomputes and records status/reason without raising;
    - Mismatched key recomputes and records status/reason without raising.
    """
    sampler = EncounterSamplerV1()
    encs = sampler.generate(t1_cruise_state)[:5]

    # Run 1: miss -> created
    _, info1 = load_or_compute_truth_cache(encs, max_workers=1, cache_dir=tmp_path)
    assert info1["status"] == "miss"

    # Run 2: hit
    _, info2 = load_or_compute_truth_cache(encs, max_workers=1, cache_dir=tmp_path)
    assert info2["status"] == "hit"

    # Corrupt the cache file
    key = compute_truth_cache_key(encs)
    cache_file = tmp_path / f"truth_{key[:16]}.json"
    assert cache_file.is_file()
    cache_file.write_text("{corrupt json content!!", encoding="utf-8")

    # Run 3: corrupt -> recomputed
    _, info3 = load_or_compute_truth_cache(encs, max_workers=1, cache_dir=tmp_path)
    assert info3["status"] == "recomputed"
    assert "corrupt" in info3["reason"]

    # Mismatch the key in the cache file
    with open(cache_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["key"] = "0000000000000000000000000000000000000000000000000000000000000000"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    # Run 4: key mismatch -> recomputed
    _, info4 = load_or_compute_truth_cache(encs, max_workers=1, cache_dir=tmp_path)
    assert info4["status"] == "recomputed"
    assert "key_mismatch" in info4["reason"]
    assert info4["save_failed"] is False


def test_truth_cache_save_failure_recorded(tmp_path, monkeypatch):
    """Test F-B6-3: when cache save encounters OSError, save_failed is recorded as True."""
    sampler = EncounterSamplerV1()
    encs = sampler.generate(t1_cruise_state)[:2]

    orig_open = open

    def mock_open(file, mode="r", *args, **kwargs):
        if "w" in mode and str(tmp_path) in str(file):
            raise OSError("Simulated disk full or permission error")
        return orig_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", mock_open)
    _, info = load_or_compute_truth_cache(encs, max_workers=1, cache_dir=tmp_path)
    assert info["save_failed"] is True
    assert "save_error" in info
    assert "Simulated disk full" in info["save_error"]


def test_d29_common_random_numbers_and_w7_miss_decomposition():
    """Verify D29 Common Random Numbers (true-target detections and INS drift identical across lambda)
    and W7 miss decomposition at t_CPA - 12.5 s.
    """
    import numpy as np
    from dataclasses import asdict
    from xvs.traffic.encounter import construct_encounter
    from xvs.tracking.measurements import SyntheticMeasurementGenerator
    from xvs.tracking.evaluate import get_ownship_states, simulate_single_encounter
    from scripts.xvs.tracking.run_lambda_sweep_v2 import (
        garwood_poisson_rate_ci,
        summarize_miss_decomposition,
    )

    sampler = EncounterSamplerV1()
    enc_params = sampler.generate(t1_cruise_state)[0]
    enc = construct_encounter(t1_cruise_state, enc_params)
    ownship_states = get_ownship_states()[:60]

    base_cfg = json.loads(
        (Path(__file__).resolve().parents[2] / "configs" / "xvs" / "tracking" / "g0_measured_d0.json").read_text(encoding="utf-8")
    )

    target_dets_by_lam = {}
    ins_by_lam = {}
    for lam in (0.0005, 0.5, 5.0):
        cfg = dict(base_cfg, fppi_lambda=lam)
        gen = SyntheticMeasurementGenerator(config=cfg, seed=1010000)
        tdets = []
        ins_list = []
        for k, st_o in enumerate(ownship_states):
            dets, meta = gen.generate_frame(k * 0.1, st_o, [enc])
            tdets.append([d.to_dict() for d in dets if not d.is_false_alarm])
            ins_list.append(meta["r_ned_body_ins"].tolist())
        target_dets_by_lam[lam] = tdets
        ins_by_lam[lam] = ins_list

    assert target_dets_by_lam[0.0005] == target_dets_by_lam[0.5] == target_dets_by_lam[5.0]
    assert ins_by_lam[0.0005] == ins_by_lam[0.5] == ins_by_lam[5.0]

    # Verify W7 miss decomposition fields and Garwood Poisson bound
    res = simulate_single_encounter(asdict(enc_params), dict(base_cfg, fppi_lambda=0.0005), seed=1010000)
    assert "miss_check_at_tcpa_minus_12_5" in res
    assert "true_track_coverage" in res
    mc = res["miss_check_at_tcpa_minus_12_5"]
    for key in ("no_confirmed_track", "tau_s_unavailable", "tau_s_ge_T", "omega_ge_thr", "joint_condition", "overall_limit_type"):
        assert key in mc

    decomp = summarize_miss_decomposition([dict(res, is_threat=True, lead_a1_warn_s=None)])
    assert decomp["late_or_missed_runs_lt_12_5s"] == 1
    gw = garwood_poisson_rate_ci(0, 75.0)
    assert gw["rate"] == 0.0
    assert 0.0 < gw["upper_95"] < 0.05


def test_d32_ever_confirmed_false_track_deleted_after_4s_is_counted(monkeypatch):
    """D32 (F2): A false track that confirms (3-of-5) and is later deleted at 4 s without updates
    must be counted in `false_tracks` (`false_tracks_ever_confirmed`) and excluded from `false_tracks_end_of_run`.
    """
    from dataclasses import asdict
    from xvs.tracking.measurements import DetectionRecord, SyntheticMeasurementGenerator
    from xvs.tracking.evaluate import aggregate_dataset_metrics, simulate_single_encounter

    sampler = EncounterSamplerV1()
    enc_params = sampler.generate(t1_cruise_state)[0]
    base_cfg = json.loads(
        (Path(__file__).resolve().parents[2] / "configs" / "xvs" / "tracking" / "g0_measured_d0.json").read_text(encoding="utf-8")
    )
    cfg = dict(base_cfg, fppi_lambda=0.0)

    orig_gen_frame = SyntheticMeasurementGenerator.generate_frame

    def patched_generate_frame(self, t_s, ownship, intruders):
        dets, meta = orig_gen_frame(self, t_s, ownship, intruders)
        # Keep only true-target detections, plus inject a 3-frame false track at t = 0.0, 0.1, 0.2 s
        true_dets = [d for d in dets if not d.is_false_alarm]
        if abs(t_s - 0.0) < 1e-6 or abs(t_s - 0.1) < 1e-6 or abs(t_s - 0.2) < 1e-6:
            true_dets.append(
                DetectionRecord(
                    t=t_s,
                    camera=self.ring.cameras[0].id,
                    u=1280.0,
                    v=400.0,
                    s_px=3.0,
                    score=25.0,
                    is_false_alarm=True,
                    truth_id=None,
                )
            )
        return true_dets, meta

    monkeypatch.setattr(SyntheticMeasurementGenerator, "generate_frame", patched_generate_frame)

    res = simulate_single_encounter(asdict(enc_params), cfg, seed=1010000)
    assert res["false_tracks"] == 1
    assert res["false_tracks_ever_confirmed"] == 1
    assert res["false_tracks_end_of_run"] == 0

    agg = aggregate_dataset_metrics([res], cfg)
    assert agg["tracking"]["false_tracks_total"] == 1
    assert agg["tracking"]["false_tracks_ever_confirmed_total"] == 1
    assert agg["tracking"]["false_tracks_end_of_run_total"] == 0


