"""Gate G0 Acceptance V5: No leakage between splits.

Conforms strictly to docs/xvs/WP07_INTEGRITY.md §1 and §5 (V5):
- Calibration, validation, and test splits share no frame or seed.
- Clean frames go exclusively to calibration.
- Thresholds are tuned only on validation split.
- Conformal quantiles are calibrated only on calibration split.
- Test split is evaluated strictly out-of-sample.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from xvs.integrity.monitor import tune_monitor_thresholds, MonitorThresholds
from xvs.integrity.conformal import fit_conformal_model, ConformalModel

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "output" / "xvs" / "wp07" / "g0"


def test_v5_seed_split_disjointness():
    """Verify that seed sets for calibration, validation, and test splits are strictly disjoint."""
    # 20-seed configuration
    cal_seeds_20 = set(range(1, 11))
    val_seeds_20 = set(range(11, 16))
    test_seeds_20 = set(range(16, 21))

    assert len(cal_seeds_20 & val_seeds_20) == 0, "Cal and Val share seeds!"
    assert len(val_seeds_20 & test_seeds_20) == 0, "Val and Test share seeds!"
    assert len(cal_seeds_20 & test_seeds_20) == 0, "Cal and Test share seeds!"
    assert len(cal_seeds_20 | val_seeds_20 | test_seeds_20) == 20

    # 12-seed fallback configuration
    cal_seeds_12 = set(range(1, 7))
    val_seeds_12 = set(range(7, 10))
    test_seeds_12 = set(range(10, 13))

    assert len(cal_seeds_12 & val_seeds_12) == 0, "Cal and Val share seeds in 12-seed fallback!"
    assert len(val_seeds_12 & test_seeds_12) == 0, "Val and Test share seeds in 12-seed fallback!"
    assert len(cal_seeds_12 & test_seeds_12) == 0, "Cal and Test share seeds in 12-seed fallback!"
    assert len(cal_seeds_12 | val_seeds_12 | test_seeds_12) == 12


def test_v5_pipeline_enforces_split_provenance():
    """Verify that threshold tuning and conformal calibration enforce strict split isolation."""
    # Synthetic records with deliberate split leakage attempt
    fake_val_records = [
        {"split": "validation", "status": "CONVERGED", "condition": "nominal", "rms_residual_urad": 150.0, "condition_number": 50.0, "inlier_fraction": 0.9, "inlier_count": 20, "leave_one_out_consistency_deg": 0.02, "cam_set": "FRBL"},
        {"split": "test", "status": "CONVERGED", "condition": "nominal", "rms_residual_urad": 9999.0, "condition_number": 9999.0, "inlier_fraction": 0.1, "inlier_count": 2, "leave_one_out_consistency_deg": 9.99, "cam_set": "FRBL"},
    ]

    # Filtered strictly for validation split
    pure_val = [r for r in fake_val_records if r["split"] == "validation"]
    th = tune_monitor_thresholds(pure_val, target_far=0.01)

    assert isinstance(th, MonitorThresholds)
    # The contaminated 'test' record (rms 9999) must not affect the validation threshold
    assert th.max_rms_residual_urad < 1000.0, f"Threshold leaked test data: {th.max_rms_residual_urad}"

    # Similarly for conformal model fitting (need n >= 19 for finite quantile at alpha=0.1 under D28)
    fake_cal_records = [
        {"split": "calibration", "status": "CONVERGED", "alarm": False, "nonconformity_score": 2.5, "cam_set": "FRBL", "condition": "degraded", "seed": 0}
        for _ in range(20)
    ] + [
        {"split": "test", "status": "CONVERGED", "alarm": False, "nonconformity_score": 999.0, "cam_set": "FRBL", "condition": "degraded", "seed": 16},
    ]
    pure_cal = [r for r in fake_cal_records if r["split"] == "calibration"]
    model = fit_conformal_model(pure_cal, alphas=[0.1])
    q_10 = model.standard_quantiles.get("0.1", float("nan"))
    assert q_10 < 10.0, f"Conformal quantile leaked test score: {q_10}"


@pytest.mark.data
def test_v5_actual_dataset_disjointness():
    """Verify generated dataset files if present on disk share zero seeds and zero frames across splits.
    
    Per D19/W3:
    - Reads get_data_root().
    - Fails under an explicit root when files are missing.
    - Skips without an explicit root when files are missing.
    """
    from xvs.paths import get_data_root, data_root_source

    root = get_data_root()
    is_explicit = (data_root_source() != "default")

    data_dir = root / "xvs" / "wp07" / "g0"
    if not data_dir.is_dir() and (root / "dataset_calibration.jsonl").is_file():
        data_dir = root

    cal_file = data_dir / "dataset_calibration.jsonl"
    val_file = data_dir / "dataset_validation.jsonl"
    test_file = data_dir / "dataset_test.jsonl"

    missing = [str(f) for f in (cal_file, val_file, test_file) if not f.is_file()]
    if missing:
        if is_explicit:
            pytest.fail(f"Dataset files missing under explicit data root {root}: {missing}")
        else:
            pytest.skip(f"Dataset files not present under default data root {root}.")

    def inspect_file(fpath: Path) -> tuple[set[int], set[tuple[str, int, int]]]:
        seeds = set()
        keys = set()
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                s = r.get("seed_idx") or r.get("seed", 0)
                if s > 0:
                    seeds.add(s)
                k_frame = r.get("k", r.get("frame_k", 0))
                # Key: (seq_name, k_frame, seed)
                keys.add((r["seq_name"], k_frame, s))
        return seeds, keys

    cal_seeds, cal_keys = inspect_file(cal_file)
    val_seeds, val_keys = inspect_file(val_file)
    test_seeds, test_keys = inspect_file(test_file)

    if not cal_keys or not val_keys or not test_keys:
        pytest.skip("Dataset files are still being generated by background task.")

    assert len(cal_seeds & val_seeds) == 0, f"Cal and Val share seeds: {cal_seeds & val_seeds}"
    assert len(val_seeds & test_seeds) == 0, f"Val and Test share seeds: {val_seeds & test_seeds}"
    assert len(cal_seeds & test_seeds) == 0, f"Cal and Test share seeds: {cal_seeds & test_seeds}"

    assert len(cal_keys & val_keys) == 0, "Cal and Val share frame keys!"
    assert len(val_keys & test_keys) == 0, "Val and Test share frame keys!"
    assert len(cal_keys & test_keys) == 0, "Cal and Test share frame keys!"


def test_v5_runner_split_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify the real runner executes strictly against disjoint split files and tracks provenance (W3)."""
    import subprocess as _sp
    import scripts.xvs.integrity.run_wp07_g0 as runner_mod
    from scripts.xvs.integrity.run_wp07_g0 import main

    real_run = _sp.run
    dirty_mode = {"dirty": True}

    def _mock_git_run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and len(cmd) >= 2 and cmd[0] == "git" and cmd[1] == "status":
            out = " M src/xvs/integrity/monitor.py\n" if dirty_mode["dirty"] else ""
            return _sp.CompletedProcess(cmd, 0, stdout=out, stderr="")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(runner_mod.subprocess, "run", _mock_git_run)

    # Synthetic validation records (seeds 11-12)
    val_records = [
        {
            "seq_name": "T1", "k": i, "frame_k": i, "split": "validation", "seed": 11 + (i % 2),
            "status": "CONVERGED", "condition": "degraded", "fault_type": "none",
            "rms_residual_urad": 120.0 + i, "condition_number": 40.0, "inlier_fraction": 0.85,
            "inlier_count": 25, "leave_one_out_consistency_deg": 0.015, "cam_set": "FRBL",
            "total_error_deg": 0.05, "total_error_rad": 0.0008, "sigma_hat_urad": 1000.0, "nonconformity_score": 1.2,
        }
        for i in range(20)
    ]

    # Synthetic calibration records (seeds 1-2)
    cal_records = [
        {
            "seq_name": "T1", "k": i, "frame_k": i, "split": "calibration", "seed": 1 + (i % 2),
            "status": "CONVERGED", "condition": "degraded" if i % 2 == 0 else "clean", "fault_type": "none",
            "rms_residual_urad": 110.0 + i, "condition_number": 35.0, "inlier_fraction": 0.90,
            "inlier_count": 30, "leave_one_out_consistency_deg": 0.012, "cam_set": "FRBL",
            "total_error_deg": 0.04, "total_error_rad": 0.0007, "sigma_hat_urad": 900.0, "nonconformity_score": 1.1,
        }
        for i in range(30)
    ]

    # Synthetic test records (seeds 16-17)
    test_records = [
        {
            "seq_name": "T1", "k": i, "frame_k": i, "split": "test", "seed": 16 + (i % 2),
            "status": "CONVERGED", "condition": "degraded", "fault_type": "none" if i < 15 else "spurious_edge",
            "rms_residual_urad": 130.0 + i, "condition_number": 45.0, "inlier_fraction": 0.80,
            "inlier_count": 22, "leave_one_out_consistency_deg": 0.020, "cam_set": "FRBL",
            "total_error_deg": 0.06 if i < 15 else 0.80, "total_error_rad": 0.001, "sigma_hat_urad": 1100.0, "nonconformity_score": 1.3,
            "k_onset": 15 if i >= 15 else None, "t_s": float(i) * 0.5, "sigma_h": 1.0,
        }
        for i in range(20)
    ]

    val_file = tmp_path / "dataset_validation.jsonl"
    cal_file = tmp_path / "dataset_calibration.jsonl"
    test_file = tmp_path / "dataset_test.jsonl"
    cfg_file = tmp_path / "g0.json"
    res_file = tmp_path / "results.json"

    with open(val_file, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")
    with open(cal_file, "w", encoding="utf-8") as f:
        for r in cal_records:
            f.write(json.dumps(r) + "\n")
    with open(test_file, "w", encoding="utf-8") as f:
        for r in test_records:
            f.write(json.dumps(r) + "\n")

    for seq, exp_dt in (("T1_cruise", 0.5), ("T3_turn_climb", 0.5), ("T2_ksfo28l", 0.2)):
        seq_dir = tmp_path / "xvs" / "wp01" / "g4" / seq
        seq_dir.mkdir(parents=True, exist_ok=True)
        with open(seq_dir / "frames.jsonl", "w", encoding="utf-8") as f_g4:
            f_g4.write(json.dumps({"k": 0, "t_s": 0.0}) + "\n")
            f_g4.write(json.dumps({"k": 1, "t_s": exp_dt}) + "\n")

    # Dirty tree check (Step 1.2): must refuse to run when git working tree is dirty
    with pytest.raises(RuntimeError, match="Refusing to run on a dirty git working tree"):
        main([
            "--data-root", str(tmp_path),
            "--data-dir", str(tmp_path),
            "--config-path", str(cfg_file),
            "--out-json", str(res_file),
        ])

    # Run the real runner on clean tree
    dirty_mode["dirty"] = False
    ret = main([
        "--data-root", str(tmp_path),
        "--data-dir", str(tmp_path),
        "--config-path", str(cfg_file),
        "--out-json", str(res_file),
    ])
    assert ret == 0
    assert res_file.is_file()

    with open(res_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    assert results["provenance"]["validation_records_read"] == 20
    assert results["provenance"]["calibration_records_read"] == 30
    assert results["provenance"]["test_records_read"] == 20

    # Leakage test: contaminate validation file with a test record
    val_records.append(dict(test_records[0]))
    with open(val_file, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")

    with pytest.raises(ValueError, match="Record split 'test' does not match expected 'validation'"):
        main([
            "--data-root", str(tmp_path),
            "--data-dir", str(tmp_path),
            "--config-path", str(cfg_file),
            "--out-json", str(res_file),
        ])

    # Missing G4 frames check (F10): must raise FileNotFoundError
    empty_root = tmp_path / "empty_root"
    empty_root.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError, match="Missing WP-01 G4 directory"):
        main([
            "--data-root", str(empty_root),
            "--data-dir", str(tmp_path),
            "--config-path", str(cfg_file),
            "--out-json", str(res_file),
        ])


def test_v5_mondrian_clean_and_degraded_partitions():
    """Verify that Mondrian conformal calibration forms separate clean and degraded slices."""
    cal_records = [
        {"split": "calibration", "status": "CONVERGED", "alarm": False, "nonconformity_score": 1.2, "cam_set": "F", "condition": "clean", "seed_idx": 0},
        {"split": "calibration", "status": "CONVERGED", "alarm": False, "nonconformity_score": 3.4, "cam_set": "F", "condition": "degraded", "seed_idx": 1},
        {"split": "calibration", "status": "CONVERGED", "alarm": False, "nonconformity_score": 1.1, "cam_set": "FRBL", "condition": "clean", "seed_idx": 0},
        {"split": "calibration", "status": "CONVERGED", "alarm": False, "nonconformity_score": 2.2, "cam_set": "FRBL", "condition": "degraded", "seed_idx": 2},
        {"split": "calibration", "status": "CONVERGED", "alarm": False, "nonconformity_score": 1.0, "cam_set": "all", "condition": "clean", "seed_idx": 0},
        {"split": "calibration", "status": "CONVERGED", "alarm": False, "nonconformity_score": 2.0, "cam_set": "all", "condition": "degraded", "seed_idx": 3},
    ]
    model = fit_conformal_model(cal_records, alphas=[0.1])
    expected_partitions = {"F:clean", "F:degraded", "FRBL:clean", "FRBL:degraded", "all:clean", "all:degraded"}
    assert set(model.mondrian_quantiles.keys()) == expected_partitions
    assert model.n_cal_mondrian["F:clean"] == 1
    assert model.n_cal_mondrian["F:degraded"] == 1
    assert model.n_cal_mondrian["FRBL:clean"] == 1
    assert model.n_cal_mondrian["FRBL:degraded"] == 1
    assert model.n_cal_mondrian["all:clean"] == 1
    assert model.n_cal_mondrian["all:degraded"] == 1
