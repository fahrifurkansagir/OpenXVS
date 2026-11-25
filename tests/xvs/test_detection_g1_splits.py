"""V1_G1 — Split and post-build pack leakage verification tests (`WP04_DETECTION.md` §9.2, §9.3, §9.7).

Verifies Decision D34 (S1, S2, S7, S9):
1. 70/15/15 (`126 / 27 / 27`) stratified encounter-disjoint partitioning of the `seed=20260926`
   `E####` G3a v0.1 subsample (`E0001`–`E0900`).
2. Disjoint ownship time-block partitioning of negative ROIs along `T1` (`[0, 42)`, `[42, 51)`, `[51, 60]`).
3. Seen-backgrounds declaration, AOT background-generalisation reference, and `total_g1_cu_cap <= 60`.
4. Post-build `.tar.zst` archive and manifest leakage audit (`verify_pack_no_leakage`, S9).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import pytest

from xvs.detection.pack import (
    build_both_g1_packs,
    build_g1_pack_archive,
    generate_synthetic_pack_items_from_splits,
    load_g1_pack_archive,
    verify_pack_no_leakage,
)
from xvs.detection.splits import (
    DEFAULT_SPLITS_CONFIG,
    assign_encounter_split,
    assign_negative_split_by_time,
    build_g1_splits_dict,
    get_g3a_v01_subsample_by_class,
    load_splits_config,
    verify_splits_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_v1_g1_stratified_encounter_disjoint_and_time_block_splits() -> None:
    cfg, sha256_hex = load_splits_config(DEFAULT_SPLITS_CONFIG)
    assert len(sha256_hex) == 64
    assert verify_splits_config(cfg) == []

    # Rebuild from scratch and confirm exact determinism
    rebuilt = build_g1_splits_dict()
    assert rebuilt == cfg

    # Verify exact 70/15/15 counts (126 / 27 / 27 = 180)
    assert cfg["encounter_counts"] == {
        "total_encounters": 180,
        "train_encounters": 126,
        "val_encounters": 27,
        "test_encounters": 27,
    }

    # Verify exact match with G3a v0.1 seed=20260926 subsample
    expected_by_class = get_g3a_v01_subsample_by_class(n_encounters=180, subsample_seed=20260926)
    expected_all = sorted(eid for eids in expected_by_class.values() for eid in eids)
    assert cfg["encounters"]["all_g3a_subsample"] == expected_all

    # Verify class stratification (each of the 6 classes has 30 total: 21 train, 4 or 5 val, 4 or 5 test)
    strat = cfg["class_stratification"]
    assert len(strat) == 6
    for cid, splits in strat.items():
        tr_c = set(splits["train"])
        va_c = set(splits["val"])
        te_c = set(splits["test"])
        assert len(tr_c) == 21
        assert len(va_c) in (4, 5)
        assert len(te_c) in (4, 5)
        assert len(va_c) + len(te_c) == 9
        assert (tr_c | va_c | te_c) == set(expected_by_class[cid])

    # Verify negative time blocks along T1 (S1)
    assert assign_negative_split_by_time(0.0, cfg) == "train"
    assert assign_negative_split_by_time(41.0, cfg) == "train"
    assert assign_negative_split_by_time(42.0, cfg) == "val"
    assert assign_negative_split_by_time(50.0, cfg) == "val"
    assert assign_negative_split_by_time(51.0, cfg) == "test"
    assert assign_negative_split_by_time(60.0, cfg) == "test"

    # Verify seen backgrounds statement, AOT generalisation role, and compute caps (S1, S9)
    assert "seen backgrounds" in cfg["seen_backgrounds_statement"]
    assert cfg["background_generalisation_test"]["dataset_id"] == "AOT-v1"
    assert cfg["compute_caps"]["total_g1_cu_cap"] == 60
    assert cfg["compute_caps"]["cu_cap"] == 60


def test_v1_g1_post_build_pack_leakage_audit_and_determinism(tmp_path: Path) -> None:
    # Build strictly into tmp_path (B8-m1)
    summary_tmp = build_both_g1_packs(out_dir=tmp_path / "packs_a")
    summary_tmp_b = build_both_g1_packs(out_dir=tmp_path / "packs_b")

    assert summary_tmp["leakage_errors"] == []
    assert summary_tmp_b["leakage_errors"] == []

    # Archives built twice from identical inputs are byte-identical
    assert (
        summary_tmp["train_val_manifest"]["archive_sha256"]
        == summary_tmp_b["train_val_manifest"]["archive_sha256"]
    )
    assert (
        summary_tmp["test_manifest"]["archive_sha256"]
        == summary_tmp_b["test_manifest"]["archive_sha256"]
    )

    # Round-trip unpack via load_g1_pack_archive and verify SHA-256 tamper detection (B8-F1)
    tv_unpacked = load_g1_pack_archive(Path(summary_tmp["train_val_archive"]))
    te_unpacked = load_g1_pack_archive(Path(summary_tmp["test_archive"]))
    assert len(tv_unpacked) == summary_tmp["train_val_manifest"]["item_count"]
    assert len(te_unpacked) == summary_tmp["test_manifest"]["item_count"]

    # Tampering with frames_sha256 in manifest raises ValueError on unpack
    man_file = tmp_path / "packs_a" / "pack_manifest.json"
    man_obj = json.loads(man_file.read_text(encoding="utf-8"))
    man_obj["items"][0]["frames_sha256"] = "0" * 64
    man_file.write_text(json.dumps(man_obj, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_g1_pack_archive(Path(summary_tmp["train_val_archive"]), manifest_path=man_file)


def test_v1_g1_pack_builder_rejects_test_leakage_and_bad_dt(tmp_path: Path) -> None:
    cfg, sha = load_splits_config(DEFAULT_SPLITS_CONFIG)
    tv_items, test_items = generate_synthetic_pack_items_from_splits(cfg)

    # 1. Attempting to include a test item in the train/val pack raises ValueError before build completes
    leaked_items = list(tv_items) + [test_items[0]]
    with pytest.raises(ValueError, match="not in allowed"):
        build_g1_pack_archive(
            archive_path=tmp_path / "bad_tv.tar.zst",
            manifest_path=tmp_path / "bad_tv.json",
            items=leaked_items,
            allowed_splits={"train", "val"},
            splits_cfg=cfg,
            splits_sha256=sha,
        )

    # 2. Attempting to build a stack with Delta_t != 1.0 s raises ValueError (D34-S2)
    bad_dt_item = copy.deepcopy(tv_items[0])
    bad_dt_item["timestamps_s"] = [1.0, 2.5, 4.0]
    with pytest.raises(ValueError, match="Delta_t=1.0 s"):
        build_g1_pack_archive(
            archive_path=tmp_path / "bad_dt.tar.zst",
            manifest_path=tmp_path / "bad_dt.json",
            items=[bad_dt_item],
            allowed_splits={"train", "val"},
            splits_cfg=cfg,
            splits_sha256=sha,
        )

    # 3. Post-build auditor catches tampered manifest containing a test encounter ID
    summary = build_both_g1_packs(out_dir=tmp_path / "tamper_check")
    tv_man_path = tmp_path / "tamper_check" / "pack_manifest.json"
    tampered = json.loads(tv_man_path.read_text(encoding="utf-8"))
    tampered["items"][0]["encounter_id"] = cfg["encounters"]["test"][0]
    tv_man_path.write_text(json.dumps(tampered, indent=2), encoding="utf-8")

    errs = verify_pack_no_leakage(
        train_val_archive=Path(summary["train_val_archive"]),
        train_val_manifest=tv_man_path,
        test_archive=Path(summary["test_archive"]),
        test_manifest=tmp_path / "tamper_check" / "test_pack_manifest.json",
        splits_cfg=cfg,
    )
    assert any("leaked into train/val pack" in e for e in errs)
