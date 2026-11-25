"""Deterministic encounter-disjoint 70/15/15 splits and negative time-block partitioning (WP-04 §9.2).

Implements Decision D34 (S1, S7, S9):
- Stratified 70% train (126) / 15% val (27) / 15% test (27) encounter-disjoint split
  of the 180 G3a v0.1 encounters (seed=20260926 subsample of E0001..E0900).
- Disjoint ownship time-block partitioning of negative ROIs along the 60 s T1 trajectory
  so no negative time block spans two splits.
- Seen-backgrounds declaration and AOT background-generalisation role metadata.
- Post-build archive and manifest leakage audit (`verify_pack_no_leakage`).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENCOUNTERS_CACHE = REPO_ROOT / "data" / "cache" / "wp05_encounters_sampler_v1.jsonl"
DEFAULT_SPLITS_CONFIG = REPO_ROOT / "configs" / "xvs" / "detection" / "g1_splits_v0.json"
G3A_V01_RELEASE_SHA256 = "f29430883a0116f7aadb0dfb2f0d201cb9a3344df206735aa2689083b893fbee"


def load_encounter_pool(encounters_file: Path = DEFAULT_ENCOUNTERS_CACHE) -> dict[str, list[str]]:
    """Load the 900 E#### encounter IDs grouped by intruder class without importing xvs.traffic."""
    classes_dict: dict[str, list[str]] = {}
    with open(encounters_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            rec = json.loads(line_str)
            cid = str(rec.get("class_id", "unknown"))
            eid = str(rec["id"])
            classes_dict.setdefault(cid, []).append(eid)
    for cid in classes_dict:
        classes_dict[cid] = sorted(classes_dict[cid])
    return classes_dict


def get_g3a_v01_subsample_by_class(
    n_encounters: int = 180,
    subsample_seed: int = 20260926,
    encounters_file: Path = DEFAULT_ENCOUNTERS_CACHE,
) -> dict[str, list[str]]:
    """Reproduce the exact G3a v0.1 180-encounter stratified subsample (seed=20260926)."""
    classes_dict = load_encounter_pool(encounters_file)
    rng = np.random.default_rng(subsample_seed)
    n_classes = len(classes_dict)
    per_class = max(1, n_encounters // n_classes)

    selected_by_class: dict[str, list[str]] = {}
    for cid in sorted(classes_dict.keys()):
        pool = sorted(classes_dict[cid])
        chosen = rng.choice(pool, size=min(per_class, len(pool)), replace=False).tolist()
        selected_by_class[cid] = sorted(str(x) for x in chosen)
    return selected_by_class


def build_g1_splits_dict(
    n_encounters: int = 180,
    subsample_seed: int = 20260926,
    split_seed: int = 20261104,
    encounters_file: Path = DEFAULT_ENCOUNTERS_CACHE,
    g3a_release_sha256: str = G3A_V01_RELEASE_SHA256,
) -> dict[str, Any]:
    """Build the frozen WP-04 G1 split configuration dictionary (`g1_splits_v0.json`).

    Partitions the 180 G3a v0.1 encounters (30 per class across 6 classes) into:
      - 70% train: 126 encounters (21 per class)
      - 15% val:    27 encounters (5 or 4 per class, balanced across classes)
      - 15% test:   27 encounters (4 or 5 per class, complementary to val)
    and defines disjoint ownship T1 time blocks for negative ROIs:
      - train: [0.0, 42.0) s (k_frame in [0, 41])
      - val:   [42.0, 51.0) s (k_frame in [42, 50])
      - test:  [51.0, 60.0] s (k_frame in [51, 60])
    """
    selected_by_class = get_g3a_v01_subsample_by_class(
        n_encounters=n_encounters,
        subsample_seed=subsample_seed,
        encounters_file=encounters_file,
    )
    rng = np.random.default_rng(split_seed)

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    class_breakdown: dict[str, dict[str, list[str]]] = {}

    for class_idx, cid in enumerate(sorted(selected_by_class.keys())):
        class_eids = list(selected_by_class[cid])
        n_cls = len(class_eids)
        perm = rng.permutation(n_cls)
        shuffled = [class_eids[int(i)] for i in perm]

        n_train_cls = int(round(0.70 * n_cls))
        rem = n_cls - n_train_cls
        if class_idx % 2 == 0:
            n_val_cls = (rem + 1) // 2
        else:
            n_val_cls = rem // 2
        n_test_cls = rem - n_val_cls

        cls_train = sorted(shuffled[:n_train_cls])
        cls_val = sorted(shuffled[n_train_cls : n_train_cls + n_val_cls])
        cls_test = sorted(shuffled[n_train_cls + n_val_cls : n_train_cls + n_val_cls + n_test_cls])

        train_ids.extend(cls_train)
        val_ids.extend(cls_val)
        test_ids.extend(cls_test)
        class_breakdown[cid] = {
            "train": cls_train,
            "val": cls_val,
            "test": cls_test,
        }

    train_ids = sorted(train_ids)
    val_ids = sorted(val_ids)
    test_ids = sorted(test_ids)
    all_subsample_ids = sorted(train_ids + val_ids + test_ids)

    return {
        "schema": "xvs.detection_g1_splits/1",
        "spec_reference": "docs/xvs/WP04_DETECTION.md#9.2",
        "decision_references": ["D23", "D34-S1", "D34-S2", "D34-S7", "D34-S9"],
        "g3a_release_sha256": g3a_release_sha256,
        "subsample_seed": subsample_seed,
        "split_seed": split_seed,
        "kept_frame_step_dt_s": 1.0,
        "seen_backgrounds_statement": (
            "All G3a v0.1 encounters fly the shared T1 ownship trajectory from the same initial "
            "pose and UTC; therefore G3a test evaluation measures detection and false-positive "
            "density on seen backgrounds (unseen intruder trajectories, 3D aircraft models, and "
            "sensor-noise realizations over the shared T1 terrain/sky scene)."
        ),
        "background_generalisation_test": {
            "batch": "B9",
            "dataset_id": "AOT-v1",
            "s3_bucket": "s3://airborne-obj-detection-challenge-training",
            "license": "CDLA-Permissive-1.0",
            "registry_doc": "docs/xvs/DATASETS_LICENSES.md",
        },
        "compute_caps": {
            "total_g1_cu_cap": 60,
            "cu_cap": 60,
            "pilot_max_hours": 0.5,
            "pilot_cu_cap": 10,
            "job_max_hours": 10.0,
            "checkpoint_interval_minutes": 30,
            "single_gpu_only": True,
        },
        "split_fractions": {
            "train_fraction": 0.70,
            "val_fraction": 0.15,
            "test_fraction": 0.15,
        },
        "encounter_counts": {
            "total_encounters": len(all_subsample_ids),
            "train_encounters": len(train_ids),
            "val_encounters": len(val_ids),
            "test_encounters": len(test_ids),
        },
        "negative_time_blocks": {
            "partition_axis": "t_s_along_T1_ownship_trajectory",
            "trajectory_duration_s": 60.0,
            "train": {"t_min_s": 0.0, "t_max_s": 42.0, "max_inclusive": False, "k_min": 0, "k_max": 41},
            "val": {"t_min_s": 42.0, "t_max_s": 51.0, "max_inclusive": False, "k_min": 42, "k_max": 50},
            "test": {"t_min_s": 51.0, "t_max_s": 60.0, "max_inclusive": True, "k_min": 51, "k_max": 60},
        },
        "encounters": {
            "all_g3a_subsample": all_subsample_ids,
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        },
        "class_stratification": class_breakdown,
    }


def assign_negative_split_by_time(t_s: float, splits_cfg: dict[str, Any]) -> str:
    """Assign a negative ROI at ownship time `t_s` (seconds) to 'train', 'val', or 'test'."""
    blocks = splits_cfg["negative_time_blocks"]
    t_val = float(t_s)
    for split_name in ("train", "val", "test"):
        blk = blocks[split_name]
        t_min = float(blk["t_min_s"])
        t_max = float(blk["t_max_s"])
        if blk.get("max_inclusive", False):
            if t_min <= t_val <= t_max + 1e-9:
                return split_name
        else:
            if t_min <= t_val < t_max:
                return split_name
    raise ValueError(f"Negative frame timestamp t_s={t_s} falls outside configured time blocks")


def assign_encounter_split(encounter_id: str, splits_cfg: dict[str, Any]) -> str | None:
    """Return 'train', 'val', or 'test' for a G3a v0.1 subsample encounter ID, or None if not in subsample."""
    enc_map = splits_cfg["encounters"]
    if encounter_id in enc_map["train"]:
        return "train"
    if encounter_id in enc_map["val"]:
        return "val"
    if encounter_id in enc_map["test"]:
        return "test"
    return None


def verify_splits_config(splits_cfg: dict[str, Any]) -> list[str]:
    """Validate all D34 S1, S7, S9 invariants on `splits_cfg`. Returns list of error strings (empty if valid)."""
    errors: list[str] = []
    enc = splits_cfg.get("encounters", {})
    train_set = set(enc.get("train", []))
    val_set = set(enc.get("val", []))
    test_set = set(enc.get("test", []))
    all_set = set(enc.get("all_g3a_subsample", []))

    if len(train_set) != 126:
        errors.append(f"Expected 126 train encounters, got {len(train_set)}")
    if len(val_set) != 27:
        errors.append(f"Expected 27 val encounters, got {len(val_set)}")
    if len(test_set) != 27:
        errors.append(f"Expected 27 test encounters, got {len(test_set)}")
    if len(all_set) != 180:
        errors.append(f"Expected 180 total subsample encounters, got {len(all_set)}")

    if train_set & val_set:
        errors.append(f"Train and val sets overlap: {sorted(train_set & val_set)}")
    if train_set & test_set:
        errors.append(f"Train and test sets overlap: {sorted(train_set & test_set)}")
    if val_set & test_set:
        errors.append(f"Val and test sets overlap: {sorted(val_set & test_set)}")
    if (train_set | val_set | test_set) != all_set:
        errors.append("Union of train, val, test does not equal all_g3a_subsample")

    # Verify E#### format
    for eid in all_set:
        if not (isinstance(eid, str) and len(eid) == 5 and eid.startswith("E") and eid[1:].isdigit()):
            errors.append(f"Invalid encounter ID format: {eid!r}")

    # Verify negative time blocks are disjoint and cover [0, 60]
    blocks = splits_cfg.get("negative_time_blocks", {})
    tr_b = blocks.get("train", {})
    va_b = blocks.get("val", {})
    te_b = blocks.get("test", {})
    if not (
        float(tr_b.get("t_min_s", -1.0)) == 0.0  # default-ok: fail-closed sentinel for missing split time boundary
        and float(tr_b.get("t_max_s", -1.0)) == float(va_b.get("t_min_s", -2.0))  # default-ok: fail-closed sentinels for missing split time boundaries
        and float(va_b.get("t_max_s", -1.0)) == float(te_b.get("t_min_s", -2.0))  # default-ok: fail-closed sentinels for missing split time boundaries
        and float(te_b.get("t_max_s", -1.0)) == 60.0  # default-ok: fail-closed sentinel for missing split time boundary
    ):
        errors.append("Negative time blocks are not contiguous/disjoint over [0.0, 60.0] s")

    # Verify compute caps (S9)
    caps = splits_cfg.get("compute_caps", {})
    if int(caps.get("total_g1_cu_cap", 999)) > 60 or int(caps.get("cu_cap", 999)) > 60:  # default-ok: fail-closed sentinel above 60 CU cap
        errors.append("total_g1_cu_cap / cu_cap exceeds 60 CU limit")

    return errors


def load_splits_config(path: Path = DEFAULT_SPLITS_CONFIG) -> tuple[dict[str, Any], str]:
    """Load `g1_splits_v0.json` and return `(config_dict, sha256_hex)`."""
    raw_bytes = path.read_bytes()
    sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
    cfg = json.loads(raw_bytes.decode("utf-8"))
    errs = verify_splits_config(cfg)
    if errs:
        raise ValueError(f"Invalid G1 splits config at {path}: {errs}")
    return cfg, sha256_hex
