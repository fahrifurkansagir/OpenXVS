"""Centralized OpenXVS v5.0 (Allowlist v2.1 / DR-21) dataset and model-weight license guard.

Implements `SP2_M2_MODELS_DESIGN.md` §3.1 (`V2-M2L-1`) upgraded to Allowlist v2.1 (`DR-21`):
- `assert_dataset_allowed(dataset_id, split_purpose)` enforces hard-banned datasets/simulators,
  eval-only research corpora (zero training!), and verified open-license training datasets.
- `assert_model_weight_allowed(weight_id, allow_non_osi_attribution=False)` enforces OSI-pure
  default weights (`mobilenetv4_hybrid_medium`, `mixvpr_clean`, `xfeat`, `lighterglue`,
  `dinov2_b14`, `boq`, `roma_v1`, etc.) and HARD-BANS copyleft / NC / custom non-OSI weights
  including `ufm_dinov2`, `roma_v2_dinov3`, `dinov3`, `convnextv2`, `salad`, `fastvit_sa12`,
  `c_radiov4`, `mobilegeo`, `mixvpr_upstream`, `megaloc`, and `superpoint`.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Literal

SplitPurpose = Literal["train", "eval"]


def _normalize_id(raw_id: str) -> str:
    return raw_id.strip().lower().replace("-", "_").replace(" ", "_")


ALLOWED_TRAIN_AND_EVAL_DATASETS: dict[str, str] = {
    "aot": "CDLA-Permissive-1.0 (Amazon Airborne Object Tracking)",
    "openxvs_g3a": "Apache-2.0 / U.S. Public Domain (OpenXVS synthetic G3a)",
    "openxvs_g4": "Apache-2.0 / U.S. Public Domain (OpenXVS synthetic G4)",
    "usda_naip": "U.S. Public Domain (USDA NAIP)",
    "faa_nasr": "U.S. Public Domain (FAA NASR)",
    "usgs_3dep": "U.S. Public Domain (USGS 3DEP)",
    "usgs_nhd": "U.S. Public Domain (USGS NHD)",
    "usgs_imagery_only": "U.S. Public Domain (USGS National Map Orthoimagery / HRO)",
    "noaa_geoid18": "U.S. Public Domain (NOAA NGS GEOID18)",
    "fema_usa_structures": (
        "U.S. Public Domain (FEMA/DHS) with CC-BY-4.0 ORNL Attribution "
        "(Oak Ridge National Laboratory USA Structures)"
    ),
    "ourairports": "Public Domain (OurAirports)",
    "lltem_v1": "CC-BY-4.0 (MIT LL LLTEM V1.0)",
    "copernicus_glo30": "ESA/Copernicus Free & Open Attribution (NOT U.S. PD)",
    "copernicus_sentinel2_l2a": (
        "ESA/Copernicus Free & Open Attribution (Sentinel-2 L2A via Planetary Computer STAC; "
        "ESA/EU credit required, NOT U.S. PD)"
    ),
    "alto": "BSD-3-Clause (CMU AirLab ALTO)",
    "eccv2026_ltc": "CC-BY-4.0 (ECCV 2026 LTC)",
    "tartanair": "CC-BY-4.0 (CMU AirLab TartanAir)",
    "irstd1k": "MIT (IRSTD-1k)",
    "hit_uav": "CC-BY-4.0 (HIT-UAV)",
    "det_fly": "MIT (Det-Fly air-to-air micro-UAV dataset, https://github.com/Jake-WU/Det-Fly)",
    "fujii_distant_bird": (
        "MIT (Fujii et al. Distant-Bird-Detection-Dataset-for-Safe-Drone-Flight, "
        "https://github.com/shi-labs/Distant-Bird-Detection-Dataset-for-Safe-Drone-Flight)"
    ),
}

EVAL_ONLY_DATASETS: dict[str, str] = {
    "icehorizon": "Charter §6: IceHorizon is strictly evaluation-only (ZERO training).",
    "lard_real": "Charter §6 / H3: LARD real split is strictly evaluation-only (ZERO training).",
    "lard": "Charter §6 / H3: LARD is strictly evaluation-only (ZERO training).",
    "cart": "SP2 §3.1: CART is strictly evaluation-only (ZERO training).",
    "massmind": "SP2 §3.1: MassMIND is strictly evaluation-only (ZERO training).",
    "uav_cb": "SP2 §3.1: UAV-CB is strictly evaluation-only (ZERO training).",
    "flir_adas_v2": "SP2 §3.1: FLIR ADAS v2 is strictly evaluation-only (ZERO training).",
    "mastr1325": "CC BY-NC-SA 4.0: MaSTr1325 is non-commercial research/eval only (ZERO training).",
}

BANNED_DATASETS: dict[str, str] = {
    "university_1652": "Google Earth / proprietary satellite tiles forbidden by allowlist.",
    "sues_200": "Proprietary satellite tiles / non-cleared terms.",
    "vigor": "Google Maps / Street View tiles forbidden by allowlist.",
    "denseuav": "Google Earth / proprietary satellite tiles forbidden by allowlist.",
    "uav_visloc": "Proprietary satellite tiles / non-cleared terms.",
    "ortholoc": "Proprietary imagery / non-cleared terms.",
    "vpair": "Non-cleared satellite imagery terms and LTC site contamination (DR-21 hard ban).",
    "lard_synthetic": "Contains Google Earth Studio, Bing Maps, ArcGIS, MSFS, and X-Plane renders.",
    "nudt_sirst": "Academic/non-cleared or ambiguous license terms.",
    "rgbt_tiny": "Non-cleared / restrictive license terms.",
    "anti_uav": "Counter-UAS / non-civil scope & restrictive terms.",
    "vedai": "Non-cleared / research-only terms.",
    "pst900": "Non-cleared / restrictive thermal dataset terms.",
    "msfs": "Microsoft Flight Simulator Game Content Usage Rules forbid ML training/commercial reuse.",
    "xplane": "X-Plane render license not cleared for ML training.",
    "aerial_uav_thermal_inferred_unified_kaggle": (
        "Invalid CC0 relicensing of 6 merged datasets and contains weapon classes outside civil scope."
    ),
    "thermal_uav_scc_loc": (
        "Ships forbidden Google Maps satellite imagery and carries academic-only restriction."
    ),
}

ALLOWED_OSI_MODEL_WEIGHTS: dict[str, str] = {
    "dinov2_b14": "Apache-2.0 (Meta DINOv2 ViT-B/14; teacher only in V5.0)",
    "boq": "MIT (Bag-of-Queries visual place recognition; teacher only in V5.0)",
    "mobilenetv4_hybrid_medium": "Apache-2.0 (Google / timm MobileNetV4-Hybrid-Medium IN-12k/1k)",
    "mixvpr_clean": "Apache-2.0 (OpenXVS clean-room MixVPR-Clean FeatureMixer aggregator)",
    "xfeat": "Apache-2.0 (verlab/accelerated_features XFeat Stage-2A local feature extractor)",
    "lighterglue": "Apache-2.0 (verlab/accelerated_features LighterGlue Stage-2B fallback matcher)",
    "roma_v1": "MIT (RoMa v1 dense feature matcher)",
    "dfine_m_coco": "Apache-2.0 (D-FINE-M trained strictly on COCO)",
    "convnext_v1_small": "MIT (ConvNeXt-V1 Small)",
    "xvs_d2_scratch": "Apache-2.0 (OpenXVS XVS-D2 trained from scratch)",
    "xvs_thermal_scratch": "Apache-2.0 (OpenXVS XVS-ThermalNet trained from scratch)",
}

# Retained as an empty mapping for backwards API compatibility; DR-21 moves roma_v2_dinov3 to HARD BAN.
OPT_IN_NON_OSI_MODEL_WEIGHTS: dict[str, str] = {}

BANNED_MODEL_WEIGHTS: dict[str, str] = {
    "ufm_dinov2": "CC-BY-NC-SA-4.0 non-commercial checkpoint weights (DR-21 hard ban).",
    "roma_v2_dinov3": "DINOv3 custom non-OSI license (DR-21 hard ban).",
    "dinov3": "Meta DINOv3 custom non-OSI commercial-restriction license (DR-21 hard ban).",
    "convnextv2": "CC-BY-NC-4.0 non-commercial weights despite MIT code (DR-21 hard ban).",
    "convnext_v2_nc": "CC-BY-NC-4.0 non-commercial weights.",
    "salad": "GPL-3.0 copyleft license contamination (serizba/salad, DR-21 hard ban).",
    "salad_gpl": "GPL copyleft license contamination.",
    "fastvit_sa12": "Apple ASCL non-OSI license without patent grant (DR-21 hard ban).",
    "c_radiov4": "NVIDIA proprietary non-OSI license (DR-21 hard ban).",
    "mobilegeo": "No license file and University-1652 proprietary satellite contamination (DR-21 hard ban).",
    "mixvpr_upstream": (
        "amaralibey/MixVPR ships NO LICENSE FILE; use clean-room Apache-2.0 mixvpr_clean (DR-21 hard ban)."
    ),
    "megaloc": "GPL-3.0 SALAD + GSV-Cities street-view contamination (DR-21 hard ban).",
    "superpoint": "Magic Leap non-commercial research license.",
    "lard_v2_yolo_agpl": "AGPL-3.0 copyleft contamination (Ultralytics / LARD v2 YOLO).",
    "dfine_obj365": "Objects365 pretraining carries non-commercial terms.",
    "rf_detr_xl": "Roboflow PML 1.0 restricted license.",
    "rf_detr_2xl": "Roboflow PML 1.0 restricted license.",
    "mast3r": "CC-BY-NC-SA-4.0 non-commercial license.",
    "dust3r": "CC-BY-NC-SA-4.0 non-commercial license.",
    "vggt": "Non-commercial / restricted research weights.",
    "eloftr": "Restricted / non-OSI license or unverified dependency terms.",
    "matchanything": "Restricted / non-OSI license terms.",
}


def assert_dataset_allowed(dataset_id: str, split_purpose: SplitPurpose) -> str:
    """Validate that `dataset_id` is permitted for `split_purpose` ('train' or 'eval').

    Returns the verified license / policy note string if allowed; raises `PermissionError`
    on any banned dataset, `eval_only` dataset requested for `'train'`, or unknown dataset.
    """
    if split_purpose not in ("train", "eval"):
        raise ValueError(f"split_purpose must be 'train' or 'eval', got {split_purpose!r}")

    key = _normalize_id(dataset_id)
    if key in BANNED_DATASETS:
        raise PermissionError(f"Dataset '{dataset_id}' is HARD-BANNED: {BANNED_DATASETS[key]}")

    if key in EVAL_ONLY_DATASETS:
        if split_purpose == "train":
            raise PermissionError(
                f"Dataset '{dataset_id}' is EVAL-ONLY and forbidden for training: "
                f"{EVAL_ONLY_DATASETS[key]}"
            )
        return EVAL_ONLY_DATASETS[key]

    if key in ALLOWED_TRAIN_AND_EVAL_DATASETS:
        return ALLOWED_TRAIN_AND_EVAL_DATASETS[key]

    raise PermissionError(
        f"Dataset '{dataset_id}' is not on the OpenXVS v2.1 allowlist (fail-closed)."
    )


def assert_model_weight_allowed(
    weight_id: str,
    allow_non_osi_attribution: bool = False,
) -> str:
    """Validate that `weight_id` satisfies OpenXVS v2.1 / DR-21 license policy.

    Returns the verified license string if allowed; raises `PermissionError` on banned weights
    (even when `allow_non_osi_attribution=True`), opt-in non-OSI weights without explicit opt-in,
    or unknown weights (fail-closed).
    """
    key = _normalize_id(weight_id)
    if key in BANNED_MODEL_WEIGHTS:
        raise PermissionError(f"Model weight '{weight_id}' is HARD-BANNED: {BANNED_MODEL_WEIGHTS[key]}")

    if key in OPT_IN_NON_OSI_MODEL_WEIGHTS:
        if not allow_non_osi_attribution:
            raise PermissionError(
                f"Model weight '{weight_id}' requires explicit opt-in "
                f"(allow_non_osi_attribution=True): {OPT_IN_NON_OSI_MODEL_WEIGHTS[key]}"
            )
        return OPT_IN_NON_OSI_MODEL_WEIGHTS[key]

    if key in ALLOWED_OSI_MODEL_WEIGHTS:
        return ALLOWED_OSI_MODEL_WEIGHTS[key]

    raise PermissionError(
        f"Model weight '{weight_id}' is not on the OpenXVS v2.1 weight allowlist (fail-closed)."
    )


def run_strict_policy_audit() -> dict[str, object]:
    """Audit internal consistency of the v2.1 license guard policy (NAV1 `--strict` gate)."""
    violations: list[str] = []

    ds_overlap = (
        (set(ALLOWED_TRAIN_AND_EVAL_DATASETS) & set(BANNED_DATASETS))
        | (set(ALLOWED_TRAIN_AND_EVAL_DATASETS) & set(EVAL_ONLY_DATASETS))
        | (set(EVAL_ONLY_DATASETS) & set(BANNED_DATASETS))
    )
    if ds_overlap:
        violations.append(f"dataset_set_overlap:{sorted(ds_overlap)}")

    wt_overlap = (
        (set(ALLOWED_OSI_MODEL_WEIGHTS) & set(BANNED_MODEL_WEIGHTS))
        | (set(ALLOWED_OSI_MODEL_WEIGHTS) & set(OPT_IN_NON_OSI_MODEL_WEIGHTS))
        | (set(OPT_IN_NON_OSI_MODEL_WEIGHTS) & set(BANNED_MODEL_WEIGHTS))
    )
    if wt_overlap:
        violations.append(f"weight_set_overlap:{sorted(wt_overlap)}")

    required_banned_weights = (
        "ufm_dinov2",
        "roma_v2_dinov3",
        "dinov3",
        "convnextv2",
        "salad",
        "fastvit_sa12",
        "c_radiov4",
        "mobilegeo",
        "mixvpr_upstream",
        "megaloc",
        "superpoint",
    )
    for wid in required_banned_weights:
        if wid not in BANNED_MODEL_WEIGHTS or wid in ALLOWED_OSI_MODEL_WEIGHTS:
            violations.append(f"required_banned_weight_not_blocked:{wid}")

    required_allowed_weights = (
        "mobilenetv4_hybrid_medium",
        "mixvpr_clean",
        "xfeat",
        "lighterglue",
    )
    for wid in required_allowed_weights:
        if wid not in ALLOWED_OSI_MODEL_WEIGHTS:
            violations.append(f"required_allowed_weight_missing:{wid}")

    if "vpair" not in BANNED_DATASETS:
        violations.append("required_banned_dataset_not_blocked:vpair")
    for ds in ("copernicus_sentinel2_l2a", "usgs_imagery_only", "fema_usa_structures"):
        if ds not in ALLOWED_TRAIN_AND_EVAL_DATASETS:
            violations.append(f"required_allowed_dataset_missing:{ds}")

    return {
        "policy_version": "2.1",
        "allowed_datasets_count": len(ALLOWED_TRAIN_AND_EVAL_DATASETS),
        "eval_only_datasets_count": len(EVAL_ONLY_DATASETS),
        "banned_datasets_count": len(BANNED_DATASETS),
        "allowed_osi_weights_count": len(ALLOWED_OSI_MODEL_WEIGHTS),
        "banned_weights_count": len(BANNED_MODEL_WEIGHTS),
        "violations": violations,
        "pass": len(violations) == 0,
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenXVS v2.1 License Guard strict policy audit")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on any policy violation")
    args = parser.parse_args(argv)
    report = run_strict_policy_audit()
    print(json.dumps(report, indent=2))
    if args.strict and not report["pass"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main())
