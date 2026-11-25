"""Gate `V2-M2L-1` verification: `data_allowlist.json` v2.1 & `license_guard.py` (DR-21 upgraded).

Verifies `SP2_M2_MODELS_DESIGN.md` §3.1 + `DR-21` Allowlist v2.1 and emits
`output/xvs/v2_m2/v1_allowlist_evidence.json`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from xvs.sim.allowlist import URLClassifier
from xvs.v2.license_guard import (
    ALLOWED_OSI_MODEL_WEIGHTS,
    ALLOWED_TRAIN_AND_EVAL_DATASETS,
    BANNED_DATASETS,
    BANNED_MODEL_WEIGHTS,
    EVAL_ONLY_DATASETS,
    assert_dataset_allowed,
    assert_model_weight_allowed,
    run_strict_policy_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST_PATH = REPO_ROOT / "docs" / "xvs" / "spec" / "data_allowlist.json"
# Do NOT clobber frozen GSN/M2 evidence under output/xvs/v2_m2/; write v2.1 evidence under output/xvs/v5_nav/
EVIDENCE_PATH = REPO_ROOT / "output" / "xvs" / "v5_nav" / "v1_allowlist_v21_evidence.json"

_EVIDENCE: dict[str, object] = {
    "schema": "xvs.v2_m2.v1_allowlist_evidence/1",
    "verified_utc": "",
    "allowlist_version_prefix": "",
    "v14_entries_preserved_count": 0,
    "v20_new_upstream_entries_count": 0,
    "v21_new_upstream_entries_count": 0,
    "url_classifier_allowed_checks_passed": 0,
    "url_classifier_forbidden_checks_passed": 0,
    "allowed_training_datasets_count": 0,
    "eval_only_datasets_blocked_on_train_count": 0,
    "banned_datasets_blocked_count": 0,
    "allowed_osi_weights_count": 0,
    "banned_weights_blocked_count": 0,
    "roma_v2_opt_in_gated": False,
    "roma_v2_hard_banned": False,
    "ufm_dinov2_hard_banned": False,
    "pass": False,
}


def _persist_evidence() -> None:
    _EVIDENCE["verified_utc"] = datetime.now(timezone.utc).isoformat()
    _EVIDENCE["pass"] = bool(
        _EVIDENCE["allowlist_version_prefix"] == "2.1"
        and int(_EVIDENCE["v14_entries_preserved_count"]) == 5
        and int(_EVIDENCE["v20_new_upstream_entries_count"]) == 5
        and int(_EVIDENCE["v21_new_upstream_entries_count"]) >= 1
        and int(_EVIDENCE["url_classifier_allowed_checks_passed"]) >= 13
        and int(_EVIDENCE["url_classifier_forbidden_checks_passed"]) >= 7
        and int(_EVIDENCE["allowed_training_datasets_count"]) == 5
        and int(_EVIDENCE["eval_only_datasets_blocked_on_train_count"]) >= 6
        and int(_EVIDENCE["banned_datasets_blocked_count"]) >= 17
        and int(_EVIDENCE["allowed_osi_weights_count"]) >= 11
        and int(_EVIDENCE["banned_weights_blocked_count"]) >= 22
        and bool(_EVIDENCE["roma_v2_hard_banned"])
        and bool(_EVIDENCE["ufm_dinov2_hard_banned"])
    )
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(_EVIDENCE, indent=2) + "\n", encoding="utf-8")


def test_allowlist_v21_schema_and_url_classifier_compatibility() -> None:
    doc = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    assert doc["version"].startswith("2.1"), f"Expected allowlist v2.1, got {doc['version']!r}"
    assert "2.0" in doc["version"] and "1.4" in doc["version"]

    upstream_by_id = {entry["id"]: entry for entry in doc["upstream_allowed"]}
    v14_ids = ("usgs_imagery_only", "usgs_3dep", "noaa_geoid18", "ourairports", "lltem_v1")
    for vid in v14_ids:
        assert vid in upstream_by_id, f"Missing preserved v1.4 entry: {vid}"

    v20_ids = ("usda_naip", "faa_nasr", "usgs_nhd", "fema_usa_structures", "copernicus_glo30")
    for vid in v20_ids:
        assert vid in upstream_by_id, f"Missing preserved v2.0 entry: {vid}"
        entry = upstream_by_id[vid]
        assert entry["host"], f"Empty host on {vid}"
        assert entry["path_prefix"], f"Empty path_prefix on {vid}"

    v21_ids = ("copernicus_sentinel2_l2a",)
    for vid in v21_ids:
        assert vid in upstream_by_id, f"Missing new v2.1 entry: {vid}"
        entry = upstream_by_id[vid]
        assert entry["host"] and entry["path_prefix"]

    # Verify Copernicus entries explicitly require ESA/Copernicus Attribution and NOT U.S. PD
    for cop_id in ("copernicus_glo30", "copernicus_sentinel2_l2a"):
        cop_lic = str(upstream_by_id[cop_id]["license"])
        assert "ESA" in cop_lic and "Copernicus" in cop_lic and "NOT U.S. Public Domain" in cop_lic

    # Verify FEMA USA Structures notes ORNL CC-BY-4.0 attribution and never uses *.arcgis.com
    fema_entry = upstream_by_id["fema_usa_structures"]
    assert "ORNL" in str(fema_entry["license"]) and "CC-BY-4.0" in str(fema_entry["license"])
    assert all("arcgis.com" not in h.lower() for h in fema_entry["host"])

    # Verify every allowed host+prefix returns ALLOWED under frozen URLClassifier
    clf = URLClassifier(ALLOWLIST_PATH)
    allowed_urls = [
        ("https://planetarycomputer.microsoft.com/api/stac/v1/collections/naip", "usda_naip"),
        ("https://naipeuwest.blob.core.windows.net/naip/v002/ca/2022/tile.tif", "usda_naip"),
        ("https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a", "copernicus_sentinel2_l2a"),
        ("https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/11/T/K/2024/B04.tif", "copernicus_sentinel2_l2a"),
        ("https://nfdc.faa.gov/webContent/28DaySub/2026-11-20/APT.zip", "faa_nasr"),
        ("https://aeronav.faa.gov/Upload_313-d/cifp/CIFP_261120.zip", "faa_nasr"),
        ("https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/0", "usgs_nhd"),
        ("https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/State/Shape/NHD_H_California.zip", "usgs_nhd"),
        ("https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/USGS_Seamless_DEM_13.vrt", "usgs_nhd"),
        ("https://fema-femadata.s3.amazonaws.com/Partners/ORNL/USA_Structures/California/CA.zip", "fema_usa_structures"),
        ("https://disasters.geoplatform.gov/USA_Structures/CA/structures.gpkg", "fema_usa_structures"),
        ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N37_00_W123_00_DEM/tile.tif", "copernicus_glo30"),
        ("https://opentopography.s3.sdsc.edu/raster/COP30/COP30_hh/tile.tif", "copernicus_glo30"),
    ]
    for url, expected_id in allowed_urls:
        verdict, reason, entry = clf.classify_upstream_url(url)
        assert verdict == "ALLOWED", f"Expected ALLOWED for {url}, got {verdict} ({reason})"
        assert entry is not None and entry["id"] == expected_id

    # Verify all forbidden hosts stay FORBIDDEN (including gis-fema.hub.arcgis.com & naip-usdaonline.hub.arcgis.com!)
    forbidden_urls = [
        "https://gis-fema.hub.arcgis.com/datasets/usa-structures",
        "https://naip-usdaonline.hub.arcgis.com/datasets/naip",
        "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/",
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer",
        "https://tile.googleapis.com/v1/3dtiles/root.json",
        "https://api.cesium.com/v1/assets/1",
        "https://ecn.t0.tiles.virtualearth.net/tiles/a123.jpeg",
        "https://tile.openstreetmap.org/12/654/1583.png",
    ]
    for url in forbidden_urls:
        verdict, reason, _ = clf.classify_upstream_url(url)
        assert verdict == "FORBIDDEN", f"Expected FORBIDDEN for {url}, got {verdict}"
        assert reason is not None

    # Verify dedicated allowed_training_datasets and expanded research_only_datasets
    train_ds_ids = {d["id"]: d["license"] for d in doc["allowed_training_datasets"]}
    assert train_ds_ids == {
        "alto": "BSD-3-Clause",
        "eccv2026_ltc": "CC-BY-4.0",
        "tartanair": "CC-BY-4.0",
        "irstd1k": "MIT",
        "hit_uav": "CC-BY-4.0",
    }
    research_ids = {d["id"] for d in doc["research_only_datasets"]}
    for rid in ("IceHorizon", "LARD", "CART", "MassMIND", "UAV-CB", "FLIR ADAS v2"):
        assert rid in research_ids, f"Missing eval-only research_only_datasets entry: {rid}"

    _EVIDENCE["allowlist_version_prefix"] = "2.1"
    _EVIDENCE["v14_entries_preserved_count"] = len(v14_ids)
    _EVIDENCE["v20_new_upstream_entries_count"] = len(v20_ids)
    _EVIDENCE["v21_new_upstream_entries_count"] = len(v21_ids)
    _EVIDENCE["url_classifier_allowed_checks_passed"] = len(allowed_urls)
    _EVIDENCE["url_classifier_forbidden_checks_passed"] = len(forbidden_urls)
    _EVIDENCE["allowed_training_datasets_count"] = len(train_ds_ids)
    _persist_evidence()


def test_license_guard_datasets_and_model_weights() -> None:
    strict = run_strict_policy_audit()
    assert strict["pass"] is True, f"Strict policy audit violations: {strict['violations']}"

    # Allowed datasets succeed for both train & eval (including copernicus_sentinel2_l2a & usgs_imagery_only)
    assert "copernicus_sentinel2_l2a" in ALLOWED_TRAIN_AND_EVAL_DATASETS
    assert "usgs_imagery_only" in ALLOWED_TRAIN_AND_EVAL_DATASETS
    assert "CC-BY-4.0" in ALLOWED_TRAIN_AND_EVAL_DATASETS["fema_usa_structures"]
    for ds_id in ALLOWED_TRAIN_AND_EVAL_DATASETS:
        assert assert_dataset_allowed(ds_id, "train")
        assert assert_dataset_allowed(ds_id, "eval")

    # Eval-only datasets MUST fail on train and pass on eval
    eval_only_required = ("icehorizon", "lard_real", "cart", "massmind", "uav_cb", "flir_adas_v2")
    for ds_id in eval_only_required:
        assert ds_id in EVAL_ONLY_DATASETS
        assert assert_dataset_allowed(ds_id, "eval")
        with pytest.raises(PermissionError):
            assert_dataset_allowed(ds_id, "train")

    # All 17 hard-banned datasets (including vpair) MUST fail on both train & eval
    banned_required = (
        "university_1652",
        "sues_200",
        "vigor",
        "denseuav",
        "uav_visloc",
        "ortholoc",
        "vpair",
        "lard_synthetic",
        "nudt_sirst",
        "rgbt_tiny",
        "anti_uav",
        "vedai",
        "pst900",
        "msfs",
        "xplane",
        "aerial_uav_thermal_inferred_unified_kaggle",
        "thermal_uav_scc_loc",
    )
    for ds_id in banned_required:
        assert ds_id in BANNED_DATASETS
        with pytest.raises(PermissionError):
            assert_dataset_allowed(ds_id, "train")
        with pytest.raises(PermissionError):
            assert_dataset_allowed(ds_id, "eval")

    # Unknown dataset fails closed
    with pytest.raises(PermissionError):
        assert_dataset_allowed("unverified_random_dataset", "train")

    # Default OSI weights pass (including v5.0 Apache-2.0 MNV4-Hybrid-M, MixVPR-Clean, XFeat, LighterGlue)
    osi_required = (
        "dinov2_b14",
        "boq",
        "mobilenetv4_hybrid_medium",
        "mixvpr_clean",
        "xfeat",
        "lighterglue",
        "roma_v1",
        "dfine_m_coco",
        "convnext_v1_small",
        "xvs_d2_scratch",
        "xvs_thermal_scratch",
    )
    for wid in osi_required:
        assert wid in ALLOWED_OSI_MODEL_WEIGHTS
        assert assert_model_weight_allowed(wid)

    # All 22 hard-banned weights (including DR-21 bans: ufm_dinov2, roma_v2_dinov3, dinov3, convnextv2,
    # salad, fastvit_sa12, c_radiov4, mobilegeo, mixvpr_upstream, megaloc, superpoint) MUST fail even with opt-in!
    banned_weights_required = (
        "ufm_dinov2",
        "roma_v2_dinov3",
        "dinov3",
        "convnextv2",
        "convnext_v2_nc",
        "salad",
        "salad_gpl",
        "fastvit_sa12",
        "c_radiov4",
        "mobilegeo",
        "mixvpr_upstream",
        "megaloc",
        "superpoint",
        "lard_v2_yolo_agpl",
        "dfine_obj365",
        "rf_detr_xl",
        "rf_detr_2xl",
        "mast3r",
        "dust3r",
        "vggt",
        "eloftr",
        "matchanything",
    )
    for wid in banned_weights_required:
        assert wid in BANNED_MODEL_WEIGHTS
        with pytest.raises(PermissionError):
            assert_model_weight_allowed(wid, allow_non_osi_attribution=False)
        with pytest.raises(PermissionError):
            assert_model_weight_allowed(wid, allow_non_osi_attribution=True)

    _EVIDENCE["eval_only_datasets_blocked_on_train_count"] = len(eval_only_required)
    _EVIDENCE["banned_datasets_blocked_count"] = len(banned_required)
    _EVIDENCE["allowed_osi_weights_count"] = len(osi_required)
    _EVIDENCE["banned_weights_blocked_count"] = len(banned_weights_required)
    _EVIDENCE["roma_v2_opt_in_gated"] = True
    _EVIDENCE["roma_v2_hard_banned"] = True
    _EVIDENCE["ufm_dinov2_hard_banned"] = True
    _persist_evidence()
