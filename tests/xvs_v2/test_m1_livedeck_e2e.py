"""Local GPU Playwright E2E Verification Suite for OpenXVS Live Deck v2.0 Milestone M1.

Covers SP1_LIVEDECK_DESIGN.md §5:
  - V2-M1-4: Pass-4 deterministic `--noise-off` parity (< 1e-4 vs Python `_oracles.py`),
             >= 300 consecutive frames with live Stage-1 `D1-KF` ONNX (`enableGraphCapture: false`,
             zero GPU crash/leak, finite outputs).
  - V2-M1-5: Explicit visible fallbacks (WebGPU vs `?forceWebGL2=1` + `ORT-WASM-SIMD`,
             `usgs_3dep_geoid18_ellipsoidal` vs `?terrain=procedural-ci`), `0` external network
             requests under `data_allowlist.json` v1.4 Playwright route guard.
  - V2-M1-6: All 4 emergent Red-Team Missions (`M1`..`M4`, fault-perturbation-only) in BOTH
             WebGPU (Pass A) and forced WebGL2 + ORT-WASM-SIMD (Pass B), saving verified
             1280x720 stills + `e2e_results.json` + `run_manifest.json` + `data_sources.json`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "tests" / "xvs_v2") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tests" / "xvs_v2"))

from _oracles import compute_pass4_reference_tensor, srgb_u8_to_electron_luma
from xvs.core_ffi import XvsCoreFFI, XvsFaultPerturbations
from xvs.sim.allowlist import URLClassifier

DIST_DIR = REPO_ROOT / "apps" / "livedeck" / "dist"
OUT_DIR = REPO_ROOT / "output" / "xvs" / "v2_m1"
STILLS_DIR = OUT_DIR / "stills"
E2E_RESULTS_PATH = OUT_DIR / "e2e_results.json"
RUN_MANIFEST_PATH = OUT_DIR / "run_manifest.json"
DATA_SOURCES_PATH = OUT_DIR / "data_sources.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class _CoopCoepHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "application/javascript",
        ".mjs": "application/javascript",
        ".wasm": "application/wasm",
        ".onnx": "application/octet-stream",
        ".bin": "application/octet-stream",
        ".json": "application/json",
        ".css": "text/css",
        ".png": "image/png",
        ".html": "text/html",
    }

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def copyfile(self, source: Any, outputfile: Any) -> None:
        try:
            super().copyfile(source, outputfile)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            return

    def log_message(self, format: str, *args: Any) -> None:
        return


class _LocalDistServer:
    def __init__(self, dist_dir: Path) -> None:
        handler = partial(_CoopCoepHandler, directory=str(dist_dir))
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = int(self._httpd.server_address[1])
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> _LocalDistServer:
        self._thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5.0)


def _install_allowlist_guard(page: Any, classifier: URLClassifier) -> list[dict[str, str]]:
    disallowed: list[dict[str, str]] = []

    def _on_route(route: Any, request: Any) -> None:
        url = str(request.url)
        verdict, reason = classifier.classify_browser_url(url)
        if verdict == "ALLOWED":
            route.continue_()
        else:
            disallowed.append({"url": url, "verdict": verdict, "reason": str(reason)})
            route.abort("blockedbyclient")

    page.route("**/*", _on_route)
    return disallowed


def _wait_livedeck_ready(page: Any, timeout_ms: int = 45_000) -> dict[str, Any]:
    page.wait_for_function(
        "() => Boolean(window.__OPENXVS_TEST_API__ && window.__OPENXVS_TEST_API__.getStatus().ready && window.__OPENXVS_TEST_API__.getStatus().stage1Ready)",
        timeout=timeout_ms,
    )
    return page.evaluate("() => window.__OPENXVS_TEST_API__.getStatus()")


CHECKPOINT_TIMES_S: tuple[float, ...] = (5.0, 9.0, 15.0, 25.0, 45.0)


def _extract_checkpoint(tel: dict[str, Any]) -> dict[str, Any]:
    return {
        "time_s": round(float(tel["time_s"]), 3),
        "tick_count": int(tel["tick_count"]),
        "simplex_mode": int(tel["simplex_mode"]),
        "healthy_cam_mask": int(tel["healthy_cam_mask"]),
        "active_nav_bank": int(tel["active_nav_bank"]),
        "mandatory_go_around": bool(tel["mandatory_go_around"]),
        "ghosts_rejected": int(tel["ghosts_rejected"]),
        "h_m": float(tel["h_m"]),
        "cusum_stat": float(tel["cusum_stat"]),
        "v_vpl_mm": float(tel["v_vpl_mm"]),
    }


def _run_headless_mission_checkpoints() -> dict[str, dict[str, dict[str, Any]]]:
    """Run headless XvsCoreFFI with exact mission schedules matching apps/livedeck/src/missions/."""
    ffi = XvsCoreFFI()
    sim = ffi.sim_create(20261101)
    checkpoints: dict[str, dict[str, dict[str, Any]]] = {}
    target_ticks = {int(round(t * 10)): f"{t:.1f}s" for t in CHECKPOINT_TIMES_S}

    mission_specs = [
        ("M1", 1, 20261101),
        ("M2", 2, 20261102),
        ("M3", 3, 20261103),
        ("M4", 4, 20261104),
    ]
    try:
        for mid, mission_num, seed in mission_specs:
            ffi.sim_reset_mission(sim, mission_id=mission_num, seed=seed)
            m_ckpts: dict[str, dict[str, Any]] = {}
            for tick in range(1, 451):
                t_prev = float(ffi.sim_snapshot(sim).time_s)
                f = XvsFaultPerturbations.nominal()
                f.ood_global_score = 0.0
                if mid == "M1" and t_prev >= 8.0:
                    f.cam_r_bias_rad = 0.020944
                elif mid == "M2" and t_prev >= 8.0:
                    f.gnss_spoof_vel_ned_mps[1] = 2.5
                elif mid == "M3" and t_prev >= 12.0:
                    f.peer3_active = 1
                elif mid == "M4":
                    f.fog_contrast = 0.22
                    if t_prev >= 10.0:
                        f.lwir_efvs_enabled = 1
                    if t_prev >= 24.0:
                        f.runway_corner_bias_px = 18.0
                ffi.sim_set_faults(sim, f)
                ffi.sim_step(sim, 1)
                if tick in target_ticks:
                    snap = ffi.sim_snapshot(sim)
                    m_ckpts[target_ticks[tick]] = {
                        "time_s": round(float(snap.time_s), 3),
                        "tick_count": int(snap.tick_count),
                        "simplex_mode": int(snap.simplex_mode),
                        "healthy_cam_mask": int(snap.healthy_cam_mask),
                        "active_nav_bank": int(snap.active_nav_bank),
                        "mandatory_go_around": bool(snap.mandatory_go_around),
                        "ghosts_rejected": int(snap.ghosts_rejected),
                        "h_m": float(snap.h_m),
                        "cusum_stat": float(snap.cusum_stat),
                        "v_vpl_mm": float(snap.v_vpl_mm),
                    }
            checkpoints[mid] = m_ckpts
    finally:
        ffi.sim_destroy(sim)
    return checkpoints


def _verify_browser_vs_headless_crosscheck(
    browser_ckpts: dict[str, dict[str, dict[str, Any]]],
    headless_ckpts: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Assert exact discrete & <1e-3 float parity between Browser WASM and Headless XvsCoreFFI."""
    rows: list[dict[str, Any]] = []
    discrete_mismatches = 0
    max_h_m_diff = 0.0
    max_cusum_diff = 0.0
    max_v_vpl_mm_diff = 0.0

    for mid in ("M1", "M2", "M3", "M4"):
        for t_s in CHECKPOINT_TIMES_S:
            key = f"{t_s:.1f}s"
            b = browser_ckpts[mid][key]
            h = headless_ckpts[mid][key]
            for field in (
                "simplex_mode",
                "healthy_cam_mask",
                "active_nav_bank",
                "mandatory_go_around",
                "ghosts_rejected",
            ):
                if b[field] != h[field]:
                    discrete_mismatches += 1
                    raise AssertionError(
                        f"Browser vs Headless discrete mismatch at {mid}@{key} field={field}: "
                        f"browser={b[field]} headless={h[field]}"
                    )
            dh = abs(float(b["h_m"]) - float(h["h_m"]))
            dc = abs(float(b["cusum_stat"]) - float(h["cusum_stat"]))
            dv = abs(float(b["v_vpl_mm"]) - float(h["v_vpl_mm"]))
            max_h_m_diff = max(max_h_m_diff, dh)
            max_cusum_diff = max(max_cusum_diff, dc)
            max_v_vpl_mm_diff = max(max_v_vpl_mm_diff, dv)
            assert dh < 1e-3, f"h_m diff {dh:.3e} >= 1e-3 at {mid}@{key}"
            assert dc < 1e-3, f"cusum_stat diff {dc:.3e} >= 1e-3 at {mid}@{key}"
            assert dv < 1e-3, f"v_vpl_mm diff {dv:.3e} >= 1e-3 at {mid}@{key}"
            rows.append(
                {
                    "mission": mid,
                    "t_s": t_s,
                    "simplex_mode": int(b["simplex_mode"]),
                    "healthy_cam_mask": int(b["healthy_cam_mask"]),
                    "active_nav_bank": int(b["active_nav_bank"]),
                    "mandatory_go_around": bool(b["mandatory_go_around"]),
                    "ghosts_rejected": int(b["ghosts_rejected"]),
                    "h_m_abs_diff": dh,
                    "cusum_abs_diff": dc,
                    "v_vpl_mm_abs_diff": dv,
                }
            )

    return {
        "checkpoints_checked": len(rows),
        "discrete_mismatches": discrete_mismatches,
        "max_h_m_diff": max_h_m_diff,
        "max_cusum_diff": max_cusum_diff,
        "max_v_vpl_mm_diff": max_v_vpl_mm_diff,
        "float_tol": 1e-3,
        "pass": True,
        "rows": rows,
    }


def _run_four_emergent_missions(
    page: Any,
    *,
    capture_prefix: str | None = None,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, dict[str, dict[str, Any]]]]:
    """Run full 45.0 s M1..M4 fault-perturbation schedules via __OPENXVS_TEST_API__ and verify §4.5 outcomes."""
    results: dict[str, Any] = {}
    stills: dict[str, Path] = {}
    browser_ckpts: dict[str, dict[str, dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Mission M1 (45.0 s): Cam-R +1.2 deg at t = 8.0 s -> ISOLATED_5CAM, bit 1 cleared, < 175 urad
    # ------------------------------------------------------------------
    page.evaluate("() => window.__OPENXVS_TEST_API__.startMission('M1')")
    m1_ckpts: dict[str, dict[str, Any]] = {}
    t_5 = page.evaluate("() => window.__OPENXVS_TEST_API__.advanceSimSeconds(5.0)")
    assert int(t_5["simplex_mode"]) == 4, f"M1 premature degradation at 5.0 s: {t_5}"
    m1_ckpts["5.0s"] = _extract_checkpoint(t_5)

    t_9 = page.evaluate("() => window.__OPENXVS_TEST_API__.advanceSimSeconds(4.0)")
    m1_ckpts["9.0s"] = _extract_checkpoint(t_9)

    m1_still_tel = page.evaluate("() => window.__OPENXVS_TEST_API__.advanceSimSeconds(1.0)")
    assert int(m1_still_tel["simplex_mode"]) == 3, f"M1 expected ISOLATED_5CAM(3), got {m1_still_tel}"
    assert (int(m1_still_tel["healthy_cam_mask"]) & 2) == 0, f"M1 Cam-R bit 1 not cleared: {m1_still_tel}"
    assert float(m1_still_tel["fused_attitude_err_urad"]) < 175.0
    if capture_prefix == "webgpu":
        p = STILLS_DIR / f"{capture_prefix}_m1_cam_drift.png"
        page.screenshot(path=str(p))
        stills["M1"] = p

    t_15 = page.evaluate("() => window.__OPENXVS_TEST_API__.advanceSimSeconds(5.0)")
    m1_ckpts["15.0s"] = _extract_checkpoint(t_15)
    t_25 = page.evaluate("() => window.__OPENXVS_TEST_API__.advanceSimSeconds(10.0)")
    m1_ckpts["25.0s"] = _extract_checkpoint(t_25)
    t_45 = page.evaluate("() => window.__OPENXVS_TEST_API__.advanceSimSeconds(20.0)")
    m1_ckpts["45.0s"] = _extract_checkpoint(t_45)
    assert int(t_45["simplex_mode"]) == 3 and (int(t_45["healthy_cam_mask"]) & 2) == 0
    assert float(t_45["fused_attitude_err_urad"]) < 175.0
    browser_ckpts["M1"] = m1_ckpts
    results["M1"] = {
        "time_s": float(t_45["time_s"]),
        "simplex_mode": int(t_45["simplex_mode"]),
        "healthy_cam_mask": int(t_45["healthy_cam_mask"]),
        "active_healthy_cams": int(t_45["active_healthy_cams"]),
        "fused_attitude_err_urad": float(t_45["fused_attitude_err_urad"]),
        "pass": True,
    }

    # ------------------------------------------------------------------
    # Mission M2 (45.0 s): +2.5 m/s East GNSS spoof at t = 8.0 s -> CUSUM > 10.530, Bank 1, FALLBACK_D0_SAFE
    # ------------------------------------------------------------------
    page.evaluate("() => window.__OPENXVS_TEST_API__.startMission('M2')")
    m2_ckpts: dict[str, dict[str, Any]] = {}
    t_5 = page.evaluate("() => window.__OPENXVS_TEST_API__.advanceSimSeconds(5.0)")
    assert int(t_5["simplex_mode"]) == 4, f"M2 premature alarm at 5.0 s: {t_5}"
    m2_ckpts["5.0s"] = _extract_checkpoint(t_5)

    t_9 = page.evaluate("() => window.__OPENXVS_TEST_API__.advanceSimSeconds(4.0)")
    m2_ckpts["9.0s"] = _extract_checkpoint(t_9)

    m2_still_tel = page.evaluate("() => window.__OPENXVS_TEST_API__.advanceSimSeconds(1.5)")
    assert float(m2_still_tel["cusum_stat"]) > 10.530, f"M2 CUSUM did not trip > 10.530: {m2_still_tel}"
    assert int(m2_still_tel["active_nav_bank"]) == 1, f"M2 expected Bank 1 active: {m2_still_tel}"
    assert int(m2_still_tel["simplex_mode"]) == 1, f"M2 expected FALLBACK_D0_SAFE(1): {m2_still_tel}"
    if capture_prefix == "webgpu":
        p = STILLS_DIR / f"{capture_prefix}_m2_gps_spoof.png"
        page.screenshot(path=str(p))
        stills["M2"] = p

    t_15 = page.evaluate("() => window.__OPENXVS_TEST_API__.advanceSimSeconds(4.5)")
    m2_ckpts["15.0s"] = _extract_checkpoint(t_15)
    t_25 = page.evaluate("() => window.__OPENXVS_TEST_API__.advanceSimSeconds(10.0)")
    m2_ckpts["25.0s"] = _extract_checkpoint(t_25)
    t_45 = page.evaluate("() => window.__OPENXVS_TEST_API__.advanceSimSeconds(20.0)")
    m2_ckpts["45.0s"] = _extract_checkpoint(t_45)
    assert float(t_45["cusum_stat"]) > 10.530 and int(t_45["active_nav_bank"]) == 1 and int(t_45["simplex_mode"]) == 1
    browser_ckpts["M2"] = m2_ckpts
    results["M2"] = {
        "time_s": float(t_45["time_s"]),
        "simplex_mode": int(t_45["simplex_mode"]),
        "cusum_stat": float(t_45["cusum_stat"]),
        "cusum_stat_at_10_5s": float(m2_still_tel["cusum_stat"]),
        "active_nav_bank": int(t_45["active_nav_bank"]),
        "pass": True,
    }

    # ------------------------------------------------------------------
    # Mission M3 (Full 45.0 s at 0.1 s resolution): 3-ship MeshDAA -> ghost rejected, 0 3D NMAC, 0 3D LoWC
    # ------------------------------------------------------------------
    page.evaluate("() => window.__OPENXVS_TEST_API__.startMission('M3')")
    m3_ckpts: dict[str, dict[str, Any]] = {}

    def _step_m3_ticks(start_tick: int, end_tick: int) -> dict[str, Any]:
        return page.evaluate(
            """([startTick, endTick]) => {
                let minHoriz = Infinity;
                let minVert = Infinity;
                let minSlant = Infinity;
                let nmacTicks = 0;
                let lowcTicks = 0;
                let maxGhosts = 0;
                let maxIntruders = 0;
                let lastTel = null;
                const ckpts = {};
                const targets = new Set([50, 90, 150, 250, 450]);
                for (let tk = startTick; tk <= endTick; tk++) {
                    const tel = window.__OPENXVS_TEST_API__.advanceSimSeconds(0.1);
                    lastTel = tel;
                    if (tel.ghosts_rejected > maxGhosts) maxGhosts = tel.ghosts_rejected;
                    if (tel.intruder_count > maxIntruders) maxIntruders = tel.intruder_count;
                    for (const rel of tel.intruder_rel_ned) {
                        const dh = Math.hypot(rel[0], rel[1]);
                        const dv = Math.abs(rel[2]);
                        const ds = Math.hypot(rel[0], rel[1], rel[2]);
                        if (dh < minHoriz) minHoriz = dh;
                        if (dv < minVert) minVert = dv;
                        if (ds < minSlant) minSlant = ds;
                        if (dh < 152.4 && dv < 30.48) nmacTicks += 1;
                        if (dh < 1219.2 && dv < 137.16) lowcTicks += 1;
                    }
                    if (targets.has(tk)) {
                        ckpts[(tk / 10).toFixed(1) + 's'] = tel;
                    }
                }
                return { minHoriz, minVert, minSlant, nmacTicks, lowcTicks, maxGhosts, maxIntruders, lastTel, ckpts };
            }""",
            [start_tick, end_tick],
        )

    m3_seg1 = _step_m3_ticks(1, 220)  # 0.0 -> 22.0 s (capture dramatic still at t = 22.0 s)
    for k, v in m3_seg1["ckpts"].items():
        m3_ckpts[k] = _extract_checkpoint(v)
    if capture_prefix == "webgpu":
        p = STILLS_DIR / f"{capture_prefix}_m3_mesh_daa.png"
        page.screenshot(path=str(p))
        stills["M3"] = p

    m3_seg2 = _step_m3_ticks(221, 450)  # 22.0 -> 45.0 s (complete full 45.0 s trajectory)
    for k, v in m3_seg2["ckpts"].items():
        m3_ckpts[k] = _extract_checkpoint(v)

    m3_tel = m3_seg2["lastTel"]
    min_horiz_m = float(min(m3_seg1["minHoriz"], m3_seg2["minHoriz"]))
    min_vert_m = float(min(m3_seg1["minVert"], m3_seg2["minVert"]))
    min_slant_m = float(min(m3_seg1["minSlant"], m3_seg2["minSlant"]))
    nmac_ticks = int(m3_seg1["nmacTicks"]) + int(m3_seg2["nmacTicks"])
    lowc_ticks = int(m3_seg1["lowcTicks"]) + int(m3_seg2["lowcTicks"])
    max_ghosts = int(max(m3_seg1["maxGhosts"], m3_seg2["maxGhosts"]))
    max_intruders = int(max(m3_seg1["maxIntruders"], m3_seg2["maxIntruders"]))

    assert max_ghosts >= 2, f"M3 expected >= 2 ghosts rejected, got {max_ghosts}"
    assert max_intruders >= 3, f"M3 expected >= 3 intruders, got {max_intruders}"
    assert nmac_ticks == 0, f"M3 3D NMAC violation: nmac_ticks={nmac_ticks}"
    assert lowc_ticks == 0, f"M3 3D LoWC violation: lowc_ticks={lowc_ticks}"
    assert min_horiz_m > 152.4, f"M3 horizontal sep violation: {min_horiz_m:.2f} m"

    m3_smoke = page.evaluate("() => window.__OPENXVS_TEST_API__.runOrtSmokeTicks(20)")
    assert m3_smoke["allFinite"] is True
    assert int(m3_smoke["completedInferences"]) >= 2
    browser_ckpts["M3"] = m3_ckpts
    results["M3"] = {
        "time_s": float(m3_tel["time_s"]),
        "simplex_mode": int(m3_tel["simplex_mode"]),
        "ghosts_rejected": max_ghosts,
        "intruder_count": max_intruders,
        "min_horizontal_sep_m": min_horiz_m,
        "min_vertical_sep_m": min_vert_m,
        "min_sep_m": min_slant_m,
        "nmac_ticks": nmac_ticks,
        "lowc_ticks": lowc_ticks,
        "live_d1_inferences_logged": int(m3_smoke["completedInferences"]),
        "live_d1_all_finite": bool(m3_smoke["allFinite"]),
        "domain_shift_note": "Live browser renderer demo; zero Pd claims (numerical parity verified in V2-M1-4).",
        "pass": True,
    }

    # ------------------------------------------------------------------
    # Mission M4 (Full 45.0 s): KSFO 28L 0.22x fog + LWIR EFVS (10s) + +18 px bias (24s) -> Go-Around + Climb-Out
    # ------------------------------------------------------------------
    page.evaluate("() => window.__OPENXVS_TEST_API__.startMission('M4')")
    m4_ckpts: dict[str, dict[str, Any]] = {}
    t_5 = page.evaluate("() => window.__OPENXVS_TEST_API__.advanceSimSeconds(5.0)")
    assert bool(t_5["mandatory_go_around"]) is False
    m4_ckpts["5.0s"] = _extract_checkpoint(t_5)
    t_9 = page.evaluate("() => window.__OPENXVS_TEST_API__.advanceSimSeconds(4.0)")
    assert bool(t_9["mandatory_go_around"]) is False
    m4_ckpts["9.0s"] = _extract_checkpoint(t_9)
    t_15 = page.evaluate("() => window.__OPENXVS_TEST_API__.advanceSimSeconds(6.0)")
    assert bool(t_15["lwir_efvs_enabled"]) is True
    assert bool(t_15["mandatory_go_around"]) is False
    m4_ckpts["15.0s"] = _extract_checkpoint(t_15)

    t_24_1 = page.evaluate("() => window.__OPENXVS_TEST_API__.advanceSimSeconds(9.1)")
    h_m_at_24_1s = float(t_24_1["h_m"])
    assert bool(t_24_1["mandatory_go_around"]) is True, f"M4 Go-Around did not trip by 24.1 s: {t_24_1}"

    t_25 = page.evaluate("() => window.__OPENXVS_TEST_API__.advanceSimSeconds(0.9)")
    m4_ckpts["25.0s"] = _extract_checkpoint(t_25)

    m4_still_tel = page.evaluate("() => window.__OPENXVS_TEST_API__.advanceSimSeconds(1.0)")
    m4_status = page.evaluate("() => window.__OPENXVS_TEST_API__.getStatus()")
    assert bool(m4_still_tel["lwir_efvs_enabled"]) is True, f"M4 EFVS not enabled: {m4_still_tel}"
    assert float(m4_still_tel["v_vpl_mm"]) > 10_000.0, f"M4 V-VPL did not exceed 10,000 mm: {m4_still_tel}"
    assert bool(m4_still_tel["mandatory_go_around"]) is True, f"M4 Go-Around not latched: {m4_still_tel}"
    assert int(m4_still_tel["simplex_mode"]) == 0, f"M4 expected INVALID_INHIBIT(0): {m4_still_tel}"
    assert bool(m4_still_tel["blank_conformal_hud"]) is True
    assert bool(m4_status["conformalBlanked"]) is True
    if capture_prefix:
        fname = (
            f"{capture_prefix}_m4_ksfo_go_around.png"
            if capture_prefix == "webgpu"
            else f"{capture_prefix}_m4_degraded_badges.png"
        )
        p = STILLS_DIR / fname
        page.screenshot(path=str(p))
        stills["M4"] = p

    t_45 = page.evaluate("() => window.__OPENXVS_TEST_API__.advanceSimSeconds(19.0)")
    m4_ckpts["45.0s"] = _extract_checkpoint(t_45)
    h_m_at_45s = float(t_45["h_m"])
    climb_gain_m = h_m_at_45s - h_m_at_24_1s
    assert climb_gain_m > 50.0, (
        f"M4 positive Go-Around climb-out failed: h(45s)={h_m_at_45s:.2f} - "
        f"h(24.1s)={h_m_at_24_1s:.2f} = {climb_gain_m:.2f} m <= 50.0 m"
    )
    assert bool(t_45["mandatory_go_around"]) is True and int(t_45["simplex_mode"]) == 0
    browser_ckpts["M4"] = m4_ckpts
    results["M4"] = {
        "time_s": float(t_45["time_s"]),
        "simplex_mode": int(t_45["simplex_mode"]),
        "lwir_efvs_enabled": bool(t_45["lwir_efvs_enabled"]),
        "v_vpl_mm": float(t_45["v_vpl_mm"]),
        "mandatory_go_around": bool(t_45["mandatory_go_around"]),
        "blank_conformal_hud": bool(t_45["blank_conformal_hud"]),
        "conformal_blanked_renderer": bool(m4_status["conformalBlanked"]),
        "h_m_at_24_1s": h_m_at_24_1s,
        "h_m_at_45s": h_m_at_45s,
        "climb_gain_m": climb_gain_m,
        "pass": True,
    }

    return results, stills, browser_ckpts


def _verify_pass4_parity(page: Any) -> dict[str, Any]:
    """Run deterministic `--noise-off` Pass-4 parity on both structured sky+target and flat zero tiles."""
    rng = np.random.default_rng(20261120)
    yy, xx = np.meshgrid(np.arange(256, dtype=np.float32), np.arange(256, dtype=np.float32), indexing="ij")
    sky = np.clip(95.0 + 45.0 * (yy / 255.0) + rng.normal(0.0, 2.0, size=(256, 256)), 0, 255)
    tgt0 = 90.0 * np.exp(-((xx - 128.0) ** 2 + (yy - 120.0) ** 2) / 3.0)
    tgt1 = 85.0 * np.exp(-((xx - 125.0) ** 2 + (yy - 119.0) ** 2) / 3.0)
    tgt2 = 80.0 * np.exp(-((xx - 122.0) ** 2 + (yy - 118.0) ** 2) / 3.0)

    rgb0 = np.stack([np.clip(sky + tgt0, 0, 255)] * 3, axis=-1).astype(np.uint8)
    rgb1 = np.stack([np.clip(sky + tgt1, 0, 255)] * 3, axis=-1).astype(np.uint8)
    rgb2 = np.stack([np.clip(sky + tgt2, 0, 255)] * 3, axis=-1).astype(np.uint8)

    rgba0 = np.concatenate([rgb0, np.full((256, 256, 1), 255, dtype=np.uint8)], axis=-1)
    e1 = srgb_u8_to_electron_luma(rgb1, full_well_e=10500.0)
    e2 = srgb_u8_to_electron_luma(rgb2, full_well_e=10500.0)

    # Identical poses => exact identity warp (overlap mask M == 1 everywhere)
    pose = {
        "pCamEcef": [-2706174.0, -4260962.0, 3885721.0],
        "rEcefCam": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }
    payload = {
        "currRgba": rgba0.reshape(-1).tolist(),
        "prev1Electrons": e1.reshape(-1).tolist(),
        "prev2Electrons": e2.reshape(-1).tolist(),
        "poseCurr": pose,
        "posePrev1": pose,
        "posePrev2": pose,
        "roiCurr": [1096, 896],
        "roiPrev1": [1096, 896],
        "roiPrev2": [1096, 896],
        "noiseOff": True,
        "applyPsf": False,
        "frameSeed": 1,
    }
    js_out = page.evaluate("(inp) => window.__OPENXVS_TEST_API__.runPass4Parity(inp)", payload)
    gpu_arr = np.asarray(js_out["data"], dtype=np.float32).reshape(1, 5, 256, 256)
    py_ref = compute_pass4_reference_tensor(rgb0, rgb1, rgb2, full_well_e=10500.0)

    assert np.all(np.isfinite(gpu_arr)), "Non-finite values in browser Pass-4 output"
    max_diff_structured = float(np.max(np.abs(gpu_arr - py_ref)))
    assert max_diff_structured < 1e-4, f"Pass-4 parity diff {max_diff_structured:.3e} >= 1e-4"

    # Flat zero-contrast tile check (Review Focus #4: std0 clamped to 1.0 -> exact 0.0, 0 NaN)
    flat_rgba = np.full((256, 256, 4), [128, 128, 128, 255], dtype=np.uint8)
    flat_e = srgb_u8_to_electron_luma(flat_rgba[:, :, :3], full_well_e=10500.0)
    flat_payload = {
        **payload,
        "currRgba": flat_rgba.reshape(-1).tolist(),
        "prev1Electrons": flat_e.reshape(-1).tolist(),
        "prev2Electrons": flat_e.reshape(-1).tolist(),
    }
    js_flat = page.evaluate("(inp) => window.__OPENXVS_TEST_API__.runPass4Parity(inp)", flat_payload)
    flat_arr = np.asarray(js_flat["data"], dtype=np.float32).reshape(1, 5, 256, 256)
    assert np.all(np.isfinite(flat_arr)), "Non-finite values in Pass-4 flat zero-contrast output"
    flat_max_abs = float(np.max(np.abs(flat_arr)))
    assert flat_max_abs < 1e-4, f"Flat zero-contrast tile expected ~0.0, got {flat_max_abs:.3e}"
    assert float(js_flat["stats"]["std0"]) >= 1.0 - 1e-6

    return {
        "compute_path": str(js_out["computePath"]),
        "structured_max_abs_diff": max_diff_structured,
        "flat_zero_contrast_max_abs": flat_max_abs,
        "flat_std0_clamped": float(js_flat["stats"]["std0"]),
        "threshold_lt": 1e-4,
        "pass": True,
    }


def _verify_still_non_blank(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"Missing screenshot: {path}"
    with Image.open(path) as im:
        w, h = im.size
        arr = np.asarray(im.convert("RGB"), dtype=np.float32)
    assert (w, h) == (1280, 720), f"Expected 1280x720 still, got {(w, h)}"
    std_rgb = float(np.std(arr))
    assert std_rgb > 5.0, f"Screenshot {path.name} appears blank (std={std_rgb:.2f})"
    return {
        "filename": path.name,
        "width": w,
        "height": h,
        "bytes": int(path.stat().st_size),
        "rgb_std": std_rgb,
        "sha256": _sha256_file(path),
    }


def _git_short_commit() -> str:
    p = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return p.stdout.strip() if p.returncode == 0 else "unknown"


def _git_dirty_tracked() -> list[str]:
    p = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]


def test_m1_livedeck_webgpu_and_webgl2_e2e() -> None:
    """Full local GTX 1650 SUPER WebGPU (Pass A) + Forced WebGL2 (Pass B) Gate V2-M1 E2E test."""
    assert DIST_DIR.is_dir(), f"Missing built apps/livedeck/dist: {DIST_DIR}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STILLS_DIR.mkdir(parents=True, exist_ok=True)

    classifier = URLClassifier()

    with _LocalDistServer(DIST_DIR) as srv, sync_playwright() as p:
        # ------------------------------------------------------------------
        # Probe default bundled Chromium flags first (SP1_LIVEDECK_DESIGN.md §2.4)
        # ------------------------------------------------------------------
        default_probe_backend = "unknown"
        default_browser = p.chromium.launch(headless=True)
        browser_version = str(default_browser.version)
        try:
            ctx0 = default_browser.new_context(viewport={"width": 1280, "height": 720})
            pg0 = ctx0.new_page()
            _install_allowlist_guard(pg0, classifier)
            pg0.goto(f"{srv.base_url}/?paused=1", wait_until="domcontentloaded")
            pg0.wait_for_function("() => Boolean(window.__OPENXVS_TEST_API__)", timeout=30_000)
            st0 = pg0.evaluate("() => window.__OPENXVS_TEST_API__.getStatus()")
            default_probe_backend = str(st0["backend"])
            ctx0.close()
        finally:
            default_browser.close()

        # Select launch configuration for Pass A WebGPU (SP1_LIVEDECK_DESIGN.md §2.4)
        # Note: Playwright's default headless=True uses `chromium_headless_shell`, which on Windows
        # omits `dxil.dll` needed by Dawn D3D12. Playwright's bundled full Chromium (`channel="chromium"`,
        # NOT system Chrome) ships `dxil.dll` and activates WebGPU + shader-f16 with zero extra flags.
        webgpu_launch_channel: str | None
        webgpu_launch_args: list[str]
        if default_probe_backend == "webgpu":
            webgpu_launch_channel = None
            webgpu_launch_args = []
        else:
            webgpu_launch_channel = "chromium"
            webgpu_launch_args = []

        # ------------------------------------------------------------------
        # PASS A: Local GPU WebGPU Mode
        # ------------------------------------------------------------------
        browser_a = (
            p.chromium.launch(headless=True, channel=webgpu_launch_channel, args=webgpu_launch_args)
            if webgpu_launch_channel
            else p.chromium.launch(headless=True, args=webgpu_launch_args)
        )
        try:
            ctx_a = browser_a.new_context(viewport={"width": 1280, "height": 720})
            page_a = ctx_a.new_page()
            disallowed_a = _install_allowlist_guard(page_a, classifier)
            page_errors_a: list[str] = []
            console_errors_a: list[str] = []
            page_a.on("pageerror", lambda err: page_errors_a.append(str(err)))
            page_a.on(
                "console",
                lambda msg: console_errors_a.append(str(msg.text)) if msg.type == "error" else None,
            )

            page_a.goto(f"{srv.base_url}/?paused=1&noiseOff=1", wait_until="domcontentloaded")
            status_a = _wait_livedeck_ready(page_a, timeout_ms=45_000)

            assert status_a["backend"] == "webgpu", (
                f"Pass A failed to activate WebGPU (got {status_a['backend']}, "
                f"degraded={status_a['degradedReasons']})"
            )
            assert status_a["aiBadge"] in (
                "AI: ORT-WebGPU-FP16",
                "AI: ORT-WebGPU-FP32 (NO F16)",
            ), f"Unexpected Pass A aiBadge: {status_a['aiBadge']}"
            assert status_a["terrainMode"] == "usgs_3dep_geoid18_ellipsoidal"
            assert status_a["lastGpuError"] is None

            # Query WebGPU adapter info & WebGL vendor/renderer for manifest
            gpu_info_a = page_a.evaluate(
                """async () => {
                    const out = { webgpuAdapter: null };
                    if (navigator.gpu) {
                        const ad = await navigator.gpu.requestAdapter();
                        if (ad && ad.info) {
                            out.webgpuAdapter = {
                                vendor: ad.info.vendor || "",
                                architecture: ad.info.architecture || "",
                                device: ad.info.device || "",
                                description: ad.info.description || "",
                                hasShaderF16: ad.features.has("shader-f16"),
                            };
                        }
                    }
                    return out;
                }"""
            )

            # V2-M1-4A: Pass-4 deterministic --noise-off parity (< 1e-4 vs Python oracle)
            pass4_parity = _verify_pass4_parity(page_a)

            # V2-M1-4B: >= 300 consecutive frames with live Stage-1 D1-KF ONNX (enableGraphCapture=false)
            smoke_a = page_a.evaluate("() => window.__OPENXVS_TEST_API__.runOrtSmokeTicks(300)")
            assert int(smoke_a["count"]) == 300
            assert int(smoke_a["completedInferences"]) >= 30
            assert bool(smoke_a["allFinite"]) is True
            assert smoke_a["lastGpuError"] is None

            # V2-M1-6A: All 4 emergent Red-Team Missions (full 45.0 s) + 1280x720 screenshots
            missions_a, stills_a, ckpts_a = _run_four_emergent_missions(
                page_a, capture_prefix="webgpu"
            )
            proof_m4_a = page_a.evaluate("() => window.__OPENXVS_TEST_API__.getMissionProof()")

            # V2-M1-5: Explicit ?terrain=procedural-ci visible badge check (ZERO silent fallback)
            page_proc = ctx_a.new_page()
            disallowed_proc = _install_allowlist_guard(page_proc, classifier)
            page_proc.goto(f"{srv.base_url}/?terrain=procedural-ci&paused=1", wait_until="domcontentloaded")
            st_proc = _wait_livedeck_ready(page_proc, timeout_ms=45_000)
            assert st_proc["terrainMode"] == "procedural_ci"
            terrain_badge_txt = page_proc.locator("[data-testid='terrain-badge']").inner_text()
            degraded_banner_vis = page_proc.locator("[data-testid='degraded-banner']").is_visible()
            assert "PROCEDURAL-CI" in terrain_badge_txt
            assert degraded_banner_vis is True
            assert len(disallowed_proc) == 0
            page_proc.close()

            assert len(disallowed_a) == 0, f"Disallowed external requests in Pass A: {disallowed_a}"
            assert len(page_errors_a) == 0, f"Uncaught page errors in Pass A: {page_errors_a}"
            ctx_a.close()
        finally:
            browser_a.close()

        # ------------------------------------------------------------------
        # PASS B: Forced WebGL2 + ORT-WASM-SIMD Mode (`?forceWebGL2=1`)
        # ------------------------------------------------------------------
        browser_b = p.chromium.launch(headless=True)
        try:
            ctx_b = browser_b.new_context(viewport={"width": 1280, "height": 720})
            page_b = ctx_b.new_page()
            disallowed_b = _install_allowlist_guard(page_b, classifier)
            page_errors_b: list[str] = []
            page_b.on("pageerror", lambda err: page_errors_b.append(str(err)))

            page_b.goto(f"{srv.base_url}/?forceWebGL2=1&paused=1", wait_until="domcontentloaded")
            status_b = _wait_livedeck_ready(page_b, timeout_ms=45_000)

            assert status_b["backend"] == "webgl2"
            assert status_b["aiBadge"] == "AI: ORT-WASM-SIMD"
            assert status_b["lastGpuError"] is None

            render_badge_b = page_b.locator("[data-testid='render-badge']").inner_text()
            ai_badge_b = page_b.locator("[data-testid='ai-badge']").inner_text()
            banner_vis_b = page_b.locator("[data-testid='degraded-banner']").is_visible()
            assert "WebGL2" in render_badge_b
            assert "ORT-WASM-SIMD" in ai_badge_b
            assert banner_vis_b is True

            gl_info_b = page_b.evaluate(
                """() => {
                    const c = document.createElement("canvas");
                    const gl = c.getContext("webgl2");
                    if (!gl) return null;
                    const ext = gl.getExtension("WEBGL_debug_renderer_info");
                    return {
                        vendor: ext ? gl.getParameter(ext.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR),
                        renderer: ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
                    };
                }"""
            )

            smoke_b = page_b.evaluate("() => window.__OPENXVS_TEST_API__.runOrtSmokeTicks(30)")
            assert int(smoke_b["completedInferences"]) >= 3
            assert bool(smoke_b["allFinite"]) is True
            assert smoke_b["lastGpuError"] is None

            missions_b, stills_b, ckpts_b = _run_four_emergent_missions(
                page_b, capture_prefix="webgl2"
            )

            # Keep only webgl2_m4_degraded_badges.png + all 4 webgpu stills as canonical still set
            for key in ("M1", "M2", "M3"):
                if key in stills_b and stills_b[key].is_file():
                    pass  # Zero-deletion rule F6: never delete files; just record canonical set

            assert len(disallowed_b) == 0, f"Disallowed external requests in Pass B: {disallowed_b}"
            assert len(page_errors_b) == 0, f"Uncaught page errors in Pass B: {page_errors_b}"
            ctx_b.close()
        finally:
            browser_b.close()

    # Explicit Browser WASM vs Headless XvsCoreFFI Cross-Check across M1..M4 at matched timestamps
    headless_ckpts = _run_headless_mission_checkpoints()
    crosscheck_a = _verify_browser_vs_headless_crosscheck(ckpts_a, headless_ckpts)
    crosscheck_b = _verify_browser_vs_headless_crosscheck(ckpts_b, headless_ckpts)

    # Verify canonical 5 stills are non-blank 1280x720 and distinct
    canonical_stills = [
        STILLS_DIR / "webgpu_m1_cam_drift.png",
        STILLS_DIR / "webgpu_m2_gps_spoof.png",
        STILLS_DIR / "webgpu_m3_mesh_daa.png",
        STILLS_DIR / "webgpu_m4_ksfo_go_around.png",
        STILLS_DIR / "webgl2_m4_degraded_badges.png",
    ]
    stills_meta = [_verify_still_non_blank(sp) for sp in canonical_stills]
    still_hashes = [m["sha256"] for m in stills_meta]
    assert len(set(still_hashes)) == len(canonical_stills), "Expected 5 distinct cockpit screenshots"

    # Write output/xvs/v2_m1/e2e_results.json
    e2e_payload = {
        "schema": "xvs.v2_m1.livedeck_e2e_results/1",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "allowlist_version": str(classifier.policy["version"]),
        "external_network_requests_count": 0,
        "default_chromium_flags_backend": default_probe_backend,
        "webgpu_launch_channel": webgpu_launch_channel or "default-headless-shell",
        "webgpu_launch_args": webgpu_launch_args,
        "pass_a_webgpu": {
            "backend": status_a["backend"],
            "has_shader_f16": bool(status_a["hasShaderF16"]),
            "ai_badge": status_a["aiBadge"],
            "ai_cache_source": status_a["aiCacheSource"],
            "terrain_mode": status_a["terrainMode"],
            "adapter": gpu_info_a["webgpuAdapter"],
            "pass4_parity": pass4_parity,
            "ort_300_frame_smoke": {
                "frames": int(smoke_a["count"]),
                "completed_inferences": int(smoke_a["completedInferences"]),
                "all_finite": bool(smoke_a["allFinite"]),
                "avg_inference_ms": float(smoke_a["avgInferenceMs"]),
                "max_heatmap_score": float(smoke_a["maxHeatmapScore"]),
                "last_gpu_error": smoke_a["lastGpuError"],
            },
            "missions": missions_a,
            "browser_vs_headless_crosscheck": crosscheck_a,
            "procedural_ci_badge_verified": True,
            "mission_m4_proof_transitions": proof_m4_a["simplex_transitions"],
        },
        "pass_b_forced_webgl2": {
            "backend": status_b["backend"],
            "ai_badge": status_b["aiBadge"],
            "render_badge_visible": render_badge_b,
            "ai_badge_visible": ai_badge_b,
            "degraded_banner_visible": bool(banner_vis_b),
            "webgl_info": gl_info_b,
            "ort_wasm_simd_smoke": {
                "frames": int(smoke_b["count"]),
                "completed_inferences": int(smoke_b["completedInferences"]),
                "all_finite": bool(smoke_b["allFinite"]),
                "avg_inference_ms": float(smoke_b["avgInferenceMs"]),
            },
            "missions": missions_b,
            "browser_vs_headless_crosscheck": crosscheck_b,
        },
        "stills": stills_meta,
        "all_checks_passed": True,
    }
    E2E_RESULTS_PATH.write_text(json.dumps(e2e_payload, indent=2) + "\n", encoding="utf-8")

    # Write output/xvs/v2_m1/run_manifest.json (CONVENTIONS.md §9)
    artifact_paths = [
        REPO_ROOT / "apps" / "livedeck" / "public" / "models" / "d1_kf_k3_fp32.onnx",
        REPO_ROOT / "apps" / "livedeck" / "public" / "models" / "d1_kf_k3_fp16.onnx",
        REPO_ROOT / "apps" / "livedeck" / "public" / "models" / "registry.json",
        REPO_ROOT / "apps" / "livedeck" / "public" / "assets" / "ksfo_3dep_geoid18_v1.bin",
        REPO_ROOT / "apps" / "livedeck" / "public" / "assets" / "b612_mono_msdf.png",
        REPO_ROOT / "apps" / "livedeck" / "src" / "wasm" / "xvs_wasm_bg.wasm",
        OUT_DIR / "kani_proofs.log",
        OUT_DIR / "d1_onnx_parity.json",
        OUT_DIR / "v1_cargo_evidence.json",
        OUT_DIR / "v5_bundle_budget.json",
        OUT_DIR / "v6_bridge_evidence.json",
        E2E_RESULTS_PATH,
        *canonical_stills,
    ]
    artifacts_sha256 = {
        str(p.relative_to(REPO_ROOT).as_posix()): _sha256_file(p)
        for p in artifact_paths
        if p.is_file()
    }
    dirty_tracked = _git_dirty_tracked()
    manifest_payload = {
        "schema": "xvs.v2_m1.run_manifest/1",
        "milestone": "V2-M1",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_short_commit(),
        "git_dirty_tracked_files": dirty_tracked,
        "browser": {
            "engine": "playwright-chromium",
            "version": browser_version,
            "default_flags_webgpu_backend": default_probe_backend,
            "webgpu_launch_channel": webgpu_launch_channel or "default-headless-shell",
            "webgpu_launch_args": webgpu_launch_args,
            "webgpu_adapter": gpu_info_a["webgpuAdapter"],
            "webgl2_info": gl_info_b,
        },
        "artifacts_sha256": artifacts_sha256,
    }
    RUN_MANIFEST_PATH.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")

    # Write output/xvs/v2_m1/data_sources.json (CONVENTIONS.md §9)
    ksfo_src_path = OUT_DIR / "ksfo_3dep_data_sources.json"
    ksfo_sources = (
        json.loads(ksfo_src_path.read_text(encoding="utf-8"))["sources"]
        if ksfo_src_path.is_file()
        else []
    )
    data_sources_payload = {
        "schema": "xvs.v2_m1.data_sources/1",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "allowlist_version": str(classifier.policy["version"]),
        "browser_external_requests_count": 0,
        "upstream_allowlisted_sources": ksfo_sources,
        "local_runtime_assets": [
            {
                "path": "apps/livedeck/public/models/d1_kf_k3_fp32.onnx",
                "sha256": artifacts_sha256["apps/livedeck/public/models/d1_kf_k3_fp32.onnx"],
            },
            {
                "path": "apps/livedeck/public/models/d1_kf_k3_fp16.onnx",
                "sha256": artifacts_sha256["apps/livedeck/public/models/d1_kf_k3_fp16.onnx"],
            },
            {
                "path": "apps/livedeck/public/assets/ksfo_3dep_geoid18_v1.bin",
                "sha256": artifacts_sha256["apps/livedeck/public/assets/ksfo_3dep_geoid18_v1.bin"],
            },
            {
                "path": "apps/livedeck/public/assets/b612_mono_msdf.png",
                "sha256": artifacts_sha256["apps/livedeck/public/assets/b612_mono_msdf.png"],
            },
        ],
    }
    DATA_SOURCES_PATH.write_text(json.dumps(data_sources_payload, indent=2) + "\n", encoding="utf-8")
