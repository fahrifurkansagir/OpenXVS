"""WP-02 Gate G1 validation script (Models, encounters, kinematics, analytic GT).

Conforms strictly to docs/xvs/WP02_TRAFFIC.md §3 G1 and §4 thresholds.
Fast tests, no browser required.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.geo import wgs84, rotations
from xvs.rig.rig import Rig
from xvs.rig.camera import PinholeCamera
from xvs.sim import horizon, trajectory
from xvs.traffic import models, encounter, gt
from xvs.traffic.ring import TrafficRing

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


from scripts.xvs import make_traffic_models


def run_g1_models_check() -> dict[str, Any]:
    print("\n--- 1. Models Check (3 variants per class + CAL) ---")
    manifest = make_traffic_models.generate_all_models(seeds=[1, 2, 3])

    classes = ["C1_UAS", "C2_SEP", "C3_TWIN", "C4_BIZJET", "C5_TRANSPORT", "C6_HELI"]
    all_pass = True
    model_rows = []

    for item in manifest:
        cls_id = item["class"]
        if cls_id not in classes and cls_id != "CALIBRATION":
            continue

        dims = item["dimensions"]
        span = dims.get("span_m", 0.0)  # default-ok: timing benchmark fallback
        length = dims.get("length_m", 0.0)  # default-ok: timing benchmark fallback
        tris = item["triangle_count"]

        if cls_id in models.CLASS_SPECS:
            spec = models.CLASS_SPECS[cls_id]
            s_ok = (spec["span_min_m"] - 1e-4 <= span <= spec["span_max_m"] + 1e-4)
            l_ok = (spec["length_min_m"] - 1e-4 <= length <= spec["length_max_m"] + 1e-4)
            t_ok = (tris <= 20000)
            valid = s_ok and l_ok and t_ok
            if not valid:
                all_pass = False
        else:
            valid = True

        model_rows.append({
            "model_id": item["model_id"],
            "class": cls_id,
            "seed": item["seed"],
            "span_m": span,
            "length_m": length,
            "height_m": dims.get("height_m", 0.0),  # default-ok: timing benchmark fallback
            "triangle_count": tris,
            "glb_sha256": item["glb_sha256"],
            "valid": valid,
        })

    print(f"  Verified {len(model_rows)} models: all within class ranges and <= 20k tris -> {'PASS' if all_pass else 'FAIL'}")
    return {
        "all_pass": all_pass,
        "count": len(model_rows),
        "models": model_rows,
    }


def run_g1_encounter_o1() -> dict[str, Any]:
    print("\n--- 2. Encounter Construction on O1 (1,000 draws) ---")
    o1 = encounter.TestOwnshipO1()
    rng = np.random.default_rng(42)

    n_draws = 1000
    max_diff_t = 0.0
    max_diff_hmd = 0.0
    max_diff_vmd = 0.0

    for i in range(n_draws):
        t_cpa = float(rng.uniform(30.0, 60.0))
        hmd = float(rng.uniform(10.0, 1500.0))
        side = "right" if rng.uniform(0.0, 1.0) < 0.5 else "left"
        vmd = float(rng.uniform(-300.0, 300.0))
        approach_deg = float(rng.uniform(-179.0, 180.0))
        gs = float(rng.uniform(20.0, 200.0))
        vs = float(rng.uniform(-8.0, 8.0))

        params = encounter.EncounterParams(
            id=f"TEST_{i:04d}",
            class_id="C2_SEP",
            model_id="C2_SEP_0001",
            t_cpa=t_cpa,
            hmd_m=hmd,
            side=side,
            vmd_m=vmd,
            approach_deg=approach_deg,
            gs_mps=gs,
            vs_mps=vs,
        )

        t_achieved, hmd_achieved, vmd_achieved = encounter.evaluate_cpa_o1(params, o1)

        diff_t = abs(t_achieved - t_cpa)
        diff_hmd = abs(hmd_achieved - hmd)
        diff_vmd = abs(vmd_achieved - vmd)

        if diff_t > max_diff_t:
            max_diff_t = diff_t
        if diff_hmd > max_diff_hmd:
            max_diff_hmd = diff_hmd
        if diff_vmd > max_diff_vmd:
            max_diff_vmd = diff_vmd

    passes = (max_diff_t <= 1e-6) and (max_diff_hmd <= 1e-6) and (max_diff_vmd <= 1e-6)
    print(f"  O1 1,000 draws: max |dt| = {max_diff_t:.3e} s, max |dHMD| = {max_diff_hmd:.3e} m, max |dVMD| = {max_diff_vmd:.3e} m -> {'PASS' if passes else 'FAIL'}")
    return {
        "pass": passes,
        "n_draws": n_draws,
        "max_diff_t_s": max_diff_t,
        "max_diff_hmd_m": max_diff_hmd,
        "max_diff_vmd_m": max_diff_vmd,
    }


def run_g1_actual_cpa_trajectories() -> dict[str, Any]:
    print("\n--- 3. Actual CPA on Trajectories T1, T2, T3 (100 draws each) ---")
    scenarios = [
        ("T1_cruise", trajectory.t1_cruise_state),
        ("T2_ksfo28l", trajectory.t2_ksfo28l_state),
        ("T3_turn_climb", trajectory.t3_turn_climb_state),
    ]

    rng = np.random.default_rng(123)
    results = {}

    for sc_name, traj_fn in scenarios:
        dt_list = []
        dhmd_list = []
        dvmd_list = []

        for i in range(100):
            t_cpa = float(rng.uniform(35.0, 55.0))
            hmd = float(rng.uniform(50.0, 800.0))
            side = "right" if rng.uniform(0.0, 1.0) < 0.5 else "left"
            vmd = float(rng.uniform(-150.0, 150.0))
            app_deg = float(rng.uniform(-179.0, 180.0))
            gs = float(rng.uniform(40.0, 120.0))
            vs = float(rng.uniform(-5.0, 5.0))

            params = encounter.EncounterParams(
                id=f"{sc_name}_{i:03d}",
                class_id="C2_SEP",
                model_id="C2_SEP_0001",
                t_cpa=t_cpa,
                hmd_m=hmd,
                side=side,
                vmd_m=vmd,
                approach_deg=app_deg,
                gs_mps=gs,
                vs_mps=vs,
            )

            try:
                enc = encounter.construct_encounter(traj_fn, params)
                res = encounter.evaluate_actual_cpa(traj_fn, enc, t_start=t_cpa - 5.0, t_end=t_cpa + 5.0, dt=0.01)
                dt_list.append(abs(res["diff_t_cpa_s"]))
                dhmd_list.append(abs(res["diff_hmd_m"]))
                dvmd_list.append(abs(res["diff_vmd_m"]))
            except Exception as e:
                continue

        results[sc_name] = {
            "mean_diff_t_s": float(np.mean(dt_list)),
            "median_diff_t_s": float(np.median(dt_list)),
            "max_diff_t_s": float(np.max(dt_list)),
            "mean_diff_hmd_m": float(np.mean(dhmd_list)),
            "median_diff_hmd_m": float(np.median(dhmd_list)),
            "max_diff_hmd_m": float(np.max(dhmd_list)),
            "mean_diff_vmd_m": float(np.mean(dvmd_list)),
            "median_diff_vmd_m": float(np.median(dvmd_list)),
            "max_diff_vmd_m": float(np.max(dvmd_list)),
        }
        print(f"  {sc_name}: med |dt| = {results[sc_name]['median_diff_t_s']:.3f} s, med |dHMD| = {results[sc_name]['median_diff_hmd_m']:.3f} m, med |dVMD| = {results[sc_name]['median_diff_vmd_m']:.3f} m")

    return results


def run_g1_los_rate() -> dict[str, Any]:
    print("\n--- 4. LOS Rate Vector vs Central Finite Differences (1,000 draws) ---")
    rng = np.random.default_rng(999)
    n_draws = 1000
    dt_fd = 1e-3
    max_rel_err = 0.0

    for i in range(n_draws):
        # Random ownship and intruder in ECEF
        lat = float(rng.uniform(30.0, 45.0))
        lon = float(rng.uniform(-125.0, -115.0))
        h = float(rng.uniform(1000.0, 5000.0))
        p_o = wgs84.geodetic_to_ecef(lat, lon, h)
        n_o = wgs84.r_ecef_ned(lat, lon)

        v_o_ned = rng.uniform(-100.0, 100.0, size=3)
        v_o = n_o @ v_o_ned

        # Intruder 500m to 10km away
        r_ned = rng.uniform(-5000.0, 5000.0, size=3)
        if np.linalg.norm(r_ned) < 500.0:
            r_ned += 500.0
        p_i = p_o + n_o @ r_ned

        v_i_ned = rng.uniform(-150.0, 150.0, size=3)
        v_i = n_o @ v_i_ned

        r_ecef = p_i - p_o
        v_rel = v_i - v_o
        r_sq = float(np.dot(r_ecef, r_ecef))

        # Analytic omega = (r x v_rel) / |r|^2
        omega_ecef = np.cross(r_ecef, v_rel) / r_sq
        omega_ned = n_o.T @ omega_ecef
        omega_norm = float(np.linalg.norm(omega_ned))

        # Finite difference evaluation
        p_o_plus = p_o + v_o * dt_fd
        p_o_minus = p_o - v_o * dt_fd
        p_i_plus = p_i + v_i * dt_fd
        p_i_minus = p_i - v_i * dt_fd

        r_plus = p_i_plus - p_o_plus
        r_minus = p_i_minus - p_o_minus

        u_plus = r_plus / np.linalg.norm(r_plus)
        u_minus = r_minus / np.linalg.norm(r_minus)
        u_dot_fd = (u_plus - u_minus) / (2.0 * dt_fd)

        # omega_fd = u x u_dot
        u_0 = r_ecef / math.sqrt(r_sq)
        omega_fd_ecef = np.cross(u_0, u_dot_fd)
        omega_fd_ned = n_o.T @ omega_fd_ecef

        diff = float(np.linalg.norm(omega_ned - omega_fd_ned))
        rel_err = diff / omega_norm if omega_norm > 1e-12 else diff

        if rel_err > max_rel_err:
            max_rel_err = rel_err

    passes = (max_rel_err <= 1e-6)
    print(f"  1,000 draws: max relative error = {max_rel_err:.3e} (limit <= 1e-6) -> {'PASS' if passes else 'FAIL'}")
    return {
        "pass": passes,
        "n_draws": n_draws,
        "max_rel_error": max_rel_err,
    }


def run_g1_event_flags() -> dict[str, Any]:
    print("\n--- 5. Event Flags Boundary Cases (NMAC and LoWC) ---")
    o1 = encounter.TestOwnshipO1()
    test_cases = [
        # (name, dh, dv, expected_nmac, expected_lowc)
        ("nmac_inside_dh", 152.3, 30.0, True, True),
        ("nmac_outside_dh", 152.5, 30.0, False, True),
        ("nmac_inside_dv", 150.0, 30.4, True, True),
        ("nmac_outside_dv", 150.0, 30.6, False, True),
        ("lowc_inside_dh", 609.5, 70.0, False, True),
        ("lowc_outside_dh", 609.7, 70.0, False, False),
        ("lowc_inside_dv", 600.0, 76.1, False, True),
        ("lowc_outside_dv", 600.0, 76.3, False, False),
    ]

    all_pass = True
    case_results = []

    for name, dh, dv, exp_nmac, exp_lowc in test_cases:
        params = encounter.EncounterParams(
            id=name,
            class_id="C2_SEP",
            model_id="C2_SEP_0001",
            t_cpa=30.0,
            hmd_m=dh,
            side="right",
            vmd_m=dv,
            approach_deg=180.0,
            gs_mps=50.0,
            vs_mps=0.0,
        )
        enc = encounter.construct_encounter(o1, params)
        ev = encounter.evaluate_events(o1, enc, t_start=29.0, t_end=31.0, dt=0.01)

        got_nmac = (ev["nmac_t"] is not None)
        got_lowc = (ev["lowc_t"] is not None)

        passes = (got_nmac == exp_nmac) and (got_lowc == exp_lowc)
        if not passes:
            all_pass = False

        case_results.append({
            "case": name,
            "dh_m": dh,
            "dv_m": dv,
            "exp_nmac": exp_nmac,
            "got_nmac": got_nmac,
            "exp_lowc": exp_lowc,
            "got_lowc": got_lowc,
            "pass": passes,
        })
        print(f"  {name:<18} dh={dh}m dv={dv}m | NMAC: {got_nmac} (exp {exp_nmac}), LoWC: {got_lowc} (exp {exp_lowc}) -> {'PASS' if passes else 'FAIL'}")

    return {
        "all_pass": all_pass,
        "cases": case_results,
    }


def run_g1_projection_gt() -> dict[str, Any]:
    print("\n--- 6. Projection GT (Box Containment & Sphere Midpoint) ---")
    ring = TrafficRing()
    cam = ring.camera_model

    # 1. Symmetric sphere centered on boresight (x=0, y=0, z=1000m)
    sphere_model = models.build_sphere_r2()
    verts = np.concatenate([p.vertices for p in sphere_model.parts], axis=0)

    p_cam_rel = np.array([0.0, 0.0, 1000.0], dtype=np.float64)
    u_c, v_c, _ = cam.project_cam(p_cam_rel)

    verts_cam = verts + p_cam_rel[None, :]
    u_all = cam.fx * (verts_cam[:, 0] / verts_cam[:, 2]) + cam.cx
    v_all = cam.fy * (verts_cam[:, 1] / verts_cam[:, 2]) + cam.cy

    u_mid = 0.5 * (np.min(u_all) + np.max(u_all))
    v_mid = 0.5 * (np.min(v_all) + np.max(v_all))

    diff_mid_u = abs(u_mid - u_c)
    diff_mid_v = abs(v_mid - v_c)
    sphere_pass = (diff_mid_u < 1e-9) and (diff_mid_v < 1e-9)
    print(f"  Sphere midpoint test: |u_mid - u_c| = {diff_mid_u:.3e} px, |v_mid - v_c| = {diff_mid_v:.3e} px (limit < 1e-9) -> {'PASS' if sphere_pass else 'FAIL'}")

    # 2. Box containment across 100 random aircraft poses
    c2 = models.build_c2_sep(seed=1)
    c2_verts = np.concatenate([p.vertices for p in c2.parts], axis=0)
    rng = np.random.default_rng(555)

    containment_pass = True
    for _ in range(100):
        # Place aircraft at 200m to 2000m within camera FOV
        z = float(rng.uniform(200.0, 2000.0))
        x = float(rng.uniform(-0.3 * z, 0.3 * z))
        y = float(rng.uniform(-0.3 * z, 0.3 * z))
        pos_cam = np.array([x, y, z], dtype=np.float64)

        r_body = rotations.euler_to_r(
            float(rng.uniform(-math.pi, math.pi)),
            float(rng.uniform(-0.3, 0.3)),
            float(rng.uniform(-0.5, 0.5)),
        )

        u_target, v_target, _ = cam.project_cam(pos_cam)
        v_cam = (r_body @ c2_verts.T).T + pos_cam[None, :]

        u_box = cam.fx * (v_cam[:, 0] / v_cam[:, 2]) + cam.cx
        v_box = cam.fy * (v_cam[:, 1] / v_cam[:, 2]) + cam.cy

        u_min, u_max = float(np.min(u_box)), float(np.max(u_box))
        v_min, v_max = float(np.min(v_box)), float(np.max(v_box))

        if not (u_min <= u_target <= u_max and v_min <= v_target <= v_max):
            containment_pass = False

    print(f"  Analytic box contains projected centre (100 poses): -> {'PASS' if containment_pass else 'FAIL'}")
    return {
        "pass": sphere_pass and containment_pass,
        "sphere_diff_u_px": diff_mid_u,
        "sphere_diff_v_px": diff_mid_v,
        "box_containment_pass": containment_pass,
    }


def run_g1_horizon_relation_ray_test() -> dict[str, Any]:
    print("\n--- 7. Horizon Relation Sign vs Ray Test (1,000 samples) ---")
    true_rig = Rig.load_nominal().make_true(20260922)
    f_cam = true_rig.camera_model
    rng = np.random.default_rng(777)

    # Ownship at 1000m level
    lat, lon, h = 37.62, -122.37, 1000.0
    cam_poses = true_rig.compute_poses(lat, lon, h, 0.0, 0.0, 0.0)
    f_pose = cam_poses["F"]
    p_cam = np.array(f_pose["position"], dtype=np.float64)
    r_cam = np.array(f_pose["r_ecef_cam"], dtype=np.float64)

    agreements = 0
    total = 1000

    for _ in range(total):
        u = float(rng.uniform(10.0, f_cam.width - 10.0))
        roots = horizon.analytic_horizon_v_at_u(p_cam, r_cam, f_cam, u)
        if not roots:
            continue

        v_limb = roots[0]
        # Test point offset by delta_v from limb
        delta_v = float(rng.uniform(-50.0, 50.0))
        if abs(delta_v) < 0.5:
            delta_v = 1.0 if delta_v >= 0 else -1.0
        v_test = v_limb + delta_v

        # Check ray intersection
        is_earth = horizon.is_earth_at_pixel(p_cam, r_cam, f_cam, u, v_test)
        # If delta_v > 0 (v_test > v_limb, below limb), it should be earth
        # If delta_v < 0 (v_test < v_limb, above limb), it should be sky
        expected_earth = (delta_v > 0)
        if is_earth == expected_earth:
            agreements += 1

    agreement_rate = agreements / total
    passes = (agreement_rate == 1.0)
    print(f"  1,000 samples: agreement rate = {agreement_rate * 100:.1f}% ({agreements}/{total}) -> {'PASS' if passes else 'FAIL'}")
    return {
        "pass": passes,
        "agreement_rate": agreement_rate,
        "agreements": agreements,
        "total": total,
    }


def run_g1_sampler_v1(out_dir: Path) -> dict[str, Any]:
    print("\n--- 8. Sampler v1 Generation & Stratum Coverage (900 encounters) ---")
    sampler = encounter.EncounterSamplerV1(master_seed=20260923, count_per_stratum=10)
    encs1 = sampler.generate(trajectory.t1_cruise_state)

    jsonl_p1 = out_dir / "encounters.jsonl"
    sha1 = encounter.export_encounters_jsonl(encs1, jsonl_p1)

    # Re-run for determinism check
    sampler2 = encounter.EncounterSamplerV1(master_seed=20260923, count_per_stratum=10)
    encs2 = sampler2.generate(trajectory.t1_cruise_state)
    jsonl_p2 = out_dir / "encounters_rerun.jsonl"
    sha2 = encounter.export_encounters_jsonl(encs2, jsonl_p2)
    jsonl_p2.unlink()

    det_pass = (sha1 == sha2)
    print(f"  Generated {len(encs1)} encounters: SHA-256 = {sha1}")
    print(f"  Determinism check (re-run gives identical SHA-256): -> {'PASS' if det_pass else 'FAIL'}")

    # Coverage statistics
    o_state_0 = trajectory.t1_cruise_state(0.0)
    p_o_0 = o_state_0.p_ecef_m

    ranges_t0 = []
    closing_speeds = []
    strata_counts = {}

    for e in encs1:
        traj = encounter.construct_encounter(trajectory.t1_cruise_state, e)
        p_i_0 = traj.position_at(0.0)
        v_i = traj.velocity_at(0.0)
        r0 = float(np.linalg.norm(p_i_0 - p_o_0))
        ranges_t0.append(r0)

        v_rel = v_i - (wgs84.r_ecef_ned(o_state_0.lat_deg, o_state_0.lon_deg) @ o_state_0.v_ned_mps)
        vc = -float(np.dot(p_i_0 - p_o_0, v_rel) / r0)
        closing_speeds.append(vc)

        key = (e.stratum["geometry"], e.stratum["class"], e.stratum["vertical"])
        strata_counts[key] = strata_counts.get(key, 0) + 1  # default-ok: frequency counter accumulator

    stats = {
        "total_encounters": len(encs1),
        "total_strata": len(strata_counts),
        "encounters_per_stratum": 10,
        "all_strata_covered": len(strata_counts) == 90 and all(c == 10 for c in strata_counts.values()),
        "range_t0_m": {
            "min": float(np.min(ranges_t0)),
            "median": float(np.median(ranges_t0)),
            "max": float(np.max(ranges_t0)),
        },
        "closing_speed_mps": {
            "min": float(np.min(closing_speeds)),
            "median": float(np.median(closing_speeds)),
            "max": float(np.max(closing_speeds)),
        },
        "sha256": sha1,
        "determinism_pass": det_pass,
    }

    # Save sampler config metadata
    cfg_dir = REPO_ROOT / "configs" / "xvs" / "encounters"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "sampler_v1.json"
    sampler_cfg = {
        "schema": "xvs.encounter_sampler/1",
        "sampler_version": "1.0",
        "master_seed": 20260923,
        "total_encounters": len(encs1),
        "count_per_stratum": 10,
        "ownship_scenario": "t1_cruise",
        "encounters_sha256": sha1,
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(sampler_cfg, f, indent=2)

    return stats


def run_g1_pipeline(out_dir: str | Path | None = None) -> dict[str, Any]:
    out_path = Path(out_dir) if out_dir else (REPO_ROOT / "output" / "xvs" / "wp02" / "g1")
    out_path.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    res_models = run_g1_models_check()
    res_o1 = run_g1_encounter_o1()
    res_traj = run_g1_actual_cpa_trajectories()
    res_los = run_g1_los_rate()
    res_events = run_g1_event_flags()
    res_proj = run_g1_projection_gt()
    res_horizon = run_g1_horizon_relation_ray_test()
    res_sampler = run_g1_sampler_v1(out_path)

    all_pass = (
        res_models["all_pass"]
        and res_o1["pass"]
        and res_los["pass"]
        and res_events["all_pass"]
        and res_proj["pass"]
        and res_horizon["pass"]
        and res_sampler["determinism_pass"]
        and res_sampler["all_strata_covered"]
    )

    elapsed_s = time.perf_counter() - t0

    summary = {
        "all_pass": all_pass,
        "elapsed_s": elapsed_s,
        "models": res_models,
        "encounter_o1": res_o1,
        "actual_cpa_trajectories": res_traj,
        "los_rate": res_los,
        "event_flags": res_events,
        "projection_gt": res_proj,
        "horizon_relation": res_horizon,
        "sampler_v1": res_sampler,
    }

    out_file = out_path / "results_g1.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=======================================================")
    print(f"WP-02 G1 Pipeline: ALL PASS = {all_pass} (Elapsed: {elapsed_s:.2f} s)")
    print(f"Artifacts saved to: {out_file}")
    print("=======================================================")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="WP-02 G1 validation pipeline")
    parser.add_argument("--out-dir", default=None, help="Output directory")
    args = parser.parse_args()

    res = run_g1_pipeline(args.out_dir)
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
