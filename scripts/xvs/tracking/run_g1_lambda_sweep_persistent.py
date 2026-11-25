"""WP-05 G1 / Batch B7 — Persistent Terrain Clutter Lambda Sweep under Decisions D29, D32, D33.

Evaluates the WP-05 G1 tracker (with 6-DoF WGS-84 ground-clutter rejection at 3-of-5 initiation,
sliding-window second-moment TTC, and Passive Well-Clear alert rule WC1) across both Decision D33
persistent 3D WGS-84 terrain clutter models (`primary_1hz_hold` and `sensitivity_10hz_geom`),
parameterized strictly from output/xvs/wp04/g0/d0_clutter_persistence.json.

Also synthesizes the conservative overall WP-04 G1 false-alarm requirement:
  lambda_req_G1 = min(lambda_req_indep, lambda_req_persist_1hz, lambda_req_persist_10hz)
and writes both:
  - output/xvs/wp05/g1/g1_target_lambda_req.json
  - configs/xvs/detection/g1_target_lambda_req.json
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xvs.sim.trajectory import t1_cruise_state
import xvs.tracking.evaluate as eval_mod
from xvs.tracking.evaluate import (
    get_ownship_states,
    load_or_compute_truth_cache,
)
from xvs.tracking.measurements import DetectionRecord
from xvs.traffic.encounter import EncounterSamplerV1, EncounterTrajectory
from scripts.xvs.tracking.run_lambda_sweep_persistent_clutter import (
    D33_MODELS,
    DEFAULT_PERSISTENT_LAMBDAS,
    PERSISTENCE_JSON_PATH,
    PersistentClutterMeasurementGenerator,
    load_persistence_parameters,
)
import scripts.xvs.tracking.run_g1_lambda_sweep_v2 as g1_sweep_mod
from scripts.xvs.tracking.run_g1_lambda_sweep_v2 import (
    EXPECTED_SAMPLER_SHA256,
    FULL_SEEDS,
    G1_CONFIG_PATH,
    compute_code_hashes,
    get_git_commit,
    get_git_status_porcelain,
    get_safe_workers,
)

G0_PERSISTENT_JSON_PATH = REPO_ROOT / "output" / "xvs" / "wp05" / "g0_lambda_sweep_v2" / "lambda_sweep_persistent_clutter.json"
G0_PERSISTENT_PER_RUN_PATH = REPO_ROOT / "output" / "xvs" / "wp05" / "g0_lambda_sweep_v2" / "per_run_records_persistent.jsonl"
G0_INDEP_JSON_PATH = REPO_ROOT / "output" / "xvs" / "wp05" / "g0_lambda_sweep_v2" / "lambda_sweep_v2_results.json"


class G1PersistentClutterMeasurementGenerator(PersistentClutterMeasurementGenerator):
    """Extends PersistentClutterMeasurementGenerator so false alarms populate s_mom_px."""

    def generate_frame(
        self,
        t_s: float,
        ownship_state: Any,
        encounters: Sequence[EncounterTrajectory],
        disable_noise: bool = False,
        pd_override: float | None = None,
    ) -> tuple[list[DetectionRecord], dict[str, Any]]:
        dets, metadata = super().generate_frame(
            t_s=t_s,
            ownship_state=ownship_state,
            encounters=encounters,
            disable_noise=disable_noise,
            pd_override=pd_override,
        )
        for d in dets:
            if d.is_false_alarm and d.s_mom_px is None:
                d.s_mom_px = d.s_px
        return dets, metadata


def _init_g1_persistent_worker() -> None:
    eval_mod.SyntheticMeasurementGenerator = G1PersistentClutterMeasurementGenerator
    get_ownship_states()


def load_g0_persistent_per_run_index(
    g0_jsonl_path: Path = G0_PERSISTENT_PER_RUN_PATH,
    g0_json_path: Path = G0_PERSISTENT_JSON_PATH,
) -> tuple[dict[str, dict[tuple[float, int, str], dict[str, Any]]], str | None]:
    """Load G0 persistent per-run records indexed by model -> (lambda, seed, encounter_id)."""
    if not g0_jsonl_path.is_file():
        return {}, None
    raw_bytes = g0_jsonl_path.read_bytes()
    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    if g0_json_path.is_file():
        g0_data = json.loads(g0_json_path.read_text(encoding="utf-8"))
        expected_sha = g0_data.get("per_run_records_sha256")
        if expected_sha and actual_sha != expected_sha:
            raise ValueError(
                f"G0 per_run_records_persistent.jsonl SHA-256 mismatch: {actual_sha} != {expected_sha}"
            )
    by_model: dict[str, dict[tuple[float, int, str], dict[str, Any]]] = {}
    for line in raw_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        m_name = str(row.get("clutter_model", "primary_1hz_hold"))
        key = (float(row["lambda"]), int(row["seed"]), str(row["encounter"]))
        by_model.setdefault(m_name, {})[key] = row
    return by_model, actual_sha


def _numeric_lambda_req(val: float | str) -> float:
    """Convert lambda_req value (float or '< 0.0005') to numeric float for min() comparison."""
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if s.startswith("<"):
        return 0.0
    if s.startswith(">="):
        return float(s.replace(">=", "").strip())
    return float(s)


def build_g1_target_lambda_req_payload(
    indep_sweep: dict[str, Any] | None,
    indep_sweep_path: Path,
    persist_sweep: dict[str, Any],
    persist_sweep_path: Path,
    git_commit: str,
) -> dict[str, Any]:
    """Build g1_target_lambda_req.json payload conforming to B7 §5.3 and config denylist rules."""
    m_1hz = persist_sweep["models"].get("primary_1hz_hold", persist_sweep)
    m_10hz = persist_sweep["models"].get("sensitivity_10hz_geom", m_1hz)

    lam_1hz_val = m_1hz["lambda_req"]
    lam_1hz_str = str(m_1hz["lambda_req_statement"])
    lam_10hz_val = m_10hz["lambda_req"]
    lam_10hz_str = str(m_10hz["lambda_req_statement"])

    if indep_sweep is not None:
        lam_indep_val = indep_sweep["lambda_req"]
        lam_indep_str = str(indep_sweep["lambda_req_statement"])
        indep_sha = hashlib.sha256(indep_sweep_path.read_bytes()).hexdigest() if indep_sweep_path.is_file() else ""
    else:
        lam_indep_val = lam_1hz_val
        lam_indep_str = lam_1hz_str
        indep_sha = ""

    num_candidates = [
        (_numeric_lambda_req(lam_indep_val), lam_indep_val, lam_indep_str, "frame_independent_poisson"),
        (_numeric_lambda_req(lam_1hz_val), lam_1hz_val, lam_1hz_str, "primary_1hz_hold"),
        (_numeric_lambda_req(lam_10hz_val), lam_10hz_val, lam_10hz_str, "sensitivity_10hz_geom"),
    ]
    min_num, min_val, min_str, limiting_regime = min(num_candidates, key=lambda x: x[0])

    g0_indep_val: float | str = 0.05
    g0_1hz_val: float | str = "< 0.0005"
    g0_10hz_val: float | str = "< 0.0005"
    if G0_INDEP_JSON_PATH.is_file():
        g0_indep_data = json.loads(G0_INDEP_JSON_PATH.read_text(encoding="utf-8"))
        g0_indep_val = g0_indep_data["lambda_req"]
    if G0_PERSISTENT_JSON_PATH.is_file():
        g0_pers_data = json.loads(G0_PERSISTENT_JSON_PATH.read_text(encoding="utf-8"))
        g0_1hz_val = g0_pers_data["models"]["primary_1hz_hold"]["lambda_req"]
        g0_10hz_val = g0_pers_data["models"]["sensitivity_10hz_geom"]["lambda_req"]

    persist_sha = hashlib.sha256(persist_sweep_path.read_bytes()).hexdigest() if persist_sweep_path.is_file() else ""
    g1_cfg_sha = hashlib.sha256(G1_CONFIG_PATH.read_bytes()).hexdigest() if G1_CONFIG_PATH.is_file() else ""

    return {
        "schema": "xvs.detection.g1_target_lambda_req/1",
        "batch": "ARGON_B7",
        "wp": "WP-05_G1_TO_WP-04_G1",
        "description": "WP-05 G1 re-derived false-alarm density requirement lambda_req under Decisions D29, D32, D33 across frame-independent and persistent 3D WGS-84 terrain clutter",
        "lambda_req_g1": min_val,
        "lambda_req_g1_numeric": float(min_num),
        "lambda_req_g1_statement": min_str,
        "limiting_regime": limiting_regime,
        "lambda_req_indep": lam_indep_val,
        "lambda_req_indep_numeric": float(_numeric_lambda_req(lam_indep_val)),
        "lambda_req_indep_statement": lam_indep_str,
        "lambda_req_persist_1hz": lam_1hz_val,
        "lambda_req_persist_1hz_numeric": float(_numeric_lambda_req(lam_1hz_val)),
        "lambda_req_persist_1hz_statement": lam_1hz_str,
        "lambda_req_persist_10hz": lam_10hz_val,
        "lambda_req_persist_10hz_numeric": float(_numeric_lambda_req(lam_10hz_val)),
        "lambda_req_persist_10hz_statement": lam_10hz_str,
        "g0_lambda_req_indep": g0_indep_val,
        "g0_lambda_req_persist_1hz": g0_1hz_val,
        "g0_lambda_req_persist_10hz": g0_10hz_val,
        "wp04_g1_suppression_target_factor": 1000.0,
        "provenance": {
            "git_commit": git_commit,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "indep_sweep_json": "output/xvs/wp05/g1_lambda_sweep_v2/lambda_sweep_v2_results.json",
            "indep_sweep_sha256": indep_sha,
            "persistent_sweep_json": "output/xvs/wp05/g1_lambda_sweep_v2/lambda_sweep_persistent_clutter.json",
            "persistent_sweep_sha256": persist_sha,
            "tracking_config_json": "configs/xvs/tracking/g1.json",
            "tracking_config_sha256": g1_cfg_sha,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WP-05 G1 Persistent Terrain Clutter Lambda Sweep (D29, D32, D33)")
    parser.add_argument("--encounters", type=int, default=900)
    parser.add_argument("--lambdas", type=float, nargs="+", default=list(DEFAULT_PERSISTENT_LAMBDAS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(FULL_SEEDS))
    parser.add_argument("--models", type=str, nargs="+", default=list(D33_MODELS))
    parser.add_argument("--persistence-json", type=Path, default=PERSISTENCE_JSON_PATH)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=get_safe_workers())
    parser.add_argument("--skip-target-config", action="store_true", help="Skip writing configs/xvs/detection/g1_target_lambda_req.json")
    args = parser.parse_args(argv)

    safe_limit = get_safe_workers()
    if args.workers > safe_limit:
        args.workers = safe_limit

    git_status_start = get_git_status_porcelain()
    if args.encounters == 900 and git_status_start:
        raise RuntimeError(f"Refusing to run full G1 persistent-clutter sweep on dirty XVS git tree:\n{git_status_start}")

    pers_params = load_persistence_parameters(args.persistence_json)
    g0_pers_index_by_model, g0_pers_sha = load_g0_persistent_per_run_index()

    eval_mod.SyntheticMeasurementGenerator = G1PersistentClutterMeasurementGenerator
    g1_sweep_mod._init_pool_worker = _init_g1_persistent_worker

    with open(G1_CONFIG_PATH, "r", encoding="utf-8") as f:
        base_config = json.load(f)

    sampler = EncounterSamplerV1()
    all_encounters = sampler.generate(t1_cruise_state)
    encounters = all_encounters[: args.encounters]

    max_workers = args.workers
    truth_cache_map, truth_cache_info = load_or_compute_truth_cache(encounters, max_workers=max_workers)

    def _compute_all_hashes() -> dict[str, str]:
        h = compute_code_hashes()
        self_rel = "scripts/xvs/tracking/run_g1_lambda_sweep_persistent.py"
        p_self = REPO_ROOT / self_rel
        if p_self.is_file():
            h[self_rel] = hashlib.sha256(p_self.read_bytes()).hexdigest()
        return h

    start_commit = get_git_commit()
    start_hashes = _compute_all_hashes()
    start_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    out_dir = args.out_dir.resolve() if args.out_dir is not None else (REPO_ROOT / "output" / "xvs" / "wp05" / "g1_lambda_sweep_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    mirror_dir = (REPO_ROOT / "output" / "xvs" / "wp05" / "g1") if args.out_dir is None else None
    if mirror_dir is not None:
        mirror_dir.mkdir(parents=True, exist_ok=True)

    models_out: dict[str, Any] = {}
    all_per_run_records: list[dict[str, Any]] = []
    t0_all = time.perf_counter()

    for mode_name in args.models:
        if mode_name not in D33_MODELS:
            raise ValueError(f"Unsupported D33 model: {mode_name}")
        print(f"\n=== Running G1 D33 Persistent Clutter Model: {mode_name} ===", flush=True)
        cfg_mode = deepcopy(base_config)
        cfg_mode["clutter_p1"] = pers_params["clutter_p1"]
        cfg_mode["clutter_p_stay"] = pers_params["clutter_p_stay"]
        cfg_mode["clutter_jitter_px"] = pers_params["clutter_jitter_px"]
        cfg_mode["clutter_time_base_mode"] = mode_name

        t_m0 = time.perf_counter()
        res_m = g1_sweep_mod.run_g1_lambda_sweep_v2(
            encounters=encounters,
            base_config=cfg_mode,
            truth_cache_map=truth_cache_map,
            max_workers=max_workers,
            lambdas=args.lambdas,
            seeds_full=args.seeds,
            g0_index=g0_pers_index_by_model.get(mode_name),
        )
        t_m1 = time.perf_counter()
        res_m["wall_s"] = float(t_m1 - t_m0)
        res_m["clutter_time_base_mode"] = mode_name
        per_run_m = res_m.pop("_per_run_records", [])
        for row in per_run_m:
            row_with_model = dict(row)
            row_with_model["clutter_model"] = mode_name
            all_per_run_records.append(row_with_model)
        models_out[mode_name] = res_m

    t1_all = time.perf_counter()

    end_commit = get_git_commit()
    end_hashes = _compute_all_hashes()
    git_status_end = get_git_status_porcelain()
    end_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    hashes_match = (start_hashes == end_hashes) and (start_commit == end_commit)
    if args.encounters == 900 and not hashes_match:
        raise RuntimeError(
            f"Long-job rule violation: code hashes or git commit changed during run! "
            f"start={start_hashes}, end={end_hashes}"
        )

    per_run_file = out_dir / "per_run_records_persistent.jsonl"
    with open(per_run_file, "w", encoding="utf-8") as f:
        for row in all_per_run_records:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    per_run_sha256 = hashlib.sha256(per_run_file.read_bytes()).hexdigest()

    primary_key = "primary_1hz_hold" if "primary_1hz_hold" in models_out else args.models[0]
    primary_res = models_out[primary_key]

    payload: dict[str, Any] = {
        "schema": "xvs.g1_lambda_sweep_persistent_clutter/1",
        "description": "WP-05 G1 / Batch B7 persistent terrain clutter lambda sweep under D29, D32, D33 (primary_1hz_hold and sensitivity_10hz_geom) with 6-DoF WGS-84 ground-clutter rejection",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "primary_model": primary_key,
        "encounters_total": primary_res["encounters_total"],
        "lambdas_swept": primary_res["lambdas_swept"],
        "bootstrap_resamples": primary_res["bootstrap_resamples"],
        "bootstrap_seed": primary_res["bootstrap_seed"],
        "pilot_projections": primary_res["pilot_projections"],
        "records": primary_res["records"],
        "analysis_table": primary_res["analysis_table"],
        "lambda_req": primary_res["lambda_req"],
        "lambda_req_statement": primary_res["lambda_req_statement"],
        "wp04_g1_false_alarm_requirement_lambda": primary_res["wp04_g1_false_alarm_requirement_lambda"],
        "models": models_out,
        "clutter_persistence_model": {
            "source": pers_params["source"],
            "source_sha256": pers_params["source_sha256"],
            "mode_matched": "6dof_wgs84_contiguous_markov",
            "primary_time_base": "primary_1hz_hold",
            "sensitivity_time_base": "sensitivity_10hz_geom",
            "clutter_p1": pers_params["clutter_p1"],
            "clutter_p_stay": pers_params["clutter_p_stay"],
            "clutter_mean_len": pers_params["clutter_mean_len_1s"],
            "clutter_jitter_px": pers_params["clutter_jitter_px"],
            "matched_persist_1": pers_params["clutter_p1"],
            "matched_persist_3": pers_params["clutter_persist_3"],
            "sensitivity_p1_step_10hz": pers_params["p1_step_10hz"],
            "sensitivity_p_stay_step_10hz": pers_params["p_stay_step_10hz"],
            "sensitivity_mean_len_step_10hz": pers_params["mean_len_step_10hz"],
        },
        "total_wall_s": float(t1_all - t0_all),
        "workers_used": int(max_workers),
        "truth_cache": truth_cache_info,
        "per_run_records_file": "output/xvs/wp05/g1_lambda_sweep_v2/per_run_records_persistent.jsonl",
        "per_run_records_sha256": per_run_sha256,
        "g0_per_run_records_persistent_sha256": g0_pers_sha,
        "provenance": {
            "batch": "ARGON_B7",
            "wp": "WP-05_G1_PERSISTENT_CLUTTER_D33",
            "status": "completed",
            "argv": list(argv) if argv is not None else list(sys.argv),
            "start_time_utc": start_time_iso,
            "end_time_utc": end_time_iso,
            "git_commit_start": start_commit,
            "git_commit_end": end_commit,
            "git_status_start": git_status_start,
            "git_status_end": git_status_end,
            "sampler_v1_sha256": EXPECTED_SAMPLER_SHA256,
            "per_run_records_file": "output/xvs/wp05/g1_lambda_sweep_v2/per_run_records_persistent.jsonl",
            "per_run_records_sha256": per_run_sha256,
            "g0_per_run_records_persistent_sha256": g0_pers_sha,
            "configs_read": {
                "configs/xvs/tracking/g1.json": hashlib.sha256(G1_CONFIG_PATH.read_bytes()).hexdigest(),
                "output/xvs/wp04/g0/d0_clutter_persistence.json": pers_params["source_sha256"],
            },
            "code_hashes_start": start_hashes,
            "code_hashes_end": end_hashes,
            "code_hashes_match": hashes_match,
            "code_hashes_unchanged_during_run": hashes_match,
            "platform": platform.platform(),
        },
    }

    out_path = out_dir / "lambda_sweep_persistent_clutter.json"
    if args.encounters == 900 or args.out_dir is not None:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved G1 persistent-clutter sweep to {out_path}", flush=True)

        if mirror_dir is not None and mirror_dir != out_dir:
            shutil.copy2(out_path, mirror_dir / "lambda_sweep_persistent_clutter.json")
            shutil.copy2(per_run_file, mirror_dir / "per_run_records_persistent.jsonl")

        indep_path = out_dir / "lambda_sweep_v2_results.json"
        indep_data = json.loads(indep_path.read_text(encoding="utf-8")) if indep_path.is_file() else None
        target_payload = build_g1_target_lambda_req_payload(
            indep_sweep=indep_data,
            indep_sweep_path=indep_path,
            persist_sweep=payload,
            persist_sweep_path=out_path,
            git_commit=end_commit,
        )
        out_target_json = out_dir / "g1_target_lambda_req.json"
        with open(out_target_json, "w", encoding="utf-8") as f:
            json.dump(target_payload, f, indent=2)
        if mirror_dir is not None:
            with open(mirror_dir / "g1_target_lambda_req.json", "w", encoding="utf-8") as f:
                json.dump(target_payload, f, indent=2)
        if not args.skip_target_config and args.encounters == 900 and args.out_dir is None:
            cfg_target_path = REPO_ROOT / "configs" / "xvs" / "detection" / "g1_target_lambda_req.json"
            cfg_target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cfg_target_path, "w", encoding="utf-8") as f:
                json.dump(target_payload, f, indent=2)
            print(f"Saved WP-04 G1 target lambda_req config to {cfg_target_path}", flush=True)

    for m_name, m_res in models_out.items():
        print(f"\n[G1 {m_name}] Persistent-clutter lambda_req = {m_res['lambda_req_statement']}")
        for rec, row in zip(m_res["records"], m_res["analysis_table"]):
            print(
                f"  lambda={row['lambda']}: missed={rec['missed_alert_rate']:.4f}, "
                f"FA_runs/hr={rec['runs_with_false_alert_per_h']:.4f}, "
                f"FA_ev/hr={rec['false_alert_events_per_h']:.4f}, "
                f"FT_ever={rec['false_tracks_count']} ({rec['false_tracks_per_hour']:.4f}/hr, UB={row['garwood_ub_false_tracks_per_hour']:.4f}), "
                f"FT_end={rec['false_tracks_end_of_run']}, "
                f"GND_rej={rec['ground_clutter_rejected_count']}, "
                f"pass_d29={row['fixed_sequence_pass']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
