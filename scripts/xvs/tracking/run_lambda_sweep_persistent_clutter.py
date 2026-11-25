"""WP-05 G0 Item 4a — Persistent Terrain Clutter Lambda Sweep under Decisions D29, D32, D33.

Subclasses SyntheticMeasurementGenerator to evaluate the D29 lambda sweep under
persistent 3D WGS-84 terrain clutter matched to
output/xvs/wp04/g0/d0_clutter_persistence.json:
  - Reads p1, persist_3, p_stay = sqrt(persist_3 / p1), and matched_displacement_std_px
    directly from output/xvs/wp04/g0/d0_clutter_persistence.json (zero typed constants).
  - Evaluates both Decision D33 time-base models:
      1. `primary_1hz_hold`: Markov survival transitions once per 1.0 s (matching G3a's
         1.0 Hz frame grid), with each surviving 3D clutter site projected in all 10
         tracker frames (0.1 s steps) of that second.
      2. `sensitivity_10hz_geom`: Per-0.1 s geometric survival transitions with
         p1_step = p1^(1/10) and p_stay_step = p_stay^(1/10).
  - Judges lambda_req under persistent clutter on the primary model (`primary_1hz_hold`)
    per Decision D33, and reports both models in lambda_sweep_persistent_clutter.json.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xvs.geo import wgs84
from xvs.rig.camera import PinholeCamera
from xvs.sim.trajectory import t1_cruise_state
import xvs.tracking.evaluate as eval_mod
from xvs.tracking.evaluate import (
    get_ownship_states,
    load_or_compute_truth_cache,
)
from xvs.tracking.measurements import DetectionRecord, SyntheticMeasurementGenerator
from xvs.traffic.encounter import EncounterSamplerV1, EncounterTrajectory
import scripts.xvs.tracking.run_lambda_sweep_v2 as sweep_v2_mod
from scripts.xvs.tracking.run_lambda_sweep_v2 import (
    EXPECTED_SAMPLER_SHA256,
    FULL_SEEDS,
    MEASURED_CONFIG_PATH,
    compute_code_hashes,
    get_git_commit,
    get_git_status_porcelain,
    get_safe_workers,
)

PERSISTENCE_JSON_PATH = REPO_ROOT / "output" / "xvs" / "wp04" / "g0" / "d0_clutter_persistence.json"
DEFAULT_PERSISTENT_LAMBDAS = [0.0005, 0.005, 0.05, 0.5]
D33_MODELS = ("primary_1hz_hold", "sensitivity_10hz_geom")


def load_persistence_parameters(persistence_path: Path = PERSISTENCE_JSON_PATH) -> dict[str, Any]:
    """Read Markov persistence parameters directly from d0_clutter_persistence.json (F5)."""
    if not persistence_path.is_file():
        raise FileNotFoundError(f"Missing d0_clutter_persistence.json at {persistence_path}")
    with open(persistence_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    r20_6dof = data["results_by_tau"]["20.0"]["6dof_wgs84"]
    lag1_all = r20_6dof["lag_match"]["all"]["lag_1"]
    cont3_all = r20_6dof["contiguous_persistence"]["all"]["persist_3"]

    p1 = float(lag1_all["fraction_in_roi"])
    persist_3 = float(cont3_all["fraction_in_roi"])
    if p1 <= 0.0:
        raise ValueError(f"Invalid lag_1 fraction_in_roi in {persistence_path}: {p1}")
    p_stay = math.sqrt(persist_3 / p1)
    if "matched_displacement_std_px" not in lag1_all:
        raise KeyError(f"Missing 'matched_displacement_std_px' in {persistence_path}:lag_match.all.lag_1")
    jitter_px = float(lag1_all["matched_displacement_std_px"])

    mean_len_1s = 1.0 + p1 / max(1e-12, 1.0 - p_stay)
    p1_step_10hz = float(p1 ** 0.1)
    p_stay_step_10hz = float(p_stay ** 0.1)
    mean_len_step_10hz = 1.0 + p1_step_10hz / max(1e-12, 1.0 - p_stay_step_10hz)

    return {
        "source": "output/xvs/wp04/g0/d0_clutter_persistence.json",
        "source_sha256": hashlib.sha256(persistence_path.read_bytes()).hexdigest(),
        "clutter_p1": p1,
        "clutter_persist_3": persist_3,
        "clutter_p_stay": p_stay,
        "clutter_mean_len_1s": mean_len_1s,
        "clutter_jitter_px": jitter_px,
        "p1_step_10hz": p1_step_10hz,
        "p_stay_step_10hz": p_stay_step_10hz,
        "mean_len_step_10hz": mean_len_step_10hz,
    }


class PersistentClutterMeasurementGenerator(SyntheticMeasurementGenerator):
    """Extends SyntheticMeasurementGenerator with D33 Markov persistent 3D WGS-84 terrain clutter."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if "clutter_p1" not in self.config or "clutter_p_stay" not in self.config or "clutter_jitter_px" not in self.config:
            raise KeyError("PersistentClutterMeasurementGenerator requires clutter_p1, clutter_p_stay, clutter_jitter_px in config")
        self.clutter_p1: float = float(self.config["clutter_p1"])
        self.clutter_p_stay: float = float(self.config["clutter_p_stay"])
        self.clutter_jitter_px: float = float(self.config["clutter_jitter_px"])
        self.clutter_time_base_mode: str = str(self.config["clutter_time_base_mode"])
        if self.clutter_time_base_mode not in D33_MODELS:
            raise ValueError(f"Unknown clutter_time_base_mode: {self.clutter_time_base_mode}")

        self.steps_per_sec: int = max(1, int(round(self.rate_hz)))
        if self.clutter_time_base_mode == "primary_1hz_hold":
            self.step_p1 = self.clutter_p1
            self.step_p_stay = self.clutter_p_stay
        else:
            inv_steps = 1.0 / float(self.steps_per_sec)
            self.step_p1 = float(self.clutter_p1 ** inv_steps)
            self.step_p_stay = float(self.clutter_p_stay ** inv_steps)

        self.clutter_mean_len: float = 1.0 + self.step_p1 / max(1e-12, 1.0 - self.step_p_stay)
        self.active_clutter_chains: dict[str, list[dict[str, Any]]] = {}

    def reset_state(self, seed: int | None = None) -> None:
        super().reset_state(seed=seed)
        self.active_clutter_chains.clear()

    def _create_clutter_site_from_pixel(
        self,
        u_c: float,
        v_c: float,
        s_base: float,
        p_cam_ecef: np.ndarray,
        r_ecef_cam: np.ndarray,
        cam_model: PinholeCamera,
        age: int = 1,
    ) -> dict[str, Any]:
        d_cam = np.array([(u_c - cam_model.cx) / cam_model.fx, (v_c - cam_model.cy) / cam_model.fy, 1.0], dtype=np.float64)
        d_ecef = r_ecef_cam @ d_cam

        a = wgs84.WGS84_A
        b = wgs84.WGS84_B
        s_vec = np.array([1.0 / a, 1.0 / a, 1.0 / b], dtype=np.float64)
        c_prime = p_cam_ecef * s_vec
        big_k = float(np.dot(c_prime, c_prime)) - 1.0
        d_prime = d_ecef * s_vec
        c_dot_d = float(np.dot(c_prime, d_prime))
        d2 = float(np.dot(d_prime, d_prime))
        disc = c_dot_d * c_dot_d - d2 * big_k

        if disc > 0.0 and c_dot_d < 0.0:
            t_hit = (-c_dot_d - math.sqrt(max(0.0, disc))) / d2
            x_ecef = p_cam_ecef + d_ecef * t_hit
            return {"is_3d": True, "x_ecef": x_ecef, "d_ecef": d_ecef, "s_px": s_base, "age": age}
        return {"is_3d": False, "x_ecef": None, "d_ecef": d_ecef, "s_px": s_base, "age": age}

    def _project_persistent_site(
        self,
        site: dict[str, Any],
        p_cam_ecef: np.ndarray,
        r_ecef_cam: np.ndarray,
        cam_model: PinholeCamera,
    ) -> tuple[float, float, bool]:
        if site["is_3d"] and site["x_ecef"] is not None:
            v_cam = r_ecef_cam.T @ (site["x_ecef"] - p_cam_ecef)
        else:
            v_cam = r_ecef_cam.T @ site["d_ecef"]
        if v_cam[2] <= 1e-3:
            return 0.0, 0.0, False
        u_p = float(cam_model.fx * (v_cam[0] / v_cam[2]) + cam_model.cx)
        v_p = float(cam_model.fy * (v_cam[1] / v_cam[2]) + cam_model.cy)
        in_img = (0.0 <= u_p < cam_model.width) and (0.0 <= v_p < cam_model.height)
        return u_p, v_p, in_img

    def generate_frame(
        self,
        t_s: float,
        ownship_state: Any,
        encounters: Sequence[EncounterTrajectory],
        disable_noise: bool = False,
        pd_override: float | None = None,
    ) -> tuple[list[DetectionRecord], dict[str, Any]]:
        orig_lambda = self.fppi_lambda
        self.fppi_lambda = 0.0
        try:
            detections, metadata = super().generate_frame(
                t_s=t_s,
                ownship_state=ownship_state,
                encounters=encounters,
                disable_noise=disable_noise,
                pd_override=pd_override,
            )
        finally:
            self.fppi_lambda = orig_lambda

        if not disable_noise and self.fppi_lambda > 0.0:
            cam_poses = self.ring.compute_poses(
                lat_deg=ownship_state.lat_deg,
                lon_deg=ownship_state.lon_deg,
                h_m=ownship_state.h_m,
                yaw_rad=ownship_state.yaw_rad,
                pitch_rad=ownship_state.pitch_rad,
                roll_rad=ownship_state.roll_rad,
            )
            lambda_birth = self.fppi_lambda / self.clutter_mean_len
            frame_idx = int(round(t_s * self.rate_hz))

            if self.clutter_time_base_mode == "primary_1hz_hold":
                # D33 Primary Model: Markov transitions once per 1.0 s (every 10 tracker frames),
                # and every active site is projected in all 10 tracker frames of that second.
                is_markov_boundary = (frame_idx % self.steps_per_sec == 0)

                for cam in self.ring.cameras:
                    c_pose = cam_poses[cam.id]
                    p_cam = np.array(c_pose["position"], dtype=np.float64)
                    r_cam = np.array(c_pose["r_ecef_cam"], dtype=np.float64)

                    if cam.id not in self.active_clutter_chains:
                        # Initial second (t_s = 0.0): spawn stationary Poisson(lambda) sites
                        n_init = int(self.rng_clutter.poisson(self.fppi_lambda))
                        init_chains: list[dict[str, Any]] = []
                        for _ in range(n_init):
                            u_0, v_0 = self._sample_false_alarm_position(p_cam, r_cam, self.ring.camera_model)
                            s_0 = float(self.rng_clutter.uniform(self.fa_size_min_px, self.fa_size_max_px))
                            site_new = self._create_clutter_site_from_pixel(
                                u_0, v_0, s_0, p_cam, r_cam, self.ring.camera_model, age=1
                            )
                            init_chains.append(site_new)
                        self.active_clutter_chains[cam.id] = init_chains
                    elif is_markov_boundary:
                        # 1.0 s boundary: transition existing sites and spawn new 1.0 s births
                        prev_chains = self.active_clutter_chains[cam.id]
                        next_chains: list[dict[str, Any]] = []
                        for site in prev_chains:
                            p_surv = self.step_p1 if site["age"] == 1 else self.step_p_stay
                            if self.rng_clutter.uniform(0.0, 1.0) < p_surv:
                                site_next = dict(site)
                                site_next["age"] = site["age"] + 1
                                next_chains.append(site_next)

                        n_new = int(self.rng_clutter.poisson(lambda_birth))
                        for _ in range(n_new):
                            u_0, v_0 = self._sample_false_alarm_position(p_cam, r_cam, self.ring.camera_model)
                            s_0 = float(self.rng_clutter.uniform(self.fa_size_min_px, self.fa_size_max_px))
                            site_new = self._create_clutter_site_from_pixel(
                                u_0, v_0, s_0, p_cam, r_cam, self.ring.camera_model, age=1
                            )
                            next_chains.append(site_new)
                        self.active_clutter_chains[cam.id] = next_chains

                    # Project all active sites in this camera for the current 0.1 s tracker frame
                    for site in self.active_clutter_chains[cam.id]:
                        u_p, v_p, in_img = self._project_persistent_site(site, p_cam, r_cam, self.ring.camera_model)
                        if in_img:
                            u_fa = float(np.clip(u_p + self.rng_clutter.normal(0.0, self.clutter_jitter_px), 0.0, self.ring.camera_model.width - 1.0))
                            v_fa = float(np.clip(v_p + self.rng_clutter.normal(0.0, self.clutter_jitter_px), 0.0, self.ring.camera_model.height - 1.0))
                            s_fa = float(np.clip(site["s_px"] + self.rng_clutter.normal(0.0, 0.25), self.fa_size_min_px, self.fa_size_max_px))
                            fa_score = float(self.detection_probability(s_fa))
                            detections.append(DetectionRecord(
                                t=t_s,
                                camera=cam.id,
                                u=u_fa,
                                v=v_fa,
                                s_px=s_fa,
                                score=fa_score,
                                is_false_alarm=True,
                                truth_id=None,
                            ))
            else:
                # D33 Sensitivity Model (`sensitivity_10hz_geom`): per-0.1 s geometric survival p_step = p^(1/10)
                for cam in self.ring.cameras:
                    c_pose = cam_poses[cam.id]
                    p_cam = np.array(c_pose["position"], dtype=np.float64)
                    r_cam = np.array(c_pose["r_ecef_cam"], dtype=np.float64)

                    prev_chains = self.active_clutter_chains.get(cam.id, [])  # default-ok: initial camera chain list
                    next_chains = []
                    for site in prev_chains:
                        p_surv = self.step_p1 if site["age"] == 1 else self.step_p_stay
                        if self.rng_clutter.uniform(0.0, 1.0) < p_surv:
                            u_p, v_p, in_img = self._project_persistent_site(site, p_cam, r_cam, self.ring.camera_model)
                            if in_img:
                                u_fa = float(np.clip(u_p + self.rng_clutter.normal(0.0, self.clutter_jitter_px), 0.0, self.ring.camera_model.width - 1.0))
                                v_fa = float(np.clip(v_p + self.rng_clutter.normal(0.0, self.clutter_jitter_px), 0.0, self.ring.camera_model.height - 1.0))
                                s_fa = float(np.clip(site["s_px"] + self.rng_clutter.normal(0.0, 0.25), self.fa_size_min_px, self.fa_size_max_px))
                                fa_score = float(self.detection_probability(s_fa))
                                detections.append(DetectionRecord(
                                    t=t_s,
                                    camera=cam.id,
                                    u=u_fa,
                                    v=v_fa,
                                    s_px=s_fa,
                                    score=fa_score,
                                    is_false_alarm=True,
                                    truth_id=None,
                                ))
                                site_next = dict(site)
                                site_next["age"] = site["age"] + 1
                                next_chains.append(site_next)

                    mean_new = self.fppi_lambda if cam.id not in self.active_clutter_chains else lambda_birth
                    n_new = int(self.rng_clutter.poisson(mean_new))
                    for _ in range(n_new):
                        u_fa, v_fa = self._sample_false_alarm_position(p_cam, r_cam, self.ring.camera_model)
                        s_fa = float(self.rng_clutter.uniform(self.fa_size_min_px, self.fa_size_max_px))
                        fa_score = float(self.detection_probability(s_fa))
                        detections.append(DetectionRecord(
                            t=t_s,
                            camera=cam.id,
                            u=u_fa,
                            v=v_fa,
                            s_px=s_fa,
                            score=fa_score,
                            is_false_alarm=True,
                            truth_id=None,
                        ))
                        site_new = self._create_clutter_site_from_pixel(
                            u_fa, v_fa, s_fa, p_cam, r_cam, self.ring.camera_model, age=1
                        )
                        next_chains.append(site_new)

                    self.active_clutter_chains[cam.id] = next_chains

        return detections, metadata


def _init_persistent_worker() -> None:
    eval_mod.SyntheticMeasurementGenerator = PersistentClutterMeasurementGenerator
    get_ownship_states()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WP-05 G0 Item 4a Persistent Terrain Clutter Lambda Sweep (D29, D32, D33)")
    parser.add_argument("--encounters", type=int, default=900)
    parser.add_argument("--lambdas", type=float, nargs="+", default=list(DEFAULT_PERSISTENT_LAMBDAS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(FULL_SEEDS))
    parser.add_argument("--models", type=str, nargs="+", default=list(D33_MODELS))
    parser.add_argument("--persistence-json", type=Path, default=PERSISTENCE_JSON_PATH)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    git_status_start = get_git_status_porcelain()
    if args.encounters == 900 and git_status_start:
        raise RuntimeError(f"Refusing to run full persistent-clutter sweep on dirty git working tree:\n{git_status_start}")

    pers_params = load_persistence_parameters(args.persistence_json)

    eval_mod.SyntheticMeasurementGenerator = PersistentClutterMeasurementGenerator
    sweep_v2_mod._init_pool_worker = _init_persistent_worker

    with open(MEASURED_CONFIG_PATH, "r", encoding="utf-8") as f:
        base_config = json.load(f)

    sampler = EncounterSamplerV1()
    all_encounters = sampler.generate(t1_cruise_state)
    encounters = all_encounters[: args.encounters]

    max_workers = get_safe_workers()
    truth_cache_map, truth_cache_info = load_or_compute_truth_cache(encounters, max_workers=max_workers)

    def _compute_all_hashes() -> dict[str, str]:
        h = compute_code_hashes()
        self_rel = "scripts/xvs/tracking/run_lambda_sweep_persistent_clutter.py"
        p_self = REPO_ROOT / self_rel
        if p_self.is_file():
            h[self_rel] = hashlib.sha256(p_self.read_bytes()).hexdigest()
        return h

    start_commit = get_git_commit()
    start_hashes = _compute_all_hashes()
    start_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    models_out: dict[str, Any] = {}
    all_per_run_records: list[dict[str, Any]] = []
    t0_all = time.perf_counter()

    for mode_name in args.models:
        if mode_name not in D33_MODELS:
            raise ValueError(f"Unsupported D33 model: {mode_name}")
        print(f"\n=== Running D33 Persistent Clutter Model: {mode_name} ===", flush=True)
        cfg_mode = deepcopy(base_config)
        cfg_mode["clutter_p1"] = pers_params["clutter_p1"]
        cfg_mode["clutter_p_stay"] = pers_params["clutter_p_stay"]
        cfg_mode["clutter_jitter_px"] = pers_params["clutter_jitter_px"]
        cfg_mode["clutter_time_base_mode"] = mode_name

        t_m0 = time.perf_counter()
        res_m = sweep_v2_mod.run_lambda_sweep_v2(
            encounters=encounters,
            base_config=cfg_mode,
            truth_cache_map=truth_cache_map,
            max_workers=max_workers,
            lambdas=args.lambdas,
            seeds_full=args.seeds,
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

    # Write strictly to worktree output (F6)
    out_dir = args.out_dir.resolve() if args.out_dir is not None else (REPO_ROOT / "output" / "xvs" / "wp05" / "g0_lambda_sweep_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run_file = out_dir / "per_run_records_persistent.jsonl"
    with open(per_run_file, "w", encoding="utf-8") as f:
        for row in all_per_run_records:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    per_run_sha256 = hashlib.sha256(per_run_file.read_bytes()).hexdigest()

    primary_key = "primary_1hz_hold" if "primary_1hz_hold" in models_out else args.models[0]
    primary_res = models_out[primary_key]

    payload: dict[str, Any] = {
        "schema": "xvs.lambda_sweep_persistent_clutter/2",
        "description": "WP-05 G0 Item 4a persistent terrain clutter lambda sweep under D29, D32, D33 (primary_1hz_hold and sensitivity_10hz_geom)",
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
        "per_run_records_file": "output/xvs/wp05/g0_lambda_sweep_v2/per_run_records_persistent.jsonl",
        "per_run_records_sha256": per_run_sha256,
        "provenance": {
            "batch": "ARGON_B6_ITEM_4A_PERSISTENT_CLUTTER_D33",
            "argv": list(argv) if argv is not None else list(sys.argv),
            "start_time_utc": start_time_iso,
            "end_time_utc": end_time_iso,
            "git_commit_start": start_commit,
            "git_commit_end": end_commit,
            "git_status_start": git_status_start,
            "git_status_end": git_status_end,
            "sampler_v1_sha256": EXPECTED_SAMPLER_SHA256,
            "per_run_records_file": "output/xvs/wp05/g0_lambda_sweep_v2/per_run_records_persistent.jsonl",
            "per_run_records_sha256": per_run_sha256,
            "configs_read": {
                "configs/xvs/tracking/g0_measured_d0.json": hashlib.sha256(MEASURED_CONFIG_PATH.read_bytes()).hexdigest(),
                "output/xvs/wp04/g0/d0_clutter_persistence.json": pers_params["source_sha256"],
            },
            "code_hashes_start": start_hashes,
            "code_hashes_end": end_hashes,
            "code_hashes_unchanged_during_run": hashes_match,
            "platform": platform.platform(),
        },
    }

    out_path = out_dir / "lambda_sweep_persistent_clutter.json"
    if args.encounters == 900 or args.out_dir is not None:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved persistent-clutter sweep to {out_path}", flush=True)

    for m_name, m_res in models_out.items():
        print(f"\n[{m_name}] Persistent-clutter lambda_req = {m_res['lambda_req_statement']}")
        for rec, row in zip(m_res["records"], m_res["analysis_table"]):
            print(
                f"  lambda={row['lambda']}: missed={rec['missed_alert_rate']:.4f}, "
                f"FA_runs/hr={rec['runs_with_false_alert_per_h']:.4f}, "
                f"FA_ev/hr={rec['false_alert_events_per_h']:.4f}, "
                f"FT_ever={rec['false_tracks_count']} ({rec['false_tracks_per_hour']:.4f}/hr, UB={row['garwood_ub_false_tracks_per_hour']:.4f}), "
                f"FT_end={rec['false_tracks_end_of_run']}, "
                f"pass_d29={row['fixed_sequence_pass']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
