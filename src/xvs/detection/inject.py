"""Controlled detectability injection study module for OpenXVS (WP-04 G0).

Conforms strictly to docs/xvs/WP04_DETECTION.md v0.1 §5.
Generates controlled aircraft silhouettes with 4x sub-pixel rasterization,
composites them in linear luminance over real or synthetic backgrounds,
applies the WP-03 degraded/clean sensor chain, and measures D0 detectability.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence
import numpy as np

from xvs.traffic.raster import glb_mesh, rasterize
from xvs.metrics.aot import compute_eiou, clopper_pearson_interval, poisson_rate_upper_bound
from xvs.sensor.chain import SensorConfig
from xvs.sensor.psf import apply_psf
from xvs.sensor.noise import apply_sensor_noise, generate_fpn_maps
from xvs.sensor.srgb import srgb_eotf, linear_rgb_to_monochrome
from xvs.detection.d0 import detect_d0, load_d0_config
from xvs.detection.fit import fit_logistic_mle, LogisticFitResult
from xvs.paths import get_data_root

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ASSETS_DIR = REPO_ROOT / "assets" / "traffic"

CLASSES = ["C1_UAS", "C2_SEP", "C3_TWIN", "C4_BIZJET", "C5_TRANSPORT", "C6_HELI"]
GRID_S_PX = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
GRID_C = [-0.4, -0.2, -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1]
GRID_BACKGROUND = ["sky", "earth"]
GRID_SENSOR = ["clean", "degraded"]
GRID_K = [3, 5, 7]


def matching_radius(s_px: float) -> float:
    """Compute primary matching radius r = max(1.5 px, 0.5 * s_px) per D23."""
    return max(1.5, 0.5 * float(s_px))


def is_primary_hit(
    u_det: float,
    v_det: float,
    u_t: float,
    v_t: float,
    s_px: float,
) -> bool:
    """Primary hit rule: detection centroid lies within r = max(1.5 px, 0.5 * s_px) of injected centre."""
    r = matching_radius(s_px)
    dist = math.hypot(float(u_det) - float(u_t), float(v_det) - float(v_t))
    return dist <= r


def is_secondary_hit(
    box_det: Sequence[float],
    box_gt: Sequence[float],
    threshold: float = 0.2,
) -> bool:
    """Secondary hit rule: AOT eIoU > 0.2 with target vertex box."""
    return compute_eiou(box_gt, box_det) > threshold


def compute_corrected_pd_bootstrap(
    y_hits: Sequence[int] | np.ndarray,
    z_null_hits: Sequence[int] | np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute raw P_d, chance-hit rate p0, and Abbott-corrected P_d with 95% bootstrap intervals.

    Corrected P_d = (P_d - p0) / (1 - p0).
    Resamples over trials using seeded generator.
    """
    y = np.asarray(y_hits, dtype=np.float64)
    z = np.asarray(z_null_hits, dtype=np.float64)
    n_y = len(y)
    n_z = len(z)
    if n_y == 0 or n_z == 0:
        return {
            "pd_raw": float("nan"),
            "pd_raw_ci95": [float("nan"), float("nan")],
            "p0": float("nan"),
            "p0_ci95": [float("nan"), float("nan")],
            "pd_corrected": float("nan"),
            "pd_corrected_ci95": [float("nan"), float("nan")],
        }

    pd_raw = float(np.mean(y))
    p0 = float(np.mean(z))
    if p0 < 1.0:
        pd_corr = (pd_raw - p0) / (1.0 - p0)
    else:
        pd_corr = 0.0

    rng = np.random.default_rng(seed)
    boot_y = rng.binomial(n_y, pd_raw, size=n_boot) / float(n_y)
    boot_z = rng.binomial(n_z, p0, size=n_boot) / float(n_z)
    safe_mask = boot_z < 1.0
    denom = np.where(safe_mask, 1.0 - boot_z, 1.0)
    boot_corr = np.where(safe_mask, (boot_y - boot_z) / denom, 0.0)
    boot_corr = np.clip(boot_corr, -1.0, 1.0)

    ci_pd_low, ci_pd_high = float(np.percentile(boot_y, 2.5)), float(np.percentile(boot_y, 97.5))
    ci_p0_low, ci_p0_high = float(np.percentile(boot_z, 2.5)), float(np.percentile(boot_z, 97.5))
    ci_corr_low, ci_corr_high = float(np.percentile(boot_corr, 2.5)), float(np.percentile(boot_corr, 97.5))

    return {
        "pd_raw": pd_raw,
        "pd_raw_ci95": [ci_pd_low, ci_pd_high],
        "p0": p0,
        "p0_ci95": [ci_p0_low, ci_p0_high],
        "pd_corrected": pd_corr,
        "pd_corrected_ci95": [ci_corr_low, ci_corr_high],
    }


TAU_LIST_EXTENDED = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0]
CROP_AREA_PX = 256 * 256
EXCLUSION_RADIUS_PX = 16.0
EXCLUSION_AREA_PX = math.pi * (EXCLUSION_RADIUS_PX ** 2)
TRIAL_SEARCH_AREA_MPX = (CROP_AREA_PX - EXCLUSION_AREA_PX) / 1.0e6  # ~0.06473175 Mpx


def evaluate_detections_for_trial(
    dets: list[dict[str, Any]],
    u_t: float,
    v_t: float,
    box_gt: list[float],
    s_px: float,
) -> dict[str, Any]:
    """Evaluate detections against hit rules and collect scores for tau sweep.

    Per G-B8 A2:
      - the best score of any detection meeting each hit rule (primary and secondary);
      - the scores of all detections more than 16 px from the injected centre.
    """
    primary_best = -float("inf")
    secondary_best = -float("inf")
    fp_scores: list[float] = []

    r = matching_radius(s_px)

    for d in dets:
        u = float(d["u"])
        v = float(d["v"])
        score = float(d["score"])
        dist = math.hypot(u - u_t, v - v_t)

        # Primary hit rule: centroid within r
        if dist <= r:
            if score > primary_best:
                primary_best = score

        # Secondary hit rule: eIoU > 0.2
        if dist <= max(32.0, 3.0 * s_px) and compute_eiou(box_gt, d["box_edges"]) > 0.2:
            if score > secondary_best:
                secondary_best = score

        # FP candidate: more than 16 px from injected centre
        if dist > EXCLUSION_RADIUS_PX:
            fp_scores.append(score)

    return {
        "primary_best_score": primary_best,
        "secondary_best_score": secondary_best,
        "fp_scores": fp_scores,
    }


def compute_fp_density_sweep(
    all_fp_scores: list[float],
    num_trials: int,
    tau_list: Sequence[float] = TAU_LIST_EXTENDED,
    search_area_per_trial_mpx: float = TRIAL_SEARCH_AREA_MPX,
) -> dict[str, dict[str, float]]:
    """Compute FP count, FP density per Mpx, and Poisson 95% upper bound for each tau."""
    total_area_mpx = float(num_trials) * search_area_per_trial_mpx
    scores_arr = np.array(all_fp_scores, dtype=np.float64)

    sweep_results: dict[str, dict[str, float]] = {}
    for tau in tau_list:
        tau_key = f"{tau:.1f}"
        if len(scores_arr) > 0:
            fp_count = int(np.sum(scores_arr >= tau))
        else:
            fp_count = 0

        fp_per_mpx = float(fp_count / total_area_mpx) if total_area_mpx > 0 else 0.0
        fp_ci_high = float(poisson_rate_upper_bound(fp_count, total_area_mpx, confidence=0.95))

        sweep_results[tau_key] = {
            "tau": float(tau),
            "fp_count": fp_count,
            "total_area_mpx": total_area_mpx,
            "fp_per_mpx": fp_per_mpx,
            "fp_per_mpx_ci95_high": fp_ci_high,
        }

    return sweep_results


def find_operating_point_tau(
    fp_sweep: dict[str, dict[str, float]],
    target_fp_upper_per_mpx: float = 1.0e-4,
) -> dict[str, Any]:
    """Find smallest tau whose Poisson FP 95% upper bound is <= target_fp_upper_per_mpx.

    Target 1.0e-4 FP/Mpx corresponds to lambda = 0.0005 per 5.01 Mpx full frame.
    Returns status 'RESOLVED' with tau, or 'UNRESOLVED' when data cannot show it.
    """
    sorted_taus = sorted(fp_sweep.keys(), key=lambda k: float(k))
    for t_k in sorted_taus:
        stat = fp_sweep[t_k]
        if stat["fp_per_mpx_ci95_high"] <= target_fp_upper_per_mpx:
            return {
                "status": "RESOLVED",
                "tau_star": float(stat["tau"]),
                "fp_per_mpx": stat["fp_per_mpx"],
                "fp_per_mpx_ci95_high": stat["fp_per_mpx_ci95_high"],
            }

    min_fp_ub = fp_sweep[sorted_taus[-1]]["fp_per_mpx_ci95_high"] if sorted_taus else float("nan")
    min_fp_rate = fp_sweep[sorted_taus[-1]]["fp_per_mpx"] if sorted_taus else float("nan")
    return {
        "status": "UNRESOLVED",
        "tau_star": None,
        "reason": f"No tau in the swept range reaches {target_fp_upper_per_mpx:.1e} FP/Mpx: minimum measured FP upper bound ({min_fp_ub:.2e}/Mpx) exceeds target",
        "min_measured_fp_per_mpx": min_fp_rate,
        "min_measured_fp_per_mpx_ci95_high": min_fp_ub,
    }


def load_traffic_models(assets_dir: Path | str | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load all 18 standard traffic 3D models into memory."""
    base = Path(assets_dir) if assets_dir is not None else DEFAULT_ASSETS_DIR
    models: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cls_name in CLASSES:
        for m_idx in (1, 2, 3):
            name = f"{cls_name}_{m_idx:04d}"
            path = base / cls_name / f"{name}.glb"
            if path.is_file():
                verts, faces = glb_mesh(path)
                models[name] = (verts, faces)
    return models


def sample_random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Sample a uniform random 3D rotation matrix from Haar measure on SO(3)."""
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    r = np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)
    return r


def sample_traffic_silhouette(
    models: dict[str, tuple[np.ndarray, np.ndarray]],
    s_px: float,
    u_t: float,
    v_t: float,
    rng: np.random.Generator,
    crop_size: int = 256,
) -> dict[str, Any]:
    """Generate a random traffic silhouette scaled to target s_px and placed at (u_t, v_t).

    Rasterized at 4x sub-pixel resolution via xvs.traffic.raster.rasterize.
    """
    model_keys = sorted(models.keys())
    if not model_keys:
        raise ValueError("No traffic models available to sample from")
    chosen_key = model_keys[int(rng.integers(0, len(model_keys)))]
    verts, faces = models[chosen_key]

    # Random attitude
    r_mat = sample_random_rotation(rng)
    v_rot = (r_mat @ verts.T).T

    x_min, x_max = float(v_rot[:, 0].min()), float(v_rot[:, 0].max())
    y_min, y_max = float(v_rot[:, 1].min()), float(v_rot[:, 1].max())
    w_model = max(1e-6, x_max - x_min)
    h_model = max(1e-6, y_max - y_min)
    s_model = max(w_model, h_model)

    scale = s_px / s_model
    x_c = (x_min + x_max) / 2.0
    y_c = (y_min + y_max) / 2.0

    u = u_t + scale * (v_rot[:, 0] - x_c)
    v = v_t + scale * (v_rot[:, 1] - y_c)

    box_gt = [
        float(u_t - scale * w_model / 2.0),
        float(v_t - scale * h_model / 2.0),
        float(u_t + scale * w_model / 2.0),
        float(v_t + scale * h_model / 2.0),
    ]

    # 4x sub-pixel rasterization (localized window for 25x speedup)
    x0 = max(0, int(np.floor(np.min(u))) - 1)
    x1 = min(crop_size, int(np.ceil(np.max(u))) + 2)
    y0 = max(0, int(np.floor(np.min(v))) - 1)
    y1 = min(crop_size, int(np.ceil(np.max(v))) + 2)
    w_loc = x1 - x0
    h_loc = y1 - y0

    u_loc_4x = 4.0 * (u - x0) + 1.5
    v_loc_4x = 4.0 * (v - y0) + 1.5
    mask_loc_4x = rasterize(u_loc_4x, v_loc_4x, faces, width=w_loc * 4, height=h_loc * 4)
    alpha_loc = mask_loc_4x.reshape(h_loc, 4, w_loc, 4).mean(axis=(1, 3))

    alpha = np.zeros((crop_size, crop_size), dtype=np.float64)
    alpha[y0:y1, x0:x1] = alpha_loc
    sum_alpha = float(np.sum(alpha_loc))

    return {
        "alpha": alpha,
        "box_gt": box_gt,
        "model_id": chosen_key,
        "sum_alpha": sum_alpha,
    }


def composite_signal(
    L_bg: np.ndarray,
    alpha: np.ndarray,
    C: float,
) -> tuple[np.ndarray, float]:
    """Composite silhouette over background in linear luminance: L' = L_bg * (1 + C * alpha)."""
    L_prime = L_bg * (1.0 + C * alpha)
    A_C = float(C * np.sum(alpha))
    return L_prime, A_C


def extract_background_crops(
    data_root: Path | None = None,
    num_crops_per_type: int = 200,
    rng: np.random.Generator | None = None,
    crop_size: int = 256,
) -> dict[str, Any]:
    """Extract 256x256 background crops from real G4 T1 and T3 rig cameras (F/R/B/L).

    Conforms to GEMINI_G-B8 A3 and F-B7-3:
      - Sky crops: band 16-200 px above the WP-01 horizon line.
      - Earth crops: band 16-200 px below the WP-01 horizon line.
      - Cameras: F, R, B, L from T1_cruise and T3_turn_climb (U/D nadir/zenith dropped).
      - No synthetic fallback: raises FileNotFoundError if G4 data missing.
      - Records source (file, camera, frame, offset) and SHA-256 of crop manifest.
      - Computes sigma_bg = 1.4826 * MAD of linear monochrome image.
    """
    if rng is None:
        rng = np.random.default_rng(20260925)

    root = data_root if data_root is not None else get_data_root()
    if root is None or not root.is_dir():
        raise FileNotFoundError(f"Data root directory does not exist: {root}")

    from PIL import Image

    cams_allowed = ["F", "R", "B", "L"]
    seqs = ["T1_cruise", "T3_turn_climb"]

    candidates: list[tuple[str, int, str, np.ndarray, Path]] = []
    for seq in seqs:
        seq_dir = root / "xvs" / "wp01" / "g4" / seq
        frames_path = seq_dir / "frames.jsonl"
        if not frames_path.is_file():
            raise FileNotFoundError(f"Missing G4 frames file: {frames_path}")

        with open(frames_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                frame_obj = json.loads(line)
                k = frame_obj.get("k", line_idx)
                for cam in cams_allowed:
                    cam_data = frame_obj.get("cams", {}).get(cam, {})
                    gt = cam_data.get("gt", {})
                    if not gt.get("horizon_visible", False):
                        continue
                    pts = np.array(gt.get("horizon_uv", []))
                    if len(pts) < 2:
                        continue
                    png_path = seq_dir / cam / f"{k:06d}.png"
                    if not png_path.is_file():
                        rel_png = cam_data.get("png")
                        if rel_png:
                            png_path = seq_dir / rel_png
                    if png_path.is_file():
                        candidates.append((seq, k, cam, pts, png_path))

    if not candidates:
        raise FileNotFoundError(f"No valid G4 frames found in {root} for sequences {seqs}")

    perm = rng.permutation(len(candidates))
    shuffled_candidates = [candidates[i] for i in perm]

    crops_sky: list[dict[str, Any]] = []
    crops_earth: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for seq, k, cam, pts, png_path in shuffled_candidates:
        if len(crops_sky) >= num_crops_per_type and len(crops_earth) >= num_crops_per_type:
            break

        sort_idx = np.argsort(pts[:, 0])
        u_sorted = pts[sort_idx, 0]
        v_sorted = pts[sort_idx, 1]

        # Load image on demand
        lin_mono: np.ndarray | None = None

        for _ in range(5):
            x0 = int(rng.integers(0, 1024 - crop_size + 1))
            xs = np.linspace(x0, x0 + crop_size, 33)
            vs = np.interp(xs, u_sorted, v_sorted)
            v_min, v_max = float(np.min(vs)), float(np.max(vs))

            # Sky crop: bottom edge (y0 + crop_size) between 16 and 200 px above horizon
            if len(crops_sky) < num_crops_per_type:
                y0_sky_min = max(0, int(math.ceil(v_max - 200 - crop_size)))
                y0_sky_max = int(math.floor(v_min - 16 - crop_size))
                if y0_sky_min <= y0_sky_max:
                    y0 = int(rng.integers(y0_sky_min, y0_sky_max + 1))
                    if lin_mono is None:
                        with Image.open(png_path) as im:
                            arr = np.asarray(im)
                        lin_rgb = srgb_eotf(arr) if (arr.ndim == 3 and arr.shape[2] == 3) else arr
                        lin_mono = linear_rgb_to_monochrome(lin_rgb) if (arr.ndim == 3 and arr.shape[2] == 3) else srgb_eotf(arr)

                    crop_img = lin_mono[y0 : y0 + crop_size, x0 : x0 + crop_size]
                    med = float(np.median(crop_img))
                    mad = float(np.median(np.abs(crop_img - med)))
                    sigma_bg = float(1.4826 * mad)

                    meta = {
                        "sequence": seq,
                        "camera": cam,
                        "frame_idx": int(k),
                        "file": f"{cam}/{k:06d}.png",
                        "offset": [int(x0), int(y0)],
                        "crop_size": [int(crop_size), int(crop_size)],
                        "type": "sky",
                        "sigma_bg": sigma_bg,
                    }
                    crops_sky.append({
                        "image": crop_img,
                        "sigma_bg": sigma_bg,
                        "type": "sky",
                        "source": f"{seq}/{cam}/{k:06d}.png",
                        "metadata": meta,
                    })
                    manifest.append(meta)

            # Earth crop: top edge y0 between 16 and 200 px below horizon
            if len(crops_earth) < num_crops_per_type:
                y0_earth_min = int(math.ceil(v_max + 16))
                y0_earth_max = min(1024 - crop_size, int(math.floor(v_min + 200)))
                if y0_earth_min <= y0_earth_max:
                    y0 = int(rng.integers(y0_earth_min, y0_earth_max + 1))
                    if lin_mono is None:
                        with Image.open(png_path) as im:
                            arr = np.asarray(im)
                        lin_rgb = srgb_eotf(arr) if (arr.ndim == 3 and arr.shape[2] == 3) else arr
                        lin_mono = linear_rgb_to_monochrome(lin_rgb) if (arr.ndim == 3 and arr.shape[2] == 3) else srgb_eotf(arr)

                    crop_img = lin_mono[y0 : y0 + crop_size, x0 : x0 + crop_size]
                    med = float(np.median(crop_img))
                    mad = float(np.median(np.abs(crop_img - med)))
                    sigma_bg = float(1.4826 * mad)

                    meta = {
                        "sequence": seq,
                        "camera": cam,
                        "frame_idx": int(k),
                        "file": f"{cam}/{k:06d}.png",
                        "offset": [int(x0), int(y0)],
                        "crop_size": [int(crop_size), int(crop_size)],
                        "type": "earth",
                        "sigma_bg": sigma_bg,
                    }
                    crops_earth.append({
                        "image": crop_img,
                        "sigma_bg": sigma_bg,
                        "type": "earth",
                        "source": f"{seq}/{cam}/{k:06d}.png",
                        "metadata": meta,
                    })
                    manifest.append(meta)

            if len(crops_sky) >= num_crops_per_type and len(crops_earth) >= num_crops_per_type:
                break

    if len(crops_sky) < num_crops_per_type or len(crops_earth) < num_crops_per_type:
        raise RuntimeError(
            f"Could not extract required crops: sky={len(crops_sky)}/{num_crops_per_type}, "
            f"earth={len(crops_earth)}/{num_crops_per_type}"
        )

    manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    return {
        "sky": crops_sky,
        "earth": crops_earth,
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
    }


class InjectionSensorSimulator:
    """Sensor simulator optimized for 256x256 injection crops."""

    def __init__(self, config_path: Path | str | None = None):
        cfg_p = Path(config_path) if config_path is not None else REPO_ROOT / "configs" / "xvs" / "sensors" / "ring_mono_v0_nodist.json"
        self.config = SensorConfig.from_file(cfg_p)
        self.psf_kernel, _, _ = self.config.get_psf_kernel()
        self.prnu_map, self.dsnu_map = generate_fpn_maps("cam_inject", 256, 256, self.config.prnu_rms, self.config.dsnu_rms_e)

    def apply_sensor(self, L_prime: np.ndarray, is_degraded: bool, seed: int) -> np.ndarray:
        """Apply sensor imaging chain to 256x256 linear image."""
        if not is_degraded:
            return np.asarray(L_prime, dtype=np.float64)

        # Degraded chain: PSF convolution + EMVA noise
        blurred = apply_psf(L_prime, self.psf_kernel)
        rng = np.random.Generator(np.random.PCG64(seed))
        raw_dn = apply_sensor_noise(
            linear_radiance=blurred,
            prnu_map=self.prnu_map,
            dsnu_map=self.dsnu_map,
            E=self.config.full_well_e,
            full_well_e=self.config.full_well_e,
            sigma_r_e=self.config.sigma_r_e,
            dark_current_e_per_s=self.config.dark_current_e_per_s,
            exposure_time_s=self.config.exposure_time_s,
            k_dn_per_e=self.config.k_dn_per_e,
            bits=self.config.bits,
            black_level_dn=self.config.black_level_dn,
            rng_temporal=rng,
        )
        L_deg = (raw_dn.astype(np.float64) - self.config.black_level_dn) / self.config.k_dn_per_e
        return L_deg


def fit_ac_logistic(
    ac_values: Sequence[float],
    detected_labels: Sequence[int],
) -> dict[str, Any]:
    """Fit logistic detection curve against ln|A_C|: P_d(A_C) = 1 / (1 + exp(-(ln|A_C| - a) / b))."""
    valid_pts = []
    for ac, d in zip(ac_values, detected_labels):
        abs_ac = abs(ac)
        if abs_ac > 1e-6:
            valid_pts.append((math.log(abs_ac), d))

    if len(valid_pts) < 10:
        return {"computable": False, "reason": "Insufficient valid points"}

    x = np.array([pt[0] for pt in valid_pts], dtype=np.float64)
    y = np.array([pt[1] for pt in valid_pts], dtype=np.float64)

    # Check variation in y
    if np.all(y == 0.0) or np.all(y == 1.0):
        return {
            "computable": True,
            "ln_ac_50": float("nan"),
            "w": float("nan"),
            "all_constant": bool(np.all(y == 1.0)),
        }

    fit_res = fit_logistic_mle(x, y)
    return {
        "computable": fit_res.converged,
        "ln_ac_50": fit_res.s50,
        "w": fit_res.w,
        "ci_s50": [fit_res.ci95_low, fit_res.ci95_high],
        "se_s50": fit_res.se_s50,
    }
