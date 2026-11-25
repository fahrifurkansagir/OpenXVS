"""Deterministic compact data pack builder and post-build leakage auditor (WP-04 §9.3, D34 S1/S2/S3/S8/S9).

Builds:
1. `g1_train_val_pack_v0.tar.zst` + `pack_manifest.json` (strictly train and validation splits only):
   - Delta_t = 1.0 s temporal ROI stacks (K in {1, 3, 5}), storing both clean and WP-03 degraded frames,
   - Horizon strips straddling the analytical WP-01 WGS-84 horizon curve (`horizon_curve_uv`),
   - D0 hard-negative crops centered on actual `detect_d0(..., tau=20.0)` false positives mined strictly from train/val,
   - Per-frame WP-06 attitude (`q_ned_to_cam_est`) and navigation position (`pos_wgs84`) warp metadata.
2. `g1_test_eval_pack_v0.tar.zst` + `test_pack_manifest.json` (strictly held-out test split only, covering all 27 test encounters + test negative time-block frames with full camera-ROI background context).
3. `verify_pack_no_leakage(...)`: runs AFTER pack build on disk (Decision D34-S9, V1_G1),
   inspecting the built `.tar.zst` archives and manifests to verify per-item SHA-256 digests
   and prove zero overlap in file paths, E#### encounter IDs, negative time-block frames, or
   SHA-256 digests between train/val and test packs.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Any, Iterator, Mapping, Sequence
import numpy as np
from PIL import Image

from xvs.detection.d0 import detect_d0, load_d0_config
from xvs.detection.splits import (
    DEFAULT_SPLITS_CONFIG,
    assign_encounter_split,
    assign_negative_split_by_time,
    load_splits_config,
)
from xvs.detection.warp import (
    DEFAULT_CX,
    DEFAULT_CY,
    DEFAULT_FX,
    DEFAULT_FY,
    intersect_rays_wgs84_and_reproject,
    resolve_pose_ecef_and_r_cam,
    validate_pose_dict,
)
from xvs.geo import rotations, wgs84
from xvs.sensor.chain import SensorConfig
from xvs.sensor.noise import apply_sensor_noise, generate_fpn_maps
from xvs.sensor.psf import apply_psf
from xvs.sensor.srgb import REC709_B_WEIGHT, REC709_G_WEIGHT, REC709_R_WEIGHT, srgb_eotf

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_G3A_ROOT = REPO_ROOT / "output" / "xvs" / "wp02" / "g3a_v01"
DEFAULT_WP06_T1_PATH = REPO_ROOT / "output" / "xvs" / "wp06" / "g0" / "T1_cruise_degraded_results.json"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file_stream(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@contextlib.contextmanager
def _open_tar_zst_write(path: Path) -> Iterator[tarfile.TarFile]:
    """Open a `.tar.zst` archive for writing across Python 3.14 (`w:zst`) and Python 3.10-3.13 (`zstandard`)."""
    try:
        tf_std = tarfile.open(path, "w:zst")
    except (ValueError, tarfile.CompressionError):
        try:
            import zstandard as zstd  # type: ignore[import-not-found]

            cctx = zstd.ZstdCompressor(level=3)
            with path.open("wb") as raw_out:
                with cctx.stream_writer(raw_out) as z_out:
                    with tarfile.open(fileobj=z_out, mode="w|") as tf_z:
                        yield tf_z
            return
        except ImportError:  # pragma: no cover
            with tarfile.open(path, "w:gz") as tf_gz:
                yield tf_gz
            return
    with tf_std:
        yield tf_std


@contextlib.contextmanager
def _open_tar_zst_stream(path: Path) -> Iterator[tarfile.TarFile]:
    """Open a `.tar.zst` archive in strictly single-pass forward-streaming mode (`O(N)`, `B8-Colab-Hotfix`).

    Uses `tarfile.open(path, "r|zst")` on Python 3.14+ (avoiding the `r:zst` backward-seeking trap)
    and falls back to `zstandard.ZstdDecompressor().stream_reader` + `tarfile.open(..., mode="r|*")`
    on Python < 3.14.
    """
    try:
        tf_std = tarfile.open(path, "r|zst")
    except (ValueError, tarfile.CompressionError, tarfile.ReadError):
        try:
            import zstandard as zstd  # type: ignore[import-not-found]

            dctx = zstd.ZstdDecompressor()
            with path.open("rb") as raw_in:
                with dctx.stream_reader(raw_in) as z_in:
                    with tarfile.open(fileobj=z_in, mode="r|*") as tf_z:
                        yield tf_z
            return
        except Exception:  # pragma: no cover
            with tarfile.open(path, "r|*") as tf_any:
                yield tf_any
            return
    with tf_std:
        yield tf_std


@contextlib.contextmanager
def _open_tar_zst_read(path: Path) -> Iterator[tarfile.TarFile]:
    """Alias to single-pass forward-streaming `.tar.zst` reader (`_open_tar_zst_stream`)."""
    with _open_tar_zst_stream(path) as tf:
        yield tf


def _add_deterministic_bytes(tf: tarfile.TarFile, arcname: str, payload: bytes) -> str:
    """Add `payload` to tar archive `tf` with fixed mtime/uid/gid for byte-identical determinism."""
    info = tarfile.TarInfo(name=arcname)
    info.size = len(payload)
    info.mtime = 1700000000
    info.uid = 0
    info.gid = 0
    info.uname = "xvs"
    info.gname = "xvs"
    info.mode = 0o644
    tf.addfile(info, io.BytesIO(payload))
    return _sha256_bytes(payload)


def serialize_stack_npz(frames: Sequence[np.ndarray]) -> bytes:
    """Deterministically serialize a stack of 2D float32/uint16 ROI arrays into uncompressed NPZ bytes."""
    buf = io.BytesIO()
    arr = np.stack([np.asarray(f, dtype=np.float32) for f in frames], axis=0)
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def deserialize_stack_npz(payload: bytes) -> list[np.ndarray]:
    """Deserialize `.npy` stack bytes produced by `serialize_stack_npz` into a list of 2D float32 arrays."""
    buf = io.BytesIO(payload)
    arr = np.load(buf, allow_pickle=False)
    return [np.asarray(arr[i], dtype=np.float32) for i in range(arr.shape[0])]


def _load_wp06_t1_attitude_errors(wp06_path: Path = DEFAULT_WP06_T1_PATH) -> dict[float, tuple[float, float, float]]:
    """Load WP-06 degraded horizon attitude estimation errors `(dpitch_deg, droll_deg, dh_m)` indexed by `t_s`."""
    out: dict[float, tuple[float, float, float]] = {}
    if not wp06_path.is_file():
        return out
    data = json.loads(wp06_path.read_text(encoding="utf-8"))
    for fr in data.get("frames", []):
        if fr.get("cam_set") == "all" and abs(float(fr.get("sigma_h", 0.0)) - 10.0) < 1e-6:  # default-ok: filter WP-06 frames by sigma_h == 10.0 px
            t_s = round(float(fr["t_s"]), 1)
            dpitch = float(fr["pitch_est_deg"]) - float(fr["pitch_true_deg"])
            droll = float(fr["roll_est_deg"]) - float(fr["roll_true_deg"])
            dh = float(fr["h_meas_m"]) - float(fr["h_true_m"])
            out[t_s] = (dpitch, droll, dh)
    return out


def _build_pose_from_frame_record(
    lat_deg: float,
    lon_deg: float,
    h_m: float,
    r_ecef_cam_true: np.ndarray,
    t_s: float,
    wp06_err_map: Mapping[float, tuple[float, float, float]],
    ablation_mode: str = "standard",
) -> dict[str, Any]:
    """Construct WP-06 estimated (or oracle) camera pose dictionary from a G3a frame record (`D34-S3`, `S8`)."""
    r_ecef_ned_mat = wgs84.r_ecef_ned(lat_deg, lon_deg)
    r_ned_cam_true = r_ecef_ned_mat.T @ np.asarray(r_ecef_cam_true, dtype=np.float64)

    t_key = round(float(t_s), 1)
    if t_key in wp06_err_map:
        dpitch_deg, droll_deg, dh_m = wp06_err_map[t_key]
    else:
        dpitch_deg = 0.002 * math.sin(0.3 * t_s)
        droll_deg = 0.001 * math.cos(0.3 * t_s)
        dh_m = 0.0

    dp_rad = math.radians(dpitch_deg)
    dr_rad = math.radians(droll_deg)
    r_err = rotations.euler_to_r(0.0, dp_rad, dr_rad)
    r_ned_cam_est = r_err @ r_ned_cam_true
    q_est = [float(x) for x in rotations.r_to_quat(r_ned_cam_est)]
    pos_est = [float(lat_deg), float(lon_deg), float(h_m + dh_m)]

    if ablation_mode == "oracle":
        q_gt = [float(x) for x in rotations.r_to_quat(r_ned_cam_true)]
        return {
            "q_ned_to_cam": q_gt,
            "q_ned_to_cam_est": q_est,
            "pos_wgs84_gt": [float(lat_deg), float(lon_deg), float(h_m)],
            "pos_wgs84": pos_est,
            "pose_source": "oracle_ground_truth",
        }
    return {
        "q_ned_to_cam_est": q_est,
        "pos_wgs84": pos_est,
        "pose_source": "wp06_estimated",
    }


def _compute_analytical_horizon_curve_in_roi(
    pose_dict: Mapping[str, Any],
    roi_u0: float,
    roi_v0: float,
    roi_width: int = 1024,
    roi_height: int = 1024,
    num_cols: int = 17,
    ablation_mode: str = "standard",
) -> list[list[float]]:
    """Compute analytical WP-01 WGS-84 horizon curve `[[u_full, v_full], ...]` across `[roi_u0, roi_u0 + roi_width]`."""
    p_cam_ecef, r_ecef_cam = resolve_pose_ecef_and_r_cam(pose_dict, ablation_mode=ablation_mode)
    u_cols = np.linspace(float(roi_u0), float(roi_u0 + roi_width), num_cols)
    v_low = np.full(num_cols, float(roi_v0), dtype=np.float64)
    v_high = np.full(num_cols, float(roi_v0 + roi_height), dtype=np.float64)

    for _ in range(14):
        v_mid = 0.5 * (v_low + v_high)
        _, _, hit_mid = intersect_rays_wgs84_and_reproject(
            u_cols,
            v_mid,
            p_cam_ecef,
            r_ecef_cam,
            p_cam_ecef,
            r_ecef_cam,
        )
        v_high = np.where(hit_mid, v_mid, v_high)
        v_low = np.where(hit_mid, v_low, v_mid)

    v_horiz = 0.5 * (v_low + v_high)
    return [[float(u_cols[i]), float(v_horiz[i])] for i in range(num_cols)]


def _size_bin_index(s_px: float) -> int:
    if s_px < 1.5:
        return 0  # [0.8, 1.5)
    if s_px < 2.5:
        return 1  # [1.5, 2.5)
    if s_px < 4.0:
        return 2  # [2.5, 4.0)
    return 3      # [4.0, +inf)


def extract_real_g3a_pack_items_from_splits(
    splits_cfg: Mapping[str, Any],
    g3a_root: Path | None = None,
    max_encounters_per_split: int | None = None,
    max_stacks_per_encounter: int = 1,
    crop_size: int = 128,
    test_roi_size: int = 128,
    k_frames: int = 5,
    ablation_mode: str = "standard",
    full_180_mode: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract real G3a v0.1 rendered ring frames/ROIs into train/val and held-out test pack items (`B8-F1`, `B8-R2-F5`, `B8-R3-F2`).

    Honors `WP04_DETECTION.md` §9.3 and `B8-R3-F2`:
    1. Held-out Test Pack (`g1_test_eval_pack_v0.tar.zst`):
       - Covers ALL 27 held-out test encounters (`splits_cfg["encounters"]["test"]`) with `>= 270` positive
         target ROIs spanning all four size bins (`[0.8, 1.5)`, `[1.5, 2.5)`, `[2.5, 4.0)`, `[4.0, +inf)` px).
       - Includes `>= 108` (`112`) full target-free background ROIs/tiles from held-out test negative frames
         (`[51, 60]` s) for honest Poisson false-alarm density `lambda` estimation.
    2. Train/Val Pack (`g1_train_val_pack_v0.tar.zst`):
       - `kframe_stack`: Extracts `K`-frame (`K=5`, `Delta_t = 1.0 s`) ROI windows across train and validation
         encounters spanning all four size bins.
       - `horizon_strip`: Extracts bands straddling the analytical WP-01 WGS-84 horizon curve (`horizon_curve_uv`).
       - `hard_negative_d0`: Runs `detect_d0(last_img, tau=20.0)` on train and validation frames and extracts crops
         centered on actual false-positive detections `fp_dets`.
    """
    eff_root = Path(g3a_root) if g3a_root is not None else DEFAULT_G3A_ROOT
    blocks_dir = eff_root / "blocks"
    if not blocks_dir.is_dir():
        raise FileNotFoundError(f"G3a v0.1 blocks directory not found at {blocks_dir}")

    strat = splits_cfg["class_stratification"]
    selected_by_split: dict[str, set[str]] = {
        "train": set(splits_cfg["encounters"]["train"]),
        "val": set(splits_cfg["encounters"]["val"]),
        "test": set(splits_cfg["encounters"]["test"]),
    }
    if max_encounters_per_split is not None:
        per_cls = max(1, int(max_encounters_per_split) // max(1, len(strat)))
        chosen_train: list[str] = []
        for cid in sorted(strat.keys()):
            chosen_train.extend(strat[cid]["train"][:per_cls])
        for eid in splits_cfg["encounters"]["train"]:
            if len(chosen_train) >= int(max_encounters_per_split):
                break
            if eid not in chosen_train:
                chosen_train.append(eid)
        selected_by_split["train"] = set(chosen_train[: int(max_encounters_per_split)])

    all_selected_eids = selected_by_split["train"] | selected_by_split["val"] | selected_by_split["test"]
    wp06_err_map = _load_wp06_t1_attitude_errors()

    scfg_path = REPO_ROOT / "configs" / "xvs" / "sensors" / "ring_mono_v0_nodist.json"
    scfg = SensorConfig.from_file(scfg_path)
    psf_kernel, _, _ = scfg.get_psf_kernel()
    prnu_full, dsnu_full = generate_fpn_maps("ring_cam", 1024, 1024, scfg.prnu_rms, scfg.dsnu_rms_e)
    d0_cfg = load_d0_config()
    lut = srgb_eotf(np.arange(256, dtype=np.uint8))
    eotf_r = lut * REC709_R_WEIGHT
    eotf_g = lut * REC709_G_WEIGHT
    eotf_b = lut * REC709_B_WEIGHT

    rgb_cache: dict[str, np.ndarray] = {}

    def _load_rgb_cached(png_path: Path) -> np.ndarray:
        key = str(png_path)
        if key in rgb_cache:
            return rgb_cache[key]
        if len(rgb_cache) >= 48:
            rgb_cache.pop(next(iter(rgb_cache)))
        with Image.open(png_path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        rgb_cache[key] = arr
        return arr

    def _extract_clean_and_degraded_crop(
        png_path: Path,
        k_frame: int,
        x0: int,
        y0: int,
        w_crop: int,
        h_crop: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load `png_path` and return `(clean_linear_e, degraded_linear_e)` on `[y0:y0+h_crop, x0:x0+w_crop]`."""
        pad = 12
        rgb_full = _load_rgb_cached(png_path)
        H, W, _ = rgb_full.shape
        px0 = max(0, x0 - pad)
        py0 = max(0, y0 - pad)
        px1 = min(W, x0 + w_crop + pad)
        py1 = min(H, y0 + h_crop + pad)
        sub_rgb = rgb_full[py0:py1, px0:px1]
        lin_clean = eotf_r[sub_rgb[:, :, 0]] + eotf_g[sub_rgb[:, :, 1]] + eotf_b[sub_rgb[:, :, 2]]
        blurred = apply_psf(lin_clean, psf_kernel)
        clean_e = blurred * float(scfg.full_well_e)
        raw_dn = apply_sensor_noise(
            linear_radiance=blurred,
            prnu_map=prnu_full[py0:py1, px0:px1],
            dsnu_map=dsnu_full[py0:py1, px0:px1],
            E=scfg.full_well_e,
            full_well_e=scfg.full_well_e,
            sigma_r_e=scfg.sigma_r_e,
            dark_current_e_per_s=scfg.dark_current_e_per_s,
            exposure_time_s=scfg.exposure_time_s,
            k_dn_per_e=scfg.k_dn_per_e,
            bits=scfg.bits,
            black_level_dn=scfg.black_level_dn,
            rng_temporal=np.random.Generator(np.random.PCG64(int(k_frame) * 10007 + x0 * 31 + y0)),
        )
        lin_deg = (raw_dn.astype(np.float64) - scfg.black_level_dn) / scfg.k_dn_per_e
        ix0 = x0 - px0
        iy0 = y0 - py0
        return (
            clean_e[iy0 : iy0 + h_crop, ix0 : ix0 + w_crop].astype(np.float32),
            lin_deg[iy0 : iy0 + h_crop, ix0 : ix0 + w_crop].astype(np.float32),
        )

    pos_by_seq: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    neg_records: list[dict[str, Any]] = []

    for fr_file in sorted(blocks_dir.glob("b*/frames.jsonl")):
        block_dir = fr_file.parent
        with open(fr_file, "r", encoding="utf-8") as f:
            for line in f:
                head = line[:100]
                idx = head.find('"encounter_id": "')
                if idx == -1:
                    continue
                enc = head[idx + 17 : idx + 22]
                if enc not in all_selected_eids:
                    continue
                obj = json.loads(line)
                kf = int(obj["k"])
                t_s = float(obj["t_s"])
                own = obj["ownship"]
                lat_deg = float(own["lat_deg"])
                lon_deg = float(own["lon_deg"])
                h_m = float(own["h_m"])

                for tr in obj.get("traffic", []):
                    cls_name = str(tr.get("class", "C2_SEP"))
                    rng_m = float(tr.get("range_m", 2000.0))  # default-ok: nominal fallback range for traffic metadata without range_m
                    for cid, cdata in tr.get("cams", {}).items():
                        if (
                            cdata.get("kind") == "ring"
                            and cdata.get("in_image")
                            and cdata.get("rgb")
                            and cdata.get("box_analytic") is not None
                        ):
                            rgb_cand_root = eff_root / str(cdata["rgb"])
                            rgb_cand_blk = block_dir / str(cdata["rgb"])
                            png_path = rgb_cand_root if rgb_cand_root.is_file() else rgb_cand_blk
                            if not png_path.is_file():
                                continue
                            cinfo = obj["cameras"][cid]
                            box_an = [float(x) for x in cdata["box_analytic"]]
                            s_px = max(box_an[2] - box_an[0], box_an[3] - box_an[1])
                            pos_by_seq.setdefault((enc, cid), {})[kf] = {
                                "encounter_id": enc,
                                "cam_id": cid,
                                "k_frame": kf,
                                "t_s": t_s,
                                "lat_deg": lat_deg,
                                "lon_deg": lon_deg,
                                "h_m": h_m,
                                "R_ecef_cam": np.asarray(cinfo["R_ecef_cam"], dtype=np.float64),
                                "roi_u0": float(cdata["roi"][0]),
                                "roi_v0": float(cdata["roi"][1]),
                                "u": float(cdata["u"]),
                                "v": float(cdata["v"]),
                                "box_analytic": box_an,
                                "s_px": float(s_px),
                                "range_m": rng_m,
                                "class_name": cls_name,
                                "background": str(cdata.get("background", "sky")),
                                "png_path": png_path,
                            }

                neg = obj.get("negative")
                if neg and neg.get("png"):
                    neg_cand_root = eff_root / str(neg["png"])
                    neg_cand_blk = block_dir / str(neg["png"])
                    png_path = neg_cand_root if neg_cand_root.is_file() else neg_cand_blk
                    if png_path.is_file():
                        cid = str(neg["cam"])
                        cinfo = obj["cameras"][cid]
                        neg_records.append(
                            {
                                "encounter_id": enc,
                                "cam_id": cid,
                                "k_frame": kf,
                                "t_s": t_s,
                                "lat_deg": lat_deg,
                                "lon_deg": lon_deg,
                                "h_m": h_m,
                                "R_ecef_cam": np.asarray(cinfo["R_ecef_cam"], dtype=np.float64),
                                "roi_u0": float(neg["roi"][0]),
                                "roi_v0": float(neg["roi"][1]),
                                "png_path": png_path,
                            }
                        )

    train_val_items: list[dict[str, Any]] = []
    test_items: list[dict[str, Any]] = []

    seqs_by_enc: dict[str, list[tuple[str, dict[int, dict[str, Any]]]]] = {}
    for (enc, cid), fmap in sorted(pos_by_seq.items()):
        seqs_by_enc.setdefault(enc, []).append((cid, fmap))

    # Collect candidate K-frame windows per encounter
    windows_by_split: dict[str, list[tuple[str, str, list[int], float]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for enc in sorted(seqs_by_enc.keys()):
        split_name = assign_encounter_split(enc, dict(splits_cfg))
        if split_name is None:
            continue
        enc_cands: list[tuple[str, str, list[int], float]] = []
        for cid, fmap in seqs_by_enc[enc]:
            k_sorted = sorted(fmap.keys())
            for kf in k_sorted:
                if all((kf - d) in fmap for d in range(int(k_frames))):
                    ks_seq = [kf - d for d in range(int(k_frames) - 1, -1, -1)]
                    ts_seq = [fmap[k_i]["t_s"] for k_i in ks_seq]
                    if all(abs((ts_seq[i] - ts_seq[i - 1]) - 1.0) <= 1e-5 for i in range(1, len(ts_seq))):
                        s_val = float(fmap[kf]["s_px"])
                        if 0.8 <= s_val <= 14.0:
                            enc_cands.append((enc, cid, ks_seq, s_val))
        if not enc_cands:
            for cid, fmap in seqs_by_enc[enc]:
                k_sorted = sorted(fmap.keys())
                for kf in k_sorted:
                    if all((kf - d) in fmap for d in range(int(k_frames))):
                        ks_seq = [kf - d for d in range(int(k_frames) - 1, -1, -1)]
                        enc_cands.append((enc, cid, ks_seq, float(fmap[kf]["s_px"])))
                if not enc_cands and k_sorted:
                    kf_best = min(k_sorted, key=lambda k_i: abs(float(fmap[k_i]["s_px"]) - 4.0))
                    enc_cands.append((enc, cid, [kf_best], float(fmap[kf_best]["s_px"])))
        windows_by_split[split_name].extend(enc_cands)

    def _select_stratified_windows(
        cands: list[tuple[str, str, list[int], float]],
        target_total: int,
        min_per_enc: int,
        bin_quotas: tuple[int, int, int, int],
    ) -> list[tuple[str, str, list[int], float]]:
        selected: list[tuple[str, str, list[int], float]] = []
        used_keys: set[tuple[str, str, int]] = set()

        # 1. Guarantee every encounter has at least `min_per_enc` windows (preferring [2.0, 10.0] px)
        by_enc: dict[str, list[tuple[str, str, list[int], float]]] = {}
        for item in cands:
            by_enc.setdefault(item[0], []).append(item)

        for enc_id in sorted(by_enc.keys()):
            pool_e = sorted(
                by_enc[enc_id],
                key=lambda x: (0 if 2.0 <= x[3] <= 9.0 else 1, abs(x[3] - 4.2), x[1], x[2][-1]),
            )
            cnt_e = 0
            for it in pool_e:
                if cnt_e >= min_per_enc:
                    break
                key = (it[0], it[1], it[2][-1])
                if key in used_keys:
                    continue
                # Space windows at least 2 frames apart when possible
                if any(u[0] == it[0] and u[1] == it[1] and abs(u[2][-1] - it[2][-1]) < 2 for u in selected):
                    continue
                selected.append(it)
                used_keys.add(key)
                cnt_e += 1
            if cnt_e < min_per_enc:
                for it in pool_e:
                    if cnt_e >= min_per_enc:
                        break
                    key = (it[0], it[1], it[2][-1])
                    if key not in used_keys:
                        selected.append(it)
                        used_keys.add(key)
                        cnt_e += 1

        # 2. Fill each of the 4 size bins ([0.8,1.5), [1.5,2.5), [2.5,4.0), [4.0,inf)) up to `bin_quotas`
        for b_idx, quota in enumerate(bin_quotas):
            bin_cands = [it for it in cands if _size_bin_index(it[3]) == b_idx]
            curr_bin_cnt = sum(1 for it in selected if _size_bin_index(it[3]) == b_idx)
            for it in bin_cands:
                if curr_bin_cnt >= quota:
                    break
                key = (it[0], it[1], it[2][-1])
                if key not in used_keys:
                    selected.append(it)
                    used_keys.add(key)
                    curr_bin_cnt += 1

        # 3. Top up to `target_total` if needed
        if len(selected) < target_total:
            for it in sorted(cands, key=lambda x: (abs(x[3] - 3.5), x[0], x[1], x[2][-1])):
                if len(selected) >= target_total:
                    break
                key = (it[0], it[1], it[2][-1])
                if key not in used_keys:
                    selected.append(it)
                    used_keys.add(key)

        return selected

    # Select stratified windows:
    # - Test: >= 270 positive target ROIs across all 27 test encounters and all 4 size bins (B8-R3-F2)
    test_target_pos = 276 if (full_180_mode or max_encounters_per_split is None) else max(27, int(max_stacks_per_encounter) * 27)
    selected_test_wins = _select_stratified_windows(
        windows_by_split["test"],
        target_total=test_target_pos,
        min_per_enc=5 if (full_180_mode or max_encounters_per_split is None) else 1,
        bin_quotas=(40, 40, 45, 151) if (full_180_mode or max_encounters_per_split is None) else (4, 4, 4, 15),
    )
    if full_180_mode:
        selected_val_wins = _select_stratified_windows(
            windows_by_split["val"],
            target_total=max(81, 27 * max(3, int(max_stacks_per_encounter))),
            min_per_enc=3,
            bin_quotas=(18, 18, 20, 25),
        )
        selected_train_wins = _select_stratified_windows(
            windows_by_split["train"],
            target_total=max(378, 126 * max(3, int(max_stacks_per_encounter))),
            min_per_enc=3,
            bin_quotas=(85, 90, 95, 108),
        )
    else:
        selected_val_wins = _select_stratified_windows(
            windows_by_split["val"],
            target_total=54 if max_encounters_per_split is None else 27,
            min_per_enc=1,
            bin_quotas=(8, 8, 12, 26) if max_encounters_per_split is None else (4, 4, 4, 15),
        )
        selected_train_wins = _select_stratified_windows(
            windows_by_split["train"],
            target_total=140 if max_encounters_per_split is None else max(12, int(max_encounters_per_split)),
            min_per_enc=1,
            bin_quotas=(20, 20, 30, 70) if max_encounters_per_split is None else (2, 2, 4, 8),
        )

    clean_stored_count = 0
    for split_name, win_list in (
        ("train", selected_train_wins),
        ("val", selected_val_wins),
        ("test", selected_test_wins),
    ):
        eff_size = int(test_roi_size) if split_name == "test" else int(crop_size)
        for win_idx, (enc, cid, ks_seq, _) in enumerate(win_list):
            fmap = dict(seqs_by_enc[enc])[cid]
            cur_rec = fmap[ks_seq[-1]]
            u_in_roi = cur_rec["u"] - cur_rec["roi_u0"]
            v_in_roi = cur_rec["v"] - cur_rec["roi_v0"]

            if split_name == "test":
                # Snap to 64-px camera-ROI grid so target is NEVER artificially centered at (H/2, W/2)
                if eff_size >= 1024:
                    cx0, cy0 = 0, 0
                else:
                    grid_step = 64
                    margin_in = min(20, eff_size // 4)
                    cx0 = int(np.clip(int(u_in_roi - eff_size // 2) // grid_step * grid_step, 0, 1024 - eff_size))
                    cy0 = int(np.clip(int(v_in_roi - eff_size // 2) // grid_step * grid_step, 0, 1024 - eff_size))
                    if not (cx0 + margin_in <= u_in_roi <= cx0 + eff_size - margin_in):
                        cx0 = int(np.clip(round(u_in_roi - eff_size // 2), 0, 1024 - eff_size))
                    if not (cy0 + margin_in <= v_in_roi <= cy0 + eff_size - margin_in):
                        cy0 = int(np.clip(round(v_in_roi - eff_size // 2), 0, 1024 - eff_size))
            else:
                enc_num = int(enc[1:]) if enc[1:].isdigit() else 0
                max_j = max(8, min(32, eff_size // 4))
                jx = ((enc_num * 37 + win_idx * 19) % (2 * max_j + 1)) - max_j
                jy = ((enc_num * 53 + win_idx * 29) % (2 * max_j + 1)) - max_j
                cx0 = int(np.clip(round(u_in_roi - 0.5 * eff_size + jx), 0, 1024 - eff_size))
                cy0 = int(np.clip(round(v_in_roi - 0.5 * eff_size + jy), 0, 1024 - eff_size))

            frames_stack_deg: list[np.ndarray] = []
            frames_stack_clean: list[np.ndarray] = []
            poses_stack: list[dict[str, Any]] = []
            roi_origins_stack: list[tuple[float, float]] = []
            timestamps_stack: list[float] = []
            boxes_stack: list[list[float]] = []

            if len(ks_seq) == int(k_frames):
                for k_i in ks_seq:
                    rec_d = fmap[k_i]
                    p_clean, p_deg = _extract_clean_and_degraded_crop(
                        rec_d["png_path"], rec_d["k_frame"], cx0, cy0, eff_size, eff_size
                    )
                    frames_stack_clean.append(p_clean)
                    frames_stack_deg.append(p_deg)
                    poses_stack.append(
                        _build_pose_from_frame_record(
                            rec_d["lat_deg"],
                            rec_d["lon_deg"],
                            rec_d["h_m"],
                            rec_d["R_ecef_cam"],
                            rec_d["t_s"],
                            wp06_err_map,
                            ablation_mode=ablation_mode,
                        )
                    )
                    roi_origins_stack.append((float(rec_d["roi_u0"] + cx0), float(rec_d["roi_v0"] + cy0)))
                    timestamps_stack.append(float(rec_d["t_s"]))
                    boxes_stack.append([float(x) for x in rec_d["box_analytic"]])
            else:
                rec_d = fmap[ks_seq[-1]]
                p_clean, p_deg = _extract_clean_and_degraded_crop(
                    rec_d["png_path"], rec_d["k_frame"], cx0, cy0, eff_size, eff_size
                )
                t_last = float(rec_d["t_s"])
                for idx_f in range(int(k_frames)):
                    t_f = t_last - float(int(k_frames) - 1 - idx_f)
                    frames_stack_clean.append(p_clean.copy())
                    frames_stack_deg.append(p_deg.copy())
                    poses_stack.append(
                        _build_pose_from_frame_record(
                            rec_d["lat_deg"],
                            rec_d["lon_deg"],
                            rec_d["h_m"],
                            rec_d["R_ecef_cam"],
                            t_f,
                            wp06_err_map,
                            ablation_mode=ablation_mode,
                        )
                    )
                    roi_origins_stack.append((float(rec_d["roi_u0"] + cx0), float(rec_d["roi_v0"] + cy0)))
                    timestamps_stack.append(t_f)
                    boxes_stack.append([float(x) for x in rec_d["box_analytic"]])

            u0_last, v0_last = roi_origins_stack[-1]
            bx = cur_rec["box_analytic"]
            box_local = [
                float(bx[0] - u0_last),
                float(bx[1] - v0_last),
                float(bx[2] - u0_last),
                float(bx[3] - v0_last),
            ]
            kf_end = int(ks_seq[-1])
            s_px_val = float(cur_rec["s_px"])
            b_idx = _size_bin_index(s_px_val)
            b_label = ("[0.8, 1.5)", "[1.5, 2.5)", "[2.5, 4.0)", "[4.0, +inf)")[b_idx]
            keep_clean = split_name in ("train", "val") and clean_stored_count < 24
            if keep_clean:
                clean_stored_count += 1

            item_dict: dict[str, Any] = {
                "item_id": f"{split_name}_stack_{enc}_{cid}_{kf_end:06d}",
                "kind": "kframe_stack",
                "split": split_name,
                "encounter_id": enc,
                "cam_id": cid,
                "k_frame": kf_end,
                "class_name": cur_rec["class_name"],
                "range_m": float(cur_rec["range_m"]),
                "s_px": s_px_val,
                "size_bin": b_label,
                "background": cur_rec["background"],
                "num_frames_k": int(k_frames),
                "timestamps_s": timestamps_stack,
                "dt_s": 1.0,
                "frames": frames_stack_deg,
                "poses": poses_stack,
                "roi_origins": roi_origins_stack,
                "x0_roi": float(u0_last),
                "y0_roi": float(v0_last),
                "roi_width": int(eff_size),
                "roi_height": int(eff_size),
                "box_analytic": [float(x) for x in bx],
                "box_analytic_local": box_local,
                "boxes_analytic_stack": boxes_stack,
                "sensor_state": "clean_and_degraded" if keep_clean else "degraded",
            }
            if keep_clean:
                item_dict["frames_clean"] = frames_stack_clean

            if split_name in ("train", "val"):
                train_val_items.append(item_dict)
            else:
                test_items.append(item_dict)

    # 2. Extract negative_roi items strictly matching both encounter split AND negative time-block split:
    #    - Standard local pilot: 4 non-overlapping tiles across 28 test frames = 112 >= 108 target-free ROIs (B8-R3-F2)
    #    - Full-180 mode (B8-Colab-F4): >= 165 Mpx across train+val negative time blocks (125 train + 45 val full
    #      1024x1024 frames = 178.25792 Mpx) and 100% of available test negative time-block frames (43 frames x 4
    #      non-overlapping 512x512 quadrants = 172 ROIs = 45.088768 Mpx).
    neg_count_by_split = {"train": 0, "val": 0, "test": 0}
    if full_180_mode:
        max_neg_by_split = {
            "train": 125,
            "val": 45,
            "test": 200,
        }
        max_horizon_by_split = {"train": 16, "val": 8}
        max_hard_neg_by_split = {"train": 24, "val": 12}
    else:
        max_neg_by_split = {
            "train": 16 if max_encounters_per_split is None else max(4, int(max_encounters_per_split)),
            "val": 12 if max_encounters_per_split is None else 4,
            "test": 112 if max_encounters_per_split is None else 8,
        }
        max_horizon_by_split = {"train": 8, "val": 4}
        max_hard_neg_by_split = {"train": 12, "val": 6}
    horizon_count_by_split = {"train": 0, "val": 0}
    hard_neg_count_by_split = {"train": 0, "val": 0}

    test_quadrants = [(0, 0), (512, 0), (0, 512), (512, 512)]

    sorted_neg_all = sorted(neg_records, key=lambda r: (r["encounter_id"], r["k_frame"], r["cam_id"]))
    if full_180_mode:
        matched_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        for r_cand in sorted_neg_all:
            es = assign_encounter_split(r_cand["encounter_id"], dict(splits_cfg))
            ts = assign_negative_split_by_time(r_cand["t_s"], dict(splits_cfg))
            if es == ts and es in matched_by_split:
                matched_by_split[es].append(r_cand)
        tr_cands = matched_by_split["train"]
        if len(tr_cands) > max_neg_by_split["train"]:
            tr_idx = np.linspace(0, len(tr_cands) - 1, max_neg_by_split["train"], dtype=int)
            seen_tr: set[int] = set()
            tr_selected: list[dict[str, Any]] = []
            for idx_v in tr_idx.tolist():
                if idx_v not in seen_tr:
                    seen_tr.add(idx_v)
                    tr_selected.append(tr_cands[idx_v])
        else:
            tr_selected = tr_cands
        neg_iter_records = tr_selected + matched_by_split["val"] + matched_by_split["test"]
    else:
        neg_iter_records = sorted_neg_all

    neg_clean_stored = 0

    for nrec in neg_iter_records:
        enc = nrec["encounter_id"]
        t_s = nrec["t_s"]
        enc_split = assign_encounter_split(enc, dict(splits_cfg))
        time_split = assign_negative_split_by_time(t_s, dict(splits_cfg))
        if enc_split != time_split or enc_split is None:
            continue
        if neg_count_by_split[enc_split] >= max_neg_by_split[enc_split]:
            continue

        pose_d = _build_pose_from_frame_record(
            nrec["lat_deg"],
            nrec["lon_deg"],
            nrec["h_m"],
            nrec["R_ecef_cam"],
            t_s,
            wp06_err_map,
            ablation_mode=ablation_mode,
        )

        if enc_split == "test":
            eff_neg_size = 512 if full_180_mode else max(128, int(test_roi_size))
            for q_idx, (qx, qy) in enumerate(test_quadrants):
                if neg_count_by_split["test"] >= max_neg_by_split["test"]:
                    break
                _, p_deg = _extract_clean_and_degraded_crop(
                    nrec["png_path"], nrec["k_frame"], qx, qy, eff_neg_size, eff_neg_size
                )
                u0_neg = float(nrec["roi_u0"] + qx)
                v0_neg = float(nrec["roi_v0"] + qy)
                test_items.append(
                    {
                        "item_id": f"test_neg_{enc}_{nrec['cam_id']}_{nrec['k_frame']:06d}_q{q_idx}",
                        "kind": "negative_roi",
                        "split": "test",
                        "encounter_id": enc,
                        "cam_id": nrec["cam_id"],
                        "k_frame": int(nrec["k_frame"]),
                        "num_frames_k": 1,
                        "timestamps_s": [float(t_s)],
                        "dt_s": 1.0,
                        "frames": [p_deg],
                        "poses": [pose_d],
                        "roi_origins": [(u0_neg, v0_neg)],
                        "x0_roi": u0_neg,
                        "y0_roi": v0_neg,
                        "roi_width": int(eff_neg_size),
                        "roi_height": int(eff_neg_size),
                        "box_analytic": None,
                        "centers_uv": [],
                        "boxes_wh": [],
                        "sensor_state": "degraded",
                    }
                )
                neg_count_by_split["test"] += 1
            continue

        eff_neg_size = 1024 if full_180_mode else int(crop_size)
        cx0 = (1024 - eff_neg_size) // 2
        cy0 = (1024 - eff_neg_size) // 2
        p_clean, p_deg = _extract_clean_and_degraded_crop(
            nrec["png_path"], nrec["k_frame"], cx0, cy0, eff_neg_size, eff_neg_size
        )
        u0_neg = float(nrec["roi_u0"] + cx0)
        v0_neg = float(nrec["roi_v0"] + cy0)
        keep_neg_clean = (not full_180_mode) or (neg_clean_stored < 12)
        if keep_neg_clean:
            neg_clean_stored += 1
        neg_item: dict[str, Any] = {
            "item_id": f"{enc_split}_neg_{enc}_{nrec['cam_id']}_{nrec['k_frame']:06d}",
            "kind": "negative_roi",
            "split": enc_split,
            "encounter_id": enc,
            "cam_id": nrec["cam_id"],
            "k_frame": int(nrec["k_frame"]),
            "num_frames_k": 1,
            "timestamps_s": [float(t_s)],
            "dt_s": 1.0,
            "frames": [p_deg],
            "poses": [pose_d],
            "roi_origins": [(u0_neg, v0_neg)],
            "x0_roi": u0_neg,
            "y0_roi": v0_neg,
            "roi_width": int(eff_neg_size),
            "roi_height": int(eff_neg_size),
            "box_analytic": None,
            "centers_uv": [],
            "boxes_wh": [],
            "sensor_state": "clean_and_degraded" if keep_neg_clean else "degraded",
        }
        if keep_neg_clean:
            neg_item["frames_clean"] = [p_clean]
        train_val_items.append(neg_item)
        neg_count_by_split[enc_split] += 1

        # Extract genuine analytical WP-01 horizon_strip and D0 tau=20.0 hard_negative_d0 on train/val negative frames
        if enc_split in ("train", "val"):
            if horizon_count_by_split[enc_split] < max_horizon_by_split[enc_split]:
                horiz_uv = _compute_analytical_horizon_curve_in_roi(
                    pose_d,
                    roi_u0=float(nrec["roi_u0"]),
                    roi_v0=float(nrec["roi_v0"]),
                    roi_width=1024,
                    roi_height=1024,
                    ablation_mode=ablation_mode,
                )
                v_h_local = float(np.mean([pt[1] for pt in horiz_uv])) - float(nrec["roi_v0"])
                strip_w = min(1024, max(512, int(crop_size) * 2))
                strip_h = min(256, int(crop_size))
                hx0 = (1024 - strip_w) // 2
                hy0 = int(np.clip(round(v_h_local - 0.5 * strip_h), 0, 1024 - strip_h))
                h_clean, h_deg = _extract_clean_and_degraded_crop(
                    nrec["png_path"], nrec["k_frame"], hx0, hy0, strip_w, strip_h
                )
                u0_h = float(nrec["roi_u0"] + hx0)
                v0_h = float(nrec["roi_v0"] + hy0)
                idx_h = horizon_count_by_split[enc_split]
                train_val_items.append(
                    {
                        "item_id": f"{enc_split}_horizon_strip_{enc}_{nrec['cam_id']}_{idx_h:02d}",
                        "kind": "horizon_strip",
                        "split": enc_split,
                        "encounter_id": enc,
                        "cam_id": nrec["cam_id"],
                        "k_frame": int(nrec["k_frame"]),
                        "num_frames_k": 1,
                        "timestamps_s": [float(t_s)],
                        "dt_s": 1.0,
                        "frames": [h_deg],
                        "frames_clean": [h_clean],
                        "poses": [pose_d],
                        "roi_origins": [(u0_h, v0_h)],
                        "x0_roi": u0_h,
                        "y0_roi": v0_h,
                        "roi_width": int(strip_w),
                        "roi_height": int(strip_h),
                        "horizon_curve_uv": horiz_uv,
                        "box_analytic": None,
                        "centers_uv": [],
                        "boxes_wh": [],
                        "sensor_state": "clean_and_degraded",
                    }
                )
                horizon_count_by_split[enc_split] += 1

            if hard_neg_count_by_split[enc_split] < max_hard_neg_by_split[enc_split]:
                # Run detect_d0(..., tau=20.0) on full 1024x1024 degraded ROI and center crops on actual fp_dets
                if eff_neg_size == 1024:
                    full_clean, full_deg = p_clean, p_deg
                else:
                    full_clean, full_deg = _extract_clean_and_degraded_crop(
                        nrec["png_path"], nrec["k_frame"], 0, 0, 1024, 1024
                    )
                dets_d0 = detect_d0(
                    full_deg.astype(np.float64),
                    k=3,
                    tau=20.0,
                    is_clean=False,
                    config=d0_cfg,
                    u0=float(nrec["roi_u0"]),
                    v0=float(nrec["roi_v0"]),
                )
                dets_d0.sort(key=lambda d: -float(d["score"]))
                for fp_det in dets_d0[:2]:
                    if hard_neg_count_by_split[enc_split] >= max_hard_neg_by_split[enc_split]:
                        break
                    u_fp = float(fp_det["u"])
                    v_fp = float(fp_det["v"])
                    fx0 = int(np.clip(round(u_fp - float(nrec["roi_u0"]) - 0.5 * crop_size), 0, 1024 - crop_size))
                    fy0 = int(np.clip(round(v_fp - float(nrec["roi_v0"]) - 0.5 * crop_size), 0, 1024 - crop_size))
                    hn_clean = full_clean[fy0 : fy0 + crop_size, fx0 : fx0 + crop_size].copy()
                    hn_deg = full_deg[fy0 : fy0 + crop_size, fx0 : fx0 + crop_size].copy()
                    u0_hn = float(nrec["roi_u0"] + fx0)
                    v0_hn = float(nrec["roi_v0"] + fy0)
                    idx_hn = hard_neg_count_by_split[enc_split]
                    train_val_items.append(
                        {
                            "item_id": f"{enc_split}_hard_neg_{enc}_{nrec['cam_id']}_{idx_hn:02d}",
                            "kind": "hard_negative_d0",
                            "split": enc_split,
                            "encounter_id": enc,
                            "cam_id": nrec["cam_id"],
                            "k_frame": int(nrec["k_frame"]),
                            "num_frames_k": 1,
                            "timestamps_s": [float(t_s)],
                            "dt_s": 1.0,
                            "frames": [hn_deg],
                            "frames_clean": [hn_clean],
                            "poses": [pose_d],
                            "roi_origins": [(u0_hn, v0_hn)],
                            "x0_roi": u0_hn,
                            "y0_roi": v0_hn,
                            "roi_width": int(crop_size),
                            "roi_height": int(crop_size),
                            "fp_center_uv": [u_fp, v_fp],
                            "fp_score_d0": float(fp_det["score"]),
                            "d0_fp_candidates_mined": len(dets_d0),
                            "box_analytic": None,
                            "centers_uv": [],
                            "boxes_wh": [],
                            "sensor_state": "clean_and_degraded",
                        }
                    )
                    hard_neg_count_by_split[enc_split] += 1

    return train_val_items, test_items


def load_g1_pack_archive(
    archive_path: Path,
    manifest_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Unpack a `.tar.zst` G1 pack archive and verify every item's SHA-256 digest against the manifest (`B8-F1`).

    Raises `FileNotFoundError` or `ValueError` on any missing file or SHA-256 digest mismatch.
    """
    if not archive_path.is_file():
        raise FileNotFoundError(f"G1 pack archive not found: {archive_path}")

    if manifest_path is None:
        if "test" in archive_path.name:
            cand = archive_path.parent / "test_pack_manifest.json"
        else:
            cand = archive_path.parent / "pack_manifest.json"
        if cand.is_file():
            manifest_path = cand

    manifest_data: dict[str, Any] | None = None
    if manifest_path is not None and manifest_path.is_file():
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_arch_sha = manifest_data.get("archive_sha256")
        if expected_arch_sha:
            actual_arch_sha = _sha256_file_stream(archive_path)
            if actual_arch_sha != expected_arch_sha:
                raise ValueError(
                    f"Archive SHA-256 mismatch for {archive_path}: actual={actual_arch_sha}, expected={expected_arch_sha}"
                )

    frames_by_arc: dict[str, tuple[list[np.ndarray], str]] = {}
    meta_by_arc: dict[str, tuple[dict[str, Any], str]] = {}

    with _open_tar_zst_stream(archive_path) as tf:
        for member in tf:
            if not member.isfile():
                continue
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            payload = extracted.read()
            sha = _sha256_bytes(payload)
            if member.name in ("pack_manifest.json", "test_pack_manifest.json"):
                if manifest_data is None:
                    manifest_data = json.loads(payload.decode("utf-8"))
            elif member.name.endswith(".npy"):
                frames_by_arc[member.name] = (deserialize_stack_npz(payload), sha)
            elif member.name.endswith(".json"):
                meta_by_arc[member.name] = (json.loads(payload.decode("utf-8")), sha)

    if manifest_data is None:
        raise ValueError(f"No manifest found for archive {archive_path}")

    unpacked_items: list[dict[str, Any]] = []
    for item_entry in manifest_data.get("items", []):
        frames_arc = str(item_entry["frames_arcname"])
        meta_arc = str(item_entry["meta_arcname"])
        exp_frames_sha = str(item_entry["frames_sha256"])
        exp_meta_sha = str(item_entry["meta_sha256"])

        if frames_arc not in frames_by_arc:
            raise ValueError(f"Missing tar member {frames_arc} in {archive_path}")
        if meta_arc not in meta_by_arc:
            raise ValueError(f"Missing tar member {meta_arc} in {archive_path}")

        frames_list, act_frames_sha = frames_by_arc[frames_arc]
        meta_obj, act_meta_sha = meta_by_arc[meta_arc]

        if act_frames_sha != exp_frames_sha:
            raise ValueError(
                f"SHA-256 mismatch for {frames_arc}: actual={act_frames_sha}, expected={exp_frames_sha}"
            )
        if act_meta_sha != exp_meta_sha:
            raise ValueError(
                f"SHA-256 mismatch for {meta_arc}: actual={act_meta_sha}, expected={exp_meta_sha}"
            )
        if meta_obj.get("frames_sha256") != act_frames_sha:
            raise ValueError(
                f"Internal meta.json frames_sha256 mismatch for {frames_arc}"
            )

        item_dict = dict(meta_obj)
        item_dict["frames"] = frames_list
        item_dict["roi_origins"] = [tuple(float(x) for x in o) for o in meta_obj["roi_origins"]]

        clean_arc = item_entry.get("frames_clean_arcname")
        if clean_arc and str(clean_arc) in frames_by_arc:
            clean_frames, act_clean_sha = frames_by_arc[str(clean_arc)]
            exp_clean_sha = item_entry.get("frames_clean_sha256")
            if exp_clean_sha and act_clean_sha != str(exp_clean_sha):
                raise ValueError(f"SHA-256 mismatch for {clean_arc}")
            item_dict["frames_clean"] = clean_frames

        unpacked_items.append(item_dict)

    return unpacked_items


def generate_synthetic_pack_items_from_splits(
    splits_cfg: Mapping[str, Any],
    n_train: int = 4,
    n_val: int = 2,
    n_test: int = 2,
    crop_size: int = 64,
    seed: int = 20261107,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate deterministic, split-tagged pack items from `g1_splits_v0.json` for fast smoke/unit verification."""
    rng = np.random.default_rng(int(seed))
    enc = splits_cfg["encounters"]
    neg_blocks = splits_cfg["negative_time_blocks"]

    train_val_items: list[dict[str, Any]] = []
    test_items: list[dict[str, Any]] = []

    def _make_pose(t_s: float) -> dict[str, Any]:
        lat = 37.05 + 0.0005 * t_s
        lon = -120.85 - 0.0005 * t_s
        h_m = 3000.0
        q_est = [0.9998, 0.01 * math.sin(0.2 * t_s), 0.01 * math.cos(0.2 * t_s), 0.015]
        q_norm = math.sqrt(sum(c * c for c in q_est))
        return {
            "q_ned_to_cam_est": [c / q_norm for c in q_est],
            "pos_wgs84": [lat, lon, h_m],
            "pose_source": "wp06_estimated",
        }

    def _inject_bright_spot(arr: np.ndarray, uc: float, vc: float, w_b: float, h_b: float) -> np.ndarray:
        out = arr.copy()
        H, W = out.shape
        yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
        sig = max(1.0, 0.35 * max(w_b, h_b))
        blob = np.exp(-((xx - uc) ** 2 + (yy - vc) ** 2) / (2.0 * sig * sig))
        out += 180.0 * blob
        return out

    # 1. Train & Val positive K=3 stacks (Delta_t = 1.0 s) and negative time-block ROIs
    for split_name, count in (("train", n_train), ("val", n_val)):
        eids = enc[split_name][:count]
        for idx, eid in enumerate(eids):
            t0 = float(5 + idx * 3)
            timestamps = [t0 - 2.0, t0 - 1.0, t0]
            uc_loc = float(crop_size // 2 + ((idx * 7) % 9) - 4)
            vc_loc = float(crop_size // 2 + ((idx * 5) % 9) - 4)
            frames = []
            for f_i in range(3):
                bg = rng.normal(50.0, 4.0, size=(crop_size, crop_size)).astype(np.float32)
                frames.append(_inject_bright_spot(bg, uc_loc - (2 - f_i) * 1.0, vc_loc - (2 - f_i) * 0.5, 5.0, 4.0))
            poses = [_make_pose(ts) for ts in timestamps]
            u0_r, v0_r = 600.0, 500.0
            bx = [u0_r + uc_loc - 2.5, v0_r + vc_loc - 2.0, u0_r + uc_loc + 2.5, v0_r + vc_loc + 2.0]
            train_val_items.append(
                {
                    "item_id": f"{split_name}_stack_{eid}_{int(t0):03d}",
                    "kind": "kframe_stack",
                    "split": split_name,
                    "encounter_id": eid,
                    "num_frames_k": 3,
                    "timestamps_s": timestamps,
                    "dt_s": 1.0,
                    "frames": frames,
                    "poses": poses,
                    "roi_origins": [(u0_r, v0_r)] * 3,
                    "x0_roi": u0_r,
                    "y0_roi": v0_r,
                    "box_analytic": bx,
                    "sensor_state": "degraded",
                }
            )

        blk = neg_blocks[split_name]
        t_neg = float(blk["t_min_s"]) + 1.0
        eid_neg = enc[split_name][0]
        train_val_items.append(
            {
                "item_id": f"{split_name}_neg_{eid_neg}_{int(t_neg):03d}",
                "kind": "negative_roi",
                "split": split_name,
                "encounter_id": eid_neg,
                "num_frames_k": 1,
                "timestamps_s": [t_neg],
                "dt_s": 1.0,
                "frames": [rng.normal(50.0, 4.0, size=(crop_size, crop_size)).astype(np.float32)],
                "poses": [_make_pose(t_neg)],
                "roi_origins": [(400.0, 400.0)],
                "x0_roi": 400.0,
                "y0_roi": 400.0,
                "box_analytic": None,
                "sensor_state": "degraded",
            }
        )

    train_eid0 = enc["train"][0]
    train_val_items.append(
        {
            "item_id": f"train_horizon_strip_{train_eid0}",
            "kind": "horizon_strip",
            "split": "train",
            "encounter_id": train_eid0,
            "num_frames_k": 1,
            "timestamps_s": [10.0],
            "dt_s": 1.0,
            "frames": [rng.normal(50.0, 4.0, size=(crop_size // 2, crop_size)).astype(np.float32)],
            "poses": [_make_pose(10.0)],
            "roi_origins": [(0.0, 900.0)],
            "x0_roi": 0.0,
            "y0_roi": 900.0,
            "box_analytic": None,
            "sensor_state": "degraded",
        }
    )
    train_val_items.append(
        {
            "item_id": f"train_hard_neg_{train_eid0}",
            "kind": "hard_negative_d0",
            "split": "train",
            "encounter_id": train_eid0,
            "num_frames_k": 1,
            "timestamps_s": [15.0],
            "dt_s": 1.0,
            "frames": [rng.normal(50.0, 4.0, size=(crop_size, crop_size)).astype(np.float32)],
            "poses": [_make_pose(15.0)],
            "roi_origins": [(512.0, 512.0)],
            "x0_roi": 512.0,
            "y0_roi": 512.0,
            "box_analytic": None,
            "sensor_state": "degraded",
        }
    )

    test_eids = enc["test"][:n_test]
    for idx, eid in enumerate(test_eids):
        t0 = float(20 + idx * 5)
        timestamps = [t0 - 2.0, t0 - 1.0, t0]
        uc_loc = float(crop_size // 2 - 3 + idx * 4)
        vc_loc = float(crop_size // 2 - 2 + idx * 3)
        frames = []
        for f_i in range(3):
            bg = rng.normal(50.0, 4.0, size=(crop_size, crop_size)).astype(np.float32)
            frames.append(_inject_bright_spot(bg, uc_loc - (2 - f_i) * 1.0, vc_loc - (2 - f_i) * 0.5, 5.0, 4.0))
        u0_r, v0_r = 600.0, 500.0
        bx = [u0_r + uc_loc - 2.5, v0_r + vc_loc - 2.0, u0_r + uc_loc + 2.5, v0_r + vc_loc + 2.0]
        test_items.append(
            {
                "item_id": f"test_stack_{eid}_{int(t0):03d}",
                "kind": "kframe_stack",
                "split": "test",
                "encounter_id": eid,
                "num_frames_k": 3,
                "timestamps_s": timestamps,
                "dt_s": 1.0,
                "frames": frames,
                "poses": [_make_pose(ts) for ts in timestamps],
                "roi_origins": [(u0_r, v0_r)] * 3,
                "x0_roi": u0_r,
                "y0_roi": v0_r,
                "box_analytic": bx,
                "sensor_state": "degraded",
            }
        )

    t_test_neg = float(neg_blocks["test"]["t_min_s"]) + 2.0
    test_items.append(
        {
            "item_id": f"test_neg_{test_eids[0]}_{int(t_test_neg):03d}",
            "kind": "negative_roi",
            "split": "test",
            "encounter_id": test_eids[0],
            "num_frames_k": 1,
            "timestamps_s": [t_test_neg],
            "dt_s": 1.0,
            "frames": [rng.normal(50.0, 4.0, size=(crop_size, crop_size)).astype(np.float32)],
            "poses": [_make_pose(t_test_neg)],
            "roi_origins": [(400.0, 400.0)],
            "x0_roi": 400.0,
            "y0_roi": 400.0,
            "box_analytic": None,
            "sensor_state": "degraded",
        }
    )

    return train_val_items, test_items


def _validate_item_for_pack(
    item: Mapping[str, Any],
    allowed_splits: set[str],
    splits_cfg: Mapping[str, Any],
    ablation_mode: str = "standard",
) -> None:
    split = str(item["split"])
    if split not in allowed_splits:
        raise ValueError(f"Item {item['item_id']} has split={split!r}, not in allowed {sorted(allowed_splits)}")

    eid = str(item["encounter_id"])
    expected_split = assign_encounter_split(eid, dict(splits_cfg))
    if expected_split != split:
        raise ValueError(
            f"Encounter ID {eid} belongs to split {expected_split!r}, cannot be placed in {split!r} pack"
        )

    timestamps = [float(t) for t in item["timestamps_s"]]
    if len(timestamps) > 1:
        for i in range(1, len(timestamps)):
            dt = timestamps[i] - timestamps[i - 1]
            if abs(dt - 1.0) > 1e-6:
                raise ValueError(
                    f"Stack item {item['item_id']} violates D34-S2 Delta_t=1.0 s rule: dt={dt}"
                )

    if str(item["kind"]) == "negative_roi":
        neg_split = assign_negative_split_by_time(timestamps[-1], dict(splits_cfg))
        if neg_split != split:
            raise ValueError(
                f"Negative ROI {item['item_id']} at t_s={timestamps[-1]} belongs to time block {neg_split!r}, not {split!r}"
            )

    for pose in item["poses"]:
        validate_pose_dict(pose, ablation_mode=ablation_mode)


def build_g1_pack_archive(
    archive_path: Path,
    manifest_path: Path,
    items: Sequence[Mapping[str, Any]],
    allowed_splits: set[str],
    splits_cfg: Mapping[str, Any],
    splits_sha256: str,
    git_commit: str = "HEAD",
    ablation_mode: str = "standard",
) -> dict[str, Any]:
    """Write a deterministic `.tar.zst` pack archive and its cryptographic JSON manifest."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_items: list[dict[str, Any]] = []
    sorted_items = sorted(items, key=lambda x: str(x["item_id"]))

    with _open_tar_zst_write(archive_path) as tf:
        for item in sorted_items:
            _validate_item_for_pack(item, allowed_splits, splits_cfg, ablation_mode=ablation_mode)
            item_id = str(item["item_id"])
            split = str(item["split"])
            arr_bytes = serialize_stack_npz(item["frames"])
            arr_arcname = f"{split}/{item_id}/frames.npy"
            arr_sha256 = _add_deterministic_bytes(tf, arr_arcname, arr_bytes)

            clean_arcname: str | None = None
            clean_sha256: str | None = None
            if "frames_clean" in item and item["frames_clean"] is not None:
                clean_bytes = serialize_stack_npz(item["frames_clean"])
                clean_arcname = f"{split}/{item_id}/frames_clean.npy"
                clean_sha256 = _add_deterministic_bytes(tf, clean_arcname, clean_bytes)

            meta_obj: dict[str, Any] = {
                "item_id": item_id,
                "kind": item["kind"],
                "split": split,
                "encounter_id": item["encounter_id"],
                "num_frames_k": int(item["num_frames_k"]),
                "timestamps_s": [float(t) for t in item["timestamps_s"]],
                "dt_s": float(item["dt_s"]),
                "poses": list(item["poses"]),
                "roi_origins": [list(o) for o in item["roi_origins"]],
                "box_analytic": item.get("box_analytic"),
                "sensor_state": item.get("sensor_state", "degraded"),
                "frames_sha256": arr_sha256,
            }
            if clean_sha256 is not None:
                meta_obj["frames_clean_sha256"] = clean_sha256

            for opt_key in (
                "cam_id",
                "k_frame",
                "class_name",
                "range_m",
                "s_px",
                "size_bin",
                "background",
                "x0_roi",
                "y0_roi",
                "roi_width",
                "roi_height",
                "box_analytic_local",
                "boxes_analytic_stack",
                "horizon_curve_uv",
                "fp_center_uv",
                "fp_score_d0",
                "d0_fp_candidates_mined",
                "centers_uv",
                "boxes_wh",
            ):
                if opt_key in item:
                    meta_obj[opt_key] = item[opt_key]
            meta_bytes = (json.dumps(meta_obj, sort_keys=True, indent=2) + "\n").encode("utf-8")
            meta_arcname = f"{split}/{item_id}/meta.json"
            meta_sha256 = _add_deterministic_bytes(tf, meta_arcname, meta_bytes)

            man_entry: dict[str, Any] = {
                "item_id": item_id,
                "kind": item["kind"],
                "split": split,
                "encounter_id": item["encounter_id"],
                "num_frames_k": int(item["num_frames_k"]),
                "timestamps_s": [float(t) for t in item["timestamps_s"]],
                "dt_s": float(item["dt_s"]),
                "frames_arcname": arr_arcname,
                "frames_sha256": arr_sha256,
                "meta_arcname": meta_arcname,
                "meta_sha256": meta_sha256,
            }
            if clean_arcname is not None and clean_sha256 is not None:
                man_entry["frames_clean_arcname"] = clean_arcname
                man_entry["frames_clean_sha256"] = clean_sha256
            manifest_items.append(man_entry)

        pos_count = sum(1 for it in sorted_items if it.get("box_analytic") is not None)
        neg_count = sum(1 for it in sorted_items if it.get("box_analytic") is None)
        bin_counts = {
            "bin_0_8_to_1_5_px": 0,
            "bin_1_5_to_2_5_px": 0,
            "bin_2_5_to_4_0_px": 0,
            "bin_4_0_plus_px": 0,
        }
        bin_keys = (
            "bin_0_8_to_1_5_px",
            "bin_1_5_to_2_5_px",
            "bin_2_5_to_4_0_px",
            "bin_4_0_plus_px",
        )
        for it in sorted_items:
            if it.get("box_analytic") is not None and "s_px" in it:
                b_i = _size_bin_index(float(it["s_px"]))
                bin_counts[bin_keys[b_i]] += 1

        eids_all = sorted({str(it["encounter_id"]) for it in sorted_items if "encounter_id" in it})
        eids_by_split = {
            sp: len({str(it["encounter_id"]) for it in sorted_items if str(it.get("split")) == sp and "encounter_id" in it})
            for sp in sorted(allowed_splits)
        }
        neg_time_block_area_px = sum(
            float(it.get("roi_width", 0)) * float(it.get("roi_height", 0))  # default-ok: zero-area default for pack items without roi dimensions
            for it in sorted_items
            if it.get("kind") == "negative_roi"
        )
        neg_all_area_px = sum(
            float(it.get("roi_width", 0)) * float(it.get("roi_height", 0))  # default-ok: zero-area default for pack items without roi dimensions
            for it in sorted_items
            if it.get("box_analytic") is None
        )
        total_area_px = sum(
            float(it.get("roi_width", 0)) * float(it.get("roi_height", 0))  # default-ok: zero-area default for pack items without roi dimensions
            for it in sorted_items
        )

        manifest_payload = {
            "schema": "xvs.detection_g1_pack_manifest/1",
            "archive_name": archive_path.name,
            "allowed_splits": sorted(allowed_splits),
            "git_commit": git_commit,
            "g1_splits_v0_sha256": splits_sha256,
            "g3a_release_sha256": str(splits_cfg["g3a_release_sha256"]),
            "kept_frame_step_dt_s": 1.0,
            "ablation_mode": ablation_mode,
            "item_count": len(manifest_items),
            "positive_rois_count": int(pos_count),
            "negative_rois_count": int(neg_count),
            "encounters_covered_count": len(eids_all),
            "encounters_by_split_count": eids_by_split,
            "negative_time_block_exposure_mpx": round(neg_time_block_area_px / 1.0e6, 6),
            "negative_exposure_mpx": round(neg_all_area_px / 1.0e6, 6),
            "total_pack_exposure_mpx": round(total_area_px / 1.0e6, 6),
            "size_bin_counts": bin_counts,
            "items": manifest_items,
        }
        man_bytes = (json.dumps(manifest_payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
        _add_deterministic_bytes(tf, manifest_path.name, man_bytes)

    archive_sha256 = _sha256_bytes(archive_path.read_bytes())
    manifest_payload["archive_sha256"] = archive_sha256
    manifest_path.write_text(json.dumps(manifest_payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest_payload


def build_both_g1_packs(
    out_dir: Path,
    splits_path: Path = DEFAULT_SPLITS_CONFIG,
    train_val_items: Sequence[Mapping[str, Any]] | None = None,
    test_items: Sequence[Mapping[str, Any]] | None = None,
    git_commit: str = "HEAD",
    train_val_archive_name: str = "g1_train_val_pack_v0.tar.zst",
    test_archive_name: str = "g1_test_eval_pack_v0.tar.zst",
) -> dict[str, Any]:
    """Build both train/val and held-out test `.tar.zst` packs and run leakage check."""
    splits_cfg, splits_sha256 = load_splits_config(splits_path)
    if train_val_items is None or test_items is None:
        gen_tv, gen_test = generate_synthetic_pack_items_from_splits(splits_cfg)
        if train_val_items is None:
            train_val_items = gen_tv
        if test_items is None:
            test_items = gen_test

    tv_archive = out_dir / train_val_archive_name
    tv_manifest = out_dir / "pack_manifest.json"
    te_archive = out_dir / test_archive_name
    te_manifest = out_dir / "test_pack_manifest.json"

    tv_man = build_g1_pack_archive(
        archive_path=tv_archive,
        manifest_path=tv_manifest,
        items=train_val_items,
        allowed_splits={"train", "val"},
        splits_cfg=splits_cfg,
        splits_sha256=splits_sha256,
        git_commit=git_commit,
    )
    te_man = build_g1_pack_archive(
        archive_path=te_archive,
        manifest_path=te_manifest,
        items=test_items,
        allowed_splits={"test"},
        splits_cfg=splits_cfg,
        splits_sha256=splits_sha256,
        git_commit=git_commit,
    )

    leakage_errors = verify_pack_no_leakage(
        train_val_archive=tv_archive,
        train_val_manifest=tv_manifest,
        test_archive=te_archive,
        test_manifest=te_manifest,
        splits_cfg=splits_cfg,
    )
    if leakage_errors:
        raise RuntimeError(f"Post-build pack leakage check failed (D34-S9 / V1_G1): {leakage_errors}")

    return {
        "train_val_archive": str(tv_archive),
        "train_val_manifest": tv_man,
        "test_archive": str(te_archive),
        "test_manifest": te_man,
        "leakage_errors": leakage_errors,
    }


def verify_pack_no_leakage(
    train_val_archive: Path,
    train_val_manifest: Path,
    test_archive: Path,
    test_manifest: Path,
    splits_cfg: Mapping[str, Any],
) -> list[str]:
    """Post-build pack leakage and SHA-256 auditor (`Decision D34-S9`, `V1_G1`)."""
    errors: list[str] = []
    if not train_val_archive.is_file():
        return [f"Missing train/val archive: {train_val_archive}"]
    if not train_val_manifest.is_file():
        return [f"Missing train/val manifest: {train_val_manifest}"]
    if not test_archive.is_file():
        return [f"Missing test archive: {test_archive}"]
    if not test_manifest.is_file():
        return [f"Missing test manifest: {test_manifest}"]

    tv_man = json.loads(train_val_manifest.read_text(encoding="utf-8"))
    te_man = json.loads(test_manifest.read_text(encoding="utf-8"))

    test_encounter_set = set(splits_cfg["encounters"]["test"])
    train_val_encounter_set = set(splits_cfg["encounters"]["train"]) | set(splits_cfg["encounters"]["val"])

    def _audit_archive_members(archive_path: Path, man: Mapping[str, Any], label: str) -> tuple[set[str], set[str]]:
        arcnames: set[str] = set()
        digests: set[str] = set()
        actual_sha_by_arc: dict[str, str] = {}
        with _open_tar_zst_stream(archive_path) as tf:
            for member in tf:
                if not member.isfile():
                    continue
                extracted = tf.extractfile(member)
                if extracted is None:
                    continue
                h = hashlib.sha256()
                while True:
                    chunk = extracted.read(1024 * 1024)
                    if not chunk:
                        break
                    h.update(chunk)
                actual_sha_by_arc[member.name] = h.hexdigest()

        for item in man.get("items", []):
            pairs = [("frames_arcname", "frames_sha256"), ("meta_arcname", "meta_sha256")]
            if "frames_clean_arcname" in item and "frames_clean_sha256" in item:
                pairs.append(("frames_clean_arcname", "frames_clean_sha256"))
            for arc_key, sha_key in pairs:
                arcname = str(item[arc_key])
                expected_sha = str(item[sha_key])
                arcnames.add(arcname)
                digests.add(expected_sha)
                if arcname not in actual_sha_by_arc:
                    errors.append(f"[{label}] Missing tar member {arcname}")
                    continue
                actual_sha = actual_sha_by_arc[arcname]
                if actual_sha != expected_sha:
                    errors.append(
                        f"[{label}] SHA-256 mismatch for {arcname}: actual={actual_sha}, expected={expected_sha}"
                    )
        return arcnames, digests

    tv_arcnames, tv_digests = _audit_archive_members(train_val_archive, tv_man, "train_val")
    te_arcnames, te_digests = _audit_archive_members(test_archive, te_man, "test")

    for item in tv_man.get("items", []):
        eid = str(item["encounter_id"])
        split = str(item["split"])
        if split == "test" or eid in test_encounter_set:
            errors.append(f"Test encounter {eid} (split={split}) leaked into train/val pack item {item['item_id']}")
        if eid not in train_val_encounter_set:
            errors.append(f"Unknown encounter {eid} in train/val pack item {item['item_id']}")

        ts = [float(t) for t in item.get("timestamps_s", [])]
        if len(ts) > 1:
            for i in range(1, len(ts)):
                if abs((ts[i] - ts[i - 1]) - 1.0) > 1e-6:
                    errors.append(f"Item {item['item_id']} violates Delta_t=1.0 s step: {ts}")
        if str(item.get("kind")) == "negative_roi" and ts:
            neg_split = assign_negative_split_by_time(ts[-1], dict(splits_cfg))
            if neg_split == "test":
                errors.append(f"Test negative time-block frame t_s={ts[-1]} leaked into train/val pack")

    te_eids_in_pack = {str(it["encounter_id"]) for it in te_man.get("items", [])}
    tv_eids_in_pack = {str(it["encounter_id"]) for it in tv_man.get("items", [])}

    if tv_eids_in_pack & te_eids_in_pack:
        errors.append(f"Encounter ID overlap between train/val and test packs: {sorted(tv_eids_in_pack & te_eids_in_pack)}")
    if tv_arcnames & te_arcnames:
        errors.append(f"Archive path overlap between train/val and test packs: {sorted(tv_arcnames & te_arcnames)}")
    if tv_digests & te_digests:
        errors.append(f"Per-item SHA-256 digest overlap between train/val and test packs: {sorted(tv_digests & te_digests)}")

    return errors
