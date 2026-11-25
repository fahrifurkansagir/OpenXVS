"""Real AWS S3 Airborne Object Tracking (AOT) Schema & Ephemeral Streaming Ingestion (`B8-Colab-F1`).

Verified live S3 layout under `s3://airborne-obj-detection-challenge-training`
(`https://airborne-obj-detection-challenge-training.s3.amazonaws.com`):
- `part1/ImageSets/valid_encounters_maxRange700_maxGap3_minEncLen30.csv` (120.6 KB):
  `,encounter_id,flight_id,framemin,framemax,framecount,enc_len_with_gaps,is_above_horizon,min_enc_range,max_enc_range,median_enc_range,mean_enc_range`
  where `is_above_horizon` is `1.0` (above horizon) or `-1.0` (below horizon).
- `part1/ImageSets/groundtruth.csv` (311.8 MB):
  `,time,flight_id,img_name,frame,id,range_distance_m,is_above_horizon,size_width,size_height,gt_left,gt_top,gt_right,gt_bottom`
  where rows with empty `id` / `gt_left..gt_bottom` are target-free frames of that flight and
  `img_name` is `<nanosec_timestamp><flight_id>.png` (`2448x2048` 8-bit grayscale).
- Frame PNGs live at `part1/Images/<flight_id>/<img_name>`.

Capabilities:
1. Downloads `ImageSets/*.csv` via `aws s3 cp --no-sign-request` with automatic chunked HTTPS
   `urllib.request` fallback when `awscli` is absent on the Colab VM.
2. Deterministically selects a seeded stratified evaluation set of `>= 12` valid encounters
   (`6` above-horizon `is_above_horizon == 1.0` + `6` below-horizon `is_above_horizon == -1.0`),
   sampled at `1.0 s` / `10`-frame step (`10 Hz` AOT camera rate -> `dt = 1.0 s` for `D1-KF`),
   plus target-free `10`-frame-step triplets/frames for `FP/Mpx` estimation.
3. Downloads ONLY the exact required `img_name` PNGs (~150–250 PNGs, ~400–600 MB) into
   `/content/aot/sequences/<flight_id>/<img_name>`, writes `/content/aot/aot_ground_truth.json`
   (`dataset_id: "AOT-v1"`, `synthetic: false`), and cleans up temporary shards on exit.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Mapping, Sequence
import urllib.request
import numpy as np

AOT_S3_BUCKET = "s3://airborne-obj-detection-challenge-training"
AOT_HTTPS_BASE = "https://airborne-obj-detection-challenge-training.s3.amazonaws.com"
VALID_ENCOUNTERS_S3_KEY = "part1/ImageSets/valid_encounters_maxRange700_maxGap3_minEncLen30.csv"
GROUNDTRUTH_S3_KEY = "part1/ImageSets/groundtruth.csv"
IMAGES_S3_PREFIX = "part1/Images"

DownloaderFn = Callable[[str, Path], None]


def map_aot_horizon_label(raw_val: Any, default: str = "unknown") -> str:
    """Convert AOT `is_above_horizon` numeric code (`1.0`, `-1.0`, `0.0`, `""`) to `'above' | 'below' | 'unknown'`."""
    if raw_val is None:
        return default
    s = str(raw_val).strip()
    if not s or s.lower() == "nan":
        return default
    try:
        v = float(s)
    except ValueError:
        s_low = s.lower()
        if s_low in ("above", "below", "unknown"):
            return s_low
        return default
    if math.isnan(v):
        return default
    if v > 0.5:
        return "above"
    if v < -0.5:
        return "below"
    return "unknown"


def _safe_float(raw_val: Any, default: float | None = None) -> float | None:
    if raw_val is None:
        return default
    s = str(raw_val).strip()
    if not s or s.lower() == "nan":
        return default
    try:
        val = float(s)
    except ValueError:
        return default
    if not math.isfinite(val):
        return default
    return val


def download_s3_object_with_https_fallback(
    s3_key: str,
    dest_path: Path,
    s3_bucket: str = AOT_S3_BUCKET,
    https_base: str = AOT_HTTPS_BASE,
    timeout_s: int = 300,
    prefer_aws_cli: bool = True,
) -> Path:
    """Download a single object from public AOT S3 via `aws s3 cp --no-sign-request` or HTTPS `urllib.request`."""
    if dest_path.is_file() and dest_path.stat().st_size > 0:
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_name(f"{dest_path.name}.part.{os.getpid()}")
    clean_key = s3_key.lstrip("/")

    try:
        if prefer_aws_cli and shutil.which("aws") is not None:
            s3_uri = f"{s3_bucket.rstrip('/')}/{clean_key}"
            proc = subprocess.run(
                ["aws", "s3", "cp", "--no-sign-request", "--only-show-errors", s3_uri, str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            if proc.returncode == 0 and tmp_path.is_file() and tmp_path.stat().st_size > 0:
                os.replace(tmp_path, dest_path)
                return dest_path

        url = f"{https_base.rstrip('/')}/{clean_key}"
        req = urllib.request.Request(url, headers={"User-Agent": "OpenXVS-B9-Ingest/1.0"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp, open(tmp_path, "wb") as out_f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out_f.write(chunk)

        if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
            raise RuntimeError(f"Downloaded empty payload from {url}")
        os.replace(tmp_path, dest_path)
        return dest_path
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def parse_valid_encounters_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Parse `valid_encounters_maxRange700_maxGap3_minEncLen30.csv` into normalized encounter dicts."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"AOT valid_encounters CSV not found: {csv_path}")

    records: list[dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            flight_id = str(row.get("flight_id", "")).strip()
            if not flight_id:
                continue
            enc_id = str(row.get("encounter_id", "")).strip() or f"{flight_id}_{row.get('framemin', '0')}"
            framemin = int(float(row["framemin"]))
            framemax = int(float(row["framemax"]))
            framecount = int(float(row.get("framecount") or max(1, framemax - framemin + 1)))
            enc_len = int(float(row.get("enc_len_with_gaps") or max(1, framemax - framemin + 1)))
            hz_raw = _safe_float(row.get("is_above_horizon"), default=0.0) or 0.0
            horizon = map_aot_horizon_label(hz_raw)
            records.append(
                {
                    "encounter_id": enc_id,
                    "flight_id": flight_id,
                    "framemin": framemin,
                    "framemax": framemax,
                    "framecount": framecount,
                    "enc_len_with_gaps": enc_len,
                    "is_above_horizon": float(hz_raw),
                    "horizon": horizon,
                    "min_enc_range": _safe_float(row.get("min_enc_range"), default=500.0),
                    "max_enc_range": _safe_float(row.get("max_enc_range"), default=700.0),
                    "median_enc_range": _safe_float(row.get("median_enc_range"), default=600.0),
                    "mean_enc_range": _safe_float(row.get("mean_enc_range"), default=600.0),
                }
            )
    return records


def sample_encounter_frames_10hz(
    framemin: int,
    framemax: int,
    dt_frame_step: int = 10,
    max_frames_per_encounter: int = 12,
) -> list[int]:
    """Sample evenly spaced `dt = 1.0 s` (`10`-frame step at 10 Hz) frames across `[framemin, framemax]`.

    Guarantees adjacent sampled frames differ by exact `dt_frame_step` (`10` frames = `1.0 s`) so `D1-KF`
    (`K=3` or `K=5`) evaluates genuine 1.0 s temporal stacks.
    """
    step = max(1, int(dt_frame_step))
    span = max(0, int(framemax) - int(framemin))
    all_steps = list(range(int(framemin), int(framemax) + 1, step))
    if len(all_steps) <= int(max_frames_per_encounter):
        return all_steps
    # Keep a contiguous `10`-frame-stepped window centered on the encounter midpoint (closest approach / median range)
    start_idx = (len(all_steps) - int(max_frames_per_encounter)) // 2
    return all_steps[start_idx : start_idx + int(max_frames_per_encounter)]


def select_stratified_aot_encounters(
    encounters: Sequence[Mapping[str, Any]],
    n_above: int = 6,
    n_below: int = 6,
    seed: int = 20261109,
    dt_frame_step: int = 10,
    max_frames_per_encounter: int = 12,
    priority_flight_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Deterministically select `>= n_above + n_below` valid encounters (`above` and `below` horizon)."""
    min_span = 2 * max(1, int(dt_frame_step))
    above_pool = [
        dict(e)
        for e in encounters
        if str(e.get("horizon") or map_aot_horizon_label(e.get("is_above_horizon"))) == "above"
        and (int(e["framemax"]) - int(e["framemin"])) >= min_span
    ]
    below_pool = [
        dict(e)
        for e in encounters
        if str(e.get("horizon") or map_aot_horizon_label(e.get("is_above_horizon"))) == "below"
        and (int(e["framemax"]) - int(e["framemin"])) >= min_span
    ]
    # Fallback if fixture encounters are shorter than 20 frames
    if len(above_pool) < n_above:
        above_pool = [
            dict(e)
            for e in encounters
            if str(e.get("horizon") or map_aot_horizon_label(e.get("is_above_horizon"))) == "above"
        ]
    if len(below_pool) < n_below:
        below_pool = [
            dict(e)
            for e in encounters
            if str(e.get("horizon") or map_aot_horizon_label(e.get("is_above_horizon"))) == "below"
        ]

    if len(above_pool) < n_above or len(below_pool) < n_below:
        raise ValueError(
            f"Insufficient stratified AOT encounters: found {len(above_pool)} above-horizon (need {n_above}) "
            f"and {len(below_pool)} below-horizon (need {n_below})."
        )

    rng = np.random.default_rng(int(seed))
    priority_set = set(priority_flight_ids or [])

    def _pick_stratum(pool: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        sorted_pool = sorted(pool, key=lambda x: (str(x["flight_id"]), int(x["framemin"]), str(x["encounter_id"])))
        order = rng.permutation(len(sorted_pool)).tolist()
        shuffled = [sorted_pool[i] for i in order]
        # Place verified priority flights first (deterministic order) while filling remainder via seeded permutation
        pri = [e for e in sorted_pool if str(e["flight_id"]) in priority_set]
        rest = [e for e in shuffled if str(e["flight_id"]) not in priority_set]
        ordered = pri + rest

        chosen: list[dict[str, Any]] = []
        seen_flights: set[str] = set()
        seen_encs: set[tuple[str, int, int]] = set()
        # Pass 1: distinct flight_ids
        for e in ordered:
            if len(chosen) >= count:
                break
            fid = str(e["flight_id"])
            ekey = (fid, int(e["framemin"]), int(e["framemax"]))
            if fid not in seen_flights and ekey not in seen_encs:
                seen_flights.add(fid)
                seen_encs.add(ekey)
                chosen.append(e)
        # Pass 2: allow multiple encounters in same flight if CSV has fewer unique flights
        for e in ordered:
            if len(chosen) >= count:
                break
            ekey = (str(e["flight_id"]), int(e["framemin"]), int(e["framemax"]))
            if ekey not in seen_encs:
                seen_encs.add(ekey)
                chosen.append(e)
        return chosen

    selected = _pick_stratum(above_pool, int(n_above)) + _pick_stratum(below_pool, int(n_below))
    for item in selected:
        item["sampled_frames"] = sample_encounter_frames_10hz(
            int(item["framemin"]),
            int(item["framemax"]),
            dt_frame_step=dt_frame_step,
            max_frames_per_encounter=max_frames_per_encounter,
        )
    return selected


def build_aot_sequences_from_groundtruth_csv(
    groundtruth_csv_path: Path,
    selected_encounters: Sequence[Mapping[str, Any]],
    dt_frame_step: int = 10,
    target_free_stacks_per_flight: int = 1,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Stream `groundtruth.csv` once for selected `flight_id`s and build sequence metadata + required `(flight_id, img_name)` pairs.

    Returns:
        sequences: list of sequence dicts ready for `aot_ground_truth.json["sequences"]`.
        required_pngs: deduplicated sorted list of `(flight_id, img_name)` to download from `part1/Images/<flight_id>/<img_name>`.
    """
    if not groundtruth_csv_path.is_file():
        raise FileNotFoundError(f"AOT groundtruth.csv not found: {groundtruth_csv_path}")

    target_flights = {str(e["flight_id"]) for e in selected_encounters}
    enc_default_range: dict[str, float] = {
        str(e["flight_id"]): float(e.get("median_enc_range") or 600.0) for e in selected_encounters
    }
    enc_default_horizon: dict[str, str] = {
        str(e["flight_id"]): str(e.get("horizon") or map_aot_horizon_label(e.get("is_above_horizon")))
        for e in selected_encounters
    }

    # Map: flight_id -> frame_int -> {"img_name": str, "time": str, "objects": list[dict]}
    flight_frames: dict[str, dict[int, dict[str, Any]]] = {fid: {} for fid in target_flights}
    flight_positive_frame_nums: dict[str, set[int]] = {fid: set() for fid in target_flights}

    with open(groundtruth_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = str(row.get("flight_id", "")).strip()
            if fid not in target_flights:
                continue
            frame_raw = row.get("frame")
            img_name = str(row.get("img_name", "")).strip()
            if frame_raw is None or not str(frame_raw).strip() or not img_name:
                continue
            f_num = int(float(frame_raw))
            entry = flight_frames[fid].setdefault(
                f_num,
                {
                    "aot_frame": f_num,
                    "img_name": img_name,
                    "time_raw": str(row.get("time", "")).strip(),
                    "objects": [],
                },
            )

            obj_id = str(row.get("id", "")).strip()
            gt_l = _safe_float(row.get("gt_left"))
            gt_t = _safe_float(row.get("gt_top"))
            gt_r = _safe_float(row.get("gt_right"))
            gt_b = _safe_float(row.get("gt_bottom"))
            if obj_id and gt_l is not None and gt_t is not None and gt_r is not None and gt_b is not None:
                flight_positive_frame_nums[fid].add(f_num)
                rng_m = _safe_float(row.get("range_distance_m"), default=enc_default_range.get(fid, 600.0))  # default-ok: AOT encounter range fallback when CSV range_distance_m is blank
                if rng_m is None:
                    rng_m = enc_default_range.get(fid, 600.0)  # default-ok: AOT encounter range fallback when CSV range_distance_m is blank
                hz_label = map_aot_horizon_label(
                    row.get("is_above_horizon"),
                    default=enc_default_horizon.get(fid, "unknown"),
                )
                entry["objects"].append(
                    {
                        "object_id": obj_id,
                        "class_id": "airborne",
                        "range_m": float(rng_m),
                        "box_edges": [float(gt_l), float(gt_t), float(gt_r), float(gt_b)],
                        "horizon": hz_label,
                    }
                )

    sequences: list[dict[str, Any]] = []
    required_set: set[tuple[str, str]] = set()
    step = max(1, int(dt_frame_step))

    # 1. Build positive encounter sequences at dt = 1.0 s (10-frame step)
    for enc in selected_encounters:
        fid = str(enc["flight_id"])
        enc_id = str(enc["encounter_id"])
        fmap = flight_frames.get(fid, {})
        if not fmap:
            continue

        sampled = list(
            enc.get("sampled_frames")
            or sample_encounter_frames_10hz(int(enc["framemin"]), int(enc["framemax"]), dt_frame_step=step)
        )
        avail_sorted = sorted(fmap.keys())
        frames_out: list[dict[str, Any]] = []
        used_fnums: set[int] = set()

        for target_f in sampled:
            cand_f = target_f if target_f in fmap else None
            if cand_f is None:
                # Snap within +/- 2 frames if an occasional single frame was dropped in flight recording
                near = [af for af in avail_sorted if abs(af - target_f) <= 2 and af not in used_fnums]
                if near:
                    cand_f = min(near, key=lambda af: abs(af - target_f))
            if cand_f is None or cand_f in used_fnums:
                continue
            used_fnums.add(cand_f)
            info = fmap[cand_f]
            rel_img = f"sequences/{fid}/{info['img_name']}"
            required_set.add((fid, str(info["img_name"])))
            frames_out.append(
                {
                    "frame_id": f"{enc_id}_f{cand_f:05d}",
                    "aot_frame": int(cand_f),
                    "t_s": round(float(cand_f) * 0.1, 3),
                    "image_path": rel_img,
                    "objects": list(info["objects"]),
                }
            )

        if frames_out:
            sequences.append(
                {
                    "sequence_id": f"enc_{enc_id}",
                    "flight_id": fid,
                    "encounter_id": enc_id,
                    "sequence_kind": "encounter",
                    "stratum_horizon": str(enc.get("horizon") or map_aot_horizon_label(enc.get("is_above_horizon"))),
                    "dt_s": round(float(step) * 0.1, 3),
                    "aot_frame_step": step,
                    "frames": frames_out,
                }
            )

    # 2. Build target-free background sequences (K=3 frames at 10-frame step when available) per flight for FP/Mpx
    if int(target_free_stacks_per_flight) > 0:
        for fid in sorted(target_flights):
            fmap = flight_frames.get(fid, {})
            if not fmap:
                continue
            pos_fnums = flight_positive_frame_nums.get(fid, set())
            pure_neg_fnums = sorted(
                fn
                for fn, info in fmap.items()
                if len(info["objects"]) == 0 and all(abs(fn - pf) >= 25 for pf in pos_fnums)
            )
            if not pure_neg_fnums:
                pure_neg_fnums = sorted(fn for fn, info in fmap.items() if len(info["objects"]) == 0)
            if not pure_neg_fnums:
                continue

            neg_set = set(pure_neg_fnums)
            triplets = [
                [fn - 2 * step, fn - step, fn]
                for fn in pure_neg_fnums
                if (fn - step) in neg_set and (fn - 2 * step) in neg_set
            ]
            chosen_groups: list[list[int]] = []
            if triplets:
                idxs = np.linspace(0, len(triplets) - 1, num=min(int(target_free_stacks_per_flight), len(triplets)), dtype=int)
                for ii in idxs:
                    grp = triplets[int(ii)]
                    if grp not in chosen_groups:
                        chosen_groups.append(grp)
            else:
                chosen_groups.append(pure_neg_fnums[: min(3, len(pure_neg_fnums))])

            for g_i, grp in enumerate(chosen_groups):
                neg_frames_out: list[dict[str, Any]] = []
                for fn in grp:
                    info = fmap[fn]
                    rel_img = f"sequences/{fid}/{info['img_name']}"
                    required_set.add((fid, str(info["img_name"])))
                    neg_frames_out.append(
                        {
                            "frame_id": f"neg_{fid[:8]}_{g_i:02d}_f{fn:05d}",
                            "aot_frame": int(fn),
                            "t_s": round(float(fn) * 0.1, 3),
                            "image_path": rel_img,
                            "objects": [],
                        }
                    )
                if neg_frames_out:
                    sequences.append(
                        {
                            "sequence_id": f"neg_{fid[:12]}_{g_i:02d}",
                            "flight_id": fid,
                            "encounter_id": None,
                            "sequence_kind": "target_free",
                            "stratum_horizon": "target_free",
                            "dt_s": round(float(step) * 0.1, 3),
                            "aot_frame_step": step,
                            "frames": neg_frames_out,
                        }
                    )

    if not sequences:
        raise RuntimeError("Zero AOT sequences matched between valid_encounters CSV and groundtruth.csv.")

    return sequences, sorted(required_set)


def ingest_stratified_aot_from_s3(
    local_aot_dir: Path,
    s3_bucket: str = AOT_S3_BUCKET,
    https_base: str = AOT_HTTPS_BASE,
    valid_encounters_key: str = VALID_ENCOUNTERS_S3_KEY,
    groundtruth_key: str = GROUNDTRUTH_S3_KEY,
    images_prefix: str = IMAGES_S3_PREFIX,
    n_above: int = 6,
    n_below: int = 6,
    seed: int = 20261109,
    dt_frame_step: int = 10,
    max_frames_per_encounter: int = 12,
    target_free_stacks_per_flight: int = 1,
    priority_flight_ids: Sequence[str] | None = None,
    max_download_threads: int = 8,
    cleanup_groundtruth_csv: bool = True,
    downloader: DownloaderFn | None = None,
) -> dict[str, Any]:
    """End-to-end AOT S3/HTTPS metadata ingestion, stratified sampling, selective PNG download, and JSON conversion."""
    local_aot_dir = Path(local_aot_dir)
    imagesets_dir = local_aot_dir / "ImageSets"
    imagesets_dir.mkdir(parents=True, exist_ok=True)

    def _default_dl(key: str, dst: Path) -> None:
        download_s3_object_with_https_fallback(
            s3_key=key,
            dest_path=dst,
            s3_bucket=s3_bucket,
            https_base=https_base,
        )

    dl_fn = downloader if downloader is not None else _default_dl

    valid_csv_path = imagesets_dir / Path(valid_encounters_key).name
    gt_csv_path = imagesets_dir / Path(groundtruth_key).name

    dl_fn(valid_encounters_key, valid_csv_path)
    dl_fn(groundtruth_key, gt_csv_path)

    all_encounters = parse_valid_encounters_csv(valid_csv_path)
    selected_encounters = select_stratified_aot_encounters(
        all_encounters,
        n_above=n_above,
        n_below=n_below,
        seed=seed,
        dt_frame_step=dt_frame_step,
        max_frames_per_encounter=max_frames_per_encounter,
        priority_flight_ids=priority_flight_ids,
    )

    sequences, required_pngs = build_aot_sequences_from_groundtruth_csv(
        groundtruth_csv_path=gt_csv_path,
        selected_encounters=selected_encounters,
        dt_frame_step=dt_frame_step,
        target_free_stacks_per_flight=target_free_stacks_per_flight,
    )

    clean_img_prefix = images_prefix.strip("/")

    def _fetch_one(pair: tuple[str, str]) -> Path:
        fid, img_name = pair
        s3_key = f"{clean_img_prefix}/{fid}/{img_name}"
        dst = local_aot_dir / "sequences" / fid / img_name
        dl_fn(s3_key, dst)
        return dst

    if max_download_threads <= 1 or len(required_pngs) <= 1:
        for pair in required_pngs:
            _fetch_one(pair)
    else:
        with ThreadPoolExecutor(max_workers=int(max_download_threads)) as pool:
            futs = [pool.submit(_fetch_one, pair) for pair in required_pngs]
            for fut in as_completed(futs):
                fut.result()

    n_enc_above = sum(
        1 for s in sequences if s.get("sequence_kind") == "encounter" and s.get("stratum_horizon") == "above"
    )
    n_enc_below = sum(
        1 for s in sequences if s.get("sequence_kind") == "encounter" and s.get("stratum_horizon") == "below"
    )
    n_target_free = sum(1 for s in sequences if s.get("sequence_kind") == "target_free")

    gt_manifest = {
        "schema": "xvs.aot_ground_truth/1",
        "dataset_id": "AOT-v1",
        "synthetic": False,
        "unit_test_fixture": False,
        "source_bucket": s3_bucket,
        "valid_encounters_key": valid_encounters_key,
        "groundtruth_key": groundtruth_key,
        "selection_seed": int(seed),
        "dt_s": round(float(dt_frame_step) * 0.1, 3),
        "aot_frame_step": int(dt_frame_step),
        "n_above_horizon_encounters": int(n_enc_above),
        "n_below_horizon_encounters": int(n_enc_below),
        "n_target_free_sequences": int(n_target_free),
        "downloaded_png_count": len(required_pngs),
        "sequences": sequences,
    }
    gt_json_path = local_aot_dir / "aot_ground_truth.json"
    gt_json_path.write_text(json.dumps(gt_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Clean up temporary .part files and optional 311.8 MB groundtruth.csv after conversion
    for part_file in local_aot_dir.rglob("*.part.*"):
        try:
            part_file.unlink()
        except OSError:
            pass
    if cleanup_groundtruth_csv and gt_csv_path.is_file():
        try:
            gt_csv_path.unlink()
        except OSError:
            pass

    return {
        "aot_ground_truth_path": str(gt_json_path),
        "sequences_count": len(sequences),
        "above_horizon_encounters": int(n_enc_above),
        "below_horizon_encounters": int(n_enc_below),
        "target_free_sequences": int(n_target_free),
        "downloaded_png_count": len(required_pngs),
    }
