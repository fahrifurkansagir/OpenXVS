"""AOT `[left, top, width, height]` Parser, `capture_day` Split & Stdlib `tarfile` WDS Pipeline (`DR-01`).

Implements `SP2_M2_MODELS_DESIGN.md` §3.2 (`V2-M2L-2`):
- Fixes AIcrowd AOT coordinate trap: parses `[left, top, width, height]` (`x, y, w, h`) and asserts
  `0 <= left + width <= 2448` and `0 <= top + height <= 2048`.
- Flags fixed `6x6` and `16x16` AOT circle-approximation labels with `m_box = 0.0` so scale
  supervision never collapses onto discrete plateaus.
- Groups flights atomically by `capture_day` extracted from the first 19-char nanosecond Unix
  timestamp (`ts_ns // 86_400_000_000_000`) -> UTC date, guaranteeing ZERO flight-day leakage.
- Pure Python stdlib `tarfile` WebDataset shard writer/reader + deterministic `70% AOT + 30% G3a`
  batch mixer (zero extra pip dependencies!).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import random
import tarfile
from typing import Any

import numpy as np

AOT_IMG_WIDTH = 2448
AOT_IMG_HEIGHT = 2048
NS_PER_DAY = 86_400_000_000_000


@dataclass(frozen=True)
class AotParsedBox:
    """Validated AOT bounding box parsed strictly as `[left, top, width, height]`."""

    left: float
    top: float
    width: float
    height: float
    cx: float
    cy: float
    is_circle_approx: bool
    m_box: float


def is_aot_circle_approx(width: float, height: float, tol: float = 0.25) -> bool:
    """Return `True` if `(width, height)` matches AOT's fixed 6x6 or 16x16 circle approximations."""
    w = float(width)
    h = float(height)
    is_6 = abs(w - 6.0) <= tol and abs(h - 6.0) <= tol
    is_16 = abs(w - 16.0) <= tol and abs(h - 16.0) <= tol
    return bool(is_6 or is_16)


def parse_aot_bbox_ltwh(
    bb: list[float] | tuple[float, float, float, float],
    *,
    img_w: int = AOT_IMG_WIDTH,
    img_h: int = AOT_IMG_HEIGHT,
) -> AotParsedBox:
    """Parse and bounds-check an AOT `[left, top, width, height]` box (`DR-01` Trap #1)."""
    if len(bb) != 4:
        raise ValueError(f"AOT bbox must have 4 elements [left, top, width, height], got {len(bb)}")
    left, top, width, height = (float(v) for v in bb)
    if not all(np.isfinite([left, top, width, height])):
        raise ValueError(f"Non-finite AOT bbox values: {bb!r}")
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"AOT bbox width and height must be > 0, got w={width}, h={height}")
    if left < 0.0 or top < 0.0 or (left + width) > float(img_w) + 1e-6 or (top + height) > float(img_h) + 1e-6:
        raise ValueError(
            f"AOT bbox [left={left}, top={top}, w={width}, h={height}] out of bounds "
            f"for {img_w}x{img_h} image"
        )
    circle = is_aot_circle_approx(width, height)
    return AotParsedBox(
        left=left,
        top=top,
        width=width,
        height=height,
        cx=left + 0.5 * width,
        cy=top + 0.5 * height,
        is_circle_approx=circle,
        m_box=0.0 if circle else 1.0,
    )


def extract_capture_day_utc(filename: str) -> str:
    """Extract UTC `YYYY-MM-DD` capture day from AOT `<19-char-ns-timestamp><32-char-uuid>.png`."""
    base = Path(filename).name
    if len(base) < 19 or not base[:19].isdigit():
        raise ValueError(f"AOT filename must begin with a 19-digit ns timestamp, got {filename!r}")
    ts_ns = int(base[:19])
    day_index = ts_ns // NS_PER_DAY
    dt = datetime.fromtimestamp(day_index * 86400, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")


def split_flights_by_capture_day(
    flight_records: list[dict[str, Any]],
    *,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    pinned_test_days: set[str] | None = None,
    seed: int = 20261121,
) -> dict[str, list[dict[str, Any]]]:
    """Partition flights atomically by `capture_day` with ZERO date overlap across splits (`DR-01:108`).

    Each record in `flight_records` must provide `flight_id` and either `capture_day` or `sample_filename`.
    """
    by_day: dict[str, list[dict[str, Any]]] = {}
    for rec in flight_records:
        if "capture_day" in rec and rec["capture_day"]:
            day = str(rec["capture_day"])
        elif "sample_filename" in rec:
            day = extract_capture_day_utc(str(rec["sample_filename"]))
        else:
            raise ValueError(f"Flight record missing 'capture_day' or 'sample_filename': {rec!r}")
        by_day.setdefault(day, []).append({**rec, "capture_day": day})

    pinned = set(pinned_test_days) if pinned_test_days is not None else set()
    all_days = sorted(by_day.keys())
    free_days = [d for d in all_days if d not in pinned]

    rng = random.Random(int(seed))
    rng.shuffle(free_days)

    n_total = len(all_days)
    n_test_target = max(len(pinned), int(round(n_total * max(0.0, 1.0 - train_frac - val_frac))))
    n_test_extra = max(0, min(len(free_days), n_test_target - len(pinned)))
    test_days = sorted(pinned.intersection(by_day.keys())) + free_days[:n_test_extra]

    remaining = free_days[n_test_extra:]
    n_val = max(1 if len(remaining) >= 2 and val_frac > 0.0 else 0, int(round(n_total * val_frac)))
    n_val = min(n_val, max(0, len(remaining) - 1)) if len(remaining) >= 2 else min(n_val, len(remaining))
    val_days = remaining[:n_val]
    train_days = remaining[n_val:]

    # Programmatic invariant check: zero capture-day overlap
    set_tr, set_va, set_te = set(train_days), set(val_days), set(test_days)
    assert set_tr.isdisjoint(set_va) and set_tr.isdisjoint(set_te) and set_va.isdisjoint(set_te)

    return {
        "train": [r for d in sorted(train_days) for r in by_day[d]],
        "val": [r for d in sorted(val_days) for r in by_day[d]],
        "test": [r for d in sorted(test_days) for r in by_day[d]],
    }


def write_wds_shard(
    shard_path: Path,
    samples: list[dict[str, Any]],
) -> Path:
    """Write a POSIX `.tar` WebDataset shard using Python stdlib `tarfile` (zero external deps).

    Each sample dict must contain:
      - `key`: unique string prefix (e.g. `aot_000001`)
      - `x_t`: numpy array `[6, H, W]` (stored as `<key>.x_t.npy`)
      - `meta`: JSON-serializable dict (stored as `<key>.json`)
    """
    shard_path = Path(shard_path)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(shard_path, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for sample in samples:
            key = str(sample["key"])
            arr = np.asarray(sample["x_t"], dtype=np.float32)
            buf = io.BytesIO()
            np.save(buf, arr, allow_pickle=False)
            arr_bytes = buf.getvalue()
            ti_arr = tarfile.TarInfo(name=f"{key}.x_t.npy")
            ti_arr.size = len(arr_bytes)
            tf.addfile(ti_arr, io.BytesIO(arr_bytes))

            meta_bytes = json.dumps(sample["meta"], sort_keys=True).encode("utf-8")
            ti_meta = tarfile.TarInfo(name=f"{key}.json")
            ti_meta.size = len(meta_bytes)
            tf.addfile(ti_meta, io.BytesIO(meta_bytes))
    return shard_path


def read_wds_shard(shard_path: Path) -> list[dict[str, Any]]:
    """Read all `(key, x_t, meta)` samples from a POSIX `.tar` WebDataset shard via stdlib `tarfile`."""
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    with tarfile.open(Path(shard_path), mode="r") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            raw = fobj.read()
            if member.name.endswith(".x_t.npy"):
                key = member.name[: -len(".x_t.npy")]
                if key not in grouped:
                    grouped[key] = {"key": key}
                    order.append(key)
                grouped[key]["x_t"] = np.load(io.BytesIO(raw), allow_pickle=False)
            elif member.name.endswith(".json"):
                key = member.name[: -len(".json")]
                if key not in grouped:
                    grouped[key] = {"key": key}
                    order.append(key)
                grouped[key]["meta"] = json.loads(raw.decode("utf-8"))
    return [grouped[k] for k in order]


class AotG3aBatchMixer:
    """Deterministic `70% Real AOT + 30% Synthetic G3a` clip mixer (`DR-01:116`)."""

    def __init__(
        self,
        aot_samples: list[dict[str, Any]],
        g3a_samples: list[dict[str, Any]],
        *,
        aot_ratio: float = 0.70,
        seed: int = 20261121,
    ) -> None:
        if not aot_samples or not g3a_samples:
            raise ValueError("Both aot_samples and g3a_samples must be non-empty")
        if not (0.0 < float(aot_ratio) < 1.0):
            raise ValueError(f"aot_ratio must be in (0, 1), got {aot_ratio}")
        self.aot_samples = list(aot_samples)
        self.g3a_samples = list(g3a_samples)
        self.aot_ratio = float(aot_ratio)
        self._rng = random.Random(int(seed))

    def sample_batch(self, batch_size: int) -> list[dict[str, Any]]:
        """Draw an exact stratified batch with `round(batch_size * 0.70)` AOT and remainder G3a."""
        bs = int(batch_size)
        n_aot = int(round(bs * self.aot_ratio))
        n_g3a = bs - n_aot
        batch: list[dict[str, Any]] = []
        for _ in range(n_aot):
            s = self.aot_samples[self._rng.randrange(len(self.aot_samples))]
            batch.append({**s, "source_domain": "aot"})
        for _ in range(n_g3a):
            s = self.g3a_samples[self._rng.randrange(len(self.g3a_samples))]
            batch.append({**s, "source_domain": "g3a"})
        self._rng.shuffle(batch)
        return batch
