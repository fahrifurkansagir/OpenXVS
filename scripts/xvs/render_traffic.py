"""WP-02 G3a — dataset v0 renderer (WP02_TRAFFIC.md v1.3 §3 G3a, §2.4, §2.6, §5).

Extends the WP-01 sequence runner (render_sequence.py) to traffic: one intruder per encounter, ownship from a
scenario, frames at t = k / rate over the scenario duration.

Frame selection (§3 G3a): a frame is kept only if the intruder's projected centre lies inside at least one
ring-camera image and the range is <= 10 km; skipped frames are counted per reason.

Images per kept frame:
  ring cameras containing the centre: a ROI (§2.4) with rgb (MSAA 4, imagery through the proxy), mask_amodal (A)
    and mask_visible (V);
  rig cameras: A and V wherever the analytic box meets the image (or the mesh crosses the camera plane, where the
    analytic box is undefined); rgb only when the A mask has >= 16 px (skips are recorded);
  every 5th kept frame: one negative rgb ROI in a ring camera that contains no intruder (seeded camera and
    top-left; the model is also hidden on that render).
GT per kept frame: §2.6 analytic fields for all 14 cameras; mask-derived fields, WP-01 horizon GT, GT-2 v1.1
raster result (A masks), GT-3(b) and mask purity for every rendered camera; the ROI (u0, v0) of ring renders.

Pages: one Cesium page per (profile, camera kind) = {rgb, A, V} x {rig, ring}; the encounter's model is loaded
once per encounter on each page. The stable-frame rule (models ready, proxy idle) is the renderer's.

Blocks and jobs: --block i/n takes encounters i, i+n, i+2n, ... (round-robin). Each block has its own
blocks/b<i>of<n>/ folder with frames.jsonl, encounters_done.jsonl, perf.json, data_sources.json,
data_violations.jsonl (appended by the proxy, never truncated), proxy_log.jsonl and run_manifest.json. A job
stops cleanly after --max-wall-h hours; re-running the same command resumes (finished encounters are skipped,
frames already recorded with matching file SHA-256 are kept).

Other modes:
  --plan <json>    analytic frame counts per rate for all encounters (no browser), for the r_ds projection;
  --frames <json>  render only the listed (encounter_id, k) frames (determinism / ref spot checks).
Encounter records: sampler v1 lines (EncounterParams) or sampled tracks ({"track": {...}}, xvs.traffic.track).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.geo import wgs84  # noqa: E402
from xvs.rig.rig import Rig  # noqa: E402
from xvs.sim import airports, trajectory  # noqa: E402
from xvs.sim.horizon import compute_horizon_gt  # noqa: E402
from xvs.sim.renderer import XVSRenderer  # noqa: E402
from xvs.sim.server import XVSServer  # noqa: E402
from xvs.traffic import raster  # noqa: E402
from xvs.traffic.encounter import EncounterParams, construct_encounter, evaluate_actual_cpa, evaluate_events  # noqa: E402
from xvs.traffic.gt import compute_intruder_analytic_gt  # noqa: E402
from xvs.traffic.ring import TrafficRing  # noqa: E402
from xvs.traffic.track import TrackTrajectory  # noqa: E402
from load_logger import LoadLogger, summarise as summarise_load  # noqa: E402  (scripts/xvs, D26)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FRAME_SCHEMA = "xvs.traffic_frame/1"
RANGE_MAX_M = 10000.0
RIG_RGB_MIN_AMODAL_PX = 16
NEGATIVE_EVERY = 5
NEGATIVE_SEED = 20260924
RIG_CAMS = ("F", "R", "B", "L", "U", "D")
RING_CAMS = ("T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7")
PASSES = ("rgb", "A", "V")
PROFILE = {"rgb": "rgb", "A": "mask_amodal", "V": "mask_visible"}
SHADER = {"rgb": None, "A": "unlit_white", "V": "unlit_white"}
ROI = 1024
MODULES = ("src/xvs/traffic/raster.py", "src/xvs/traffic/gt.py", "src/xvs/traffic/encounter.py", "src/xvs/traffic/ring.py",
           "src/xvs/traffic/track.py", "src/xvs/sim/horizon.py", "src/xvs/sim/renderer.py", "src/xvs/sim/server.py",
           "src/xvs/rig/camera.py", "src/xvs/rig/rig.py", "src/xvs/geo/wgs84.py")


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------
def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git_state() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            return f"unavailable: {exc}"
    return {"head": run(["rev-parse", "HEAD"]), "branch": run(["rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty_files": [ln for ln in run(["status", "--porcelain", "--untracked-files=no"]).splitlines() if ln][:200]}


def load_records(path: Path) -> list[dict[str, Any]]:
    recs = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    ids = [r["id"] for r in recs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate encounter ids")
    return recs


def block_records(recs: list[dict[str, Any]], i: int, n: int) -> list[dict[str, Any]]:
    """Round-robin block i of n (0-based): encounters i, i+n, i+2n, ..."""
    if not (0 <= i < n):
        raise ValueError(f"block {i}/{n} out of range")
    return recs[i::n]


def ownship_from_scenario(cfg: dict[str, Any]):
    kind = cfg["trajectory_type"]
    if kind == "t1_cruise":
        return trajectory.t1_cruise_state
    if kind == "t2_ksfo28l":
        thr = airports.load_ksfo_28l_threshold()
        return lambda t: trajectory.t2_ksfo28l_state(t, thr=thr)
    raise ValueError(f"unsupported ownship trajectory '{kind}'")


def make_trajectory(rec: dict[str, Any], own_fn):
    if "track" in rec:
        return TrackTrajectory.from_record(rec)
    fields = EncounterParams.__dataclass_fields__
    return construct_encounter(own_fn, EncounterParams(**{k: v for k, v in rec.items() if k in fields}))


def enc_hash(enc_id: str) -> int:
    return int(hashlib.sha256(enc_id.encode("utf-8")).hexdigest()[:8], 16)


def clean(x: Any) -> Any:
    """JSON-safe copy: NaN / inf -> None, numpy scalars -> Python."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return f if math.isfinite(f) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# ---------------------------------------------------------------------------------------------
# Geometry helpers (pure; unit-tested)
# ---------------------------------------------------------------------------------------------
def camera_set(rig: Rig, ring: TrafficRing, st: Any) -> dict[str, dict[str, Any]]:
    rig_poses = rig.compute_poses(st.lat_deg, st.lon_deg, st.h_m, st.yaw_rad, st.pitch_rad, st.roll_rad)
    ring_poses = ring.compute_poses(st.lat_deg, st.lon_deg, st.h_m, st.yaw_rad, st.pitch_rad, st.roll_rad)
    cams: dict[str, dict[str, Any]] = {}
    for cid in RIG_CAMS:
        p = rig_poses[cid]
        cams[cid] = {"kind": "rig", "pose": p, "model": rig.camera_model, "fov_rad": float(p["fov_rad"]),
                     "p": np.array(p["position"], float), "R": np.array(p["r_ecef_cam"], float)}
    for cid in RING_CAMS:
        p = ring_poses[cid]
        cams[cid] = {"kind": "ring", "pose": p, "model": ring.camera_model, "fov_rad": math.radians(ring.fov_h_deg),
                     "p": np.array(p["position"], float), "R": np.array(p["r_ecef_cam"], float)}
    return cams


def full_window(model: Any) -> tuple[float, float, float, float]:
    return (-0.5, -0.5, model.width - 0.5, model.height - 0.5)


def box_meets(box: list[float] | None, win: tuple[float, float, float, float]) -> bool:
    return box is not None and box[2] >= win[0] and box[0] <= win[2] and box[3] >= win[1] and box[1] <= win[3]


def select_frame(range_m: float, ring_in_image: dict[str, bool], range_max_m: float = RANGE_MAX_M) -> str | None:
    """None if the frame is kept, else the skip reason (§3 G3a frame selection)."""
    if not (range_m <= range_max_m):
        return "range_gt_10km"
    if not any(ring_in_image.values()):
        return "centre_not_in_ring"
    return None


def straddles(z: np.ndarray, z_near: float = raster.Z_NEAR_M) -> bool:
    return bool(np.any(z >= z_near) and np.any(z < z_near))


def negative_choice(enc_id: str, k: int, ring_free: list[str], width: int, height: int) -> tuple[str, int, int] | None:
    """Seeded negative ROI: a ring camera without the intruder and a uniform ROI top-left (§3 G3a)."""
    if not ring_free:
        return None
    rng = np.random.default_rng([NEGATIVE_SEED, enc_hash(enc_id), int(k)])
    cid = ring_free[int(rng.integers(0, len(ring_free)))]
    u0 = int(rng.integers(0, width - ROI + 1))
    v0 = int(rng.integers(0, height - ROI + 1))
    return cid, u0, v0


def mask_stats(m01: np.ndarray, u0: int, v0: int) -> dict[str, Any]:
    rows, cols = np.nonzero(m01)
    if rows.size == 0:
        return {"px": 0, "box_px": None, "centroid": None}
    return {"px": int(rows.size),
            "box_px": [int(cols.min()) + u0, int(rows.min()) + v0, int(cols.max()) + u0, int(rows.max()) + v0],
            "centroid": [float(cols.mean()) + u0, float(rows.mean()) + v0]}


def mask_grey(img: np.ndarray) -> tuple[np.ndarray, int]:
    """(grey uint8, impure pixel count): A/V masks must contain only (0,0,0) and (255,255,255)."""
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    pure = (r == g) & (g == b) & ((r == 0) | (r == 255))
    return r.copy(), int(np.count_nonzero(~pure))


def frame_plan(gt: Any, cams: dict[str, dict[str, Any]], z_by_cam: dict[str, np.ndarray]) -> dict[str, Any]:
    """Which cameras a kept frame renders (analytic only)."""
    ring = [cid for cid in RING_CAMS if gt.cams[cid].in_image]
    rig = [cid for cid in RIG_CAMS
           if box_meets(gt.cams[cid].box_analytic, full_window(cams[cid]["model"])) or straddles(z_by_cam[cid])]
    ring_free = [cid for cid in RING_CAMS
                 if not gt.cams[cid].in_image and not straddles(z_by_cam[cid])
                 and not box_meets(gt.cams[cid].box_analytic, full_window(cams[cid]["model"]))]
    return {"ring": ring, "rig": rig, "ring_free": ring_free}


# ---------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------
class PageSet:
    def __init__(self, backend: str, log_dir: Path, offline: bool, fresh_rgb: bool = True):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.server = XVSServer(
            server_address=("127.0.0.1", self.port),
            cesium_build_dir=REPO_ROOT / "sim" / "cesium" / "node_modules" / "cesium" / "Build" / "Cesium",
            page_dir=REPO_ROOT / "sim" / "cesium" / "page",
            cache_dir=REPO_ROOT / "data" / "cache" / "usgs_imagery",
            log_dir=log_dir,
            offline=offline,
        )
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.r = XVSRenderer(port=self.port, backend=backend, server=self.server).start()
        self.pages: dict[tuple[str, str], Any] = {}
        self.resolved: dict[str, Any] = {}
        for ps in PASSES:
            for kind in ("rig", "ring"):
                self.pages[(ps, kind)] = self.r.new_camera_page(f"g3a_{ps}_{kind}", profile=PROFILE[ps])
                self.resolved[f"{ps}_{kind}"] = self.r.resolved_settings
        self.loaded: str | None = None
        self.model_load_ms: list[float] = []
        self.fresh_rgb = fresh_rgb
        self.rgb_pages_opened = 2
        self.log_dir = log_dir
        self.clock: str | None = None
        self.d26a = {"retries": 0, "violations": 0, "retry_ok": 0}

    def renew_rgb_page(self, kind: str) -> None:
        """A fresh rgb page of one camera kind, with the current model and clock (D25; D26a retry)."""
        old = self.pages[("rgb", kind)]
        pg = self.r.new_camera_page(f"g3a_rgb_{kind}", profile=PROFILE["rgb"])
        self.pages[("rgb", kind)] = pg
        old.close()
        self.rgb_pages_opened += 1
        if self.loaded is not None:
            self.r.add_model(url=f"http://127.0.0.1:{self.port}/{self.loaded}", custom_shader=SHADER["rgb"], page=pg)
        if self.clock is not None:
            self.r.set_clock(self.clock, page=pg)

    def start_encounter(self) -> None:
        """D25 (was D24 fix candidate 1): new rgb pages for every encounter, so the imagery tile state of a page never
        carries over from earlier encounters. Long-lived rgb pages are not allowed until the cause is known. The mask
        pages stay resident (no imagery; masks are bit-exact)."""
        if not self.fresh_rgb:
            return
        for kind in ("rig", "ring"):
            old = self.pages[("rgb", kind)]
            self.pages[("rgb", kind)] = self.r.new_camera_page(f"g3a_rgb_{kind}", profile=PROFILE["rgb"])
            old.close()
            self.rgb_pages_opened += 1
        self.loaded = None  # the new pages have no model yet

    def tiles(self) -> dict[str, int]:
        with self.server._stats_lock:
            st = self.server.stats["usgs_imagery"]
            return {"served": st["tiles_served_total"], "upstream": st["tiles_upstream_fetches"],
                    "errors": st["tile_errors"], "bytes_up": st["bytes_upstream_fetched"]}

    def usgs_stats(self) -> dict[str, Any]:
        with self.server._stats_lock:
            return json.loads(json.dumps(self.server.stats["usgs_imagery"]))

    def load(self, rel_url: str) -> None:
        if self.loaded == rel_url:
            return
        t0 = time.perf_counter()
        for (ps, _kind), pg in self.pages.items():
            self.r.clear_models(page=pg)
            self.r.add_model(url=f"http://127.0.0.1:{self.port}/{rel_url}", custom_shader=SHADER[ps], page=pg)
        self.model_load_ms.append((time.perf_counter() - t0) * 1000.0)
        self.loaded = rel_url

    def set_clock(self, utc: str) -> None:
        self.clock = utc
        for kind in ("rig", "ring"):
            self.r.set_clock(utc, page=self.pages[("rgb", kind)])

    def render(self, ps: str, cam: dict[str, Any], roi: tuple[int, int] | None, model_m: list[float],
               hide_model: bool = False) -> tuple[Any, dict[str, Any]]:
        """One render. rgb (D26a): captured only with the globe's tiles loaded and no pending tile requests; if that
        is not reached within the render cap, the frame is retried once in a fresh page, and a second failure is
        recorded as a violation in render_violations.jsonl (never a silent capture)."""
        res, info = self._render_once(ps, cam, roi, model_m, hide_model)
        if ps != "rgb" or not res.incomplete:
            return res, info
        self.d26a["retries"] += 1
        first = {k: info[k] for k in ("n_renders", "tiles_requested", "tiles_cache_miss")}
        self.renew_rgb_page(cam["kind"])
        res, info = self._render_once(ps, cam, roi, model_m, hide_model)
        info["d26a_retry"] = {"first_attempt": first, "ok": not res.incomplete}
        if res.incomplete:
            self.d26a["violations"] += 1
            with open(self.log_dir / "render_violations.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "rule": "D26a", "kind": cam["kind"],
                                    "roi": list(roi) if roi else None, "clock": self.clock, "model": self.loaded,
                                    "hide_model": hide_model, "tile_state": res.tile_state, "n_renders": res.n_renders}) + "\n")
        else:
            self.d26a["retry_ok"] += 1
        return res, info

    def _render_once(self, ps: str, cam: dict[str, Any], roi: tuple[int, int] | None, model_m: list[float],
                     hide_model: bool = False) -> tuple[Any, dict[str, Any]]:
        pg = self.pages[(ps, cam["kind"])]
        self.r.set_model_pose(0, model_m, page=pg)
        if hide_model:
            self.r.set_model_show(0, False, page=pg)
        try:
            p = cam["pose"]
            _readback, ms_set = self.r.set_camera(position=p["position"], direction=p["direction"], up=p["up"],
                                                  fov_rad=cam["fov_rad"], aspect_ratio=1.0, page=pg)
            if roi is not None:
                m = cam["model"]
                self.r.set_roi_frustum(u0=roi[0], v0=roi[1], fx=m.fx, fy=m.fy, cx=m.cx, cy=m.cy, page=pg)
            before = self.tiles()
            res = self.r.render_frame(page=pg, require_no_pending=(ps == "rgb"))
            after = self.tiles()
        finally:
            if hide_model:
                self.r.set_model_show(0, True, page=pg)
        info = {"n_renders": res.n_renders, "incomplete": res.incomplete, "ms_set": ms_set,
                "ms_render": res.ms_render_loop, "ms_readback": res.ms_readback, "rgb_sha256": res.rgb_sha256,
                "tiles_requested": after["served"] - before["served"], "tiles_cache_miss": after["upstream"] - before["upstream"],
                "tile_errors": after["errors"] - before["errors"], "bytes_upstream": after["bytes_up"] - before["bytes_up"]}
        if res.tile_state is not None:
            info["tile_state"] = res.tile_state
        return res, info

    def close(self) -> None:
        try:
            self.r.close()
        finally:
            self.server.shutdown()


def save_png(arr: np.ndarray, path: Path, mode: str) -> tuple[int, float, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    Image.fromarray(arr, mode=mode).save(path, format="PNG", compress_level=6)
    ms = (time.perf_counter() - t0) * 1000.0
    data = path.read_bytes()
    return len(data), ms, hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------------------------
# Output bookkeeping
# ---------------------------------------------------------------------------------------------
def read_jsonl(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                pass  # a line cut by an interrupted job; the frame is re-rendered
    return out


def ensure_trailing_newline(p: Path) -> bool:
    """End a frames file cut by a crash with a newline, so the next appended record starts on its own line.
    Returns True if a newline was added."""
    if not p.exists() or p.stat().st_size == 0:
        return False
    with open(p, "rb") as f:
        f.seek(-1, 2)
        if f.read(1) == b"\n":
            return False
    with open(p, "ab") as f:
        f.write(b"\n")
    return True


def add_resumed_counts(summ: dict[str, Any], rec: dict[str, Any]) -> None:
    """Count a frame kept by an earlier job into the encounter summary, the same way render_encounter counts a frame
    it renders, so an encounter resumed mid-way reports all its frames (D16 review R-G3a-6)."""
    cams = rec["traffic"][0]["cams"]
    for cid, rr in rec["renders"].items():
        summ["images"]["A"] += 1
        summ["images"]["V"] += 1
        if rr.get("rgb"):
            summ["images"]["rgb_ring" if cams[cid]["kind"] == "ring" else "rgb_rig"] += 1
    summ["rig_rgb_skipped_small"] += len(rec.get("rig_rgb_skipped_small") or [])
    if rec.get("negative"):
        summ["negatives"] += 1
        summ["images"]["negative"] += 1
    elif rec["kept_index"] % NEGATIVE_EVERY == 0:
        summ["negatives_none_free"] += 1
    summ["bytes"] += rec["bytes"]


def frame_files_valid(root: Path, rec: dict[str, Any]) -> bool:
    for f in rec.get("files", []):
        p = root / f["path"]
        if not p.exists() or sha256_file(p) != f["sha256"]:
            return False
    return True


def stats(v: list[float]) -> dict[str, float] | None:
    if not v:
        return None
    a = np.asarray(v, dtype=np.float64)
    return {"n": int(a.size), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "mean": float(a.mean()), "max": float(a.max()), "sum": float(a.sum())}


def write_sequence_json(out: Path, args: argparse.Namespace, recs: list[dict[str, Any]], ring: TrafficRing, rig: Rig,
                        scenario: dict[str, Any], n_frames: int) -> None:
    enc_path = Path(args.encounters)
    models = {}
    for r in recs:
        glb = REPO_ROOT / "assets" / "traffic" / r["class_id"] / f"{r['model_id']}.glb"
        models[r["model_id"]] = {"path": glb.relative_to(REPO_ROOT).as_posix(), "sha256": sha256_file(glb)}
    sampler = None
    if args.sampler_config:
        sp = Path(args.sampler_config)
        scfg = json.loads(sp.read_text(encoding="utf-8"))
        if scfg.get("encounters_sha256") != sha256_file(enc_path):
            raise RuntimeError(f"{enc_path}: SHA-256 differs from {sp} encounters_sha256")
        sampler = {"path": sp.as_posix(), "sha256": sha256_file(sp), "config": scfg}
    seq = {
        "schema": "xvs.traffic_sequence/1", "spec": "WP02_TRAFFIC.md v1.3 §3 G3a",
        "encounters": {"path": enc_path.as_posix(), "sha256": sha256_file(enc_path), "n": len(recs)},
        "sampler": sampler,
        "ownship": {"path": Path(args.ownship).as_posix(), "sha256": sha256_file(Path(args.ownship)), "scenario": scenario},
        "rate_hz": args.rate_hz, "n_frames_per_encounter": n_frames,
        "selection": {"range_max_m": RANGE_MAX_M, "centre_in_ring_image": True},
        "images": {"ring": "ROI rgb (MSAA 4) + mask_amodal + mask_visible", "rig": "A + V where the analytic box meets the image; rgb if A >= 16 px",
                   "negatives": f"every {NEGATIVE_EVERY}th kept frame, seeded ring camera without the intruder, rgb only (seed {NEGATIVE_SEED})",
                   "rig_rgb_min_amodal_px": RIG_RGB_MIN_AMODAL_PX},
        "naming": {"masks": "masks/<cam>/<k:06d>_<enc>_<A|V>.png", "rgb": "rgb/<cam>/<k:06d>_<enc>.png",
                   "negatives": "negatives/<cam>/<k:06d>_<enc>.png"},
        "ring": ring.spec, "rig": rig.record(),
        "models": models,
        "script_sha256": sha256_file(Path(__file__)),
        "page_sha256": sha256_file(REPO_ROOT / "sim" / "cesium" / "page" / "xvs_page.js"),
        "profiles_sha256": sha256_file(REPO_ROOT / "configs" / "xvs" / "render_profiles.json"),
        "module_sha256": {m: sha256_file(REPO_ROOT / m) for m in MODULES},
    }
    seq["rgb_pages"] = "fresh per encounter (D25)"
    p = out / "sequence.json"
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8"))
        keys = ("encounters", "ownship", "rate_hz", "n_frames_per_encounter", "models", "script_sha256", "page_sha256", "profiles_sha256",
                "module_sha256", "rgb_pages")
        diff = [k for k in keys if old.get(k) != json.loads(json.dumps(seq.get(k)))]
        if diff:
            raise RuntimeError(f"{p} exists with different {diff}: use a new --out (no silent mixing of runs)")
        return
    p.write_text(json.dumps(clean(seq), indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# One encounter
# ---------------------------------------------------------------------------------------------
def render_encounter(ctx: dict[str, Any], rec: dict[str, Any], existing: dict[tuple[str, int], dict[str, Any]],
                     frames_file: Any, only: set[tuple[str, int]] | None, acc: dict[str, Any]) -> dict[str, Any]:
    own_fn, rig, ring, ps_ = ctx["own_fn"], ctx["rig"], ctx["ring"], ctx["pages"]
    out, rate, n_frames, start_dt = ctx["out"], ctx["rate"], ctx["n_frames"], ctx["start_dt"]
    enc_id = rec["id"]
    traj = make_trajectory(rec, own_fn)
    is_track = isinstance(traj, TrackTrajectory)
    t0e, t1e = (max(0.0, traj.t_first), min(ctx["duration"], traj.t_last)) if is_track else (0.0, ctx["duration"])
    cpa = evaluate_actual_cpa(own_fn, traj, t0e, t1e, 0.01)
    events = evaluate_events(own_fn, traj, t0e, t1e, 0.01)
    V, F = ctx["mesh"](rec["class_id"], rec["model_id"])
    max_dim = float(np.max(np.ptp(V, axis=0)))
    model_rel = f"assets/traffic/{rec['class_id']}/{rec['model_id']}.glb"
    if ps_ is not None and (only is None or any(e == enc_id for e, _ in only)):
        ps_.start_encounter()
        ps_.load(model_rel)

    summ = {"encounter_id": enc_id, "class_id": rec["class_id"], "model_id": rec["model_id"],
            "stratum": rec.get("stratum"), "weight": rec.get("weight", 1.0), "frames_total": n_frames, "kept": 0,  # default-ok: optional fallback default
            "skipped": {"range_gt_10km": 0, "centre_not_in_ring": 0, "outside_track": 0},
            "rendered": 0, "resumed": 0, "negatives": 0, "negatives_none_free": 0, "rig_rgb_skipped_small": 0,
            "images": {"rgb_ring": 0, "rgb_rig": 0, "A": 0, "V": 0, "negative": 0}, "bytes": 0}
    kept_idx = 0
    for k in range(n_frames):
        t_s = k / rate
        if is_track and not traj.covers(t_s):
            summ["skipped"]["outside_track"] += 1
            continue
        st = own_fn(t_s)
        cams = camera_set(rig, ring, st)
        v_o = wgs84.r_ecef_ned(st.lat_deg, st.lon_deg) @ st.v_ned_mps
        gt = compute_intruder_analytic_gt(t_s, traj, st.p_ecef_m, v_o, st.r_ecef_body,
                                          {cid: (c["p"], c["R"], c["model"]) for cid, c in cams.items()},
                                          model_vertices=V, max_body_dim_m=max_dim)
        reason = select_frame(gt.range_m, {cid: gt.cams[cid].in_image for cid in RING_CAMS})
        if reason is not None:
            summ["skipped"][reason] += 1
            continue
        kept_idx += 1
        summ["kept"] += 1
        key = (enc_id, k)
        if only is not None and key not in only:
            continue
        if key in existing and frame_files_valid(out, existing[key]):
            summ["resumed"] += 1
            add_resumed_counts(summ, existing[key])
            continue
        if ps_ is None:  # plan-only caller never gets here
            continue

        t_frame = time.perf_counter()
        p_i = traj.position_at(t_s)
        r_i, _ = traj.attitude_at(t_s)
        m4 = np.eye(4)
        m4[:3, :3] = r_i
        m4[:3, 3] = p_i
        model_m = m4.T.flatten().tolist()  # Cesium column-major
        pw = p_i[None, :] + V @ r_i.T
        pc_by = {cid: (pw - c["p"][None, :]) @ c["R"] for cid, c in cams.items()}
        plan = frame_plan(gt, cams, {cid: pc[:, 2] for cid, pc in pc_by.items()})
        frame_dt = start_dt + timedelta(microseconds=int(round(t_s * 1e6)))
        utc = frame_dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{frame_dt.microsecond // 1000:03d}Z"
        ps_.set_clock(utc)

        files: list[dict[str, Any]] = []
        renders: dict[str, dict[str, Any]] = {}
        cam_rec: dict[str, dict[str, Any]] = {}
        horizon: dict[str, Any] = {}
        ms_gt = 0.0
        skipped_rig_rgb: list[str] = []

        def do(ps: str, cid: str, roi: tuple[int, int] | None, rel: str, hide: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
            res, info = ps_.render(ps, cams[cid], roi, model_m, hide_model=hide)
            if ps in ("A", "V"):
                grey, impure = mask_grey(res.image)
                nbytes, ms_enc, sha = save_png(grey, out / rel, "L")
                info["impure_px"] = impure
                arr = grey
            else:
                nbytes, ms_enc, sha = save_png(res.image, out / rel, "RGB")
                arr = res.image
            info.update({"png": rel, "sha256": sha, "bytes": nbytes, "ms_encode": ms_enc})
            files.append({"path": rel, "sha256": sha})
            acc["renders"].append({"pass": "neg" if hide else ps, "kind": cams[cid]["kind"], **{k2: info[k2] for k2 in
                                   ("n_renders", "ms_set", "ms_render", "ms_readback", "ms_encode", "bytes", "tiles_requested",
                                    "tiles_cache_miss", "bytes_upstream")}})
            if info["incomplete"]:
                acc["incomplete"] += 1
            return arr, info

        for cid in plan["ring"] + plan["rig"]:
            c = cams[cid]
            g = gt.cams[cid]
            roi = ring.compute_roi(g.u, g.v) if c["kind"] == "ring" else None
            u0, v0 = roi if roi else (0, 0)
            base = f"{k:06d}_{enc_id}"
            a_grey, a_info = do("A", cid, roi, f"masks/{cid}/{base}_A.png")
            v_grey, v_info = do("V", cid, roi, f"masks/{cid}/{base}_V.png")
            a01, v01 = a_grey > 127, v_grey > 127
            sa, sv = mask_stats(a01, u0, v0), mask_stats(v01, u0, v0)
            tg = time.perf_counter()
            m = c["model"]
            gt2 = raster.compare_with_mask(a01, pc_by[cid], F, m.fx, m.fy, m.cx, m.cy, u0, v0)
            if gt2.get("ref_box_px"):
                b = gt2["ref_box_px"]
                gt2["ref_box_px"] = [b[0] + u0, b[1] + v0, b[2] + u0, b[3] + v0]
            outside_box = None
            box = g.box_analytic
            if sa["px"] == 0:
                outside_box = 0
            elif box is not None:
                ys, xs = np.nonzero(a01)
                tol = raster.TIE_PX
                xs, ys = xs + u0, ys + v0
                outside_box = int(np.count_nonzero((xs < box[0] - tol) | (xs > box[2] + tol) | (ys < box[1] - tol) | (ys > box[3] + tol)))
            horizon[cid] = compute_horizon_gt(c["p"], c["R"], m)
            ms_gt += (time.perf_counter() - tg) * 1000.0
            summ["images"]["A"] += 1
            summ["images"]["V"] += 1
            rgb_info = None
            if c["kind"] == "ring" or sa["px"] >= RIG_RGB_MIN_AMODAL_PX:
                _img, rgb_info = do("rgb", cid, roi, f"rgb/{cid}/{base}.png")
                summ["images"]["rgb_ring" if c["kind"] == "ring" else "rgb_rig"] += 1
            else:
                skipped_rig_rgb.append(cid)
                summ["rig_rgb_skipped_small"] += 1
            renders[cid] = {"A": a_info, "V": v_info, "rgb": rgb_info}
            cam_rec[cid] = {
                "roi": list(roi) if roi else None,
                "amodal_px": sa["px"], "box_amodal_px": sa["box_px"], "centroid_amodal": sa["centroid"],
                "visible_px": sv["px"], "box_px": sv["box_px"],
                "visibility": (sv["px"] / sa["px"]) if sa["px"] else None,
                "visible_outside_amodal_px": int(np.count_nonzero(v01 & ~a01)),
                "mask_A": a_info["png"], "mask_V": v_info["png"], "rgb": rgb_info["png"] if rgb_info else None,
                "gt2": gt2, "outside_analytic_box_px": outside_box,
            }

        negative = None
        if kept_idx % NEGATIVE_EVERY == 0:
            ch = negative_choice(enc_id, k, plan["ring_free"], ring.width, ring.height)
            if ch is None:
                summ["negatives_none_free"] += 1
            else:
                cid, u0, v0 = ch
                _img, n_info = do("rgb", cid, (u0, v0), f"negatives/{cid}/{k:06d}_{enc_id}.png", hide=True)
                negative = {"cam": cid, "roi": [u0, v0], "model_hidden": True, **n_info}
                horizon[cid] = compute_horizon_gt(cams[cid]["p"], cams[cid]["R"], cams[cid]["model"])
                summ["negatives"] += 1
                summ["images"]["negative"] += 1

        cams_out = {}
        for cid in RIG_CAMS + RING_CAMS:
            g = gt.cams[cid]
            d = {"kind": cams[cid]["kind"], "u": g.u, "v": g.v, "z_m": g.z_m, "in_image": g.in_image,
                 "box_analytic": g.box_analytic, "limb_dv_px": g.limb_dv_px, "background": g.background}
            if cid in cam_rec:
                d.update(cam_rec[cid])
            cams_out[cid] = d
        frame_bytes = sum(r2["bytes"] for rr in renders.values() for r2 in rr.values() if r2) + (negative["bytes"] if negative else 0)
        frame = {
            "schema": FRAME_SCHEMA, "encounter_id": enc_id, "k": k, "t_s": t_s, "utc": utc, "kept_index": kept_idx,
            "ownship": {"lat_deg": st.lat_deg, "lon_deg": st.lon_deg, "h_m": st.h_m, "p_ecef_m": st.p_ecef_m.tolist(),
                        "v_ned_mps": st.v_ned_mps.tolist(), "ypr_deg": list(st.ypr_deg), "R_ecef_body": st.r_ecef_body.tolist()},
            "cameras": {cid: {"p_ecef_m": c["p"].tolist(), "R_ecef_cam": c["R"].tolist()} for cid, c in cams.items()},
            "traffic": [{
                "id": enc_id, "model_id": rec["model_id"], "class": rec["class_id"], "stratum": rec.get("stratum"),
                "weight": rec.get("weight", 1.0),  # default-ok: optional fallback default
                "p_ecef_m": gt.p_ecef_m, "v_ecef_mps": gt.v_ecef_mps, "ypr_deg": gt.ypr_deg,
                "model_matrix_colmajor": model_m,
                "range_m": gt.range_m, "range_rate_mps": gt.range_rate_mps, "closing_speed_mps": gt.closing_speed_mps,
                "los_ned": gt.los_ned, "az_el_ned_deg": gt.az_el_ned_deg, "az_el_body_deg": gt.az_el_body_deg,
                "los_rate_ned_radps": gt.los_rate_ned_radps, "az_el_rates_ned_degps": gt.az_el_rates_ned_degps,
                "cpa": {"t_s": cpa["t_cpa_s"], "hmd_m": cpa["hmd_m"], "vmd_m": cpa["vmd_m"]},
                "cpa_extrapolated": gt.cpa_extrapolated, "events": {"nmac_t": events["nmac_t"], "lowc_t": events["lowc_t"]},
                "angular_span_rad": gt.angular_span_rad, "cams": cams_out,
            }],
            "renders": renders, "negative": negative, "rig_rgb_skipped_small": skipped_rig_rgb, "horizon": horizon,
            "files": files, "bytes": frame_bytes,
            "ms": {"frame": (time.perf_counter() - t_frame) * 1000.0, "gt_raster_horizon": ms_gt},
        }
        frames_file.write(json.dumps(clean(frame)) + "\n")
        frames_file.flush()
        summ["rendered"] += 1
        summ["bytes"] += frame_bytes
        acc["frame_ms"].append(frame["ms"]["frame"])
        acc["frame_bytes"].append(frame_bytes)
        acc["gt_ms"].append(ms_gt)
        if acc["deadline"] is not None and time.time() > acc["deadline"]:
            summ["interrupted"] = True
            return summ
    summ["cpa"] = clean(cpa)
    summ["events"] = clean(events)
    return summ


# ---------------------------------------------------------------------------------------------
# Plan mode (analytic counts, no browser)
# ---------------------------------------------------------------------------------------------
def run_plan(args: argparse.Namespace) -> int:
    scenario = json.loads(Path(args.ownship).read_text(encoding="utf-8"))
    own_fn = ownship_from_scenario(scenario)
    duration = float(scenario.get("duration_s", 60.0))  # default-ok: optional fallback default
    rig = Rig.load_nominal().make_true(20260922)
    ring = TrafficRing()
    recs = load_records(Path(args.encounters))
    if args.encounter_ids:
        want = set(args.encounter_ids.split(","))
        recs = [r for r in recs if r["id"] in want]
    rates = sorted(float(x) for x in args.plan_rates.split(","))
    base_rate = max(rates)
    for r in rates:
        if abs(base_rate / r - round(base_rate / r)) > 1e-9:
            raise ValueError("--plan-rates must divide the largest rate")
    meshes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows = []
    t0 = time.perf_counter()
    for n, rec in enumerate(recs):
        traj = make_trajectory(rec, own_fn)
        if rec["model_id"] not in meshes:
            meshes[rec["model_id"]] = raster.glb_mesh(REPO_ROOT / "assets" / "traffic" / rec["class_id"] / f"{rec['model_id']}.glb")
        V, _F = meshes[rec["model_id"]]
        max_dim = float(np.max(np.ptp(V, axis=0)))
        n_base = int(round(duration * base_rate))
        per_k = []
        for k in range(n_base):
            t_s = k / base_rate
            if isinstance(traj, TrackTrajectory) and not traj.covers(t_s):
                per_k.append(("outside_track", 0, 0, 0))
                continue
            st = own_fn(t_s)
            cams = camera_set(rig, ring, st)
            v_o = wgs84.r_ecef_ned(st.lat_deg, st.lon_deg) @ st.v_ned_mps
            gt = compute_intruder_analytic_gt(t_s, traj, st.p_ecef_m, v_o, st.r_ecef_body,
                                              {cid: (c["p"], c["R"], c["model"]) for cid, c in cams.items()},
                                              model_vertices=V, max_body_dim_m=max_dim)
            reason = select_frame(gt.range_m, {cid: gt.cams[cid].in_image for cid in RING_CAMS})
            if reason is not None:
                per_k.append((reason, 0, 0, 0))
                continue
            p_i = traj.position_at(t_s)
            r_i, _ = traj.attitude_at(t_s)
            pw = p_i[None, :] + V @ r_i.T
            plan = frame_plan(gt, cams, {cid: ((pw - c["p"][None, :]) @ c["R"])[:, 2] for cid, c in cams.items()})
            per_k.append(("kept", len(plan["ring"]), len(plan["rig"]), len(plan["ring_free"])))
        row = {"encounter_id": rec["id"], "class_id": rec["class_id"], "stratum": rec.get("stratum"), "rates": {}}
        for r in rates:
            step = int(round(base_rate / r))
            sub = per_k[::step]
            kept = [x for x in sub if x[0] == "kept"]
            skipped: dict[str, int] = {}
            for x in sub:
                if x[0] != "kept":
                    skipped[x[0]] = skipped.get(x[0], 0) + 1  # default-ok: frequency counter accumulator
            row["rates"][str(r)] = {"frames": len(sub), "kept": len(kept), "skipped": skipped,
                                    "ring_rois": sum(x[1] for x in kept), "rig_cams": sum(x[2] for x in kept),
                                    "negatives": len(kept) // NEGATIVE_EVERY}
        rows.append(row)
        if (n + 1) % 50 == 0:
            print(f"  plan: {n + 1}/{len(recs)} encounters ({time.perf_counter() - t0:.0f} s)", flush=True)
    totals = {}
    for r in rates:
        key = str(r)
        totals[key] = {f: sum(row["rates"][key][f] for row in rows) for f in ("frames", "kept", "ring_rois", "rig_cams", "negatives")}
        sk: dict[str, int] = {}
        for row in rows:
            for kk, vv in row["rates"][key]["skipped"].items():
                sk[kk] = sk.get(kk, 0) + vv  # default-ok: optional fallback default
        totals[key]["skipped"] = sk
    out = {"schema": "xvs.traffic_plan/1", "encounters": {"path": args.encounters, "sha256": sha256_file(Path(args.encounters)), "n": len(recs)},
           "ownship": args.ownship, "rates_hz": rates, "selection": {"range_max_m": RANGE_MAX_M},
           "totals": totals, "per_encounter": rows, "wall_s": time.perf_counter() - t0, "script_sha256": sha256_file(Path(__file__))}
    Path(args.plan).parent.mkdir(parents=True, exist_ok=True)
    Path(args.plan).write_text(json.dumps(clean(out), indent=1), encoding="utf-8")
    print(json.dumps(clean(totals), indent=1))
    return 0


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    t_start = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scenario = json.loads(Path(args.ownship).read_text(encoding="utf-8"))
    own_fn = ownship_from_scenario(scenario)
    duration = float(scenario.get("duration_s", 60.0))  # default-ok: optional fallback default
    n_frames = int(round(duration * args.rate_hz))
    if args.max_frames is not None:
        n_frames = min(n_frames, args.max_frames)
    start_dt = datetime.strptime(scenario["start_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    rig = Rig.load_nominal().make_true(20260922)
    ring = TrafficRing()
    all_recs = load_records(Path(args.encounters))
    only = None
    if args.frames:
        only = {(d["encounter_id"], int(d["k"])) for d in json.loads(Path(args.frames).read_text(encoding="utf-8"))["frames"]}
    if args.encounter_ids:
        want = args.encounter_ids.split(",")
        missing = [w for w in want if w not in {r["id"] for r in all_recs}]
        if missing:
            raise ValueError(f"unknown encounter ids {missing}")
        all_recs = [r for r in all_recs if r["id"] in set(want)]
    bi, bn = (int(x) for x in args.block.split("/"))
    recs = block_records(all_recs, bi, bn)
    if only is not None:
        recs = [r for r in recs if any(e == r["id"] for e, _ in only)]
    write_sequence_json(out, args, all_recs, ring, rig, scenario, n_frames)

    bdir = out / "blocks" / f"b{bi:02d}of{bn:02d}"
    bdir.mkdir(parents=True, exist_ok=True)
    frames_p, done_p = bdir / "frames.jsonl", bdir / "encounters_done.jsonl"
    existing = {(f["encounter_id"], f["k"]): f for f in read_jsonl(frames_p)}
    done = {d["encounter_id"]: d for d in read_jsonl(done_p)}
    todo = [r for r in recs if r["id"] not in done]
    print(f"G3a block {bi}/{bn}: {len(recs)} encounters ({len(done)} done, {len(todo)} to do), {n_frames} frames each "
          f"@ {args.rate_hz} Hz, backend {args.backend} -> {out}", flush=True)

    meshes: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def mesh(cls: str, mid: str) -> tuple[np.ndarray, np.ndarray]:
        if mid not in meshes:
            meshes[mid] = raster.glb_mesh(REPO_ROOT / "assets" / "traffic" / cls / f"{mid}.glb")
        return meshes[mid]

    acc: dict[str, Any] = {"renders": [], "frame_ms": [], "frame_bytes": [], "gt_ms": [], "incomplete": 0,
                           "deadline": (t_start + args.max_wall_h * 3600.0) if args.max_wall_h else None}
    pages = PageSet(args.backend, bdir, offline=args.offline, fresh_rgb=True) if todo else None
    run_rec: dict[str, Any] = {"start_utc": datetime.now(timezone.utc).isoformat(), "backend": args.backend,
                               "argv": sys.argv[1:], "git": git_state(), "complete": False, "interrupted": False,
                               "fresh_rgb_pages": True}
    new_done = 0
    loadlog = LoadLogger(bdir / "load_log.jsonl").start() if pages is not None else None  # D26
    try:
        if pages is not None:
            run_rec["webgl"] = pages.r.get_webgl_diagnostics(page=pages.pages[("rgb", "ring")])
            run_rec["browser_args"] = pages.r.get_browser_args()
            run_rec["resolved_settings"] = pages.resolved
        ctx = {"own_fn": own_fn, "rig": rig, "ring": ring, "pages": pages, "out": out, "rate": args.rate_hz,
               "n_frames": n_frames, "start_dt": start_dt, "duration": duration, "mesh": mesh}
        ensure_trailing_newline(frames_p)
        with open(frames_p, "a", encoding="utf-8") as ff:
            for n, rec in enumerate(todo):
                te = time.perf_counter()
                summ = render_encounter(ctx, rec, existing, ff, only, acc)
                summ["wall_s"] = time.perf_counter() - te
                if summ.get("interrupted"):
                    run_rec["interrupted"] = True
                    print(f"  wall-time limit reached in {rec['id']}; stopping (resume with the same command)", flush=True)
                    break
                if only is None:
                    with open(done_p, "a", encoding="utf-8") as fd:
                        fd.write(json.dumps(clean(summ)) + "\n")
                new_done += 1
                print(f"  [{n + 1:4d}/{len(todo)}] {rec['id']} {rec['class_id']:<13s} kept {summ['kept']:4d} rendered {summ['rendered']:4d} "
                      f"skip {summ['skipped']}  {summ['wall_s']:7.1f}s  (job {time.time() - t_start:8.0f}s)", flush=True)
                if acc["deadline"] is not None and time.time() > acc["deadline"]:
                    run_rec["interrupted"] = True
                    print("  wall-time limit reached; stopping (resume with the same command)", flush=True)
                    break
        run_rec["complete"] = (only is None and not run_rec["interrupted"] and len(read_jsonl(done_p)) == len(recs)) or \
                              (only is not None and not run_rec["interrupted"])
    finally:
        usgs = pages.usgs_stats() if pages is not None else None
        if pages is not None:
            pages.close()
        run_rec["end_utc"] = datetime.now(timezone.utc).isoformat()
        run_rec["wall_s"] = time.time() - t_start
        run_rec["encounters_finished"] = new_done
        run_rec["usgs"] = usgs
        run_rec["model_load_ms"] = stats(pages.model_load_ms) if pages is not None else None
        run_rec["rgb_pages_opened"] = pages.rgb_pages_opened if pages is not None else None
        run_rec["d26a"] = dict(pages.d26a) if pages is not None else None
        if loadlog is not None:
            run_rec["load_log"] = {"path": (bdir / "load_log.jsonl").as_posix(), "readings_this_run": loadlog.stop(),
                                   "summary_all_runs": summarise_load(bdir / "load_log.jsonl")}
        finish_block(bdir, out, run_rec, acc)
    return 0


def finish_block(bdir: Path, out: Path, run_rec: dict[str, Any], acc: dict[str, Any]) -> None:
    """perf.json (runs appended), data_sources.json (event-based, summed over runs), run_manifest.json."""
    groups: dict[str, dict[str, list[float]]] = {}
    for r in acc["renders"]:
        g = groups.setdefault(f"{r['pass']}_{r['kind']}", {})
        for k in ("n_renders", "ms_set", "ms_render", "ms_readback", "ms_encode", "bytes", "tiles_requested", "tiles_cache_miss", "bytes_upstream"):
            g.setdefault(k, []).append(r[k])
    run_perf = {**run_rec, "frames_rendered": len(acc["frame_ms"]), "incomplete_renders": acc["incomplete"],
                "frame_ms": stats(acc["frame_ms"]), "frame_bytes": stats(acc["frame_bytes"]), "gt_raster_horizon_ms": stats(acc["gt_ms"]),
                "per_pass": {k: {m: stats(v) for m, v in g.items()} for k, g in sorted(groups.items())}}
    pp = bdir / "perf.json"
    perf = json.loads(pp.read_text(encoding="utf-8")) if pp.exists() else {"schema": "xvs.traffic_perf/1", "runs": []}
    perf["runs"].append(clean(run_perf))
    done = read_jsonl(bdir / "encounters_done.jsonl")
    vp = bdir / "data_violations.jsonl"
    vp.touch(exist_ok=True)
    n_viol = sum(1 for ln in vp.read_text(encoding="utf-8").splitlines() if ln.strip())
    perf["totals"] = {
        "encounters_done": len(done), "frames_kept": sum(d["kept"] for d in done), "frames_rendered": sum(d["rendered"] for d in done),
        "skipped": {k: sum(d["skipped"].get(k, 0) for d in done) for k in ("range_gt_10km", "centre_not_in_ring", "outside_track")},  # default-ok: optional fallback default
        "images": {k: sum(d["images"][k] for d in done) for k in ("rgb_ring", "rgb_rig", "A", "V", "negative")},
        "rig_rgb_skipped_small": sum(d["rig_rgb_skipped_small"] for d in done), "bytes": sum(d["bytes"] for d in done),
        "wall_s_runs": sum(r.get("wall_s", 0.0) for r in perf["runs"]),  # default-ok: timing benchmark fallback
        "incomplete_renders_runs": sum(r.get("incomplete_renders", 0) for r in perf["runs"]), "violations": n_viol,  # default-ok: optional fallback default
    }
    pp.write_text(json.dumps(clean(perf), indent=1), encoding="utf-8")
    # Data sources: event-based, summed over the runs of this block (each run's proxy counters start at 0).
    us = [r["usgs"] for r in perf["runs"] if r.get("usgs")]
    sources = []
    if us:
        tot = dict(us[-1])
        for k in ("tiles_served_total", "tiles_cache_hits", "tiles_upstream_fetches", "bytes_served_total", "bytes_upstream_fetched", "tile_errors"):
            tot[k] = sum(u.get(k, 0) for u in us)  # default-ok: optional fallback default
        firsts = [u["first_access_utc"] for u in us if u.get("first_access_utc")]
        lasts = [u["last_access_utc"] for u in us if u.get("last_access_utc")]
        tot["first_access_utc"] = min(firsts) if firsts else None
        tot["last_access_utc"] = max(lasts) if lasts else None
        if tot["tiles_served_total"] > 0 or tot["tile_errors"] > 0:
            sources.append(tot)
    (bdir / "data_sources.json").write_text(json.dumps({"schema": "xvs.data_sources/1", "run_id": f"{out.name}/{bdir.name}",
                                                        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                        "sources": sources}, indent=2), encoding="utf-8")
    manifest = {"schema": "xvs.run_manifest/1", "run_id": f"{out.name}/{bdir.name}", "script": "scripts/xvs/render_traffic.py",
                "script_sha256": sha256_file(Path(__file__)), "python": sys.version.split()[0], "platform": platform.platform(),
                "runs": [{k: r.get(k) for k in ("start_utc", "end_utc", "wall_s", "backend", "argv", "complete", "interrupted",
                                                "encounters_finished", "git")} for r in perf["runs"]],
                "totals": perf["totals"]}
    (bdir / "run_manifest.json").write_text(json.dumps(clean(manifest), indent=1), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="WP-02 G3a dataset v0 renderer (WP02 v1.3 §3 G3a)")
    ap.add_argument("--encounters", required=True, help="encounters JSONL (sampler v1 records or sampled tracks)")
    ap.add_argument("--ownship", required=True, help="ownship scenario JSON (t1_cruise or t2_ksfo28l)")
    ap.add_argument("--rate-hz", type=float, default=1.0)
    ap.add_argument("--out", default=str(REPO_ROOT / "output" / "xvs" / "wp02" / "g3a"))
    ap.add_argument("--block", default="0/1", help="i/n: round-robin block i (0-based) of n")
    ap.add_argument("--backend", choices=["hw", "ref"], default="hw")
    ap.add_argument("--sampler-config", default=None, help="sampler config whose encounters_sha256 must match --encounters")
    ap.add_argument("--encounter-ids", default=None, help="comma-separated subset (pilot)")
    ap.add_argument("--frames", default=None, help="JSON {'frames': [{'encounter_id', 'k'}, ...]}: render only these")
    ap.add_argument("--max-frames", type=int, default=None, help="debug only: frames per encounter")
    ap.add_argument("--max-wall-h", type=float, default=3.9, help="stop cleanly after this many hours (resume later)")
    ap.add_argument("--offline", action="store_true", help="tiles from the cache only (tests)")
    ap.add_argument("--fresh-rgb-pages", action="store_true",
                    help="no effect: fresh rgb pages per encounter are always used since D25 (kept so older commands still run)")
    ap.add_argument("--plan", default=None, help="analytic counts only: write this JSON and exit")
    ap.add_argument("--plan-rates", default="0.5,1,2")
    a = ap.parse_args()
    if a.plan:
        return run_plan(a)
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
