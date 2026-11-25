"""WP-02 G3a checks on the outputs of scripts/xvs/render_traffic.py (WP02 v1.2 §3 G3a). Reads outputs only.

Subcommands (every result is a JSON under the given --out; report tables come from report_tables.py):
  project     r_ds choice from the measured pilot and the analytic plan (limits: total <= 16 h, disk <= 80 GB,
              every job <= 4 h; the largest of {0.5, 1, 2} Hz that meets all three).
  acceptance  items 1-2: incomplete renders, violations counted from the real logs, GT-2 v1.1 on every A mask,
              GT-3(b), mask purity; plus completeness (every encounter done, every recorded file present).
  select      seeded choice of kept frames (determinism: 20; ref spot check: 10).
  compare     pixel comparison of the selected frames between two output roots (masks identical; rgb
              max_diff <= 1 and n_diff <= 1 %).
  datacard    statistics for the data card (item 5).
  isolate     where two or more renders of the same rgb images differ: whole image and outside a +-12 px
              neighbourhood of the intruder's projected centre (item 3 diagnosis).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
LIMITS = {"total_h": 16.0, "disk_gb": 80.0, "job_h": 4.0}
JOB_TARGET_H = 3.5  # block size target; each job also stops itself at --max-wall-h 3.9


def jsonl(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return out


def load_frames(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """All frame records of all blocks; for a (encounter, k) recorded twice (re-render after a resume), the last wins."""
    frames: dict[tuple[str, int], dict[str, Any]] = {}
    for b in sorted((root / "blocks").glob("b*of*")):
        for f in jsonl(b / "frames.jsonl"):
            frames[(f["encounter_id"], f["k"])] = f
    return frames


def st(v: list[float]) -> dict[str, float] | None:
    if not v:
        return None
    a = np.asarray(v, float)
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "min": float(a.min()), "max": float(a.max()), "sum": float(a.sum())}


# ---------------------------------------------------------------------------------------------
def cmd_project(a: argparse.Namespace) -> int:
    pilot = Path(a.pilot)
    frames = list(load_frames(pilot).values())
    done = [d for b in sorted((pilot / "blocks").glob("b*of*")) for d in jsonl(b / "encounters_done.jsonl")]
    perf_runs = [r for b in sorted((pilot / "blocks").glob("b*of*")) for r in json.loads((b / "perf.json").read_text(encoding="utf-8"))["runs"]]
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    ms = np.array([f["ms"]["frame"] for f in frames], float)
    by = np.array([f["bytes"] for f in frames], float)
    n_ring = np.array([sum(1 for c in f["renders"] if c.startswith("T")) for f in frames], float)
    n_rig = np.array([sum(1 for c in f["renders"] if not c.startswith("T")) for f in frames], float)
    n_neg = np.array([1.0 if f.get("negative") else 0.0 for f in frames])
    X = np.column_stack([np.ones_like(ms), n_ring, n_rig, n_neg])
    coef_ms, *_ = np.linalg.lstsq(X, ms, rcond=None)
    coef_by, *_ = np.linalg.lstsq(X, by, rcond=None)
    # per-encounter overhead: encounter wall time minus its frames (model loads, CPA/events, analytic GT of skipped frames)
    fr_ms_by_enc: dict[str, float] = {}
    for f in frames:
        fr_ms_by_enc[f["encounter_id"]] = fr_ms_by_enc.get(f["encounter_id"], 0.0) + f["ms"]["frame"]  # default-ok: timing benchmark fallback
    overhead_s = [d["wall_s"] - fr_ms_by_enc.get(d["encounter_id"], 0.0) / 1000.0 for d in done]  # default-ok: timing benchmark fallback
    per_frame_overhead_s = float(np.sum(overhead_s) / max(1, sum(d["frames_total"] for d in done)))
    enc_overhead_fixed_s = float(np.median(overhead_s)) if overhead_s else 0.0
    out: dict[str, Any] = {
        "pilot": {"root": pilot.as_posix(), "encounters": [d["encounter_id"] for d in done], "frames": len(frames),
                  "frame_ms": st(ms.tolist()), "frame_bytes": st(by.tolist()),
                  "per_frame_counts": {"ring_rois": st(n_ring.tolist()), "rig_cams": st(n_rig.tolist()), "negatives": st(n_neg.tolist())},
                  "linear_model_ms": dict(zip(["const", "per_ring_roi", "per_rig_cam", "per_negative"], coef_ms.tolist())),
                  "linear_model_bytes": dict(zip(["const", "per_ring_roi", "per_rig_cam", "per_negative"], coef_by.tolist())),
                  "encounter_overhead_s": st(overhead_s), "overhead_s_per_scenario_frame": per_frame_overhead_s,
                  "tiles": {"served": sum((r.get("usgs") or {}).get("tiles_served_total", 0) for r in perf_runs),  # default-ok: optional fallback default
                            "upstream_fetches": sum((r.get("usgs") or {}).get("tiles_upstream_fetches", 0) for r in perf_runs),  # default-ok: optional fallback default
                            "upstream_bytes": sum((r.get("usgs") or {}).get("bytes_upstream_fetched", 0) for r in perf_runs),  # default-ok: optional fallback default
                            "errors": sum((r.get("usgs") or {}).get("tile_errors", 0) for r in perf_runs)},  # default-ok: optional fallback default
                  "n_renders": {k: v.get("n_renders") for r in perf_runs for k, v in r.get("per_pass", {}).items()},
                  "per_pass_ms_render": {k: v.get("ms_render") for r in perf_runs for k, v in r.get("per_pass", {}).items()},
                  "wall_s": sum(r.get("wall_s", 0.0) for r in perf_runs),  # default-ok: timing benchmark fallback
                  "incomplete_renders": sum(r.get("incomplete_renders", 0) for r in perf_runs)},  # default-ok: optional fallback default
        "plan": {"path": Path(a.plan).as_posix(), "encounters": plan["encounters"]["n"]},
        "limits": LIMITS, "job_target_h": JOB_TARGET_H, "rates": {},
    }
    mean_ms, mean_by = float(ms.mean()), float(by.mean())
    for r in ("0.5", "1.0", "2.0"):
        t = plan["totals"][r]
        lin_ms = coef_ms[0] * t["kept"] + coef_ms[1] * t["ring_rois"] + coef_ms[2] * t["rig_cams"] + coef_ms[3] * t["negatives"]
        lin_by = coef_by[0] * t["kept"] + coef_by[1] * t["ring_rois"] + coef_by[2] * t["rig_cams"] + coef_by[3] * t["negatives"]
        frames_ms = max(lin_ms, mean_ms * t["kept"])            # conservative: the larger of the two models
        disk = max(lin_by, mean_by * t["kept"])
        overhead_s = per_frame_overhead_s * t["frames"]
        total_h = (frames_ms / 1000.0 + overhead_s) / 3600.0
        n_jobs = max(1, math.ceil(total_h / JOB_TARGET_H))
        job_h = total_h / n_jobs
        ok = total_h <= LIMITS["total_h"] and disk / 1e9 <= LIMITS["disk_gb"] and job_h <= LIMITS["job_h"]
        out["rates"][r] = {"planned": t, "frames_ms_linear": lin_ms, "frames_ms_mean": mean_ms * t["kept"],
                           "overhead_s": overhead_s, "total_h": total_h, "disk_gb": disk / 1e9, "n_jobs": n_jobs,
                           "job_h": job_h, "meets_limits": bool(ok)}
    ok_rates = [float(r) for r, v in out["rates"].items() if v["meets_limits"]]
    out["r_ds_hz"] = max(ok_rates) if ok_rates else None
    out["hard_stop_total_over_24h"] = bool(out["r_ds_hz"] is None and min(v["total_h"] for v in out["rates"].values()) > 24.0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"r_ds_hz": out["r_ds_hz"], "rates": {r: {k: v[k] for k in ("total_h", "disk_gb", "n_jobs", "job_h", "meets_limits")}
                                                           for r, v in out["rates"].items()}}, indent=1))
    return 0


# ---------------------------------------------------------------------------------------------
def cmd_acceptance(a: argparse.Namespace) -> int:
    root = Path(a.root)
    frames = load_frames(root)
    seq = json.loads((root / "sequence.json").read_text(encoding="utf-8"))
    blocks = sorted((root / "blocks").glob("b*of*"))
    done = [d for b in blocks for d in jsonl(b / "encounters_done.jsonl")]
    viol = {b.name: sum(1 for ln in (b / "data_violations.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip())
            if (b / "data_violations.jsonl").exists() else None for b in blocks}
    incomplete, a_masks, not_comp, mismatch, ties, outside, gt3b, impure = [], 0, [], 0, 0, [], [], []
    missing_files, n_files = [], 0
    for key, f in frames.items():
        for cid, rr in f["renders"].items():
            for ps in ("A", "V", "rgb"):
                if rr.get(ps) and rr[ps]["incomplete"]:
                    incomplete.append([*key, cid, ps])
            for ps in ("A", "V"):
                if rr[ps].get("impure_px"):
                    impure.append([*key, cid, ps, rr[ps]["impure_px"]])
            c = f["traffic"][0]["cams"][cid]
            a_masks += 1
            g = c["gt2"]
            if not g.get("computable"):
                not_comp.append([*key, cid])
            else:
                mismatch += g["mismatch_px"]
                ties += g["tie_px"]
            if c["outside_analytic_box_px"]:
                outside.append([*key, cid, c["outside_analytic_box_px"]])
            if c["visible_outside_amodal_px"]:
                gt3b.append([*key, cid, c["visible_outside_amodal_px"]])
        if f.get("negative") and f["negative"]["incomplete"]:
            incomplete.append([*key, f["negative"]["cam"], "negative"])
        if a.check_files:
            for fl in f["files"]:
                n_files += 1
                p = root / fl["path"]
                if not p.exists() or (a.check_files == "sha" and hashlib.sha256(p.read_bytes()).hexdigest() != fl["sha256"]):
                    missing_files.append(fl["path"])
    enc_total = seq["encounters"]["n"]
    out = {
        "root": root.as_posix(), "encounters_total": enc_total, "encounters_done": len(done),
        "frames_kept_done": sum(d["kept"] for d in done), "frame_records": len(frames),
        "item1": {"incomplete_renders": len(incomplete), "incomplete_list": incomplete[:50], "violations_per_block": viol,
                  "violations_total": sum(v for v in viol.values() if v is not None),
                  "pass": not incomplete and all(v == 0 for v in viol.values())},
        "item2": {"a_masks": a_masks, "gt2_not_computable": len(not_comp), "gt2_non_tie_mismatch_px": mismatch, "gt2_tie_px": ties,
                  "outside_analytic_box": len(outside), "gt3b_violations": len(gt3b), "gt3b_list": gt3b[:50],
                  "impure_masks": len(impure), "impure_list": impure[:50],
                  "pass": not not_comp and mismatch == 0 and not gt3b and not impure and not outside},
        "completeness": {"all_encounters_done": len(done) == enc_total,
                         "frames_recorded_equal_kept": len(frames) == sum(d["kept"] for d in done),
                         "files_checked": n_files, "files_missing_or_changed": len(missing_files), "missing_list": missing_files[:50]},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if not kk.endswith("_list")}) for k, v in out.items()}, indent=1))
    return 0


# ---------------------------------------------------------------------------------------------
def job_groups(root: Path, head_tail: float | None, every: int | None) -> list[tuple[str, str, list[str]]]:
    """Encounter groups by position in each block job (order of encounters_done.jsonl):
    head_tail f -> ("head", first round(f n)) and ("tail", last round(f n)) per block;
    every K     -> consecutive groups of K encounters per block, labelled by their positions."""
    out = []
    for b in sorted((root / "blocks").glob("b*of*")):
        order = [d["encounter_id"] for d in jsonl(b / "encounters_done.jsonl")]
        if head_tail is not None:
            m = int(round(head_tail * len(order)))
            out += [(b.name, f"head 1-{m}", order[:m]), (b.name, f"tail {len(order) - m + 1}-{len(order)}", order[len(order) - m:])]
        else:
            out += [(b.name, f"{i + 1}-{min(i + every, len(order))}", order[i:i + every]) for i in range(0, len(order), every)]
    return out


def cmd_select(a: argparse.Namespace) -> int:
    keys = sorted(load_frames(Path(a.root)).keys())
    rng = np.random.default_rng(a.seed)
    if a.head_tail is None and a.every_encounters is None:
        idx = sorted(rng.choice(len(keys), size=min(a.n, len(keys)), replace=False).tolist())
        sel = [{"encounter_id": keys[i][0], "k": keys[i][1]} for i in idx]
    else:
        # n seeded kept frames per encounter group (B3: head/tail audit, cause experiment probes)
        sel = []
        for blk, grp, encs in job_groups(Path(a.root), a.head_tail, a.every_encounters):
            es = set(encs)
            cand = [kk for kk in keys if kk[0] in es]
            pick = sorted(rng.choice(len(cand), size=min(a.n, len(cand)), replace=False).tolist()) if cand else []
            sel += [{"encounter_id": cand[i][0], "k": cand[i][1], "block": blk, "group": grp} for i in pick]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"root": a.root, "seed": a.seed, "n_candidates": len(keys), "n_per_group": a.n,
                                       "head_tail": a.head_tail, "every_encounters": a.every_encounters, "frames": sel}, indent=1), encoding="utf-8")
    print(json.dumps(sel))
    return 0


_MESH: dict[str, Any] = {}


def mask_diff_ties(frame: dict[str, Any], path: str, diff: np.ndarray) -> int:
    """Number of differing mask pixels whose centre has the union boundary of the projected GLB mesh within 1/16 px
    (spec v1.3 D-B2a, the GT-2 D13 tie rule), from the pose, intrinsics and ROI recorded in the frame."""
    from xvs.rig.rig import Rig
    from xvs.traffic import raster
    from xvs.traffic.ring import TrafficRing
    cid = path.split("/")[1]
    t = frame["traffic"][0]
    key = t["model_id"]
    if key not in _MESH:
        _MESH[key] = raster.glb_mesh(REPO_ROOT / "assets" / "traffic" / t["class"] / f"{key}.glb")
    if "rig" not in _MESH:
        _MESH["rig"] = Rig.load_nominal().camera_model
        _MESH["ring"] = TrafficRing().camera_model
    V, F = _MESH[key]
    m4 = np.array(t["model_matrix_colmajor"], float).reshape(4, 4).T
    cam = frame["cameras"][cid]
    pc = (m4[:3, 3][None, :] + V @ m4[:3, :3].T - np.array(cam["p_ecef_m"])[None, :]) @ np.array(cam["R_ecef_cam"])
    m = _MESH["ring" if cid.startswith("T") else "rig"]
    u0, v0 = t["cams"][cid]["roi"] or (0, 0)
    z = pc[:, 2]
    if np.any(z < raster.Z_NEAR_M):
        pc, F = raster.clip_near(pc, F)
    u = m.fx * pc[:, 0] / pc[:, 2] + m.cx - u0
    v = m.fy * pc[:, 1] / pc[:, 2] + m.cy - v0
    ys, xs = np.nonzero(diff)
    return int(np.count_nonzero(raster.near_union_boundary(xs.astype(float), ys.astype(float), u, v, F)))


def cmd_compare(a: argparse.Namespace) -> int:
    ra, rb = Path(a.a), Path(a.b)
    fa, fb = load_frames(ra), load_frames(rb)
    sel = [(d["encounter_id"], d["k"]) for d in json.loads(Path(a.frames).read_text(encoding="utf-8"))["frames"]]
    rows = []
    gt_rows = []
    for key in sel:
        if key not in fa or key not in fb:
            rows.append({"frame": list(key), "present_a": key in fa, "present_b": key in fb, "pass": False})
            continue
        # GT of the frame: the traffic record (analytic fields of all 14 cameras, mask-derived fields, GT-2 results),
        # the horizon GT and the negative's camera / ROI must be bit-identical (render timings are excluded)
        neg = lambda f: None if not f.get("negative") else {k: f["negative"][k] for k in ("cam", "roi")}
        same = {part: json.dumps(get(fa[key]), sort_keys=True) == json.dumps(get(fb[key]), sort_keys=True)
                for part, get in (("traffic", lambda f: f["traffic"]), ("horizon", lambda f: f["horizon"]), ("negative", neg))}
        gt_rows.append({"frame": list(key), **same, "identical": all(same.values())})
        pa = {fl["path"] for fl in fa[key]["files"]}
        pb = {fl["path"] for fl in fb[key]["files"]}
        paths = sorted(pa | pb)
        for p in paths:
            is_mask = p.startswith("masks/")
            if a.masks_only and not is_mask:
                continue
            row = {"frame": list(key), "path": p, "kind": "mask" if is_mask else "rgb", "in_a": p in pa, "in_b": p in pb}
            if p in pa and p in pb:
                ia = np.array(Image.open(ra / p), dtype=np.int16)
                ib = np.array(Image.open(rb / p), dtype=np.int16)
                d = np.abs(ia - ib)
                n_diff = int(np.count_nonzero(d.reshape(d.shape[0], d.shape[1], -1).max(axis=2)))
                row.update({"max_diff": int(d.max()), "n_diff_px": n_diff, "n_px": int(d.shape[0] * d.shape[1])})
                if is_mask and n_diff and a.mask_ties_allowed:
                    row["n_tie_px"] = mask_diff_ties(fa[key], p, d.reshape(d.shape[0], d.shape[1], -1).max(axis=2) > 0)
                    row["pass"] = row["n_tie_px"] == n_diff
                else:
                    row["pass"] = (row["max_diff"] == 0) if is_mask else (row["max_diff"] <= 1 and n_diff <= 0.01 * row["n_px"])
            else:
                row["pass"] = False
            rows.append(row)
    masks = [r for r in rows if r.get("kind") == "mask"]
    rgbs = [r for r in rows if r.get("kind") == "rgb"]
    out = {"a": ra.as_posix(), "b": rb.as_posix(), "frames": len(sel), "masks_only": bool(a.masks_only),
           "masks": {"n": len(masks), "identical": sum(1 for r in masks if r.get("max_diff") == 0),
                     "differing_px": sum(r.get("n_diff_px", 0) for r in masks), "tie_px": sum(r.get("n_tie_px", 0) for r in masks),  # default-ok: geometry dimension fallback
                     "ties_allowed": bool(a.mask_ties_allowed), "pass": all(r["pass"] for r in masks) and bool(masks)},
           "rgb": {"n": len(rgbs), "pass_n": sum(1 for r in rgbs if r["pass"]), "worst_max_diff": max((r.get("max_diff", 999) for r in rgbs), default=None),  # default-ok: optional fallback default
                   "worst_n_diff_frac": max((r["n_diff_px"] / r["n_px"] for r in rgbs if "n_px" in r), default=None),
                   "pass": all(r["pass"] for r in rgbs)},
           "gt": {"frames": len(gt_rows), "identical": sum(1 for g in gt_rows if g["identical"]), "rows": gt_rows},
           "rows": rows}
    out["pass"] = out["masks"]["pass"] and (a.masks_only or out["rgb"]["pass"]) and all(r.get("pass", False) for r in rows)
    out["masks_and_gt_bit_exact"] = (out["masks"]["identical"] == out["masks"]["n"] and out["gt"]["identical"] == out["gt"]["frames"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: (v if k != "gt" else {kk: vv for kk, vv in v.items() if kk != "rows"}) for k, v in out.items() if k != "rows"}, indent=1))
    return 0


# ---------------------------------------------------------------------------------------------
def cmd_datacard(a: argparse.Namespace) -> int:
    root = Path(a.root)
    frames = load_frames(root)
    blocks = sorted((root / "blocks").glob("b*of*"))
    done = [d for b in blocks for d in jsonl(b / "encounters_done.jsonl")]
    perfs = [json.loads((b / "perf.json").read_text(encoding="utf-8")) for b in blocks]
    seq = json.loads((root / "sequence.json").read_text(encoding="utf-8"))
    per_class: dict[str, dict[str, int]] = {}
    per_stratum: dict[str, dict[str, int]] = {}
    for d in done:
        s = d.get("stratum") or {}
        for tab, key in ((per_class, d["class_id"]), (per_stratum, f"{s.get('geometry')}/{s.get('vertical')}")):
            e = tab.setdefault(key, {"encounters": 0, "encounters_with_frames": 0, "frames_kept": 0, "skipped_range": 0, "skipped_not_in_ring": 0})
            e["encounters"] += 1
            e["encounters_with_frames"] += int(d["kept"] > 0)
            e["frames_kept"] += d["kept"]
            e["skipped_range"] += d["skipped"]["range_gt_10km"]
            e["skipped_not_in_ring"] += d["skipped"]["centre_not_in_ring"]
    ranges, box_ring, box_rig, vis_ring, vis_rig = [], [], [], [], []
    bg = {"ring": {"sky": 0, "earth": 0}, "rig": {"sky": 0, "earth": 0}}
    images = {"rgb_ring": 0, "rgb_rig": 0, "A": 0, "V": 0, "negative": 0}
    # special cases the data card must explain (D16 review R-G3a-4, -10, -11)
    special = {"box_side_gt_1e4_px": {"ring": 0, "rig": 0}, "box_side_gt_1e4_px_near_clipped": {"ring": 0, "rig": 0},
               "renders_near_clipped": 0, "nonempty_A": 0, "nonempty_A_box_undefined": 0, "V_equals_A": 0,
               "frames_range_lt_20m": 0, "frames_range_lt_50m": 0}
    nbytes = 0
    for f in frames.values():
        t = f["traffic"][0]
        ranges.append(t["range_m"])
        special["frames_range_lt_20m"] += int(t["range_m"] < 20.0)
        special["frames_range_lt_50m"] += int(t["range_m"] < 50.0)
        for cid, rr in f["renders"].items():
            c = t["cams"][cid]
            kind = "ring" if cid.startswith("T") else "rig"
            b = c["box_analytic"]
            near = bool((c.get("gt2") or {}).get("near_clipped"))
            special["renders_near_clipped"] += int(near)
            if b is not None:
                side = max(b[2] - b[0], b[3] - b[1])
                (box_ring if kind == "ring" else box_rig).append(side)
                if side > 1e4:
                    special["box_side_gt_1e4_px"][kind] += 1
                    special["box_side_gt_1e4_px_near_clipped"][kind] += int(near)
            if c["amodal_px"]:
                special["nonempty_A"] += 1
                special["nonempty_A_box_undefined"] += int(b is None)
            special["V_equals_A"] += int(c["visible_px"] == c["amodal_px"] and c["box_px"] == c["box_amodal_px"])
            if c["visibility"] is not None:
                (vis_ring if kind == "ring" else vis_rig).append(c["visibility"])
            bg[kind][c["background"]] += 1
            images["A"] += 1
            images["V"] += 1
            if rr.get("rgb"):
                images["rgb_ring" if kind == "ring" else "rgb_rig"] += 1
        if f.get("negative"):
            images["negative"] += 1
        nbytes += f["bytes"]

    def hist(v: list[float], edges: list[float]) -> dict[str, int]:
        h, _ = np.histogram(np.asarray(v, float), bins=edges)
        if int(h.sum()) != len(v):  # np.histogram drops values outside the edges (R-G3a-4)
            raise ValueError(f"histogram edges {edges[0]:g}..{edges[-1]:g} miss {len(v) - int(h.sum())} of {len(v)} values")
        return {f"{edges[i]:g}-{edges[i + 1]:g}": int(h[i]) for i in range(len(h))}
    wall = sum(r.get("wall_s", 0.0) for p in perfs for r in p["runs"])  # default-ok: timing benchmark fallback
    frame_ms = [f["ms"]["frame"] for f in frames.values()]
    ds = [json.loads((b / "data_sources.json").read_text(encoding="utf-8")) for b in blocks if (b / "data_sources.json").exists()]
    usgs = {"tiles_served_total": 0, "tiles_cache_hits": 0, "tiles_upstream_fetches": 0, "bytes_upstream_fetched": 0, "tile_errors": 0}
    attribution, licence = None, None
    for d in ds:
        for s in d["sources"]:
            if s.get("id") == "usgs_imagery_only":
                for k in usgs:
                    usgs[k] += s.get(k, 0)  # default-ok: optional fallback default
                attribution, licence = s.get("attribution"), s.get("license")
    out = {
        "root": root.as_posix(), "encounters": {"total": seq["encounters"]["n"], "done": len(done),
                                                "with_kept_frames": sum(1 for d in done if d["kept"] > 0)},
        "rate_hz": seq["rate_hz"], "frames": {"kept": len(frames), "scenario_frames": sum(d["frames_total"] for d in done),
                                              "skipped_range_gt_10km": sum(d["skipped"]["range_gt_10km"] for d in done),
                                              "skipped_centre_not_in_ring": sum(d["skipped"]["centre_not_in_ring"] for d in done)},
        "per_class": per_class, "per_stratum": per_stratum,
        "range_m": {"stats": st(ranges), "hist": hist(ranges, [0, 500, 1000, 2000, 3000, 5000, 7500, 10000])},
        "box_px_ring": {"stats": st(box_ring), "hist": hist(box_ring, [0, 2, 4, 8, 16, 32, 64, 128, 256, 1e4, np.inf])},
        "box_px_rig": {"stats": st(box_rig), "hist": hist(box_rig, [0, 2, 4, 8, 16, 32, 64, 128, 256, 1e4, np.inf])},
        "special": special,
        "visibility_ring": {"stats": st(vis_ring), "hist": hist(vis_ring, [0, 0.25, 0.5, 0.75, 0.999999, 1.000001])},
        "visibility_rig": {"stats": st(vis_rig), "hist": hist(vis_rig, [0, 0.25, 0.5, 0.75, 0.999999, 1.000001])},
        "background": bg, "images": images, "bytes": nbytes,
        "rig_rgb_skipped_small": sum(d["rig_rgb_skipped_small"] for d in done),
        "compute": {"wall_s_all_runs": wall, "runs": sum(len(p["runs"]) for p in perfs), "frame_ms": st(frame_ms)},
        "sources": {"usgs": usgs, "usgs_attribution": attribution, "usgs_license": licence,
                    "models": "own procedural models (assets/traffic, generated by scripts/xvs/make_traffic_models.py)",
                    "encounters": seq["encounters"], "sampler": (seq.get("sampler") or {}).get("path")},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("encounters", "frames", "images", "bytes")}, indent=1))
    return 0


def cmd_isolate(a: argparse.Namespace) -> int:
    roots = dict(x.split("=", 1) for x in a.roots)
    names = list(roots)
    frames = load_frames(Path(a.gt_root))
    rows = []
    for path in a.paths:
        cam = path.split("/")[1]
        k_str, enc = Path(path).stem.split("_", 1)
        f = frames[(enc, int(k_str))]
        c = f["traffic"][0]["cams"][cam]
        u0, v0 = c.get("roi") or (0, 0)
        u, v = int(round(c["u"] - u0)), int(round(c["v"] - v0))
        imgs = {n: np.array(Image.open(Path(r) / path)).astype(np.int16) for n, r in roots.items()}
        keep = np.ones(imgs[names[0]].shape[:2], bool)
        keep[max(0, v - a.halo):v + a.halo + 1, max(0, u - a.halo):u + a.halo + 1] = False
        row = {"path": path, "intruder_px_in_window": [u, v], "pairs": {}}
        for i, n1 in enumerate(names):
            for n2 in names[i + 1:]:
                d = np.abs(imgs[n1] - imgs[n2]).max(axis=2)
                row["pairs"][f"{n1}|{n2}"] = {"max": int(d.max()), "px_gt1": int((d > 1).sum()),
                                             "max_outside_intruder": int((d * keep).max()), "px_gt1_outside_intruder": int(((d * keep) > 1).sum())}
        rows.append(row)
    out = {"roots": roots, "gt_root": a.gt_root, "halo_px": a.halo, "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(rows, indent=1)[:4000])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="WP-02 G3a checks")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("project"); p.add_argument("--pilot", required=True); p.add_argument("--plan", required=True); p.add_argument("--out", required=True)
    p = sp.add_parser("acceptance"); p.add_argument("--root", required=True); p.add_argument("--out", required=True)
    p.add_argument("--check-files", choices=["exists", "sha"], default="exists")
    p = sp.add_parser("select"); p.add_argument("--root", required=True); p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, required=True); p.add_argument("--out", required=True)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--head-tail", type=float, default=None, help="n frames from the first and n from the last FRAC of each block job")
    g.add_argument("--every-encounters", type=int, default=None, help="n frames from every K consecutive encounters of each block job")
    p = sp.add_parser("compare"); p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p.add_argument("--frames", required=True); p.add_argument("--out", required=True); p.add_argument("--masks-only", action="store_true")
    p.add_argument("--mask-ties-allowed", action="store_true", help="spec v1.3 D-B2a: differing mask pixels must all be union-boundary ties")
    p = sp.add_parser("datacard"); p.add_argument("--root", required=True); p.add_argument("--out", required=True)
    p = sp.add_parser("isolate"); p.add_argument("--roots", nargs="+", required=True, help="name=dir ...")
    p.add_argument("--gt-root", required=True); p.add_argument("--paths", nargs="+", required=True)
    p.add_argument("--halo", type=int, default=12); p.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"project": cmd_project, "acceptance": cmd_acceptance, "select": cmd_select, "compare": cmd_compare,
            "datacard": cmd_datacard, "isolate": cmd_isolate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
