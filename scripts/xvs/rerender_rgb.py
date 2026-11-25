"""WP-02 G3a dataset v0.1 (Batch B3 rule, D25): re-render the rgb images of an existing G3a dataset with fresh rgb pages.

Only the rgb images are rendered again: the ring ROI rgb, the rig rgb and the negatives of every kept frame, with the
camera poses recomputed exactly as in render_traffic.py, the ROI, model matrix and UTC recorded in the source frame
record, and a fresh pair of rgb pages per encounter (D25). Masks and GT are not re-rendered: the mask PNGs are
hard-linked (copied if linking fails) from the source, and each output frame record is the source record with its rgb
and negative entries (render info, SHA-256, bytes) replaced. The source frame time is kept under "ms_v0".

usage: rerender_rgb.py --src output/xvs/wp02/g3a --out output/xvs/wp02/g3a_v01 --block i/n [--backend hw]
                       [--max-wall-h 3.9] [--frames <select JSON>] [--redo-encounters E1,E2,... --redo-tag b4]
--redo-encounters (Batch B4): the listed encounters of this block are rendered again in place. Before that, their frame
records and encounter summaries are moved to blocks/<b>/replaced_<tag>.jsonl and replaced_<tag>_done.jsonl, and their
rgb and negative PNGs to <out>/replaced_<tag>/, so nothing is lost; masks are hard links and stay.
Blocks are the source's block folders (b<i>of<n>); encounters are processed in the source job order. A job stops cleanly
after --max-wall-h; re-running the same command resumes (frames already written with matching SHA-256 are kept).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "xvs"))
import render_traffic as RT  # noqa: E402
from load_logger import LoadLogger, summarise as summarise_load  # noqa: E402  (D26)

SCHEMA = "xvs.traffic_rgb_rerender/1"


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_sequence(src: Path, out: Path, args: argparse.Namespace) -> dict[str, Any]:
    seq = json.loads((src / "sequence.json").read_text(encoding="utf-8"))
    if getattr(args, "redo_encounters", None):
        # an in-place redo (Batch B4) appends its own provenance instead of requiring the original one
        p = out / "sequence.json"
        cur = json.loads(p.read_text(encoding="utf-8"))
        entry = {"tag": args.redo_tag, "block": args.block, "encounters": sorted(args.redo_encounters.split(",")),
                 "utc": datetime.now(timezone.utc).isoformat(), "script_sha256": RT.sha256_file(Path(__file__)),
                 "renderer_sha256": RT.sha256_file(REPO_ROOT / "scripts" / "xvs" / "render_traffic.py"),
                 "page_sha256": RT.sha256_file(REPO_ROOT / "sim" / "cesium" / "page" / "xvs_page.js"),
                 "renderer_module_sha256": RT.sha256_file(REPO_ROOT / "src" / "xvs" / "sim" / "renderer.py"),
                 "rules": "D25 fresh rgb pages, D26 quiet machine and load log, D26a no capture while tiles load"}
        cur.setdefault("rgb_redo", []).append(entry)
        p.write_text(json.dumps(cur, indent=1), encoding="utf-8")
        return seq
    rer = {"schema": SCHEMA, "source_root": src.as_posix(), "source_sequence_sha256": RT.sha256_file(src / "sequence.json"),
           "rgb_pages": "fresh per encounter (D25)", "script_sha256": RT.sha256_file(Path(__file__)),
           "renderer_sha256": RT.sha256_file(REPO_ROOT / "scripts" / "xvs" / "render_traffic.py"),
           "page_sha256": RT.sha256_file(REPO_ROOT / "sim" / "cesium" / "page" / "xvs_page.js"),
           "profiles_sha256": RT.sha256_file(REPO_ROOT / "configs" / "xvs" / "render_profiles.json"),
           "rule": "BATCH_B3.md §1: rgb above the D24 bounds in the head/tail audit -> rgb re-rendered, masks and GT kept"}
    p = out / "sequence.json"
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8")).get("rgb_rerender", {})
        diff = [k for k in ("source_sequence_sha256", "script_sha256", "renderer_sha256", "page_sha256", "profiles_sha256") if old.get(k) != rer[k]]
        if diff:
            raise RuntimeError(f"{p} exists with different {diff}: use a new --out (no silent mixing of runs)")
        return seq
    out.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({**seq, "rgb_rerender": rer}, indent=1), encoding="utf-8")
    return seq


def archive_encounters(out: Path, ob: Path, enc_ids: set[str], tag: str) -> dict[str, int]:
    """Move the records and rgb/negative PNGs of enc_ids out of the way, so the normal resume path renders them again."""
    frames_p, done_p = ob / "frames.jsonl", ob / "encounters_done.jsonl"
    keep, moved = [], []
    for ln in frames_p.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        (moved if json.loads(ln)["encounter_id"] in enc_ids else keep).append(ln)
    dkeep, dmoved = [], []
    for ln in done_p.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            (dmoved if json.loads(ln)["encounter_id"] in enc_ids else dkeep).append(ln)
    n_png = 0
    for ln in moved:
        f = json.loads(ln)
        rels = [rr["rgb"]["png"] for rr in f["renders"].values() if rr.get("rgb")] + ([f["negative"]["png"]] if f.get("negative") else [])
        for rel in rels:
            src, dst = out / rel, out / f"replaced_{tag}" / rel
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.replace(src, dst)
                n_png += 1
    with open(ob / f"replaced_{tag}.jsonl", "a", encoding="utf-8") as fa:
        fa.writelines(ln + "\n" for ln in moved)
    with open(ob / f"replaced_{tag}_done.jsonl", "a", encoding="utf-8") as fa:
        fa.writelines(ln + "\n" for ln in dmoved)
    for p, lines in ((frames_p, keep), (done_p, dkeep)):
        tmp = p.with_suffix(".tmp")
        tmp.write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")
        os.replace(tmp, p)
    return {"frames": len(moved), "encounters": len(dmoved), "pngs": n_png}


def main() -> int:
    ap = argparse.ArgumentParser(description="G3a v0.1: re-render the rgb images of a G3a dataset with fresh rgb pages (D25)")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", required=True, help="i/n: the source block folder b<i>of<n>")
    ap.add_argument("--backend", choices=["hw", "ref"], default="hw")
    ap.add_argument("--max-wall-h", type=float, default=3.9)
    ap.add_argument("--frames", default=None, help="select JSON: re-render only these frames (tests, controls)")
    ap.add_argument("--redo-encounters", default=None, help="comma-separated encounter ids of this block to render again in place")
    ap.add_argument("--redo-tag", default="redo", help="suffix of the archive of the replaced records and PNGs")
    a = ap.parse_args()
    t_start = time.time()
    src, out = Path(a.src), Path(a.out)
    seq = write_sequence(src, out, a)
    own_fn = RT.ownship_from_scenario(seq["ownship"]["scenario"])
    rig = RT.Rig.load_nominal().make_true(20260922)
    ring = RT.TrafficRing()
    bi, bn = (int(x) for x in a.block.split("/"))
    name = f"b{bi:02d}of{bn:02d}"
    sb, ob = src / "blocks" / name, out / "blocks" / name
    ob.mkdir(parents=True, exist_ok=True)
    redo = None
    if a.redo_encounters:
        redo = set(a.redo_encounters.split(","))
        known = {d["encounter_id"] for d in RT.read_jsonl(sb / "encounters_done.jsonl")}
        if redo - known:
            raise ValueError(f"not in block {name}: {sorted(redo - known)}")
        arch = archive_encounters(out, ob, redo, a.redo_tag)
        print(f"archived for re-render ({a.redo_tag}): {arch}", flush=True)
    only = None
    if a.frames:
        only = {(d["encounter_id"], int(d["k"])) for d in json.loads(Path(a.frames).read_text(encoding="utf-8"))["frames"]}
    src_done = RT.read_jsonl(sb / "encounters_done.jsonl")
    src_frames: dict[str, list[dict[str, Any]]] = {}
    for f in RT.read_jsonl(sb / "frames.jsonl"):
        src_frames.setdefault(f["encounter_id"], []).append(f)
    frames_p, done_p = ob / "frames.jsonl", ob / "encounters_done.jsonl"
    existing = {(f["encounter_id"], f["k"]): f for f in RT.read_jsonl(frames_p)}
    done_ids = {d["encounter_id"] for d in RT.read_jsonl(done_p)}
    todo = [d for d in src_done if d["encounter_id"] not in done_ids
            and (only is None or any(e == d["encounter_id"] for e, _ in only))
            and (redo is None or d["encounter_id"] in redo)]
    print(f"G3a v0.1 rgb re-render {name}: {len(src_done)} encounters ({len(done_ids)} done, {len(todo)} to do) -> {out}", flush=True)
    recs = {r["id"]: r for r in RT.load_records(Path(seq["encounters"]["path"]))}
    acc: dict[str, Any] = {"renders": [], "frame_ms": [], "frame_bytes": [], "gt_ms": [], "incomplete": 0}
    deadline = t_start + a.max_wall_h * 3600.0 if a.max_wall_h else None
    pages = RT.PageSet(a.backend, ob, offline=False, fresh_rgb=True) if todo else None
    run_rec: dict[str, Any] = {"start_utc": datetime.now(timezone.utc).isoformat(), "backend": a.backend, "argv": sys.argv[1:],
                               "git": RT.git_state(), "complete": False, "interrupted": False, "fresh_rgb_pages": True,
                               "mode": "rgb re-render (masks and GT from the source)" + (f"; redo {a.redo_tag}: {sorted(redo)}" if redo else "")}
    new_done = 0
    loadlog = LoadLogger(ob / "load_log.jsonl").start() if pages is not None else None  # D26
    try:
        if pages is not None:
            run_rec["webgl"] = pages.r.get_webgl_diagnostics(page=pages.pages[("rgb", "ring")])
            run_rec["resolved_settings"] = pages.resolved
        RT.ensure_trailing_newline(frames_p)
        with open(frames_p, "a", encoding="utf-8") as ff:
            for n, sd in enumerate(todo):
                te = time.perf_counter()
                enc_id = sd["encounter_id"]
                rec = recs[enc_id]
                pages.start_encounter()
                pages.load(f"assets/traffic/{rec['class_id']}/{rec['model_id']}.glb")
                summ = copy.deepcopy(sd)
                summ.update({"bytes": 0, "rendered": 0, "resumed": 0, "rgb_rerendered": 0})
                for f in sorted(src_frames.get(enc_id, []), key=lambda x: x["k"]):
                    key = (enc_id, f["k"])
                    if only is not None and key not in only:
                        continue
                    if key in existing and RT.frame_files_valid(out, existing[key]):
                        summ["resumed"] += 1
                        summ["bytes"] += existing[key]["bytes"]
                        continue
                    tf = time.perf_counter()
                    st = own_fn(f["t_s"])
                    cams = RT.camera_set(rig, ring, st)
                    model_m = f["traffic"][0]["model_matrix_colmajor"]
                    pages.set_clock(f["utc"])
                    nf = copy.deepcopy(f)
                    new_sha: dict[str, tuple[str, int]] = {}

                    def rgb(cid: str, roi: Any, rel: str, hide: bool = False) -> dict[str, Any]:
                        res, info = pages.render("rgb", cams[cid], tuple(roi) if roi else None, model_m, hide_model=hide)
                        nbytes, ms_enc, sha = RT.save_png(res.image, out / rel, "RGB")
                        info.update({"png": rel, "sha256": sha, "bytes": nbytes, "ms_encode": ms_enc})
                        new_sha[rel] = (sha, nbytes)
                        acc["renders"].append({"pass": "neg" if hide else "rgb", "kind": cams[cid]["kind"], **{k2: info[k2] for k2 in
                                               ("n_renders", "ms_set", "ms_render", "ms_readback", "ms_encode", "bytes", "tiles_requested",
                                                "tiles_cache_miss", "bytes_upstream")}})
                        if info["incomplete"]:
                            acc["incomplete"] += 1
                        return info

                    for cid, rr in f["renders"].items():
                        for ps in ("A", "V"):
                            link_or_copy(src / rr[ps]["png"], out / rr[ps]["png"])
                        if rr.get("rgb"):
                            nf["renders"][cid]["rgb"] = rgb(cid, f["traffic"][0]["cams"][cid]["roi"], rr["rgb"]["png"])
                            summ["rgb_rerendered"] += 1
                    if f.get("negative"):
                        ng = f["negative"]
                        info = rgb(ng["cam"], ng["roi"], ng["png"], hide=True)
                        nf["negative"] = {"cam": ng["cam"], "roi": ng["roi"], "model_hidden": True, **info}
                        summ["rgb_rerendered"] += 1
                    for fl in nf["files"]:
                        if fl["path"] in new_sha:
                            fl["sha256"] = new_sha[fl["path"]][0]
                    mask_bytes = sum(r2["bytes"] for rr in f["renders"].values() for p2, r2 in rr.items() if p2 in ("A", "V") and r2)
                    nf["bytes"] = mask_bytes + sum(b for _, b in new_sha.values())
                    nf["ms_v0"] = f["ms"]
                    nf["ms"] = {"frame": (time.perf_counter() - tf) * 1000.0, "gt_raster_horizon": 0.0}
                    nf["rgb_rerender"] = "D25, BATCH_B3 §1 (masks and GT from the source)"
                    ff.write(json.dumps(RT.clean(nf)) + "\n")
                    ff.flush()
                    summ["rendered"] += 1
                    summ["bytes"] += nf["bytes"]
                    acc["frame_ms"].append(nf["ms"]["frame"])
                    acc["frame_bytes"].append(nf["bytes"])
                    if deadline is not None and time.time() > deadline:
                        run_rec["interrupted"] = True
                        break
                if run_rec["interrupted"]:
                    print(f"  wall-time limit reached in {enc_id}; stopping (resume with the same command)", flush=True)
                    break
                summ["wall_s"] = time.perf_counter() - te
                if only is None:
                    with open(done_p, "a", encoding="utf-8") as fd:
                        fd.write(json.dumps(RT.clean(summ)) + "\n")
                new_done += 1
                print(f"  [{n + 1:4d}/{len(todo)}] {enc_id} {rec['class_id']:<13s} frames {summ['rendered']:4d} rgb {summ['rgb_rerendered']:4d} "
                      f"{summ['wall_s']:7.1f}s  (job {time.time() - t_start:8.0f}s)", flush=True)
                if deadline is not None and time.time() > deadline:
                    run_rec["interrupted"] = True
                    print("  wall-time limit reached; stopping (resume with the same command)", flush=True)
                    break
        run_rec["complete"] = (not run_rec["interrupted"]) and (only is not None or len(RT.read_jsonl(done_p)) == len(src_done))
    finally:
        usgs = pages.usgs_stats() if pages is not None else None
        if pages is not None:
            pages.close()
        run_rec.update({"end_utc": datetime.now(timezone.utc).isoformat(), "wall_s": time.time() - t_start, "encounters_finished": new_done,
                        "usgs": usgs, "rgb_pages_opened": pages.rgb_pages_opened if pages is not None else None,
                        "d26a": dict(pages.d26a) if pages is not None else None})
        if loadlog is not None:
            run_rec["load_log"] = {"path": (ob / "load_log.jsonl").as_posix(), "readings_this_run": loadlog.stop(),
                                   "summary_all_runs": summarise_load(ob / "load_log.jsonl")}
        RT.finish_block(ob, out, run_rec, acc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
