"""WP-02 G3a renderer (scripts/xvs/render_traffic.py) and sampled tracks (xvs.traffic.track)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from xvs.sim import trajectory
from xvs.traffic.encounter import EncounterParams, construct_encounter
from xvs.traffic.track import TrackParams, TrackTrajectory

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "xvs"))
import render_traffic as RT  # noqa: E402


def test_blocks_are_a_round_robin_partition():
    recs = [{"id": f"E{i:04d}"} for i in range(23)]
    blocks = [RT.block_records(recs, i, 5) for i in range(5)]
    ids = [r["id"] for b in blocks for r in b]
    assert sorted(ids) == sorted(r["id"] for r in recs) and len(ids) == len(set(ids))
    assert [r["id"] for r in blocks[2]] == ["E0002", "E0007", "E0012", "E0017", "E0022"]
    with pytest.raises(ValueError):
        RT.block_records(recs, 5, 5)


def test_frame_selection_reasons():
    ring_in = {c: False for c in RT.RING_CAMS}
    assert RT.select_frame(12000.0, {**ring_in, "T0": True}) == "range_gt_10km"
    assert RT.select_frame(5000.0, ring_in) == "centre_not_in_ring"
    assert RT.select_frame(10000.0, {**ring_in, "T3": True}) is None
    assert RT.select_frame(float("nan"), {**ring_in, "T3": True}) == "range_gt_10km"


def test_negative_choice_is_seeded_and_in_range():
    a = RT.negative_choice("E0042", 17, ["T1", "T4", "T6"], 2448, 2048)
    assert a == RT.negative_choice("E0042", 17, ["T1", "T4", "T6"], 2448, 2048)
    cid, u0, v0 = a
    assert cid in ("T1", "T4", "T6") and 0 <= u0 <= 2448 - 1024 and 0 <= v0 <= 2048 - 1024
    draws = {RT.negative_choice("E0042", k, ["T1", "T4", "T6"], 2448, 2048) for k in range(40)}
    assert len(draws) > 30
    assert RT.negative_choice("E0042", 17, [], 2448, 2048) is None


def test_mask_grey_counts_impure_pixels():
    img = np.zeros((4, 4, 3), np.uint8)
    img[0, 0] = 255
    img[1, 1] = (255, 0, 0)
    img[2, 2] = 128
    grey, impure = RT.mask_grey(img)
    assert impure == 2 and grey[0, 0] == 255


def test_clean_makes_json_safe():
    assert RT.clean({"a": float("nan"), "b": [np.float32(1.5), np.int64(3), np.bool_(True)]}) == {"a": None, "b": [1.5, 3, True]}


def test_resumed_frame_counts_like_a_rendered_frame():
    """R-G3a-6: a frame kept by an earlier job is counted into the encounter summary like a rendered one."""
    rec = {"kept_index": 5, "bytes": 1234, "rig_rgb_skipped_small": ["U"],
           "traffic": [{"cams": {"T0": {"kind": "ring"}, "F": {"kind": "rig"}, "U": {"kind": "rig"}}}],
           "renders": {"T0": {"A": {"png": "a"}, "V": {"png": "v"}, "rgb": {"png": "c"}}, "F": {"A": {"png": "a"}, "V": {"png": "v"}, "rgb": {"png": "c"}},
                       "U": {"A": {"png": "a"}, "V": {"png": "v"}, "rgb": None}},
           "negative": {"cam": "T3"}}
    summ = {"negatives": 0, "negatives_none_free": 0, "rig_rgb_skipped_small": 0, "bytes": 0,
            "images": {"rgb_ring": 0, "rgb_rig": 0, "A": 0, "V": 0, "negative": 0}}
    RT.add_resumed_counts(summ, rec)
    assert summ["images"] == {"rgb_ring": 1, "rgb_rig": 1, "A": 3, "V": 3, "negative": 1}
    assert (summ["negatives"], summ["negatives_none_free"], summ["rig_rgb_skipped_small"], summ["bytes"]) == (1, 0, 1, 1234)
    RT.add_resumed_counts(summ, {**rec, "negative": None, "kept_index": 10})
    assert summ["negatives"] == 1 and summ["negatives_none_free"] == 1 and summ["bytes"] == 2468


def test_trailing_newline_repair(tmp_path):
    p = tmp_path / "frames.jsonl"
    assert not RT.ensure_trailing_newline(p)
    p.write_text('{"a": 1}\n{"b": 2', encoding="utf-8")  # a record cut by a crash
    assert RT.ensure_trailing_newline(p) and not RT.ensure_trailing_newline(p)
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"c": 3}\n')
    assert [r for r in RT.read_jsonl(p)] == [{"a": 1}, {"c": 3}]


def test_track_reproduces_a_parametric_encounter():
    params = EncounterParams(id="E9", class_id="C2_SEP", model_id="C2_SEP_0001", t_cpa=40.0, hmd_m=300.0, side="left",
                             vmd_m=-120.0, approach_deg=100.0, gs_mps=60.0, vs_mps=3.0, alpha_trim_deg=2.0)
    enc = construct_encounter(trajectory.t1_cruise_state, params)
    ts = np.arange(0.0, 60.0 + 1e-9, 2.5)
    tr = TrackTrajectory(TrackParams(id="E9", class_id="C2_SEP", model_id="C2_SEP_0001", alpha_trim_deg=2.0),
                         ts, np.array([enc.position_at(t) for t in ts]), np.array([enc.velocity_at(t) for t in ts]))
    for t in (0.0, 1.3, 17.77, 40.0, 59.999, 60.0):
        assert np.abs(tr.position_at(t) - enc.position_at(t)).max() < 1e-6
        assert np.abs(tr.velocity_at(t) - enc.velocity_at(t)).max() < 1e-9
        assert np.abs(tr.attitude_at(t)[0] - enc.attitude_at(t)[0]).max() < 1e-9
    assert tr.covers(0.0) and not tr.covers(60.01)
    with pytest.raises(ValueError):
        tr.position_at(-0.1)


def test_track_from_record_and_validation():
    rec = {"id": "L1", "class_id": "C3_TWIN", "model_id": "C3_TWIN_0002", "weight": 0.25,
           "track": {"t_s": [0.0, 1.0], "p_ecef_m": [[0, 0, 6.4e6], [10, 0, 6.4e6]], "v_ecef_mps": [[10, 0, 0], [10, 0, 0]]}}
    tr = TrackTrajectory.from_record(rec)
    assert tr.params.weight == 0.25 and np.isnan(tr.params.t_cpa)
    assert np.allclose(tr.position_at(0.5), [5, 0, 6.4e6])
    with pytest.raises(ValueError):
        TrackTrajectory(tr.params, np.array([0.0, 0.0]), np.zeros((2, 3)), np.zeros((2, 3)))


@pytest.mark.render
def test_render_traffic_end_to_end_ref(tmp_path):
    """One near encounter, 5 frames at 1 Hz on ref (tiles from the cache only): masks, GT-2, purity, negative."""
    enc = {"id": "E_TEST", "class_id": "C2_SEP", "model_id": "C2_SEP_0001", "t_cpa": 6.0, "hmd_m": 120.0, "side": "right",
           "vmd_m": 80.0, "approach_deg": 180.0, "gs_mps": 60.0, "vs_mps": 0.0, "alpha_trim_deg": 2.0, "weight": 1.0,
           "stratum": {"test": "near"}}
    ep = tmp_path / "enc.jsonl"
    ep.write_text(json.dumps(enc) + "\n", encoding="utf-8")
    out = tmp_path / "g3a"
    cp = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "xvs" / "render_traffic.py"), "--encounters", str(ep),
                         "--ownship", str(REPO_ROOT / "configs" / "xvs" / "scenarios" / "t1_cruise.json"), "--rate-hz", "1",
                         "--max-frames", "5", "--backend", "ref", "--offline", "--out", str(out), "--max-wall-h", "0.3"],
                        cwd=REPO_ROOT, capture_output=True, text=True, timeout=1500)
    assert cp.returncode == 0, cp.stdout[-3000:] + cp.stderr[-3000:]
    bdir = out / "blocks" / "b00of01"
    frames = [json.loads(ln) for ln in (bdir / "frames.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    done = [json.loads(ln) for ln in (bdir / "encounters_done.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(done) == 1 and done[0]["kept"] == len(frames) == 5
    seq = json.loads((out / "sequence.json").read_text(encoding="utf-8"))
    assert seq["n_frames_per_encounter"] == 5 and "C2_SEP_0001" in seq["models"]
    n_masks = 0
    for f in frames:
        for f2 in f["files"]:
            assert (out / f2["path"]).exists()
        for cid, rr in f["renders"].items():
            assert not rr["A"]["incomplete"] and rr["A"]["impure_px"] == 0 and rr["V"]["impure_px"] == 0
            c = f["traffic"][0]["cams"][cid]
            assert c["gt2"]["computable"] and c["gt2"]["mismatch_px"] == 0
            assert c["visible_outside_amodal_px"] == 0 and c["outside_analytic_box_px"] in (0, None)
            n_masks += 2
        assert any(f["traffic"][0]["cams"][c]["in_image"] for c in RT.RING_CAMS)
    assert n_masks >= 10
    assert frames[4]["negative"] is not None and frames[4]["negative"]["model_hidden"]
    assert all(fr["negative"] is None for fr in frames[:4])
    perf = json.loads((bdir / "perf.json").read_text(encoding="utf-8"))
    assert perf["totals"]["frames_kept"] == 5 and perf["totals"]["violations"] == 0
    # resume: a second run of the same command renders nothing new
    cp2 = subprocess.run(cp.args, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    assert cp2.returncode == 0
    assert len((bdir / "frames.jsonl").read_text(encoding="utf-8").splitlines()) == 5
    # mid-encounter resume (R-G3a-6): keep 3 frame records plus a record cut by a crash, drop the encounter summary;
    # the resumed encounter must count all 5 frames
    lines = (bdir / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    (bdir / "frames.jsonl").write_text("\n".join(lines[:3]) + "\n" + lines[3][: len(lines[3]) // 2], encoding="utf-8")
    (bdir / "encounters_done.jsonl").unlink()
    cp3 = subprocess.run(cp.args, cwd=REPO_ROOT, capture_output=True, text=True, timeout=1500)
    assert cp3.returncode == 0, cp3.stdout[-3000:] + cp3.stderr[-3000:]
    frames3 = RT.read_jsonl(bdir / "frames.jsonl")
    done3 = RT.read_jsonl(bdir / "encounters_done.jsonl")
    assert sorted(f["k"] for f in frames3) == sorted(f["k"] for f in frames) and len(done3) == 1
    d0, d3 = done[0], done3[0]
    assert d3["resumed"] == 3 and d3["rendered"] == 2 and d3["kept"] == 5
    for key in ("images", "negatives", "negatives_none_free", "rig_rgb_skipped_small"):
        assert d3[key] == d0[key], key
    assert d3["bytes"] == sum(f["bytes"] for f in frames3)


def test_d26a_retry_once_then_violation(tmp_path):
    """D26a: an incomplete rgb render is retried once in a fresh page; a second failure is a recorded violation."""
    from types import SimpleNamespace
    ps = RT.PageSet.__new__(RT.PageSet)
    ps.d26a = {"retries": 0, "violations": 0, "retry_ok": 0}
    ps.log_dir, ps.clock, ps.loaded, ps.rgb_pages_opened = tmp_path, "2026-01-01T00:00:00.000Z", "m.glb", 2
    renewed = []
    ps.renew_rgb_page = lambda kind: renewed.append(kind)
    outcomes = []

    def once(p, cam, roi, m, hide):
        inc = outcomes.pop(0)
        return SimpleNamespace(incomplete=inc, tile_state={"pendingRequests": 3 if inc else 0}, n_renders=600 if inc else 9), \
            {"n_renders": 600 if inc else 9, "tiles_requested": 1, "tiles_cache_miss": 0}
    ps._render_once = once
    cam = {"kind": "ring"}
    outcomes[:] = [False]
    res, info = ps.render("rgb", cam, (0, 0), [])
    assert not res.incomplete and ps.d26a == {"retries": 0, "violations": 0, "retry_ok": 0} and not renewed
    outcomes[:] = [True, False]
    res, info = ps.render("rgb", cam, (0, 0), [])
    assert not res.incomplete and info["d26a_retry"]["ok"] and ps.d26a["retry_ok"] == 1 and renewed == ["ring"]
    outcomes[:] = [True, True]
    res, info = ps.render("rgb", cam, (0, 0), [])
    assert res.incomplete and ps.d26a == {"retries": 2, "violations": 1, "retry_ok": 1}
    v = [json.loads(ln) for ln in (tmp_path / "render_violations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(v) == 1 and v[0]["rule"] == "D26a" and v[0]["tile_state"]["pendingRequests"] == 3
    outcomes[:] = [True]
    res, info = ps.render("A", cam, (0, 0), [])  # masks: no D26a retry
    assert res.incomplete and ps.d26a["retries"] == 2
