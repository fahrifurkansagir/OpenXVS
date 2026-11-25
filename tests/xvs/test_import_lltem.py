"""WP-02 G3b LLTEM importer (scripts/xvs/import_lltem.py): binary layout, truncated prefix, geometry preservation."""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "xvs"))
import import_lltem as IL  # noqa: E402

from xvs.geo import wgs84  # noqa: E402
from xvs.sim import airports, trajectory  # noqa: E402


def _encounter_bytes(own: np.ndarray, intr: np.ndarray) -> bytes:
    """own/intr: (n+1, 4) rows (t, along, cross, alt); row 0 is the initial position at t = 0."""
    b = struct.pack("<6d", *own[0, 1:], *intr[0, 1:])
    for tr in (own, intr):
        b += struct.pack("<H", len(tr) - 1) + tr[1:].astype("<f8").tobytes()
    return b


def _synthetic(n_updates: int = 60) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(0.0, n_updates + 1.0)
    own = np.column_stack([t, -12000.0 + 200.0 * t, np.zeros_like(t), 1500.0 - 5.0 * t])
    intr = np.column_stack([t, -3000.0 + 0.0 * t, 4000.0 - 120.0 * t, 1200.0 + 0.0 * t])
    return own, intr


def test_parse_dat_complete_encounters_and_truncated_tail(tmp_path):
    own, intr = _synthetic()
    enc = _encounter_bytes(own, intr)
    data = struct.pack("<II", 1_000_000, 2) + enc + enc + enc[: len(enc) // 2]
    p = tmp_path / "x.dat"
    p.write_bytes(data)
    header, encs = IL.parse_dat(p)
    assert header["n_encounters"] == 1_000_000 and header["n_aircraft"] == 2
    assert header["complete_encounters"] == 2 and header["bytes_truncated_tail"] == len(enc) // 2
    assert np.array_equal(encs[1]["tracks"][0], own) and np.array_equal(encs[1]["tracks"][1], intr)


def test_to_t2_track_keeps_the_relative_geometry_at_cpa():
    own, intr = _synthetic()
    k = 30  # CPA sample of this synthetic encounter
    row = {"tcpa": float(own[k, 0]), "int_intent": 1.0, "hmd_ft": 0.0, "vmd_ft": 0.0}
    thr = airports.load_ksfo_28l_threshold()
    rec, info = IL.to_t2_track({"index": 7, "tracks": [own, intr]}, row, thr, variant=2)
    assert rec["model_id"] == "C2_SEP_0002" and rec["weight"] == 1.0
    t = np.array(rec["track"]["t_s"])
    i = int(np.argmin(np.abs(t - IL.T_CPA_T2)))
    assert abs(t[i] - IL.T_CPA_T2) < 1e-9
    st = trajectory.t2_ksfo28l_state(IL.T_CPA_T2, thr=thr)
    rel_ned = wgs84.r_ecef_ned(thr.lat_deg, thr.lon_deg).T @ (np.array(rec["track"]["p_ecef_m"][i]) - st.p_ecef_m)
    along, cross, up = (intr[k, 1:] - own[k, 1:]) * IL.FT
    h = math.radians(thr.heading_deg)
    expect = along * np.array([math.cos(h), math.sin(h), 0]) + cross * np.array([-math.sin(h), math.cos(h), 0]) + up * np.array([0, 0, -1])
    assert np.abs(rel_ned - expect).max() < 1e-6
    # horizontal miss distance and vertical separation of the LLTEM sample are preserved exactly
    assert abs(math.hypot(rel_ned[0], rel_ned[1]) - math.hypot(along, cross)) < 1e-6
    assert abs(-rel_ned[2] - up) < 1e-6
