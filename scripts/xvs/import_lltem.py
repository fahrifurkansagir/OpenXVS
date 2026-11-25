"""WP-02 G3b: parse and validate LLTEM v1.0 (MIT LL Terminal Airspace Encounter Set V1.0) and convert a seeded,
weight-aware sample of its encounters to intruder tracks relative to ownship T2 (KSFO 28L straight-in).

Inputs (scripts/xvs/fetch_lltem.py, D17, owner option 1): data/cache/lltem/README.txt, the full encounter-information
CSV and a byte prefix of the binary trajectory file. Every file is parsed; in the .dat prefix every *complete*
encounter is parsed and the truncated tail is recorded, not guessed.

Format (README §1-2; confirmed on the data, see the validation JSON):
  .dat, little-endian: uint32 n_encounters, uint32 n_aircraft; per encounter: initial positions (3 float64 per aircraft:
  along-runway ft, cross-runway ft, altitude ft above the airport) at t = 0, then per aircraft uint16 n_updates and
  n_updates x (t s, along, cross, altitude) float64. Aircraft 1 is the ownship (UA on a straight-in approach).
  CSV: 19 columns per README §1, one row per encounter; tcpa relative to the encounter start; vmd = intruder - ownship.
  Axes (checked against the CSV bearings): along-runway = runway (landing) direction, cross-runway positive = to the
  right of it, altitude up.

Conversion (the most conservative reading; see the G3b report, Open questions):
  - the intruder's position RELATIVE to the LLTEM ownship is kept exactly (in runway axes, feet -> metres), so range,
    bearing, HMD, VMD and closure of the LLTEM encounter are preserved;
  - runway axes map to KSFO 28L: along -> the runway's true heading (landing direction), cross -> 90 deg to its right,
    altitude -> local up at the threshold;
  - the relative vector is added to the T2 ownship position, with LLTEM's CPA placed at T2 t = 45 s (as E-VAL-12);
    the track covers the T2 window [0, 60] s where the LLTEM encounter has samples;
  - class: LLTEM gives only "fixed wing"; the smallest fixed-wing class C2_SEP is used (the hardest to see), variant
    seeded 1-3; alpha_trim 2 deg (§2.5);
  - weights: LLTEM has none (encounters are samples of the model), so every encounter has weight 1.0 and the
    weight-aware sample is uniform.
usage: python scripts/xvs/import_lltem.py [--n 100 --seed 20260924] -> output/xvs/wp02/g3b/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from xvs.geo import wgs84  # noqa: E402
from xvs.sim import airports, trajectory  # noqa: E402

FT = 0.3048
NM_FT = 6076.115486
CACHE = REPO_ROOT / "data" / "cache" / "lltem"
OUT = REPO_ROOT / "output" / "xvs" / "wp02" / "g3b"
T_CPA_T2 = 45.0
T2_DURATION = 60.0
CSV_COLS = ["id", "airspace_class", "own_intent", "int_intent", "int_type", "int_runway", "own_distance", "own_bearing",
            "own_alt", "own_speed", "own_trk_angle", "int_distance", "int_bearing", "int_alt", "int_speed", "int_trk_angle",
            "tcpa", "hmd_ft", "vmd_ft"]
DOC = {"n_encounters": 1_000_000, "n_aircraft": 2, "max_radius_nm": 8.0, "max_alt_ft": 3000.0, "max_duration_s": 240.0,
       "codes": {"airspace_class": {1}, "own_intent": {1}, "int_intent": {1, 2}, "int_type": {1}, "int_runway": {1}}}


def parse_dat(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    b = path.read_bytes()
    n_enc, n_ac = struct.unpack_from("<II", b, 0)
    off = 8
    encs = []
    while True:
        start = off
        try:
            if off + 24 * n_ac > len(b):
                raise EOFError
            init = np.frombuffer(b, dtype="<f8", count=3 * n_ac, offset=off).reshape(n_ac, 3).copy()
            off += 24 * n_ac
            tracks = []
            for a in range(n_ac):
                if off + 2 > len(b):
                    raise EOFError
                n = struct.unpack_from("<H", b, off)[0]
                off += 2
                if off + 32 * n > len(b):
                    raise EOFError
                arr = np.frombuffer(b, dtype="<f8", count=4 * n, offset=off).reshape(n, 4).copy()
                off += 32 * n
                tracks.append(np.vstack([np.concatenate([[0.0], init[a]]), arr]))  # t = 0 initial position first
        except EOFError:
            off = start
            break
        encs.append({"index": len(encs) + 1, "tracks": tracks})
        if len(encs) == n_enc:
            break
    header = {"n_encounters": n_enc, "n_aircraft": n_ac, "bytes": len(b), "bytes_parsed": off,
              "bytes_truncated_tail": len(b) - off, "complete_encounters": len(encs)}
    return header, encs


def parse_csv(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        head = [h.strip() for h in next(r) if h.strip()]
        if head != CSV_COLS:
            raise ValueError(f"CSV header {head} != README columns {CSV_COLS}")
        for line_no, row in enumerate(r, start=2):
            vals = [x.strip() for x in row if x.strip() != ""]
            if len(vals) != 19:
                raise ValueError(f"CSV line {line_no}: {len(vals)} values, expected 19")
            d = dict(zip(CSV_COLS, (float(x) for x in vals)))
            rows[int(d["id"])] = d
    return rows


def validate(header: dict[str, Any], encs: list[dict[str, Any]], rows: dict[int, dict[str, float]], readme: str) -> dict[str, Any]:
    v: dict[str, Any] = {"readme_licence_sentence_present":
                         "This work is licensed under a Creative Commons Attribution 4.0 International License (CC BY 4.0)." in readme}
    v["header"] = header
    v["header_matches_doc"] = header["n_encounters"] == DOC["n_encounters"] and header["n_aircraft"] == DOC["n_aircraft"]
    v["csv_rows"] = len(rows)
    v["csv_ids_1_to_n"] = sorted(rows) == list(range(1, len(rows) + 1))
    code_bad = {k: int(sum(1 for d in rows.values() if int(d[k]) not in ok)) for k, ok in DOC["codes"].items()}
    v["csv_code_values_outside_doc"] = code_bad
    ints = np.array([d["int_intent"] for d in rows.values()])
    v["csv_intruder_intent_share"] = {"landing": float(np.mean(ints == 1)), "takeoff": float(np.mean(ints == 2))}
    for k in ("own_alt", "int_alt", "own_distance", "int_distance", "tcpa", "hmd_ft", "vmd_ft", "own_speed", "int_speed"):
        a = np.array([d[k] for d in rows.values()])
        v.setdefault("csv_ranges", {})[k] = {"min": float(a.min()), "median": float(np.median(a)), "max": float(a.max())}
    # trajectories of the parsed prefix
    dur, rad, alt, dt_bad, hmd_err, vmd_err, tcpa_bad, n_samples = [], [], [], 0, [], [], 0, 0
    bearing_err: dict[str, list[float]] = {}
    dist_err: list[float] = []
    for e in encs:
        own, intr = e["tracks"][0], e["tracks"][1]
        for tr in e["tracks"]:
            dur.append(float(tr[-1, 0] - tr[0, 0]))
            rad.extend(np.hypot(tr[:, 1], tr[:, 2]).tolist())
            alt.extend(tr[:, 3].tolist())
            dt_bad += int(np.count_nonzero(np.abs(np.diff(tr[:, 0]) - 1.0) > 1e-9))
            n_samples += len(tr)
        row = rows[e["index"]]
        k = np.nonzero(np.isclose(own[:, 0], row["tcpa"]))[0]
        if len(k) != 1 or not np.allclose(own[:, 0], intr[:, 0]):
            tcpa_bad += 1
            continue
        k = int(k[0])
        hmd_err.append(abs(math.hypot(intr[k, 1] - own[k, 1], intr[k, 2] - own[k, 2]) - row["hmd_ft"]))
        vmd_err.append(abs((intr[k, 3] - own[k, 3]) - row["vmd_ft"]))
        # axis check (statistical): the CSV holds the CPA conditions as SAMPLED from the model (README §1), which the
        # generated trajectories approximate; compare the CSV bearing with atan2(cross, along) (cross positive to the
        # right of the runway direction) and with the mirrored hypothesis atan2(-cross, along)
        for tr, pre in ((own, "own"), (intr, "int")):
            if math.hypot(tr[k, 1], tr[k, 2]) <= 500.0:  # bearing ill-defined near the airport reference point
                continue
            for hyp, sign in (("right", 1.0), ("mirrored", -1.0)):
                b = math.degrees(math.atan2(sign * tr[k, 2], tr[k, 1])) % 360.0
                bearing_err.setdefault(hyp, []).append(abs((b - row[f"{pre}_bearing"] + 180.0) % 360.0 - 180.0))
            dist_err.append(math.hypot(tr[k, 1], tr[k, 2]) / NM_FT - row[f"{pre}_distance"])
    v["dat_prefix"] = {
        "encounters_parsed": len(encs), "samples": n_samples,
        "duration_s_max": max(dur), "duration_over_doc_max": int(sum(1 for d in dur if d > DOC["max_duration_s"])),
        "radius_nm_max": max(rad) / NM_FT, "radius_over_doc_max": int(sum(1 for r in rad if r / NM_FT > DOC["max_radius_nm"] + 1e-9)),
        "alt_ft_min": min(alt), "alt_ft_max": max(alt), "alt_over_doc_max": int(sum(1 for a in alt if a > DOC["max_alt_ft"] + 1e-9)),
        "time_steps_not_1s": dt_bad, "tcpa_not_on_a_sample_or_times_differ": tcpa_bad,
        "csv_vs_dat_hmd_abs_err_ft_max": max(hmd_err) if hmd_err else None,
        "csv_vs_dat_vmd_abs_err_ft_max": max(vmd_err) if vmd_err else None,
        "axis_check": {
            "note": "CSV CPA conditions are sampled model conditions (README §1); the trajectories approximate them",
            "n": len(bearing_err.get("right", [])),
            "bearing_abs_diff_deg_cross_right": {q: float(np.quantile(bearing_err["right"], v)) for q, v in (("median", 0.5), ("p90", 0.9), ("p99", 0.99))},
            "bearing_abs_diff_deg_cross_mirrored": {q: float(np.quantile(bearing_err["mirrored"], v)) for q, v in (("median", 0.5), ("p90", 0.9), ("p99", 0.99))},
            "distance_diff_nm_dat_minus_csv": {q: float(np.quantile(dist_err, v)) for q, v in (("p01", 0.01), ("median", 0.5), ("p99", 0.99))},
        },
    }
    # is the prefix (encounters 1..n) a representative sample of the full set? CSV statistics, prefix vs all rows
    pre = [rows[e["index"]] for e in encs]
    comp = {}
    for k in ("int_intent", "hmd_ft", "vmd_ft", "tcpa", "own_alt", "int_alt", "own_distance", "int_speed"):
        a_all = np.array([d[k] for d in rows.values()])
        a_pre = np.array([d[k] for d in pre])
        comp[k] = {"all_mean": float(a_all.mean()), "prefix_mean": float(a_pre.mean()),
                   "all_median": float(np.median(a_all)), "prefix_median": float(np.median(a_pre)),
                   "prefix_mean_z": float((a_pre.mean() - a_all.mean()) / (a_all.std() / math.sqrt(len(a_pre))))}
    v["prefix_vs_all_csv"] = comp
    v["errors"] = (int(not v["readme_licence_sentence_present"]) + int(not v["header_matches_doc"]) + int(not v["csv_ids_1_to_n"])
                   + sum(code_bad.values()) + v["dat_prefix"]["duration_over_doc_max"] + v["dat_prefix"]["radius_over_doc_max"]
                   + v["dat_prefix"]["alt_over_doc_max"] + dt_bad + tcpa_bad)
    return v


def to_t2_track(e: dict[str, Any], row: dict[str, float], thr: Any, variant: int) -> tuple[dict[str, Any], dict[str, Any]]:
    own, intr = e["tracks"][0], e["tracks"][1]
    rel_ft = intr[:, 1:] - own[:, 1:]                       # along, cross, up (ft), intruder - ownship
    t_l = own[:, 0]
    t_t2 = t_l - row["tcpa"] + T_CPA_T2
    h = math.radians(thr.heading_deg)
    a_ned = np.array([math.cos(h), math.sin(h), 0.0])
    c_ned = np.array([math.cos(h + math.pi / 2), math.sin(h + math.pi / 2), 0.0])  # right of the landing direction
    u_ned = np.array([0.0, 0.0, -1.0])
    n_thr = wgs84.r_ecef_ned(thr.lat_deg, thr.lon_deg)
    m = n_thr @ np.column_stack([a_ned, c_ned, u_ned]) * FT  # runway-axis feet -> ECEF metres
    rel_ecef = rel_ft @ m.T
    drel = np.gradient(rel_ecef, t_l, axis=0)
    keep = (t_t2 >= 0.0) & (t_t2 <= T2_DURATION)
    idx = np.nonzero(keep)[0]
    if len(idx) >= 1:  # one sample of margin on each side, for interpolation up to the window edges
        idx = np.arange(max(0, idx[0] - 1), min(len(t_l), idx[-1] + 2))
    p, v, below = [], [], 0
    for i in idx:
        st = trajectory.t2_ksfo28l_state(float(t_t2[i]), thr=thr)
        v_o = wgs84.r_ecef_ned(st.lat_deg, st.lon_deg) @ st.v_ned_mps
        pi = st.p_ecef_m + rel_ecef[i]
        p.append(pi.tolist())
        v.append((v_o + drel[i]).tolist())
        below += int(wgs84.ecef_to_geodetic(*pi)[2] < thr.h_m)
    rec = {"id": f"LLTEM_{e['index']:07d}", "class_id": "C2_SEP", "model_id": f"C2_SEP_{variant:04d}", "alpha_trim_deg": 2.0,
           "weight": 1.0, "stratum": {"source": "lltem_v1", "intruder_intent": "landing" if int(row["int_intent"]) == 1 else "takeoff"},
           "source": {"dataset": "lltem_v1", "encounter": e["index"], "tcpa_s": row["tcpa"], "hmd_ft": row["hmd_ft"], "vmd_ft": row["vmd_ft"],
                      "t2_cpa_s": T_CPA_T2},
           "track": {"t_s": t_t2[idx].tolist(), "p_ecef_m": p, "v_ecef_mps": v}}
    info = {"id": rec["id"], "samples": int(len(idx)), "t2_window_covered_s": [float(t_t2[idx][0]), float(t_t2[idx][-1])] if len(idx) else None,
            "samples_below_threshold_height": below}
    return rec, info


def main() -> int:
    ap = argparse.ArgumentParser(description="Import LLTEM v1.0 for WP-02 G3b")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--out-dir", default=None, help="output folder (default: output/xvs/wp02/g3b)")
    a = ap.parse_args()
    out = Path(a.out_dir) if a.out_dir else OUT
    out.mkdir(parents=True, exist_ok=True)
    readme = (CACHE / "README.txt").read_text(encoding="utf-8", errors="replace")
    rows = parse_csv(CACHE / "terminal_encounter_info_20200630.csv")
    header, encs = parse_dat(CACHE / "terminal_encounters_20200630.dat")
    val = validate(header, encs, rows, readme)
    val["files_sha256"] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(CACHE.glob("*")) if f.suffix != ".part"}
    (out / "lltem_validation.json").write_text(json.dumps(val, indent=1), encoding="utf-8")
    print(json.dumps({k: val[k] for k in ("errors", "csv_rows", "header", "dat_prefix")}, indent=1))
    if val["errors"]:
        print("validation errors: no import (fix or report)")
        return 1
    # weight-aware seeded sample: LLTEM carries no weights -> weight 1.0 each -> uniform without replacement
    rng = np.random.default_rng(a.seed)
    w = np.ones(len(encs))
    pick = sorted(rng.choice(len(encs), size=min(a.n, len(encs)), replace=False, p=w / w.sum()).tolist())
    thr = airports.load_ksfo_28l_threshold()
    recs, infos = [], []
    for i in pick:
        e = encs[i]
        variant = int(np.random.default_rng([a.seed, e["index"]]).integers(1, 4))
        rec, info = to_t2_track(e, rows[e["index"]], thr, variant)
        if info["samples"] < 2:
            info["dropped"] = "fewer than 2 samples in the T2 window"
        else:
            recs.append(rec)
        infos.append(info)
    enc_path = out / "lltem_t2_encounters.jsonl"
    with open(enc_path, "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    summary = {"schema": "xvs.lltem_import/1", "seed": a.seed, "n_requested": a.n, "n_parsed_encounters": len(encs),
               "picked_encounters": [encs[i]["index"] for i in pick], "n_tracks": len(recs),
               "tracks_path": enc_path.resolve().relative_to(REPO_ROOT).as_posix(), "tracks_sha256": hashlib.sha256(enc_path.read_bytes()).hexdigest(),
               "mapping": {"relative_to": "T2 ownship", "t2_cpa_s": T_CPA_T2, "runway": f"{thr.airport_ident} {thr.runway_ident}",
                           "heading_deg_true": thr.heading_deg, "threshold": [thr.lat_deg, thr.lon_deg, thr.h_m], "ft_to_m": FT,
                           "axes": "along -> landing heading, cross -> 90 deg right, altitude -> up at the threshold"},
               "class": "C2_SEP (LLTEM gives only 'fixed wing'), variant seeded 1-3", "weights": "none in LLTEM -> 1.0 each",
               "samples_below_threshold_height": sum(i["samples_below_threshold_height"] for i in infos),
               "samples_total": sum(i["samples"] for i in infos),
               "tracks_with_samples_below_threshold_height": sum(1 for i in infos if i["samples_below_threshold_height"] > 0),
               "per_track": infos}
    (out / "lltem_import.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_track", "picked_encounters")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
