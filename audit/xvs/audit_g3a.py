"""Director's independent audit of a WP-02 G3a-format dataset (G3a on T1, G3b on T2), on a seeded sample of kept frames.

Uses only oracle.py (own geodesy, rig_true, ring, T1, encounter construction, GLB parser, projection, dense limb,
point-sampled rasterizer) and the saved frame records and PNGs. For every sampled frame and every one of the 14 cameras
it recomputes the GT and compares it with the record; for every rendered camera it compares the saved A mask with the
oracle's rasterization (meshes crossing the 0.5 m near plane are clipped here, with this file's own clipper) and
classifies each differing pixel with the oracle-side union-boundary test (tie rule of GT-2 v1.1).

- Intruder pose: parametric encounters are rebuilt with oracle.construct; sampled tracks (records with "track", G3b)
  are interpolated here (cubic Hermite on the recorded samples) and oriented like oracle.intruder_pose.
- Ownship: T1 is recomputed with oracle.t1_state. The oracle has no T2; with `--ownship recorded` the recorded
  ownship state is used and only its downstream quantities are checked (reported as such).
usage: python audit/xvs/audit_g3a.py <root> <encounters.jsonl> <n> <seed> [--ownship t1|recorded] [--extra ENC:K ...] [--out DIR]
       -> DIR/audit_g3a_<root name>.json (default DIR: output/xvs/audit/wp02_g3a)
"""
import json, math, re, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import oracle as O  # noqa: E402

ARGS, OUT = O.parse_out(sys.argv[1:], 'wp02_g3a')
OWN_MODE = 't1'
if '--ownship' in ARGS:
    i = ARGS.index('--ownship'); OWN_MODE = ARGS[i + 1]; del ARGS[i:i + 2]
EXTRA = []
if '--extra' in ARGS:
    i = ARGS.index('--extra'); EXTRA = [(e.split(':')[0], int(e.split(':')[1])) for e in ARGS[i + 1:]]; del ARGS[i:]
ROOT, ENC_P, N, SEED = Path(ARGS[0]), Path(ARGS[1]), int(ARGS[2]), int(ARGS[3])
ASSETS = O.REPO_ROOT / 'assets' / 'traffic'
Z_NEAR, TIE = 0.5, 1.0 / 16.0

# ---- index of kept frames (block, byte offset, encounter, k) ----
rx = re.compile(rb'"encounter_id": "([^"]+)", "k": (\d+)')
idx = []
for b in sorted((ROOT / 'blocks').glob('b*of*')):
    off = 0
    with open(b / 'frames.jsonl', 'rb') as fh:
        for ln in fh:
            m = rx.search(ln[:300])
            if m and ln.endswith(b'\n'):
                idx.append((b.name, off, m.group(1).decode(), int(m.group(2))))
            off += len(ln)
rng = np.random.default_rng(SEED)
pick = sorted(rng.choice(len(idx), size=min(N, len(idx)), replace=False).tolist())
extra_set = set(EXTRA)
sel = [idx[i] for i in pick] + [r for r in idx if (r[2], r[3]) in extra_set and idx.index(r) not in pick]
encs = {}
for ln in ENC_P.read_text(encoding='utf-8').splitlines():
    if ln.strip():
        d = json.loads(ln); encs[d['id']] = d
MESH = {}


def mesh(cls, mid):
    if mid not in MESH:
        MESH[mid] = O.glb_mesh(ASSETS / cls / f'{mid}.glb')
    return MESH[mid]


def hermite(tr, t):
    """Cubic Hermite position and velocity at t from the recorded track samples."""
    ts = np.asarray(tr['t_s'], float); P = np.asarray(tr['p_ecef_m'], float); Vv = np.asarray(tr['v_ecef_mps'], float)
    j = int(np.clip(np.searchsorted(ts, t, side='right') - 1, 0, len(ts) - 2))
    h = ts[j + 1] - ts[j]; s = (t - ts[j]) / h
    h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
    p = h00 * P[j] + h10 * h * Vv[j] + h01 * P[j + 1] + h11 * h * Vv[j + 1]
    d00, d10, d01, d11 = 6 * s**2 - 6 * s, 3 * s**2 - 4 * s + 1, -6 * s**2 + 6 * s, 3 * s**2 - 2 * s
    v = (d00 * P[j] + d01 * P[j + 1]) / h + d10 * Vv[j] + d11 * Vv[j + 1]
    return p, v


def intruder(e, t):
    cls = e['class_id']
    if 'track' in e:
        p, v = hermite(e['track'], t)
        E = {'P_cpa': p, 'V': v, 't_cpa': t}
        trim = e.get('alpha_trim_deg', O.ALPHA_TRIM[cls])
        p_i, R_i, ypr = O.intruder_pose(E, t, cls)
        if trim != O.ALPHA_TRIM[cls]:  # pitch = flight-path angle + the record's trim
            lat, lon, _ = O.ecef_to_geodetic(p_i)
            Ni = O.r_ecef_ned(lat, lon)
            R_i = Ni @ O.r_ned_body(ypr[0], ypr[1] - O.ALPHA_TRIM[cls] + trim, 0.0)
        return p_i, R_i
    E = O.construct(O.t1_state, e['t_cpa'], e['hmd_m'], e['side'], e['vmd_m'], e['approach_deg'], e['gs_mps'], e['vs_mps'])
    p_i, R_i, _ = O.intruder_pose(E, t, cls)
    return p_i, R_i


def clip_near(pc, F):
    """Clip camera-frame triangles at z = Z_NEAR (Sutherland-Hodgman on one plane), fan-triangulated."""
    vs, fs = [], []
    for tri in F:
        P = pc[tri]; poly = []
        for i in range(3):
            a, b = P[i], P[(i + 1) % 3]
            ina, inb = a[2] >= Z_NEAR, b[2] >= Z_NEAR
            if ina:
                poly.append(a)
            if ina != inb:
                q = a + (Z_NEAR - a[2]) / (b[2] - a[2]) * (b - a); q[2] = Z_NEAR; poly.append(q)
        for j in range(1, len(poly) - 1):
            base = len(vs); vs += [poly[0], poly[j], poly[j + 1]]; fs.append([base, base + 1, base + 2])
    return np.array(vs, float).reshape(-1, 3), np.array(fs, int).reshape(-1, 3)


def union_boundary_near(px, py, uu, vv, F, r=TIE, nn=32):
    ang = 2 * np.pi * np.arange(nn) / nn
    qx = px + np.concatenate([[0.0], r * np.cos(ang), 0.5 * r * np.cos(ang)])
    qy = py + np.concatenate([[0.0], r * np.sin(ang), 0.5 * r * np.sin(ang)])
    inside = np.zeros(len(qx), bool)
    for a, b, c in F:
        x1, y1, x2, y2, x3, y3 = uu[a], vv[a], uu[b], vv[b], uu[c], vv[c]
        d = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        if d == 0:
            continue
        l1 = ((x2 - qx) * (y3 - qy) - (y2 - qy) * (x3 - qx)) / d
        l2 = ((x3 - qx) * (y1 - qy) - (y3 - qy) * (x1 - qx)) / d
        inside |= (l1 >= 0) & (l2 >= 0) & (1 - l1 - l2 >= 0)
    return bool(inside.any() and not inside.all())


def mstats(m, u0, v0):
    ys, xs = np.nonzero(m)
    if not len(xs):
        return 0, None, None
    return int(len(xs)), [int(xs.min()) + u0, int(ys.min()) + v0, int(xs.max()) + u0, int(ys.max()) + v0], [float(xs.mean()) + u0, float(ys.mean()) + v0]


rows = []
t0 = time.time()
for blk, off, enc_id, k in sel:
    with open(ROOT / 'blocks' / blk / 'frames.jsonl', 'rb') as fh:
        fh.seek(off); f = json.loads(fh.readline().decode('utf-8'))
    assert (f['encounter_id'], f['k']) == (enc_id, k)
    t = f['traffic'][0]; e = encs[enc_id]
    if OWN_MODE == 't1':
        own = O.t1_state(f['t_s'])
        own_err = (float(np.abs(own['p'] - np.array(f['ownship']['p_ecef_m'])).max()),
                   float(np.abs(own['R_ecef_body'] - np.array(f['ownship']['R_ecef_body'])).max()))
    else:
        own = {'p': np.array(f['ownship']['p_ecef_m']), 'R_ecef_body': np.array(f['ownship']['R_ecef_body'])}
        own_err = (None, None)
    p_i, R_i = intruder(e, f['t_s'])
    M = np.array(t['model_matrix_colmajor']).reshape(4, 4).T
    row = {'frame': [enc_id, k], 'class': e['class_id'], 'range_m': t['range_m'], 'own_p_err_m': own_err[0], 'own_R_err': own_err[1],
           'intr_p_err_m': float(np.abs(p_i - np.array(t['p_ecef_m'])).max()), 'intr_p_vs_matrix_m': float(np.abs(M[:3, 3] - np.array(t['p_ecef_m'])).max()),
           'intr_R_err': float(np.abs(R_i - M[:3, :3]).max()), 'range_err_m': abs(float(np.linalg.norm(p_i - own['p'])) - t['range_m']), 'cams': {}}
    V, F = mesh(e['class_id'], e['model_id'])
    Pw = p_i + V @ R_i.T
    Pw_rec = M[:3, 3] + V @ M[:3, :3].T
    for cid, c in t['cams'].items():
        C, R, intr = O.cam_pose(own, cid)
        Crec, Rrec = np.array(f['cameras'][cid]['p_ecef_m']), np.array(f['cameras'][cid]['R_ecef_cam'])
        cr = {'cam_p_err_m': float(np.abs(C - Crec).max()), 'cam_R_err': float(np.abs(R - Rrec).max())}
        u, v, z = (float(x[0]) for x in O.project(C, R, intr, p_i))
        cr['z_err_m'] = abs(z - c['z_m'])
        if z > 1e-3:
            cr['uv_err_px'] = max(abs(u - c['u']), abs(v - c['v']))
            cr['in_image_ok'] = ((-0.5 <= u <= intr.W - 0.5) and (-0.5 <= v <= intr.H - 0.5)) == c['in_image']
            cr['background_ok'] = ('earth' if O.ray_hits_earth(C, O.pixel_ray(R, intr, u, v)) else 'sky') == c['background']
            uu, vv, zz = O.project(C, R, intr, Pw)
            if np.all(zz > 1e-3):  # the analytic vertex box is defined when every vertex is in front of the camera
                box = [float(uu.min()), float(vv.min()), float(uu.max()), float(vv.max())]
                cr['box_err_px'] = 'oracle_defined_impl_undefined' if c['box_analytic'] is None else float(np.abs(np.array(box) - np.array(c['box_analytic'])).max())
            elif c['box_analytic'] is not None:
                cr['box_err_px'] = 'oracle_undefined_impl_defined'
            if c['in_image'] and c.get('limb_dv_px') is not None and cid in f['renders']:
                lv, _ = O.limb_crossings(C, R, intr, 0, 0, intr.W, intr.H)
                c0 = int(math.floor(u)); c1 = c0 + 1
                if 0 <= c0 and c1 < intr.W and np.isfinite(lv[c0]) and np.isfinite(lv[c1]):
                    cr['limb_dv_err_px'] = abs((v - (lv[c0] + (u - c0) * (lv[c1] - lv[c0]))) - c['limb_dv_px'])
        if cid in f['renders']:
            u0, v0 = c['roi'] if c['roi'] else (0, 0)
            if c['roi']:
                cr['roi_ok'] = c['roi'] == [int(np.clip(round(u - 511.5), 0, intr.W - 1024)), int(np.clip(round(v - 511.5), 0, intr.H - 1024))]
            A = np.array(Image.open(ROOT / c['mask_A'])); Vm = np.array(Image.open(ROOT / c['mask_V']))
            cr['png_values_ok'] = bool(set(np.unique(A).tolist()) <= {0, 255} and set(np.unique(Vm).tolist()) <= {0, 255})
            a, vm = A > 127, Vm > 127
            pa, ba, ca = mstats(a, u0, v0); pv, bv, _ = mstats(vm, u0, v0)
            cr['mask_stats_ok'] = (pa == c['amodal_px'] and ba == c['box_amodal_px'] and pv == c['visible_px'] and bv == c['box_px']
                                   and (ca is None or max(abs(ca[0] - c['centroid_amodal'][0]), abs(ca[1] - c['centroid_amodal'][1])) < 1e-9))
            cr['v_outside_a'] = int((vm & ~a).sum()); cr['amodal_px'] = pa
            pc = (Pw_rec - Crec) @ Rrec
            near = bool(np.any(pc[:, 2] < Z_NEAR))
            cv, cf = clip_near(pc, F) if near else (pc, F)
            if len(cf) == 0:
                ref, uu, vv = np.zeros(a.shape, bool), np.zeros(0), np.zeros(0)
            else:
                uu = intr.fx * cv[:, 0] / cv[:, 2] + intr.cx - u0
                vv = intr.fy * cv[:, 1] / cv[:, 2] + intr.cy - v0
                off_img = uu.max() < -1 or vv.max() < -1 or uu.min() > a.shape[1] or vv.min() > a.shape[0]
                ref = np.zeros(a.shape, bool) if off_img else O.rasterize_point_sampled(uu, vv, cf, W=a.shape[1], H=a.shape[0])
            ys, xs = np.nonzero(a ^ ref)
            non_tie = sum(1 for px, py in zip(xs, ys) if not union_boundary_near(float(px), float(py), uu, vv, cf))
            cr['raster'] = {'near_clipped': near, 'ref_px': int(ref.sum()), 'xor_px': int(len(xs)), 'xor_not_union_boundary': int(non_tie),
                            'impl_ref_px': c['gt2']['ref_px'], 'impl_tie_px': c['gt2']['tie_px'], 'impl_mismatch_px': c['gt2']['mismatch_px']}
            if c['box_analytic'] is not None and pa:
                b = c['box_analytic']
                cr['a_outside_box'] = int(np.count_nonzero((np.nonzero(a)[1] + u0 < b[0] - TIE) | (np.nonzero(a)[1] + u0 > b[2] + TIE)
                                                           | (np.nonzero(a)[0] + v0 < b[1] - TIE) | (np.nonzero(a)[0] + v0 > b[3] + TIE)))
            elif pa:
                cr['a_box_not_applicable'] = True
        row['cams'][cid] = cr
    rows.append(row)
    print(enc_id, k, f'{time.time() - t0:.0f}s', flush=True)


def cams():
    return [cr for r in rows for cr in r['cams'].values()]


def mx(key):
    vals = [cr[key] for cr in cams() if isinstance(cr.get(key), (int, float)) and not isinstance(cr.get(key), bool)]
    return max(vals) if vals else None


def mxr(key):
    vals = [r[key] for r in rows if r[key] is not None]
    return max(vals) if vals else None


R_ = [cr['raster'] for cr in cams() if 'raster' in cr]
summary = {
    'root': ROOT.as_posix(), 'encounters': ENC_P.as_posix(), 'ownship': OWN_MODE, 'frames_indexed': len(idx), 'frames': len(rows),
    'seed': SEED, 'n_random': N, 'extra': EXTRA,
    'max_own_p_err_m': mxr('own_p_err_m'), 'max_own_R_err': mxr('own_R_err'), 'max_intr_p_err_m': mxr('intr_p_err_m'),
    'max_intr_R_err': mxr('intr_R_err'), 'max_range_err_m': mxr('range_err_m'),
    'max_cam_p_err_m': mx('cam_p_err_m'), 'max_cam_R_err': mx('cam_R_err'), 'max_uv_err_px': mx('uv_err_px'), 'max_z_err_m': mx('z_err_m'),
    'max_box_err_px': mx('box_err_px'), 'box_defined_disagree': sum(1 for cr in cams() if cr.get('box_err_px') in ('oracle_defined_impl_undefined', 'oracle_undefined_impl_defined')),
    'max_limb_dv_err_px': mx('limb_dv_err_px'), 'n_limb_checked': sum(1 for cr in cams() if 'limb_dv_err_px' in cr),
    'cameras': len(cams()),
    'in_image_bad': sum(1 for cr in cams() if cr.get('in_image_ok') is False),
    'background_bad': sum(1 for cr in cams() if cr.get('background_ok') is False),
    'roi_bad': sum(1 for cr in cams() if cr.get('roi_ok') is False),
    'rendered_cams': len(R_), 'rendered_nonempty': sum(1 for cr in cams() if cr.get('amodal_px')),
    'mask_stats_bad': sum(1 for cr in cams() if cr.get('mask_stats_ok') is False),
    'png_values_bad': sum(1 for cr in cams() if cr.get('png_values_ok') is False),
    'v_outside_a_px': sum(cr.get('v_outside_a', 0) for cr in cams()),
    'a_outside_box_px': sum(cr.get('a_outside_box', 0) for cr in cams()),
    'a_box_not_applicable': sum(1 for cr in cams() if cr.get('a_box_not_applicable')),
    'raster_near_clipped': sum(1 for x in R_ if x['near_clipped']),
    'raster_xor_px': sum(x['xor_px'] for x in R_), 'raster_xor_not_union_boundary': sum(x['xor_not_union_boundary'] for x in R_),
    'raster_ref_px_vs_impl_disagree': sum(1 for x in R_ if x['ref_px'] != x['impl_ref_px']),
    'impl_mismatch_px': sum(x['impl_mismatch_px'] for x in R_),
    'time_s': time.time() - t0,
}
summary['pass'] = bool(
    (OWN_MODE != 't1' or (summary['max_own_p_err_m'] <= 1e-6 and summary['max_own_R_err'] <= 1e-9))
    and summary['max_intr_p_err_m'] <= 1e-3 and summary['max_intr_R_err'] <= 1e-6 and summary['max_range_err_m'] <= 1e-3
    and summary['max_cam_p_err_m'] <= 1e-6 and summary['max_cam_R_err'] <= 1e-9
    and (summary['max_uv_err_px'] or 0) <= 1e-3 and (summary['max_box_err_px'] or 0) <= 1e-3 and summary['box_defined_disagree'] == 0
    and (summary['max_limb_dv_err_px'] or 0) <= 0.05
    and summary['in_image_bad'] == summary['background_bad'] == summary['roi_bad'] == 0
    and summary['mask_stats_bad'] == summary['png_values_bad'] == summary['v_outside_a_px'] == summary['a_outside_box_px'] == 0
    and summary['raster_xor_not_union_boundary'] == 0)
out_p = OUT / f'audit_g3a_{ROOT.name}.json'
out_p.write_text(json.dumps({'summary': summary, 'rows': rows}, indent=1), encoding='utf-8')
print(json.dumps(summary, indent=1))
print('wrote', out_p)
