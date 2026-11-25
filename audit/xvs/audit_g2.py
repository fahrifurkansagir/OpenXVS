"""Director's independent audit of WP-02 G2 results (any backend).

Uses only oracle.py (own geodesy, rig_true, ring, T1, encounter construction, GLB parser, projection,
dense limb) and the saved mask PNGs. Recomputes every GT quantity and every GT-1..GT-4 metric and
compares them with results_<backend>.json.
usage: python audit/xvs/audit_g2.py <g2_root> <backend> [assets_root] [--out DIR]
       -> DIR/audit_g2_<backend>.json (default DIR: output/xvs/audit/wp02_g2)
(2026-09-24, Engineering Automation on the PC: implementer_summary reads gt2_v10_box, the raster-GT-2 results schema.)
"""
import hashlib, json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import oracle as O  # noqa: E402

ARGS, OUT = O.parse_out(sys.argv[1:], 'wp02_g2')
ROOT = Path(ARGS[0]); BK = ARGS[1]; ASSETS = Path(ARGS[2]) if len(ARGS) > 2 else O.REPO_ROOT / 'assets' / 'traffic'
res = json.loads((ROOT / f'results_{BK}.json').read_text(encoding='utf-8'))
out = {'backend': BK}

# ---- encounters ----
own = O.t1_state
enc_err = []
encs = {}
for e in res['eval12']['encounters']:
    cls = e['class_id']
    cc = next(c for c in res['eval12']['classes'] if c['class_id'] == cls)
    bc = next(b for b in res['eval12']['backgrounds'] if b['bg'] == e['bg'])
    E = O.construct(own, res['eval12']['t_cpa_s'], res['eval12']['hmd_m'], res['eval12']['side'], bc['vmd_m'],
                    bc['approach_deg'], cc['gs_mps'], res['eval12']['vs_mps'])
    enc_err.append(max(np.abs(E['P_cpa'] - np.array(e['p_cpa_ecef_m'])).max(), np.abs(E['V'] - np.array(e['v_ecef_mps'])).max()))
    encs[e['encounter_id']] = (E, cls)
    for inst in e['instants']:
        o = own(inst['t_s'])
        p = E['P_cpa'] + E['V'] * (inst['t_s'] - E['t_cpa'])
        enc_err.append(abs(np.linalg.norm(p - o['p']) - inst['target_range_m']))
out['encounter_max_abs_err'] = float(max(enc_err))

VERTS = {}
def verts(model_id, cls):
    key = model_id
    if key not in VERTS:
        sub = 'CALIBRATION' if model_id == 'SPHERE_R2' else cls
        V, info = O.glb_vertices(ASSETS / sub / f'{model_id}.glb')
        assert info['nodes_with_transform'] == 0
        VERTS[key] = V
    return VERTS[key]

# ---- per instant ----
dpos, drot, dcam, dproj, dbox, dsbox, dlimb, readback_pos, readback_dir, roi_bad, bg_bad = ([] for _ in range(11))
gt1, gt2, gt3a, gt3b, gt4, purity = [], [], [], [], [], []
grey_unverified, n_grey_checked = [], 0
for rec in res['records']:
    E, cls = encs[rec['encounter_id']]
    o = own(rec['t_s'])
    dpos.append(np.abs(o['p'] - np.array(rec['ownship']['p_ecef_m'])).max())
    p_i, R_i, ypr = O.intruder_pose(E, rec['t_s'], cls)
    dpos.append(np.abs(p_i - np.array(rec['intruder']['p_ecef_m'])).max())
    drot.append(np.abs(R_i - np.array(rec['intruder']['R_ecef_body'])).max())
    Vb = verts(rec['model_id'], cls)
    Vs = verts('SPHERE_R2', cls)
    Pw = p_i + Vb @ R_i.T
    Sw = p_i + Vs @ R_i.T
    for cid, c in rec['cams'].items():
        C, R, intr = O.cam_pose(o, cid)
        dcam.append(max(np.abs(C - np.array(c['p_ecef_m'])).max(), np.abs(R - np.array(c['R_ecef_cam'])).max() * 1e3))
        u, v, z = O.project(C, R, intr, p_i)
        u, v, z = float(u[0]), float(v[0]), float(z[0])
        g = c['gt']
        in_img = z > 0 and -0.5 <= u <= intr.W - 0.5 and -0.5 <= v <= intr.H - 0.5
        if z > 0:
            dproj.append(max(abs(u - g['u']), abs(v - g['v'])))
        uu, vv, zz = O.project(C, R, intr, Pw)
        box = [float(uu.min()), float(vv.min()), float(uu.max()), float(vv.max())] if np.all(zz > 1e-3) else None
        if box is not None and g['box_analytic'] is not None:
            dbox.append(max(abs(a - b) for a, b in zip(box, g['box_analytic'])))
        elif (box is None) != (g['box_analytic'] is None):
            dbox.append(float('inf'))
        su, sv, sz = O.project(C, R, intr, Sw)
        sbox = [float(su.min()), float(sv.min()), float(su.max()), float(sv.max())] if np.all(sz > 1e-3) else None
        if sbox is not None and c['gt_sphere']['box_analytic'] is not None:
            dsbox.append(max(abs(a - b) for a, b in zip(sbox, c['gt_sphere']['box_analytic'])))
        # background label and limb relation at the centre
        if z > 0:
            bg = 'earth' if O.ray_hits_earth(C, O.pixel_ray(R, intr, u, v)) else 'sky'
            if bg != g['background']:
                bg_bad.append((rec['instant_id'], cid, bg, g['background']))
        if in_img and g['limb_dv_px'] is not None:
            u0r = int(np.floor(u))
            vcol, _ = O.limb_crossings(C, R, intr, u0r, 0, W=2, H=intr.H, n=2_000_000)
            if np.all(np.isfinite(vcol)):
                vl = vcol[0] + (u - u0r) * (vcol[1] - vcol[0])
                dlimb.append(abs((v - vl) - g['limb_dv_px']))
        # ROI
        if c['kind'] == 'ring':
            u0 = int(np.clip(np.round(u - 511.5), 0, intr.W - 1024)); v0 = int(np.clip(np.round(v - 511.5), 0, intr.H - 1024))
            if [u0, v0] != c['roi']:
                roi_bad.append((rec['instant_id'], cid, [u0, v0], c['roi']))
            win = (u0 - 0.5, v0 - 0.5, u0 + 1023.5, v0 + 1023.5)
        else:
            u0 = v0 = 0
            win = (-0.5, -0.5, intr.W - 0.5, intr.H - 0.5)
        # camera readback from Cesium (M3-like)
        for ps, rr in c['renders'].items():
            rb = rr['readback']
            readback_pos.append(np.abs(np.array(rb['position']) - C).max())
            ang = math.degrees(math.acos(min(1.0, float(np.dot(np.array(rb['direction']) / np.linalg.norm(rb['direction']), R[:, 2])))))
            ang_up = math.degrees(math.acos(min(1.0, float(np.dot(np.array(rb['up']) / np.linalg.norm(rb['up']), -R[:, 1])))))
            readback_dir.append(max(ang, ang_up))
        # masks re-measured from the PNGs
        imgs = {ps: np.array(Image.open(ROOT / rr['png'])) for ps, rr in c['renders'].items()}
        for ps, im in imgs.items():
            if ps in 'AVS':
                bad = int(np.count_nonzero((im != 0) & (im != 255)))
                if bad:
                    purity.append((rec['instant_id'], cid, ps, bad))
            # R-G2-3: the PNG keeps the R channel only; the recorded rgb_sha256 hashes the full RGB readback, so
            # R = G = B for every pixel iff the hash of the grey image replicated to 3 channels equals it.
            rgb3 = np.repeat(np.asarray(im, np.uint8)[:, :, None], 3, axis=2)
            if hashlib.sha256(np.ascontiguousarray(rgb3).tobytes()).hexdigest() != c['renders'][ps]['rgb_sha256']:
                grey_unverified.append((rec['instant_id'], cid, ps))
            n_grey_checked += 1
        A = imgs['A'] > 127
        rows, cols = np.nonzero(A)
        # GT-3(a)
        exp = box is not None and (box[2] - box[0]) >= 1 and (box[3] - box[1]) >= 1 and \
            box[2] >= win[0] and box[0] <= win[2] and box[3] >= win[1] and box[1] <= win[3]
        if exp != (rows.size > 0):
            gt3a.append({'id': rec['instant_id'], 'cam': cid, 'expected': exp, 'px': int(rows.size),
                         'box_wh': None if box is None else [box[2] - box[0], box[3] - box[1]]})
        # GT-2
        if rows.size >= 10 and box is not None:
            edges = [cols.min() + u0 - 0.5, rows.min() + v0 - 0.5, cols.max() + u0 + 0.5, rows.max() + v0 + 0.5]
            errs = []
            for k in range(4):
                lo, hi = (win[0], win[2]) if k in (0, 2) else (win[1], win[3])
                if lo + 1.0 <= box[k] <= hi - 1.0:
                    errs.append(float(edges[k] - box[k]))
            if errs:
                gt2.append({'id': rec['instant_id'], 'cam': cid, 'cls': cls, 'range': rec['target_range_m'], 'px': int(rows.size),
                            'max_abs': max(abs(e) for e in errs), 'errs': errs})
        # GT-3(b)
        if 'V' in imgs:
            gt3b.append(int(np.count_nonzero((imgs['V'] > 127) & ~A)))
        # GT-1
        if 'S' in imgs:
            S = imgs['S'] > 127
            r2, c2 = np.nonzero(S)
            if r2.size >= 50 and sbox is not None and sbox[0] >= win[0] and sbox[1] >= win[1] and sbox[2] <= win[2] and sbox[3] <= win[3]:
                gt1.append({'id': rec['instant_id'], 'cam': cid, 'px': int(r2.size),
                            'du': float(c2.mean() + u0 - u), 'dv': float(r2.mean() + v0 - v)})
        # GT-4
        if 'E' in imgs:
            iu, iv = int(round(u)) - u0, int(round(v)) - v0
            val = int(imgs['E'][iv, iu])
            near = g['limb_dv_px'] is not None and abs(g['limb_dv_px']) <= 1.0
            bg = 'earth' if O.ray_hits_earth(C, O.pixel_ray(R, intr, u, v)) else 'sky'
            gt4.append({'id': rec['instant_id'], 'cam': cid, 'val': val, 'near': near, 'bg': bg,
                        'agree': ('earth' if val >= 128 else 'sky') == bg})

F = lambda f, x: float(f(x)) if len(x) else None
e1 = [math.hypot(r['du'], r['dv']) for r in gt1]
e2 = [r['max_abs'] for r in gt2]
sc4 = [r for r in gt4 if not r['near']]
out.update({
    'max_abs_err': {'positions_m': float(max(dpos)), 'R_intruder': float(max(drot)), 'camera_pose(m | mrad)': float(max(dcam)),
                    'centre_px': float(max(dproj)), 'model_box_px': float(max(dbox)), 'sphere_box_px': float(max(dsbox)),
                    'limb_dv_px': float(max(dlimb)) if dlimb else None,
                    'cesium_readback_position_m': float(max(readback_pos)), 'cesium_readback_dir_up_deg': float(max(readback_dir))},
    'n_limb_checked': len(dlimb), 'roi_mismatches': roi_bad, 'background_mismatches': bg_bad,
    'GT1': {'n': len(e1), 'median': F(np.median, e1), 'max': F(np.max, e1),
            'mean_du': F(np.mean, [r['du'] for r in gt1]), 'mean_dv': F(np.mean, [r['dv'] for r in gt1]),
            'kinds': sorted({r['cam'] for r in gt1})},
    'GT2': {'n': len(e2), 'median': F(np.median, e2), 'p95': F(lambda x: np.percentile(x, 95), e2), 'max': F(np.max, e2),
            'n_over_1px': sum(1 for x in e2 if x > 1.0),
            'worst': sorted(gt2, key=lambda r: -r['max_abs'])[:8]},
    'GT3a': {'exceptions': len(gt3a), 'records': gt3a},
    'GT3b': {'n': len(gt3b), 'violations': sum(1 for x in gt3b if x > 0)},
    'GT4': {'n': len(gt4), 'scored': len(sc4), 'agree': sum(r['agree'] for r in sc4),
            'values': sorted({r['val'] for r in gt4}), 'disagree': [r for r in sc4 if not r['agree']]},
    'purity_bad_images': purity,
    'rgb_readback_grey_checked': n_grey_checked, 'rgb_readback_not_grey': grey_unverified,
})
# compare with the implementer's summary
s = res['summary']
out['implementer_summary'] = {'gt1': [s['gt1']['n'], s['gt1']['median_px'], s['gt1']['max_px']],
                              'gt2_v10_box': [s['gt2_v10_box']['n'], s['gt2_v10_box']['median_px'], s['gt2_v10_box']['max_px'],
                                              s['gt2_v10_box']['n_over_threshold']],
                              'gt3a': s['gt3a']['exceptions'], 'gt3b': s['gt3b']['violations'],
                              'gt4': [s['gt4']['n_scored'], s['gt4']['agree']],
                              'purity_impure_images': s['purity']['impure_images']}
print(json.dumps(out, indent=1, default=str)[:12000])
(OUT / f'audit_g2_{BK}.json').write_text(json.dumps(out, indent=1, default=str), encoding='utf-8')
