"""All A/S masks of a G2 run vs the oracle's point-sampled rasterization; XOR pixels and their edge distances.

usage: python audit/xvs/raster_xor.py <g2_root> <backend> [assets_root] [--out DIR]
       -> DIR/raster_xor_<backend>.json (default DIR: output/xvs/audit/wp02_g2)

Meshes with vertices on both sides of the camera plane (the implementer clips them at z = 0.5 m) get an
independent check here: each triangle is rejected if all three vertices lie outside one plane of the
window's viewing frustum (z >= 0.5, and the four side planes through the window's outer pixel edges);
any triangle that is not rejected is ray cast against every pixel centre of the window.
(2026-09-24, Engineering Automation on the PC: paths made cwd-independent; straddling meshes no longer skipped.)
"""
import json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
import oracle as O
ARGS, OUT = O.parse_out(sys.argv[1:], 'wp02_g2')
root = ARGS[0]; bk = ARGS[1]; ASSETS = ARGS[2] if len(ARGS) > 2 else str(O.REPO_ROOT / 'assets' / 'traffic')
Z_NEAR = 0.5
d = json.load(open(f'{root}/results_{bk}.json', encoding='utf-8'))
MESH = {}
def mesh(cls, mid):
    if mid not in MESH:
        sub = 'CALIBRATION' if mid == 'SPHERE_R2' else cls
        MESH[mid] = O.glb_mesh(f'{ASSETS}/{sub}/{mid}.glb')
    return MESH[mid]
def seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / np.where(L == 0, 1, L), 0, 1)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))
def union_boundary_near(px, py, uu, vv, F, r=1.0 / 16.0, n=32):
    """R-G2-2: is the boundary of the union of the projected triangles within r of (px, py)? The coverage over the
    centre and 2 x n points on circles of radius r and r/2 is mixed (the oracle's own point-in-triangle test)."""
    ang = 2 * np.pi * np.arange(n) / n
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


def straddling_reference(pc, F, intr, u0, v0):
    """Pixels of the 1024x1024 window whose ray hits a triangle at z >= Z_NEAR; plus counts."""
    # window pixel-edge bounds in normalised image coordinates x/z, y/z
    xl, xr = (u0 - 0.5 - intr.cx) / intr.fx, (u0 + 1023.5 - intr.cx) / intr.fx
    yt, yb = (v0 - 0.5 - intr.cy) / intr.fy, (v0 + 1023.5 - intr.cy) / intr.fy
    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    planes = [z - Z_NEAR, x - xl * z, xr * z - x, y - yt * z, yb * z - y]   # >= 0 inside
    out = np.zeros(len(F), bool)
    for s in planes:
        out |= np.all(s[F] < 0, axis=1)
    live = F[~out]
    m = np.zeros((1024, 1024), bool)
    if len(live):
        jj, ii = np.mgrid[0:1024, 0:1024]
        dr = np.stack([(ii.ravel() + u0 - intr.cx) / intr.fx, (jj.ravel() + v0 - intr.cy) / intr.fy, np.ones(ii.size)], 1)
        hit = np.zeros(ii.size, bool)
        for a, b, c in live:
            A, B, Cc = pc[a], pc[b], pc[c]
            e1, e2 = B - A, Cc - A
            p = np.cross(dr, e2); det = p @ e1
            ok = np.abs(det) > 1e-15
            inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
            tv = -A; q = np.cross(tv, e1)
            uu = (p @ tv) * inv; vv = (dr * q).sum(1) * inv; s = (q @ e2) * inv
            hit |= ok & (uu >= 0) & (vv >= 0) & (uu + vv <= 1) & (s >= Z_NEAR)
        m = hit.reshape(1024, 1024)
    return m, int(out.sum()), int(len(live))
rows = []
t0 = time.time()
for rec in d['records']:
    R_i = np.array(rec['intruder']['R_ecef_body']); p_i = np.array(rec['intruder']['p_ecef_m'])
    for cid, c in rec['cams'].items():
        C = np.array(c['p_ecef_m']); R = np.array(c['R_ecef_cam'])
        intr = O.RING_INTR if c['kind'] == 'ring' else O.RIG_INTR
        u0, v0 = c['roi'] if c['roi'] else (0, 0)
        for ps, mid in (('A', rec['model_id']), ('S', 'SPHERE_R2')):
            if ps not in c['renders']:
                continue
            V, F = mesh(rec['class_id'], mid)
            P = p_i + V @ R_i.T
            u, v, z = O.project(C, R, intr, P)
            g = np.array(Image.open(f"{root}/{c['renders'][ps]['png']}")) > 127
            row = {'id': rec['instant_id'], 'cam': cid, 'pass': ps, 'gpu': int(g.sum())}
            if np.any(z <= Z_NEAR) and np.any(z > Z_NEAR):
                ref, n_rej, n_cast = straddling_reference((P - C) @ R, F, intr, u0, v0)
                x = g ^ ref
                row.update({'straddling': True, 'tri_rejected': n_rej, 'tri_ray_cast': n_cast, 'ref': int(ref.sum()),
                            'xor': int(x.sum()), 'xor_gpu_only': int((g & ~ref).sum()), 'xor_ref_only': int((ref & ~g).sum()),
                            'max_edge_dist': None})
                rows.append(row)
                continue
            if np.all(z <= Z_NEAR):
                ref = np.zeros((1024, 1024), bool)
            else:
                uu, vv = u - u0, v - v0
                if uu.max() < -1 or vv.max() < -1 or uu.min() > 1024 or vv.min() > 1024:
                    ref = np.zeros((1024, 1024), bool)
                else:
                    ref = O.rasterize_point_sampled(uu, vv, F)
            x = g ^ ref
            ys, xs = np.nonzero(x)
            dmin, n_not_boundary = [], 0
            if len(xs):
                uu, vv = u - u0, v - v0
                E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
                for px, py in zip(xs, ys):
                    dmin.append(float(seg_dist(px, py, uu[E[:, 0]], vv[E[:, 0]], uu[E[:, 1]], vv[E[:, 1]]).min()))
                    n_not_boundary += int(not union_boundary_near(float(px), float(py), uu, vv, F))
            row.update({'straddling': False, 'ref': int(ref.sum()),
                        'xor': int(x.sum()), 'xor_gpu_only': int((g & ~ref).sum()), 'xor_ref_only': int((ref & ~g).sum()),
                        'max_edge_dist': max(dmin) if dmin else None, 'xor_not_near_union_boundary': n_not_boundary})
            rows.append(row)
xs = [r for r in rows if r['xor']]
summary = {'backend': bk, 'images': len(rows), 'straddling': sum(r['straddling'] for r in rows),
           'straddling_xor_px': sum(r['xor'] for r in rows if r['straddling']),
           'images_with_xor': len(xs), 'total_xor_px': sum(r['xor'] for r in rows),
           'max_xor_px': max((r['xor'] for r in rows), default=0),
           'max_edge_dist_of_xor_px': max((r['max_edge_dist'] for r in xs if r['max_edge_dist'] is not None), default=None),
           'xor_px_not_near_union_boundary': sum(r.get('xor_not_near_union_boundary', 0) for r in rows),
           'straddling_nonempty': sum(1 for r in rows if r['straddling'] and (r['gpu'] or r['ref'])),
           'straddling_triangles_ray_cast': sum(r.get('tri_ray_cast', 0) for r in rows if r['straddling']),
           'time_s': time.time() - t0}
print(json.dumps(summary))
for r in sorted(xs, key=lambda r: -(r['max_edge_dist'] or 1e9))[:8]:
    print(r)
for r in rows:
    if r['straddling']:
        print('straddling:', r)
(OUT / f'raster_xor_{bk}.json').write_text(json.dumps({'summary': summary, 'rows': rows}, indent=1), encoding='utf-8')
