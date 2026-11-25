"""Director's independent oracle for WP-02 (no `xvs` imports).

Own geodesy (camera_model.py from the Director probe), own rig_true regeneration, own ring
poses from spec/traffic_ring_v1.json, own T1 integration, own encounter construction from
WP02 §2.5, own GLB parser, own projection, own dense-sampled limb and coverage estimator.
"""
import json, struct, sys
from pathlib import Path
import numpy as np

import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from camera_model import geodetic_to_ecef, r_ecef_ned, r_ned_body, limb_points, A, B, E2  # noqa: E402

# spec dir: $XVS_SPEC_DIR, else <repo>/docs/xvs/spec found by walking up from this file
SPEC = os.environ.get('XVS_SPEC_DIR') or next(
    (os.path.join(d, 'docs', 'xvs', 'spec') for d in [HERE] + [os.path.dirname(HERE)] * 0 + [
        os.path.abspath(os.path.join(HERE, *(['..'] * k))) for k in range(1, 8)]
     if os.path.isfile(os.path.join(d, 'docs', 'xvs', 'spec', 'rig_nominal.json'))), None)
assert SPEC, 'set XVS_SPEC_DIR to <repo>/docs/xvs/spec'
REPO_ROOT = Path(SPEC).resolve().parents[2]  # <repo>/docs/xvs/spec -> <repo>
RIG = json.load(open(SPEC + '/rig_nominal.json'))
RING = json.load(open(SPEC + '/traffic_ring_v1.json'))


class Intr:
    def __init__(self, fx, fy, cx, cy, W, H):
        self.fx, self.fy, self.cx, self.cy, self.W, self.H = fx, fy, cx, cy, W, H


RIG_INTR = Intr(512.0 / np.tan(np.radians(50.0)), 512.0 / np.tan(np.radians(50.0)), 511.5, 511.5, 1024, 1024)
_ri = RING['intrinsics']
RING_INTR = Intr(_ri['fx'], _ri['fy'], _ri['cx'], _ri['cy'], RING['image']['width'], RING['image']['height'])


def expso3(w):
    th = np.linalg.norm(w)
    if th < 1e-15:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def rig_true(seed=20260922):
    cams = {c['id']: c for c in RIG['cameras']}
    rng = np.random.default_rng(seed)
    out = {}
    for cid in ['F', 'R', 'B', 'L', 'U', 'D']:
        om = np.clip(rng.normal(0.0, 0.20, 3), -0.60, 0.60)
        de = np.clip(rng.normal(0.0, 0.005, 3), -0.015, 0.015)
        out[cid] = (np.array(cams[cid]['R_body_cam'], float) @ expso3(np.radians(om)),
                    np.array(cams[cid]['lever_arm_m'], float) + de)
    return out


RIG_TRUE = rig_true()
RING_CAMS = {c['id']: (np.array(c['R_body_cam'], float), np.array(c['lever_arm_m'], float)) for c in RING['cameras']}


def ecef_to_geodetic(p):
    x, y, z = p
    lon = np.arctan2(y, x)
    r = np.hypot(x, y)
    lat = np.arctan2(z, r * (1 - E2))
    for _ in range(20):
        n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
        h = r / np.cos(lat) - n
        lat_new = np.arctan2(z, r * (1 - E2 * n / (n + h)))
        if abs(lat_new - lat) < 1e-15:
            lat = lat_new
            break
        lat = lat_new
    n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    h = r / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), h


# ---------------- T1 ownship (WP01 §2.2) ----------------
T1 = dict(lat0=37.05, lon0=-120.85, h0=3000.0, track=315.0, gs=120.0)


def _t1_rates(s):
    lat, lon, h = s
    sl = np.sin(lat)
    M = A * (1 - E2) / (1 - E2 * sl * sl) ** 1.5
    N = A / np.sqrt(1 - E2 * sl * sl)
    vn = T1['gs'] * np.cos(np.radians(T1['track']))
    ve = T1['gs'] * np.sin(np.radians(T1['track']))
    return np.array([vn / (M + h), ve / ((N + h) * np.cos(lat)), 0.0])


def _rk4(s, dt):
    k1 = _t1_rates(s)
    k2 = _t1_rates(s + 0.5 * dt * k1)
    k3 = _t1_rates(s + 0.5 * dt * k2)
    k4 = _t1_rates(s + dt * k3)
    return s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def t1_state(t):
    s = np.array([np.radians(T1['lat0']), np.radians(T1['lon0']), T1['h0']])
    nstep = int(np.floor(t / 0.01 + 1e-9))
    for _ in range(nstep):
        s = _rk4(s, 0.01)
    rem = t - nstep * 0.01
    if rem > 1e-12:
        s = _rk4(s, rem)
    lat, lon, h = np.degrees(s[0]), np.degrees(s[1]), s[2]
    yaw = T1['track']
    pitch = 2.0 + 1.0 * np.sin(2 * np.pi * t / 15.0)
    roll = 5.0 * np.sin(2 * np.pi * t / 20.0)
    Rn = r_ecef_ned(lat, lon)
    v_ned = np.array([T1['gs'] * np.cos(np.radians(yaw)), T1['gs'] * np.sin(np.radians(yaw)), 0.0])
    return dict(t=t, lat=lat, lon=lon, h=h, p=geodetic_to_ecef(lat, lon, h), v=Rn @ v_ned, v_ned=v_ned,
                ypr=(yaw, pitch, roll), R_ecef_body=Rn @ r_ned_body(yaw, pitch, roll))


# ---------------- Encounters (WP02 §2.5) ----------------
ALPHA_TRIM = {'C1_UAS': 0.0, 'C2_SEP': 2.0, 'C3_TWIN': 2.0, 'C4_BIZJET': 2.0, 'C5_TRANSPORT': 2.0, 'C6_HELI': -3.0}


def construct(own_fn, t_cpa, hmd, side, vmd, approach_deg, gs, vs):
    o = own_fn(t_cpa)
    Po, Vo = o['p'], o['v']
    N = r_ecef_ned(o['lat'], o['lon'])
    vned_o = N.T @ Vo
    track = np.degrees(np.arctan2(vned_o[1], vned_o[0]))
    chi = np.radians(track + approach_deg)
    Vi = N @ np.array([gs * np.cos(chi), gs * np.sin(chi), -vs])
    Vrel = Vi - Vo
    Vh = N @ np.diag([1.0, 1.0, 0.0]) @ N.T @ Vrel
    d = N[:, 2]
    s = 1.0 if side == 'right' else -1.0
    hh = np.cross(d, Vh)
    hh = s * hh / np.linalg.norm(hh)
    Pi_cpa = Po + hmd * hh + vmd * (-d)
    return dict(t_cpa=t_cpa, P_cpa=Pi_cpa, V=Vi)


def intruder_pose(enc, t, cls):
    p = enc['P_cpa'] + enc['V'] * (t - enc['t_cpa'])
    lat, lon, h = ecef_to_geodetic(p)
    Ni = r_ecef_ned(lat, lon)
    vn = Ni.T @ enc['V']
    yaw = np.degrees(np.arctan2(vn[1], vn[0]))
    gamma = np.degrees(np.arctan2(-vn[2], np.hypot(vn[0], vn[1])))
    pitch = gamma + ALPHA_TRIM[cls]
    return p, Ni @ r_ned_body(yaw, pitch, 0.0), (yaw, pitch, 0.0)


# ---------------- Cameras ----------------
def cam_pose(own, cam_id):
    if cam_id in RIG_TRUE:
        Rbc, lev = RIG_TRUE[cam_id]
        intr = RIG_INTR
    else:
        Rbc, lev = RING_CAMS[cam_id]
        intr = RING_INTR
    R = own['R_ecef_body'] @ Rbc
    C = own['p'] + own['R_ecef_body'] @ lev
    return C, R, intr


def project(C, R, intr, P):
    pc = (np.atleast_2d(P) - C) @ R
    z = pc[:, 2]
    u = intr.fx * pc[:, 0] / z + intr.cx
    v = intr.fy * pc[:, 1] / z + intr.cy
    return u, v, z


def ray_hits_earth(C, d):
    S = np.array([1 / A, 1 / A, 1 / B])
    c, dd = C * S, d * S
    k = c @ c - 1.0
    cd = c @ dd
    return bool((cd * cd - (dd @ dd) * k > 0) and (cd < 0))


def pixel_ray(R, intr, u, v):
    d = R @ np.array([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, 1.0])
    return d / np.linalg.norm(d)


# ---------------- GLB ----------------
def _node_matrix(node):
    if 'matrix' in node:
        return np.array(node['matrix'], float).reshape(4, 4).T
    M = np.eye(4)
    if 'scale' in node:
        M = np.diag(list(node['scale']) + [1.0]) @ M
    if 'rotation' in node:
        x, y, z, w = node['rotation']
        Rq = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                       [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                       [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        T = np.eye(4); T[:3, :3] = Rq
        M = T @ M
    if 'translation' in node:
        T = np.eye(4); T[:3, 3] = node['translation']
        M = T @ M
    return M


def glb_vertices(path):
    b = open(path, 'rb').read()
    magic, ver, length = struct.unpack_from('<4sII', b, 0)
    assert magic == b'glTF' and ver == 2 and length == len(b)
    off = 12
    js, bin_ = None, None
    while off < len(b):
        clen, ctype = struct.unpack_from('<I4s', b, off)
        chunk = b[off + 8: off + 8 + clen]
        if ctype == b'JSON':
            js = json.loads(chunk.decode('utf-8'))
        elif ctype == b'BIN\x00':
            bin_ = chunk
        off += 8 + clen
    out = []
    info = dict(nodes_with_transform=0, n_prims=0, materials=len(js.get('materials', [])))

    def acc_positions(ai):
        acc = js['accessors'][ai]
        bv = js['bufferViews'][acc['bufferView']]
        assert acc['componentType'] == 5126 and acc['type'] == 'VEC3'
        stride = bv.get('byteStride', 12)
        base = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
        n = acc['count']
        arr = np.empty((n, 3))
        for i in range(n):
            arr[i] = struct.unpack_from('<3f', bin_, base + i * stride)
        return arr

    def visit(ni, M):
        node = js['nodes'][ni]
        if any(k in node for k in ('matrix', 'rotation', 'translation', 'scale')):
            info['nodes_with_transform'] += 1
        M2 = M @ _node_matrix(node)
        if 'mesh' in node:
            for prim in js['meshes'][node['mesh']]['primitives']:
                info['n_prims'] += 1
                P = acc_positions(prim['attributes']['POSITION'])
                Ph = np.c_[P, np.ones(len(P))] @ M2.T
                out.append(Ph[:, :3])
        for c in node.get('children', []):
            visit(c, M2)

    scene = js['scenes'][js.get('scene', 0)]
    for ni in scene['nodes']:
        visit(ni, np.eye(4))
    V = np.concatenate(out, 0)
    return V, info


# ---------------- Limb ----------------
def limb_crossings(C, R, intr, u0=0, v0=0, W=1024, H=1024, n=3_000_000):
    """Per ROI column c (full column u0+c): limb v in ROI coords (NaN if absent / not unique), slope dv/du."""
    X = limb_points(C, n)
    pc = (X - C) @ R
    z = pc[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = intr.fx * pc[:, 0] / z + intr.cx - u0
        v = intr.fy * pc[:, 1] / z + intr.cy - v0
    u2, v2, z2 = np.roll(u, -1), np.roll(v, -1), np.roll(z, -1)
    ok = (z > 0) & (z2 > 0) & (np.abs(u2 - u) < 5) & (np.abs(v2 - v) < 5)
    ua, ub, va, vb = u[ok], u2[ok], v[ok], v2[ok]
    lo, hi = np.minimum(ua, ub), np.maximum(ua, ub)
    c = np.ceil(lo)
    has = (c < hi) & (c >= 0) & (c <= W - 1)
    t = (c[has] - ua[has]) / (ub[has] - ua[has])
    p = va[has] + t * (vb[has] - va[has])
    slope = (vb[has] - va[has]) / (ub[has] - ua[has])
    inside = (p >= -0.5) & (p <= H - 0.5)
    ci = c[has][inside].astype(int)
    cnt = np.bincount(ci, minlength=W)
    val = np.full(W, np.nan); slp = np.full(W, np.nan)
    val[ci] = p[inside]; slp[ci] = slope[inside]
    val[cnt != 1] = np.nan; slp[cnt != 1] = np.nan
    return val, slp


def coverage_edges(mask01, vcol, scol, earth_below=True, H=1024):
    """WP01 §2.6 coverage estimator on columns with |dv/du| <= 1 (earth below the sky)."""
    res = []
    skipped = 0
    for col in range(mask01.shape[1]):
        vg = vcol[col]
        if not np.isfinite(vg) or abs(scol[col]) > 1.0:
            continue
        prof = mask01[:, col]
        if np.count_nonzero(np.diff((prof > 0.5).astype(int))) != 1:
            skipped += 1
            continue
        s = prof.sum()
        ve = (H - 0.5) - s if earth_below else s - 0.5
        res.append((col, ve - vg))
    return res, skipped


# ---------------- GLB mesh with faces, and a point-sampling rasterizer ----------------
def glb_mesh(path):
    """(V, F): vertices (N,3) in model coordinates and triangle indices (M,3); no node transforms allowed."""
    b = open(path, 'rb').read()
    off = 12
    js, bin_ = None, None
    while off < len(b):
        clen, ctype = struct.unpack_from('<I4s', b, off)
        chunk = b[off + 8: off + 8 + clen]
        if ctype == b'JSON':
            js = json.loads(chunk.decode('utf-8'))
        elif ctype == b'BIN\x00':
            bin_ = chunk
        off += 8 + clen

    def read_acc(ai):
        acc = js['accessors'][ai]
        bv = js['bufferViews'][acc['bufferView']]
        base = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
        ncomp = {'SCALAR': 1, 'VEC3': 3}[acc['type']]
        dt = {5126: '<f4', 5125: '<u4', 5123: '<u2', 5121: 'u1'}[acc['componentType']]
        isz = np.dtype(dt).itemsize
        stride = bv.get('byteStride', isz * ncomp)
        n = acc['count']
        raw = np.frombuffer(bin_, dtype=np.uint8, count=stride * (n - 1) + isz * ncomp, offset=base)
        idx = (np.arange(n)[:, None] * stride + np.arange(isz * ncomp)[None, :]).ravel()
        return raw[idx].view(dt).reshape(n, ncomp).astype(np.float64 if dt == '<f4' else np.int64)

    Vs, Fs, nv = [], [], 0
    for node in js['nodes']:
        assert not any(k in node for k in ('matrix', 'rotation', 'translation', 'scale'))
        if 'mesh' not in node:
            continue
        for prim in js['meshes'][node['mesh']]['primitives']:
            assert prim.get('mode', 4) == 4
            V = read_acc(prim['attributes']['POSITION'])
            F = read_acc(prim['indices']).reshape(-1, 3)
            Vs.append(V); Fs.append(F + nv); nv += len(V)
    return np.concatenate(Vs), np.concatenate(Fs)


def rasterize_point_sampled(u, v, F, W=1024, H=1024):
    """Binary mask of pixel centres (integer coordinates) inside any triangle (no AA, no culling)."""
    m = np.zeros((H, W), bool)
    for a, b_, c in F:
        xs = np.array([u[a], u[b_], u[c]]); ys = np.array([v[a], v[b_], v[c]])
        x0, x1 = int(np.ceil(xs.min())), int(np.floor(xs.max()))
        y0, y1 = int(np.ceil(ys.min())), int(np.floor(ys.max()))
        x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, W - 1), min(y1, H - 1)
        if x1 < x0 or y1 < y0:
            continue
        X, Y = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        def e(px, py, qx, qy):
            return (qx - px) * (Y - py) - (qy - py) * (X - px)
        e0 = e(xs[0], ys[0], xs[1], ys[1]); e1 = e(xs[1], ys[1], xs[2], ys[2]); e2 = e(xs[2], ys[2], xs[0], ys[0])
        inside = ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))
        area = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (ys[1] - ys[0]) * (xs[2] - xs[0])
        if area == 0:
            continue
        m[y0:y1 + 1, x0:x1 + 1] |= inside
    return m


def parse_out(argv, default_sub):
    """Split an optional `--out DIR` from argv; default output/xvs/audit/<default_sub> under the repo root."""
    argv = list(argv)
    if "--out" in argv:
        i = argv.index("--out")
        out = Path(argv[i + 1]); del argv[i:i + 2]
    else:
        out = REPO_ROOT / "output" / "xvs" / "audit" / default_sub
    out.mkdir(parents=True, exist_ok=True)
    return argv, out
