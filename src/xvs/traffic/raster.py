"""Point-sampled reference rasterization of traffic meshes (WP-02 G2, spec v1.1 GT-2).

A GPU mask rendered without anti-aliasing (MSAA 1) contains exactly the pixels whose centre lies
inside at least one projected triangle. This module computes that set on the CPU from the same GLB
mesh and the analytic pinhole projection, so a rendered mask can be compared pixel by pixel with
the analytic geometry. Pixels whose centre lies within `tie_px` of a projected triangle edge are
"ties": fixed-point vertex snapping in the GPU may put them on either side (SwiftShader snaps to
1/16 px, D3D11 to 1/256 px), so they are counted separately and never treated as errors. A tie needs the
boundary of the *union* of the projected triangles within `tie_px` (independent review R-G2-2): an interior edge
shared by two covering triangles cannot make a pixel flip, so a differing pixel there is a mismatch.

Why: a box comparison (spec v1.0 GT-2) fails for correct renders when thin parts (wings seen
edge-on, rotor blades) are thinner than a pixel and cover no pixel centre.

A mesh that crosses the camera plane (some vertices in front, some behind) is clipped to
z >= Z_NEAR_M in camera coordinates before projection, like the GPU clips at the frustum near plane.
Clipping in camera coordinates is exact: edges stay straight lines there.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

TIE_PX = 1.0 / 16.0
Z_NEAR_M = 0.5  # near plane of the rig and ring-ROI frusta (WP02 §2.4, xvs_page.js)


def glb_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (N, 3) float64 in model (BODY FRD) coordinates and triangles (M, 3) int64.

    Node transforms are not allowed (BODY vertices are used as-is, WP02 §2.2).
    """
    import pygltflib

    g = pygltflib.GLTF2().load(str(path))
    blob = g.binary_blob()

    def read(acc_index: int) -> np.ndarray:
        acc = g.accessors[acc_index]
        bv = g.bufferViews[acc.bufferView]
        ncomp = {"SCALAR": 1, "VEC3": 3}[acc.type]
        dt = {5126: "<f4", 5125: "<u4", 5123: "<u2", 5121: "u1"}[acc.componentType]
        isz = np.dtype(dt).itemsize
        stride = bv.byteStride or isz * ncomp
        start = (bv.byteOffset or 0) + (acc.byteOffset or 0)
        rows = [np.frombuffer(blob, dtype=dt, count=ncomp, offset=start + i * stride) for i in range(acc.count)]
        return np.array(rows).reshape(acc.count, ncomp)

    verts, faces, base = [], [], 0
    for node in g.nodes:
        identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        if (node.matrix not in (None, identity)) or node.translation or node.rotation or node.scale:
            raise ValueError(f"{Path(path).name}: node transforms are not supported")
        if node.mesh is None:
            continue
        for prim in g.meshes[node.mesh].primitives:
            if prim.mode not in (None, 4):
                raise ValueError(f"{Path(path).name}: only TRIANGLES primitives are supported")
            v = read(prim.attributes.POSITION).astype(np.float64)
            f = read(prim.indices).astype(np.int64).reshape(-1, 3)
            verts.append(v)
            faces.append(f + base)
            base += len(v)
    return np.concatenate(verts), np.concatenate(faces)


def rasterize(u: np.ndarray, v: np.ndarray, faces: np.ndarray, width: int = 1024, height: int = 1024) -> np.ndarray:
    """Pixels (integer centres of a width x height window) inside any projected triangle.

    u, v are window pixel coordinates of the vertices (OpenCV convention: pixel centres at integers).
    Vectorised over triangles, inside-or-on-edge rule; ties are handled by `compare_with_mask`.
    """
    mask = np.zeros((height, width), dtype=bool)
    if len(faces) == 0:
        return mask
    ax, ay = u[faces[:, 0]], v[faces[:, 0]]
    bx, by = u[faces[:, 1]], v[faces[:, 1]]
    cx, cy = u[faces[:, 2]], v[faces[:, 2]]
    area2 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    keep = area2 != 0.0
    ax, ay, bx, by, cx, cy, area2 = ax[keep], ay[keep], bx[keep], by[keep], cx[keep], cy[keep], area2[keep]
    sgn = np.sign(area2)
    x0 = np.maximum(np.ceil(np.minimum(np.minimum(ax, bx), cx)), 0).astype(int)
    x1 = np.minimum(np.floor(np.maximum(np.maximum(ax, bx), cx)), width - 1).astype(int)
    y0 = np.maximum(np.ceil(np.minimum(np.minimum(ay, by), cy)), 0).astype(int)
    y1 = np.minimum(np.floor(np.maximum(np.maximum(ay, by), cy)), height - 1).astype(int)
    live = (x1 >= x0) & (y1 >= y0)
    if not np.any(live):
        return mask
    gx0, gx1, gy0, gy1 = x0[live].min(), x1[live].max(), y0[live].min(), y1[live].max()
    ys, xs = np.mgrid[gy0:gy1 + 1, gx0:gx1 + 1]
    px = xs.ravel().astype(np.float64)[:, None]
    py = ys.ravel().astype(np.float64)[:, None]
    idx = np.nonzero(live)[0]
    hit = np.zeros(px.shape[0], dtype=bool)
    chunk = max(1, int(4_000_000 // max(1, px.shape[0])))
    for s in range(0, len(idx), chunk):
        k = idx[s:s + chunk]
        e0 = ((bx[k] - ax[k]) * (py - ay[k]) - (by[k] - ay[k]) * (px - ax[k])) * sgn[k]
        e1 = ((cx[k] - bx[k]) * (py - by[k]) - (cy[k] - by[k]) * (px - bx[k])) * sgn[k]
        e2 = ((ax[k] - cx[k]) * (py - cy[k]) - (ay[k] - cy[k]) * (px - cx[k])) * sgn[k]
        hit |= np.any((e0 >= 0) & (e1 >= 0) & (e2 >= 0), axis=1)
    mask[gy0:gy1 + 1, gx0:gx1 + 1] = hit.reshape(ys.shape)
    return mask


def covered(qx: np.ndarray, qy: np.ndarray, u: np.ndarray, v: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Point sampling at arbitrary points: True where the point lies inside (or on an edge of) any triangle."""
    qx = np.asarray(qx, dtype=np.float64).ravel()[:, None]
    qy = np.asarray(qy, dtype=np.float64).ravel()[:, None]
    ax, ay = u[faces[:, 0]], v[faces[:, 0]]
    bx, by = u[faces[:, 1]], v[faces[:, 1]]
    cx, cy = u[faces[:, 2]], v[faces[:, 2]]
    area2 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    keep = area2 != 0.0
    ax, ay, bx, by, cx, cy, sgn = ax[keep], ay[keep], bx[keep], by[keep], cx[keep], cy[keep], np.sign(area2[keep])
    hit = np.zeros(qx.shape[0], dtype=bool)
    chunk = max(1, int(2_000_000 // max(1, len(ax))))
    for s in range(0, qx.shape[0], chunk):
        px, py = qx[s:s + chunk], qy[s:s + chunk]
        e0 = ((bx - ax) * (py - ay) - (by - ay) * (px - ax)) * sgn
        e1 = ((cx - bx) * (py - by) - (cy - by) * (px - bx)) * sgn
        e2 = ((ax - cx) * (py - cy) - (ay - cy) * (px - cx)) * sgn
        hit[s:s + chunk] = np.any((e0 >= 0) & (e1 >= 0) & (e2 >= 0), axis=1)
    return hit


def near_union_boundary(px: np.ndarray, py: np.ndarray, u: np.ndarray, v: np.ndarray, faces: np.ndarray,
                        r: float = TIE_PX, n: int = 32) -> np.ndarray:
    """True where the boundary of the union of the projected triangles passes within r of the point: the
    coverage is not constant over the centre and n points on each of the circles of radius r and r/2."""
    ang = 2.0 * np.pi * np.arange(n) / n
    ox = np.concatenate([[0.0], r * np.cos(ang), 0.5 * r * np.cos(ang)])
    oy = np.concatenate([[0.0], r * np.sin(ang), 0.5 * r * np.sin(ang)])
    out = np.zeros(len(px), dtype=bool)
    for i, (x, y) in enumerate(zip(px, py)):
        c = covered(x + ox, y + oy, u, v, faces)
        out[i] = bool(c.any() and not c.all())
    return out


def edge_distance(px: np.ndarray, py: np.ndarray, u: np.ndarray, v: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Distance (px) from each point to the nearest projected triangle edge (segment)."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    ax, ay, bx, by = u[e[:, 0]], v[e[:, 0]], u[e[:, 1]], v[e[:, 1]]
    dx, dy = bx - ax, by - ay
    ll = dx * dx + dy * dy
    out = np.empty(len(px))
    for i, (x, y) in enumerate(zip(px, py)):
        t = np.clip(((x - ax) * dx + (y - ay) * dy) / np.where(ll == 0.0, 1.0, ll), 0.0, 1.0)
        out[i] = float(np.min(np.hypot(x - (ax + t * dx), y - (ay + t * dy))))
    return out


def clip_near(p_cam: np.ndarray, faces: np.ndarray, z_near: float = Z_NEAR_M) -> tuple[np.ndarray, np.ndarray]:
    """Clip triangles to the half-space z >= z_near (camera coordinates, Sutherland-Hodgman).

    Returns (vertices (N', 3), triangles (M', 3)). A triangle with one vertex in front becomes one
    triangle, one with two vertices in front becomes a quad split into two triangles (fan). Triangles
    entirely in front are kept with their original vertices; triangles entirely behind are dropped.
    """
    inside = p_cam[:, 2] >= z_near
    n_in = inside[faces].sum(axis=1)
    keep = faces[n_in == 3]
    verts = [p_cam]
    tris = [keep]
    base = len(p_cam)
    for f in faces[(n_in == 1) | (n_in == 2)]:
        poly = []
        for k in range(3):
            a, b = p_cam[f[k]], p_cam[f[(k + 1) % 3]]
            ia, ib = a[2] >= z_near, b[2] >= z_near
            if ia:
                poly.append(a)
            if ia != ib:
                t = (z_near - a[2]) / (b[2] - a[2])
                q = a + t * (b - a)
                q[2] = z_near
                poly.append(q)
        pts = np.array(poly)
        idx = base + np.arange(len(pts))
        verts.append(pts)
        tris.append(np.array([[idx[0], idx[j], idx[j + 1]] for j in range(1, len(pts) - 1)], dtype=np.int64))
        base += len(pts)
    return np.concatenate(verts), np.concatenate(tris).astype(np.int64).reshape(-1, 3)


def compare_with_mask(gpu_mask: np.ndarray, p_cam: np.ndarray, faces: np.ndarray, fx: float, fy: float,
                      cx: float, cy: float, u0: int = 0, v0: int = 0, tie_px: float = TIE_PX,
                      z_near: float = Z_NEAR_M) -> dict:
    """Compare a rendered binary mask with the reference rasterization.

    p_cam: vertices in camera coordinates (x right, y down, z forward, m); fx, fy, cx, cy: full-image
    intrinsics; (u0, v0): window offset (ROI top-left, 0 for rig cameras). Window pixel coordinates are
    u = fx·x/z + cx − u0, v = fy·y/z + cy − v0. A mesh with vertices on both sides of z = z_near is
    clipped there first (`near_clipped`); a mesh entirely behind it has an empty reference.
    Returns counts of reference pixels, differing pixels, ties (the boundary of the union of the projected
    triangles within tie_px, `near_union_boundary`) and non-tie mismatches, plus the worst distance of a differing
    pixel to any projected triangle edge (reported only).
    """
    h, w = gpu_mask.shape
    z = p_cam[:, 2]
    clipped = bool(np.any(z < z_near) and np.any(z >= z_near))
    if np.all(z < z_near):
        u = v = None
        ref = np.zeros_like(gpu_mask)
    else:
        if clipped:
            p_cam, faces = clip_near(p_cam, faces, z_near)
        u = fx * p_cam[:, 0] / p_cam[:, 2] + cx - u0
        v = fy * p_cam[:, 1] / p_cam[:, 2] + cy - v0
        ref = rasterize(u, v, faces, w, h)
    diff = gpu_mask ^ ref
    ys, xs = np.nonzero(diff)
    if len(xs) and u is not None:
        dist = edge_distance(xs.astype(float), ys.astype(float), u, v, faces)
        tie = near_union_boundary(xs.astype(float), ys.astype(float), u, v, faces, tie_px)
    else:
        dist = np.full(len(xs), np.inf)
        tie = np.zeros(len(xs), dtype=bool)
    ties = int(np.count_nonzero(tie))
    return {
        "computable": True,
        "tie_rule": "union boundary within tie_px",
        "near_clipped": clipped,
        "ref_px": int(ref.sum()),
        "diff_px": int(len(xs)),
        "gpu_only_px": int(np.count_nonzero(gpu_mask & ~ref)),
        "ref_only_px": int(np.count_nonzero(ref & ~gpu_mask)),
        "tie_px": ties,
        "mismatch_px": int(len(xs) - ties),
        "max_diff_edge_distance_px": (float(dist.max()) if len(xs) else None),
        "ref_box_px": (None if not ref.any() else
                       [int(np.nonzero(ref.any(0))[0].min()), int(np.nonzero(ref.any(1))[0].min()),
                        int(np.nonzero(ref.any(0))[0].max()), int(np.nonzero(ref.any(1))[0].max())]),
    }
