"""Curved-Earth WGS-84 hierarchical Min-Max Mipmap ray-DEM intersection (WP-01b, DR-03).

Strictly obeys CONVENTIONS.md v1.0:
  - All inputs/outputs in ECEF or local NED (x=North, y=East, z=Down); zero internal ENU.
  - Curved-Earth vertical drop relative to the local NED tangent plane at grid origin:
      Delta_z_curv(N, E) = -(N^2 + E^2) / (2 * R_E),  R_E = 6_371_008.8 m.
  - Hierarchical ascend/descend traversal skips empty coarse cells via `levels_max[lvl]`,
    descending to L0 leaf cells and solving the analytical quadratic bilinear-patch equation
    to achieve < 0.1 m vertical accuracy vs dense reference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from xvs.geo.wgs84 import geodetic_to_ecef, r_ecef_ned
from xvs.v2.wp01b.mipmap import MinMaxMipmapPyramid, sample_height_bilinear_m

EARTH_MEAN_RADIUS_M: float = 6_371_008.8


@dataclass(frozen=True)
class RayDemHit:
    """Result of curved-Earth ray-DEM intersection in local grid NED coordinates."""

    hit: bool
    t_m: float
    north_m: float
    east_m: float
    down_m: float
    h_wgs84_m: float
    curv_drop_m: float
    steps: int
    leaf_evals: int


def earth_curvature_drop_m(north_m: float, east_m: float, radius_m: float = EARTH_MEAN_RADIUS_M) -> float:
    """Curved-Earth elevation drop `Delta_z_curv = -(N^2 + E^2) / (2 * R_E)` (negative metres)."""
    return -((north_m * north_m + east_m * east_m) / (2.0 * radius_m))


def _solve_quadratic_bilinear_leaf(
    h00: float,
    h10: float,
    h01: float,
    h11: float,
    u0: float,
    du: float,
    v0: float,
    dv: float,
    z0: float,
    dz: float,
    t_enter: float,
    t_exit: float,
    curv0: float,
    dcurv: float,
) -> float | None:
    """Analytical quadratic intersection of ray `z_ray(s) = z0 + dz*s` with curved bilinear patch.

    Parameterized by local `s = t - t_enter in [0, dt]`:
      u(s) = u0 + du*s,  v(s) = v0 + dv*s
      h(s) = h00*(1-u)*(1-v) + h10*u*(1-v) + h01*(1-u)*v + h11*u*v
      f(s) = (z0 + dz*s) - (h(s) + curv0 + dcurv*s) = a*s^2 + b*s + c = 0.
    """
    dt = max(0.0, t_exit - t_enter)
    k00 = h00
    k10 = h10 - h00
    k01 = h01 - h00
    k11 = h00 - h10 - h01 + h11

    # h(s) = (k00 + k10*u0 + k01*v0 + k11*u0*v0)
    #      + (k10*du + k01*dv + k11*(u0*dv + v0*du)) * s
    #      + (k11 * du * dv) * s^2
    h_c = k00 + k10 * u0 + k01 * v0 + k11 * u0 * v0
    h_b = k10 * du + k01 * dv + k11 * (u0 * dv + v0 * du)
    h_a = k11 * du * dv

    c = z0 - (h_c + curv0)
    if c <= 0.0:
        return t_enter

    b = dz - (h_b + dcurv)
    a = -h_a

    roots: list[float] = []
    if abs(a) < 1e-15:
        if abs(b) > 1e-15:
            s = -c / b
            if -1e-9 <= s <= dt + 1e-9:
                roots.append(max(0.0, min(dt, s)))
    else:
        disc = b * b - 4.0 * a * c
        if disc >= 0.0:
            sq = math.sqrt(max(0.0, disc))
            for s in ((-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)):
                if -1e-9 <= s <= dt + 1e-9:
                    roots.append(max(0.0, min(dt, s)))

    if not roots:
        # Guard against tiny numerical miss when endpoint is strictly below patch
        f_end = a * dt * dt + b * dt + c
        if f_end <= 0.0:
            return t_exit
        return None
    return t_enter + min(roots)


def _clip_ray_2d(
    n0: float,
    e0: float,
    dn: float,
    de: float,
    n_min: float,
    n_max: float,
    e_min: float,
    e_max: float,
    t_min: float,
    t_max: float,
) -> tuple[float, float] | None:
    """Slab clip ray `(n0 + t*dn, e0 + t*de)` against `[n_min, n_max] x [e_min, e_max]`."""
    t0 = t_min
    t1 = t_max

    for p0, dp, lo, hi in ((n0, dn, n_min, n_max), (e0, de, e_min, e_max)):
        if abs(dp) < 1e-15:
            if p0 < lo or p0 > hi:
                return None
        else:
            inv_d = 1.0 / dp
            ta = (lo - p0) * inv_d
            tb = (hi - p0) * inv_d
            if ta > tb:
                ta, tb = tb, ta
            t0 = max(t0, ta)
            t1 = min(t1, tb)
            if t0 > t1:
                return None
    return t0, t1


def _min_r2_in_box(n_lo: float, n_hi: float, e_lo: float, e_hi: float) -> float:
    """Minimum squared horizontal distance `N^2 + E^2` inside an axis-aligned NED rectangle."""
    n_closest = 0.0 if (n_lo <= 0.0 <= n_hi) else min(abs(n_lo), abs(n_hi))
    e_closest = 0.0 if (e_lo <= 0.0 <= e_hi) else min(abs(e_lo), abs(e_hi))
    return n_closest * n_closest + e_closest * e_closest


def intersect_ray_dem_ned(
    pyramid: MinMaxMipmapPyramid,
    origin_ned_m: np.ndarray | tuple[float, float, float],
    dir_ned: np.ndarray | tuple[float, float, float],
    *,
    t_min_m: float = 0.0,
    t_max_m: float = 60_000.0,
    max_steps: int = 8192,
    earth_radius_m: float = EARTH_MEAN_RADIUS_M,
) -> RayDemHit:
    """Hierarchical ascend/descend Min-Max Mipmap curved-Earth ray-DEM intersection in grid NED.

    Grid NED convention (CONVENTIONS.md §3):
      - `origin_ned_m = [N, E, D]` where `D = -z_tangent_plane` (Down is positive).
      - Curved surface tangent-plane elevation at `(N, E)` is `h_DEM(N, E) - (N^2 + E^2)/(2 R_E)`.
    """
    o = np.asarray(origin_ned_m, dtype=np.float64)
    d_raw = np.asarray(dir_ned, dtype=np.float64)
    d_norm = float(np.linalg.norm(d_raw))
    if d_norm <= 0.0 or not math.isfinite(d_norm):
        raise ValueError("Ray direction must be finite and non-zero")
    d = d_raw / d_norm

    n0, e0, d0 = float(o[0]), float(o[1]), float(o[2])
    dn, de, dd = float(d[0]), float(d[1]), float(d[2])
    # Ray tangent-plane elevation z_ray(t) = -D(t) = -d0 - dd * t
    z0_ray = -d0
    dz_ray = -dd

    clipped = _clip_ray_2d(
        n0,
        e0,
        dn,
        de,
        pyramid.north_min_m,
        pyramid.north_max_m,
        pyramid.east_min_m,
        pyramid.east_max_m,
        t_min_m,
        t_max_m,
    )
    if clipped is None:
        return RayDemHit(False, math.inf, math.nan, math.nan, math.nan, math.nan, math.nan, 0, 0)

    t_start, t_end = clipped
    grid_n = pyramid.grid_size
    n_cells = grid_n - 1  # Bilinear patches span [0..grid_n-2]
    max_lvl = pyramid.num_levels - 1
    cell_n_m = pyramid.extent_north_m / float(n_cells)
    cell_e_m = pyramid.extent_east_m / float(n_cells)
    g0 = pyramid.levels_min[0]

    # Map local NED to continuous sample index (row in [0, grid_n-1], col in [0, grid_n-1])
    def _ned_to_rc(nm: float, em: float) -> tuple[float, float]:
        r_f = (pyramid.north_max_m - nm) / cell_n_m
        c_f = (em - pyramid.east_min_m) / cell_e_m
        return r_f, c_f

    # Start near top of pyramid for rapid empty-space skipping
    lvl = max(0, max_lvl - 1)
    t = t_start
    steps = 0
    leaf_evals = 0
    eps_t = 1e-6

    while t <= t_end and steps < max_steps:
        steps += 1
        n_cur = n0 + t * dn
        e_cur = e0 + t * de
        r_f, c_f = _ned_to_rc(n_cur, e_cur)

        # Nudge slightly inward on exact boundary entry
        if r_f < -1e-6 or r_f > float(n_cells) + 1e-6 or c_f < -1e-6 or c_f > float(n_cells) + 1e-6:
            break

        cell_span = 1 << lvl
        r_idx = max(0, min(grid_n - 1, int(math.floor(max(0.0, min(float(grid_n - 1), r_f))))))
        c_idx = max(0, min(grid_n - 1, int(math.floor(max(0.0, min(float(grid_n - 1), c_f))))))
        mr = r_idx >> lvl
        mc = c_idx >> lvl

        # Metric extent covered by this mip cell's bilinear patch envelope:
        # texels [mr*span .. min((mr+1)*span, grid_n-1)]
        r_lo_i = mr * cell_span
        r_hi_i = min(r_lo_i + cell_span, n_cells)
        c_lo_i = mc * cell_span
        c_hi_i = min(c_lo_i + cell_span, n_cells)

        # If on last texel row/col, clamp patch to last valid cell [n_cells-1, n_cells]
        if r_lo_i >= n_cells:
            r_lo_i = n_cells - 1
            r_hi_i = n_cells
        if c_lo_i >= n_cells:
            c_lo_i = n_cells - 1
            c_hi_i = n_cells

        box_n_hi = pyramid.north_max_m - float(r_lo_i) * cell_n_m
        box_n_lo = pyramid.north_max_m - float(r_hi_i) * cell_n_m
        box_e_lo = pyramid.east_min_m + float(c_lo_i) * cell_e_m
        box_e_hi = pyramid.east_min_m + float(c_hi_i) * cell_e_m

        # Compute exit parameter t_exit from this cell's NED box
        t_exit = t_end
        if dn > 1e-15:
            t_exit = min(t_exit, (box_n_hi - n0) / dn)
        elif dn < -1e-15:
            t_exit = min(t_exit, (box_n_lo - n0) / dn)
        if de > 1e-15:
            t_exit = min(t_exit, (box_e_hi - e0) / de)
        elif de < -1e-15:
            t_exit = min(t_exit, (box_e_lo - e0) / de)
        t_exit = max(t, min(t_exit, t_end))

        # Conservative upper bound on terrain height in this mip cell (including boundary corner +1)
        if lvl == 0:
            q_max = int(
                max(
                    g0[r_lo_i, c_lo_i],
                    g0[r_lo_i, c_hi_i],
                    g0[r_hi_i, c_lo_i],
                    g0[r_hi_i, c_hi_i],
                )
            )
        else:
            # Include neighbor mip cell along +row/+col because bilinear patches share right/bottom edges
            max_grid = pyramid.levels_max[lvl]
            mr_n = min(mr + 1, int(max_grid.shape[0]) - 1)
            mc_n = min(mc + 1, int(max_grid.shape[1]) - 1)
            q_max = int(
                max(
                    max_grid[mr, mc],
                    max_grid[mr, mc_n],
                    max_grid[mr_n, mc],
                    max_grid[mr_n, mc_n],
                )
            )

        h_cell_max = pyramid.h_min_m + float(q_max) * pyramid.h_step_m
        curv_cell_max = -_min_r2_in_box(box_n_lo, box_n_hi, box_e_lo, box_e_hi) / (2.0 * earth_radius_m)
        z_cell_max = h_cell_max + curv_cell_max

        z_ray_enter = z0_ray + dz_ray * t
        z_ray_exit = z0_ray + dz_ray * t_exit
        z_ray_min = min(z_ray_enter, z_ray_exit)

        if z_ray_min > z_cell_max + 1e-9:
            # Ray is strictly above this entire cell -> skip to t_exit and ascend if possible
            t_next = t_exit + eps_t
            if t_next <= t:
                t_next = t + max(1e-3, 1e-6 * max(1.0, t))
            t = t_next
            if lvl < max_lvl - 1:
                lvl += 1
            continue

        if lvl > 0:
            # Potential intersection -> descend into finer mip level at current t
            lvl -= 1
            continue

        # Leaf level (L0): analytical quadratic bilinear-patch intersection
        leaf_evals += 1
        h00 = pyramid.h_min_m + float(g0[r_lo_i, c_lo_i]) * pyramid.h_step_m
        h10 = pyramid.h_min_m + float(g0[r_lo_i, c_hi_i]) * pyramid.h_step_m
        h01 = pyramid.h_min_m + float(g0[r_hi_i, c_lo_i]) * pyramid.h_step_m
        h11 = pyramid.h_min_m + float(g0[r_hi_i, c_hi_i]) * pyramid.h_step_m

        n_ent = n0 + t * dn
        e_ent = e0 + t * de
        n_ex = n0 + t_exit * dn
        e_ex = e0 + t_exit * de

        u0 = (e_ent - box_e_lo) / cell_e_m
        v0 = (box_n_hi - n_ent) / cell_n_m
        du = de / cell_e_m
        dv = -dn / cell_n_m

        curv_ent = earth_curvature_drop_m(n_ent, e_ent, earth_radius_m)
        curv_ex = earth_curvature_drop_m(n_ex, e_ex, earth_radius_m)
        dt_leaf = t_exit - t
        dcurv = (curv_ex - curv_ent) / dt_leaf if dt_leaf > 1e-12 else 0.0

        t_hit = _solve_quadratic_bilinear_leaf(
            h00,
            h10,
            h01,
            h11,
            u0,
            du,
            v0,
            dv,
            z_ray_enter,
            dz_ray,
            t,
            t_exit,
            curv_ent,
            dcurv,
        )
        if t_hit is not None:
            nh = n0 + t_hit * dn
            eh = e0 + t_hit * de
            dh = d0 + t_hit * dd
            curv_h = earth_curvature_drop_m(nh, eh, earth_radius_m)
            h_surf = sample_height_bilinear_m(pyramid, nh, eh)
            return RayDemHit(
                hit=True,
                t_m=float(t_hit),
                north_m=float(nh),
                east_m=float(eh),
                down_m=float(dh),
                h_wgs84_m=float(h_surf),
                curv_drop_m=float(curv_h),
                steps=steps,
                leaf_evals=leaf_evals,
            )

        # No hit in this leaf cell -> step into next cell
        t_next = t_exit + eps_t
        if t_next <= t:
            t_next = t + max(1e-3, 1e-6 * max(1.0, t))
        t = t_next
        if ((r_lo_i + 1) & 1) == 0 or ((c_lo_i + 1) & 1) == 0:
            lvl = min(lvl + 1, max_lvl - 1)

    return RayDemHit(False, math.inf, math.nan, math.nan, math.nan, math.nan, math.nan, steps, leaf_evals)


def intersect_ray_dem_ecef(
    pyramid: MinMaxMipmapPyramid,
    origin_ecef_m: np.ndarray | tuple[float, float, float],
    dir_ecef: np.ndarray | tuple[float, float, float],
    *,
    t_min_m: float = 0.0,
    t_max_m: float = 60_000.0,
) -> RayDemHit:
    """Transform an ECEF ray into the DEM pyramid's local NED tangent frame and intersect."""
    p_orig_ecef = geodetic_to_ecef(pyramid.origin_lat_deg, pyramid.origin_lon_deg, 0.0)
    r_mat = r_ecef_ned(pyramid.origin_lat_deg, pyramid.origin_lon_deg)
    rel_ecef = np.asarray(origin_ecef_m, dtype=np.float64) - p_orig_ecef
    o_ned = r_mat.T @ rel_ecef
    d_ned = r_mat.T @ np.asarray(dir_ecef, dtype=np.float64)
    return intersect_ray_dem_ned(pyramid, o_ned, d_ned, t_min_m=t_min_m, t_max_m=t_max_m)


def intersect_ray_dem_dense_ref(
    pyramid: MinMaxMipmapPyramid,
    origin_ned_m: np.ndarray | tuple[float, float, float],
    dir_ned: np.ndarray | tuple[float, float, float],
    *,
    step_m: float = 1.0,
    t_min_m: float = 0.0,
    t_max_m: float = 60_000.0,
    earth_radius_m: float = EARTH_MEAN_RADIUS_M,
) -> RayDemHit:
    """Dense fine-step + bisection curved-Earth reference oracle for verifying mipmap accuracy."""
    o = np.asarray(origin_ned_m, dtype=np.float64)
    d_raw = np.asarray(dir_ned, dtype=np.float64)
    d = d_raw / float(np.linalg.norm(d_raw))
    n0, e0, d0 = float(o[0]), float(o[1]), float(o[2])
    dn, de, dd = float(d[0]), float(d[1]), float(d[2])

    clipped = _clip_ray_2d(
        n0,
        e0,
        dn,
        de,
        pyramid.north_min_m,
        pyramid.north_max_m,
        pyramid.east_min_m,
        pyramid.east_max_m,
        t_min_m,
        t_max_m,
    )
    if clipped is None:
        return RayDemHit(False, math.inf, math.nan, math.nan, math.nan, math.nan, math.nan, 0, 0)

    t_start, t_end = clipped

    def _f(t_val: float) -> float:
        nv = n0 + t_val * dn
        ev = e0 + t_val * de
        zv = -(d0 + t_val * dd)
        hv = sample_height_bilinear_m(pyramid, nv, ev) + earth_curvature_drop_m(nv, ev, earth_radius_m)
        return zv - hv

    t_prev = t_start
    f_prev = _f(t_prev)
    steps = 1
    if f_prev <= 0.0:
        nv, ev, dv = n0 + t_prev * dn, e0 + t_prev * de, d0 + t_prev * dd
        return RayDemHit(
            True,
            t_prev,
            nv,
            ev,
            dv,
            sample_height_bilinear_m(pyramid, nv, ev),
            earth_curvature_drop_m(nv, ev, earth_radius_m),
            steps,
            steps,
        )

    t_cur = t_prev + step_m
    while t_cur <= t_end + 1e-9:
        tc = min(t_cur, t_end)
        f_cur = _f(tc)
        steps += 1
        if f_cur <= 0.0:
            lo, hi = t_prev, tc
            for _ in range(32):
                mid = 0.5 * (lo + hi)
                if _f(mid) <= 0.0:
                    hi = mid
                else:
                    lo = mid
            t_hit = 0.5 * (lo + hi)
            nv, ev, dv = n0 + t_hit * dn, e0 + t_hit * de, d0 + t_hit * dd
            return RayDemHit(
                True,
                t_hit,
                nv,
                ev,
                dv,
                sample_height_bilinear_m(pyramid, nv, ev),
                earth_curvature_drop_m(nv, ev, earth_radius_m),
                steps,
                steps,
            )
        t_prev = tc
        t_cur += step_m

    return RayDemHit(False, math.inf, math.nan, math.nan, math.nan, math.nan, math.nan, steps, steps)
