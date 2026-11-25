"""Procedural aircraft model generators and glTF 2.0 binary builder (WP02 §2.1).

Generates civil aircraft models for size classes C1_UAS to C6_HELI,
plus calibration assets (CAL_SPHERES, CAL, SPHERE_R2, PANEL).
Conforms strictly to docs/xvs/WP02_TRAFFIC.md §2.1.
"""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
import pygltflib

REPO_ROOT = Path(__file__).resolve().parents[3]
ASSETS_DIR = REPO_ROOT / "assets" / "traffic"

# Civil color palette (spec §2.1)
CIVIL_PALETTE = {
    "white": [0.95, 0.95, 0.95, 1.0],
    "light_grey": [0.75, 0.75, 0.75, 1.0],
    "cream": [0.92, 0.90, 0.82, 1.0],
    "light_blue": [0.60, 0.75, 0.90, 1.0],
    "trainer_yellow": [0.95, 0.80, 0.10, 1.0],
    "red": [0.85, 0.15, 0.15, 1.0],
    "dark_blue": [0.10, 0.20, 0.50, 1.0],
    "black": [0.10, 0.10, 0.10, 1.0],
}


@dataclass
class MeshPart:
    """A single mesh primitive with vertices, normals, indices, and color."""
    name: str
    vertices: np.ndarray  # (N, 3) float32
    normals: np.ndarray   # (N, 3) float32
    indices: np.ndarray   # (M,) uint16 or uint32
    color: list[float]    # [r, g, b, a]
    # Lifting surfaces only (wing, tailplane, T-tail, fin): the thickness axis (2 = z for wings, 1 = y for fins) and
    # each vertex's signed offset from its section's mid-plane along it. _finalize_model uses them to keep the
    # thickness/chord ratio of every section under per-axis scaling (WP02 §2.1, Director decision D14).
    thickness_axis: int | None = None
    thickness_offsets: np.ndarray | None = None


def _smooth_normals(verts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Area-weighted per-vertex normals of a triangle list, unit length (float32)."""
    normals = np.zeros_like(verts, dtype=np.float32)
    for i in range(0, len(indices), 3):
        i0, i1, i2 = indices[i], indices[i + 1], indices[i + 2]
        fn = np.cross(verts[i1] - verts[i0], verts[i2] - verts[i0])
        norm = np.linalg.norm(fn)
        if norm > 1e-8:
            fn /= norm
            normals[i0] += fn
            normals[i1] += fn
            normals[i2] += fn
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    lengths[lengths < 1e-8] = 1.0
    return (normals / lengths).astype(np.float32)


@dataclass
class ModelResult:
    model_id: str
    cls_id: str
    seed: int
    parts: list[MeshPart]
    dimensions_m: dict[str, float]  # span, length, height
    bbox_min: list[float]
    bbox_max: list[float]
    triangle_count: int
    alpha_trim_deg: float = 0.0


CLASS_SPECS: dict[str, dict[str, Any]] = {
    "C1_UAS": {
        "span_min_m": 0.35, "span_max_m": 1.20,
        "length_min_m": 0.30, "length_max_m": 1.00,
        "alpha_trim_deg": 0.0,
    },
    "C2_SEP": {
        "span_min_m": 9.0, "span_max_m": 12.0,
        "length_min_m": 7.0, "length_max_m": 9.0,
        "alpha_trim_deg": 2.0,
    },
    "C3_TWIN": {
        "span_min_m": 12.0, "span_max_m": 18.0,
        "length_min_m": 10.0, "length_max_m": 14.0,
        "alpha_trim_deg": 2.0,
    },
    "C4_BIZJET": {
        "span_min_m": 14.0, "span_max_m": 22.0,
        "length_min_m": 15.0, "length_max_m": 22.0,
        "alpha_trim_deg": 2.0,
    },
    "C5_TRANSPORT": {
        "span_min_m": 28.0, "span_max_m": 36.0,
        "length_min_m": 30.0, "length_max_m": 40.0,
        "alpha_trim_deg": 2.0,
    },
    "C6_HELI": {
        "span_min_m": 10.0, "span_max_m": 14.0,
        "length_min_m": 10.0, "length_max_m": 14.0,
        "alpha_trim_deg": -3.0,
    },
}


def create_icosphere(
    center: list[float] | np.ndarray,
    radius: float,
    subdivisions: int = 4,
    color: list[float] | None = None,
    name: str = "icosphere",
) -> MeshPart:
    """Generate an icosphere with unit-length normals and CCW winding."""
    c = np.array(center, dtype=np.float32)
    phi = (1.0 + math.sqrt(5.0)) / 2.0

    # 12 base vertices of icosahedron
    base_verts = [
        [-1.0, phi, 0.0], [1.0, phi, 0.0], [-1.0, -phi, 0.0], [1.0, -phi, 0.0],
        [0.0, -1.0, phi], [0.0, 1.0, phi], [0.0, -1.0, -phi], [0.0, 1.0, -phi],
        [phi, 0.0, -1.0], [phi, 0.0, 1.0], [-phi, 0.0, -1.0], [-phi, 0.0, 1.0],
    ]
    # Normalize to unit sphere
    verts = [v / np.linalg.norm(v) for v in base_verts]

    # 20 base faces (CCW seen from outside)
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]

    midpoint_cache: dict[tuple[int, int], int] = {}

    def get_midpoint(i1: int, i2: int) -> int:
        edge = (min(i1, i2), max(i1, i2))
        if edge in midpoint_cache:
            return midpoint_cache[edge]
        mid = (verts[i1] + verts[i2]) * 0.5
        mid /= np.linalg.norm(mid)
        verts.append(mid)
        idx = len(verts) - 1
        midpoint_cache[edge] = idx
        return idx

    current_faces = faces
    for _ in range(subdivisions):
        new_faces = []
        for tri in current_faces:
            a = get_midpoint(tri[0], tri[1])
            b = get_midpoint(tri[1], tri[2])
            c_pt = get_midpoint(tri[2], tri[0])
            new_faces.append([tri[0], a, c_pt])
            new_faces.append([tri[1], b, a])
            new_faces.append([tri[2], c_pt, b])
            new_faces.append([a, b, c_pt])
        current_faces = new_faces

    v_arr = np.array(verts, dtype=np.float32)
    normals = v_arr.copy()
    scaled_verts = c + v_arr * float(radius)
    indices = np.array(current_faces, dtype=np.uint32).flatten()

    return MeshPart(
        name=name,
        vertices=scaled_verts.astype(np.float32),
        normals=normals.astype(np.float32),
        indices=indices,
        color=color or [1.0, 1.0, 1.0, 1.0],
    )


def create_box(
    center: list[float] | np.ndarray,
    size: list[float] | tuple[float, float, float],
    color: list[float],
    name: str = "box",
) -> MeshPart:
    """Generate an axis-aligned box with outward normals and CCW winding."""
    cx, cy, cz = center
    sx, sy, sz = [s * 0.5 for s in size]

    # 6 faces: +X, -X, +Y, -Y, +Z, -Z
    # 4 vertices per face
    faces_data = [
        # +X (Front)
        ([[cx + sx, cy - sy, cz - sz], [cx + sx, cy + sy, cz - sz], [cx + sx, cy + sy, cz + sz], [cx + sx, cy - sy, cz + sz]], [1, 0, 0]),
        # -X (Back)
        ([[cx - sx, cy + sy, cz - sz], [cx - sx, cy - sy, cz - sz], [cx - sx, cy - sy, cz + sz], [cx - sx, cy + sy, cz + sz]], [-1, 0, 0]),
        # +Y (Right)
        ([[cx + sx, cy + sy, cz - sz], [cx - sx, cy + sy, cz - sz], [cx - sx, cy + sy, cz + sz], [cx + sx, cy + sy, cz + sz]], [0, 1, 0]),
        # -Y (Left)
        ([[cx - sx, cy - sy, cz - sz], [cx + sx, cy - sy, cz - sz], [cx + sx, cy - sy, cz + sz], [cx - sx, cy - sy, cz + sz]], [0, -1, 0]),
        # +Z (Bottom)
        ([[cx - sx, cy - sy, cz + sz], [cx + sx, cy - sy, cz + sz], [cx + sx, cy + sy, cz + sz], [cx - sx, cy + sy, cz + sz]], [0, 0, 1]),
        # -Z (Top)
        ([[cx - sx, cy + sy, cz - sz], [cx + sx, cy + sy, cz - sz], [cx + sx, cy - sy, cz - sz], [cx - sx, cy - sy, cz - sz]], [0, 0, -1]),
    ]

    all_v = []
    all_n = []
    all_idx = []

    for f_idx, (corners, normal) in enumerate(faces_data):
        base_i = f_idx * 4
        all_v.extend(corners)
        all_n.extend([normal] * 4)
        all_idx.extend([base_i, base_i + 1, base_i + 2, base_i, base_i + 2, base_i + 3])

    return MeshPart(
        name=name,
        vertices=np.array(all_v, dtype=np.float32),
        normals=np.array(all_n, dtype=np.float32),
        indices=np.array(all_idx, dtype=np.uint32),
        color=color,
    )


def create_cylinder(
    p0: list[float] | np.ndarray,
    p1: list[float] | np.ndarray,
    r0: float,
    r1: float | None = None,
    n_segments: int = 16,
    color: list[float] | None = None,
    name: str = "cylinder",
) -> MeshPart:
    """Generate a cylinder or truncated cone along line segment p0 -> p1."""
    if r1 is None:
        r1 = r0
    p0 = np.array(p0, dtype=np.float32)
    p1 = np.array(p1, dtype=np.float32)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        axis_norm = np.array([1, 0, 0], dtype=np.float32)
    else:
        axis_norm = axis / length

    # Construct perpendicular frame
    if abs(axis_norm[2]) < 0.9:
        ref = np.array([0, 0, 1], dtype=np.float32)
    else:
        ref = np.array([0, 1, 0], dtype=np.float32)
    u_vec = np.cross(axis_norm, ref)
    u_vec /= np.linalg.norm(u_vec)
    v_vec = np.cross(axis_norm, u_vec)

    angles = np.linspace(0, 2 * math.pi, n_segments, endpoint=False)
    cos_a = np.cos(angles).astype(np.float32)
    sin_a = np.sin(angles).astype(np.float32)

    circle_dirs = cos_a[:, None] * u_vec[None, :] + sin_a[:, None] * v_vec[None, :]  # (S, 3)

    v0 = p0[None, :] + r0 * circle_dirs
    v1 = p1[None, :] + r1 * circle_dirs

    # Side normals
    normals0 = circle_dirs
    normals1 = circle_dirs

    # Combine side vertices
    verts = np.concatenate([v0, v1], axis=0)
    normals = np.concatenate([normals0, normals1], axis=0)

    indices = []
    for i in range(n_segments):
        nxt = (i + 1) % n_segments
        # CCW winding
        indices.extend([i, nxt, i + n_segments])
        indices.extend([nxt, nxt + n_segments, i + n_segments])

    # End caps
    cap0_center_idx = len(verts)
    verts = np.concatenate([verts, [p0, p1]], axis=0)
    normals = np.concatenate([normals, [-axis_norm, axis_norm]], axis=0)
    cap1_center_idx = cap0_center_idx + 1

    for i in range(n_segments):
        nxt = (i + 1) % n_segments
        # Cap 0 (-axis)
        indices.extend([cap0_center_idx, nxt, i])
        # Cap 1 (+axis)
        indices.extend([cap1_center_idx, i + n_segments, nxt + n_segments])

    return MeshPart(
        name=name,
        vertices=verts.astype(np.float32),
        normals=normals.astype(np.float32),
        indices=np.array(indices, dtype=np.uint32),
        color=color or [0.8, 0.8, 0.8, 1.0],
    )


def create_lofted_fuselage(
    length_m: float,
    width_m: float,
    height_m: float,
    n_sections: int = 16,
    n_slices: int = 16,
    color: list[float] | None = None,
    name: str = "fuselage",
) -> MeshPart:
    """Generate lofted streamlined elliptical fuselage with tapered nose and tail."""
    # Stations along x from +length/2 (nose) to -length/2 (tail)
    xs = np.linspace(length_m * 0.5, -length_m * 0.5, n_sections, dtype=np.float32)

    # Normalized station s in [0, 1] from nose to tail
    s = (length_m * 0.5 - xs) / length_m

    # Radii profile: smooth teardrop
    # Nose taper in first 25%, middle cruise in 25%-60%, tail taper in 60%-100%
    ry_profile = np.zeros_like(s)
    rz_profile = np.zeros_like(s)
    for i, si in enumerate(s):
        if si <= 0.25:
            # Elliptical nose
            factor = math.sqrt(max(0.0, 1.0 - ((0.25 - si) / 0.25) ** 2))
        elif si <= 0.65:
            factor = 1.0
        else:
            # Linear / cubic tail taper
            factor = max(0.05, 1.0 - ((si - 0.65) / 0.35) ** 1.5)
        ry_profile[i] = width_m * 0.5 * factor
        rz_profile[i] = height_m * 0.5 * factor

    angles = np.linspace(0, 2 * math.pi, n_slices, endpoint=False)
    cos_a = np.cos(angles).astype(np.float32)
    sin_a = np.sin(angles).astype(np.float32)

    verts = []
    normals = []
    indices = []

    for i in range(n_sections):
        x = xs[i]
        ry = ry_profile[i]
        rz = rz_profile[i]
        ring_y = ry * cos_a
        ring_z = rz * sin_a
        ring_x = np.full_like(ring_y, x)
        ring_v = np.stack([ring_x, ring_y, ring_z], axis=-1)
        verts.append(ring_v)

        # Normals
        ring_ny = cos_a
        ring_nz = sin_a
        ring_nx = np.zeros_like(ring_ny)
        ring_n = np.stack([ring_nx, ring_ny, ring_nz], axis=-1)
        normals.append(ring_n)

    verts_arr = np.concatenate(verts, axis=0)  # (N_sec * N_slices, 3)
    normals_arr = np.concatenate(normals, axis=0)

    for i in range(n_sections - 1):
        for j in range(n_slices):
            nxt_j = (j + 1) % n_slices
            v0 = i * n_slices + j
            v1 = i * n_slices + nxt_j
            v2 = (i + 1) * n_slices + nxt_j
            v3 = (i + 1) * n_slices + j
            # counter-clockwise seen from outside (§2.1; independent review R-G2-1: these were wound inward)
            indices.extend([v0, v2, v1, v0, v3, v2])

    # Close nose and tail caps
    nose_center_idx = len(verts_arr)
    tail_center_idx = nose_center_idx + 1
    nose_pt = np.array([xs[0], 0, 0], dtype=np.float32)
    tail_pt = np.array([xs[-1], 0, 0], dtype=np.float32)
    verts_arr = np.concatenate([verts_arr, [nose_pt, tail_pt]], axis=0)
    normals_arr = np.concatenate([normals_arr, [[1, 0, 0], [-1, 0, 0]]], axis=0)

    for j in range(n_slices):
        nxt_j = (j + 1) % n_slices
        # Nose cap (+X)
        indices.extend([nose_center_idx, nxt_j, j])  # outward (R-G2-1)
        # Tail cap (-X)
        last_ring = (n_sections - 1) * n_slices
        indices.extend([tail_center_idx, last_ring + nxt_j, last_ring + j])

    return MeshPart(
        name=name,
        vertices=verts_arr.astype(np.float32),
        normals=normals_arr.astype(np.float32),
        indices=np.array(indices, dtype=np.uint32),
        color=color or CIVIL_PALETTE["white"],
    )


def create_trapezoid_wing(
    span_m: float,
    root_chord_m: float,
    tip_chord_m: float,
    sweep_deg: float = 0.0,
    dihedral_deg: float = 0.0,
    thickness_ratio: float = 0.12,
    root_x_offset: float = 0.0,
    root_z_offset: float = 0.0,
    color: list[float] | None = None,
    name: str = "wing",
) -> MeshPart:
    """Generate bilateral trapezoid wings (left and right) with symmetric airfoil thickness."""
    half_span = span_m * 0.5
    sweep_rad = math.radians(sweep_deg)
    dihedral_rad = math.radians(dihedral_deg)

    # 4 stations: right tip, right root, left root, left tip
    # Root: y = 0, chord = root_chord_m, x in [x0, x0 - root_chord_m]
    # Right tip: y = +half_span, x shifted by sweep, z shifted by dihedral
    tip_x_shift = half_span * math.tan(sweep_rad)
    tip_z_shift = -half_span * math.sin(dihedral_rad)  # z down, positive dihedral raises wingtip (-z)

    # Simplified 4-facet airfoil per section (upper, lower, LE, TE)
    # Profiles defined at y = 0 and y = ±half_span. Thickness = thickness_ratio * local chord, no absolute floor
    # (WP02 §2.1, D14). The 4th component is the vertex's z offset from the section mid-plane (see MeshPart).
    def get_section_quad(y_val: float, chord: float, x_le: float, z_pos: float):
        t = chord * thickness_ratio * 0.5
        le = np.array([x_le, y_val, z_pos, 0.0], dtype=np.float32)
        top = np.array([x_le - chord * 0.3, y_val, z_pos - t, -t], dtype=np.float32)
        bottom = np.array([x_le - chord * 0.3, y_val, z_pos + t, t], dtype=np.float32)
        te = np.array([x_le - chord, y_val, z_pos, 0.0], dtype=np.float32)
        return le, top, te, bottom

    # Right wing
    r_root_le, r_root_top, r_root_te, r_root_bot = get_section_quad(0.0, root_chord_m, root_x_offset, root_z_offset)
    r_tip_le, r_tip_top, r_tip_te, r_tip_bot = get_section_quad(half_span, tip_chord_m, root_x_offset - tip_x_shift, root_z_offset + tip_z_shift)

    # Left wing
    l_root_le, l_root_top, l_root_te, l_root_bot = get_section_quad(0.0, root_chord_m, root_x_offset, root_z_offset)
    l_tip_le, l_tip_top, l_tip_te, l_tip_bot = get_section_quad(-half_span, tip_chord_m, root_x_offset - tip_x_shift, root_z_offset + tip_z_shift)

    # Assemble quad strips
    verts = []
    indices = []

    def add_quad(p0, p1, p2, p3):
        base = len(verts)
        verts.extend([p0, p1, p2, p3])
        # Two triangles CCW
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])

    # Right wing top: root_le -> root_top -> tip_top -> tip_le
    add_quad(r_root_le, r_root_top, r_tip_top, r_tip_le)
    add_quad(r_root_top, r_root_te, r_tip_te, r_tip_top)
    # Right wing bottom: root_te -> root_bot -> tip_bot -> tip_te
    add_quad(r_root_te, r_root_bot, r_tip_bot, r_tip_te)
    add_quad(r_root_bot, r_root_le, r_tip_le, r_tip_bot)
    # Right tip cap
    add_quad(r_tip_le, r_tip_top, r_tip_te, r_tip_bot)

    # Left wing top
    add_quad(l_tip_le, l_tip_top, l_root_top, l_root_le)
    add_quad(l_tip_top, l_tip_te, l_root_te, l_root_top)
    # Left wing bottom
    add_quad(l_tip_te, l_tip_bot, l_root_bot, l_root_te)
    add_quad(l_tip_bot, l_tip_le, l_root_le, l_root_bot)
    # Left tip cap
    add_quad(l_tip_bot, l_tip_te, l_tip_top, l_tip_le)

    verts4 = np.array(verts, dtype=np.float32)
    verts_arr = np.ascontiguousarray(verts4[:, :3])
    indices_arr = np.array(indices, dtype=np.uint32)

    # Compute per-vertex smooth normals
    normals = np.zeros_like(verts_arr)
    for i in range(0, len(indices_arr), 3):
        i0, i1, i2 = indices_arr[i], indices_arr[i + 1], indices_arr[i + 2]
        v0, v1, v2 = verts_arr[i0], verts_arr[i1], verts_arr[i2]
        fn = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(fn)
        if norm > 1e-8:
            fn /= norm
            normals[i0] += fn
            normals[i1] += fn
            normals[i2] += fn

    norm_lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    norm_lengths[norm_lengths < 1e-8] = 1.0
    normals /= norm_lengths

    return MeshPart(
        name=name,
        vertices=verts_arr,
        normals=normals.astype(np.float32),
        indices=indices_arr,
        color=color or CIVIL_PALETTE["white"],
        thickness_axis=2,
        thickness_offsets=np.ascontiguousarray(verts4[:, 3]),
    )


def create_vertical_fin(
    height_m: float,
    root_chord_m: float,
    tip_chord_m: float,
    sweep_deg: float = 25.0,
    thickness_ratio: float = 0.12,
    root_x_offset: float = 0.0,
    root_z_offset: float = 0.0,
    color: list[float] | None = None,
    name: str = "fin",
) -> MeshPart:
    """Generate vertical stabilizer extending upwards (-Z in FRD)."""
    sweep_rad = math.radians(sweep_deg)
    tip_x_shift = height_m * math.tan(sweep_rad)

    # Thickness = thickness_ratio * local chord, no absolute floor (WP02 §2.1, D14). 4th component: y offset from
    # the section mid-plane (see MeshPart).
    def get_section(z_val: float, chord: float, x_le: float):
        t = chord * thickness_ratio * 0.5
        le = np.array([x_le, 0, z_val, 0.0], dtype=np.float32)
        r = np.array([x_le - chord * 0.3, t, z_val, t], dtype=np.float32)
        l = np.array([x_le - chord * 0.3, -t, z_val, -t], dtype=np.float32)
        te = np.array([x_le - chord, 0, z_val, 0.0], dtype=np.float32)
        return le, r, te, l

    r_le, r_r, r_te, r_l = get_section(root_z_offset, root_chord_m, root_x_offset)
    t_le, t_r, t_te, t_l = get_section(root_z_offset - height_m, tip_chord_m, root_x_offset - tip_x_shift)

    verts = []
    indices = []

    def add_quad(p0, p1, p2, p3):
        base = len(verts)
        verts.extend([p0, p1, p2, p3])
        # counter-clockwise seen from outside (§2.1; R-G2-1: the fin quads were wound inward, and so were the
        # normals derived from them)
        indices.extend([base, base + 2, base + 1, base, base + 3, base + 2])

    # Right side
    add_quad(r_le, r_r, t_r, t_le)
    add_quad(r_r, r_te, t_te, t_r)
    # Left side
    add_quad(t_le, t_l, r_l, r_le)
    add_quad(t_l, t_te, r_te, r_l)
    # Tip cap
    add_quad(t_le, t_r, t_te, t_l)

    verts4 = np.array(verts, dtype=np.float32)
    verts_arr = np.ascontiguousarray(verts4[:, :3])
    indices_arr = np.array(indices, dtype=np.uint32)

    normals = np.zeros_like(verts_arr)
    for i in range(0, len(indices_arr), 3):
        i0, i1, i2 = indices_arr[i], indices_arr[i + 1], indices_arr[i + 2]
        fn = np.cross(verts_arr[i1] - verts_arr[i0], verts_arr[i2] - verts_arr[i0])
        norm = np.linalg.norm(fn)
        if norm > 1e-8:
            fn /= norm
            normals[i0] += fn
            normals[i1] += fn
            normals[i2] += fn

    norm_lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    norm_lengths[norm_lengths < 1e-8] = 1.0
    normals /= norm_lengths

    return MeshPart(
        name=name,
        vertices=verts_arr,
        normals=normals.astype(np.float32),
        indices=indices_arr,
        color=color or CIVIL_PALETTE["white"],
        thickness_axis=1,
        thickness_offsets=np.ascontiguousarray(verts4[:, 3]),
    )


# --- Aircraft Size Classes Generators (spec §2.1) ---

def build_c1_uas(seed: int = 1) -> ModelResult:
    """C1_UAS: Small quadcopter UAS. Span 0.35-1.2 m, Length 0.3-1.0 m."""
    spec = CLASS_SPECS["C1_UAS"]
    rng = np.random.default_rng(seed)
    diag_span = float(rng.uniform(spec["span_min_m"], spec["span_max_m"]))
    length = float(rng.uniform(spec["length_min_m"], spec["length_max_m"]))

    arm_len = diag_span * 0.5
    body_w = length * 0.35
    body_l = length * 0.50
    body_h = length * 0.18

    parts = []
    # 1. Central body
    parts.append(create_box([0, 0, 0], [body_l, body_w, body_h], CIVIL_PALETTE["dark_blue"], name="body"))

    # 2. 4 arms
    angles = [math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4]
    motor_radius = length * 0.04
    motor_h = length * 0.08
    rotor_r = length * 0.22

    for i, a in enumerate(angles):
        mx = arm_len * math.cos(a) * 0.707
        my = arm_len * math.sin(a) * 0.707
        parts.append(create_cylinder([0, 0, 0], [mx, my, 0], r0=0.015 * length, color=CIVIL_PALETTE["black"], name=f"arm_{i}"))
        # Motor
        parts.append(create_cylinder([mx, my, motor_h * 0.5], [mx, my, -motor_h * 0.5], r0=motor_radius, color=CIVIL_PALETTE["light_grey"], name=f"motor_{i}"))
        # 2-blade static rotor. Absolute thickness floor (0.015 m): rotors are thin solids, not lifting-surface
        # sections under the §2.1 thickness/chord rule, so D14 allows a floor here.
        parts.append(create_box([mx, my, -motor_h * 0.5], [rotor_r * 2.0, motor_radius * 0.8, max(0.015, length * 0.02)], CIVIL_PALETTE["black"], name=f"rotor_{i}"))

    return _finalize_model("C1_UAS", f"C1_UAS_{seed:04d}", seed, parts, target_span=diag_span, target_length=length, alpha_trim_deg=spec["alpha_trim_deg"])


def build_c2_sep(seed: int = 1) -> ModelResult:
    """C2_SEP: Light single-engine fixed wing. Span 9-12 m, Length 7-9 m."""
    spec = CLASS_SPECS["C2_SEP"]
    rng = np.random.default_rng(seed)
    span = float(rng.uniform(spec["span_min_m"], spec["span_max_m"]))
    length = float(rng.uniform(spec["length_min_m"], spec["length_max_m"]))
    color_name = rng.choice(["white", "cream", "light_blue", "trainer_yellow"])
    base_color = CIVIL_PALETTE[color_name]

    parts = []
    # Fuselage
    fuse_w = length * 0.14
    fuse_h = length * 0.16
    parts.append(create_lofted_fuselage(length, fuse_w, fuse_h, color=base_color, name="fuselage"))

    # Wing
    root_chord = length * 0.20
    tip_chord = length * 0.12
    wing_z = -fuse_h * 0.2  # low/mid wing
    parts.append(create_trapezoid_wing(span, root_chord, tip_chord, sweep_deg=1.0, dihedral_deg=3.0, root_x_offset=length * 0.1, root_z_offset=wing_z, color=base_color, name="wing"))

    # Tailplane
    tail_span = span * 0.35
    tail_chord = length * 0.13
    parts.append(create_trapezoid_wing(tail_span, tail_chord, tail_chord * 0.6, sweep_deg=5.0, root_x_offset=-length * 0.38, root_z_offset=-fuse_h * 0.1, color=base_color, name="tailplane"))

    # Fin
    fin_h = length * 0.22
    parts.append(create_vertical_fin(fin_h, length * 0.18, length * 0.09, sweep_deg=30.0, root_x_offset=-length * 0.33, root_z_offset=-fuse_h * 0.3, color=base_color, name="fin"))

    # Nose propeller disc
    prop_r = fuse_w * 0.9
    parts.append(create_cylinder([length * 0.50, 0, 0], [length * 0.52, 0, 0], r0=prop_r, color=CIVIL_PALETTE["black"], n_segments=24, name="propeller"))

    return _finalize_model("C2_SEP", f"C2_SEP_{seed:04d}", seed, parts, target_span=span, target_length=length, alpha_trim_deg=spec["alpha_trim_deg"])


def build_c3_twin(seed: int = 1) -> ModelResult:
    """C3_TWIN: Light twin / turboprop. Span 12-18 m, Length 10-14 m."""
    spec = CLASS_SPECS["C3_TWIN"]
    rng = np.random.default_rng(seed)
    span = float(rng.uniform(spec["span_min_m"], spec["span_max_m"]))
    length = float(rng.uniform(spec["length_min_m"], spec["length_max_m"]))
    base_color = CIVIL_PALETTE[rng.choice(["white", "light_grey", "cream"])]

    parts = []
    # Fuselage
    fuse_w = length * 0.14
    fuse_h = length * 0.16
    parts.append(create_lofted_fuselage(length, fuse_w, fuse_h, color=base_color, name="fuselage"))

    # Wing
    root_chord = length * 0.22
    tip_chord = length * 0.11
    wing_z = -fuse_h * 0.2
    parts.append(create_trapezoid_wing(span, root_chord, tip_chord, sweep_deg=2.0, dihedral_deg=4.0, root_x_offset=length * 0.1, root_z_offset=wing_z, color=base_color, name="wing"))

    # Tailplane & Fin
    parts.append(create_trapezoid_wing(span * 0.36, length * 0.14, length * 0.08, sweep_deg=8.0, root_x_offset=-length * 0.38, root_z_offset=-fuse_h * 0.1, color=base_color, name="tailplane"))
    parts.append(create_vertical_fin(length * 0.24, length * 0.20, length * 0.10, sweep_deg=35.0, root_x_offset=-length * 0.32, root_z_offset=-fuse_h * 0.3, color=base_color, name="fin"))

    # 2 engine nacelles on wings + props
    nacelle_y = span * 0.22
    nacelle_len = length * 0.25
    nacelle_r = length * 0.05
    for sign in (+1, -1):
        ny = sign * nacelle_y
        parts.append(create_cylinder([length * 0.18, ny, wing_z], [length * 0.18 - nacelle_len, ny, wing_z], r0=nacelle_r, color=CIVIL_PALETTE["dark_blue"], name=f"nacelle_{sign}"))
        parts.append(create_cylinder([length * 0.19, ny, wing_z], [length * 0.20, ny, wing_z], r0=nacelle_r * 1.8, color=CIVIL_PALETTE["black"], n_segments=20, name=f"prop_{sign}"))

    return _finalize_model("C3_TWIN", f"C3_TWIN_{seed:04d}", seed, parts, target_span=span, target_length=length, alpha_trim_deg=spec["alpha_trim_deg"])


def build_c4_bizjet(seed: int = 1) -> ModelResult:
    """C4_BIZJET: Business jet. Span 14-22 m, Length 15-22 m."""
    spec = CLASS_SPECS["C4_BIZJET"]
    rng = np.random.default_rng(seed)
    span = float(rng.uniform(spec["span_min_m"], spec["span_max_m"]))
    length = float(rng.uniform(spec["length_min_m"], spec["length_max_m"]))
    base_color = CIVIL_PALETTE["white"]

    parts = []
    fuse_w = length * 0.11
    fuse_h = length * 0.12
    parts.append(create_lofted_fuselage(length, fuse_w, fuse_h, color=base_color, name="fuselage"))

    # Swept low wing
    root_chord = length * 0.22
    tip_chord = length * 0.07
    parts.append(create_trapezoid_wing(span, root_chord, tip_chord, sweep_deg=22.0, dihedral_deg=3.0, root_x_offset=length * 0.05, root_z_offset=fuse_h * 0.1, color=base_color, name="wing"))

    # T-tail fin & horizontal stabilizer mounted on top
    fin_h = length * 0.26
    parts.append(create_vertical_fin(fin_h, length * 0.22, length * 0.12, sweep_deg=40.0, root_x_offset=-length * 0.30, root_z_offset=-fuse_h * 0.2, color=base_color, name="fin"))
    t_tail_x = -length * 0.30 - fin_h * math.tan(math.radians(40.0)) - length * 0.04
    parts.append(create_trapezoid_wing(span * 0.32, length * 0.12, length * 0.05, sweep_deg=25.0, root_x_offset=t_tail_x, root_z_offset=-fuse_h * 0.2 - fin_h, color=base_color, name="t_tail"))

    # 2 rear fuselage engines
    eng_len = length * 0.22
    eng_r = length * 0.04
    eng_y = fuse_w * 0.85
    eng_z = -fuse_h * 0.2
    for sign in (+1, -1):
        parts.append(create_cylinder([-length * 0.12, sign * eng_y, eng_z], [-length * 0.12 - eng_len, sign * eng_y, eng_z], r0=eng_r, color=CIVIL_PALETTE["light_grey"], name=f"engine_{sign}"))

    return _finalize_model("C4_BIZJET", f"C4_BIZJET_{seed:04d}", seed, parts, target_span=span, target_length=length, alpha_trim_deg=spec["alpha_trim_deg"])


def build_c5_transport(seed: int = 1) -> ModelResult:
    """C5_TRANSPORT: Narrow-body transport jet. Span 28-36 m, Length 30-40 m."""
    spec = CLASS_SPECS["C5_TRANSPORT"]
    rng = np.random.default_rng(seed)
    span = float(rng.uniform(spec["span_min_m"], spec["span_max_m"]))
    length = float(rng.uniform(spec["length_min_m"], spec["length_max_m"]))
    base_color = CIVIL_PALETTE["white"]

    parts = []
    fuse_w = length * 0.10
    fuse_h = length * 0.11
    parts.append(create_lofted_fuselage(length, fuse_w, fuse_h, color=base_color, name="fuselage"))

    # Swept wing
    root_chord = length * 0.22
    tip_chord = length * 0.06
    wing_z = fuse_h * 0.1
    parts.append(create_trapezoid_wing(span, root_chord, tip_chord, sweep_deg=25.0, dihedral_deg=5.0, root_x_offset=length * 0.06, root_z_offset=wing_z, color=base_color, name="wing"))

    # Tailplane & Fin
    parts.append(create_trapezoid_wing(span * 0.34, length * 0.13, length * 0.05, sweep_deg=28.0, dihedral_deg=4.0, root_x_offset=-length * 0.38, root_z_offset=-fuse_h * 0.05, color=base_color, name="tailplane"))
    parts.append(create_vertical_fin(length * 0.28, length * 0.22, length * 0.09, sweep_deg=40.0, root_x_offset=-length * 0.32, root_z_offset=-fuse_h * 0.3, color=base_color, name="fin"))

    # 2 underwing turbofans
    eng_y = span * 0.18
    eng_len = length * 0.16
    eng_r = length * 0.035
    eng_z = wing_z + length * 0.04
    for sign in (+1, -1):
        parts.append(create_cylinder([length * 0.08, sign * eng_y, eng_z], [length * 0.08 - eng_len, sign * eng_y, eng_z], r0=eng_r, color=CIVIL_PALETTE["light_grey"], name=f"turbofan_{sign}"))

    return _finalize_model("C5_TRANSPORT", f"C5_TRANSPORT_{seed:04d}", seed, parts, target_span=span, target_length=length, alpha_trim_deg=spec["alpha_trim_deg"])


def build_c6_heli(seed: int = 1) -> ModelResult:
    """C6_HELI: Civil helicopter. Rotor diameter 10-14 m, Length 10-14 m."""
    spec = CLASS_SPECS["C6_HELI"]
    rng = np.random.default_rng(seed)
    rotor_dia = float(rng.uniform(spec["span_min_m"], spec["span_max_m"]))
    length = float(rng.uniform(spec["length_min_m"], spec["length_max_m"]))
    base_color = CIVIL_PALETTE[rng.choice(["white", "red", "trainer_yellow"])]

    parts = []
    # Cabin fuselage
    cabin_len = length * 0.45
    cabin_w = length * 0.18
    cabin_h = length * 0.22
    parts.append(create_lofted_fuselage(cabin_len, cabin_w, cabin_h, color=base_color, name="cabin"))

    # Tail boom
    boom_len = length * 0.55
    parts.append(create_cylinder([-cabin_len * 0.3, 0, -cabin_h * 0.1], [-cabin_len * 0.3 - boom_len, 0, -cabin_h * 0.1], r0=length * 0.03, color=base_color, name="tail_boom"))

    # Vertical fin at boom end
    fin_x = -cabin_len * 0.3 - boom_len
    parts.append(create_vertical_fin(length * 0.18, length * 0.12, length * 0.06, sweep_deg=20.0, root_x_offset=fin_x, root_z_offset=-cabin_h * 0.1, color=base_color, name="fin"))

    # Landing skids
    skid_r = length * 0.015
    skid_z = cabin_h * 0.55
    skid_y = cabin_w * 0.45
    for sign in (+1, -1):
        parts.append(create_cylinder([cabin_len * 0.4, sign * skid_y, skid_z], [-cabin_len * 0.4, sign * skid_y, skid_z], r0=skid_r, color=CIVIL_PALETTE["black"], name=f"skid_{sign}"))
        parts.append(create_cylinder([cabin_len * 0.15, sign * skid_y, skid_z], [cabin_len * 0.15, sign * skid_y * 0.7, cabin_h * 0.3], r0=skid_r, color=CIVIL_PALETTE["black"], name=f"strut_f_{sign}"))
        parts.append(create_cylinder([-cabin_len * 0.15, sign * skid_y, skid_z], [-cabin_len * 0.15, sign * skid_y * 0.7, cabin_h * 0.3], r0=skid_r, color=CIVIL_PALETTE["black"], name=f"strut_r_{sign}"))

    # Main rotor: mast + 4 static blades
    mast_top_z = -cabin_h * 0.55
    parts.append(create_cylinder([0, 0, -cabin_h * 0.3], [0, 0, mast_top_z], r0=length * 0.02, color=CIVIL_PALETTE["black"], name="mast"))

    blade_r = rotor_dia * 0.5
    blade_w = length * 0.025
    # Absolute thickness floor (0.15 m) on the main-rotor blades: they are thin solids, not lifting-surface sections
    # under the §2.1 thickness/chord rule, so D14 allows a floor here.
    blade_thick = max(0.15, length * 0.015)
    for i in range(4):
        a = i * (math.pi / 2.0)
        bx = (blade_r * 0.5) * math.cos(a)
        by = (blade_r * 0.5) * math.sin(a)
        # blade rectangle
        parts.append(create_box([bx, by, mast_top_z], [blade_r if i % 2 == 0 else blade_w, blade_w if i % 2 == 0 else blade_r, blade_thick], CIVIL_PALETTE["black"], name=f"blade_{i}"))

    # Tail rotor
    parts.append(create_cylinder([fin_x, length * 0.04, -cabin_h * 0.1], [fin_x, length * 0.05, -cabin_h * 0.1], r0=length * 0.09, color=CIVIL_PALETTE["black"], n_segments=16, name="tail_rotor"))

    return _finalize_model("C6_HELI", f"C6_HELI_{seed:04d}", seed, parts, target_span=rotor_dia, target_length=length, alpha_trim_deg=spec["alpha_trim_deg"])


# --- Calibration Assets (spec §2.1) ---

def build_cal_spheres() -> ModelResult:
    """CAL_SPHERES: six spheres (radius 0.25 m, icosphere subdivision >= 4), pure unlit colors.

    Centred at BODY:
      nose (+5.5, 0, 0) red [1.0, 0.0, 0.0, 1.0]
      tail (-5.5, 0, 0) green [0.0, 1.0, 0.0, 1.0]
      right wingtip (0, +6.5, 0) blue [0.0, 0.0, 1.0, 1.0]
      left wingtip (0, -6.5, 0) yellow [1.0, 1.0, 0.0, 1.0]
      fin top (-4.5, 0, -3.0) magenta [1.0, 0.0, 1.0, 1.0]
      belly (0, 0, +1.5) cyan [0.0, 1.0, 1.0, 1.0]
    """
    spheres_spec = [
        ("nose", [+5.5, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]),
        ("tail", [-5.5, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]),
        ("right_wingtip", [0.0, +6.5, 0.0], [0.0, 0.0, 1.0, 1.0]),
        ("left_wingtip", [0.0, -6.5, 0.0], [1.0, 1.0, 0.0, 1.0]),
        ("fin_top", [-4.5, 0.0, -3.0], [1.0, 0.0, 1.0, 1.0]),
        ("belly", [0.0, 0.0, +1.5], [0.0, 1.0, 1.0, 1.0]),
    ]
    parts = []
    for name, pos, col in spheres_spec:
        parts.append(create_icosphere(center=pos, radius=0.25, subdivisions=4, color=col, name=name))

    return _finalize_calibration_model("CAL_SPHERES", parts)


def build_cal() -> ModelResult:
    """CAL: C2-like airframe (span 12 m, length 10 m) together with the six calibration spheres."""
    # Base C2-like airframe
    span = 12.0
    length = 10.0
    parts = []

    fuse_w = length * 0.14
    fuse_h = length * 0.16
    parts.append(create_lofted_fuselage(length, fuse_w, fuse_h, color=CIVIL_PALETTE["white"], name="fuselage"))

    root_chord = length * 0.20
    tip_chord = length * 0.12
    parts.append(create_trapezoid_wing(span, root_chord, tip_chord, root_x_offset=length * 0.05, root_z_offset=-fuse_h * 0.2, color=CIVIL_PALETTE["white"], name="wing"))
    parts.append(create_trapezoid_wing(span * 0.35, length * 0.13, length * 0.07, root_x_offset=-length * 0.40, root_z_offset=-fuse_h * 0.1, color=CIVIL_PALETTE["white"], name="tailplane"))
    parts.append(create_vertical_fin(length * 0.24, length * 0.18, length * 0.09, sweep_deg=30.0, root_x_offset=-length * 0.35, root_z_offset=-fuse_h * 0.3, color=CIVIL_PALETTE["white"], name="fin"))

    # Add the six calibration spheres
    spheres_spec = [
        ("sphere_nose", [+5.5, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]),
        ("sphere_tail", [-5.5, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]),
        ("sphere_rwing", [0.0, +6.5, 0.0], [0.0, 0.0, 1.0, 1.0]),
        ("sphere_lwing", [0.0, -6.5, 0.0], [1.0, 1.0, 0.0, 1.0]),
        ("sphere_fin", [-4.5, 0.0, -3.0], [1.0, 0.0, 1.0, 1.0]),
        ("sphere_belly", [0.0, 0.0, +1.5], [0.0, 1.0, 1.0, 1.0]),
    ]
    for name, pos, col in spheres_spec:
        parts.append(create_icosphere(center=pos, radius=0.25, subdivisions=4, color=col, name=name))

    return _finalize_calibration_model("CAL", parts)


def build_sphere_r2() -> ModelResult:
    """SPHERE_R2: White sphere of radius 2 m (icosphere subdivision >= 4) centred at BODY origin."""
    part = create_icosphere(center=[0.0, 0.0, 0.0], radius=2.0, subdivisions=4, color=[1.0, 1.0, 1.0, 1.0], name="sphere_r2")
    return _finalize_calibration_model("SPHERE_R2", [part])


def build_panel() -> ModelResult:
    """PANEL: Box 100 km wide x 20 km tall x 100 m thick for T-OCC."""
    # In FRD: x=100 m (thickness), y=100,000 m (width), z=20,000 m (height)
    part = create_box(center=[0.0, 0.0, 0.0], size=[100.0, 100000.0, 20000.0], color=[1.0, 1.0, 1.0, 1.0], name="panel")
    return _finalize_calibration_model("PANEL", [part])


# --- Finalization & glTF Binary Export ---

def _finalize_model(
    cls_id: str,
    model_id: str,
    seed: int,
    parts: list[MeshPart],
    target_span: float | None = None,
    target_length: float | None = None,
    alpha_trim_deg: float = 0.0,
) -> ModelResult:
    """Center model at axis-aligned bounding box center and compute metadata (spec §2.1)."""
    all_verts = np.concatenate([p.vertices for p in parts], axis=0)
    bbox_min = np.min(all_verts, axis=0)
    bbox_max = np.max(all_verts, axis=0)
    center = (bbox_min + bbox_max) * 0.5

    # Shift vertices so origin is exact AABB center
    for p in parts:
        p.vertices = (p.vertices - center).astype(np.float32)

    cur_len = float(bbox_max[0] - bbox_min[0])
    cur_span = float(bbox_max[1] - bbox_min[1])

    if target_length is not None and cur_len > 0 and target_span is not None and cur_span > 0:
        sx = float(target_length / cur_len)
        sy = float(target_span / cur_span)
        sz = float((sx + sy) * 0.5)
        scale_vec = np.array([sx, sy, sz], dtype=np.float32)
        inv_scale = np.array([1.0 / sx, 1.0 / sy, 1.0 / sz], dtype=np.float32)
        for p in parts:
            p.vertices *= scale_vec
            p.normals *= inv_scale
            norms = np.linalg.norm(p.normals, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            p.normals = (p.normals / norms).astype(np.float32)
            if p.thickness_axis is not None:
                # Lifting surface: the chord (x) was scaled by sx but the thickness by the scale of its own axis.
                # Re-scale each vertex's offset from its section mid-plane by sx, so every section keeps its
                # designed thickness/chord ratio (WP02 §2.1, D14), then recompute the normals.
                ax = p.thickness_axis
                p.vertices[:, ax] += (p.thickness_offsets * np.float32(sx - scale_vec[ax])).astype(np.float32)
                p.normals = _smooth_normals(p.vertices, p.indices)

    new_all_verts = np.concatenate([p.vertices for p in parts], axis=0)
    final_bbox_min = np.min(new_all_verts, axis=0).tolist()
    final_bbox_max = np.max(new_all_verts, axis=0).tolist()

    dims = {
        "length_m": float(final_bbox_max[0] - final_bbox_min[0]),
        "span_m": float(final_bbox_max[1] - final_bbox_min[1]),
        "height_m": float(final_bbox_max[2] - final_bbox_min[2]),
    }
    tri_count = sum(len(p.indices) // 3 for p in parts)

    return ModelResult(
        model_id=model_id,
        cls_id=cls_id,
        seed=seed,
        parts=parts,
        dimensions_m=dims,
        bbox_min=final_bbox_min,
        bbox_max=final_bbox_max,
        triangle_count=tri_count,
        alpha_trim_deg=alpha_trim_deg,
    )


def _finalize_calibration_model(model_id: str, parts: list[MeshPart]) -> ModelResult:
    """Keep explicit coordinates for calibration assets."""
    all_verts = np.concatenate([p.vertices for p in parts], axis=0)
    final_bbox_min = np.min(all_verts, axis=0).tolist()
    final_bbox_max = np.max(all_verts, axis=0).tolist()
    dims = {
        "length_m": float(final_bbox_max[0] - final_bbox_min[0]),
        "span_m": float(final_bbox_max[1] - final_bbox_min[1]),
        "height_m": float(final_bbox_max[2] - final_bbox_min[2]),
    }
    tri_count = sum(len(p.indices) // 3 for p in parts)

    return ModelResult(
        model_id=model_id,
        cls_id="CALIBRATION",
        seed=0,
        parts=parts,
        dimensions_m=dims,
        bbox_min=final_bbox_min,
        bbox_max=final_bbox_max,
        triangle_count=tri_count,
    )


def export_glb(model: ModelResult, out_glb_path: str | Path) -> str:
    """Export ModelResult to glTF 2.0 binary (.glb) using pygltflib.

    Returns the GLB SHA-256 string.
    """
    out_path = Path(out_glb_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    materials = []
    primitives = []
    accessors = []
    buffer_views = []
    binary_chunks = []
    current_byte_offset = 0

    for idx, part in enumerate(model.parts):
        # Material
        mat = pygltflib.Material(
            name=f"mat_{part.name}",
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorFactor=part.color,
                metallicFactor=0.1,
                roughnessFactor=0.6,
            ),
            doubleSided=False,
        )
        materials.append(mat)

        # Buffer data: Vertices (VEC3 FLOAT)
        v_bytes = part.vertices.tobytes()
        v_view_idx = len(buffer_views)
        buffer_views.append(
            pygltflib.BufferView(
                buffer=0,
                byteOffset=current_byte_offset,
                byteLength=len(v_bytes),
                target=pygltflib.ARRAY_BUFFER,
            )
        )
        v_acc_idx = len(accessors)
        v_min = np.min(part.vertices, axis=0).tolist()
        v_max = np.max(part.vertices, axis=0).tolist()
        accessors.append(
            pygltflib.Accessor(
                bufferView=v_view_idx,
                byteOffset=0,
                componentType=pygltflib.FLOAT,
                count=len(part.vertices),
                type=pygltflib.VEC3,
                min=v_min,
                max=v_max,
            )
        )
        binary_chunks.append(v_bytes)
        current_byte_offset += len(v_bytes)

        # Buffer data: Normals (VEC3 FLOAT)
        n_bytes = part.normals.tobytes()
        n_view_idx = len(buffer_views)
        buffer_views.append(
            pygltflib.BufferView(
                buffer=0,
                byteOffset=current_byte_offset,
                byteLength=len(n_bytes),
                target=pygltflib.ARRAY_BUFFER,
            )
        )
        n_acc_idx = len(accessors)
        n_min = np.min(part.normals, axis=0).tolist()
        n_max = np.max(part.normals, axis=0).tolist()
        accessors.append(
            pygltflib.Accessor(
                bufferView=n_view_idx,
                byteOffset=0,
                componentType=pygltflib.FLOAT,
                count=len(part.normals),
                type=pygltflib.VEC3,
                min=n_min,
                max=n_max,
            )
        )
        binary_chunks.append(n_bytes)
        current_byte_offset += len(n_bytes)

        # Buffer data: Indices (SCALAR UNSIGNED_INT)
        idx_bytes = part.indices.astype(np.uint32).tobytes()
        idx_view_idx = len(buffer_views)
        buffer_views.append(
            pygltflib.BufferView(
                buffer=0,
                byteOffset=current_byte_offset,
                byteLength=len(idx_bytes),
                target=pygltflib.ELEMENT_ARRAY_BUFFER,
            )
        )
        idx_acc_idx = len(accessors)
        accessors.append(
            pygltflib.Accessor(
                bufferView=idx_view_idx,
                byteOffset=0,
                componentType=pygltflib.UNSIGNED_INT,
                count=len(part.indices),
                type=pygltflib.SCALAR,
                min=[int(np.min(part.indices))],
                max=[int(np.max(part.indices))],
            )
        )
        binary_chunks.append(idx_bytes)
        current_byte_offset += len(idx_bytes)

        # Primitive
        primitives.append(
            pygltflib.Primitive(
                attributes=pygltflib.Attributes(POSITION=v_acc_idx, NORMAL=n_acc_idx),
                indices=idx_acc_idx,
                material=idx,
                mode=pygltflib.TRIANGLES,
            )
        )

    # Pad final binary buffer to 4 bytes
    total_bin = b"".join(binary_chunks)
    pad = (4 - (len(total_bin) % 4)) % 4
    if pad:
        total_bin += b"\x00" * pad

    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(mesh=0)],
        meshes=[pygltflib.Mesh(primitives=primitives)],
        materials=materials,
        accessors=accessors,
        bufferViews=buffer_views,
        buffers=[pygltflib.Buffer(byteLength=len(total_bin))],
    )
    gltf.set_binary_blob(total_bin)

    raw_glb = b"".join(gltf.save_to_bytes())
    with open(out_path, "wb") as f:
        f.write(raw_glb)

    glb_sha = hashlib.sha256(raw_glb).hexdigest()

    # Write metadata sidecar JSON
    meta_path = out_path.with_suffix(".json")
    meta = {
        "schema": "xvs.traffic_model/1",
        "model_id": model.model_id,
        "class": model.cls_id,
        "seed": model.seed,
        "dimensions": model.dimensions_m,
        "bbox_min": model.bbox_min,
        "bbox_max": model.bbox_max,
        "origin": [0.0, 0.0, 0.0],
        "triangle_count": model.triangle_count,
        "materials": [p.name for p in model.parts],
        "glb_sha256": glb_sha,
        "file_size_bytes": len(raw_glb),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return glb_sha
