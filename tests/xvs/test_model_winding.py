"""WP02 §2.1: faces wind counter-clockwise seen from outside, and normals point outward (independent review R-G2-1).

Reads the shipped GLBs. For every primitive (one per part): the signed volume sum(a . (b x c)) / 6 is positive, and
at least 95 % of the triangles have a winding normal that agrees with the sum of their stored vertex normals.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GLBS = sorted((REPO_ROOT / "assets" / "traffic").rglob("*.glb"))


def _primitives(path: Path):
    import pygltflib

    g = pygltflib.GLTF2().load(str(path))
    blob = g.binary_blob()

    def read(ai: int, ncomp: int, dt: str) -> np.ndarray:
        a = g.accessors[ai]
        bv = g.bufferViews[a.bufferView]
        start = (bv.byteOffset or 0) + (a.byteOffset or 0)
        stride = bv.byteStride or np.dtype(dt).itemsize * ncomp
        return np.array([np.frombuffer(blob, dtype=dt, count=ncomp, offset=start + i * stride) for i in range(a.count)]).reshape(a.count, ncomp)

    for pr in g.meshes[0].primitives:
        v = read(pr.attributes.POSITION, 3, "<f4").astype(np.float64)
        n = read(pr.attributes.NORMAL, 3, "<f4").astype(np.float64)
        dt = {5125: "<u4", 5123: "<u2", 5121: "u1"}[g.accessors[pr.indices].componentType]
        f = read(pr.indices, 1, dt).reshape(-1, 3).astype(np.int64)
        yield g.materials[pr.material].name, v, n, f


@pytest.mark.parametrize("path", GLBS, ids=[p.relative_to(REPO_ROOT / "assets" / "traffic").as_posix() for p in GLBS])
def test_parts_wind_outward_and_normals_agree(path):
    for name, v, n, f in _primitives(path):
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        fn = np.cross(b - a, c - a)
        area = np.linalg.norm(fn, axis=1)
        keep = area > 1e-12
        volume = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
        agree = float(np.mean(np.einsum("ij,ij->i", fn[keep], (n[f[:, 0]] + n[f[:, 1]] + n[f[:, 2]])[keep]) > 0.0))
        assert volume > 0.0, f"{path.name} {name}: signed volume {volume:.4g} <= 0 (faces wound inward)"
        assert agree >= 0.95, f"{path.name} {name}: only {agree:.3f} of the triangles agree with their stored normals"
