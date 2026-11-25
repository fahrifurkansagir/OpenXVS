"""WP02 §2.1 / Director decision D14 (closes C-G1-4): every lifting-surface section of every shipped model has a
thickness/chord ratio in [0.10, 0.14].

The check reads the committed GLBs, not the builder, so it covers exactly what is rendered. Lifting surfaces are
the primitives whose material is `mat_wing`, `mat_tailplane`, `mat_t_tail` or `mat_fin` (one primitive per part,
models.export_glb). A section is the set of vertices in one chordwise plane: constant y for wings and tails
(thickness along z), constant z for fins (thickness along y). Chord = x extent, thickness = extent along the
thickness axis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAFFIC = REPO_ROOT / "assets" / "traffic"
LIFTING = {"mat_wing": ("y", "z"), "mat_tailplane": ("y", "z"), "mat_t_tail": ("y", "z"), "mat_fin": ("z", "y")}
AXIS = {"x": 0, "y": 1, "z": 2}
MODELS = [f"{c}/{c}_{s:04d}.glb" for c in ("C2_SEP", "C3_TWIN", "C4_BIZJET", "C5_TRANSPORT", "C6_HELI") for s in (1, 2, 3)]
MODELS.append("CALIBRATION/CAL.glb")


def _lifting_sections(path: Path) -> list[tuple[str, float, float, float]]:
    import pygltflib

    g = pygltflib.GLTF2().load(str(path))
    blob = g.binary_blob()
    out = []
    for prim in g.meshes[0].primitives:
        mat = g.materials[prim.material].name
        if mat not in LIFTING:
            continue
        acc = g.accessors[prim.attributes.POSITION]
        bv = g.bufferViews[acc.bufferView]
        start = (bv.byteOffset or 0) + (acc.byteOffset or 0)
        stride = bv.byteStride or 12
        v = np.array([np.frombuffer(blob, dtype="<f4", count=3, offset=start + i * stride) for i in range(acc.count)],
                     dtype=np.float64)
        plane_ax, thick_ax = AXIS[LIFTING[mat][0]], AXIS[LIFTING[mat][1]]
        keys = np.round(v[:, plane_ax], 4)
        for k in np.unique(keys):
            s = v[keys == k]
            chord = float(np.ptp(s[:, 0]))
            thickness = float(np.ptp(s[:, thick_ax]))
            out.append((mat, float(k), chord, thickness))
    return out


@pytest.mark.parametrize("rel", MODELS)
def test_lifting_surface_thickness_ratio_per_section(rel):
    sections = _lifting_sections(TRAFFIC / rel)
    mats = {m for m, *_ in sections}
    expected = {"mat_fin"} if rel.startswith("C6_HELI") else {"mat_wing", "mat_fin"}  # helicopter: fin only
    assert expected <= mats, f"{rel}: lifting surfaces not found ({sorted(mats)})"
    for mat, plane, chord, thickness in sections:
        assert chord > 0.0
        ratio = thickness / chord
        assert 0.10 <= ratio <= 0.14, f"{rel} {mat} section at {plane:+.4f}: thickness/chord = {ratio:.4f}"
