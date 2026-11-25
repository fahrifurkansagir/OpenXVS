"""Director check of WP02 §2.1 / D14 on the shipped GLBs: thickness/chord of every lifting-surface section.

Reads the GLBs with pygltflib only (no `xvs` code). One glTF primitive per part; lifting surfaces are the
primitives whose material is mat_wing, mat_tailplane, mat_t_tail or mat_fin. A section is the vertex set of one
chordwise plane (constant y for wings and tails, thickness along z; constant z for fins, thickness along y).
usage: python audit/xvs/audit_g1_thickness.py [assets_dir] [--out DIR]
       -> DIR/audit_g1_thickness.json (default DIR: output/xvs/audit/wp02_g1)
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import oracle as O
import pygltflib

ARGS, OUT = O.parse_out(sys.argv[1:], 'wp02_g1')
ASSETS = Path(ARGS[0]) if ARGS else O.REPO_ROOT / 'assets' / 'traffic'
LIFTING = {'mat_wing': (1, 2), 'mat_tailplane': (1, 2), 'mat_t_tail': (1, 2), 'mat_fin': (2, 1)}  # (plane axis, thickness axis)

rows, per_model = [], {}
for glb in sorted(ASSETS.rglob('*.glb')):
    g = pygltflib.GLTF2().load(str(glb)); blob = g.binary_blob()
    for prim in g.meshes[0].primitives:
        mat = g.materials[prim.material].name
        if mat not in LIFTING:
            continue
        acc = g.accessors[prim.attributes.POSITION]; bv = g.bufferViews[acc.bufferView]
        start = (bv.byteOffset or 0) + (acc.byteOffset or 0); stride = bv.byteStride or 12
        v = np.array([np.frombuffer(blob, dtype='<f4', count=3, offset=start + i * stride) for i in range(acc.count)], float)
        pa, ta = LIFTING[mat]
        keys = np.round(v[:, pa], 4)
        for k in np.unique(keys):
            s = v[keys == k]
            chord = float(np.ptp(s[:, 0])); thick = float(np.ptp(s[:, ta]))
            r = thick / chord
            rows.append({'model': glb.relative_to(ASSETS).as_posix(), 'part': mat[4:], 'plane_m': float(k),
                         'chord_m': chord, 'thickness_m': thick, 'ratio': r})
            m = per_model.setdefault(glb.relative_to(ASSETS).as_posix(), [r, r])
            m[0] = min(m[0], r); m[1] = max(m[1], r)
try:
    assets_rel = ASSETS.resolve().relative_to(O.REPO_ROOT).as_posix()
except ValueError:
    assets_rel = ASSETS.as_posix()
out = {'assets': assets_rel, 'n_sections': len(rows), 'n_models': len(per_model),
       'n_outside_10_14pct': sum(1 for r in rows if not (0.10 <= r['ratio'] <= 0.14)),
       'ratio_min': min(r['ratio'] for r in rows), 'ratio_max': max(r['ratio'] for r in rows),
       'per_model_min_max': per_model, 'rows': rows}
(OUT / 'audit_g1_thickness.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
print(json.dumps({k: v for k, v in out.items() if k not in ('rows', 'per_model_min_max')}))
