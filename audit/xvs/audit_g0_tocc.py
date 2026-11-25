"""Director check of G0 T-OCC on the saved mask PNGs, with the independent oracle (no `xvs` imports).

usage: python audit/xvs/audit_g0_tocc.py <g0_dir> [--out DIR]   -> DIR/audit_g0_tocc.json (default output/xvs/audit/wp02_g0)
(2026-09-24, Engineering Automation on the PC: evaluates the saved t_occ_depthTest_{false,true}_{ref,hw}.png of a G0 run
instead of re-rendering them in a scratch folder; the sandbox version rendered with the repo code first.)
"""
import sys, json
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
import oracle as O
ARGS, OUT = O.parse_out(sys.argv[1:], 'wp02_g0')
G0 = Path(ARGS[0])
own = dict(p=O.geodetic_to_ecef(37.62, -122.37, 10000.0), R_ecef_body=O.r_ecef_ned(37.62, -122.37) @ O.r_ned_body(0, 0, 0))
C, R, intr = O.cam_pose(own, 'F')
vcol, scol = O.limb_crossings(C, R, intr)
out = {}
for bk in ('ref', 'hw'):
    impl = json.loads((G0 / f'results_{bk}.json').read_text(encoding='utf-8'))['t_occ']['evaluations']
    for dt in ('false', 'true'):
        p = G0 / f't_occ_depthTest_{dt}_{bk}.png'
        if not p.exists():
            continue
        m = np.array(Image.open(p).convert('L')) > 128
        cols = np.nonzero(m.any(axis=0))[0]
        inner = cols[3:-3]  # skip the panel's side edges
        d = np.array([np.nonzero(m[:, c])[0].max() + 0.5 - vcol[c] for c in inner])
        frac = (vcol[inner] + 0.5) % 1.0
        ev = impl.get('False' if dt == 'false' else 'True', {})
        smp = np.isin(inner, np.arange(256, 768, 8))  # the implementer's sampled columns
        row = {'cols': int(len(inner)), 'median_abs': float(np.median(np.abs(d))), 'max_abs': float(np.abs(d).max()),
               'signed_mean': float(d.mean()), 'limb_frac_range': [float(frac.min()), float(frac.max())],
               'sampled_cols': int(smp.sum()), 'sampled_median_abs': float(np.median(np.abs(d[smp]))),
               'sampled_max_abs': float(np.abs(d[smp]).max()),
               'implementer_median': ev.get('median_diff_px'), 'implementer_max': ev.get('max_diff_px'),
               'pass_035_075': bool(np.median(np.abs(d)) <= 0.35 and np.abs(d).max() <= 0.75)}
        out[f'{bk}_depthTest_{dt}'] = row
        print(bk, dt, json.dumps(row))
(OUT / 'audit_g0_tocc.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
