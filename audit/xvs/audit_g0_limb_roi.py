"""Director check of G0 T-LIMB-ROI (spec v1.1: WP-01 L1 limits) on the saved ROI mask PNGs, with the oracle.

usage: python audit/xvs/audit_g0_limb_roi.py <g0_dir> [--out DIR]   -> DIR/audit_g0_limb_roi.json (default output/xvs/audit/wp02_g0)
Own ring pose, own dense-sampled limb and own coverage estimator (oracle.py); same evaluation as exp_limb_roi.py
(sandbox), but on the PNGs saved by validate_traffic_g0.py instead of a new render. (Engineering Automation, 2026-09-24.)
"""
import sys, json
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
import oracle as O
ARGS, OUT = O.parse_out(sys.argv[1:], 'wp02_g0')
G0 = Path(ARGS[0])
L1 = {'median': 0.15, 'p95': 0.35, 'max': 0.50, 'abs_mean': 0.05}
rows = []
for bk in ('ref', 'hw'):
    impl = {tuple(r['roi']) + (r['h_m'],): r for r in json.loads((G0 / f'results_{bk}.json').read_text(encoding='utf-8'))['t_limb_roi']['results']}
    for h in (600.0, 10000.0):
        own = dict(p=O.geodetic_to_ecef(37.62, -122.37, h), R_ecef_body=O.r_ecef_ned(37.62, -122.37) @ O.r_ned_body(0, 0, 0))
        C, R, intr = O.cam_pose(own, 'T0')
        vfull, _ = O.limb_crossings(C, R, intr, 0, 0, W=2448, H=2048)
        v_l = vfull[1224] if np.isfinite(vfull[1224]) else vfull[1223]
        v0 = int(np.clip(np.round(v_l - 512.0), 0, 1024))
        for u0 in (712, 0):
            img = np.array(Image.open(G0 / f't_limb_roi_{int(h)}m_{u0}_{v0}_{bk}.png'))
            m = (img[:, :, 0] if img.ndim == 3 else img) / 255.0
            vcol, scol = O.limb_crossings(C, R, intr, u0, v0)
            est, skipped = O.coverage_edges(m, vcol, scol)
            d = np.array([e for _, e in est])
            ir = impl[(u0, v0, h)]['l1']
            row = dict(backend=bk, h=h, roi=[u0, v0], used=len(d), skipped=skipped,
                       median=float(np.median(np.abs(d))), p95=float(np.percentile(np.abs(d), 95)),
                       max=float(np.max(np.abs(d))), signed_mean=float(np.mean(d)),
                       impl=[ir['median_abs_delta_px'], ir['p95_abs_delta_px'], ir['max_abs_delta_px'], ir['signed_mean_px'], ir['lines_used']])
            row['pass_L1'] = bool(row['median'] <= L1['median'] and row['p95'] <= L1['p95'] and row['max'] <= L1['max']
                                  and abs(row['signed_mean']) <= L1['abs_mean'])
            rows.append(row)
            print(json.dumps(row))
(OUT / 'audit_g0_limb_roi.json').write_text(json.dumps({'L1': L1, 'rows': rows}, indent=1), encoding='utf-8')
