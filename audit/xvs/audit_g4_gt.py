"""Director check of the G4 T1 horizon GT (frames.jsonl) with the independent oracle.
usage: python audit/xvs/audit_g4_gt.py <T1 sequence dir> [--out DIR]   -> DIR/audit_g4_gt.json (default output/xvs/audit/wp01_g4)
(2026-09-24, Engineering Automation on the PC: sandbox paths replaced by argv; result also saved as JSON.)
"""
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import oracle as O
ARGS, OUT = O.parse_out(sys.argv[1:], 'wp01_g4')
SEQ = Path(ARGS[0]); F = SEQ / 'frames.jsonl'
frames = [json.loads(l) for l in open(F, encoding='utf-8')]
print('frames', len(frames))
S = np.array([1 / O.A, 1 / O.A, 1 / O.B])
worst_pose, worst_tan_px, n_pts, worst_state = 0, 0, 0, 0
vis_mismatch = []
for fr in frames[::10]:
    t = fr['t_s']
    o = O.t1_state(t)
    worst_state = max(worst_state, np.abs(o['p'] - np.array(fr['aircraft']['p_ecef_m'])).max())
    for cid, c in fr['cams'].items():
        C, R, intr = O.cam_pose(o, cid)
        worst_pose = max(worst_pose, np.abs(C - np.array(c['p_ecef_m'])).max(), np.abs(R - np.array(c['R_ecef_cam'])).max())
        pts = c['gt']['horizon_uv']
        # visibility: does the analytic limb cross the image?
        vcol, _ = O.limb_crossings(C, R, intr, n=400000)
        vis_oracle = bool(np.isfinite(vcol).any())
        if vis_oracle != c['gt']['horizon_visible'] and cid not in ('U',):
            vis_mismatch.append((fr['k'], cid, vis_oracle, c['gt']['horizon_visible']))
        for seg in (pts if pts and isinstance(pts[0][0], list) else [pts]):
            for u, v in seg:
                d = R @ np.array([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, 1.0])
                c_s, d_s = C * S, d * S
                d_s /= np.linalg.norm(d_s)
                # distance from origin to the scaled ray line; tangent <=> 1
                dist = np.linalg.norm(np.cross(c_s, d_s))
                # angular error ~ (dist-1)/sqrt(|c|^2-1)  -> pixels via fx
                ang = (dist - 1.0) / math.sqrt(c_s @ c_s - 1.0)
                worst_tan_px = max(worst_tan_px, abs(ang) * intr.fx)
                n_pts += 1
res = {'frames_checked': len(frames[::10]), 'state_max_m': worst_state, 'pose_max': worst_pose, 'n_gt_points': n_pts,
       'max_tangency_error_px': worst_tan_px, 'visibility_mismatch': vis_mismatch[:10]}
print(json.dumps(res, indent=1))
(OUT / 'audit_g4_gt.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
