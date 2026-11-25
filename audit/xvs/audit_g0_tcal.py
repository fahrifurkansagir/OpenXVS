"""Director check of G0 T-CAL from the saved centroids, with the independent oracle.
usage: python audit/xvs/audit_g0_tcal.py <g0_dir> [--out DIR]   -> DIR/audit_g0_tcal.json (default output/xvs/audit/wp02_g0)
"""
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import oracle as O
ARGS, OUT = O.parse_out(sys.argv[1:], 'wp02_g0')
G0 = Path(ARGS[0])
SPH = {'nose': (5.5, 0, 0), 'tail': (-5.5, 0, 0), 'right_wingtip': (0, 6.5, 0), 'left_wingtip': (0, -6.5, 0),
       'fin_top': (-4.5, 0, -3.0), 'belly': (0, 0, 1.5)}
RES = {}
for bk in ('ref', 'hw'):
    d = json.load(open(G0 / f'results_{bk}.json', encoding='utf-8'))
    own = dict(p=O.geodetic_to_ecef(37.62, -122.37, 1000.0), R_ecef_body=O.r_ecef_ned(37.62, -122.37) @ O.r_ned_body(0, 0, 0))
    C, R, intr = O.cam_pose(own, 'F')
    Pm = C + 30.0 * R[:, 2]
    errs, area_ratio, dproj = [], [], []
    for pose in d['t_cal']['poses']:
        y, p, r = pose['attitude_ypr_deg']
        Rm = O.r_ecef_ned(37.62, -122.37) @ O.r_ned_body(y, p, r)
        for s in pose['spheres']:
            X = Pm + Rm @ np.array(SPH[s['sphere']], float)
            u, v, z = O.project(C, R, intr, X)
            dproj.append(max(abs(u[0] - s['u_proj']), abs(v[0] - s['v_proj'])))
            errs.append(math.hypot(s['u_centroid'] - u[0], s['v_centroid'] - v[0]))
            # analytic silhouette area of a sphere (radius 0.25 m): exact ellipse area of the projected cone
            dvec = X - C; dist = np.linalg.norm(dvec)
            ca = np.dot(dvec / dist, R[:, 2])      # cos of off-axis angle
            sa = 0.25 / dist                         # sin of cone half-angle
            ta = sa / math.sqrt(1 - sa * sa)
            # ellipse of a circular cone (half-angle a, axis at off-axis angle b) on the image plane z=f:
            # semi-axes a1 = f*sin(a)*cos(a)/(cos^2 b - sin^2 a) , a2 = f*sin(a)/sqrt(cos^2 b - sin^2 a)
            cb2 = ca * ca; sa2 = sa * sa
            a1 = intr.fx * math.sqrt(sa2) * math.sqrt(1 - sa2) / (cb2 - sa2)
            a2 = intr.fx * math.sqrt(sa2) / math.sqrt(cb2 - sa2)
            area = math.pi * a1 * a2
            area_ratio.append(s['pixel_count'] / area)
    RES[bk] = {'proj_diff_max_px': float(max(dproj)), 'centroid_err_median_px': float(np.median(errs)),
               'centroid_err_max_px': float(max(errs)), 'area_ratio_min': float(min(area_ratio)), 'area_ratio_max': float(max(area_ratio)),
               'n_spheres': len(errs)}
    print(bk, 'proj_diff_max', max(dproj), 'centroid err median %.4f max %.4f' % (np.median(errs), max(errs)),
          'area ratio min %.3f max %.3f' % (min(area_ratio), max(area_ratio)))

(OUT / 'audit_g0_tcal.json').write_text(json.dumps(RES, indent=1), encoding='utf-8')
