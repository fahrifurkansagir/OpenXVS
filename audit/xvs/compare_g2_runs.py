"""Compare the per-image rgb_sha256 of two G2 results files, and their GT-2 raster records.

usage: python audit/xvs/compare_g2_runs.py <results_a.json> <results_b.json> <out.json>
Used for: PC ref vs the sandbox (Linux SwiftShader) dry run; hw before vs after the near-plane clipping change.
(Engineering Automation, 2026-09-24.)
"""
import json, sys
from pathlib import Path
a = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')); b = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
bm = {r['instant_id']: r for r in b['records']}
n = eq = 0; missing = 0
for rec in a['records']:
    o = bm.get(rec['instant_id'])
    if o is None:
        missing += 1; continue
    for cid, c in rec['cams'].items():
        for ps, rr in c['renders'].items():
            oo = o['cams'].get(cid, {}).get('renders', {}).get(ps)
            if oo is None:
                continue
            n += 1; eq += rr['rgb_sha256'] == oo['rgb_sha256']
out = {'a': sys.argv[1], 'b': sys.argv[2], 'images_compared': n, 'rgb_sha256_identical': eq, 'instants_missing_in_b': missing}
ga = a.get('checks', {}).get('gt2', {}).get('records'); gb = b.get('checks', {}).get('gt2', {}).get('records')
if ga and gb and 'computable' in ga[0] and 'computable' in gb[0]:
    key = lambda r: (r['instant_id'], r['cam'], r['pass'])
    B = {key(r): r for r in gb}
    fields = ('ref_px', 'diff_px', 'tie_px', 'mismatch_px', 'gpu_px', 'max_diff_edge_distance_px')
    same, changed = 0, []
    for r in ga:
        s = B.get(key(r))
        if s and r['computable'] and s['computable'] and all(r.get(f) == s.get(f) for f in fields):
            same += 1
        else:
            changed.append({'key': list(key(r)), 'a_computable': r['computable'], 'b_computable': s and s['computable'],
                            'b_near_clipped': s and s.get('near_clipped'), 'b_ref_px': s and s.get('ref_px'),
                            'b_diff_px': s and s.get('diff_px'), 'b_gpu_px': s and s.get('gpu_px')})
    out['gt2_records'] = {'n': len(ga), 'identical': same, 'different': changed}
Path(sys.argv[3]).parent.mkdir(parents=True, exist_ok=True)
Path(sys.argv[3]).write_text(json.dumps(out, indent=1), encoding='utf-8')
print(json.dumps(out, indent=1))
