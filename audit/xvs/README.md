# OpenXVS audit oracle (Director-owned)

Decision D15 (Batch B2). Charter v1.3: Architecture Lead is both implementer and Director from WP-02 G2 on. Every gate is
therefore re-checked by this independent oracle, plus a separate reviewer pass (D16).

## Purpose

The oracle recomputes the ground truth and the gate metrics from the saved images and results files, using its
own code:
- geodesy and WGS84 frames (`camera_model.py`);
- rig_true regeneration, ring poses, T1 integration and encounter construction (`oracle.py`);
- GLB parser, pinhole projection, dense-sampled Earth limb, coverage estimator and point-sampled rasterizer (`oracle.py`).

It then compares its numbers with the implementer's `results_*.json`. Agreement is evidence that the implementation
is right. A disagreement is a finding.

## Independence rule

- No file in this folder may import the implementation package (`src/xvs`), directly or through a path trick, and
  no module is shared with it. The oracle reads only the spec files (`docs/xvs/spec/`), the assets, and the saved
  outputs.
- `tests/xvs/test_audit_independence.py` fails if any file here contains an import of that package, a `src` path
  insertion or a sandbox path.
- Checks that use the repo code on purpose (a reproduction, a re-render, the repo's own rasterizer) are **not**
  oracle checks. They live in `scripts/xvs/director_checks/`.

## Paths

- The repo root is located from this folder (`docs/xvs/spec/` is found by walking up; override with `XVS_SPEC_DIR`).
- Inputs come from the command line.
- Outputs go to `output/xvs/audit/<gate>/` under the repo root by default; `--out DIR` overrides.
- No absolute or sandbox paths are written in the code.

## Scripts

Run from the repo root with the project venv (`.venv/Scripts/python` on the PC).

| Script | Gate | Command | Output |
|---|---|---|---|
| `audit_g2.py` | WP-02 G2 | `audit/xvs/audit_g2.py output/xvs/wp02/g2 <ref\|hw> [assets_dir]` | `output/xvs/audit/wp02_g2/audit_g2_<backend>.json`: every GT quantity, GT-1…GT-4 and purity recomputed from the PNGs, with the implementer's summary alongside |
| `raster_xor.py` | WP-02 G2 | `audit/xvs/raster_xor.py output/xvs/wp02/g2 <ref\|hw> [assets_dir]` | `output/xvs/audit/wp02_g2/raster_xor_<backend>.json`: every A and S mask vs the oracle's point-sampled rasterization; meshes crossing the camera plane are checked with a frustum half-space test and ray casting |
| `audit_g3a.py` | WP-02 G3a, G3b | `audit/xvs/audit_g3a.py <root> <encounters.jsonl> <n> <seed> [--ownship t1\|recorded] [--extra ENC:K ...]` | `output/xvs/audit/wp02_g3a/audit_g3a_<root name>.json`: on a seeded sample of kept frames, every GT quantity of all 14 cameras and every rendered A mask vs the oracle's rasterization (own near-plane clipping, union-boundary tie test); sampled tracks interpolated with an own cubic Hermite; T2 ownship taken from the record (the oracle has no T2) |
| `compare_g2_runs.py` | WP-02 G2 | `audit/xvs/compare_g2_runs.py <results_a.json> <results_b.json> <out.json>` | per-image `rgb_sha256` identity of two runs and their GT-2 records |
| `audit_g0_tcal.py` | WP-02 G0 | `audit/xvs/audit_g0_tcal.py <g0_dir>` | `output/xvs/audit/wp02_g0/audit_g0_tcal.json`: sphere projections, centroid errors and mask/analytic area ratios |
| `audit_g0_tocc.py` | WP-02 G0 | `audit/xvs/audit_g0_tocc.py <g0_dir>` | `output/xvs/audit/wp02_g0/audit_g0_tocc.json`: panel boundary vs the oracle limb, on all inner columns and on the script's sampled columns |
| `audit_g0_limb_roi.py` | WP-02 G0 | `audit/xvs/audit_g0_limb_roi.py <g0_dir>` | `output/xvs/audit/wp02_g0/audit_g0_limb_roi.json`: T-LIMB-ROI L1 metrics with the oracle's limb and coverage estimator |
| `audit_g1_thickness.py` | WP-02 G1 (D14) | `audit/xvs/audit_g1_thickness.py [assets_dir]` | `output/xvs/audit/wp02_g1/audit_g1_thickness.json`: thickness/chord of every lifting-surface section of the GLBs |
| `audit_g4_gt.py` | WP-01 G4 | `audit/xvs/audit_g4_gt.py output/xvs/wp01/g4/T1_cruise` | `output/xvs/audit/wp01_g4/audit_g4_gt.json`: T1 horizon GT tangency and poses vs the oracle |

Library modules: `oracle.py` (spec loading, poses, trajectories, encounters, GLB, projection, limb, rasterizer,
`REPO_ROOT`, `parse_out`) and `camera_model.py` (WGS84 and frame conversions).

Every table in the reports is printed from these JSON files by `scripts/xvs/report_tables.py audit-g0 | audit-g1 |
audit-g2 | audit-g3a | audit-g4`.

## History

The oracle was written in the Cowork sandbox session (hand-over of 2026-09-24) and adapted to the PC in Batch B1:
- sandbox paths were removed;
- `raster_xor.py` stopped skipping meshes that cross the camera plane;
- the G0 audits read the saved images.

It was committed here in Batch B2 (D15).
