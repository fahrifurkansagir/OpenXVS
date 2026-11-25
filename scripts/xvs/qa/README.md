# OpenXVS Visual QA Suite

Tooling for visual verification and statistical quality assurance of rendered aircraft traffic targets and ground-truth alignment across camera passes.

---

## 1. Overview

The QA suite enables the Director and perception team to rapidly inspect every rendered intruder target and verify that:
1. Projected geometric ground truth (analytic bounding box, center cross, Earth limb) accurately aligns with the WebGL rendered model.
2. Rasterized amodal and visible masks contain no drift, clipping, or false boundaries.
3. ROI offsets for high-resolution ring cameras are correctly accounted for across all passes.

## 2. Tools

### `make_contact_sheets.py`
Generates high-resolution PNG contact sheets grouped by intruder aircraft `class` $\times$ `range` (e.g. `C1_UAS_4000m`), accompanied by a self-contained static HTML gallery.

**Usage:**
```bash
python scripts/xvs/qa/make_contact_sheets.py --input output/xvs/wp02/g2_b1 --backend ref --out output/xvs/qa/contact_sheets_ref
```

**Options:**
- `--input <path>`: Root directory containing G2 outputs (`results_ref.json` / `results_hw.json`) or G3a sequence outputs (`frames.jsonl`).
- `--backend <ref|hw>`: Render backend (`ref` or `hw`). Default: `ref`.
- `--out <dir>`: Target directory for contact sheets and `index.html`.

**Tile Specifications:**
- **Dimensions**: $128 \times 128$ pixel crop centered on the projected center, upscaled $3\times$ using nearest-neighbor interpolation to $384 \times 384$ pixels.
- **Overlays**:
  - `Analytic Box` (Green): Full projected 3D bounding box edges transformed by camera ROI offset and crop bounds.
  - `Amodal Mask Outline` (Red): Boundary perimeter of rendered amodal mask pixels ($>0$).
  - `Mask box_px` (Yellow): Axis-aligned bounding box computed from non-zero mask pixel coordinates ($\pm 0.5$ px edge convention).
  - `Centre Cross` (Cyan): Projected center $(u, v)$ crosshair.
  - `Earth Limb` (Sky Blue): Analytic horizon curve where the planetary limb traverses the crop window.
- **Label Bar**: Bottom metadata header displaying `instant_id`, `camera_id`, `class_id`, `range_m`, `background` (`sky` or `earth`), `visibility` ratio, and `mask_px` pixel count.
- **Sheet Composition**: Maximum $6 \times 6$ tiles per PNG sheet with unified header. Groups with $>36$ tiles automatically paginate.

### `qa_stats.py`
Computes dataset-level distribution metrics and produces diagnostic charts.

**Usage:**
```bash
python scripts/xvs/qa/qa_stats.py --input output/xvs/wp02/g2_b1 --backend ref --out output/xvs/qa/stats_ref
```

**Outputs:**
- `qa_stats.json`: Quantitative breakdown per class and range bin (count, min/median/mean/max analytic box span in pixels, visibility mean/median, fraction of empty masks).
- `plot_range_vs_span.png`: Log-log plot of intruder range ($m$) versus analytic bounding box span ($px$) by aircraft class.
- `plot_visibility_hist.png`: Histogram distribution of intruder visibility fractions ($\text{visible\_px} / \text{amodal\_px}$).

### `reader.py`
Unified data adapter converting both G2 execution records and G3a WP02 §5 stream records into canonical `QATileItem` representations.

---

## 3. Dataset Support

- **G2 Validation Sets**: Reads `results_ref.json` / `results_hw.json` and associated mask PNG files under `<backend>/masks/<instant>/<cam>_A.png`.
- **G3a WP02 §5 Schema**: Reads `frames.jsonl` with full intruder traffic definitions (`cams`, `traffic`, `roi`, `amodal_px`, `box_amodal_px`). Marked *untested until G3a data exists* if run before G3a renders.
