"""Generate visual QA contact sheets for OpenXVS traffic renders.

Usage:
  python scripts/xvs/qa/make_contact_sheets.py --input <g2 or g3a root> --backend <ref|hw> --out <dir>

Conforms to docs/xvs/batches/GEMINI_G-B1.md §2:
- One tile per (instant or frame, camera, pass A):
  - 128x128 crop centred on projected centre, upscaled 3x with nearest-neighbour (384x384).
  - Overlays: analytic box (green), amodal mask outline (red), mask box_px (yellow),
    centre cross, and horizon line where the limb crosses the crop.
  - Label: instant id, camera, class, range, background, visibility, mask px.
- Sheets grouped by class x range: PNG grids of at most 6x6 tiles.
- Static index.html linking the sheets with zero external scripts/fonts.
"""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add src to path if needed
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.xvs.qa.reader import QATileItem, load_qa_items_from_g2, load_qa_items_from_g3a

try:
    from xvs.rig.camera import PinholeCamera
    from xvs.sim import horizon
    HAS_XVS_SIM = True
except ImportError:
    HAS_XVS_SIM = False


# Colors for overlays (RGB)
COLOR_ANALYTIC_BOX = (0, 255, 0)       # Green
COLOR_MASK_OUTLINE = (255, 0, 0)       # Red
COLOR_MASK_BOX_PX  = (255, 230, 0)     # Yellow
COLOR_CENTER_CROSS = (0, 255, 255)     # Cyan
COLOR_HORIZON_LINE = (50, 150, 255)    # Sky blue


def render_single_tile(item: QATileItem) -> Image.Image:
    """Render a single 128x128 crop upscaled 3x to 384x384 with overlays and label."""
    tile_img_size = 384
    label_h = 56
    total_w = tile_img_size
    total_h = tile_img_size + label_h

    # ROI offset
    u0, v0 = item.roi if item.roi else (0, 0)

    # Centre in mask image coordinates
    u_c_img = item.center_u - u0
    v_c_img = item.center_v - v0

    # 128x128 crop bounding box in mask image coords
    u_crop_min = int(round(u_c_img)) - 64
    v_crop_min = int(round(v_c_img)) - 64
    u_crop_max = u_crop_min + 128
    v_crop_max = v_crop_min + 128

    # Load mask if available
    mask_arr = np.zeros((128, 128), dtype=np.uint8)
    if item.mask_path and item.mask_path.is_file():
        try:
            full_mask = Image.open(item.mask_path).convert("L")
            full_arr = np.array(full_mask)
            f_h, f_w = full_arr.shape

            # Overlap with full image
            src_u1 = max(0, u_crop_min)
            src_v1 = max(0, v_crop_min)
            src_u2 = min(f_w, u_crop_max)
            src_v2 = min(f_h, v_crop_max)

            if src_u2 > src_u1 and src_v2 > src_v1:
                dst_u1 = src_u1 - u_crop_min
                dst_v1 = src_v1 - v_crop_min
                dst_u2 = dst_u1 + (src_u2 - src_u1)
                dst_v2 = dst_v1 + (src_v2 - src_v1)
                mask_arr[dst_v1:dst_v2, dst_u1:dst_u2] = full_arr[src_v1:src_v2, src_u1:src_u2]
        except Exception:
            pass

    # Create 384x384 image from 128x128 crop using nearest-neighbour
    # Render background: dark grey (20, 20, 25) with mask pixels shown in white (240, 240, 240)
    rgb_crop = np.zeros((128, 128, 3), dtype=np.uint8)
    rgb_crop[:, :] = [25, 27, 32]
    mask_pixels = mask_arr > 0
    rgb_crop[mask_pixels] = [235, 235, 240]

    pil_crop = Image.fromarray(rgb_crop, mode="RGB")
    tile_crop_3x = pil_crop.resize((tile_img_size, tile_img_size), Image.Resampling.NEAREST)

    # Full tile with label bar
    tile = Image.new("RGB", (total_w, total_h), (18, 19, 23))
    tile.paste(tile_crop_3x, (0, 0))

    draw = ImageDraw.Draw(tile)

    # Helper coordinate transformer from full image coords to tile crop coords
    def to_tile_coords(u_full: float, v_full: float) -> tuple[float, float]:
        u_tile = ((u_full - u0) - u_crop_min) * 3.0
        v_tile = ((v_full - v0) - v_crop_min) * 3.0
        return (u_tile, v_tile)

    # 1. Overlay: Amodal mask outline (Red)
    # Find boundary pixels in the 128x128 mask
    if np.any(mask_pixels):
        outline = np.zeros((128, 128), dtype=bool)
        # Shift in 4 directions
        up = np.pad(mask_pixels[1:, :], ((0, 1), (0, 0)), constant_values=False)
        down = np.pad(mask_pixels[:-1, :], ((1, 0), (0, 0)), constant_values=False)
        left = np.pad(mask_pixels[:, 1:], ((0, 0), (0, 1)), constant_values=False)
        right = np.pad(mask_pixels[:, :-1], ((0, 0), (1, 0)), constant_values=False)
        outline = mask_pixels & (~(up & down & left & right))

        for vy, ux in zip(*np.where(outline)):
            tx = ux * 3
            ty = vy * 3
            draw.rectangle([tx, ty, tx + 2, ty + 2], outline=COLOR_MASK_OUTLINE)

    # 2. Overlay: Analytic box (Green)
    if item.box_analytic:
        bx1, by1 = to_tile_coords(item.box_analytic[0], item.box_analytic[1])
        bx2, by2 = to_tile_coords(item.box_analytic[2], item.box_analytic[3])
        draw.rectangle([bx1, by1, bx2, by2], outline=COLOR_ANALYTIC_BOX, width=2)

    # 3. Overlay: Mask box_px (Yellow)
    if item.box_px:
        # box_px is [u_min, v_min, u_max, v_max] over pixel centres, ±0.5 for outer edges
        px1, py1 = to_tile_coords(item.box_px[0] - 0.5, item.box_px[1] - 0.5)
        px2, py2 = to_tile_coords(item.box_px[2] + 0.5, item.box_px[3] + 0.5)
        draw.rectangle([px1, py1, px2, py2], outline=COLOR_MASK_BOX_PX, width=1)

    # 4. Overlay: Centre Cross (Cyan)
    cx, cy = to_tile_coords(item.center_u, item.center_v)
    arm = 9
    draw.line([(cx - arm, cy), (cx + arm, cy)], fill=COLOR_CENTER_CROSS, width=2)
    draw.line([(cx, cy - arm), (cx, cy + arm)], fill=COLOR_CENTER_CROSS, width=2)

    # 5. Overlay: Horizon line where limb crosses the crop (Sky Blue)
    horizon_points: list[tuple[float, float]] = []
    # If sim model available, evaluate exact curve across columns
    if HAS_XVS_SIM and item.p_ecef_cam and item.r_ecef_cam and item.intrinsics:
        try:
            cam_model = PinholeCamera(
                width=item.intrinsics.get("width", 1024),  # default-ok: geometry dimension fallback
                height=item.intrinsics.get("height", 1024),  # default-ok: geometry dimension fallback
                fx=item.intrinsics.get("fx", 429.6),  # default-ok: QA tool metadata default
                fy=item.intrinsics.get("fy", 429.6),  # default-ok: QA tool metadata default
                cx=item.intrinsics.get("cx", 511.5),  # default-ok: QA tool metadata default
                cy=item.intrinsics.get("cy", 511.5),  # default-ok: QA tool metadata default
            )
            p_cam = np.array(item.p_ecef_cam, dtype=np.float64)
            r_cam = np.array(item.r_ecef_cam, dtype=np.float64)

            for col_crop in range(0, 128, 4):
                u_full = u0 + u_crop_min + col_crop
                roots = horizon.analytic_horizon_v_at_u(p_cam, r_cam, cam_model, float(u_full))
                if len(roots) >= 1:
                    v_full = roots[0]
                    tx, ty = to_tile_coords(u_full, v_full)
                    if -20 <= ty <= tile_img_size + 20:
                        horizon_points.append((tx, ty))
        except Exception:
            pass

    # Fallback to limb_dv_px if available and no points gathered
    if not horizon_points and item.limb_dv_px is not None:
        v_limb = item.center_v - item.limb_dv_px
        x1, y1 = to_tile_coords(u0 + u_crop_min, v_limb)
        x2, y2 = to_tile_coords(u0 + u_crop_max, v_limb)
        if -20 <= y1 <= tile_img_size + 20 or -20 <= y2 <= tile_img_size + 20:
            horizon_points = [(x1, y1), (x2, y2)]

    if len(horizon_points) >= 2:
        draw.line(horizon_points, fill=COLOR_HORIZON_LINE, width=2)

    # Label text in bottom bar
    font = ImageFont.load_default()
    l1 = f"{item.instant_id}  cam:{item.camera_id}  class:{item.class_id}"
    l2 = f"rng:{item.range_m:.0f}m  bg:{item.background}  vis:{item.visibility:.2f}  mask_px:{item.mask_px}"

    draw.rectangle([0, tile_img_size, total_w, total_h], fill=(28, 30, 36))
    draw.line([(0, tile_img_size), (total_w, tile_img_size)], fill=(48, 52, 62), width=1)
    draw.text((8, tile_img_size + 8), l1, fill=(230, 235, 245), font=font)
    draw.text((8, tile_img_size + 28), l2, fill=(160, 175, 195), font=font)

    return tile


def make_contact_sheets(
    input_dir: Path | str,
    out_dir: Path | str,
    backend: str = "ref",
    max_grid: int = 6,
) -> list[Path]:
    """Generate contact sheets grouped by class x range and index.html."""
    in_path = Path(input_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Detect dataset format
    if (in_path / "frames.jsonl").is_file():
        items = load_qa_items_from_g3a(in_path)
    else:
        items = load_qa_items_from_g2(in_path, backend=backend)

    if not items:
        raise ValueError(f"No QA intruder items loaded from {in_path}")

    # Group by class_id and range (approximate rounded range)
    grouped: dict[tuple[str, int], list[QATileItem]] = {}
    for it in items:
        r_rounded = int(round(it.range_m))
        grouped.setdefault((it.class_id, r_rounded), []).append(it)

    sheet_paths: list[Path] = []
    sheet_metadata: list[dict[str, Any]] = []

    # Sort groups by class then range
    for (cls_id, rng_m), group_items in sorted(grouped.items(), key=lambda k: (k[0][0], k[0][1])):
        # Sort items inside group by instant_id, camera_id
        group_items.sort(key=lambda x: (x.instant_id, x.camera_id))

        max_tiles_per_sheet = max_grid * max_grid
        n_sheets = math.ceil(len(group_items) / max_tiles_per_sheet)

        for s_idx in range(n_sheets):
            sub_items = group_items[s_idx * max_tiles_per_sheet : (s_idx + 1) * max_tiles_per_sheet]
            n_tiles = len(sub_items)

            # Determine grid columns and rows (up to 6x6)
            cols = min(n_tiles, max_grid)
            rows = math.ceil(n_tiles / cols)

            # Render tiles
            tiles = [render_single_tile(it) for it in sub_items]
            t_w, t_h = tiles[0].size

            sheet_w = cols * t_w
            # Include a top header banner for the sheet (60 px)
            header_h = 60
            sheet_h = rows * t_h + header_h

            sheet = Image.new("RGB", (sheet_w, sheet_h), (12, 13, 16))
            draw = ImageDraw.Draw(sheet)

            # Draw header
            header_text = f"OpenXVS Visual QA — Class: {cls_id} | Range: {rng_m} m | Backend: {backend} | Sheet {s_idx + 1}/{n_sheets} ({n_tiles} tiles)"
            draw.rectangle([0, 0, sheet_w, header_h], fill=(22, 24, 30))
            draw.line([(0, header_h), (sheet_w, header_h)], fill=(40, 44, 54), width=1)
            font = ImageFont.load_default()
            draw.text((16, 20), header_text, fill=(240, 245, 255), font=font)

            # Paste tiles into grid
            for idx, tile in enumerate(tiles):
                r = idx // cols
                c = idx % cols
                sheet.paste(tile, (c * t_w, header_h + r * t_h))

            # Save sheet
            sheet_filename = f"sheet_{cls_id}_{rng_m}m_s{s_idx + 1:02d}.png"
            sheet_file = out_path / sheet_filename
            sheet.save(sheet_file)
            sheet_paths.append(sheet_file)

            sheet_metadata.append({
                "filename": sheet_filename,
                "class_id": cls_id,
                "range_m": rng_m,
                "sheet_num": s_idx + 1,
                "total_sheets": n_sheets,
                "tile_count": n_tiles,
                "cols": cols,
                "rows": rows,
            })

    # Generate static index.html
    html_content = generate_index_html(backend, sheet_metadata, len(items))
    with open(out_path / "index.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    return sheet_paths


def generate_index_html(backend: str, metadata: list[dict[str, Any]], total_items: int) -> str:
    """Generate clean self-contained static HTML gallery."""
    classes = sorted(list({m["class_id"] for m in metadata}))
    ranges = sorted(list({m["range_m"] for m in metadata}))

    rows_html = []
    for m in metadata:
        fname = html.escape(m["filename"])
        cls_esc = html.escape(m["class_id"])
        rows_html.append(
            f"""
            <div class="sheet-card">
              <div class="card-header">
                <h3>{cls_esc} — {m['range_m']} m (Sheet {m['sheet_num']}/{m['total_sheets']})</h3>
                <span class="badge">{m['tile_count']} tiles</span>
              </div>
              <a href="{fname}" target="_blank">
                <img src="{fname}" alt="{fname}" loading="lazy" class="sheet-thumb" />
              </a>
              <div class="card-footer">
                <a href="{fname}" download class="btn-link">Download Full Sheet PNG</a>
              </div>
            </div>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>OpenXVS Dataset Visual QA Contact Sheets ({backend})</title>
  <style>
    :root {{
      --bg: #0e1014;
      --card-bg: #181a20;
      --border: #282c35;
      --text: #e6ebf5;
      --text-muted: #9aa4b5;
      --accent: #3b82f6;
      --accent-hover: #60a5fa;
      --badge-bg: #222733;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      padding: 24px 32px;
      line-height: 1.5;
    }}
    header {{
      margin-bottom: 24px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px;
    }}
    h1 {{ font-size: 24px; font-weight: 600; margin-bottom: 6px; }}
    .subtitle {{ color: var(--text-muted); font-size: 14px; }}
    .legend {{
      display: flex;
      gap: 16px;
      margin-top: 14px;
      font-size: 13px;
    }}
    .legend-item {{ display: flex; align-items: center; gap: 6px; }}
    .color-box {{ width: 14px; height: 14px; border-radius: 2px; }}
    .c-green {{ background: #00ff00; }}
    .c-red {{ background: #ff0000; }}
    .c-yellow {{ background: #ffe600; }}
    .c-cyan {{ background: #00ffff; }}
    .c-blue {{ background: #3296ff; }}
    .stats-bar {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      padding: 12px 18px;
      border-radius: 6px;
      margin-bottom: 24px;
      display: flex;
      gap: 32px;
      font-size: 14px;
    }}
    .stats-bar strong {{ color: #ffffff; }}
    .gallery-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(540px, 1fr));
      gap: 24px;
    }}
    .sheet-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }}
    .card-header {{
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .card-header h3 {{ font-size: 15px; font-weight: 600; }}
    .badge {{
      background: var(--badge-bg);
      color: var(--accent-hover);
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 12px;
      font-weight: 500;
    }}
    .sheet-thumb {{
      width: 100%;
      height: auto;
      display: block;
      transition: opacity 0.2s;
    }}
    .sheet-thumb:hover {{ opacity: 0.92; }}
    .card-footer {{
      padding: 10px 16px;
      border-top: 1px solid var(--border);
      font-size: 13px;
    }}
    .btn-link {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 500;
    }}
    .btn-link:hover {{ text-decoration: underline; color: var(--accent-hover); }}
  </style>
</head>
<body>
  <header>
    <h1>OpenXVS Dataset Visual QA Gallery</h1>
    <p class="subtitle">Direct inspection sheets for rendered intruders and ground-truth alignment (Backend: <strong>{backend}</strong>)</p>
    <div class="legend">
      <div class="legend-item"><div class="color-box c-green"></div> Analytic Box</div>
      <div class="legend-item"><div class="color-box c-red"></div> Mask Outline</div>
      <div class="legend-item"><div class="color-box c-yellow"></div> Mask box_px</div>
      <div class="legend-item"><div class="color-box c-cyan"></div> Centre Cross</div>
      <div class="legend-item"><div class="color-box c-blue"></div> Earth Limb</div>
    </div>
  </header>

  <div class="stats-bar">
    <div>Total Intruder Tiles: <strong>{total_items}</strong></div>
    <div>Total Contact Sheets: <strong>{len(metadata)}</strong></div>
    <div>Classes: <strong>{len(classes)}</strong> ({", ".join(classes)})</div>
    <div>Ranges: <strong>{len(ranges)}</strong> ({", ".join(str(r) + "m" for r in ranges)})</div>
  </div>

  <div class="gallery-grid">
    {''.join(rows_html)}
  </div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visual QA contact sheets.")
    parser.add_argument("--input", type=str, default="output/xvs/wp02/g2_b1", help="Path to G2 or G3a output directory")
    parser.add_argument("--backend", type=str, choices=["ref", "hw"], default="ref", help="Render backend (ref|hw)")
    parser.add_argument("--out", type=str, default="output/xvs/qa/contact_sheets", help="Output directory")
    args = parser.parse_args()

    print(f"Generating contact sheets: input={args.input}, backend={args.backend}, out={args.out}")
    sheets = make_contact_sheets(args.input, args.out, backend=args.backend)
    print(f"Done. Generated {len(sheets)} contact sheets and index.html in {args.out}")


if __name__ == "__main__":
    main()
