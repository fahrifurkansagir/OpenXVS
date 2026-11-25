"""Compute visual QA dataset statistics and generate diagnostic plots.

Usage:
  python scripts/xvs/qa/qa_stats.py --input <g2 or g3a root> --backend <ref|hw> --out <dir>

Outputs:
  - <out>/qa_stats.json: Per class and range counts, analytic spans, visibility, and empty mask fraction.
  - <out>/plot_range_vs_span.png: Scatter/trend of Range vs Analytic Box Span per class.
  - <out>/plot_visibility_hist.png: Histogram of visibility distribution across the dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.xvs.qa.reader import QATileItem, load_qa_items_from_g2, load_qa_items_from_g3a


def compute_qa_statistics(items: list[QATileItem]) -> dict[str, Any]:
    """Compute per class and range statistics from loaded QATileItems."""
    # Group items by (class_id, range_m)
    groups: dict[tuple[str, int], list[QATileItem]] = {}
    for it in items:
        r_rounded = int(round(it.range_m))
        groups.setdefault((it.class_id, r_rounded), []).append(it)

    breakdown: dict[str, dict[str, Any]] = {}
    all_spans = []
    all_visibilities = []
    all_ranges = []
    all_classes = []

    for (cls_id, rng_m), group_items in sorted(groups.items(), key=lambda k: (k[0][0], k[0][1])):
        count = len(group_items)
        spans = []
        visibilities = []
        empty_masks = 0

        for it in group_items:
            visibilities.append(it.visibility)
            all_visibilities.append(it.visibility)
            all_ranges.append(it.range_m)
            all_classes.append(it.class_id)

            if it.mask_px == 0:
                empty_masks += 1

            if it.box_analytic:
                w = max(0.0, it.box_analytic[2] - it.box_analytic[0])
                h = max(0.0, it.box_analytic[3] - it.box_analytic[1])
                span = max(w, h)
                spans.append(span)
                all_spans.append(span)

        empty_fraction = empty_masks / count if count > 0 else 0.0

        group_key = f"{cls_id}_{rng_m}m"
        breakdown[group_key] = {
            "class_id": cls_id,
            "range_m": rng_m,
            "count": count,
            "empty_mask_count": empty_masks,
            "empty_mask_fraction": round(empty_fraction, 4),
            "analytic_span_px": {
                "mean": round(float(np.mean(spans)), 3) if spans else None,
                "median": round(float(np.median(spans)), 3) if spans else None,
                "min": round(float(np.min(spans)), 3) if spans else None,
                "max": round(float(np.max(spans)), 3) if spans else None,
                "std": round(float(np.std(spans)), 3) if spans else None,
            },
            "visibility": {
                "mean": round(float(np.mean(visibilities)), 4) if visibilities else None,
                "median": round(float(np.median(visibilities)), 4) if visibilities else None,
                "min": round(float(np.min(visibilities)), 4) if visibilities else None,
                "max": round(float(np.max(visibilities)), 4) if visibilities else None,
            },
        }

    summary = {
        "total_items": len(items),
        "total_groups": len(breakdown),
        "overall_empty_mask_count": sum(1 for it in items if it.mask_px == 0),
        "overall_empty_mask_fraction": round(sum(1 for it in items if it.mask_px == 0) / len(items), 4) if items else 0.0,
        "breakdown": breakdown,
    }

    return summary


def plot_range_vs_span(items: list[QATileItem], out_path: Path) -> None:
    """Generate scatter / trend plot of range vs analytic box span per class."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

    # Palette for aircraft classes
    classes = sorted(list({it.class_id for it in items}))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(classes), 1)))

    for c_idx, cls_id in enumerate(classes):
        c_items = [it for it in items if it.class_id == cls_id and it.box_analytic is not None]
        ranges = [it.range_m for it in c_items]
        spans = [max(1e-3, max(it.box_analytic[2] - it.box_analytic[0], it.box_analytic[3] - it.box_analytic[1])) for it in c_items]

        # Group by unique ranges for mean lines
        unique_r = sorted(list(set(ranges)))
        mean_spans = [float(np.mean([s for r, s in zip(ranges, spans) if r == u_r])) for u_r in unique_r]

        ax.scatter(ranges, spans, color=colors[c_idx], label=cls_id, alpha=0.6, edgecolors="none", s=25)
        ax.plot(unique_r, mean_spans, color=colors[c_idx], linestyle="--", linewidth=1.5, alpha=0.85)

    ax.set_title("Target Range vs Analytic Box Span by Class", fontsize=14, pad=12)
    ax.set_xlabel("Range (m)", fontsize=11)
    ax.set_ylabel("Analytic Box Span (px)", fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(title="Class", frameon=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_visibility_hist(items: list[QATileItem], out_path: Path) -> None:
    """Generate histogram of visibility distribution."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    visibilities = [it.visibility for it in items]
    bins = np.linspace(0.0, 1.0, 21)

    counts, edges, patches = ax.hist(
        visibilities,
        bins=bins,
        color="#3b82f6",
        edgecolor="#1e293b",
        linewidth=1.2,
        rwidth=0.85,
    )

    ax.set_title("Intruder Target Visibility Distribution (Visible / Amodal)", fontsize=13, pad=12)
    ax.set_xlabel("Visibility Ratio", fontsize=11)
    ax.set_ylabel("Camera-Instant Count", fontsize=11)
    ax.set_xticks(np.linspace(0.0, 1.0, 11))
    ax.grid(True, axis="y", linestyle=":", alpha=0.7)

    # Annotate summary counts
    mean_vis = float(np.mean(visibilities)) if visibilities else 0.0
    median_vis = float(np.median(visibilities)) if visibilities else 0.0
    ax.axvline(mean_vis, color="#ef4444", linestyle="--", linewidth=1.5, label=f"Mean: {mean_vis:.3f}")
    ax.axvline(median_vis, color="#10b981", linestyle=":", linewidth=1.5, label=f"Median: {median_vis:.3f}")
    ax.legend(frameon=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_qa_stats(
    input_dir: Path | str,
    out_dir: Path | str,
    backend: str = "ref",
) -> Path:
    """Run full QA statistics computation and plot generation."""
    in_path = Path(input_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if (in_path / "frames.jsonl").is_file():
        items = load_qa_items_from_g3a(in_path)
    else:
        items = load_qa_items_from_g2(in_path, backend=backend)

    if not items:
        raise ValueError(f"No QA items loaded from {in_path}")

    # 1. Compute stats & save JSON
    stats_data = compute_qa_statistics(items)
    stats_data["backend"] = backend
    stats_data["input_dir"] = str(in_path)

    json_file = out_path / "qa_stats.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2)

    # 2. Plots
    plot_range_vs_span(items, out_path / "plot_range_vs_span.png")
    plot_visibility_hist(items, out_path / "plot_visibility_hist.png")

    return json_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute visual QA stats and plots.")
    parser.add_argument("--input", type=str, default="output/xvs/wp02/g2_b1", help="Path to G2 or G3a output directory")
    parser.add_argument("--backend", type=str, choices=["ref", "hw"], default="ref", help="Render backend (ref|hw)")
    parser.add_argument("--out", type=str, default="output/xvs/qa/stats", help="Output directory")
    args = parser.parse_args()

    print(f"Running QA stats: input={args.input}, backend={args.backend}, out={args.out}")
    json_path = run_qa_stats(args.input, args.out, backend=args.backend)
    print(f"Done. Statistics saved to {json_file_path(json_path)}")


def json_file_path(p: Path) -> str:
    return str(p)


if __name__ == "__main__":
    main()
