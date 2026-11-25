"""Dataset readers for OpenXVS QA contact sheets and statistics.

Supports:
1. G2 results (results_ref.json, results_hw.json and mask directory structures).
2. G3a schema (WP02 §5 frames.jsonl and mask structures).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Iterable

from xvs.paths import get_data_root, REPO_ROOT


@dataclasses.dataclass
class QATileItem:
    """Standardized intruder record for a single camera and instant."""
    instant_id: str
    camera_id: str
    class_id: str
    range_m: float
    background: str
    visibility: float
    mask_px: int
    center_u: float
    center_v: float
    roi: tuple[int, int] | None  # (u0, v0)
    box_analytic: list[float] | None  # [u_min, v_min, u_max, v_max]
    box_px: list[float] | None  # [u_min, v_min, u_max, v_max]
    mask_path: Path | None
    limb_dv_px: float | None
    p_ecef_cam: list[float] | None = None
    r_ecef_cam: list[list[float]] | None = None
    intrinsics: dict[str, Any] | None = None


def find_g2_results_file(input_dir: Path, backend: str) -> tuple[Path, Path]:
    """Find results JSON and base directory for G2 outputs.

    Checks:
    1. input_dir / f"results_{backend}.json"
    2. input_dir with _b1 suffix
    3. XVS_DATA_ROOT or REPO_ROOT output/xvs/wp02/g2_b1 or g2
    """
    data_root = get_data_root()
    candidates = [
        input_dir / f"results_{backend}.json",
        input_dir.parent / f"{input_dir.name}_b1" / f"results_{backend}.json",
        data_root / "xvs" / "wp02" / "g2_b1" / f"results_{backend}.json",
        data_root / "xvs" / "wp02" / "g2" / f"results_{backend}.json",
    ]

    for p in candidates:
        if p.is_file():
            return p, p.parent

    raise FileNotFoundError(
        f"Could not find results_{backend}.json in {input_dir} or candidate locations: {candidates}"
    )


def load_qa_items_from_g2(input_dir: Path | str, backend: str = "ref") -> list[QATileItem]:
    """Load QATileItem records from a G2 run for the specified backend."""
    base_input = Path(input_dir)
    json_path, root_dir = find_g2_results_file(base_input, backend)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items: list[QATileItem] = []
    records = data.get("records", [])

    for r in records:
        instant_id = r.get("instant_id", "")
        class_id = r.get("class_id", "")
        target_range_m = float(r.get("target_range_m", r.get("intruder", {}).get("range_m", 0.0)))  # default-ok: QA tool metadata default
        cams = r.get("cams", {})

        for cam_id, cam in cams.items():
            gt = cam.get("gt", {})
            in_image = bool(gt.get("in_image", False))
            renders = cam.get("renders", {})
            pass_a = renders.get("A", {})
            mask_px = int(pass_a.get("px", 0))  # default-ok: geometry dimension fallback

            # Include tile if in_image is True or mask_px > 0
            if not in_image and mask_px == 0:
                continue

            center_u = float(gt.get("u", 0.0))  # default-ok: QA tool metadata default
            center_v = float(gt.get("v", 0.0))  # default-ok: QA tool metadata default
            roi_raw = cam.get("roi")
            roi = (int(roi_raw[0]), int(roi_raw[1])) if roi_raw else None

            box_analytic = gt.get("box_analytic")
            box_px = pass_a.get("box_px")

            # Visibility calculation: visible_px / amodal_px
            pass_v = renders.get("V", {})
            vis_px = int(pass_v.get("px", 0))  # default-ok: geometry dimension fallback
            if mask_px > 0:
                visibility = float(vis_px) / float(mask_px)
            else:
                visibility = 1.0 if vis_px == 0 else 0.0

            # Mask file path
            rel_png = pass_a.get("png")
            mask_path: Path | None = None
            if rel_png:
                cand = root_dir / rel_png
                if cand.is_file():
                    mask_path = cand
                else:
                    alt = data_root / "xvs" / "wp02" / root_dir.name / rel_png
                    if alt.is_file():
                        mask_path = alt

            limb_dv_px = gt.get("limb_dv_px")
            bg = gt.get("background", "sky")

            items.append(
                QATileItem(
                    instant_id=instant_id,
                    camera_id=cam_id,
                    class_id=class_id,
                    range_m=target_range_m,
                    background=bg,
                    visibility=visibility,
                    mask_px=mask_px,
                    center_u=center_u,
                    center_v=center_v,
                    roi=roi,
                    box_analytic=[float(x) for x in box_analytic] if box_analytic else None,
                    box_px=[float(x) for x in box_px] if box_px else None,
                    mask_path=mask_path,
                    limb_dv_px=float(limb_dv_px) if limb_dv_px is not None else None,
                    p_ecef_cam=cam.get("p_ecef_m"),
                    r_ecef_cam=cam.get("R_ecef_cam"),
                    intrinsics=cam.get("intrinsics"),
                )
            )

    return items


def load_qa_items_from_g3a(input_dir: Path | str) -> list[QATileItem]:
    """Load QATileItem records from G3a output directory (WP02 §5 schema).

    Note: Marked 'untested until G3a data exists' per §2 GEMINI_G-B1.md.
    """
    root_dir = Path(input_dir)
    frames_path = root_dir / "frames.jsonl"
    if not frames_path.is_file():
        raise FileNotFoundError(f"G3a frames.jsonl not found at {frames_path}")

    items: list[QATileItem] = []

    with open(frames_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frame_data = json.loads(line)
            frame_id = str(frame_data.get("frame_id", ""))
            traffic_list = frame_data.get("traffic", [])

            for intruder in traffic_list:
                obj_id = str(intruder.get("id", ""))
                class_id = str(intruder.get("class", ""))
                range_m = float(intruder.get("range_m", 0.0))  # default-ok: QA tool metadata default
                cams = intruder.get("cams", {})

                for cam_id, cam_info in cams.items():
                    if not cam_info.get("in_image", False):
                        continue

                    center_u = float(cam_info.get("u", 0.0))  # default-ok: QA tool metadata default
                    center_v = float(cam_info.get("v", 0.0))  # default-ok: QA tool metadata default
                    roi_raw = cam_info.get("roi")
                    roi = (int(roi_raw[0]), int(roi_raw[1])) if roi_raw else None

                    box_an = cam_info.get("box_analytic")
                    box_px = cam_info.get("box_px") or cam_info.get("box_amodal_px")
                    mask_px = int(cam_info.get("amodal_px", 0))  # default-ok: geometry dimension fallback
                    vis = float(cam_info.get("visibility", 1.0))  # default-ok: QA tool metadata default
                    bg = cam_info.get("background", "sky")
                    limb_dv = cam_info.get("limb_dv_px")

                    rel_mask = cam_info.get("mask")
                    mask_path = (root_dir / rel_mask) if rel_mask else None

                    items.append(
                        QATileItem(
                            instant_id=f"{frame_id}_{obj_id}",
                            camera_id=cam_id,
                            class_id=class_id,
                            range_m=range_m,
                            background=bg,
                            visibility=vis,
                            mask_px=mask_px,
                            center_u=center_u,
                            center_v=center_v,
                            roi=roi,
                            box_analytic=[float(x) for x in box_an] if box_an else None,
                            box_px=[float(x) for x in box_px] if box_px else None,
                            mask_path=mask_path if mask_path and mask_path.is_file() else None,
                            limb_dv_px=float(limb_dv) if limb_dv is not None else None,
                            p_ecef_cam=cam_info.get("p_ecef_m"),
                            r_ecef_cam=cam_info.get("R_ecef_cam"),
                            intrinsics=cam_info.get("intrinsics"),
                        )
                    )

    return items
