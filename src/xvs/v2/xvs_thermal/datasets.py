"""License-clean thermal dataset adapters (IRSTD-1k MIT, HIT-UAV CC-BY-4.0) & denylist guard.

Strictly enforces DR-04 §2 & SP2_M2_MODELS_DESIGN.md §3.1/§3.3:
  - Allowed for training & eval: `irstd1k` (MIT), `hit_uav` (CC-BY-4.0) (+ WP-03 synthetic LWIR).
  - Eval-only (NEVER training): `cart`, `massmind`, `uav_cb`, `flir_adas_v2`, `sirst_v1`,
    `segfly`, `kaist`, `llvip`, `irdst`, `m3fd`.
  - Hard-banned (neither training nor eval): `nudt_sirst`, `rgbt_tiny`, `anti_uav`, `vedai`,
    `pst900`, `aerial_uav_thermal_inferred_unified_kaggle`, `thermal_uav_scc_loc`,
    `dronergbt`, `dronevehicle`, `mfnet`, `msrs`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from xvs.sensor.lwir import build_lwir_two_channel_input

_IMAGE_EXTS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})

SplitPurpose = Literal["train", "eval"]

ALLOWED_THERMAL_TRAIN_DATASETS: dict[str, dict[str, str]] = {
    "irstd1k": {
        "name": "IRSTD-1k (Zhang et al., CVPR 2022)",
        "license": "MIT",
        "modality": "thermal_ir",
        "split_policy": "train_and_eval",
    },
    "hit_uav": {
        "name": "HIT-UAV (Suo et al., Sci. Data 2023)",
        "license": "CC-BY-4.0",
        "modality": "aerial_lwir_640x512",
        "split_policy": "train_and_eval",
    },
}

EVAL_ONLY_THERMAL_DATASETS: frozenset[str] = frozenset(
    {
        "cart",
        "massmind",
        "uav_cb",
        "flir_adas_v2",
        "sirst_v1",
        "segfly",
        "kaist",
        "llvip",
        "irdst",
        "m3fd",
    }
)

BANNED_THERMAL_DATASETS: frozenset[str] = frozenset(
    {
        "nudt_sirst",
        "rgbt_tiny",
        "anti_uav",
        "vedai",
        "pst900",
        "aerial_uav_thermal_inferred_unified_kaggle",
        "thermal_uav_scc_loc",
        "dronergbt",
        "dronevehicle",
        "mfnet",
        "msrs",
    }
)


def _normalize_dataset_id(dataset_id: str) -> str:
    return dataset_id.strip().lower().replace("-", "_").replace(" ", "_")


def assert_thermal_dataset_allowed(
    dataset_id: str,
    split_purpose: SplitPurpose = "train",
) -> None:
    """Fail-closed license & scope guard for thermal datasets (DR-04 §2 / SP2 §3.1)."""
    if split_purpose not in ("train", "eval"):
        raise ValueError(f"split_purpose must be 'train' or 'eval', got {split_purpose!r}")
    norm = _normalize_dataset_id(dataset_id)

    # Delegate to central license_guard when available to guarantee single source of truth
    try:
        from xvs.v2.license_guard import assert_dataset_allowed  # type: ignore[import-not-found]
    except ImportError:
        assert_dataset_allowed = None

    if norm in BANNED_THERMAL_DATASETS:
        raise PermissionError(
            f"Thermal dataset {dataset_id!r} is HARD-BANNED (military/C-UAS, GPL/copyleft, "
            f"unlicensed, or NIR spectral mismatch; DR-04 §2 / SP2 §3.1)."
        )
    if norm in EVAL_ONLY_THERMAL_DATASETS:
        if split_purpose == "train":
            raise PermissionError(
                f"Thermal dataset {dataset_id!r} is EVAL-ONLY (NC/proprietary/unclear license); "
                f"training open OpenXVS weights on it is strictly forbidden."
            )
        if assert_dataset_allowed is not None:
            assert_dataset_allowed(norm, split_purpose)
        return
    if norm not in ALLOWED_THERMAL_TRAIN_DATASETS:
        raise PermissionError(
            f"Unknown thermal dataset {dataset_id!r} is not on the OpenXVS v2.0 allowlist."
        )
    if assert_dataset_allowed is not None:
        assert_dataset_allowed(norm, split_purpose)


@dataclass(frozen=True)
class ThermalSample:
    """Standardized 2-channel LWIR sample returned by `Irstd1kDataset` and `HitUavDataset`."""

    dataset_id: str
    sample_id: str
    lwir_2ch: torch.Tensor  # [2, H, W] float32: [norm_radiance, local_contrast]
    mask: torch.Tensor  # [1, H, W] float32 binary target mask
    boxes_xyxy: torch.Tensor  # [N, 4] float32 pixel boxes (x1, y1, x2, y2)


def _resize_binary_mask_peak_preserving(
    mask_2d: torch.Tensor,
    target_hw: tuple[int, int],
) -> torch.Tensor:
    """Resize a 2D binary mask while strictly preserving tiny 2-4 px targets via max-pooling."""
    m = (mask_2d.float() > 0.5).to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    th, tw = int(target_hw[0]), int(target_hw[1])
    nh, nw = int(m.shape[-2]), int(m.shape[-1])
    if (nh, nw) == (th, tw):
        return m[0]
    if nh >= th and nw >= tw:
        return F.adaptive_max_pool2d(m, output_size=(th, tw))[0]
    mid_h = max(nh, th)
    mid_w = max(nw, tw)
    if (mid_h, mid_w) != (nh, nw):
        m = F.interpolate(m, size=(mid_h, mid_w), mode="nearest")
    return F.adaptive_max_pool2d(m, output_size=(th, tw))[0]


def _load_grayscale_tensor(path: Path, target_hw: tuple[int, int] | None = None) -> torch.Tensor:
    with Image.open(path) as im:
        arr = np.asarray(im)
        if arr.ndim == 3:
            arr = arr[..., 0]
        if target_hw is not None and (arr.shape[0], arr.shape[1]) != target_hw:
            pil_g = Image.fromarray(arr)
            pil_g = pil_g.resize((target_hw[1], target_hw[0]), resample=Image.Resampling.BILINEAR)
            arr = np.asarray(pil_g)
    return torch.from_numpy(arr.astype(np.float32))


class Irstd1kDataset(Dataset[ThermalSample]):
    """Pure-Python/torch adapter for IRSTD-1k (MIT License: images + binary target masks)."""

    dataset_id: str = "irstd1k"
    license_id: str = "MIT"

    def __init__(
        self,
        root_dir: Path | str,
        split_purpose: SplitPurpose = "train",
        target_hw: tuple[int, int] = (512, 512),
    ) -> None:
        assert_thermal_dataset_allowed(self.dataset_id, split_purpose=split_purpose)
        self.root_dir = Path(root_dir)
        self.split_purpose = split_purpose
        self.target_hw = target_hw
        img_dir = self.root_dir / "IRSTD1k_Img"
        mask_dir = self.root_dir / "IRSTD1k_Label"
        if not img_dir.is_dir():
            for cand_img, cand_mask in (
                ("images", "masks"),
                ("images", "labels"),
                ("image", "mask"),
                ("image", "label"),
                ("imgs", "masks"),
            ):
                if (self.root_dir / cand_img).is_dir() and (self.root_dir / cand_mask).is_dir():
                    img_dir = self.root_dir / cand_img
                    mask_dir = self.root_dir / cand_mask
                    break
        self.image_paths: list[Path] = (
            sorted(
                p
                for p in img_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
            )
            if img_dir.is_dir()
            else []
        )
        self.mask_dir = mask_dir
        self._mask_by_stem: dict[str, Path] = {}
        if mask_dir.is_dir():
            for mp in sorted(mask_dir.rglob("*")):
                if mp.is_file() and mp.suffix.lower() in _IMAGE_EXTS:
                    self._mask_by_stem[mp.stem] = mp

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> ThermalSample:
        img_path = self.image_paths[index]
        raw = _load_grayscale_tensor(img_path, target_hw=self.target_hw)
        # Scale 8-bit/16-bit grayscale into radiance-equivalent proxy [0, L_sat]
        scale = 65535.0 if float(raw.max().item()) > 255.0 else 255.0
        rad_proxy = (raw / scale) * 215.7
        lwir_2ch = build_lwir_two_channel_input(rad_proxy, l_sat=215.7)

        mask_path: Path | None = self.mask_dir / img_path.name
        if not mask_path.is_file():
            mask_path = self._mask_by_stem[img_path.stem] if img_path.stem in self._mask_by_stem else None
        if mask_path is not None and mask_path.is_file():
            m_raw = _load_grayscale_tensor(mask_path, target_hw=None)
            mask = _resize_binary_mask_peak_preserving(m_raw, self.target_hw)
        else:
            mask = torch.zeros((1, self.target_hw[0], self.target_hw[1]), dtype=torch.float32)

        ys, xs = torch.where(mask[0] > 0.5)
        if ys.numel() > 0:
            boxes = torch.tensor(
                [[float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0]],
                dtype=torch.float32,
            )
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        return ThermalSample(
            dataset_id=self.dataset_id,
            sample_id=img_path.stem,
            lwir_2ch=lwir_2ch,
            mask=mask,
            boxes_xyxy=boxes,
        )


def _index_hit_uav_coco_boxes(
    json_dirs: list[Path],
) -> dict[str, tuple[int, int, list[list[float]]]]:
    """Parse HIT-UAV COCO-style `normal_json/*.json` annotations keyed by image stem."""
    by_stem: dict[str, tuple[int, int, list[list[float]]]] = {}
    for jdir in json_dirs:
        if not jdir.is_dir():
            continue
        for jp in sorted(jdir.rglob("*.json")):
            try:
                doc: Any = json.loads(jp.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(doc, dict) or "images" not in doc or "annotations" not in doc:
                continue
            images_raw = doc["images"]
            anns_raw = doc["annotations"]
            if not isinstance(images_raw, list) or not isinstance(anns_raw, list):
                continue
            img_info: dict[int, tuple[str, int, int]] = {}
            for im in images_raw:
                if not isinstance(im, dict) or "id" not in im or "file_name" not in im:
                    continue
                img_id = int(im["id"])
                stem = Path(str(im["file_name"])).stem
                iw = int(im["width"]) if "width" in im and im["width"] else 640
                ih = int(im["height"]) if "height" in im and im["height"] else 512
                img_info[img_id] = (stem, max(1, iw), max(1, ih))
                if stem not in by_stem:
                    by_stem[stem] = (max(1, iw), max(1, ih), [])
            for ann in anns_raw:
                if not isinstance(ann, dict) or "image_id" not in ann or "bbox" not in ann:
                    continue
                img_id = int(ann["image_id"])
                if img_id not in img_info:
                    continue
                bbox = ann["bbox"]
                if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                    continue
                stem, iw, ih = img_info[img_id]
                bx, by, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                if bw > 0.0 and bh > 0.0:
                    by_stem[stem][2].append([bx, by, bx + bw, by + bh])
    return by_stem


class HitUavDataset(Dataset[ThermalSample]):
    """Pure-Python/torch adapter for HIT-UAV (CC0 / CC-BY-4.0: 640x512 aerial LWIR + YOLO/COCO labels)."""

    dataset_id: str = "hit_uav"
    license_id: str = "CC-BY-4.0"

    def __init__(
        self,
        root_dir: Path | str,
        split_purpose: SplitPurpose = "train",
        target_hw: tuple[int, int] = (512, 640),
    ) -> None:
        assert_thermal_dataset_allowed(self.dataset_id, split_purpose=split_purpose)
        self.root_dir = Path(root_dir)
        self.split_purpose = split_purpose
        self.target_hw = target_hw

        # Locate image directories across standard YOLO (`images/train`, `dataset/images/...`) & Zenodo layouts
        img_roots: list[Path] = []
        for cand in (
            self.root_dir / "images",
            self.root_dir / "dataset" / "images",
            self.root_dir / "normal_images",
            self.root_dir / "train" / "images",
            self.root_dir / "valid" / "images",
        ):
            if cand.is_dir():
                img_roots.append(cand)
        if not img_roots:
            # Fallback: scan root for non-readme image directories
            for p in sorted(self.root_dir.rglob("*")):
                if p.is_dir() and p.name.lower() in ("images", "img", "imgs") and "readme" not in str(p).lower():
                    img_roots.append(p)

        seen_paths: set[Path] = set()
        img_list: list[Path] = []
        for ir in img_roots:
            for p in sorted(ir.rglob("*")):
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                    rp = p.resolve()
                    if rp not in seen_paths:
                        seen_paths.add(rp)
                        img_list.append(p)
        self.image_paths: list[Path] = img_list

        # Index YOLO `.txt` label files recursively (supports `labels/`, `labels/train/`, `yolo_labels/`, etc.)
        self.label_dir = self.root_dir / "labels"
        self._yolo_labels_by_stem: dict[str, Path] = {}
        for l_cand in (
            self.root_dir / "labels",
            self.root_dir / "dataset" / "labels",
            self.root_dir / "yolo_labels",
            self.root_dir / "train" / "labels",
            self.root_dir / "valid" / "labels",
        ):
            if l_cand.is_dir():
                for lp in sorted(l_cand.rglob("*.txt")):
                    if lp.is_file() and lp.stem not in self._yolo_labels_by_stem:
                        self._yolo_labels_by_stem[lp.stem] = lp

        # Index COCO-style `normal_json/*.json` annotations if present
        json_dirs = [
            p
            for p in (
                self.root_dir / "normal_json",
                self.root_dir / "annotations",
                self.root_dir / "dataset" / "normal_json",
            )
            if p.is_dir()
        ]
        self._coco_boxes_by_stem = _index_hit_uav_coco_boxes(json_dirs)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> ThermalSample:
        img_path = self.image_paths[index]
        raw = _load_grayscale_tensor(img_path, target_hw=self.target_hw)
        scale = 65535.0 if float(raw.max().item()) > 255.0 else 255.0
        rad_proxy = (raw / scale) * 215.7
        lwir_2ch = build_lwir_two_channel_input(rad_proxy, l_sat=215.7)

        h, w = self.target_hw
        boxes_list: list[list[float]] = []
        mask = torch.zeros((1, h, w), dtype=torch.float32)

        lbl_path: Path | None = self.label_dir / f"{img_path.stem}.txt"
        if not lbl_path.is_file():
            lbl_path = (
                self._yolo_labels_by_stem[img_path.stem]
                if img_path.stem in self._yolo_labels_by_stem
                else None
            )

        if lbl_path is not None and lbl_path.is_file():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                xc, yc, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                x1 = max(0.0, (xc - 0.5 * bw) * w)
                y1 = max(0.0, (yc - 0.5 * bh) * h)
                x2 = min(float(w), (xc + 0.5 * bw) * w)
                y2 = min(float(h), (yc + 0.5 * bh) * h)
                boxes_list.append([x1, y1, x2, y2])
                mask[0, int(y1) : max(int(y1) + 1, int(y2)), int(x1) : max(int(x1) + 1, int(x2))] = 1.0
        elif img_path.stem in self._coco_boxes_by_stem:
            src_w, src_h, raw_boxes = self._coco_boxes_by_stem[img_path.stem]
            sx = float(w) / float(max(1, src_w))
            sy = float(h) / float(max(1, src_h))
            for bx1, by1, bx2, by2 in raw_boxes:
                x1 = max(0.0, min(float(w - 1), bx1 * sx))
                y1 = max(0.0, min(float(h - 1), by1 * sy))
                x2 = max(x1 + 1.0, min(float(w), bx2 * sx))
                y2 = max(y1 + 1.0, min(float(h), by2 * sy))
                boxes_list.append([x1, y1, x2, y2])
                mask[0, int(y1) : max(int(y1) + 1, int(y2)), int(x1) : max(int(x1) + 1, int(x2))] = 1.0

        boxes = (
            torch.tensor(boxes_list, dtype=torch.float32)
            if boxes_list
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        return ThermalSample(
            dataset_id=self.dataset_id,
            sample_id=img_path.stem,
            lwir_2ch=lwir_2ch,
            mask=mask,
            boxes_xyxy=boxes,
        )
