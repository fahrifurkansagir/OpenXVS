"""Export verified Gate-G1 D1-KF (`d1_kf_k3_synth.pt`) to FP32 and FP16 ONNX for OpenXVS Live Deck.

Wraps `D1HeatmapDetector.forward_normalized` ONLY (never double-normalizing `forward()`),
exports opset 17 (`dynamo=False`) static `[1, 5, 256, 256]` graphs, converts FP32 -> FP16 using
pure `onnx` + `numpy` (zero extra dependencies, `<= 512,000` bytes), verifies numerical parity
against PyTorch CPU `forward_normalized` on synthetic + recorded Gate-G1 G3a + flat zero-contrast
tiles, and writes `apps/livedeck/public/models/registry.json` + `output/xvs/v2_m1/d1_onnx_parity.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
from PIL import Image
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.detection.d1_model import D1HeatmapDetector, build_d1_kf

DEFAULT_CHECKPOINT_PATH = (
    REPO_ROOT / "output" / "xvs" / "wp04" / "g1" / "checkpoints" / "d1_kf_k3_synth.pt"
)
DEFAULT_MODELS_DIR = REPO_ROOT / "apps" / "livedeck" / "public" / "models"
DEFAULT_PARITY_OUT = REPO_ROOT / "output" / "xvs" / "v2_m1" / "d1_onnx_parity.json"
DEFAULT_G3A_ROOT = REPO_ROOT / "output" / "xvs" / "wp02" / "g3a"

FP32_FILENAME = "d1_kf_k3_fp32.onnx"
FP16_FILENAME = "d1_kf_k3_fp16.onnx"
REGISTRY_FILENAME = "registry.json"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class HierarchicalGroupNorm(nn.Module):
    """Exact mathematical equivalent of `nn.GroupNorm` using 3-axis hierarchical reductions.

    Avoids TorchScript's flat `InstanceNormalization` rewrite (which sums up to 1,048,576 float32
    elements in a single accumulator and loses ~5e-3 precision on 256x256 tiles) while exporting
    100% standard opset-17 `Reshape` + `ReduceMean` ops supported natively on ORT CPU & WebGPU.
    """

    def __init__(self, gn: nn.GroupNorm) -> None:
        super().__init__()
        self.num_groups = int(gn.num_groups)
        self.num_channels = int(gn.num_channels)
        self.eps = float(gn.eps)
        self.register_buffer("weight", gn.weight.detach().clone().view(1, self.num_channels, 1, 1))
        self.register_buffer("bias", gn.bias.detach().clone().view(1, self.num_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = int(x.shape[2])
        w = int(x.shape[3])
        c_per_g = self.num_channels // self.num_groups
        xg = x.reshape(1, self.num_groups, c_per_g, h, w)
        mean = xg.mean(dim=2, keepdim=True).mean(dim=4, keepdim=True).mean(dim=3, keepdim=True)
        diff = xg - mean
        var = (diff * diff).mean(dim=2, keepdim=True).mean(dim=4, keepdim=True).mean(dim=3, keepdim=True)
        y = (diff / torch.sqrt(var + self.eps)).reshape(1, self.num_channels, h, w)
        return y * self.weight + self.bias


class D1NormalizedExportWrapper(nn.Module):
    """Thin export wrapper calling `D1HeatmapDetector.forward_normalized` ONLY."""

    def __init__(self, detector: D1HeatmapDetector) -> None:
        super().__init__()
        self.detector = detector
        self.gn_enc1 = HierarchicalGroupNorm(detector.enc1.gn)
        self.gn_enc2_1 = HierarchicalGroupNorm(detector.enc2.gn1)
        self.gn_enc2_2 = HierarchicalGroupNorm(detector.enc2.gn2)
        self.gn_bot_1 = HierarchicalGroupNorm(detector.bottleneck.gn1)
        self.gn_bot_2 = HierarchicalGroupNorm(detector.bottleneck.gn2)
        self.gn_dec2_1 = HierarchicalGroupNorm(detector.dec2.gn1)
        self.gn_dec2_2 = HierarchicalGroupNorm(detector.dec2.gn2)
        self.gn_dec1_1 = HierarchicalGroupNorm(detector.dec1.gn1)
        self.gn_dec1_2 = HierarchicalGroupNorm(detector.dec1.gn2)

    def _run_convblock(
        self,
        cb: nn.Module,
        gn1: HierarchicalGroupNorm,
        gn2: HierarchicalGroupNorm,
        x: torch.Tensor,
    ) -> torch.Tensor:
        y = cb.act1(gn1(cb.conv1(x)))
        y = gn2(cb.conv2(y)) + cb.skip(x)
        return cb.act2(y)

    def forward(
        self, x_norm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.detector.enc1
        f1 = self.gn_enc1(
            e1.act3(e1.conv3(e1.act2(e1.conv2(e1.act1(e1.conv1(x_norm))))))
        )
        f2 = self._run_convblock(
            self.detector.enc2, self.gn_enc2_1, self.gn_enc2_2, self.detector.down1(f1)
        )
        fb = self._run_convblock(
            self.detector.bottleneck, self.gn_bot_1, self.gn_bot_2, self.detector.down2(f2)
        )
        u2 = torch.nn.functional.interpolate(
            self.detector.up2(fb), size=f2.shape[-2:], mode="bilinear", align_corners=False
        )
        d2 = self._run_convblock(
            self.detector.dec2, self.gn_dec2_1, self.gn_dec2_2, torch.cat([u2, f2], dim=1)
        )
        u1 = torch.nn.functional.interpolate(
            self.detector.up1(d2), size=f1.shape[-2:], mode="bilinear", align_corners=False
        )
        feat = self._run_convblock(
            self.detector.dec1, self.gn_dec1_1, self.gn_dec1_2, torch.cat([u1, f1], dim=1)
        )
        hm_logits = self.detector.head_heatmap(feat)
        hm = torch.sigmoid(hm_logits)
        offset = torch.tanh(self.detector.head_offset(feat))
        log_scale = self.detector.head_scale(feat).clamp(min=-1.0, max=5.0)
        return hm, offset, log_scale


def load_gate_g1_detector(ckpt_path: Path = DEFAULT_CHECKPOINT_PATH) -> tuple[D1HeatmapDetector, dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_d1_kf(
        num_frames_k=int(ckpt["num_frames_k"]),
        ablation_mode=str(ckpt["ablation_mode"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device="cpu", dtype=torch.float32)
    return model, ckpt


def _srgb_u8_to_linear_e(rgb_u8: np.ndarray, full_well_e: float = 10500.0) -> np.ndarray:
    """Convert sRGB uint8 [H, W, 3] -> Rec.709 linear luminance scaled to photoelectrons."""
    s = rgb_u8.astype(np.float64) / 255.0
    lin = np.where(s <= 0.04045, s / 12.92, np.power((s + 0.055) / 1.055, 2.4))
    luma = 0.2126 * lin[:, :, 0] + 0.7152 * lin[:, :, 1] + 0.0722 * lin[:, :, 2]
    return (luma * float(full_well_e)).astype(np.float32)


def _load_recorded_g3a_fixture(g3a_root: Path = DEFAULT_G3A_ROOT) -> np.ndarray:
    """Load a deterministic 3-frame recorded Gate-G1 G3a ring-camera ROI stack [1, 5, 256, 256]."""
    b0_frames = g3a_root / "blocks" / "b00of05" / "frames.jsonl"
    if b0_frames.is_file():
        by_enc_cam: dict[tuple[str, str], list[tuple[int, Path, tuple[float, float]]]] = {}
        with b0_frames.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                eid = str(obj["encounter_id"])
                k_idx = int(obj["k"])
                traffic_list = obj["traffic"] if "traffic" in obj else []
                for tr in traffic_list:
                    cams = tr["cams"] if "cams" in tr else {}
                    for cid, cdata in cams.items():
                        if (
                            cdata.get("kind") == "ring"
                            and cdata.get("in_image")
                            and cdata.get("rgb")
                            and cdata.get("box_analytic") is not None
                        ):
                            png_p = g3a_root / str(cdata["rgb"])
                            if not png_p.is_file():
                                png_p = b0_frames.parent / str(cdata["rgb"])
                            if png_p.is_file():
                                box = cdata["box_analytic"]
                                cu = 0.5 * (float(box[0]) + float(box[2]))
                                cv = 0.5 * (float(box[1]) + float(box[3]))
                                by_enc_cam.setdefault((eid, cid), []).append((k_idx, png_p, (cu, cv)))

        for key in sorted(by_enc_cam.keys()):
            seq = sorted(by_enc_cam[key], key=lambda item: item[0])
            if len(seq) >= 3:
                # Pick 3 frames near the middle of the encounter
                mid = len(seq) // 2
                triplet = seq[max(0, mid - 2) : max(0, mid - 2) + 3]
                if len(triplet) == 3:
                    _, _, (cu, cv) = triplet[-1]
                    crops: list[np.ndarray] = []
                    for _, png_p, _ in triplet:
                        with Image.open(png_p) as im:
                            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
                        h_img, w_img, _ = rgb.shape
                        x0 = int(max(0, min(w_img - 256, round(cu - 128.0))))
                        y0 = int(max(0, min(h_img - 256, round(cv - 128.0))))
                        crops.append(_srgb_u8_to_linear_e(rgb[y0 : y0 + 256, x0 : x0 + 256]))
                    f_m2, f_m1, f_0 = crops[0], crops[1], crops[2]
                    stack = np.stack([f_0, f_m1, f_0 - f_m1, f_m2, f_0 - f_m2], axis=0)
                    return stack[np.newaxis, :, :, :].astype(np.float32)

    # Fallback: scan direct rgb/T0/*.png if frames.jsonl is absent
    t0_dir = g3a_root / "rgb" / "T0"
    pngs = sorted(t0_dir.glob("*.png")) if t0_dir.is_dir() else []
    if len(pngs) >= 3:
        crops = []
        for p in pngs[:3]:
            with Image.open(p) as im:
                rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
            h_img, w_img, _ = rgb.shape
            y0 = max(0, (h_img - 256) // 2)
            x0 = max(0, (w_img - 256) // 2)
            crops.append(_srgb_u8_to_linear_e(rgb[y0 : y0 + 256, x0 : x0 + 256]))
        f_m2, f_m1, f_0 = crops[0], crops[1], crops[2]
        stack = np.stack([f_0, f_m1, f_0 - f_m1, f_m2, f_0 - f_m2], axis=0)
        return stack[np.newaxis, :, :, :].astype(np.float32)

    raise FileNotFoundError(f"No recorded Gate-G1 G3a ring PNG fixtures found under {g3a_root}")


def build_parity_fixtures(
    model: D1HeatmapDetector,
    g3a_root: Path = DEFAULT_G3A_ROOT,
) -> dict[str, np.ndarray]:
    """Build deterministic pre-normalized `[1, 5, 256, 256]` float32 fixtures for ONNX parity tests."""
    rng = np.random.default_rng(20261120)
    yy, xx = np.meshgrid(
        np.arange(256, dtype=np.float32),
        np.arange(256, dtype=np.float32),
        indexing="ij",
    )

    # 1. Deterministic synthetic moving small-target fixture (electron scale -> normalize_input)
    sky_bg = 3200.0 + 180.0 * (yy / 255.0)
    noise_0 = rng.normal(0.0, 45.0, size=(256, 256)).astype(np.float32)
    noise_1 = rng.normal(0.0, 45.0, size=(256, 256)).astype(np.float32)
    noise_2 = rng.normal(0.0, 45.0, size=(256, 256)).astype(np.float32)

    tgt_0 = 2600.0 * np.exp(-((xx - 134.3) ** 2 + (yy - 118.7) ** 2) / (2.0 * 1.15**2))
    tgt_1 = 2400.0 * np.exp(-((xx - 131.8) ** 2 + (yy - 117.9) ** 2) / (2.0 * 1.15**2))
    tgt_2 = 2200.0 * np.exp(-((xx - 129.3) ** 2 + (yy - 117.1) ** 2) / (2.0 * 1.15**2))

    i0 = (sky_bg + noise_0 + tgt_0).astype(np.float32)
    i1 = (sky_bg + noise_1 + tgt_1).astype(np.float32)
    i2 = (sky_bg + noise_2 + tgt_2).astype(np.float32)
    synth_raw = np.stack([i0, i1, i0 - i1, i2, i0 - i2], axis=0)[np.newaxis, :, :, :]

    # 2. Recorded Gate-G1 G3a ROI fixture
    g3a_raw = _load_recorded_g3a_fixture(g3a_root)

    # 3. Flat zero-contrast tile (Review Focus #4: std0 == 0 clamped to 1.0 -> all zeros)
    flat_raw = np.zeros((1, 5, 256, 256), dtype=np.float32)

    with torch.no_grad():
        synth_norm = model.normalize_input(torch.from_numpy(synth_raw)).numpy().astype(np.float32)
        g3a_norm = model.normalize_input(torch.from_numpy(g3a_raw)).numpy().astype(np.float32)
        flat_norm = model.normalize_input(torch.from_numpy(flat_raw)).numpy().astype(np.float32)

    return {
        "synthetic_point_target": synth_norm,
        "g3a_recorded_roi": g3a_norm,
        "flat_zero_contrast": flat_norm,
    }


def _collect_resize_non_data_inputs(graph: onnx.GraphProto) -> set[str]:
    """Identify Resize ROI/scales tensor names that ONNX opset 17 requires to stay FLOAT32."""
    keep_f32: set[str] = set()
    for node in graph.node:
        if node.op_type == "Resize":
            # Resize inputs: [X, roi, scales, sizes] -> roi (1) and scales (2) MUST remain FLOAT32
            for idx in (1, 2):
                if idx < len(node.input) and node.input[idx]:
                    keep_f32.add(node.input[idx])
    return keep_f32


def convert_fp32_onnx_to_fp16(fp32_model: onnx.ModelProto) -> onnx.ModelProto:
    """Convert an FP32 ONNX model to FP16 weights & internal Conv activations using pure onnx+numpy.

    Stores all Conv weights/biases in FLOAT16 (`<= 512 KB`), runs all Conv/Pad/Concat/Resize/
    activation layers in FLOAT16, keeps `HierarchicalGroupNorm` (`/gn_*`) reduction/variance ops
    and final output heads (`head_heatmap`, `head_offset`, `head_scale`) numerically tight so
    FP16 max abs diff stays `< 2e-3` and peak centroid stays `< 0.05 px` while file size stays
    well under `512,000` bytes.
    """
    model = onnx.ModelProto()
    model.CopyFrom(fp32_model)
    graph = model.graph

    # Build tensor dtype map from shape inference on the original FP32 model
    inferred = onnx.shape_inference.infer_shapes(fp32_model)
    tensor_dtype: dict[str, int] = {}
    for vi in list(inferred.graph.input) + list(inferred.graph.output) + list(inferred.graph.value_info):
        if vi.type.HasField("tensor_type"):
            tensor_dtype[vi.name] = vi.type.tensor_type.elem_type
    for init in fp32_model.graph.initializer:
        tensor_dtype[init.name] = init.data_type

    # Identify nodes that stay in FP32:
    # - GroupNorm variance/mean blocks (`/gn_`),
    # - Directional second-difference wavelet stem `enc1` (`/conv1/`, `/conv2/`, `/conv3/`, `/act1..3/`)
    #   where exact sky-gradient step cancellation requires FP32 weights,
    # - Resize non-data inputs (`roi`/`scales`).
    # The entire U-Net encoder/bottleneck/decoder/heads (`down1`, `enc2`, `down2`, `bottleneck`,
    # `up2`, `dec2`, `up1`, `dec1`, `head_*`) run in FLOAT16 (`<= 512,000` B, `< 2e-3` max diff)!
    keep_f32_tensors = _collect_resize_non_data_inputs(graph)
    fp32_node_indices: set[int] = set()
    for idx, node in enumerate(graph.node):
        if node.op_type == "Constant":
            continue
        lname = node.name.lower()
        is_gn = "/gn_" in lname
        is_stem = any(
            lname.startswith(p)
            for p in ("/conv1/", "/conv2/", "/conv3/", "/act1/", "/act2/", "/act3/")
        )
        is_norm_op = node.op_type in {
            "GroupNormalization",
            "InstanceNormalization",
            "ReduceMean",
            "Sqrt",
            "Div",
        }
        if is_gn or is_stem or is_norm_op:
            fp32_node_indices.add(idx)

    # Initializers/Constants consumed ONLY by FP32 nodes stay FP32; all other FLOAT initializers
    # (Conv weights, biases, etc.) are converted to FLOAT16!
    tensor_consumers: dict[str, set[int]] = {}
    for idx, node in enumerate(graph.node):
        for inp in node.input:
            if inp:
                tensor_consumers.setdefault(inp, set()).add(idx)

    def _is_f32_only_tensor(tname: str) -> bool:
        if tname in keep_f32_tensors:
            return True
        consumers = tensor_consumers.get(tname)
        return bool(consumers) and consumers.issubset(fp32_node_indices)

    # Convert FLOAT initializers to FLOAT16 unless used strictly by FP32 nodes / Resize scales
    init_is_f16: set[str] = set()
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT and not _is_f32_only_tensor(init.name):
            arr = numpy_helper.to_array(init)
            arr16 = np.clip(arr, -65504.0, 65504.0).astype(np.float16)
            init.CopyFrom(numpy_helper.from_array(arr16, name=init.name))
            init_is_f16.add(init.name)

    for node in graph.node:
        if node.op_type == "Constant" and node.output and not _is_f32_only_tensor(node.output[0]):
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT:
                    arr = numpy_helper.to_array(attr.t)
                    arr16 = np.clip(arr, -65504.0, 65504.0).astype(np.float16)
                    attr.t.CopyFrom(numpy_helper.from_array(arr16))
                    init_is_f16.add(node.output[0])

    # Track live dtype of every tensor produced so far (FLOAT or FLOAT16 for float tensors)
    live_dtype: dict[str, int] = dict(tensor_dtype)
    for name in init_is_f16:
        live_dtype[name] = TensorProto.FLOAT16

    orig_in_name = graph.input[0].name
    live_dtype[orig_in_name] = TensorProto.FLOAT
    out_orig_names = [o.name for o in graph.output]
    out_internal_map = {n: f"{n}_pre_out_cast" for n in out_orig_names}

    new_nodes: list[onnx.NodeProto] = []
    cast_cache: dict[tuple[str, int], str] = {}
    cast_counter = 0

    def _ensure_dtype(tname: str, target_dt: int) -> str:
        nonlocal cast_counter
        if not tname:
            return tname
        cur_dt = live_dtype.get(tname, TensorProto.FLOAT)  # default-ok: fallback for uninferred float tensors
        if cur_dt not in (TensorProto.FLOAT, TensorProto.FLOAT16) or cur_dt == target_dt:
            return tname
        key = (tname, target_dt)
        if key in cast_cache:
            return cast_cache[key]
        suffix = "f16" if target_dt == TensorProto.FLOAT16 else "f32"
        cname = f"{tname}__cast_{suffix}_{cast_counter}"
        cast_counter += 1
        new_nodes.append(
            helper.make_node("Cast", [tname], [cname], name=f"cast_{suffix}_{cast_counter}", to=target_dt)
        )
        live_dtype[cname] = target_dt
        cast_cache[key] = cname
        return cname

    for idx, node in enumerate(graph.node):
        if node.op_type == "Constant":
            new_nodes.append(node)
            continue

        node_dt = TensorProto.FLOAT if idx in fp32_node_indices else TensorProto.FLOAT16

        if node.op_type == "Cast":
            orig_in = node.input[0]
            new_out = out_internal_map.get(node.output[0], node.output[0])
            target_to = None
            for attr in node.attribute:
                if attr.name == "to":
                    if attr.i == TensorProto.FLOAT and not _is_f32_only_tensor(node.output[0]):
                        attr.i = node_dt
                    target_to = attr.i
            node.ClearField("input")
            node.input.append(orig_in)
            node.ClearField("output")
            node.output.append(new_out)
            if target_to is not None:
                live_dtype[new_out] = target_to
            new_nodes.append(node)
            continue

        new_inputs: list[str] = []
        for in_idx, inp in enumerate(node.input):
            if not inp:
                new_inputs.append(inp)
                continue
            if node.op_type == "Resize" and in_idx in (1, 2):
                new_inputs.append(_ensure_dtype(inp, TensorProto.FLOAT))
            else:
                orig_dt = tensor_dtype.get(inp, TensorProto.FLOAT)  # default-ok: fallback for uninferred float tensors
                if orig_dt == TensorProto.FLOAT:
                    new_inputs.append(_ensure_dtype(inp, node_dt))
                else:
                    new_inputs.append(inp)

        new_outputs: list[str] = []
        for out_n in node.output:
            mapped = out_internal_map.get(out_n, out_n)
            orig_dt = tensor_dtype.get(out_n, TensorProto.FLOAT)  # default-ok: fallback for uninferred float tensors
            live_dtype[mapped] = node_dt if orig_dt == TensorProto.FLOAT else orig_dt
            new_outputs.append(mapped)

        node.ClearField("input")
        node.input.extend(new_inputs)
        node.ClearField("output")
        node.output.extend(new_outputs)
        new_nodes.append(node)

    for out_name in out_orig_names:
        internal_name = out_internal_map[out_name]
        cur_dt = live_dtype.get(internal_name, TensorProto.FLOAT)  # default-ok: output default float
        if cur_dt == TensorProto.FLOAT:
            new_nodes.append(
                helper.make_node("Identity", [internal_name], [out_name], name=f"id_out_{out_name}")
            )
        else:
            new_nodes.append(
                helper.make_node(
                    "Cast",
                    [internal_name],
                    [out_name],
                    name=f"cast_output_{out_name}_fp32",
                    to=TensorProto.FLOAT,
                )
            )

    graph.ClearField("node")
    graph.node.extend(new_nodes)
    graph.ClearField("value_info")
    onnx.checker.check_model(model)
    return model


def _peak_centroid_uv(hm: np.ndarray, offset: np.ndarray) -> tuple[float, float]:
    h = hm[0, 0].copy()
    h[:2, :] = -1.0
    h[-2:, :] = -1.0
    h[:, :2] = -1.0
    h[:, -2:] = -1.0
    flat_idx = int(np.argmax(h))
    vy, ux = divmod(flat_idx, h.shape[1])
    return float(ux) + float(offset[0, 0, vy, ux]), float(vy) + float(offset[0, 1, vy, ux])


def export_and_verify(
    ckpt_path: Path = DEFAULT_CHECKPOINT_PATH,
    models_dir: Path = DEFAULT_MODELS_DIR,
    parity_out: Path = DEFAULT_PARITY_OUT,
    g3a_root: Path = DEFAULT_G3A_ROOT,
) -> dict[str, Any]:
    models_dir.mkdir(parents=True, exist_ok=True)
    parity_out.parent.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_gate_g1_detector(ckpt_path)
    wrapper = D1NormalizedExportWrapper(model).eval()

    fp32_path = models_dir / FP32_FILENAME
    fp16_path = models_dir / FP16_FILENAME
    registry_path = models_dir / REGISTRY_FILENAME

    dummy_x = torch.zeros((1, 5, 256, 256), dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_x,),
            str(fp32_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["x_norm"],
            output_names=["heatmap", "offset", "log_scale"],
            dynamo=False,
        )

    fp32_onnx = onnx.load(str(fp32_path))
    onnx.checker.check_model(fp32_onnx)

    fp16_onnx = convert_fp32_onnx_to_fp16(fp32_onnx)
    onnx.save(fp16_onnx, str(fp16_path))
    onnx.checker.check_model(onnx.load(str(fp16_path)))

    # Verify numerical parity across synthetic + recorded G3a + flat zero-contrast tiles
    fixtures = build_parity_fixtures(model, g3a_root=g3a_root)
    sess_f32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    sess_f16 = ort.InferenceSession(str(fp16_path), providers=["CPUExecutionProvider"])

    fixture_reports: dict[str, Any] = {}
    max_fp32_diff = 0.0
    max_fp16_diff = 0.0
    max_fp16_centroid_px = 0.0

    for name, x_norm_np in fixtures.items():
        with torch.no_grad():
            pt_out = model.forward_normalized(torch.from_numpy(x_norm_np))
        pt_hm = pt_out["heatmap"].numpy()
        pt_off = pt_out["offset"].numpy()
        pt_lsc = pt_out["log_scale"].numpy()

        f32_hm, f32_off, f32_lsc = sess_f32.run(None, {"x_norm": x_norm_np})
        f16_hm, f16_off, f16_lsc = sess_f16.run(None, {"x_norm": x_norm_np})

        d32_hm = float(np.max(np.abs(f32_hm - pt_hm)))
        d32_off = float(np.max(np.abs(f32_off - pt_off)))
        d32_lsc = float(np.max(np.abs(f32_lsc - pt_lsc)))
        d32_max = max(d32_hm, d32_off, d32_lsc)

        d16_hm = float(np.max(np.abs(f16_hm - pt_hm)))
        d16_off = float(np.max(np.abs(f16_off - pt_off)))
        d16_lsc = float(np.max(np.abs(f16_lsc - pt_lsc)))
        d16_max = max(d16_hm, d16_off, d16_lsc)

        max_fp32_diff = max(max_fp32_diff, d32_max)
        max_fp16_diff = max(max_fp16_diff, d16_max)

        centroid_diff_px = 0.0
        if name != "flat_zero_contrast":
            pt_u, pt_v = _peak_centroid_uv(pt_hm, pt_off)
            f16_u, f16_v = _peak_centroid_uv(f16_hm, f16_off)
            centroid_diff_px = float(math.hypot(f16_u - pt_u, f16_v - pt_v))
            max_fp16_centroid_px = max(max_fp16_centroid_px, centroid_diff_px)

        fixture_reports[name] = {
            "fp32_max_abs_diff": d32_max,
            "fp32_heatmap_max_abs_diff": d32_hm,
            "fp32_offset_max_abs_diff": d32_off,
            "fp32_log_scale_max_abs_diff": d32_lsc,
            "fp16_max_abs_diff": d16_max,
            "fp16_heatmap_max_abs_diff": d16_hm,
            "fp16_offset_max_abs_diff": d16_off,
            "fp16_log_scale_max_abs_diff": d16_lsc,
            "fp16_peak_centroid_diff_px": centroid_diff_px,
            "all_finite": bool(
                np.all(np.isfinite(f32_hm))
                and np.all(np.isfinite(f32_off))
                and np.all(np.isfinite(f32_lsc))
                and np.all(np.isfinite(f16_hm))
                and np.all(np.isfinite(f16_off))
                and np.all(np.isfinite(f16_lsc))
            ),
        }

    fp32_bytes = int(fp32_path.stat().st_size)
    fp16_bytes = int(fp16_path.stat().st_size)
    fp32_sha = file_sha256(fp32_path)
    fp16_sha = file_sha256(fp16_path)
    ckpt_sha = file_sha256(ckpt_path)

    all_passed = bool(
        max_fp32_diff < 1e-5
        and max_fp16_diff < 2e-3
        and max_fp16_centroid_px < 0.05
        and fp16_bytes <= 512_000
        and all(r["all_finite"] for r in fixture_reports.values())
    )

    registry = {
        "schema": "xvs.livedeck.model_registry/1",
        "active_model_id": "d1_kf_k3_g1",
        "models": [
            {
                "id": "d1_kf_k3_g1",
                "display_name": "Gate-G1 D1-KF (K=3, Colab L4)",
                "status": "active",
                "milestone": "V2-M1",
                "source_checkpoint": "output/xvs/wp04/g1/checkpoints/d1_kf_k3_synth.pt",
                "source_checkpoint_sha256": ckpt_sha,
                "tau_star": float(ckpt["tau_star"]),
                "opset": 17,
                "input": {
                    "name": "x_norm",
                    "dtype": "float32",
                    "shape": [1, 5, 256, 256],
                },
                "outputs": [
                    {"name": "heatmap", "dtype": "float32", "shape": [1, 1, 256, 256]},
                    {"name": "offset", "dtype": "float32", "shape": [1, 2, 256, 256]},
                    {"name": "log_scale", "dtype": "float32", "shape": [1, 2, 256, 256]},
                ],
                "preprocessing": {
                    "forward_normalized_only": True,
                    "ring_intrinsics": "traffic_ring_v1",
                    "fx_px": 2319.0,
                    "fy_px": 2319.0,
                    "kept_frame_step_dt_s": 1.0,
                    "full_well_e": 10500.0,
                    "std0_clamp_min": 1.0,
                    "norm_clamp_range": [-20.0, 20.0],
                },
                "artifacts": {
                    "fp32": {
                        "path": f"models/{FP32_FILENAME}",
                        "bytes": fp32_bytes,
                        "sha256": fp32_sha,
                    },
                    "fp16": {
                        "path": f"models/{FP16_FILENAME}",
                        "bytes": fp16_bytes,
                        "sha256": fp16_sha,
                    },
                },
            },
            {
                "id": "xvs_d2_web_v2",
                "display_name": "XVS-D2-Web (Reserved M2 Slot)",
                "status": "reserved_m2",
                "milestone": "V2-M2",
                "notes": "Reserved drop-in slot for Colab G4 96 GB M2 multi-scale detector.",
            },
        ],
    }
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")

    parity_payload = {
        "schema": "xvs.v2_m1.d1_onnx_parity/1",
        "checkpoint_path": "output/xvs/wp04/g1/checkpoints/d1_kf_k3_synth.pt",
        "checkpoint_sha256": ckpt_sha,
        "opset": 17,
        "forward_normalized_only": True,
        "fp32_onnx_path": f"apps/livedeck/public/models/{FP32_FILENAME}",
        "fp32_onnx_bytes": fp32_bytes,
        "fp32_onnx_sha256": fp32_sha,
        "fp16_onnx_path": f"apps/livedeck/public/models/{FP16_FILENAME}",
        "fp16_onnx_bytes": fp16_bytes,
        "fp16_onnx_sha256": fp16_sha,
        "registry_sha256": file_sha256(registry_path),
        "fp32_max_abs_diff": max_fp32_diff,
        "fp16_max_abs_diff": max_fp16_diff,
        "fp16_max_centroid_diff_px": max_fp16_centroid_px,
        "thresholds": {
            "fp32_max_abs_diff_lt": 1e-5,
            "fp16_max_abs_diff_lt": 2e-3,
            "fp16_max_centroid_diff_px_lt": 0.05,
            "fp16_max_bytes_le": 512_000,
        },
        "fixtures": fixture_reports,
        "all_checks_passed": all_passed,
    }
    parity_out.write_text(json.dumps(parity_payload, indent=2) + "\n", encoding="utf-8")

    if not all_passed:
        raise RuntimeError(
            f"D1 ONNX parity check FAILED: fp32_diff={max_fp32_diff:.3e}, "
            f"fp16_diff={max_fp16_diff:.3e}, fp16_centroid={max_fp16_centroid_px:.4f} px, "
            f"fp16_bytes={fp16_bytes}"
        )
    return parity_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Gate-G1 D1-KF forward_normalized to FP32/FP16 ONNX")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--parity-out", type=Path, default=DEFAULT_PARITY_OUT)
    args = parser.parse_args()

    res = export_and_verify(
        ckpt_path=args.checkpoint,
        models_dir=args.models_dir,
        parity_out=args.parity_out,
    )
    print(
        f"Exported D1-KF ONNX OK: FP32={res['fp32_onnx_bytes']} B (max_diff={res['fp32_max_abs_diff']:.2e}), "
        f"FP16={res['fp16_onnx_bytes']} B (max_diff={res['fp16_max_abs_diff']:.2e}, "
        f"centroid_diff={res['fp16_max_centroid_diff_px']:.4f} px)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
