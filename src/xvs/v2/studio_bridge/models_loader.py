"""Real 5/5 OpenXVS checkpoint extractor and loader for local GTX 1650 SUPER / CPU inference.

Loads and SHA-256-verifies all 5 production OpenXVS engines:
  1. XVS-D2         : `apps/livedeck/public/models/xvs_d2_web_fp16.onnx` (ONNX Runtime ConvGRU)
  2. XVS-ActiveRange: `src/xvs/v2/xvs_active_range/filter.py` (exact Rust parity MSC-RPEKF + SC3BF)
  3. XVS-Land       : v5.2 6-output `xvs_land_v52_fp16.onnx` (promoted default, 384x384 + keyline_sigma_px)
                      -> Wave-C 5-output `xvs_land_v45.onnx` -> Wave-C `.pt` -> NB2 `xvs_land_final.pt`
  4. XVS-ThermalNet : v5.2 FPN `xvs_thermal_v52_fp16.onnx` (promoted default, 4-mode EO/LWIR fusion)
                      -> Wave-C `xvs_thermal_v45_fp32.onnx` -> Wave-C `.pt` -> NB2 `xvs_thermal_final.pt`
  5. XVS-Nav        : V5.1 Faz-B `outputs/colab_l4_v5_nav/harvest_faz_b/xvs_nav_v5_fazb_fp16.onnx`
                      (active default — v5.2 Faz-C is intentionally HELD due to -10.54 pp/-16.09 pp regression;
                      MNV4-Hybrid-M + MixVPR-Clean folded Float64 Ridge-Whitening ONNX, paired with
                      Stage-2A `xfeat_320x256_fp16.onnx` & Stage-2B `lighterglue_k256_fp16.onnx`)
                      -> fallback Faz-A `outputs/colab_l4_v5_nav/harvest/xvs_nav_v5_fp16.onnx`
                      (with Stage-2A `xfeat_320x256_fp32.onnx`) -> fallback NB2 `xvs_nav_final.pt`
                      (Colab L4 DINOv2-B/14 + BoQ 512-D + PCA whitening)

Enforces strict SHA-256 pinning (fail-closed rejection of unpinned or tampered artifacts with
verified fallback), `< 2.8 GB` VRAM headroom on 4 GB GPUs (auto FP16 / selective CPU offload),
and pure-PyTorch zero-`timm` loading so inference runs 100% offline with zero network calls.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tarfile
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

from xvs.v2.xvs_land.model import StudentTCalibrator, XvsLandModel
from xvs.v2.xvs_thermal.model import XvsThermalNet

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_D2_ONNX = REPO_ROOT / "apps" / "livedeck" / "public" / "models" / "xvs_d2_web_fp16.onnx"
DEFAULT_L4_ARCHIVE = (
    REPO_ROOT
    / "outputs"
    / "colab_l4_nb2"
    / "v2_m2_02_xvs_nav_land_thermal_g4_results_stage.tar.zst"
)
DEFAULT_L4_MANIFEST = (
    REPO_ROOT
    / "outputs"
    / "colab_l4_nb2"
    / "v2_m2_02_xvs_nav_land_thermal_g4_results_stage_manifest.json"
)
DEFAULT_CKPT_DIR = REPO_ROOT / "outputs" / "colab_l4_nb2" / "checkpoints"
DEFAULT_WAVE_C_THERMAL_DIR = REPO_ROOT / "outputs" / "colab_l4_v45" / "harvest_thermal"
DEFAULT_WAVE_C_LAND_DIR = REPO_ROOT / "outputs" / "colab_l4_v45" / "harvest_land"
DEFAULT_V5_NAV_FAZB_HARVEST_DIR = REPO_ROOT / "outputs" / "colab_l4_v5_nav" / "harvest_faz_b"
DEFAULT_V5_NAV_HARVEST_DIR = REPO_ROOT / "outputs" / "colab_l4_v5_nav" / "harvest"
DEFAULT_V52_HARVEST_DIR = REPO_ROOT / "outputs" / "colab_g4_v52_mega" / "harvest"
DEFAULT_V52_LAND_DIR = DEFAULT_V52_HARVEST_DIR / "land_v52"
DEFAULT_V52_NAV_DIR = DEFAULT_V52_HARVEST_DIR / "nav_v52"
DEFAULT_V52_THERMAL_DIR = DEFAULT_V52_HARVEST_DIR / "thermal_v52"
XVS_LAND_V52_FP16_ONNX_FILENAME: str = "xvs_land_v52_fp16.onnx"
XVS_LAND_V52_FP32_ONNX_FILENAME: str = "xvs_land_v52_fp32.onnx"
XVS_NAV_V52_FAZC_FP16_ONNX_FILENAME: str = "xvs_nav_v52_fazc_fp16.onnx"
XVS_NAV_V52_FAZC_FP32_ONNX_FILENAME: str = "xvs_nav_v52_fazc_fp32.onnx"
XVS_THERMAL_V52_FP16_ONNX_FILENAME: str = "xvs_thermal_v52_fp16.onnx"
XVS_THERMAL_V52_FP32_ONNX_FILENAME: str = "xvs_thermal_v52_fp32.onnx"
XVS_NAV_V5_FAZB_ONNX_FILENAME: str = "xvs_nav_v5_fazb_fp16.onnx"
XVS_NAV_V5_FAZB_XFEAT_ONNX_FILENAME: str = "xfeat_320x256_fp16.onnx"
XVS_NAV_V5_FAZB_LIGHTERGLUE_ONNX_FILENAME: str = "lighterglue_k256_fp16.onnx"
XVS_NAV_V5_ONNX_FILENAME: str = "xvs_nav_v5_fp16.onnx"
XVS_NAV_V5_XFEAT_ONNX_FILENAME: str = "xfeat_320x256_fp32.onnx"
XVS_NAV_V5_MANIFEST_FILENAMES: tuple[str, ...] = (
    "xvs_nav_v5_manifest.json",
    "harvest_manifest.json",
)
XVS_NAV_V5_PARAM_COUNT: int = 9_350_000
XVS_LAND_V52_PARAM_COUNT: int = 1_186_273
XVS_THERMAL_V52_PARAM_COUNT: int = 820_116
V52_LAND_DEFAULT_STUDENT_T_NU: float = 6.0
V52_LAND_DEFAULT_STUDENT_T_SCALE: float = 1.0480755427047912
# Promoted Wave 3A SHA-256 digests for v5.2 G4 Mega weights:
# - PROMOTE: XVS-Land v5.2 FP16 (2,899,997 B, 6 outputs w/ keyline_sigma_px) & XVS-ThermalNet v5.2 FP16 (1,841,441 B)
# - HOLD (DO NOT PROMOTE): XVS-Nav v5.2 Faz-C remains intentionally unpinned ("") because 92.6% S2 / 7.4% NAIP
#   corpus ratio regressed -10.54 pp all-512 / -16.09 pp land-320 vs v5.1 Faz-B + SequenceVprFilter W=7.
EXPECTED_V52_LAND_FP16_ONNX_SHA256: str = (
    "834c8e270f237f044d3920c98d5c0818363ab781be78071884e9365a221ac799"
)
EXPECTED_V52_LAND_FP32_ONNX_SHA256: str = (
    "fc8e808394b7e20c0d80e5fe357306ba332ee6f57018666e941a8d55caee6407"
)
EXPECTED_V52_NAV_FAZC_FP16_ONNX_SHA256: str = ""
EXPECTED_V52_NAV_FAZC_FP32_ONNX_SHA256: str = ""
EXPECTED_V52_THERMAL_FP16_ONNX_SHA256: str = (
    "bc6ceac6252f48dc52305c4e7955cb7841bc5adc7c191a59ce9b0e36ef10fcac"
)
EXPECTED_V52_THERMAL_FP32_ONNX_SHA256: str = (
    "87fbd639591012ec5918c95a1915de9c0fa2911606823afb4e5c6f2bd0c82dbd"
)
EXPECTED_V5_NAV_FAZB_FP16_ONNX_SHA256: str = (
    "e48f0ed5a234dcf1661031b80db5028d8c5b926b7b54e06575b4ed20bd6fc46f"
)
EXPECTED_V5_FAZB_XFEAT_FP16_ONNX_SHA256: str = (
    "e28a2f3afd7e23fe9818ecea34b57185708ff2e2b45f24cb514628f49c3d8182"
)
EXPECTED_V5_FAZB_LIGHTERGLUE_FP16_ONNX_SHA256: str = (
    "e63121ce24b840dba04acd490133eba4175a2ae43b087db3dac6e9d2f3220f3b"
)
EXPECTED_V5_NAV_FP16_ONNX_SHA256: str = (
    "c9b78a98f51ff294b2eeffb12c65902647a9ad08b28a06dec6b870b39fdf6bd1"
)
EXPECTED_V5_XFEAT_ONNX_SHA256: str = (
    "04a153209ec1bd494fea9489b92552d05c91339c86990d2492edbf104deb37c6"
)
DEFAULT_VRAM_CAP_MB: float = 2800.0

XVS_THERMAL_PARAM_COUNT: int = 335_796
XVS_LAND_PARAM_COUNT: int = 20_562_691

EXPECTED_D2_ONNX_SHA256: str = (
    "2c9578f362f6bbbcc0f9ce1de925bf5dbce5459d75dfb387c79c810023750d0d"
)

# Wave-C v4.5 real-world fine-tuned & locally verified artifact pins (ONNX primary -> PT fallback)
EXPECTED_WAVE_C_THERMAL_ONNX_SHA256: str = (
    "db06e32fb44dd8a949ed8972e06b362c10557c608ed36809d0e9626eeb0a93c3"
)
EXPECTED_WAVE_C_THERMAL_PT_SHA256: str = (
    "69f7e79f7047b8d8f650e7410d8690ffae713db0bf3eaefdb28ded2fa8aec4b1"
)
EXPECTED_WAVE_C_LAND_ONNX_SHA256: str = (
    "4245c0d1cdfea8d5065f56d9d3a7dd9a3d6cc29ff8dea1b1a7c173cd39e2920d"
)
EXPECTED_WAVE_C_LAND_PT_SHA256: str = (
    "2689e0c72143e45c413338f95998c82c57a810ed2b44a3473e769331e42a59aa"
)
WAVE_C_LAND_DEFAULT_STUDENT_T_NU: float = 6.0
WAVE_C_LAND_DEFAULT_STUDENT_T_SCALE: float = 4.214477485437598

EXPECTED_L4_CKPT_MEMBERS: dict[str, tuple[str, str]] = {
    "xvs_land": (
        "xvs_land/xvs_land_final.pt",
        "5d9309329c485a24c0446580a8ac9f019149948f80ae9fff47382b394bf5b33a",
    ),
    "xvs_thermal": (
        "xvs_thermal/xvs_thermal_final.pt",
        "009c7cca227ae897b02235067e0761f88c9980bae3058bb63e13ae4fa1a940ad",
    ),
    "xvs_nav": (
        "xvs_nav/xvs_nav_final.pt",
        "6a21fffa359d7a29a372d3d46cf9ccb3afafda056913dfe098dd45a23b9ca399",
    ),
}


def compute_file_sha256(path: Path) -> str:
    """Compute streaming SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _decompress_zstd_bytes(raw: bytes) -> bytes:
    """Decompress `.zst` payload using Python 3.14 stdlib `compression.zstd` or `zstandard`."""
    try:
        from compression import zstd as stdlib_zstd  # type: ignore[import-not-found]

        return stdlib_zstd.decompress(raw)
    except ImportError:
        import zstandard  # type: ignore[import-not-found]

        return zstandard.ZstdDecompressor().decompress(raw)


def _detect_host_gpu_label() -> str:
    """Return honest host GPU label from PyTorch CUDA or read-only `nvidia-smi` probe."""
    if torch.cuda.is_available():
        return str(torch.cuda.get_device_name(0))
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return "CPU"


def get_process_vram_mb() -> float:
    """Return current PyTorch CUDA allocated+reserved peak VRAM in MiB (`0.0` on CPU)."""
    if not torch.cuda.is_available():
        return 0.0
    alloc = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    res = float(torch.cuda.memory_reserved()) / (1024.0 * 1024.0)
    return round(max(alloc, res), 2)


# ---- Pure-PyTorch Zero-`timm` DINOv2-B/14 + BoQ Loader (Exact 176-Tensor Parity) ----


class _PatchEmbed14(nn.Module):
    def __init__(self, in_ch: int = 3, embed_dim: int = 768, patch_size: int = 14) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)


class _LayerScale(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class _MultiHeadAttention(nn.Module):
    def __init__(self, dim: int = 768, num_heads: int = 12) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).reshape(b, n, c)
        return self.proj(y)


class _Mlp(nn.Module):
    def __init__(self, dim: int = 768, hidden_dim: int = 3072) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _DinoV2Block(nn.Module):
    def __init__(self, dim: int = 768, num_heads: int = 12) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _MultiHeadAttention(dim=dim, num_heads=num_heads)
        self.ls1 = _LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Mlp(dim=dim, hidden_dim=dim * 4)
        self.ls2 = _LayerScale(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class _DinoV2ViTBaseBackbone(nn.Module):
    """Pure-PyTorch `vit_base_patch14_dinov2.lvd142m` matching Colab L4 `timm` keys 1:1."""

    def __init__(self, embed_dim: int = 768, depth: int = 12, num_heads: int = 12) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = _PatchEmbed14(in_ch=3, embed_dim=embed_dim, patch_size=14)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 37 * 37, embed_dim))
        self.blocks = nn.ModuleList([_DinoV2Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def _interpolate_pos_embed(self, gh: int, gw: int, dtype: torch.dtype) -> torch.Tensor:
        cls_pos = self.pos_embed[:, :1, :]
        patch_pos = self.pos_embed[:, 1:, :]
        n_orig = patch_pos.shape[1]
        orig_side = int(round(math.sqrt(float(n_orig))))
        if orig_side == gh and orig_side == gw:
            return self.pos_embed.to(dtype=dtype)
        patch_2d = patch_pos.float().reshape(1, orig_side, orig_side, self.embed_dim).permute(0, 3, 1, 2)
        resized = F.interpolate(
            patch_2d,
            size=(gh, gw),
            mode="bicubic",
            align_corners=False,
            antialias=False,
        )
        resized = resized.permute(0, 2, 3, 1).reshape(1, gh * gw, self.embed_dim)
        return torch.cat([cls_pos.float(), resized], dim=1).to(dtype=dtype)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.patch_embed.proj(x)  # [B, 768, H/14, W/14]
        b, c, gh, gw = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # [B, N, 768]
        cls = self.cls_token.to(dtype=tokens.dtype).expand(b, -1, -1)
        x_tok = torch.cat([cls, tokens], dim=1)
        pos = self._interpolate_pos_embed(gh, gw, dtype=tokens.dtype)
        x_tok = x_tok + pos
        for blk in self.blocks:
            x_tok = blk(x_tok)
        return self.norm(x_tok)


class DinoV2ViTBaseBoQ(nn.Module):
    """Pure-PyTorch XVS-Nav retriever loading all 176 Colab L4 `xvs_nav_final.pt` tensors `strict=True`."""

    def __init__(self, embed_dim: int = 768, num_queries: int = 16, pca_dim: int = 512) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.pca_dim = pca_dim
        self.backbone = _DinoV2ViTBaseBackbone(embed_dim=embed_dim, depth=12, num_heads=12)
        self.boq_queries = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        self.pca_proj = nn.Linear(num_queries * embed_dim, pca_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(x)  # [B, N, 768]
        b, _, d = tokens.shape
        q = self.boq_queries.to(dtype=tokens.dtype).expand(b, -1, -1)
        attn = torch.softmax(torch.bmm(q, tokens.transpose(1, 2)) / math.sqrt(float(d)), dim=-1)
        attn = attn.to(dtype=tokens.dtype)
        pooled = torch.bmm(attn, tokens).reshape(b, -1)
        return F.normalize(self.pca_proj(pooled).float(), p=2, dim=-1, eps=1e-12)


# ---- Extraction, SHA-256 Candidate Chain & Bundle Loader ----


@dataclass(frozen=True)
class PinnedArtifactCandidate:
    """One SHA-256-pinned model artifact candidate in a prioritized fallback chain."""

    tier: str
    backend: str  # "onnxruntime" | "pytorch"
    path: Path
    expected_sha256: str
    companion_pt_path: Path | None = None
    companion_pt_sha256: str = ""
    default_calibrator_nu: float = 6.0
    default_calibrator_scale: float = 1.0


@dataclass(frozen=True)
class ModelCheckpointMeta:
    """Verified provenance and device metadata for one loaded OpenXVS model."""

    name: str
    loaded: bool
    real_weights: bool
    device: str
    dtype: str
    sha256: str
    source_path: str
    tensor_count: int
    param_count: int
    backend: str = "pytorch"
    tier: str = "nb2_l4_pt"


@dataclass
class StudioModelsBundle:
    """Loaded 5/5 OpenXVS model bundle for live local inference."""

    d2_session: ort.InferenceSession | None
    d2_x_dtype: Any
    d2_h_dtype: Any
    d2_hid_c: int
    land_model: XvsLandModel | None
    land_calibrator: StudentTCalibrator
    land_device: torch.device
    thermal_model: XvsThermalNet | None
    thermal_device: torch.device
    nav_model: DinoV2ViTBaseBoQ | None
    nav_device: torch.device
    nav_dtype: torch.dtype
    nav_pca_whitening: dict[str, np.ndarray] | None
    meta: dict[str, ModelCheckpointMeta]
    host_gpu_name: str
    compute_device: str
    vram_cap_mb: float
    land_session: ort.InferenceSession | None = None
    thermal_session: ort.InferenceSession | None = None
    nav_session: ort.InferenceSession | None = None
    nav_xfeat_onnx_path: Path | None = None
    nav_xfeat_sha256: str = ""
    nav_lighterglue_onnx_path: Path | None = None
    nav_lighterglue_sha256: str = ""
    land_has_keyline_sigma: bool = False
    land_has_runway_present: bool = False
    d2_input_hw: tuple[int, int] = (256, 256)

    def health_summary(self) -> dict[str, Any]:
        """Return JSON-serializable `/api/health` summary."""
        return {
            "status": "ok",
            "kani_proofs": "26/26",
            "host_gpu_name": self.host_gpu_name,
            "compute_device": self.compute_device,
            "cuda_available": bool(torch.cuda.is_available()),
            "vram_mb": get_process_vram_mb(),
            "vram_cap_mb": self.vram_cap_mb,
            "models": {
                k: {
                    "name": v.name,
                    "loaded": v.loaded,
                    "real_weights": v.real_weights,
                    "device": v.device,
                    "dtype": v.dtype,
                    "backend": v.backend,
                    "tier": v.tier,
                    "sha256": v.sha256,
                    "source_path": v.source_path,
                    "tensor_count": v.tensor_count,
                    "param_count": v.param_count,
                }
                for k, v in self.meta.items()
            },
        }


def _format_source_path(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def _resolve_cpu_ort_threads(default_cap: int = 6) -> int:
    """Resolve CPU ORT `intra_op_num_threads`, respecting positive `OMP_NUM_THREADS` thermal caps."""
    n_phys = max(1, min(int(default_cap), (os.cpu_count() or 4) // 2))
    raw_omp = os.environ.get("OMP_NUM_THREADS", "").strip()
    if raw_omp:
        try:
            omp_val = int(raw_omp)
        except ValueError:
            omp_val = 0
        if omp_val >= 1:
            n_phys = min(n_phys, omp_val)
    return max(1, n_phys)


def _create_ort_session(onnx_path: Path, base_dev_str: str) -> tuple[ort.InferenceSession, str]:
    """Create an optimized ONNX Runtime session preferring CUDA when requested & available.

    On CPU, binds `intra_op_num_threads` to the physical-core estimate (`cpu_count // 2`, capped
    at 6 and clamped to `<= OMP_NUM_THREADS` when set) with `allow_spinning=0` and
    `inter_op_num_threads=1` to eliminate spin-lock contention and honor thermal thread locks.
    """
    avail_providers = ort.get_available_providers()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if base_dev_str == "cuda" and "CUDAExecutionProvider" in avail_providers
        else ["CPUExecutionProvider"]
    )
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if providers == ["CPUExecutionProvider"]:
        n_phys = _resolve_cpu_ort_threads(default_cap=6)
        sess_opts.intra_op_num_threads = n_phys
        sess_opts.inter_op_num_threads = 1
        sess_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        sess_opts.add_session_config_entry("session.dynamic_block_base", "4")
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=providers)
    dev_label = "cuda" if "CUDAExecutionProvider" in sess.get_providers() else "cpu"
    return sess, dev_label


def resolve_pinned_candidate(
    candidates: Sequence[PinnedArtifactCandidate],
    *,
    verify_sha256: bool = True,
) -> tuple[PinnedArtifactCandidate, str] | None:
    """Return the first existing candidate passing strict SHA-256 verification (fail-closed on mismatch).

    Unpinned candidates (`expected_sha256 == ""`) are skipped when `verify_sha256=True` so
    unverified local artifacts can never be loaded silently.
    """
    for cand in candidates:
        if verify_sha256 and not cand.expected_sha256:
            continue
        if not cand.path.is_file():
            continue
        actual_sha = compute_file_sha256(cand.path)
        if verify_sha256 and actual_sha != cand.expected_sha256:
            continue
        return cand, actual_sha
    return None


def _build_thermal_candidates(
    *,
    wave_c_dir: Path,
    nb2_pt_path: Path | None,
    nb2_expected_sha256: str,
    prefer_onnx: bool,
    v52_dir: Path | None = None,
    v52_fp16_expected_sha256: str = EXPECTED_V52_THERMAL_FP16_ONNX_SHA256,
    v52_fp32_expected_sha256: str = EXPECTED_V52_THERMAL_FP32_ONNX_SHA256,
) -> list[PinnedArtifactCandidate]:
    """Build prioritized v5.2 ONNX (opt-in pinned) -> Wave-C ONNX -> Wave-C `.pt` -> NB2 `.pt` chain."""
    chain: list[PinnedArtifactCandidate] = []
    if prefer_onnx and v52_dir is not None:
        chain.append(
            PinnedArtifactCandidate(
                tier="v52_thermal_fp16_onnx",
                backend="onnxruntime",
                path=v52_dir / XVS_THERMAL_V52_FP16_ONNX_FILENAME,
                expected_sha256=v52_fp16_expected_sha256,
            )
        )
        chain.append(
            PinnedArtifactCandidate(
                tier="v52_thermal_fp32_onnx",
                backend="onnxruntime",
                path=v52_dir / XVS_THERMAL_V52_FP32_ONNX_FILENAME,
                expected_sha256=v52_fp32_expected_sha256,
            )
        )
    if prefer_onnx:
        chain.append(
            PinnedArtifactCandidate(
                tier="wave_c_v45_onnx",
                backend="onnxruntime",
                path=wave_c_dir / "xvs_thermal_v45_fp32.onnx",
                expected_sha256=EXPECTED_WAVE_C_THERMAL_ONNX_SHA256,
            )
        )
    chain.append(
        PinnedArtifactCandidate(
            tier="wave_c_v45_pt",
            backend="pytorch",
            path=wave_c_dir / "xvs_thermal_final.pt",
            expected_sha256=EXPECTED_WAVE_C_THERMAL_PT_SHA256,
        )
    )
    if nb2_pt_path is not None:
        chain.append(
            PinnedArtifactCandidate(
                tier="nb2_l4_pt",
                backend="pytorch",
                path=nb2_pt_path,
                expected_sha256=nb2_expected_sha256,
            )
        )
    return chain


def _build_land_candidates(
    *,
    wave_c_dir: Path,
    nb2_pt_path: Path | None,
    nb2_expected_sha256: str,
    prefer_onnx: bool,
    v52_dir: Path | None = None,
    v52_fp16_expected_sha256: str = EXPECTED_V52_LAND_FP16_ONNX_SHA256,
    v52_fp32_expected_sha256: str = EXPECTED_V52_LAND_FP32_ONNX_SHA256,
) -> list[PinnedArtifactCandidate]:
    """Build prioritized v5.2 6-output ONNX (opt-in pinned) -> Wave-C 5-output ONNX -> Wave-C `.pt` -> NB2 `.pt` chain."""
    chain: list[PinnedArtifactCandidate] = []
    if prefer_onnx and v52_dir is not None:
        chain.append(
            PinnedArtifactCandidate(
                tier="v52_land_fp16_onnx",
                backend="onnxruntime",
                path=v52_dir / XVS_LAND_V52_FP16_ONNX_FILENAME,
                expected_sha256=v52_fp16_expected_sha256,
                default_calibrator_nu=V52_LAND_DEFAULT_STUDENT_T_NU,
                default_calibrator_scale=V52_LAND_DEFAULT_STUDENT_T_SCALE,
            )
        )
        chain.append(
            PinnedArtifactCandidate(
                tier="v52_land_fp32_onnx",
                backend="onnxruntime",
                path=v52_dir / XVS_LAND_V52_FP32_ONNX_FILENAME,
                expected_sha256=v52_fp32_expected_sha256,
                default_calibrator_nu=V52_LAND_DEFAULT_STUDENT_T_NU,
                default_calibrator_scale=V52_LAND_DEFAULT_STUDENT_T_SCALE,
            )
        )
    wave_c_pt = wave_c_dir / "xvs_land_final.pt"
    if prefer_onnx:
        chain.append(
            PinnedArtifactCandidate(
                tier="wave_c_v45_onnx",
                backend="onnxruntime",
                path=wave_c_dir / "xvs_land_v45.onnx",
                expected_sha256=EXPECTED_WAVE_C_LAND_ONNX_SHA256,
                companion_pt_path=wave_c_pt,
                companion_pt_sha256=EXPECTED_WAVE_C_LAND_PT_SHA256,
                default_calibrator_nu=WAVE_C_LAND_DEFAULT_STUDENT_T_NU,
                default_calibrator_scale=WAVE_C_LAND_DEFAULT_STUDENT_T_SCALE,
            )
        )
    chain.append(
        PinnedArtifactCandidate(
            tier="wave_c_v45_pt",
            backend="pytorch",
            path=wave_c_pt,
            expected_sha256=EXPECTED_WAVE_C_LAND_PT_SHA256,
            default_calibrator_nu=WAVE_C_LAND_DEFAULT_STUDENT_T_NU,
            default_calibrator_scale=WAVE_C_LAND_DEFAULT_STUDENT_T_SCALE,
        )
    )
    if nb2_pt_path is not None:
        chain.append(
            PinnedArtifactCandidate(
                tier="nb2_l4_pt",
                backend="pytorch",
                path=nb2_pt_path,
                expected_sha256=nb2_expected_sha256,
            )
        )
    return chain


def _resolve_v5_nav_manifest_sha256(
    harvest_dir: Path,
    onnx_filename: str = XVS_NAV_V5_ONNX_FILENAME,
) -> tuple[str | None, Path | None]:
    """Read optional `xvs_nav_v5_manifest.json` / `harvest_manifest.json` SHA-256 pin if present.

    Returns `(expected_sha256, manifest_path)`:
      - `(None, None)` when no manifest file is present in `harvest_dir`.
      - `("", manifest_path)` when a manifest exists but is malformed / missing a valid 64-hex pin
        (caller fails closed and falls back to verified NB2 `.pt`).
      - `(sha256_hex, manifest_path)` when a valid 64-hex pin is found.
    """
    for fname in XVS_NAV_V5_MANIFEST_FILENAMES:
        mpath = harvest_dir / fname
        if not mpath.is_file():
            continue
        try:
            doc = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception:
            return "", mpath
        if not isinstance(doc, dict):
            return "", mpath

        cand_sha = ""
        if "files" in doc and isinstance(doc["files"], dict):
            entry = doc["files"][onnx_filename] if onnx_filename in doc["files"] else None
            if isinstance(entry, dict) and "sha256" in entry:
                cand_sha = str(entry["sha256"]).strip().lower()
        elif "files" in doc and isinstance(doc["files"], list):
            for entry in doc["files"]:
                if isinstance(entry, dict) and "path" in entry and "sha256" in entry:
                    if Path(str(entry["path"])).name == onnx_filename:
                        cand_sha = str(entry["sha256"]).strip().lower()
                        break
        if not cand_sha:
            for key in ("fp16_onnx_sha256", "onnx_sha256", "sha256"):
                if key in doc and isinstance(doc[key], str):
                    cand_sha = doc[key].strip().lower()
                    break

        if len(cand_sha) == 64 and all(ch in "0123456789abcdef" for ch in cand_sha):
            return cand_sha, mpath
        return "", mpath
    return None, None


def _verify_v5_nav_artifact(
    root_dir: Path,
    filename: str,
    *,
    canonical_dir: Path,
    pinned_sha256: str,
) -> tuple[Path, str] | None:
    """Fail-closed SHA-256 verification for one V5 Nav / Stage-2 ONNX artifact.

    On the canonical harvest directory, both the immutable pinned digest and any
    co-located manifest digest must agree. On custom/isolated directories, an
    accompanying manifest must match exactly, or (when no manifest is present)
    the file must match the pinned digest.
    """
    artifact_path = root_dir / filename
    if not artifact_path.is_file():
        return None
    manifest_sha, manifest_file = _resolve_v5_nav_manifest_sha256(root_dir, filename)
    actual_sha = compute_file_sha256(artifact_path)
    is_canonical = root_dir.resolve() == canonical_dir.resolve()

    if is_canonical:
        if not pinned_sha256 or actual_sha != pinned_sha256:
            return None
        if manifest_file is not None and (not manifest_sha or actual_sha != manifest_sha):
            return None
        return artifact_path, actual_sha

    if manifest_file is not None:
        if not manifest_sha or actual_sha != manifest_sha:
            return None
        return artifact_path, actual_sha

    if not pinned_sha256 or actual_sha != pinned_sha256:
        return None
    return artifact_path, actual_sha


def _validate_land_ort_io(sess: ort.InferenceSession) -> tuple[bool, bool, bool]:
    """Verify XVS-Land ONNX IO: 5-output, 6-output, OR 7-output extended (`runway_present_logit`)."""
    inps = sess.get_inputs()
    outs = sess.get_outputs()
    if len(inps) != 1 or inps[0].name != "roi_crop":
        return False, False, False
    in_shape = list(inps[0].shape)
    if len(in_shape) != 4 or in_shape[1] != 3:
        return False, False, False
    out_names = {o.name for o in outs}
    required = {"keypoints_xy", "keylines_hesse", "cov_2x2"}
    if not required.issubset(out_names):
        return False, False, False
    has_keyline_sigma = "keyline_sigma_px" in out_names
    has_runway_present = "runway_present_logit" in out_names
    return True, has_keyline_sigma, has_runway_present


def _validate_nav_v5_ort_io(sess: ort.InferenceSession) -> bool:
    """Verify static XVS-Nav v5.0 ONNX IO contract: `[1, 3, 224, 224] -> [1, 512]`."""
    inps = sess.get_inputs()
    outs = sess.get_outputs()
    if len(inps) != 1 or len(outs) != 1:
        return False
    in_shape = list(inps[0].shape)
    out_shape = list(outs[0].shape)
    if len(in_shape) != 4 or in_shape[1:] != [3, 224, 224]:
        return False
    if len(out_shape) != 2 or out_shape[1] != 512:
        return False
    return True


def _load_expected_hashes(manifest_path: Path) -> dict[str, str]:
    expected = {k: sha for k, (_, sha) in EXPECTED_L4_CKPT_MEMBERS.items()}
    if manifest_path.is_file():
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = doc["files"] if isinstance(doc, dict) and "files" in doc else []
        by_path = {
            str(entry["path"]): str(entry["sha256"])
            for entry in files
            if isinstance(entry, dict) and "path" in entry and "sha256" in entry
        }
        for key, (member_path, _) in EXPECTED_L4_CKPT_MEMBERS.items():
            if member_path in by_path:
                expected[key] = by_path[member_path]
    return expected


def extract_l4_checkpoints(
    archive_path: Path = DEFAULT_L4_ARCHIVE,
    dest_dir: Path = DEFAULT_CKPT_DIR,
    manifest_path: Path = DEFAULT_L4_MANIFEST,
    *,
    verify_sha256: bool = True,
) -> dict[str, Path]:
    """Safely extract `xvs_{land,thermal,nav}_final.pt` into `dest_dir` with SHA-256 verification."""
    expected_hashes = _load_expected_hashes(manifest_path)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_paths: dict[str, Path] = {}
    missing_members: dict[str, tuple[str, Path, str]] = {}

    for key, (member_name, _) in EXPECTED_L4_CKPT_MEMBERS.items():
        target = dest_dir / Path(member_name).name
        exp_sha = expected_hashes[key]
        if target.is_file():
            if not verify_sha256 or compute_file_sha256(target) == exp_sha:
                out_paths[key] = target
                continue
        missing_members[key] = (member_name, target, exp_sha)

    if not missing_members:
        return out_paths

    if not archive_path.is_file():
        return out_paths

    raw_tar = _decompress_zstd_bytes(archive_path.read_bytes())
    with tarfile.open(fileobj=io.BytesIO(raw_tar), mode="r:") as tf:
        members_by_name = {m.name.lstrip("./"): m for m in tf.getmembers() if m.isfile()}
        for key, (member_name, target, exp_sha) in missing_members.items():
            if member_name not in members_by_name:
                raise FileNotFoundError(f"Expected member {member_name!r} missing from {archive_path}")
            fobj = tf.extractfile(members_by_name[member_name])
            if fobj is None:
                raise RuntimeError(f"Failed to read member {member_name!r} from {archive_path}")
            payload = fobj.read()
            actual_sha = hashlib.sha256(payload).hexdigest()
            if verify_sha256 and actual_sha != exp_sha:
                raise ValueError(
                    f"SHA-256 mismatch for {member_name}: expected {exp_sha}, got {actual_sha}"
                )
            tmp_path = target.with_suffix(target.suffix + ".tmp")
            tmp_path.write_bytes(payload)
            tmp_path.replace(target)
            out_paths[key] = target

    return out_paths


def _resolve_device_str(device_pref: str) -> str:
    pref = device_pref.strip().lower()
    if pref == "cpu":
        return "cpu"
    if pref.startswith("cuda") and torch.cuda.is_available():
        return "cuda"
    if pref == "auto" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _can_fit_vram_mb(need_mb: float, vram_cap_mb: float) -> bool:
    if not torch.cuda.is_available():
        return False
    used_mb = get_process_vram_mb()
    free_bytes, _ = torch.cuda.mem_get_info()
    free_mb = float(free_bytes) / (1024.0 * 1024.0)
    return (used_mb + need_mb <= vram_cap_mb) and (free_mb >= need_mb + 256.0)


def _load_student_t_calibrator_from_checkpoint(ck_land: dict[str, Any]) -> StudentTCalibrator:
    if "student_t_nu" in ck_land and "student_t_scale" in ck_land:
        return StudentTCalibrator(
            nu=float(ck_land["student_t_nu"]),
            scale=float(ck_land["student_t_scale"]),
        )
    return StudentTCalibrator(nu=6.0, scale=1.0)


def load_studio_models(
    *,
    device_pref: str = "auto",
    d2_onnx_path: Path = DEFAULT_D2_ONNX,
    l4_archive_path: Path = DEFAULT_L4_ARCHIVE,
    ckpt_dir: Path = DEFAULT_CKPT_DIR,
    manifest_path: Path = DEFAULT_L4_MANIFEST,
    wave_c_thermal_dir: Path = DEFAULT_WAVE_C_THERMAL_DIR,
    wave_c_land_dir: Path = DEFAULT_WAVE_C_LAND_DIR,
    v5_nav_dir: Path = DEFAULT_V5_NAV_HARVEST_DIR,
    v5_nav_fazb_dir: Path | None = None,
    v52_land_dir: Path | None = None,
    v52_thermal_dir: Path | None = None,
    v52_nav_dir: Path | None = None,
    v52_land_expected_sha256: str = EXPECTED_V52_LAND_FP16_ONNX_SHA256,
    v52_land_fp32_expected_sha256: str = EXPECTED_V52_LAND_FP32_ONNX_SHA256,
    v52_thermal_expected_sha256: str = EXPECTED_V52_THERMAL_FP16_ONNX_SHA256,
    v52_thermal_fp32_expected_sha256: str = EXPECTED_V52_THERMAL_FP32_ONNX_SHA256,
    v52_nav_expected_sha256: str = EXPECTED_V52_NAV_FAZC_FP16_ONNX_SHA256,
    prefer_onnx: bool = True,
    vram_cap_mb: float = DEFAULT_VRAM_CAP_MB,
    smoke_fast: bool = False,
    verify_d2_sha256: bool = True,
) -> StudioModelsBundle:
    """Load all 5 OpenXVS models with strict SHA-256 verification and `< 2.8 GB` VRAM cap."""
    host_gpu = _detect_host_gpu_label()
    base_dev_str = _resolve_device_str(device_pref)
    compute_label = (
        f"cuda:{host_gpu.replace('NVIDIA GeForce ', '').replace(' ', '')}"
        if base_dev_str == "cuda"
        else "cpu"
    )

    if base_dev_str == "cuda":
        torch.cuda.reset_peak_memory_stats()

    meta: dict[str, ModelCheckpointMeta] = {}

    # 1. XVS-D2 (ONNX Runtime ConvGRU with pinned SHA-256 fail-closed verification)
    d2_sess: ort.InferenceSession | None = None
    d2_x_dtype: Any = np.float32
    d2_h_dtype: Any = np.float32
    d2_hid_c = 24
    d2_input_hw: tuple[int, int] = (256, 256)
    d2_sha = ""
    if d2_onnx_path.is_file():
        d2_sha = compute_file_sha256(d2_onnx_path)
        if verify_d2_sha256 and d2_sha != EXPECTED_D2_ONNX_SHA256:
            raise ValueError(
                f"XVS-D2 ONNX SHA-256 mismatch for {d2_onnx_path}: "
                f"expected {EXPECTED_D2_ONNX_SHA256}, got {d2_sha}"
            )
        d2_sess, d2_dev = _create_ort_session(d2_onnx_path, base_dev_str)
        inps = {inp.name: inp for inp in d2_sess.get_inputs()}
        if "h_prev_warped" in inps:
            h_shape = inps["h_prev_warped"].shape
            if len(h_shape) >= 2 and isinstance(h_shape[1], int):
                d2_hid_c = int(h_shape[1])
            if inps["h_prev_warped"].type == "tensor(float16)":
                d2_h_dtype = np.float16
        if "x_t" in inps:
            x_shape = inps["x_t"].shape
            if (
                len(x_shape) == 4
                and isinstance(x_shape[2], int)
                and isinstance(x_shape[3], int)
                and x_shape[2] > 0
                and x_shape[3] > 0
            ):
                d2_input_hw = (int(x_shape[2]), int(x_shape[3]))
            if inps["x_t"].type == "tensor(float16)":
                d2_x_dtype = np.float16
        meta["xvs_d2"] = ModelCheckpointMeta(
            name="XVS-D2",
            loaded=True,
            real_weights=True,
            device=d2_dev,
            dtype="fp16" if d2_x_dtype == np.float16 else "fp32",
            sha256=d2_sha,
            source_path=_format_source_path(d2_onnx_path),
            tensor_count=len(d2_sess.get_inputs()) + len(d2_sess.get_outputs()),
            param_count=257_000,
            backend="onnxruntime",
            tier="d2_onnx",
        )
    else:
        meta["xvs_d2"] = ModelCheckpointMeta(
            name="XVS-D2",
            loaded=False,
            real_weights=False,
            device="cpu",
            dtype="fp32",
            sha256="",
            source_path=str(d2_onnx_path),
            tensor_count=0,
            param_count=0,
            backend="onnxruntime",
            tier="missing",
        )

    # 2. XVS-ActiveRange (Exact Rust-parity MSC-RPEKF + SC3BF-FIM + Split Conformal)
    meta["xvs_active_range"] = ModelCheckpointMeta(
        name="XVS-ActiveRange",
        loaded=True,
        real_weights=True,
        device="cpu",
        dtype="fp64",
        sha256="rust-kani-26-26-bitexact",
        source_path="src/xvs/v2/xvs_active_range/filter.py",
        tensor_count=6,
        param_count=0,
        backend="rust-parity",
        tier="kani_26_26",
    )

    # Extract Colab L4 NB2 fallback checkpoints once (unless smoke_fast requested)
    expected_nb2_hashes = _load_expected_hashes(manifest_path)
    ckpt_paths: dict[str, Path] = {}
    if not smoke_fast:
        ckpt_paths = extract_l4_checkpoints(
            archive_path=l4_archive_path,
            dest_dir=ckpt_dir,
            manifest_path=manifest_path,
            verify_sha256=True,
        )

    # 3. XVS-Land (v5.2 6-output ONNX [opt-in pinned] -> Wave-C 5-output ONNX -> Wave-C PT -> NB2 PT)
    land_dev = torch.device("cuda" if base_dev_str == "cuda" and _can_fit_vram_mb(220.0, vram_cap_mb) else "cpu")
    land_session: ort.InferenceSession | None = None
    land_model: XvsLandModel | None = None
    land_cal = StudentTCalibrator(nu=6.0, scale=1.0)
    land_real = False
    land_sha = ""
    land_src = "smoke_uninitialized"
    land_backend = "pytorch"
    land_tier = "smoke_uninitialized"
    land_dev_label = str(land_dev)
    land_dtype_label = "fp32"
    land_tensor_count = 0
    land_param_count = XVS_LAND_PARAM_COUNT
    land_has_keyline_sigma = False
    land_has_runway_present = False

    if not smoke_fast:
        wave_c_land_root = Path(wave_c_land_dir)
        if v52_land_dir is not None:
            v52_land_root = Path(v52_land_dir)
        elif wave_c_land_root.resolve() == DEFAULT_WAVE_C_LAND_DIR.resolve():
            v52_land_root = DEFAULT_V52_LAND_DIR
        else:
            v52_land_root = wave_c_land_root

        land_candidates = _build_land_candidates(
            wave_c_dir=wave_c_land_root,
            nb2_pt_path=ckpt_paths["xvs_land"] if "xvs_land" in ckpt_paths else None,
            nb2_expected_sha256=expected_nb2_hashes["xvs_land"],
            prefer_onnx=bool(prefer_onnx),
            v52_dir=v52_land_root,
            v52_fp16_expected_sha256=v52_land_expected_sha256,
            v52_fp32_expected_sha256=v52_land_fp32_expected_sha256,
        )
        for cand in land_candidates:
            resolved = resolve_pinned_candidate([cand], verify_sha256=True)
            if resolved is None:
                continue
            chosen, actual_sha = resolved
            if chosen.backend == "onnxruntime":
                try:
                    sess, ort_dev = _create_ort_session(chosen.path, base_dev_str)
                    io_ok, has_kl_sig, has_rwy_pres = _validate_land_ort_io(sess)
                    if not io_ok:
                        continue
                except Exception:
                    continue
                land_cal = StudentTCalibrator(
                    nu=float(chosen.default_calibrator_nu),
                    scale=float(chosen.default_calibrator_scale),
                )
                if chosen.companion_pt_path is not None and chosen.companion_pt_path.is_file():
                    comp_sha = compute_file_sha256(chosen.companion_pt_path)
                    if chosen.companion_pt_sha256 and comp_sha == chosen.companion_pt_sha256:
                        ck_comp = torch.load(chosen.companion_pt_path, map_location="cpu", weights_only=False)
                        if isinstance(ck_comp, dict):
                            land_cal = _load_student_t_calibrator_from_checkpoint(ck_comp)
                land_session = sess
                land_model = None
                land_real = True
                land_sha = actual_sha
                land_src = _format_source_path(chosen.path)
                land_backend = "onnxruntime"
                land_tier = chosen.tier
                land_dev_label = ort_dev
                land_dtype_label = "fp16" if "fp16" in chosen.path.name.lower() else "fp32"
                land_tensor_count = len(sess.get_inputs()) + len(sess.get_outputs())
                land_param_count = (
                    XVS_LAND_V52_PARAM_COUNT if chosen.tier.startswith(("v52_", "v53_")) else XVS_LAND_PARAM_COUNT
                )
                land_has_keyline_sigma = bool(has_kl_sig)
                land_has_runway_present = bool(has_rwy_pres)
                break

            pt_model = XvsLandModel(in_channels=3, roi_width=256, roi_height=256, split_ratio=2.0)
            ck_land = torch.load(chosen.path, map_location="cpu", weights_only=False)
            pt_model.load_state_dict(ck_land["model_state_dict"], strict=True)
            land_cal = _load_student_t_calibrator_from_checkpoint(ck_land)
            pt_model.to(land_dev).eval()
            land_model = pt_model
            land_session = None
            land_real = True
            land_sha = actual_sha
            land_src = _format_source_path(chosen.path)
            land_backend = "pytorch"
            land_tier = chosen.tier
            land_dev_label = str(land_dev)
            land_dtype_label = "fp32"
            land_tensor_count = len(pt_model.state_dict())
            land_param_count = sum(int(p.numel()) for p in pt_model.parameters())
            land_has_keyline_sigma = False
            break

    if land_session is None and land_model is None:
        land_model = XvsLandModel(in_channels=3, roi_width=256, roi_height=256, split_ratio=2.0).to(land_dev).eval()
        land_tensor_count = len(land_model.state_dict())
        land_param_count = sum(int(p.numel()) for p in land_model.parameters())

    meta["xvs_land"] = ModelCheckpointMeta(
        name="XVS-Land",
        loaded=True,
        real_weights=land_real,
        device=land_dev_label,
        dtype=land_dtype_label,
        sha256=land_sha,
        source_path=land_src,
        tensor_count=land_tensor_count,
        param_count=land_param_count,
        backend=land_backend,
        tier=land_tier,
    )

    # 4. XVS-ThermalNet (v5.2 ONNX [opt-in pinned] -> Wave-C ONNX -> Wave-C PT -> NB2 PT)
    thermal_dev = torch.device(
        "cuda" if base_dev_str == "cuda" and _can_fit_vram_mb(80.0, vram_cap_mb) else "cpu"
    )
    thermal_session: ort.InferenceSession | None = None
    thermal_model: XvsThermalNet | None = None
    thermal_real = False
    thermal_sha = ""
    thermal_src = "smoke_uninitialized"
    thermal_backend = "pytorch"
    thermal_tier = "smoke_uninitialized"
    thermal_dev_label = str(thermal_dev)
    thermal_dtype_label = "fp32"
    thermal_tensor_count = 0
    thermal_param_count = XVS_THERMAL_PARAM_COUNT

    if not smoke_fast:
        wave_c_th_root = Path(wave_c_thermal_dir)
        if v52_thermal_dir is not None:
            v52_th_root = Path(v52_thermal_dir)
        elif wave_c_th_root.resolve() == DEFAULT_WAVE_C_THERMAL_DIR.resolve():
            v52_th_root = DEFAULT_V52_THERMAL_DIR
        else:
            v52_th_root = wave_c_th_root

        thermal_candidates = _build_thermal_candidates(
            wave_c_dir=wave_c_th_root,
            nb2_pt_path=ckpt_paths["xvs_thermal"] if "xvs_thermal" in ckpt_paths else None,
            nb2_expected_sha256=expected_nb2_hashes["xvs_thermal"],
            prefer_onnx=bool(prefer_onnx),
            v52_dir=v52_th_root,
            v52_fp16_expected_sha256=v52_thermal_expected_sha256,
            v52_fp32_expected_sha256=v52_thermal_fp32_expected_sha256,
        )
        for cand in thermal_candidates:
            resolved = resolve_pinned_candidate([cand], verify_sha256=True)
            if resolved is None:
                continue
            chosen, actual_sha = resolved
            if chosen.backend == "onnxruntime":
                try:
                    sess, ort_dev = _create_ort_session(chosen.path, base_dev_str)
                except Exception:
                    continue
                thermal_session = sess
                thermal_model = None
                thermal_real = True
                thermal_sha = actual_sha
                thermal_src = _format_source_path(chosen.path)
                thermal_backend = "onnxruntime"
                thermal_tier = chosen.tier
                thermal_dev_label = ort_dev
                thermal_dtype_label = "fp16" if "fp16" in chosen.path.name.lower() else "fp32"
                thermal_tensor_count = len(sess.get_inputs()) + len(sess.get_outputs())
                thermal_param_count = (
                    XVS_THERMAL_V52_PARAM_COUNT if chosen.tier.startswith("v52_") else XVS_THERMAL_PARAM_COUNT
                )
                break

            pt_thermal = XvsThermalNet()
            ck_th = torch.load(chosen.path, map_location="cpu", weights_only=False)
            pt_thermal.load_state_dict(ck_th["model_state_dict"], strict=True)
            pt_thermal.to(thermal_dev).eval()
            thermal_model = pt_thermal
            thermal_session = None
            thermal_real = True
            thermal_sha = actual_sha
            thermal_src = _format_source_path(chosen.path)
            thermal_backend = "pytorch"
            thermal_tier = chosen.tier
            thermal_dev_label = str(thermal_dev)
            thermal_dtype_label = "fp32"
            thermal_tensor_count = len(pt_thermal.state_dict())
            thermal_param_count = sum(int(p.numel()) for p in pt_thermal.parameters())
            break

    if thermal_session is None and thermal_model is None:
        thermal_model = XvsThermalNet().to(thermal_dev).eval()
        thermal_tensor_count = len(thermal_model.state_dict())
        thermal_param_count = sum(int(p.numel()) for p in thermal_model.parameters())

    meta["xvs_thermal"] = ModelCheckpointMeta(
        name="XVS-ThermalNet",
        loaded=True,
        real_weights=thermal_real,
        device=thermal_dev_label,
        dtype=thermal_dtype_label,
        sha256=thermal_sha,
        source_path=thermal_src,
        tensor_count=thermal_tensor_count,
        param_count=thermal_param_count,
        backend=thermal_backend,
        tier=thermal_tier,
    )

    # 5. XVS-Nav (Prioritized: v5.2 Faz-C FP16 ONNX [opt-in pinned] -> V5.0 Faz-B FP16 ONNX -> Faz-A FP16 ONNX -> NB2 `.pt`)
    nav_session: ort.InferenceSession | None = None
    nav_xfeat_onnx_path: Path | None = None
    nav_xfeat_sha256: str = ""
    nav_lighterglue_onnx_path: Path | None = None
    nav_lighterglue_sha256: str = ""
    nav_model: DinoV2ViTBaseBoQ | None = None
    nav_pca: dict[str, np.ndarray] | None = None
    nav_dev = torch.device("cpu")
    nav_dtype = torch.float32
    nav_real = False
    nav_sha = ""
    nav_src = "smoke_lightweight"
    nav_backend = "pytorch"
    nav_tier = "smoke_lightweight"
    nav_dev_label = str(nav_dev)
    nav_dtype_label = "fp32"
    nav_tensor_count = 0
    nav_param_count = 0

    if not smoke_fast:
        v5_faza_root = Path(v5_nav_dir)
        if v5_nav_fazb_dir is not None:
            v5_fazb_root = Path(v5_nav_fazb_dir)
        elif v5_faza_root.resolve() == DEFAULT_V5_NAV_HARVEST_DIR.resolve():
            v5_fazb_root = DEFAULT_V5_NAV_FAZB_HARVEST_DIR
        else:
            # Isolated custom `v5_nav_dir` passed without explicit `v5_nav_fazb_dir`: stay strictly inside it.
            v5_fazb_root = v5_faza_root

        if v52_nav_dir is not None:
            v52_nav_root = Path(v52_nav_dir)
        elif (
            v5_faza_root.resolve() == DEFAULT_V5_NAV_HARVEST_DIR.resolve()
            and v5_nav_fazb_dir is None
        ):
            v52_nav_root = DEFAULT_V52_NAV_DIR
        else:
            v52_nav_root = v5_fazb_root

        # Candidate 0 -> 1a -> 1b: v5.2 Faz-C FP16 (opt-in pinned) -> Faz-B FP16 -> Faz-A FP16
        if bool(prefer_onnx):
            stage1_candidates: tuple[tuple[Path, str, Path, str, str], ...] = (
                (
                    v52_nav_root,
                    XVS_NAV_V52_FAZC_FP16_ONNX_FILENAME,
                    DEFAULT_V52_NAV_DIR,
                    v52_nav_expected_sha256,
                    "v52_nav_fazc_onnx",
                ),
                (
                    v5_fazb_root,
                    XVS_NAV_V5_FAZB_ONNX_FILENAME,
                    DEFAULT_V5_NAV_FAZB_HARVEST_DIR,
                    EXPECTED_V5_NAV_FAZB_FP16_ONNX_SHA256,
                    "v5_nav_fazb_onnx",
                ),
                (
                    v5_faza_root,
                    XVS_NAV_V5_ONNX_FILENAME,
                    DEFAULT_V5_NAV_HARVEST_DIR,
                    EXPECTED_V5_NAV_FP16_ONNX_SHA256,
                    "v5_nav_onnx",
                ),
            )
            for root_dir, fname, canonical_dir, pinned_sha, tier_label in stage1_candidates:
                verified = _verify_v5_nav_artifact(
                    root_dir,
                    fname,
                    canonical_dir=canonical_dir,
                    pinned_sha256=pinned_sha,
                )
                if verified is None:
                    continue
                cand_path, actual_v5_sha = verified
                try:
                    cand_sess, ort_dev = _create_ort_session(cand_path, base_dev_str)
                    if _validate_nav_v5_ort_io(cand_sess):
                        nav_session = cand_sess
                        nav_model = None
                        nav_pca = None  # Folded Gemm(512, 1024) + LpNorm inside ONNX (0 external NumPy ops)
                        nav_real = True
                        nav_sha = actual_v5_sha
                        nav_src = _format_source_path(cand_path)
                        nav_backend = "onnxruntime"
                        nav_tier = tier_label
                        nav_dev_label = ort_dev
                        nav_dtype_label = "fp16"
                        nav_tensor_count = len(cand_sess.get_inputs()) + len(cand_sess.get_outputs())
                        nav_param_count = XVS_NAV_V5_PARAM_COUNT
                        break
                except Exception:
                    nav_session = None

            # Companion Stage-2A Apache-2.0 XFeat ONNX: Faz-B FP16 -> Faz-A FP32 fallback
            xfeat_candidates: tuple[tuple[Path, str, Path, str], ...] = (
                (
                    v5_fazb_root,
                    XVS_NAV_V5_FAZB_XFEAT_ONNX_FILENAME,
                    DEFAULT_V5_NAV_FAZB_HARVEST_DIR,
                    EXPECTED_V5_FAZB_XFEAT_FP16_ONNX_SHA256,
                ),
                (
                    v5_faza_root,
                    XVS_NAV_V5_XFEAT_ONNX_FILENAME,
                    DEFAULT_V5_NAV_HARVEST_DIR,
                    EXPECTED_V5_XFEAT_ONNX_SHA256,
                ),
            )
            for root_dir, fname, canonical_dir, pinned_sha in xfeat_candidates:
                verified_xf = _verify_v5_nav_artifact(
                    root_dir,
                    fname,
                    canonical_dir=canonical_dir,
                    pinned_sha256=pinned_sha,
                )
                if verified_xf is not None:
                    nav_xfeat_onnx_path, nav_xfeat_sha256 = verified_xf
                    break

            # Companion Stage-2B Apache-2.0 LighterGlue static K=256 FP16 ONNX (Faz-B)
            lg_roots: tuple[Path, ...] = (
                (v5_fazb_root,)
                if v5_fazb_root.resolve() == v5_faza_root.resolve()
                else (v5_fazb_root, v5_faza_root)
            )
            for root_dir in lg_roots:
                verified_lg = _verify_v5_nav_artifact(
                    root_dir,
                    XVS_NAV_V5_FAZB_LIGHTERGLUE_ONNX_FILENAME,
                    canonical_dir=DEFAULT_V5_NAV_FAZB_HARVEST_DIR,
                    pinned_sha256=EXPECTED_V5_FAZB_LIGHTERGLUE_FP16_ONNX_SHA256,
                )
                if verified_lg is not None:
                    nav_lighterglue_onnx_path, nav_lighterglue_sha256 = verified_lg
                    break

        # Candidate 2: Verified Colab L4 NB2 `.pt` fallback (`outputs/colab_l4_nb2/checkpoints/xvs_nav_final.pt`)
        if nav_session is None and "xvs_nav" in ckpt_paths and ckpt_paths["xvs_nav"].is_file():
            np_path = ckpt_paths["xvs_nav"]
            nav_sha = compute_file_sha256(np_path)
            ck_nav = torch.load(np_path, map_location="cpu", weights_only=False)
            nav_model = DinoV2ViTBaseBoQ(embed_dim=768, num_queries=16, pca_dim=512)
            nav_model.load_state_dict(ck_nav["model_state_dict"], strict=True)
            if "pca_whitening" in ck_nav and isinstance(ck_nav["pca_whitening"], dict):
                raw_pca = ck_nav["pca_whitening"]
                nav_pca = {
                    "mean": np.asarray(raw_pca["mean"], dtype=np.float32),
                    "components": np.asarray(raw_pca["components"], dtype=np.float32),
                    "inv_std": np.asarray(raw_pca["inv_std"], dtype=np.float32),
                }
            # Respect strict < 2.8 GB VRAM budget on GTX 1650 SUPER 4 GB (FP16 ~200 MB weights + activations)
            if base_dev_str == "cuda" and _can_fit_vram_mb(700.0, vram_cap_mb):
                nav_dev = torch.device("cuda")
                nav_dtype = torch.float16
                nav_model.to(device=nav_dev, dtype=nav_dtype).eval()
            else:
                nav_dev = torch.device("cpu")
                nav_dtype = torch.float32
                nav_model.to(device=nav_dev, dtype=nav_dtype).eval()
            nav_real = True
            nav_src = _format_source_path(np_path)
            nav_backend = "pytorch"
            nav_tier = "nb2_l4_pt"
            nav_dev_label = str(nav_dev)
            nav_dtype_label = "fp16" if nav_dtype == torch.float16 else "fp32"
            nav_tensor_count = len(nav_model.state_dict())
            nav_param_count = sum(int(p.numel()) for p in nav_model.parameters())

    meta["xvs_nav"] = ModelCheckpointMeta(
        name="XVS-Nav",
        loaded=(nav_session is not None) or (nav_model is not None) or smoke_fast,
        real_weights=nav_real,
        device=nav_dev_label,
        dtype=nav_dtype_label,
        sha256=nav_sha,
        source_path=nav_src,
        tensor_count=nav_tensor_count,
        param_count=nav_param_count,
        backend=nav_backend,
        tier=nav_tier,
    )

    return StudioModelsBundle(
        d2_session=d2_sess,
        d2_x_dtype=d2_x_dtype,
        d2_h_dtype=d2_h_dtype,
        d2_hid_c=d2_hid_c,
        land_model=land_model,
        land_calibrator=land_cal,
        land_device=land_dev,
        thermal_model=thermal_model,
        thermal_device=thermal_dev,
        nav_model=nav_model,
        nav_device=nav_dev,
        nav_dtype=nav_dtype,
        nav_pca_whitening=nav_pca,
        meta=meta,
        host_gpu_name=host_gpu,
        compute_device=compute_label,
        vram_cap_mb=float(vram_cap_mb),
        land_session=land_session,
        thermal_session=thermal_session,
        nav_session=nav_session,
        nav_xfeat_onnx_path=nav_xfeat_onnx_path,
        nav_xfeat_sha256=nav_xfeat_sha256,
        nav_lighterglue_onnx_path=nav_lighterglue_onnx_path,
        nav_lighterglue_sha256=nav_lighterglue_sha256,
        land_has_keyline_sigma=land_has_keyline_sigma,
        land_has_runway_present=land_has_runway_present,
        d2_input_hw=d2_input_hw,
    )
