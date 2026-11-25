"""Shared Colab G4 Blackwell (`96 GB VRAM`, `sm_120`) Runtime Utilities (`SP-2B` §3.0).

Implements the normative shared interface (`BC-G4-B2`, `BC-G4-M1`, `BC-G4-m4`):
- `AutoVRAMBatchTuner`: eager VRAM batch probing BEFORE `torch.compile`, catching OOM cleanly
  and saturating 82%-90% of available VRAM with safety headroom.
- `DriveCheckpointManager`: atomic `.pt.tmp -> os.replace` + `fsync`/`os.sync()` checkpoint
  persistence with optional Google Drive mirror and resumable `load_latest`.
- `collect_blackwell_env_report`: enables TF32 + `expandable_segments:True` and reports exact
  9-key Blackwell / CUDA / BF16 runtime capabilities.
- `safe_colab_unassign`: safe `finally:`-block `google.colab.runtime.unassign()` guard.
- `install_colab_zstd_shim`: Python 3.11/3.12 `compression.zstd` compatibility shim translating
  `level=` to `zstandard.ZstdCompressor(level=level)` so `scripts.xvs.v2_m2_stage` stays 100%
  pure Python 3.14 stdlib while running seamlessly on Colab Python 3.11/3.12.
"""

from __future__ import annotations

import gc
import importlib.machinery
import os
from pathlib import Path
import sys
import tempfile
import types
from typing import Any, Callable, Sequence

import torch
from torch import nn

__all__ = [
    "AutoVRAMBatchTuner",
    "DriveCheckpointManager",
    "collect_blackwell_env_report",
    "safe_colab_unassign",
    "install_colab_zstd_shim",
    "build_zstd_compat_module",
    "snapshot_eager_state",
    "restore_eager_state",
]


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return ("out of memory" in msg) or ("cuda error: out of memory" in msg)
    return False


def _clear_cuda_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def snapshot_eager_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot eager model weights onto CPU before VRAM probing (`BC-G4-m4`)."""
    raw = getattr(model, "_orig_mod", model)
    return {k: v.detach().cpu().clone() for k, v in raw.state_dict().items()}


def restore_eager_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Restore eager model weights and clear gradients after VRAM probing (`BC-G4-m4`)."""
    raw = getattr(model, "_orig_mod", model)
    raw.load_state_dict(state, strict=True)
    raw.zero_grad(set_to_none=True)


class AutoVRAMBatchTuner:
    """Eager VRAM batch-size prober targeting 82%-90% GPU memory utilization (`SP-2B` §3.0).

    Always run `probe_batch_size(...)` on the eager model BEFORE `torch.compile(model)` and
    restore initial weights via `snapshot_eager_state` / `restore_eager_state` (`BC-G4-m4`).
    """

    def __init__(
        self,
        target_frac: float = 0.86,
        safety_headroom_gib: float = 8.0,
        device: str | torch.device | None = None,
    ) -> None:
        if not (0.10 <= float(target_frac) <= 0.98):
            raise ValueError(f"target_frac must be in [0.10, 0.98], got {target_frac}")
        if float(safety_headroom_gib) < 0.0:
            raise ValueError(f"safety_headroom_gib must be >= 0.0, got {safety_headroom_gib}")
        self.target_frac = float(target_frac)
        self.safety_headroom_gib = float(safety_headroom_gib)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def probe_batch_size(
        self,
        step_fn: Callable[[int], Any],
        candidates: Sequence[int],
        *,
        total_vram_bytes: int | None = None,
        mem_probe_fn: Callable[[int], int] | None = None,
    ) -> dict[str, Any]:
        """Probe candidate batch sizes ascending until target fraction or headroom is reached."""
        sorted_candidates = sorted({int(c) for c in candidates if int(c) > 0})
        if not sorted_candidates:
            raise ValueError("candidates must contain at least one positive integer batch size")

        if total_vram_bytes is not None:
            total_bytes = int(total_vram_bytes)
        elif self.device.type == "cuda" and torch.cuda.is_available():
            total_bytes = int(torch.cuda.get_device_properties(self.device).total_memory)
        else:
            total_bytes = int(8 * (1024**3))

        if total_bytes <= 0:
            raise ValueError(f"total_vram_bytes must be > 0, got {total_bytes}")

        headroom_bytes = int(self.safety_headroom_gib * float(1024**3))
        max_allowed_bytes = min(
            int(float(total_bytes) * self.target_frac),
            max(1, total_bytes - headroom_bytes),
        )

        probed_results: list[dict[str, Any]] = []
        oom_rejected: list[int] = []
        selected_bs: int | None = None
        selected_peak: int = 0
        fallback_bs: int = sorted_candidates[0]
        fallback_peak: int = 0
        have_fallback_peak = False

        use_real_cuda_stats = (
            mem_probe_fn is None and self.device.type == "cuda" and torch.cuda.is_available()
        )

        for bs in sorted_candidates:
            if use_real_cuda_stats:
                _clear_cuda_cache(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)

            try:
                step_fn(bs)
                if mem_probe_fn is not None:
                    peak_bytes = int(mem_probe_fn(bs))
                elif use_real_cuda_stats:
                    torch.cuda.synchronize(self.device)
                    peak_bytes = int(torch.cuda.max_memory_allocated(self.device))
                else:
                    peak_bytes = 0
            except BaseException as exc:
                if not _is_oom_error(exc):
                    raise
                oom_rejected.append(bs)
                probed_results.append(
                    {
                        "batch_size": bs,
                        "status": "oom",
                        "peak_vram_bytes": None,
                        "utilization_frac": None,
                    }
                )
                _clear_cuda_cache(self.device)
                break

            util_frac = float(peak_bytes) / float(total_bytes)
            if not have_fallback_peak:
                fallback_bs = bs
                fallback_peak = peak_bytes
                have_fallback_peak = True

            if peak_bytes <= max_allowed_bytes:
                selected_bs = bs
                selected_peak = peak_bytes
                probed_results.append(
                    {
                        "batch_size": bs,
                        "status": "ok",
                        "peak_vram_bytes": peak_bytes,
                        "utilization_frac": util_frac,
                    }
                )
            else:
                probed_results.append(
                    {
                        "batch_size": bs,
                        "status": "over_budget",
                        "peak_vram_bytes": peak_bytes,
                        "utilization_frac": util_frac,
                    }
                )
                _clear_cuda_cache(self.device)
                break

            _clear_cuda_cache(self.device)

        if selected_bs is None:
            selected_bs = fallback_bs
            selected_peak = fallback_peak

        return {
            "selected_batch_size": int(selected_bs),
            "peak_vram_bytes": int(selected_peak),
            "total_vram_bytes": int(total_bytes),
            "utilization_frac": float(selected_peak) / float(total_bytes),
            "probed_results": probed_results,
            "device": str(self.device),
            "oom_rejected_candidates": oom_rejected,
        }


def _atomic_torch_save(path: Path, state: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    if hasattr(os, "sync"):
        os.sync()
    return path


class DriveCheckpointManager:
    """Atomic `.pt.tmp -> os.replace` checkpoint persistence with Drive mirror (`SP-2B` §3.0)."""

    def __init__(
        self,
        local_dir: Path | str,
        drive_dir: Path | str | None = None,
    ) -> None:
        self.local_dir = Path(local_dir)
        self.drive_dir = Path(drive_dir) if drive_dir is not None else None
        self.local_dir.mkdir(parents=True, exist_ok=True)
        if self.drive_dir is not None:
            self.drive_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_tag(tag: str) -> str:
        clean = str(tag).strip()
        if not clean or "/" in clean or "\\" in clean or ".." in clean:
            raise ValueError(f"Unsafe checkpoint tag: {tag!r}")
        return clean

    def save(self, tag: str, state: dict[str, Any]) -> Path:
        """Atomically persist `state` to `local_dir/<tag>.pt` and optionally mirror to `drive_dir`."""
        clean_tag = self._validate_tag(tag)
        filename = f"{clean_tag}.pt" if not clean_tag.endswith(".pt") else clean_tag
        local_path = _atomic_torch_save(self.local_dir / filename, state)
        if self.drive_dir is not None:
            _atomic_torch_save(self.drive_dir / filename, state)
        return local_path

    def load_latest(self, tag_prefix: str = "") -> dict[str, Any] | None:
        """Load the most recently modified valid checkpoint matching `<tag_prefix>*.pt`."""
        candidates: list[Path] = []
        pattern = f"{tag_prefix}*.pt"
        if self.local_dir.is_dir():
            for p in self.local_dir.glob(pattern):
                if p.is_file() and not p.name.endswith(".tmp"):
                    candidates.append(p)
        if self.drive_dir is not None and self.drive_dir.is_dir():
            for p in self.drive_dir.glob(pattern):
                if p.is_file() and not p.name.endswith(".tmp"):
                    candidates.append(p)

        if not candidates:
            return None

        candidates.sort(key=lambda p: (p.stat().st_mtime_ns, p.name), reverse=True)
        for path in candidates:
            try:
                loaded = torch.load(path, map_location="cpu", weights_only=False)
                if isinstance(loaded, dict):
                    return loaded
            except Exception:
                continue
        return None


def collect_blackwell_env_report() -> dict[str, Any]:
    """Configure TF32 & expandable CUDA allocator segments and report GPU capabilities (`SP-2B` §3.0)."""
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ or not os.environ["PYTORCH_CUDA_ALLOC_CONF"]:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    expandable_env = os.environ["PYTORCH_CUDA_ALLOC_CONF"]

    tf32_enabled = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
        tf32_enabled = bool(torch.backends.cuda.matmul.allow_tf32)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    cuda_avail = torch.cuda.is_available()
    gpu_name = "CPU"
    compute_capability = "0.0"
    total_vram_gib = 0.0
    is_blackwell_sm120 = False
    bf16_supported = False

    if cuda_avail:
        props = torch.cuda.get_device_properties(0)
        gpu_name = str(props.name)
        compute_capability = f"{props.major}.{props.minor}"
        total_vram_gib = round(float(props.total_memory) / float(1024**3), 3)
        is_blackwell_sm120 = (int(props.major) > 12) or (int(props.major) == 12 and int(props.minor) >= 0)
        bf16_supported = bool(torch.cuda.is_bf16_supported())
    else:
        bf16_supported = hasattr(torch, "bfloat16")

    cuda_ver = str(torch.version.cuda) if torch.version.cuda is not None else "none"

    return {
        "gpu_name": gpu_name,
        "compute_capability": compute_capability,
        "total_vram_gib": float(total_vram_gib),
        "is_blackwell_sm120": bool(is_blackwell_sm120),
        "torch_version": str(torch.__version__),
        "cuda_version": cuda_ver,
        "bf16_supported": bool(bf16_supported),
        "tf32_enabled": bool(tf32_enabled),
        "expandable_segments_env": str(expandable_env),
    }


def safe_colab_unassign(auto_unassign: bool) -> dict[str, Any]:
    """Safely release a Google Colab runtime inside a `finally:` block (`SP-2B` §3.0)."""
    if not bool(auto_unassign):
        return {
            "requested": False,
            "unassigned": False,
            "reason": "auto_unassign_disabled",
        }
    try:
        from google.colab import runtime as colab_runtime  # type: ignore[import-not-found]
    except ImportError:
        return {
            "requested": True,
            "unassigned": False,
            "reason": "not_running_in_google_colab",
        }
    try:
        colab_runtime.unassign()
        return {
            "requested": True,
            "unassigned": True,
            "reason": "unassigned_via_google_colab_runtime",
        }
    except Exception as exc:
        return {
            "requested": True,
            "unassigned": False,
            "reason": f"unassign_error:{type(exc).__name__}",
        }


class _SpoolingZstdWriter:
    """Seekable write buffer that compresses via `ZstdCompressor.copy_stream` on close (`D16-G4-B1`)."""

    def __init__(
        self,
        target: Path | str | Any,
        zstandard_mod: Any,
        *,
        level: int | None = None,
        max_mem_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self._target = target
        self._zstandard = zstandard_mod
        self._level = 3 if level is None else int(level)
        self._spool = tempfile.SpooledTemporaryFile(max_size=int(max_mem_bytes))
        self._closed = False

    def write(self, data: bytes | bytearray | memoryview) -> int:
        return int(self._spool.write(data))

    def writelines(self, lines: Any) -> None:
        self._spool.writelines(lines)

    def flush(self) -> None:
        self._spool.flush()

    def tell(self) -> int:
        return int(self._spool.tell())

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return int(self._spool.seek(offset, whence))

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._spool.flush()
            self._spool.seek(0)
            cctx = self._zstandard.ZstdCompressor(level=self._level)
            if isinstance(self._target, (str, os.PathLike)):
                dest = Path(self._target)
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(f"{dest.name}.zst.tmp")
                try:
                    with open(tmp, "wb") as out_f:
                        cctx.copy_stream(self._spool, out_f)
                        out_f.flush()
                        os.fsync(out_f.fileno())
                    os.replace(tmp, dest)
                finally:
                    if tmp.exists():
                        try:
                            tmp.unlink()
                        except OSError:
                            pass
            else:
                cctx.copy_stream(self._spool, self._target)
                if hasattr(self._target, "flush"):
                    self._target.flush()
        finally:
            self._spool.close()

    def __enter__(self) -> _SpoolingZstdWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is not None:
            if not self._closed:
                self._closed = True
                self._spool.close()
            return
        self.close()


def build_zstd_compat_module(
    zstandard_mod: Any,
    *,
    max_spool_bytes: int = 64 * 1024 * 1024,
) -> types.ModuleType:
    """Build a fully seekable `compression.zstd`-compatible module around `zstandard` (`D16-G4-B1`).

    Read mode decompresses via `ZstdDecompressor().copy_stream(...)` into a seekable
    `tempfile.SpooledTemporaryFile`, and write mode spools into a seekable writer that compresses
    atomically via `ZstdCompressor(level=...).copy_stream(...)` on close so both
    `tarfile.open(..., mode="r:")` and `tarfile.open(..., mode="w" / "w|")` work identically on
    Colab Python 3.11/3.12 without `OSError: stream not seekable`.
    """
    zstd_mod = types.ModuleType("compression.zstd")
    zstd_mod.__spec__ = importlib.machinery.ModuleSpec("compression.zstd", loader=None)

    def _compat_open(
        filename: Path | str | Any,
        mode: str = "rb",
        *,
        level: int | None = None,
        **kwargs: Any,
    ) -> Any:
        _ = kwargs
        if "w" in mode or "a" in mode or "x" in mode:
            return _SpoolingZstdWriter(
                filename,
                zstandard_mod,
                level=level,
                max_mem_bytes=max_spool_bytes,
            )
        spool = tempfile.SpooledTemporaryFile(max_size=int(max_spool_bytes))
        dctx = zstandard_mod.ZstdDecompressor()
        try:
            if isinstance(filename, (str, os.PathLike)):
                with open(filename, "rb") as raw_f:
                    dctx.copy_stream(raw_f, spool)
            else:
                dctx.copy_stream(filename, spool)
            spool.seek(0)
            return spool
        except BaseException:
            spool.close()
            raise

    zstd_mod.open = _compat_open  # type: ignore[attr-defined]
    zstd_mod.ZstdError = getattr(zstandard_mod, "ZstdError", OSError)  # type: ignore[attr-defined]
    return zstd_mod


def install_colab_zstd_shim() -> str:
    """Ensure `from compression import zstd` works on both Python 3.14+ and Colab 3.11/3.12 (`BC-G4-M1`)."""
    try:
        from compression import zstd as _stdlib_zstd  # type: ignore[import-not-found]

        _ = _stdlib_zstd
        return "stdlib_compression_zstd"
    except ImportError:
        import zstandard  # type: ignore[import-not-found]

        try:
            import urllib3.response  # type: ignore[import-not-found]  # noqa: F401
        except Exception:
            pass

        comp_pkg = types.ModuleType("compression")
        comp_pkg.__path__ = []  # type: ignore[attr-defined]
        comp_pkg.__spec__ = importlib.machinery.ModuleSpec("compression", loader=None, is_package=True)
        zstd_mod = build_zstd_compat_module(zstandard)
        comp_pkg.zstd = zstd_mod  # type: ignore[attr-defined]
        sys.modules["compression"] = comp_pkg
        sys.modules["compression.zstd"] = zstd_mod
        return "zstandard_shim"
