"""Gates `G4R-1` & `G4R-2` verification: `colab_g4_utils.py`, `xvs_d2/train_g4.py` & `v2_m2_01` Notebook."""

from __future__ import annotations

import ast
from compression import zstd as stdlib_zstd
import csv
import importlib
import importlib.machinery
import importlib.util
import inspect
import io
import json
import math
from pathlib import Path
import sys
import tarfile
import types
from typing import Any

import cv2
import numpy as np
import pytest
import torch

import scripts.xvs.v2_m2_stage as v2_m2_stage_mod
from xvs.v2.colab_g4_utils import (
    AutoVRAMBatchTuner,
    DriveCheckpointManager,
    build_zstd_compat_module,
    collect_blackwell_env_report,
    restore_eager_state,
    safe_colab_unassign,
    snapshot_eager_state,
)
from xvs.v2.xvs_d2.aot_wds import parse_aot_bbox_ltwh, split_flights_by_capture_day
from xvs.v2.xvs_d2.tiers import build_xvs_d2
from xvs.v2.xvs_d2.train_g4 import (
    AotFrameAnnotation,
    AotS3EncounterDataset,
    XvsD2G4TrainConfig,
    parse_aot_groundtruth_csv,
    run_xvs_d2_g4_training,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
NB01_PATH = REPO_ROOT / "tools" / "colab" / "v2_m2_01_xvs_d2_aot_g4.ipynb"

FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "university_1652",
    "sues_200",
    "lard_synthetic",
    "convnext_v2_nc",
    "dfine_obj365",
    "rf_detr_xl",
    "rf_detr_2xl",
    "lard_v2_yolo_agpl",
    "msfs",
    "xplane",
    "arcgis.com",
    "arcgisonline.com",
)


def test_auto_vram_batch_tuner_96gib_simulation_oom_and_eager_weight_restore() -> None:
    gib = 1024**3
    total_96gib = 96 * gib

    tuner = AutoVRAMBatchTuner(target_frac=0.86, safety_headroom_gib=8.0, device="cpu")
    # Simulated Blackwell G4 memory curve: 4 GiB base + 2.45 GiB/sample; bs=48 triggers OOM
    # At bs=32 -> 4 + 32*2.45 = 82.4 GiB -> 85.83% of 96 GiB (strictly in [80%, 91%] with >= 8 GiB headroom)
    model = build_xvs_d2("web")
    saved_state = snapshot_eager_state(model)

    def _step_fn(bs: int) -> None:
        if bs >= 40:
            raise RuntimeError("CUDA out of memory. Tried to allocate 12.00 GiB")
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.add_(1e-3)
                    break

    def _mem_probe_fn(bs: int) -> int:
        return int((4.0 + 2.45 * float(bs)) * float(gib))

    report = tuner.probe_batch_size(
        _step_fn,
        [4, 8, 16, 24, 32, 48, 64],
        total_vram_bytes=total_96gib,
        mem_probe_fn=_mem_probe_fn,
    )
    restore_eager_state(model, saved_state)

    expected_keys = {
        "selected_batch_size",
        "peak_vram_bytes",
        "total_vram_bytes",
        "utilization_frac",
        "probed_results",
        "device",
        "oom_rejected_candidates",
    }
    assert set(report.keys()) == expected_keys
    assert report["selected_batch_size"] == 32
    assert report["total_vram_bytes"] == total_96gib
    assert 0.80 <= float(report["utilization_frac"]) <= 0.91
    assert report["oom_rejected_candidates"] == [48]

    # Verify exact eager weight restoration after probing (BC-G4-m4)
    for key, orig_tensor in saved_state.items():
        assert torch.equal(model.state_dict()[key], orig_tensor), f"Weight drift on {key}"

    # Verify Blackwell env report exact 9-key schema & safe_colab_unassign outside Colab
    env = collect_blackwell_env_report()
    assert set(env.keys()) == {
        "gpu_name",
        "compute_capability",
        "total_vram_gib",
        "is_blackwell_sm120",
        "torch_version",
        "cuda_version",
        "bf16_supported",
        "tf32_enabled",
        "expandable_segments_env",
    }
    assert "expandable_segments" in str(env["expandable_segments_env"])

    exit_disabled = safe_colab_unassign(False)
    assert exit_disabled["requested"] is False and exit_disabled["unassigned"] is False
    exit_enabled = safe_colab_unassign(True)
    assert exit_enabled["requested"] is True and exit_enabled["unassigned"] is False


def test_drive_checkpoint_manager_atomic_save_and_resume(tmp_path: Path) -> None:
    local_dir = tmp_path / "local_ckpts"
    drive_dir = tmp_path / "drive_ckpts"
    mgr = DriveCheckpointManager(local_dir=local_dir, drive_dir=drive_dir)

    assert mgr.load_latest("stage1_") is None

    p1 = mgr.save("stage1_ep001", {"epoch": 1, "weight": torch.tensor([1.0, 2.0])})
    p2 = mgr.save("stage1_ep002", {"epoch": 2, "weight": torch.tensor([3.0, 4.0])})
    assert p1.is_file() and p2.is_file()
    assert (drive_dir / "stage1_ep002.pt").is_file()
    assert list(local_dir.glob("*.tmp")) == []
    assert list(drive_dir.glob("*.tmp")) == []

    latest = mgr.load_latest("stage1_")
    assert latest is not None
    assert int(latest["epoch"]) == 2
    assert torch.equal(latest["weight"], torch.tensor([3.0, 4.0]))

    # Remove local copy and prove Drive mirror fallback restores cleanly
    p2.unlink()
    p1.unlink()
    resumed_from_drive = mgr.load_latest("stage1_")
    assert resumed_from_drive is not None
    assert int(resumed_from_drive["epoch"]) == 2

    with pytest.raises(ValueError, match="Unsafe checkpoint tag"):
        mgr.save("../escape", {"epoch": 99})


def test_python311_zstd_compat_shim_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `D16-G4-B1`: strictly non-seekable `zstandard` shim passes `build_stage_archive`,
    `verify_results_archive`, and `tarfile.open(..., mode='r:').extractall(...)` without `OSError`.
    """
    calls: list[tuple[str, Any]] = []

    class _NonSeekableRejectStream:
        """Simulates PyPI `python-zstandard`'s non-seekable reader/writer raising OSError."""

        def seek(self, *_args: Any, **_kwargs: Any) -> int:
            raise OSError("cannot seek zstd decompression stream backwards")

        def tell(self) -> int:
            raise OSError("stream not seekable")

    class _FakeCompressor:
        def __init__(self, level: int = 3) -> None:
            self.level = int(level)
            calls.append(("compressor", self.level))

        def copy_stream(self, reader: Any, writer: Any) -> tuple[int, int]:
            raw = reader.read()
            compressed = stdlib_zstd.compress(raw, level=self.level)
            writer.write(compressed)
            return len(raw), len(compressed)

    class _FakeDecompressor:
        def __init__(self) -> None:
            calls.append(("decompressor", None))

        def copy_stream(self, reader: Any, writer: Any) -> tuple[int, int]:
            comp = reader.read()
            raw = stdlib_zstd.decompress(comp)
            writer.write(raw)
            return len(comp), len(raw)

    fake_zstandard = types.ModuleType("zstandard")
    fake_zstandard.ZstdCompressor = _FakeCompressor  # type: ignore[attr-defined]
    fake_zstandard.ZstdDecompressor = _FakeDecompressor  # type: ignore[attr-defined]

    def _forbidden_direct_open(*_args: Any, **_kwargs: Any) -> Any:
        # Must NEVER be returned directly to tarfile (would crash with non-seekable OSError)
        return _NonSeekableRejectStream()

    fake_zstandard.open = _forbidden_direct_open  # type: ignore[attr-defined]
    compat_mod = build_zstd_compat_module(fake_zstandard, max_spool_bytes=4096)

    # 1. Direct file round-trip
    out_file = tmp_path / "payload.zst"
    payload = b"OpenXVS v2.0 Blackwell G4 seekable zstd shim payload" * 128
    with compat_mod.open(out_file, mode="wb", level=5) as wf:
        wf.write(payload)
    with compat_mod.open(out_file, mode="rb") as rf:
        restored = rf.read()
        rf.seek(0)
        assert rf.read(16) == payload[:16]
    assert restored == payload
    assert ("compressor", 5) in calls
    assert ("decompressor", None) in calls

    # 2. Full tarfile + build_stage_archive + verify_results_archive + extractall(mode='r:') round-trip
    src_root = tmp_path / "stage_src"
    (src_root / "sub").mkdir(parents=True)
    (src_root / "sub" / "metrics.json").write_text(json.dumps({"ok": True, "val": 1.25}) + "\n", encoding="utf-8")
    (src_root / "weights.pt").write_bytes(b"")
    torch.save({"w": torch.tensor([1.0, 2.0])}, src_root / "weights.pt")

    monkeypatch.setattr(v2_m2_stage_mod, "zstd", compat_mod)
    packed = v2_m2_stage_mod.build_stage_archive(
        repo_root=src_root,
        out_dir=tmp_path / "archives",
        job_id="shim_nonseekable_test",
        include_paths=["sub/metrics.json", "weights.pt"],
    )
    verified = v2_m2_stage_mod.verify_results_archive(
        Path(packed["archive_path"]),
        expected_sha256=packed["archive_sha256"],
        check_tensors=True,
    )
    assert verified["pass"] is True

    unpack_dir = tmp_path / "unpacked"
    unpack_dir.mkdir()
    with compat_mod.open(Path(packed["archive_path"]), mode="rb") as zf:
        with tarfile.open(fileobj=zf, mode="r:") as tf:
            tf.extractall(unpack_dir, filter="data")
    assert (unpack_dir / "sub" / "metrics.json").is_file()
    assert (unpack_dir / "weights.pt").is_file()


def test_aot_groundtruth_csv_border_clip_degenerate_drop_circle_mask_and_capture_day_split(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "groundtruth.csv"
    fieldnames = [
        "",
        "time",
        "flight_id",
        "img_name",
        "frame",
        "id",
        "range_distance_m",
        "is_above_horizon",
        "size_width",
        "size_height",
        "gt_left",
        "gt_top",
        "gt_right",
        "gt_bottom",
    ]
    # 5 distinct UTC days (spaced by 86400e9 ns)
    base_ns = 1_560_000_000_000_000_000
    day_ns = 86_400_000_000_000

    rows = [
        # Negative target-free frame
        ["0", "0.0", "f0", f"{base_ns + 0 * day_ns:019d}f0.png", "0", "", "", "1.0", "", "", "", "", "", ""],
        # Border overshoot left/top -> clipped to [0, 0, 37.5, 40.0] (valid non-circle)
        ["1", "0.1", "f0", f"{base_ns + 0 * day_ns:019d}f0_1.png", "1", "a1", "450.0", "1.0", "40", "40", "-2.5", "-1.0", "37.5", "39.0"],
        # Border overshoot right/bottom -> clipped to [2430, 2030, 18, 18]
        ["2", "0.2", "f1", f"{base_ns + 1 * day_ns:019d}f1.png", "2", "a2", "510.0", "-1.0", "20", "20", "2430.0", "2030.0", "2452.0", "2055.0"],
        # Fixed 6x6 circle approx -> m_box == 0.0
        ["3", "0.3", "f2", f"{base_ns + 2 * day_ns:019d}f2.png", "3", "a3", "620.0", "1.0", "6", "6", "1000.0", "900.0", "1006.0", "906.0"],
        # Fixed 16x16 circle approx -> m_box == 0.0
        ["4", "0.4", "f3", f"{base_ns + 3 * day_ns:019d}f3.png", "4", "a4", "320.0", "-1.0", "16", "16", "500.0", "500.0", "516.0", "516.0"],
        # Degenerate inverted box (x2 <= x1) -> dropped
        ["5", "0.5", "f4", f"{base_ns + 4 * day_ns:019d}f4.png", "5", "a5", "400.0", "1.0", "0", "10", "100.0", "100.0", "95.0", "110.0"],
        # Degenerate out-of-image box clipped to zero width -> dropped
        ["6", "0.6", "f4", f"{base_ns + 4 * day_ns:019d}f4_1.png", "6", "a6", "400.0", "1.0", "10", "10", "2450.0", "100.0", "2460.0", "110.0"],
        # Degenerate missing bottom coordinate -> dropped
        ["7", "0.7", "f4", f"{base_ns + 4 * day_ns:019d}f4_2.png", "7", "a7", "400.0", "1.0", "10", "10", "100.0", "100.0", "110.0", ""],
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        writer.writerows(rows)

    parsed = parse_aot_groundtruth_csv(csv_path, part="part1")
    assert parsed.clipped_border_boxes >= 2
    assert parsed.dropped_degenerate_boxes == 3
    assert parsed.circle_approx_boxes == 2
    assert len(parsed.positive_frames) == 4
    assert len(parsed.negative_frames) == 4

    # Verify exact m_box=0 on 6x6 / 16x16 and m_box=1 on clipped non-circle boxes
    m_boxes = sorted(b.m_box for ann in parsed.positive_frames for b in ann.boxes)
    assert m_boxes == [0.0, 0.0, 1.0, 1.0]

    # Verify strict 0% capture_day overlap across train/val/test
    splits = split_flights_by_capture_day(parsed.flight_records, seed=20261121)
    tr_days = {r["capture_day"] for r in splits["train"]}
    va_days = {r["capture_day"] for r in splits["val"]}
    te_days = {r["capture_day"] for r in splits["test"]}
    assert tr_days.isdisjoint(va_days)
    assert tr_days.isdisjoint(te_days)
    assert va_days.isdisjoint(te_days)


def test_xvs_d2_g4_smoke_local_4_stage_resume_eval_and_onnx_export(tmp_path: Path) -> None:
    out_dir = tmp_path / "xvs_d2_run"
    drive_dir = tmp_path / "drive_mirror"

    assert XvsD2G4TrainConfig(out_dir=out_dir).max_full_frame_clips == 16

    cfg1 = XvsD2G4TrainConfig(
        out_dir=out_dir,
        drive_ckpt_dir=drive_dir,
        smoke_local=True,
        max_full_frame_clips=2,
        epochs_stage1_warmup=1,
        epochs_stage2_hnm=1,
        epochs_stage3_fullframe=1,
        epochs_stage4_distill=1,
        use_torch_compile=False,
        seed=20261121,
    )
    summary1 = run_xvs_d2_g4_training(cfg1)
    assert summary1["pass"] is True
    assert summary1["artifact_tag"] == "SMOKE_LOCAL_CPU"
    assert summary1["stage1_warmup"]["resumed_from_epoch"] == 0
    assert len(summary1["stage1_warmup"]["losses"]) == 1
    assert summary1["stage2_hnm"]["resumed_from_epoch"] == 0
    assert summary1["stage2_hnm"]["mined_hard_negatives"] >= 1
    assert summary1["stage3_fullframe"]["resumed_from_epoch"] == 0
    assert summary1["stage3_fullframe"]["native_resolution_hw"] == [96, 96]
    assert summary1["stage3_fullframe"]["upsampled_from_crop"] is False
    assert summary1["stage3_fullframe"]["max_full_frame_clips_cap"] == 2
    assert summary1["stage3_fullframe"]["full_frame_clips_built"] == 3
    assert len(summary1["stage3_fullframe"]["losses"]) == 1
    assert set(summary1["stage4_distill"].keys()) == {"edge", "edge_slim", "web"}

    for tier in ("teacher", "edge", "edge_slim", "web"):
        ckpt_p = out_dir / "checkpoints" / f"xvs_d2_{tier}.pt"
        assert ckpt_p.is_file()
        assert (drive_dir / f"xvs_d2_{tier}.pt").is_file()
        loaded_ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        assert loaded_ckpt["artifact_tag"] == "SMOKE_LOCAL_CPU"

    onnx_rep = summary1["onnx_export_report"]
    assert onnx_rep["all_checks_passed"] is True
    assert int(onnx_rep["fp16_onnx_bytes"]) <= 1_048_576
    assert len(onnx_rep["forbidden_ops_fp32"]) == 0
    assert len(onnx_rep["forbidden_ops_fp16"]) == 0

    for key in ("pd_at_fpi_0p1", "pd_at_fpi_0p5", "pd_at_fpi_1p0", "ap50", "centroid_rmse_px", "above_horizon_pd", "below_horizon_pd"):
        assert math.isfinite(float(summary1["eval_metrics"]["teacher"][key]))
        assert math.isfinite(float(summary1["eval_metrics"]["web"][key]))

    # Re-run with Stage 1=2 ep & Stages 2-4=1 ep -> prove Stage 1 resumes + Stages 2, 3, 4 skip cleanly!
    cfg2 = XvsD2G4TrainConfig(
        out_dir=out_dir,
        drive_ckpt_dir=drive_dir,
        smoke_local=True,
        epochs_stage1_warmup=2,
        epochs_stage2_hnm=1,
        epochs_stage3_fullframe=1,
        epochs_stage4_distill=1,
        use_torch_compile=False,
        seed=20261121,
    )
    summary2 = run_xvs_d2_g4_training(cfg2)
    assert summary2["pass"] is True
    assert summary2["stage1_warmup"]["resumed_from_epoch"] == 1
    assert len(summary2["stage1_warmup"]["losses"]) == 1
    assert summary2["stage2_hnm"]["resumed_from_epoch"] == 1
    assert len(summary2["stage2_hnm"]["losses"]) == 0
    assert summary2["stage3_fullframe"]["resumed_from_epoch"] == 1
    assert len(summary2["stage3_fullframe"]["losses"]) == 0
    assert summary2["stage3_fullframe"]["full_frame_clips_built"] == 0
    for tier in ("edge", "edge_slim", "web"):
        assert summary2["stage4_distill"][tier]["resumed_from_epoch"] == 1
        assert summary2["stage4_distill"][tier]["epochs_executed_this_run"] == 0


def test_aot_native_full_frame_clips_from_cached_pngs(tmp_path: Path) -> None:
    """Verify `D16-G4-M5`: Stage 3 builds genuine native-res clips from cached PNGs (no crop upsampling)."""
    ds = AotS3EncounterDataset(cache_dir=tmp_path / "aot_cache", crop_hw=(64, 64), clip_len=3)
    paths: list[Path] = []
    anns: list[AotFrameAnnotation] = []
    native_h, native_w = 128, 160
    box = parse_aot_bbox_ltwh([70.0, 50.0, 12.0, 10.0], img_w=native_w, img_h=native_h)

    for idx in range(5):
        p = tmp_path / f"frame_{idx:02d}.png"
        img = np.full((native_h, native_w), 140 + idx * 3, dtype=np.uint8)
        img[50:60, 70:82] = 60
        assert cv2.imwrite(str(p), img)
        paths.append(p)
        anns.append(
            AotFrameAnnotation(
                part="part1",
                flight_id="f_native",
                img_name=f"156000000000000000{idx}f_native.png",
                frame_idx=idx,
                capture_day="2019-06-08",
                horizon="above",
                boxes=(box,),
            )
        )
    ds.train_pos_win_records.append({"paths": paths, "anns": anns})

    native_clips = ds.build_native_full_frame_clips(full_hw=(native_h, native_w))
    assert len(native_clips) == 1
    clip = native_clips[0]
    assert tuple(clip["x_clip"].shape) == (3, 6, native_h, native_w)
    assert clip["native_hw"] == [native_h, native_w]
    assert abs(float(clip["boxes_per_step"][-1]["cx"]) - 76.0) < 1e-6
    assert abs(float(clip["boxes_per_step"][-1]["cy"]) - 55.0) < 1e-6


def test_notebook_01_ast_importlib_signatures_and_hygiene() -> None:
    assert NB01_PATH.is_file(), f"Missing notebook: {NB01_PATH}"
    raw_text = NB01_PATH.read_text(encoding="utf-8")
    nb = json.loads(raw_text)

    assert nb["nbformat"] == 4
    assert int(nb["nbformat_minor"]) >= 5
    assert nb["metadata"]["accelerator"] == "GPU"

    lower_text = raw_text.lower()
    for bad in FORBIDDEN_SUBSTRINGS:
        assert bad not in lower_text, f"Forbidden substring {bad!r} found in {NB01_PATH.name}"

    code_parts: list[str] = []
    imported_symbols: dict[str, Any] = {}
    symbol_count = 0
    call_count = 0

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else str(cell["source"])
        for line in src.splitlines():
            stripped = line.lstrip()
            assert not stripped.startswith(("!", "%")), f"IPython magic forbidden in cell {idx}: {stripped!r}"
        tree = ast.parse(src, filename=f"{NB01_PATH.name}#cell{idx}")
        code_parts.append(src)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(("xvs.", "scripts.xvs.")):
                mod = importlib.import_module(node.module)
                for alias in node.names:
                    assert hasattr(mod, alias.name), f"Missing symbol {alias.name} in {node.module}"
                    imported_symbols[alias.asname or alias.name] = getattr(mod, alias.name)
                    symbol_count += 1

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in imported_symbols:
                target = imported_symbols[node.func.id]
                if callable(target):
                    sig = inspect.signature(target)
                    dummy_pos = [object() for _ in node.args]
                    dummy_kw = {str(kw.arg): object() for kw in node.keywords if kw.arg is not None}
                    sig.bind(*dummy_pos, **dummy_kw)
                    call_count += 1

    joined = "\n".join(code_parts)
    for required in (
        "COLAB_G4_GPU_ARTIFACT",
        "from compression import zstd",
        "except ImportError",
        "import zstandard",
        "SpooledTemporaryFile",
        "copy_stream",
        "import compression.zstd",
        "ModuleSpec('compression'",
        "/content/openxvs_v2_m2_g4_stage.tar.zst",
        "xvs_v2_g4_stage/openxvs_v2_m2_g4_stage.tar.zst",
        "xvs_v2_g4_runs/openxvs_v2_m2_g4_stage.tar.zst",
        "FileNotFoundError",
        "USE_DRIVE",
        "AUTO_UNASSIGN_ON_EXIT",
        "safe_colab_unassign",
        "xvs.v2.license_guard",
        "xvs.v2.colab_g4_utils",
        "xvs.v2.xvs_d2",
        "scripts.xvs.v2_m2_stage",
    ):
        assert required in joined, f"Missing required token {required!r} in {NB01_PATH.name}"

    assert nb["metadata"]["language_info"]["version"] == "3.12"
    assert symbol_count >= 12
    assert call_count >= 12


def test_python313_zstd_shim_spec_allows_repeated_find_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Python 3.13 Cell-1 re-run safety: `__spec__` is set so `find_spec` never raises ValueError."""
    fake_zstd = types.ModuleType("zstandard")
    compat_mod = build_zstd_compat_module(fake_zstd)
    assert compat_mod.__spec__ is not None

    comp_pkg = types.ModuleType("compression")
    comp_pkg.__path__ = []  # type: ignore[attr-defined]
    comp_pkg.__spec__ = importlib.machinery.ModuleSpec("compression", loader=None, is_package=True)
    comp_pkg.zstd = compat_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "compression", comp_pkg)
    monkeypatch.setitem(sys.modules, "compression.zstd", compat_mod)

    spec_parent = importlib.util.find_spec("compression")
    spec_child = importlib.util.find_spec("compression.zstd")
    assert spec_parent is not None and spec_parent.name == "compression"
    assert spec_child is not None and spec_child.name == "compression.zstd"


def test_aot_download_key_uses_direct_https_no_aws_cli_fork(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify AOT PNG downloads disable `aws` CLI subprocess fork (`prefer_aws_cli=False`)."""
    import xvs.v2.xvs_d2.train_g4 as train_g4_mod

    captured_kwargs: list[dict[str, Any]] = []

    def _fake_download(
        *,
        s3_key: str,
        dest_path: Path,
        s3_bucket: str,
        https_base: str,
        prefer_aws_cli: bool = True,
    ) -> Path:
        captured_kwargs.append(
            {
                "s3_key": s3_key,
                "dest_path": dest_path,
                "s3_bucket": s3_bucket,
                "https_base": https_base,
                "prefer_aws_cli": prefer_aws_cli,
            }
        )
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(b"ok")
        return dest_path

    monkeypatch.setattr(train_g4_mod, "download_s3_object_with_https_fallback", _fake_download)
    ds = AotS3EncounterDataset(cache_dir=tmp_path / "aot_cache", crop_hw=(64, 64), clip_len=3)
    out = ds._download_key("part1/Images/f0/test.png")
    assert out.is_file()
    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["prefer_aws_cli"] is False
