"""Regression tests for Colab G4 XVS-D2 Stage-4 true CSCWD VRAM probing & teacher-done resume."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from xvs.v2.colab_g4_utils import DriveCheckpointManager, snapshot_eager_state
from xvs.v2.xvs_d2.export_onnx import export_xvs_d2_web_onnx
from xvs.v2.xvs_d2.heads import CROP_PRIOR_BIAS
from xvs.v2.xvs_d2.tiers import build_xvs_d2
import xvs.v2.xvs_d2.train_g4 as train_g4_mod
from xvs.v2.xvs_d2.train_g4 import (
    SyntheticG3aClipDataset,
    XvsD2G4TrainConfig,
    _probe_batch_eager,
    run_xvs_d2_g4_training,
)
from xvs.v2.xvs_land.train_g4 import XvsLandG4Config, run_xvs_land_g4_training
import xvs.v2.xvs_nav.train_g4 as xvs_nav_train_g4_mod
from xvs.v2.xvs_nav.train_g4 import DinoV2BoQRetriever, XvsNavG4Config, build_xvs_nav_retriever
import xvs.v2.xvs_thermal.train_g4 as xvs_thermal_train_g4_mod
from xvs.v2.xvs_thermal.train_g4 import XvsThermalG4Config, download_hit_uav_and_irstd1k


def test_stage4_probe_batch_eager_exercises_distillation_and_restores_all_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `_probe_batch_eager` runs true teacher+CSCWD+CrossKD graph and restores student + adapters."""
    device = torch.device("cpu")
    teacher = build_xvs_d2("teacher", prior_bias=CROP_PRIOR_BIAS).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = build_xvs_d2("web", prior_bias=CROP_PRIOR_BIAS).to(device)
    t_spec = teacher.spec
    s_spec = student.spec
    cscwd_adapters = nn.ModuleDict(
        {
            "p2": nn.Conv2d(s_spec.c_fpn, t_spec.c_fpn, kernel_size=1, bias=False),
            "p3": nn.Conv2d(s_spec.c_fpn, t_spec.c_fpn, kernel_size=1, bias=False),
            "p4": nn.Conv2d(s_spec.c_fpn, t_spec.c_fpn, kernel_size=1, bias=False),
        }
    ).to(device)
    fp2_adapter = nn.Conv2d(s_spec.c_fpn, t_spec.c_fpn, kernel_size=1, bias=False).to(device)

    saved_student = snapshot_eager_state(student)
    saved_cscwd = snapshot_eager_state(cscwd_adapters)
    saved_fp2 = snapshot_eager_state(fp2_adapter)

    ds = SyntheticG3aClipDataset(num_pos=4, num_neg=0, height=64, width=64, clip_len=3, seed=123)
    observed_calls: list[dict[str, Any]] = []
    orig_train_step = train_g4_mod._train_one_clip_batch

    def _spy_train_step(*args: Any, **kwargs: Any) -> torch.Tensor:
        observed_calls.append(
            {
                "has_teacher": kwargs["teacher_model"] is teacher if "teacher_model" in kwargs else False,
                "has_cscwd": kwargs["cscwd_adapters"] is cscwd_adapters if "cscwd_adapters" in kwargs else False,
                "has_fp2": kwargs["fp2_adapter"] is fp2_adapter if "fp2_adapter" in kwargs else False,
                "lambda_kd": float(kwargs["lambda_kd"]) if "lambda_kd" in kwargs else 0.0,
            }
        )
        # Mutate student & adapter weights slightly to prove eager snapshot/restore undoes any drift
        with torch.no_grad():
            for p in student.parameters():
                if p.requires_grad:
                    p.add_(1e-3)
                    break
            cscwd_adapters["p2"].weight.add_(1e-3)
            fp2_adapter.weight.add_(1e-3)
        return orig_train_step(*args, **kwargs)

    monkeypatch.setattr(train_g4_mod, "_train_one_clip_batch", _spy_train_step)

    report = _probe_batch_eager(
        student,
        ds.positive_clips,
        target_hw=(64, 64),
        candidates=[2, 4],
        device=device,
        use_bf16=False,
        target_frac=0.86,
        headroom_gib=12.0,
        teacher_model=teacher,
        cscwd_adapters=cscwd_adapters,
        fp2_adapter=fp2_adapter,
        lambda_kd=0.5,
    )

    assert int(report["selected_batch_size"]) == 4
    assert len(observed_calls) == 2
    for call in observed_calls:
        assert call["has_teacher"] is True
        assert call["has_cscwd"] is True
        assert call["has_fp2"] is True
        assert call["lambda_kd"] == pytest.approx(0.5)

    # Bit-for-bit restoration of student, cscwd_adapters, and fp2_adapter
    for key, orig in saved_student.items():
        assert torch.equal(student.state_dict()[key], orig), f"Student drift on {key}"
    for key, orig in saved_cscwd.items():
        assert torch.equal(cscwd_adapters.state_dict()[key], orig), f"CSCWD adapter drift on {key}"
    for key, orig in saved_fp2.items():
        assert torch.equal(fp2_adapter.state_dict()[key], orig), f"FP2 adapter drift on {key}"

    # Gradients must be cleared after probing
    for mod in (student, cscwd_adapters, fp2_adapter):
        for p in mod.parameters():
            assert p.grad is None


def test_teacher_done_resume_skips_stages_1_to_3_and_runs_stage4_with_kd_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When `xvs_d2_teacher.pt` exists, Stages 1-3 skip cleanly (0s) and Stage 4 + FP16 ONNX succeed."""
    out_dir = tmp_path / "run_resume"
    ckpts_dir = out_dir / "checkpoints"
    drive_dir = tmp_path / "drive_ckpts"

    # Pre-seed completed teacher checkpoint on Drive mirror only (simulating fresh Colab VM resume)
    seeded_teacher = build_xvs_d2("teacher", prior_bias=CROP_PRIOR_BIAS)
    mgr = DriveCheckpointManager(local_dir=tmp_path / "seed_local", drive_dir=drive_dir)
    mgr.save(
        "xvs_d2_teacher",
        {
            "stage": "stage3_fullframe",
            "tier": "teacher",
            "artifact_tag": "SMOKE_LOCAL_CPU",
            "model_state_dict": {k: v.detach().cpu() for k, v in seeded_teacher.state_dict().items()},
        },
    )

    def _forbid_hnm(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("_mine_hard_negatives must NOT run when teacher is already completed!")

    monkeypatch.setattr(train_g4_mod, "_mine_hard_negatives", _forbid_hnm)

    probe_records: list[dict[str, Any]] = []
    orig_probe = train_g4_mod._probe_batch_eager

    def _spy_probe(*args: Any, **kwargs: Any) -> dict[str, Any]:
        probe_records.append(
            {
                "has_teacher": ("teacher_model" in kwargs and kwargs["teacher_model"] is not None),
                "has_cscwd": ("cscwd_adapters" in kwargs and kwargs["cscwd_adapters"] is not None),
                "has_fp2": ("fp2_adapter" in kwargs and kwargs["fp2_adapter"] is not None),
                "lambda_kd": float(kwargs["lambda_kd"]) if "lambda_kd" in kwargs else 0.0,
                "headroom_gib": float(kwargs["headroom_gib"]),
            }
        )
        return orig_probe(*args, **kwargs)

    monkeypatch.setattr(train_g4_mod, "_probe_batch_eager", _spy_probe)

    cfg = XvsD2G4TrainConfig(
        out_dir=out_dir,
        drive_ckpt_dir=drive_dir,
        smoke_local=True,
        epochs_stage1_warmup=2,
        epochs_stage2_hnm=2,
        epochs_stage3_fullframe=1,
        epochs_stage4_distill=1,
        student_tiers=("edge", "web"),
        vram_headroom_gib=8.0,
        use_torch_compile=False,
        seed=20261121,
    )
    summary = run_xvs_d2_g4_training(cfg)
    captured_out = capsys.readouterr().out

    assert summary["pass"] is True
    assert "Found completed teacher checkpoint -> skipping Stages 1-3 cleanly!" in captured_out

    # Stages 1-3 skipped completely
    assert summary["stage1_warmup"]["losses"] == []
    assert summary["stage1_warmup"]["vram_probe"] == {"skipped_resumed": True}
    assert summary["stage2_hnm"]["losses"] == []
    assert summary["stage2_hnm"]["mined_hard_negatives"] == 0
    assert summary["stage3_fullframe"]["losses"] == []
    assert summary["stage3_fullframe"]["full_frame_clips_built"] == 0
    assert summary["stage3_fullframe"]["vram_probe"] == {"skipped_resumed": True}

    # Only Stage 4 student probes ran (2 tiers: edge + web), both with true KD + >= 12 GiB headroom
    assert len(probe_records) == 2
    for rec in probe_records:
        assert rec["has_teacher"] is True
        assert rec["has_cscwd"] is True
        assert rec["has_fp2"] is True
        assert rec["lambda_kd"] == pytest.approx(0.5)
        assert rec["headroom_gib"] >= 12.0

    # Stage 4 executed and exported valid FP16 ONNX + local teacher mirror exists
    assert (ckpts_dir / "xvs_d2_teacher.pt").is_file()
    for tier in ("edge", "web"):
        assert summary["stage4_distill"][tier]["epochs_executed_this_run"] == 1
        assert (ckpts_dir / f"xvs_d2_{tier}.pt").is_file()
        assert (drive_dir / f"xvs_d2_{tier}.pt").is_file()

    onnx_rep = summary["onnx_export_report"]
    assert onnx_rep["all_checks_passed"] is True
    assert int(onnx_rep["fp16_onnx_bytes"]) <= 1_048_576
    assert (out_dir / "onnx" / "xvs_d2_web_fp16.onnx").is_file()


def test_export_xvs_d2_web_onnx_fp16_parity_untrained_trained_regime_and_real_g4_ckpt(
    tmp_path: Path,
) -> None:
    """Verify IEEE-754 FP16 parity gate on untrained init, trained Focal-loss sky logits, and real G4 ckpt."""
    # 1. Untrained init web model (preserves strict < 2e-3 absolute FP16 parity)
    init_model = build_xvs_d2("web", prior_bias=CROP_PRIOR_BIAS).eval()
    rep_init = export_xvs_d2_web_onnx(tmp_path / "init_web", model=init_model, height=256, width=256)
    assert rep_init["all_checks_passed"] is True
    assert rep_init["fp32_max_abs_diff"] < 1e-5
    assert rep_init["fp16_max_abs_diff"] < 2e-3
    assert rep_init["fp16_hm_prob_max_abs_diff"] < 2e-3
    assert rep_init["fp16_max_rel_diff"] < 5e-3
    assert rep_init["fp16_non_hm_max_abs_diff"] < 2e-3

    # 2. Trained-regime web model with Focal-loss sky logits around -10.0 (FP16 ULP ~ 0.0078125)
    trained_sim = build_xvs_d2("web", prior_bias=-10.0).eval()
    with torch.no_grad():
        trained_sim.heads.hm_out.bias.fill_(-10.0)
        trained_sim.heads.hm_out.weight.mul_(3.0)
    rep_sim = export_xvs_d2_web_onnx(tmp_path / "trained_sim_web", model=trained_sim, height=256, width=256)
    assert rep_sim["all_checks_passed"] is True
    assert rep_sim["fp32_max_abs_diff"] < 1e-5
    assert rep_sim["fp16_hm_prob_max_abs_diff"] < 2e-3
    assert rep_sim["fp16_max_rel_diff"] < 5e-3
    assert rep_sim["fp16_max_abs_diff"] < 2e-2
    assert int(rep_sim["fp16_onnx_bytes"]) <= 1_048_576
    assert rep_sim["forbidden_ops_fp16"] == []

    # 3. Real Colab G4 Blackwell trained xvs_d2_web.pt checkpoint (when present locally in output/)
    repo_root = Path(__file__).resolve().parents[2]
    real_g4_ckpt = (
        repo_root
        / "output"
        / "xvs"
        / "compute"
        / "results"
        / "v2_m2_01_xvs_d2_aot_g4_final"
        / "extracted"
        / "checkpoints"
        / "xvs_d2_web.pt"
    )
    if real_g4_ckpt.is_file():
        ckpt = torch.load(real_g4_ckpt, map_location="cpu", weights_only=False)
        real_web = build_xvs_d2("web", prior_bias=CROP_PRIOR_BIAS).eval()
        real_web.load_state_dict(ckpt["model_state_dict"], strict=True)
        rep_real = export_xvs_d2_web_onnx(tmp_path / "real_g4_web", model=real_web, height=256, width=256)
        assert rep_real["all_checks_passed"] is True
        assert rep_real["fp32_max_abs_diff"] < 1e-5
        assert rep_real["fp16_hm_prob_max_abs_diff"] < 2e-3
        assert rep_real["fp16_max_rel_diff"] < 5e-3
        assert rep_real["fp16_max_abs_diff"] < 2e-2
        assert int(rep_real["fp16_onnx_bytes"]) <= 1_048_576
        assert rep_real["forbidden_ops_fp16"] == []


def test_v2_m2_02_colab_py313_cuda13_and_dr15_19_quarantine_preflight() -> None:
    """Wave 11B pre-flight gate for Notebook 2 (`XVS-Nav`, `XVS-Land`, `XVS-ThermalNet`) on Py3.13/CUDA13."""
    repo_root = Path(__file__).resolve().parents[2]
    nb2_path = repo_root / "tools" / "colab" / "v2_m2_02_xvs_nav_land_thermal_g4.ipynb"
    nb = json.loads(nb2_path.read_text(encoding="utf-8"))

    assert nb["nbformat"] == 4
    assert int(nb["nbformat_minor"]) >= 5
    assert nb["metadata"]["colab"]["gpuType"] == "G4"
    assert nb["metadata"]["colab"]["machine_shape"] == "hm"

    code_cells = [
        "".join(cell["source"]) if isinstance(cell["source"], list) else str(cell["source"])
        for cell in nb["cells"]
        if cell["cell_type"] == "code"
    ]
    for idx, src in enumerate(code_cells):
        ast.parse(src, filename=f"{nb2_path.name}#cell{idx}")

    joined_code = "\n".join(code_cells)
    # 1. Py3.13 urllib3.response pre-import + ZstdError shim + zero GradScaler calls
    assert "import urllib3.response" in joined_code
    assert "zstd_mod.ZstdError = _zstd_compat.ZstdError" in joined_code
    assert "GradScaler(" not in joined_code
    assert joined_code.index("import urllib3.response") < joined_code.index("sys.modules['compression.zstd']")

    # 2. Every pip install invocation in NB2 enforces --only-binary=:all: and bounded timeout=
    tree = ast.parse(joined_code, filename=nb2_path.name)
    pip_calls = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "run":
            call_src = ast.unparse(node)
            if "'pip'" in call_src or '"pip"' in call_src:
                pip_calls += 1
                assert "--only-binary=:all:" in call_src, f"Missing --only-binary=:all: in {call_src}"
                assert any(kw.arg == "timeout" for kw in node.keywords), f"Missing timeout= in {call_src}"
    assert pip_calls >= 2
    assert "subprocess.TimeoutExpired" in joined_code

    # 3. DR-15..DR-19 quarantined source / OFAC / commercial-imagery check across NB2 & all 3 trainers
    quarantined_tokens = (
        "lard 2.0",
        "lardon",
        "msfs",
        "x-plane",
        "bing maps",
        "arcgis",
        "opencellid",
        "space-track",
        "infiray",
    )
    audit_files = (
        nb2_path,
        repo_root / "src" / "xvs" / "v2" / "xvs_nav" / "train_g4.py",
        repo_root / "src" / "xvs" / "v2" / "xvs_land" / "train_g4.py",
        repo_root / "src" / "xvs" / "v2" / "xvs_thermal" / "train_g4.py",
    )
    for fpath in audit_files:
        text_low = fpath.read_text(encoding="utf-8").lower()
        assert "gradscaler(" not in text_low, f"GradScaler call found in {fpath}"
        for tok in quarantined_tokens:
            assert tok not in text_low, f"Quarantined DR-15..19 token {tok!r} found in {fpath}"


def test_nb2_trainers_vram_headroom_and_candidate_floor() -> None:
    """Confirm all 3 NB2 G4 configs include batch size 1 fallback and >= 12.0 GiB VRAM headroom."""
    nav_cfg = XvsNavG4Config(out_dir=Path("tmp_nav"))
    land_cfg = XvsLandG4Config(out_dir=Path("tmp_land"), lard_manifest=None)
    thermal_cfg = XvsThermalG4Config(out_dir=Path("tmp_thermal"))

    for cfg in (nav_cfg, land_cfg, thermal_cfg):
        assert 1 in cfg.batch_candidates
        assert cfg.batch_candidates[0] == 1
        assert cfg.vram_headroom_gib >= 12.0


def test_nb2_fallbacks_and_xvs_land_stage2b_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify DINOv2 offline fallback, bounded thermal downloader timeout, and XVS-Land Stage-2B resume."""
    # 1. XVS-Nav graceful fallback when timm/HF pretrained DINOv2 fails offline
    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("simulated offline timm failure")

    monkeypatch.setattr(xvs_nav_train_g4_mod, "PretrainedDinoV2BoQRetriever", _boom)
    fallback_model = build_xvs_nav_retriever(
        XvsNavG4Config(out_dir=tmp_path / "nav_fb", use_pretrained_dinov2=True)
    )
    assert isinstance(fallback_model, DinoV2BoQRetriever)

    # 2. XVS-ThermalNet bounded socket timeout downloader fallback (never hangs, leaves no .tmp)
    seen_timeouts: list[float] = []

    def _timeout_urlopen(req: Any, timeout: float = 0.0) -> Any:
        seen_timeouts.append(float(timeout))
        raise TimeoutError("simulated github stall")

    monkeypatch.setattr(xvs_thermal_train_g4_mod.urllib.request, "urlopen", _timeout_urlopen)
    dl_cache = tmp_path / "thermal_dl"
    dl_status = download_hit_uav_and_irstd1k(dl_cache, timeout_s=15.0)
    assert seen_timeouts == [15.0, 15.0]
    assert dl_status["irstd1k"].startswith("fallback_physics_only")
    assert dl_status["hit_uav"].startswith("fallback_physics_only")
    assert list(dl_cache.glob("*.tmp")) == []

    # 3. XVS-Land Stage-2B checkpoint resume using real repo LARD manifest
    repo_root = Path(__file__).resolve().parents[2]
    lard_manifest = repo_root / "data" / "reference" / "lard_real_airports_manifest.json"
    out_dir = tmp_path / "land_resume"
    drive_dir = tmp_path / "land_drive"
    cfg = XvsLandG4Config(
        out_dir=out_dir,
        lard_manifest=lard_manifest,
        drive_ckpt_dir=drive_dir,
        smoke_local=True,
        stage2a_epochs=2,
        stage2b_epochs=2,
        steps_per_epoch=4,
        seed=20261121,
    )
    rep1 = run_xvs_land_g4_training(cfg)
    assert rep1["pass"] is True
    assert rep1["resumed_stage2a"] is False
    assert rep1["resumed_stage2b"] is False

    rep2 = run_xvs_land_g4_training(cfg)
    assert rep2["pass"] is True
    assert rep2["resumed_stage2a"] is True
    assert rep2["resumed_stage2b"] is True


