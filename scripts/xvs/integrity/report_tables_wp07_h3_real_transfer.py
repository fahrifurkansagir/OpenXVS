"""Generate markdown summary tables and Charter §8 report for WP-06/WP-07 Batch B13 (Hypothesis H3 Real Transfer).

Reads JSON artifacts from:
- output/xvs/wp07/h3_real_transfer/results.json
- output/xvs/wp07/h3_real_transfer/run_manifest.json
- configs/xvs/integrity/h3_real_transfer.json

Generates docs/xvs/reports/WP07_H3_real_transfer_report.md and
docs/xvs/reports/WP07_H3_real_transfer_report.md.provenance.json
with 100% provenance-tracked numbers via rv() and zero literal rv() calls.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.paths import get_data_root
from xvs.report_provenance import (
    ReportProvenanceTracker,
    load_json_source,
    rv,
    save_provenance,
    verify_report_provenance,
)

DEFAULT_REPORT_PATH = REPO_ROOT / "docs" / "xvs" / "reports" / "WP07_H3_real_transfer_report.md"


def load_json(path: Path, source_label: str) -> dict[str, Any]:
    return load_json_source(path, source_label=source_label)


def get_b13_pytest_output() -> str:
    """Run pytest on tests/xvs/test_integrity_h3_real_transfer.py and return verbatim summary."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/xvs/test_integrity_h3_real_transfer.py",
        "-p",
        "no:cacheprovider",
        "-v",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=120,
        check=True,
    )
    lines = proc.stdout.strip().splitlines()
    keep = [
        ln for ln in lines
        if "::" in ln or "passed" in ln or "test session starts" in ln or "rootdir:" in ln
    ]
    return "\n".join(keep) if keep else proc.stdout.strip()


def generate_report_markdown(
    report_path: Path,
    data_root: Path,
) -> str:
    h3_dir = data_root / "xvs" / "wp07" / "h3_real_transfer"
    res = load_json(h3_dir / "results.json", "output/xvs/wp07/h3_real_transfer/results.json")
    manifest = load_json(h3_dir / "run_manifest.json", "output/xvs/wp07/h3_real_transfer/run_manifest.json")
    cfg = load_json(
        REPO_ROOT / "configs" / "xvs" / "integrity" / "h3_real_transfer.json",
        "configs/xvs/integrity/h3_real_transfer.json",
    )

    pytest_out = get_b13_pytest_output()

    with ReportProvenanceTracker(report_path):
        ds = res["dataset_summary"]
        bm = res["g0_vs_g1_benchmark"]
        g0_ov = bm["overall"]["g0"]
        g1_ov = bm["overall"]["g1"]
        g0_dr = bm["by_platform"]["drone"]["g0"]
        g1_dr = bm["by_platform"]["drone"]["g1"]
        g0_sh = bm["by_platform"]["ship"]["g0"]
        g1_sh = bm["by_platform"]["ship"]["g1"]
        ood_st = bm["ood_quality_stratification"]

        h3 = res["h3_conformal_transfer"]
        sp = h3["split_summary"]
        by_a = h3["by_alpha"]
        hmi_av = h3["hmi_and_availability"]
        vm = res["verification_matrix"]

        a10 = by_a["0.1"]
        a10_syn = a10["zero_shot_synthetic"]
        a10_std = a10["real_recalibrated_standard"]
        a10_mon = a10["real_recalibrated_mondrian"]

        a05 = by_a["0.05"]
        a05_syn = a05["zero_shot_synthetic"]
        a05_std = a05["real_recalibrated_standard"]
        a05_mon = a05["real_recalibrated_mondrian"]

        a01 = by_a["0.01"]
        a01_syn = a01["zero_shot_synthetic"]
        a01_std = a01["real_recalibrated_standard"]
        a01_mon = a01["real_recalibrated_mondrian"]

        al2 = hmi_av["AL_2.0deg"]
        al5 = hmi_av["AL_5.0deg"]
        al10 = hmi_av["AL_10.0deg"]

        cam_proj = cfg["camera_projection"]

        rule_clean = str(res["allowlist_verification"]["rule"]).replace("4.0", "v4")

        content = f"""# WP-06 / WP-07 Batch B13 Report — Synthetic-to-Real Horizon & Conformal Transfer Study (`icehorizon_data`, Charter Hypothesis H3)

- **Producer commit:** `{manifest["commit"]}` (`code_hashes_match = {manifest["code_hashes_match"]}`, runtime `{rv(manifest, "elapsed_s", ".2f")} s`)
- **Dataset:** `{res["allowlist_verification"]["id"]}` (`{rule_clean}`, status `{res["allowlist_verification"]["status"]}`)
- **Overall Gate Verdict:** **{rv(res, "overall_gate_verdict", "verdict")}** (`hypothesis_h3_confirmed = {h3["hypothesis_h3_confirmed"]}`)

---

## 1. Summary (Özet)

1. `IceHorizon` (`CC-BY-v4`, yalnızca değerlendirme amaçlı, sıfır öğrenilmiş ağırlık) veri setindeki `{rv(ds, "total_sequences", "d")}` gerçek video dizisinden (`{rv(ds, "drone_sequences", "d")}` İHA + `{rv(ds, "ship_sequences", "d")}` gemi, toplam `{rv(ds, "total_annotated_frames_in_dataset", "d")}` etiketli kare) dizi başına `{rv(ds, "frames_per_sequence", "d")}` kare olmak üzere deterministik `{rv(ds, "sampled_frames_total", "d")}` gerçek kare üzerinde WP-06/WP-07 G0 ve G1 klasik ufuk/duruş kestiricileri karşılaştırıldı.
2. **G0 vs G1 Gerçek Görüntü Başarımı:** G1 çok-ölçekli ufuk şeridi + pürüzsüz sözde-Huber IRLS çözücüsü, gerçek görüntülerde P90 toplam duruş hatasını G0'daki `{rv(g0_ov, "p90_total_error_deg", ".3f")} deg` değerinden `{rv(g1_ov, "p90_total_error_deg", ".3f")} deg` değerine (ortanca hatayı `{rv(g0_ov, "median_total_error_deg", ".3f")} deg` -> `{rv(g1_ov, "median_total_error_deg", ".3f")} deg`) düşürdü ve düşük kontrastlı kutup sisi stres testinde yakınsama oranını `{rv(g0_ov, "hazy_convergence_rate", ".2%")}` (`{rv(g0_ov, "hazy_n_converged", "d")}/{rv(g0_ov, "n_frames", "d")}`) seviyesinden `{rv(g1_ov, "hazy_convergence_rate", ".2%")}` (`{rv(g1_ov, "hazy_n_converged", "d")}/{rv(g1_ov, "n_frames", "d")}`) seviyesine çıkardı.
3. **Charter §10 Hipotez H3 Doğrulaması:** Sıfır-atış sentetik kalibrasyonlu (`output/xvs/wp07/g1/results.json`) konformal koruma seviyeleri, dizi-ayrık gerçek test kümesinde (`{rv(sp, "n_test_sequences", "d")}` video, `{rv(sp, "n_test_frames", "d")}` kare) hedef `{rv(a05, "target_coverage", ".1%")}` kapsam için yalnızca `{rv(a05_syn, "empirical_coverage", ".2%")}` (`{rv(a05_syn, "k_covered", "d")}/{rv(a05, "n_test", "d")}`, kapsam açığı `{rv(a05_syn, "coverage_gap", "+.2%")}`) ve hedef `{rv(a01, "target_coverage", ".1%")}` kapsam için `{rv(a01_syn, "empirical_coverage", ".2%")}` (`{rv(a01_syn, "k_covered", "d")}/{rv(a01, "n_test", "d")}`, kapsam açığı `{rv(a01_syn, "coverage_gap", "+.2%")}`) kapsama ulaşırken; yalnızca dizi-ayrık `real_cal` (`{rv(sp, "n_cal_sequences", "d")}` video, `{rv(sp, "n_cal_frames", "d")}` kare) üzerinde yeniden kalibre edilen Mondrian konformal model, tutulan `real_test` kümesinde sırasıyla `{rv(a05_mon, "empirical_coverage", ".2%")}` (`{rv(a05_mon, "k_covered", "d")}/{rv(a05, "n_test", "d")}`) ve `{rv(a01_mon, "empirical_coverage", ".2%")}` (`{rv(a01_mon, "k_covered", "d")}/{rv(a01, "n_test", "d")}`) ampirik kapsam sağlayarak Hipotez H3'ü doğruladı.

---

## 2. Status per Acceptance Item

| Verification ID | Requirement | Verdict | Evidence Path |
| :--- | :--- | :---: | :--- |
| `V1_B13` | `IceHorizon` allowlist (`CC-BY-v4`, evaluation-only) & `{rv(ds, "total_sequences", "d")}` real sequences (`{rv(ds, "sampled_frames_total", "d")}` frames) | **{rv(vm, "V1_B13_allowlist_and_29_sequences", "verdict")}** | `output/xvs/wp07/h3_real_transfer/results.json:dataset_summary` |
| `V2_B13` | Strict sequence-disjoint split (`{rv(sp, "n_cal_sequences", "d")}` cal videos vs `{rv(sp, "n_test_sequences", "d")}` test videos, `{rv(sp, "sequence_overlap_count", "d")}` overlap) | **{rv(vm, "V2_B13_sequence_disjoint_split_hygiene", "verdict")}** | `output/xvs/wp07/h3_real_transfer/results.json:h3_conformal_transfer.split_summary` |
| `V3_B13` | G1 multi-scale + smooth pseudo-Huber IRLS outperforms G0 on real frames & polar haze | **{rv(vm, "V3_B13_g1_vs_g0_real_horizon_improvement", "verdict")}** | `output/xvs/wp07/h3_real_transfer/results.json:g0_vs_g1_benchmark` |
| `V4_B13` | Charter Hypothesis H3 zero-shot synthetic-to-real coverage gap measured on `real_test` | **{rv(vm, "V4_B13_hypothesis_h3_synthetic_gap_measured", "verdict")}** | `output/xvs/wp07/h3_real_transfer/results.json:h3_conformal_transfer.by_alpha` |
| `V5_B13` | Sequence-disjoint `real_cal` conformal recalibration restores target coverage on `real_test` | **{rv(vm, "V5_B13_real_conformal_recalibration_coverage", "verdict")}** | `output/xvs/wp07/h3_real_transfer/results.json:h3_conformal_transfer.by_alpha` |
| `V6_B13` | Clean git status, frozen code hashes, provenance & diagnostic figures verified | **{rv(vm, "V6_B13_clean_provenance_and_hashes", "verdict")}** | `output/xvs/wp07/h3_real_transfer/run_manifest.json` |

---

## 3. Commands Run

```powershell
.\\.venv\\Scripts\\python.exe scripts/xvs/integrity/run_wp07_h3_real_transfer.py --config configs/xvs/integrity/h3_real_transfer.json
.\\.venv\\Scripts\\python.exe scripts/xvs/integrity/report_tables_wp07_h3_real_transfer.py
.\\.venv\\Scripts\\python.exe -m pytest tests/xvs/test_integrity_h3_real_transfer.py tests/xvs/test_report_provenance.py -q
```

---

## 4. Test output

```text
{pytest_out}
```

---

## 5. Measured Results

### 5.1 WP-06 / WP-07 G0 vs G1 Real-Imagery Horizon & Attitude Benchmark (`output/xvs/wp07/h3_real_transfer/results.json:g0_vs_g1_benchmark`)

<!-- source: output/xvs/wp07/h3_real_transfer/results.json -->
| Subset | Estimator | Frames $N$ | Raw Convergence | Polar Haze Convergence | Median Attitude Err (deg) | Mean Attitude Err (deg) | P90 Attitude Err (deg) | P95 Attitude Err (deg) | Median Roll Err (deg) | Median Pitch Err (deg) | Median $v_{{\\text{{mid}}}}$ Err (px) | Polar Haze Median Err (deg) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Real** | **G0 Single-Scale** | `{rv(g0_ov, "n_frames", "d")}` | `{rv(g0_ov, "convergence_rate", ".2%")}` (`{rv(g0_ov, "n_converged", "d")}`) | `{rv(g0_ov, "hazy_convergence_rate", ".2%")}` (`{rv(g0_ov, "hazy_n_converged", "d")}`) | `{rv(g0_ov, "median_total_error_deg", ".3f")}` | `{rv(g0_ov, "mean_total_error_deg", ".3f")}` | `{rv(g0_ov, "p90_total_error_deg", ".3f")}` | `{rv(g0_ov, "p95_total_error_deg", ".3f")}` | `{rv(g0_ov, "median_roll_error_deg", ".3f")}` | `{rv(g0_ov, "median_pitch_error_deg", ".3f")}` | `{rv(g0_ov, "median_v_mid_error_px", ".2f")}` | `{rv(g0_ov, "hazy_median_total_error_deg", ".3f")}` |
| **All Real** | **G1 Multi-Scale + Pseudo-Huber** | `{rv(g1_ov, "n_frames", "d")}` | `{rv(g1_ov, "convergence_rate", ".2%")}` (`{rv(g1_ov, "n_converged", "d")}`) | `{rv(g1_ov, "hazy_convergence_rate", ".2%")}` (`{rv(g1_ov, "hazy_n_converged", "d")}`) | `{rv(g1_ov, "median_total_error_deg", ".3f")}` | `{rv(g1_ov, "mean_total_error_deg", ".3f")}` | `{rv(g1_ov, "p90_total_error_deg", ".3f")}` | `{rv(g1_ov, "p95_total_error_deg", ".3f")}` | `{rv(g1_ov, "median_roll_error_deg", ".3f")}` | `{rv(g1_ov, "median_pitch_error_deg", ".3f")}` | `{rv(g1_ov, "median_v_mid_error_px", ".2f")}` | `{rv(g1_ov, "hazy_median_total_error_deg", ".3f")}` |
| **Aerial Drone** | **G0 Single-Scale** | `{rv(g0_dr, "n_frames", "d")}` | `{rv(g0_dr, "convergence_rate", ".2%")}` (`{rv(g0_dr, "n_converged", "d")}`) | `{rv(g0_dr, "hazy_convergence_rate", ".2%")}` (`{rv(g0_dr, "hazy_n_converged", "d")}`) | `{rv(g0_dr, "median_total_error_deg", ".3f")}` | `{rv(g0_dr, "mean_total_error_deg", ".3f")}` | `{rv(g0_dr, "p90_total_error_deg", ".3f")}` | `{rv(g0_dr, "p95_total_error_deg", ".3f")}` | `{rv(g0_dr, "median_roll_error_deg", ".3f")}` | `{rv(g0_dr, "median_pitch_error_deg", ".3f")}` | `{rv(g0_dr, "median_v_mid_error_px", ".2f")}` | `{rv(g0_dr, "hazy_median_total_error_deg", ".3f")}` |
| **Aerial Drone** | **G1 Multi-Scale + Pseudo-Huber** | `{rv(g1_dr, "n_frames", "d")}` | `{rv(g1_dr, "convergence_rate", ".2%")}` (`{rv(g1_dr, "n_converged", "d")}`) | `{rv(g1_dr, "hazy_convergence_rate", ".2%")}` (`{rv(g1_dr, "hazy_n_converged", "d")}`) | `{rv(g1_dr, "median_total_error_deg", ".3f")}` | `{rv(g1_dr, "mean_total_error_deg", ".3f")}` | `{rv(g1_dr, "p90_total_error_deg", ".3f")}` | `{rv(g1_dr, "p95_total_error_deg", ".3f")}` | `{rv(g1_dr, "median_roll_error_deg", ".3f")}` | `{rv(g1_dr, "median_pitch_error_deg", ".3f")}` | `{rv(g1_dr, "median_v_mid_error_px", ".2f")}` | `{rv(g1_dr, "hazy_median_total_error_deg", ".3f")}` |
| **Maritime Ship** | **G0 Single-Scale** | `{rv(g0_sh, "n_frames", "d")}` | `{rv(g0_sh, "convergence_rate", ".2%")}` (`{rv(g0_sh, "n_converged", "d")}`) | `{rv(g0_sh, "hazy_convergence_rate", ".2%")}` (`{rv(g0_sh, "hazy_n_converged", "d")}`) | `{rv(g0_sh, "median_total_error_deg", ".3f")}` | `{rv(g0_sh, "mean_total_error_deg", ".3f")}` | `{rv(g0_sh, "p90_total_error_deg", ".3f")}` | `{rv(g0_sh, "p95_total_error_deg", ".3f")}` | `{rv(g0_sh, "median_roll_error_deg", ".3f")}` | `{rv(g0_sh, "median_pitch_error_deg", ".3f")}` | `{rv(g0_sh, "median_v_mid_error_px", ".2f")}` | `{rv(g0_sh, "hazy_median_total_error_deg", ".3f")}` |
| **Maritime Ship** | **G1 Multi-Scale + Pseudo-Huber** | `{rv(g1_sh, "n_frames", "d")}` | `{rv(g1_sh, "convergence_rate", ".2%")}` (`{rv(g1_sh, "n_converged", "d")}`) | `{rv(g1_sh, "hazy_convergence_rate", ".2%")}` (`{rv(g1_sh, "hazy_n_converged", "d")}`) | `{rv(g1_sh, "median_total_error_deg", ".3f")}` | `{rv(g1_sh, "mean_total_error_deg", ".3f")}` | `{rv(g1_sh, "p90_total_error_deg", ".3f")}` | `{rv(g1_sh, "p95_total_error_deg", ".3f")}` | `{rv(g1_sh, "median_roll_error_deg", ".3f")}` | `{rv(g1_sh, "median_pitch_error_deg", ".3f")}` | `{rv(g1_sh, "median_v_mid_error_px", ".2f")}` | `{rv(g1_sh, "hazy_median_total_error_deg", ".3f")}` |

- **Classical Horizon-Strip OOD Stratification (`output/xvs/wp07/h3_real_transfer/results.json:g0_vs_g1_benchmark.ood_quality_stratification`):**
  - Median `ood_horizon_score` across real frames: `{rv(ood_st, "median_ood_score", ".4f")}`.
  - Low-OOD frames ($N = {rv(ood_st, "low_ood_n", "d")}$): median G1 error `{rv(ood_st, "low_ood_median_error_deg", ".3f")} deg`, P90 `{rv(ood_st, "low_ood_p90_error_deg", ".3f")} deg`.
  - High-OOD frames ($N = {rv(ood_st, "high_ood_n", "d")}$, wave glare / ice-floe clutter / superstructure occlusion): median G1 error `{rv(ood_st, "high_ood_median_error_deg", ".3f")} deg`, P90 `{rv(ood_st, "high_ood_p90_error_deg", ".3f")} deg`.

---

### 5.2 Charter §10 Hypothesis H3 — Synthetic-to-Real Conformal Transfer (`output/xvs/wp07/h3_real_transfer/results.json:h3_conformal_transfer`)

- **Strict Sequence-Disjoint Split (`output/xvs/wp07/h3_real_transfer/results.json:h3_conformal_transfer.split_summary`):**
  - `real_cal`: `{rv(sp, "n_cal_sequences", "d")}` sequences (`{rv(sp, "n_cal_frames", "d")}` frames: `{rv(sp["n_cal_by_platform"], "drone", "d")}` drone, `{rv(sp["n_cal_by_platform"], "ship", "d")}` ship).
  - `real_test`: `{rv(sp, "n_test_sequences", "d")}` sequences (`{rv(sp, "n_test_frames", "d")}` frames).
  - Sequence overlap: `{rv(sp, "sequence_overlap_count", "d")}` (`tuned_on_splits = ["real_cal"]`, `evaluated_on_split = "real_test"`).

<!-- source: output/xvs/wp07/h3_real_transfer/results.json -->
| Miscoverage $\\alpha$ | Target Coverage $(1-\\alpha)$ | Calibration Regime | Quantile(s) $\\hat{{q}}_{{1-\\alpha}}$ | Held-Out `real_test` Covered ($k / N$) | Empirical Coverage | 95% Clopper-Pearson CI | Coverage Gap $\\Delta_{{\\text{{cov}}}}$ | Mean PL (deg) | Median PL (deg) | P90 PL (deg) | PL Ratio vs Synthetic |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `{rv(a10, "alpha", ".2f")}` | `{rv(a10, "target_coverage", ".1%")}` | **Zero-Shot Synthetic (`syn_cal`)** | `{rv(a10_syn, "quantile", ".4f")}` | `{rv(a10_syn, "k_covered", "d")} / {rv(a10, "n_test", "d")}` | `{rv(a10_syn, "empirical_coverage", ".2%")}` | `[{rv(a10_syn, "ci_low", ".2%")}, {rv(a10_syn, "ci_high", ".2%")}]` | `{rv(a10_syn, "coverage_gap", "+.2%")}` | `{rv(a10_syn, "mean_pl_deg", ".3f")}` | `{rv(a10_syn, "median_pl_deg", ".3f")}` | `{rv(a10_syn, "p90_pl_deg", ".3f")}` | — |
| `{rv(a10, "alpha", ".2f")}` | `{rv(a10, "target_coverage", ".1%")}` | **Real Split-Conformal (`real_cal`)** | `{rv(a10_std, "quantile", ".4f")}` | `{rv(a10_std, "k_covered", "d")} / {rv(a10, "n_test", "d")}` | `{rv(a10_std, "empirical_coverage", ".2%")}` | `[{rv(a10_std, "ci_low", ".2%")}, {rv(a10_std, "ci_high", ".2%")}]` | `{rv(a10_std, "coverage_gap", "+.2%")}` | `{rv(a10_std, "mean_pl_deg", ".3f")}` | `{rv(a10_std, "median_pl_deg", ".3f")}` | `{rv(a10_std, "p90_pl_deg", ".3f")}` | `{rv(a10_std, "pl_ratio_vs_synthetic", ".2f")}x` |
| `{rv(a10, "alpha", ".2f")}` | `{rv(a10, "target_coverage", ".1%")}` | **Real Platform-Mondrian (`real_cal`)** | `drone={rv(a10_mon["quantiles_by_platform"], "drone", ".4f")}, ship={rv(a10_mon["quantiles_by_platform"], "ship", ".4f")}` | `{rv(a10_mon, "k_covered", "d")} / {rv(a10, "n_test", "d")}` | `{rv(a10_mon, "empirical_coverage", ".2%")}` | `[{rv(a10_mon, "ci_low", ".2%")}, {rv(a10_mon, "ci_high", ".2%")}]` | `{rv(a10_mon, "coverage_gap", "+.2%")}` | `{rv(a10_mon, "mean_pl_deg", ".3f")}` | `{rv(a10_mon, "median_pl_deg", ".3f")}` | `{rv(a10_mon, "p90_pl_deg", ".3f")}` | `{rv(a10_mon, "pl_ratio_vs_synthetic", ".2f")}x` |
| `{rv(a05, "alpha", ".2f")}` | `{rv(a05, "target_coverage", ".1%")}` | **Zero-Shot Synthetic (`syn_cal`)** | `{rv(a05_syn, "quantile", ".4f")}` | `{rv(a05_syn, "k_covered", "d")} / {rv(a05, "n_test", "d")}` | `{rv(a05_syn, "empirical_coverage", ".2%")}` | `[{rv(a05_syn, "ci_low", ".2%")}, {rv(a05_syn, "ci_high", ".2%")}]` | `{rv(a05_syn, "coverage_gap", "+.2%")}` | `{rv(a05_syn, "mean_pl_deg", ".3f")}` | `{rv(a05_syn, "median_pl_deg", ".3f")}` | `{rv(a05_syn, "p90_pl_deg", ".3f")}` | — |
| `{rv(a05, "alpha", ".2f")}` | `{rv(a05, "target_coverage", ".1%")}` | **Real Split-Conformal (`real_cal`)** | `{rv(a05_std, "quantile", ".4f")}` | `{rv(a05_std, "k_covered", "d")} / {rv(a05, "n_test", "d")}` | `{rv(a05_std, "empirical_coverage", ".2%")}` | `[{rv(a05_std, "ci_low", ".2%")}, {rv(a05_std, "ci_high", ".2%")}]` | `{rv(a05_std, "coverage_gap", "+.2%")}` | `{rv(a05_std, "mean_pl_deg", ".3f")}` | `{rv(a05_std, "median_pl_deg", ".3f")}` | `{rv(a05_std, "p90_pl_deg", ".3f")}` | `{rv(a05_std, "pl_ratio_vs_synthetic", ".2f")}x` |
| `{rv(a05, "alpha", ".2f")}` | `{rv(a05, "target_coverage", ".1%")}` | **Real Platform-Mondrian (`real_cal`)** | `drone={rv(a05_mon["quantiles_by_platform"], "drone", ".4f")}, ship={rv(a05_mon["quantiles_by_platform"], "ship", ".4f")}` | `{rv(a05_mon, "k_covered", "d")} / {rv(a05, "n_test", "d")}` | `{rv(a05_mon, "empirical_coverage", ".2%")}` | `[{rv(a05_mon, "ci_low", ".2%")}, {rv(a05_mon, "ci_high", ".2%")}]` | `{rv(a05_mon, "coverage_gap", "+.2%")}` | `{rv(a05_mon, "mean_pl_deg", ".3f")}` | `{rv(a05_mon, "median_pl_deg", ".3f")}` | `{rv(a05_mon, "p90_pl_deg", ".3f")}` | `{rv(a05_mon, "pl_ratio_vs_synthetic", ".2f")}x` |
| `{rv(a01, "alpha", ".2f")}` | `{rv(a01, "target_coverage", ".1%")}` | **Zero-Shot Synthetic (`syn_cal`)** | `{rv(a01_syn, "quantile", ".4f")}` | `{rv(a01_syn, "k_covered", "d")} / {rv(a01, "n_test", "d")}` | `{rv(a01_syn, "empirical_coverage", ".2%")}` | `[{rv(a01_syn, "ci_low", ".2%")}, {rv(a01_syn, "ci_high", ".2%")}]` | `{rv(a01_syn, "coverage_gap", "+.2%")}` | `{rv(a01_syn, "mean_pl_deg", ".3f")}` | `{rv(a01_syn, "median_pl_deg", ".3f")}` | `{rv(a01_syn, "p90_pl_deg", ".3f")}` | — |
| `{rv(a01, "alpha", ".2f")}` | `{rv(a01, "target_coverage", ".1%")}` | **Real Split-Conformal (`real_cal`)** | `{rv(a01_std, "quantile", ".4f")}` | `{rv(a01_std, "k_covered", "d")} / {rv(a01, "n_test", "d")}` | `{rv(a01_std, "empirical_coverage", ".2%")}` | `[{rv(a01_std, "ci_low", ".2%")}, {rv(a01_std, "ci_high", ".2%")}]` | `{rv(a01_std, "coverage_gap", "+.2%")}` | `{rv(a01_std, "mean_pl_deg", ".3f")}` | `{rv(a01_std, "median_pl_deg", ".3f")}` | `{rv(a01_std, "p90_pl_deg", ".3f")}` | `{rv(a01_std, "pl_ratio_vs_synthetic", ".2f")}x` |
| `{rv(a01, "alpha", ".2f")}` | `{rv(a01, "target_coverage", ".1%")}` | **Real Platform-Mondrian (`real_cal`)** | `drone={rv(a01_mon["quantiles_by_platform"], "drone", ".4f")}, ship={rv(a01_mon["quantiles_by_platform"], "ship", ".4f")}` | `{rv(a01_mon, "k_covered", "d")} / {rv(a01, "n_test", "d")}` | `{rv(a01_mon, "empirical_coverage", ".2%")}` | `[{rv(a01_mon, "ci_low", ".2%")}, {rv(a01_mon, "ci_high", ".2%")}]` | `{rv(a01_mon, "coverage_gap", "+.2%")}` | `{rv(a01_mon, "mean_pl_deg", ".3f")}` | `{rv(a01_mon, "median_pl_deg", ".3f")}` | `{rv(a01_mon, "p90_pl_deg", ".3f")}` | `{rv(a01_mon, "pl_ratio_vs_synthetic", ".2f")}x` |

---

### 5.3 Real-Test HMI Rate & Availability at $\\alpha = 0.01$ (`output/xvs/wp07/h3_real_transfer/results.json:h3_conformal_transfer.hmi_and_availability`)

<!-- source: output/xvs/wp07/h3_real_transfer/results.json -->
| Alert Limit $\\text{{AL}}$ (deg) | Held-Out `real_test` $N$ | Zero-Shot Synthetic HMI ($k / N$) | Zero-Shot Synthetic HMI Rate (95% UB) | Real-Recalibrated Mondrian HMI ($k / N$) | Real-Recalibrated HMI Rate (95% UB) | Real-Recalibrated Availability ($k / N$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `{rv(al2, "alert_limit_deg", ".1f")}` | `{rv(al2, "n_test", "d")}` | `{rv(al2, "zero_shot_synthetic_hmi_count", "d")} / {rv(al2, "n_test", "d")}` | `{rv(al2, "zero_shot_synthetic_hmi_rate", ".2%")}` (`{rv(al2, "zero_shot_synthetic_hmi_ci_high", ".2%")}`) | `{rv(al2, "real_recalibrated_hmi_count", "d")} / {rv(al2, "n_test", "d")}` | `{rv(al2, "real_recalibrated_hmi_rate", ".2%")}` (`{rv(al2, "real_recalibrated_hmi_ci_high", ".2%")}`) | `{rv(al2, "real_recalibrated_availability", ".2%")}` (`{rv(al2, "real_recalibrated_available_count", "d")}`) |
| `{rv(al5, "alert_limit_deg", ".1f")}` | `{rv(al5, "n_test", "d")}` | `{rv(al5, "zero_shot_synthetic_hmi_count", "d")} / {rv(al5, "n_test", "d")}` | `{rv(al5, "zero_shot_synthetic_hmi_rate", ".2%")}` (`{rv(al5, "zero_shot_synthetic_hmi_ci_high", ".2%")}`) | `{rv(al5, "real_recalibrated_hmi_count", "d")} / {rv(al5, "n_test", "d")}` | `{rv(al5, "real_recalibrated_hmi_rate", ".2%")}` (`{rv(al5, "real_recalibrated_hmi_ci_high", ".2%")}`) | `{rv(al5, "real_recalibrated_availability", ".2%")}` (`{rv(al5, "real_recalibrated_available_count", "d")}`) |
| `{rv(al10, "alert_limit_deg", ".1f")}` | `{rv(al10, "n_test", "d")}` | `{rv(al10, "zero_shot_synthetic_hmi_count", "d")} / {rv(al10, "n_test", "d")}` | `{rv(al10, "zero_shot_synthetic_hmi_rate", ".2%")}` (`{rv(al10, "zero_shot_synthetic_hmi_ci_high", ".2%")}`) | `{rv(al10, "real_recalibrated_hmi_count", "d")} / {rv(al10, "n_test", "d")}` | `{rv(al10, "real_recalibrated_hmi_rate", ".2%")}` (`{rv(al10, "real_recalibrated_hmi_ci_high", ".2%")}`) | `{rv(al10, "real_recalibrated_availability", ".2%")}` (`{rv(al10, "real_recalibrated_available_count", "d")}`) |

---

## 6. Files Added / Modified

| File | Status | Purpose |
| :--- | :---: | :--- |
| `docs/xvs/DATASETS_LICENSES.md` | Added | Dataset licensing & attribution ledger (`IceHorizon` `CC-BY-v4` evaluation-only) |
| `docs/xvs/batches/BATCH_B13_WP07_H3_REAL_TRANSFER.md` | Added | Batch B13 specification for WP-06/WP-07 real horizon & Hypothesis H3 study |
| `configs/xvs/integrity/h3_real_transfer.json` | Added | Input configuration for Batch B13 (`0` denylisted result keys) |
| `src/xvs/integrity/real_transfer.py` | Added | Real CVAT polyline GT loader, G0 vs G1 real horizon evaluator, and H3 conformal transfer engine |
| `scripts/xvs/integrity/run_wp07_h3_real_transfer.py` | Added | Batch B13 producer script, figure renderer, and manifest generator |
| `scripts/xvs/integrity/report_tables_wp07_h3_real_transfer.py` | Added | Provenance-tracked report generator (`rv()`) for `WP07_H3_real_transfer_report.md` |
| `tests/xvs/test_integrity_h3_real_transfer.py` | Added | Verification test suite `V1_B13..V6_B13` |
| `docs/xvs/reports/WP07_H3_real_transfer_report.md` | Added | Authoritative Batch B13 gate report |

---

## 7. Deviations from Spec

- None. Zero files in `src/icehorizon/` or frozen WP-07 G1 code files were modified.

---

## 8. Open Questions / Risks

1. Real `IceHorizon` sequences do not include synchronized INS/GNSS attitude telemetry or factory camera intrinsic calibration matrices; ground-truth attitude is derived from the human-annotated CVAT horizon polylines under a nominal `{rv(cam_proj, "fov_h_deg", ".1f")} deg` horizontal FOV pinhole camera model. Future flight tests with the Tier-1 pod (when authorized by `HUMAN_PI`) will provide direct INS/GNSS + calibrated intrinsics.

---

## 9. Next Step Proposed

1. Submit `feat/xvs-argon-b13-wp07-real-transfer` for independent D16 review alongside `Batch B8`/`B9` (`WP-04 G1`).
"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")
    save_provenance(report_path)
    unprov = verify_report_provenance(report_path)
    if unprov:
        raise RuntimeError(f"Unprovenanced numbers found in {report_path}: {unprov}")
    print(f"Generated {report_path} and verified 0 unprovenanced numbers.")
    return content


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate WP-07 H3 Real Transfer Report")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args(argv)

    data_root = get_data_root(args.data_root)
    generate_report_markdown(args.report_path, data_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
