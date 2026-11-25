"""Gate G0 Acceptance V5: Determinism test for OpenXVS detection and injection.

Conforms strictly to docs/xvs/WP04_DETECTION.md v0.1 §7 (V5):
"The same seeds give byte-identical JSON."
"""

from __future__ import annotations

import hashlib
import json
import numpy as np
import pytest

from xvs.detection.inject import (
    load_traffic_models,
    sample_traffic_silhouette,
    composite_signal,
    InjectionSensorSimulator,
)
from xvs.detection.d0 import detect_d0, load_d0_config


def run_mini_injection_sequence(seed: int) -> str:
    """Run a deterministic multi-step injection simulation and return canonical JSON string."""
    rng = np.random.default_rng(seed)
    models = load_traffic_models()
    d0_cfg = load_d0_config()
    sim = InjectionSensorSimulator()

    results = []

    # Run 5 deterministic trials
    for trial_idx in range(5):
        s_px = float(rng.choice([1.0, 2.0, 4.0, 6.0]))
        contrast_c = float(rng.choice([-0.2, -0.05, 0.05, 0.2]))
        u_t = float(rng.uniform(30.0, 220.0))
        v_t = float(rng.uniform(30.0, 220.0))

        # Synthetic background
        base_lum = float(rng.uniform(0.1, 0.4))
        bg = np.full((256, 256), base_lum, dtype=np.float64)

        sil = sample_traffic_silhouette(models, s_px=s_px, u_t=u_t, v_t=v_t, rng=rng, crop_size=256)
        L_prime, a_c = composite_signal(bg, sil["alpha"], contrast_c)

        # Clean D0
        dets_clean = detect_d0(L_prime, k=5, tau=4.0, is_clean=True, config=d0_cfg)

        # Degraded D0
        seed_temporal = int(rng.integers(0, 2**31 - 1))
        L_deg = sim.apply_sensor(L_prime, is_degraded=True, seed=seed_temporal)
        dets_deg = detect_d0(L_deg, k=5, tau=4.0, is_clean=False, config=d0_cfg)

        results.append({
            "trial": trial_idx,
            "s_px": s_px,
            "contrast_c": contrast_c,
            "u_t": u_t,
            "v_t": v_t,
            "a_c": a_c,
            "box_gt": [round(coord, 6) for coord in sil["box_gt"]],
            "n_dets_clean": len(dets_clean),
            "n_dets_deg": len(dets_deg),
            "clean_scores": [round(d["score"], 6) for d in dets_clean],
            "deg_scores": [round(d["score"], 6) for d in dets_deg],
        })

    json_str = json.dumps(results, indent=2, sort_keys=True)
    return json_str


def test_v5_determinism_byte_identical():
    """Verify that identical seeds produce byte-identical JSON outputs."""
    seed = 424242

    run1_json = run_mini_injection_sequence(seed)
    run2_json = run_mini_injection_sequence(seed)

    sha1 = hashlib.sha256(run1_json.encode("utf-8")).hexdigest()
    sha2 = hashlib.sha256(run2_json.encode("utf-8")).hexdigest()

    print(f"\n[V5 Determinism] Run 1 SHA256: {sha1}")
    print(f"[V5 Determinism] Run 2 SHA256: {sha2}")

    assert run1_json == run2_json, "JSON outputs with identical seeds must be identical"
    assert sha1 == sha2, "JSON SHA-256 hashes must match byte-for-byte"
