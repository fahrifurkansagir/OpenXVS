"""OpenXVS Integrity Layer (WP-07 G0 & G1).

Provides fault injection, runtime monitoring (M0 / M1), leave-one-camera-out consistency,
fault isolation and exclusion (F_isolated), boresight perturbation, horizon-strip OOD
monitoring, and conformal prediction (split, Mondrian, and altitude-binned Mondrian)
for horizon attitude estimation.
"""

from __future__ import annotations

# Import G0 modules first before G1 extensions to avoid circular imports
from xvs.integrity import conformal as _conformal  # noqa: F401
from xvs.integrity import faults as _faults  # noqa: F401
from xvs.integrity import monitor as _monitor  # noqa: F401

from xvs.integrity.g1_extensions import (
    ALTITUDE_BANDS,
    ConformalModelG1,
    MonitorThresholdsG1,
    calibrate_boresight_pl_sensitivity_g1,
    classify_altitude_band,
    compute_horizon_strip_ood_features,
    compute_horizon_strip_ood_score,
    compute_pl_deg_altitude_mondrian,
    detect_camera_edges_multiscale,
    evaluate_monitor_alarm_g1,
    fit_conformal_model_g1,
    fit_pipeline_attitude_g1,
    fit_sigma_sys_g1,
    inject_single_cam_misalignment,
    isolate_and_exclude_faulty_camera,
    perturb_rig_boresight,
    solve_attitude_gn_g1,
    tune_monitor_thresholds_g1,
)

__version__ = "0.2.0"
