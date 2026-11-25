"""Unit tests for WP-05 G0 passive collision risk estimation (risk.py).

Conforms strictly to docs/xvs/WP05_TRACKING.md §3 and GEMINI_G-B3.md §1 item 4.
Hand-built test cases for TTC, range intervals, miss-distance proxy, and alert rules.
"""

import math
import pytest
import numpy as np

from xvs.tracking.tracker import Track, TrackStatus
from xvs.tracking.risk import (
    compute_ttc_from_expansion,
    compute_range_interval,
    compute_hmd_interval,
    evaluate_track_risk,
    WP02_CLASS_DIMENSIONS,
    UNION_L_MIN,
    UNION_L_MAX,
    LOWC_HMD_LIMIT_M,
)


def test_risk_ttc_from_expansion():
    """Unit test for TTC = s / s_dot with k=3 sigma validity guard."""
    # 1. Valid expansion: s_dot > 3 * sigma_s_dot
    s = 0.004
    s_dot = 0.0002
    sigma_s_dot = 0.00005  # 3 * sigma = 0.00015 < 0.0002
    ttc = compute_ttc_from_expansion(s, s_dot, sigma_s_dot, k_sigma=3.0)
    assert ttc is not None
    assert abs(ttc - 20.0) < 1e-9

    # 2. Invalid expansion: s_dot <= 3 * sigma_s_dot
    s_dot_small = 0.00012
    ttc_invalid = compute_ttc_from_expansion(s, s_dot_small, sigma_s_dot, k_sigma=3.0)
    assert ttc_invalid is None

    # 3. Receding or zero expansion
    assert compute_ttc_from_expansion(s, 0.0, sigma_s_dot) is None
    assert compute_ttc_from_expansion(s, -0.001, sigma_s_dot) is None


def test_risk_range_interval_and_hmd_proxy():
    """Unit test for class-specific and union range intervals and HMD proxy."""
    s = 0.002

    # Class-specific: C2_SEP [7 m, 12 m]
    r_c2 = compute_range_interval(s, class_id="C2_SEP")
    assert abs(r_c2[0] - 3500.0) < 1e-6
    assert abs(r_c2[1] - 6000.0) < 1e-6

    # Union over all classes [0.3 m, 40 m]
    r_union = compute_range_interval(s, class_id=None)
    assert abs(r_union[0] - (0.3 / 0.002)) < 1e-6
    assert abs(r_union[1] - (40.0 / 0.002)) < 1e-6

    # HMD proxy: HMD = R * |omega_LOS| * tau_s
    tau_s = 20.0
    omega_los = 0.002  # rad/s
    hmd_c2 = compute_hmd_interval(r_c2, omega_los, tau_s)
    assert hmd_c2 is not None
    assert abs(hmd_c2[0] - (3500.0 * 0.002 * 20.0)) < 1e-6  # 140 m
    assert abs(hmd_c2[1] - (6000.0 * 0.002 * 20.0)) < 1e-6  # 240 m

    # If tau_s is None, HMD proxy should be None
    assert compute_hmd_interval(r_c2, omega_los, None) is None


def test_risk_alert_rules_a1_and_a2():
    """Unit test for Alert Rules A1 (collision course) and A2 (range interval)."""
    # Create track on collision course
    # az_rate = 0.001, el_rate = 0.001 -> omega_los = sqrt(2)*0.001 ~ 0.001414 rad/s
    # s = 0.005, s_dot = 0.00025 -> tau_s = 20.0 s
    x = np.array([0.0, 0.001, 0.0, 0.001, 0.005, 0.00025], dtype=np.float64)
    P = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, (0.00005)**2])

    trk = Track(
        track_id="TRK_TEST_ALERT",
        t_init=0.0,
        status=TrackStatus.CONFIRMED,
        x=x,
        P=P,
    )

    # 1. Collision course with HMD_max < 609.6 m -> Both A1 and A2 trigger
    # In C2_SEP: range in [7/0.005, 12/0.005] = [1400, 2400] m
    # HMD in [1400 * 0.001414 * 20, 2400 * 0.001414 * 20] = [39.6, 67.9] m < 609.6 m
    risk_coll = evaluate_track_risk(trk, t_s=10.0, omega_thr_radps=0.005, class_id="C2_SEP")
    assert risk_coll.alert_a1_warning is True
    assert risk_coll.alert_a1_corrective is True
    assert risk_coll.alert_a2_warning is True
    assert risk_coll.alert_a2_corrective is True

    # 2. Non-collision due to high angular rate (|omega_LOS| > omega_thr)
    # az_rate = 0.010, el_rate = 0.010 -> omega_los = 0.01414 > 0.005
    trk.x[1] = 0.010
    trk.x[3] = 0.010
    risk_diverg = evaluate_track_risk(trk, t_s=10.0, omega_thr_radps=0.005, class_id="C2_SEP")
    assert risk_diverg.alert_a1_warning is False
    assert risk_diverg.alert_a1_corrective is False
    assert risk_diverg.alert_a2_warning is False
    assert risk_diverg.alert_a2_corrective is False

    # 3. Collision course in A1, but HMD_est,max exceeds 609.6 m -> A1 triggers, A2 does not
    # az_rate = 0.003, el_rate = 0.003 -> omega_los = 0.00424 < 0.005 (A1 passes)
    # Using Union: L_max = 40 m -> R_max = 40 / 0.005 = 8000 m
    # HMD_max = 8000 * 0.00424 * 20.0 = 678.8 m > 609.6 m (A2 fails)
    trk.x[1] = 0.003
    trk.x[3] = 0.003
    risk_a1_only = evaluate_track_risk(trk, t_s=10.0, omega_thr_radps=0.005, class_id=None)
    assert risk_a1_only.alert_a1_warning is True
    assert risk_a1_only.alert_a1_corrective is True
    assert risk_a1_only.alert_a2_warning is False
    assert risk_a1_only.alert_a2_corrective is False
