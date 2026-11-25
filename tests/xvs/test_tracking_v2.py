"""Unit test for WP-05 G0 acceptance item V2: Filter consistency.

Conforms strictly to docs/xvs/WP05_TRACKING.md §5 and GEMINI_G-B3.md §1 item 2.
Evaluates NEES and NIS consistency bounds on a straight-line synthetic trajectory.
"""

import math
import pytest
import numpy as np
import scipy.stats as stats

from xvs.tracking.tracker import MultiCameraTracker, MergedDetection, wrap_to_pi, TrackStatus


def test_v2_filter_consistency_nees_and_nis():
    """V2, filter consistency:
    On noise-only straight-line synthetic cases, the NEES and NIS lie inside
    their two-sided 95 % chi^2 bounds for at least 90 % of the steps.
    Also report the average NEES.
    """
    tracker = MultiCameraTracker()
    dt = tracker.dt
    F = tracker.F
    Q = tracker.Q
    H = tracker.H
    R = tracker.R

    # Ground truth initial state [az, az_rate, el, el_rate, s, s_dot]
    x_true = np.array([0.05, 0.002, -0.02, 0.001, 0.003, 0.0001], dtype=np.float64)

    # Initialise single track with true initial values plus initial uncertainty
    rng = np.random.default_rng(2026)
    init_det = MergedDetection(
        u=tracker.ring.cx,
        v=tracker.ring.cy,
        camera="T0",
        s_px=x_true[4] * tracker.ring.fx,
        score=1.0,
        los_ned=np.array([math.cos(x_true[0])*math.cos(x_true[2]), math.sin(x_true[0])*math.cos(x_true[2]), -math.sin(x_true[2])]),
        az=x_true[0],
        el=x_true[2],
        s_rad=x_true[4],
        is_false_alarm=False,
        truth_id="V2_TARGET",
    )

    trk = tracker.initiate_track(init_det, t_s=0.0)
    trk.status = TrackStatus.CONFIRMED

    nees_list = []
    nis_list = []

    n_steps = 600
    warmup_steps = 30  # Allow filter covariance to reach steady state

    for step in range(1, n_steps + 1):
        t_s = step * dt

        # 1. True state propagation with process noise
        w = rng.multivariate_normal(np.zeros(6), Q)
        x_true = F @ x_true + w
        x_true[0] = wrap_to_pi(float(x_true[0]))

        # 2. Synthetic noisy measurement
        v = rng.multivariate_normal(np.zeros(3), R)
        z = H @ x_true + v
        z[0] = wrap_to_pi(float(z[0]))

        # 3. Predict track
        tracker.tracks = [trk]
        tracker.predict_tracks(dt)

        # 4. Compute NIS before update
        hx = np.array([trk.x[0], trk.x[2], trk.x[4]], dtype=np.float64)
        y = z - hx
        y[0] = wrap_to_pi(float(y[0]))
        S = H @ trk.P @ H.T + R
        S_inv = np.linalg.inv(S)
        nis = float(y @ S_inv @ y)

        # 5. Update track
        meas_det = MergedDetection(
            u=0.0, v=0.0, camera="T0",
            s_px=z[2] * tracker.ring.fx, score=1.0,
            los_ned=np.zeros(3), az=z[0], el=z[1], s_rad=z[2],
            is_false_alarm=False, truth_id="V2_TARGET",
        )
        tracker.update_track(trk, meas_det, t_s)

        # 6. Compute NEES after update
        e_x = trk.x - x_true
        e_x[0] = wrap_to_pi(float(e_x[0]))
        P_inv = np.linalg.inv(trk.P)
        nees = float(e_x @ P_inv @ e_x)

        if step > warmup_steps:
            nees_list.append(nees)
            nis_list.append(nis)

    # Chi-square two-sided 95% bounds
    # NEES has df=6, NIS has df=3
    chi2_nees_low, chi2_nees_high = stats.chi2.ppf([0.025, 0.975], df=6)
    chi2_nis_low, chi2_nis_high = stats.chi2.ppf([0.025, 0.975], df=3)

    in_nees = np.mean([(chi2_nees_low <= n <= chi2_nees_high) for n in nees_list])
    in_nis = np.mean([(chi2_nis_low <= n <= chi2_nis_high) for n in nis_list])
    avg_nees = float(np.mean(nees_list))
    avg_nis = float(np.mean(nis_list))

    print(f"\nV2 Filter Consistency Results ({len(nees_list)} steady-state steps):")
    print(f"  NEES inside 95% bounds [{chi2_nees_low:.3f}, {chi2_nees_high:.3f}]: {in_nees * 100:.1f} % (avg NEES: {avg_nees:.2f})")
    print(f"  NIS  inside 95% bounds [{chi2_nis_low:.3f}, {chi2_nis_high:.3f}]: {in_nis * 100:.1f} % (avg NIS:  {avg_nis:.2f})")

    # Spec requirement: at least 90 % of steps inside two-sided 95 % chi^2 bounds
    assert in_nees >= 0.90, f"NEES consistency {in_nees * 100:.1f}% below 90% threshold"
    assert in_nis >= 0.90, f"NIS consistency {in_nis * 100:.1f}% below 90% threshold"
    # Average NEES should be close to state dimension (6.0)
    assert 4.5 <= avg_nees <= 7.5, f"Average NEES {avg_nees:.2f} too far from theoretical 6.0"
