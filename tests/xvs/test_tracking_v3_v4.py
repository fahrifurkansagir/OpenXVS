"""Unit tests for WP-05 G0 acceptance items V3 (association) and V4 (seams).

Conforms strictly to docs/xvs/WP05_TRACKING.md §5 and GEMINI_G-B3.md §1 item 3.
"""

import math
import pytest
import numpy as np

from xvs.tracking.tracker import MultiCameraTracker
from xvs.tracking.measurements import DetectionRecord
from xvs.traffic.ring import TrafficRing


def test_v3_association_crossing_zero_id_switches():
    """V3, association:
    Two intruders crossing at a minimum separation of 20 * sigma_ang:
    zero ID switches over 100 seeded runs at baseline noise.
    """
    tracker = MultiCameraTracker()
    sigma_px = tracker.sigma_px
    r_ned_body = np.eye(3, dtype=np.float64)

    id_switches = 0
    n_runs = 100
    n_steps = 50

    for seed in range(n_runs):
        rng = np.random.default_rng(seed)
        tracker.reset()

        for k in range(n_steps):
            t_s = k * 0.1
            # Intruder A: az moves from -0.05 to +0.05 rad, el = +0.010 rad
            az_A = -0.05 + 0.10 * (k / (n_steps - 1.0))
            el_A = 0.010

            # Intruder B: az moves from +0.05 to -0.05 rad, el = -0.010 rad
            az_B = 0.05 - 0.10 * (k / (n_steps - 1.0))
            el_B = -0.010
            # Note: At crossing point (az=0), vertical separation is 0.020 rad = 20 * sigma_ang

            dets = []
            for az_t, el_t, tid in [(az_A, el_A, "TARGET_A"), (az_B, el_B, "TARGET_B")]:
                xc = math.tan(az_t)
                yc = -math.tan(el_t) / math.cos(az_t)
                u = tracker.ring.fx * xc + tracker.ring.cx + rng.normal(0.0, sigma_px)
                v = tracker.ring.fy * yc + tracker.ring.cy + rng.normal(0.0, sigma_px)
                dets.append(DetectionRecord(
                    t=t_s,
                    camera="T0",
                    u=float(u),
                    v=float(v),
                    s_px=10.0,
                    score=0.99,
                    is_false_alarm=False,
                    truth_id=tid,
                ))

            tracker.step(t_s, dets, r_ned_body)

        confirmed = [trk for trk in tracker.tracks if trk.status.value == "confirmed"]
        assert len(confirmed) == 2, f"Run {seed}: expected 2 confirmed tracks, got {len(confirmed)}"

        for trk in confirmed:
            # Check associations after the crossing point (last 20 frames)
            tail = trk.associated_truth_ids[-20:]
            unique_truths = set(tail)
            if len(unique_truths) > 1:
                id_switches += 1

    print(f"\nV3 Association crossing: {id_switches} ID switches across {n_runs} runs.")
    assert id_switches == 0, f"V3 failed with {id_switches} ID switches"


def test_v4_seams_360_deg_azimuth_sweep_zero_breaks():
    """V4, seams:
    An intruder sweeping 360° in azimuth at constant elevation (a synthetic trajectory)
    keeps one track ID across all 8 camera boundaries.
    Zero seam breaks in 100 seeded runs.
    """
    ring = TrafficRing()
    tracker = MultiCameraTracker(ring=ring)
    sigma_px = tracker.sigma_px
    r_ned_body = np.eye(3, dtype=np.float64)

    seam_breaks = 0
    n_runs = 100
    n_steps = 360  # 1 deg per step at 10 Hz (36 s)

    for seed in range(n_runs):
        rng = np.random.default_rng(seed)
        tracker.reset()

        for k in range(n_steps):
            t_s = k * 0.1
            az_deg = -180.0 + (360.0 * k / n_steps)
            az_rad = math.radians(az_deg)
            el_rad = 0.0
            los_ned = np.array([math.cos(az_rad), math.sin(az_rad), 0.0], dtype=np.float64)

            dets = []
            for cam in ring.cameras:
                r_cam = cam.r_body_cam.T @ los_ned
                if r_cam[2] > 1e-3:
                    u, v, in_front = ring.camera_model.project_cam(r_cam)
                    if in_front and ring.camera_model.in_image(u, v):
                        u_noisy = u + rng.normal(0.0, sigma_px)
                        v_noisy = v + rng.normal(0.0, sigma_px)
                        dets.append(DetectionRecord(
                            t=t_s,
                            camera=cam.id,
                            u=float(u_noisy),
                            v=float(v_noisy),
                            s_px=8.0,
                            score=0.99,
                            is_false_alarm=False,
                            truth_id="TARGET_360",
                        ))

            tracker.step(t_s, dets, r_ned_body)

        confirmed = [t for t in tracker.tracks if t.status.value == "confirmed"]
        assert len(confirmed) == 1, f"Expected exactly 1 confirmed track, got {len(confirmed)}"
        # Find single track that successfully navigated all 8 ring camera boundaries
        surviving = [t for t in confirmed if len(set(t.history_cams)) >= 8 and len(t.history_t) >= 300]
        if len(surviving) != 1:
            seam_breaks += 1

    print(f"\nV4 Seams: {seam_breaks} seam breaks across {n_runs} runs.")
    assert seam_breaks == 0, f"V4 failed with {seam_breaks} seam breaks"
