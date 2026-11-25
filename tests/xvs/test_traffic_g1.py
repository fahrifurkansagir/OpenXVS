"""Unit tests for WP-02 G1: Encounters, kinematics, event flags, GT, and sampler."""

import math
from pathlib import Path
import pytest
import numpy as np

from xvs.geo import wgs84, rotations
from xvs.rig.rig import Rig
from xvs.sim import horizon, trajectory
from xvs.traffic import models, encounter, gt
from xvs.traffic.ring import TrafficRing


def test_g1_models_dimensions_and_triangles():
    manifest = models.CLASS_SPECS
    for cls_id in ["C1_UAS", "C2_SEP", "C3_TWIN", "C4_BIZJET", "C5_TRANSPORT", "C6_HELI"]:
        builder = getattr(models, f"build_{cls_id.lower()}")
        for s in (1, 2, 3):
            res = builder(seed=s)
            assert res.triangle_count <= 20000
            spec = manifest[cls_id]
            span = res.dimensions_m["span_m"]
            length = res.dimensions_m["length_m"]
            assert spec["span_min_m"] - 1e-3 <= span <= spec["span_max_m"] + 1e-3
            assert spec["length_min_m"] - 1e-3 <= length <= spec["length_max_m"] + 1e-3


def test_g1_encounter_construction_o1():
    o1 = encounter.TestOwnshipO1()
    rng = np.random.default_rng(42)

    for i in range(100):
        t_cpa = float(rng.uniform(30.0, 60.0))
        hmd = float(rng.uniform(10.0, 1500.0))
        side = "right" if rng.uniform(0.0, 1.0) < 0.5 else "left"
        vmd = float(rng.uniform(-300.0, 300.0))
        approach_deg = float(rng.uniform(-179.0, 180.0))
        gs = float(rng.uniform(20.0, 200.0))
        vs = float(rng.uniform(-8.0, 8.0))

        params = encounter.EncounterParams(
            id=f"TEST_{i:04d}",
            class_id="C2_SEP",
            model_id="C2_SEP_0001",
            t_cpa=t_cpa,
            hmd_m=hmd,
            side=side,
            vmd_m=vmd,
            approach_deg=approach_deg,
            gs_mps=gs,
            vs_mps=vs,
        )

        t_achieved, hmd_achieved, vmd_achieved = encounter.evaluate_cpa_o1(params, o1)
        assert abs(t_achieved - t_cpa) <= 1e-6
        assert abs(hmd_achieved - hmd) <= 1e-6
        assert abs(vmd_achieved - vmd) <= 1e-6


def test_g1_los_rate_vs_finite_differences():
    rng = np.random.default_rng(999)
    dt_fd = 1e-3

    for _ in range(100):
        lat = float(rng.uniform(30.0, 45.0))
        lon = float(rng.uniform(-125.0, -115.0))
        h = float(rng.uniform(1000.0, 5000.0))
        p_o = wgs84.geodetic_to_ecef(lat, lon, h)
        n_o = wgs84.r_ecef_ned(lat, lon)

        v_o = n_o @ rng.uniform(-100.0, 100.0, size=3)
        r_ned = rng.uniform(-5000.0, 5000.0, size=3)
        if np.linalg.norm(r_ned) < 500.0:
            r_ned += 500.0
        p_i = p_o + n_o @ r_ned
        v_i = n_o @ rng.uniform(-150.0, 150.0, size=3)

        r_ecef = p_i - p_o
        v_rel = v_i - v_o
        r_sq = float(np.dot(r_ecef, r_ecef))

        omega_ecef = np.cross(r_ecef, v_rel) / r_sq
        omega_ned = n_o.T @ omega_ecef
        omega_norm = float(np.linalg.norm(omega_ned))

        p_o_plus = p_o + v_o * dt_fd
        p_o_minus = p_o - v_o * dt_fd
        p_i_plus = p_i + v_i * dt_fd
        p_i_minus = p_i - v_i * dt_fd

        r_plus = p_i_plus - p_o_plus
        r_minus = p_i_minus - p_o_minus
        u_plus = r_plus / np.linalg.norm(r_plus)
        u_minus = r_minus / np.linalg.norm(r_minus)
        u_dot_fd = (u_plus - u_minus) / (2.0 * dt_fd)

        u_0 = r_ecef / math.sqrt(r_sq)
        omega_fd_ecef = np.cross(u_0, u_dot_fd)
        omega_fd_ned = n_o.T @ omega_fd_ecef

        diff = float(np.linalg.norm(omega_ned - omega_fd_ned))
        rel_err = diff / omega_norm if omega_norm > 1e-12 else diff
        assert rel_err <= 1e-6


def test_g1_event_flags_boundary():
    o1 = encounter.TestOwnshipO1()
    test_cases = [
        ("nmac_inside_dh", 152.3, 30.0, True, True),
        ("nmac_outside_dh", 152.5, 30.0, False, True),
        ("nmac_inside_dv", 150.0, 30.4, True, True),
        ("nmac_outside_dv", 150.0, 30.6, False, True),
        ("lowc_inside_dh", 609.5, 70.0, False, True),
        ("lowc_outside_dh", 609.7, 70.0, False, False),
        ("lowc_inside_dv", 600.0, 76.1, False, True),
        ("lowc_outside_dv", 600.0, 76.3, False, False),
    ]

    for name, dh, dv, exp_nmac, exp_lowc in test_cases:
        params = encounter.EncounterParams(
            id=name,
            class_id="C2_SEP",
            model_id="C2_SEP_0001",
            t_cpa=30.0,
            hmd_m=dh,
            side="right",
            vmd_m=dv,
            approach_deg=180.0,
            gs_mps=50.0,
            vs_mps=0.0,
        )
        enc = encounter.construct_encounter(o1, params)
        ev = encounter.evaluate_events(o1, enc, t_start=29.0, t_end=31.0, dt=0.01)
        assert (ev["nmac_t"] is not None) == exp_nmac
        assert (ev["lowc_t"] is not None) == exp_lowc


def test_g1_projection_gt_sphere_and_containment():
    ring = TrafficRing()
    cam = ring.camera_model

    sphere_model = models.build_sphere_r2()
    verts = np.concatenate([p.vertices for p in sphere_model.parts], axis=0)

    p_cam_rel = np.array([0.0, 0.0, 1000.0], dtype=np.float64)
    u_c, v_c, _ = cam.project_cam(p_cam_rel)

    verts_cam = verts + p_cam_rel[None, :]
    u_all = cam.fx * (verts_cam[:, 0] / verts_cam[:, 2]) + cam.cx
    v_all = cam.fy * (verts_cam[:, 1] / verts_cam[:, 2]) + cam.cy

    u_mid = 0.5 * (np.min(u_all) + np.max(u_all))
    v_mid = 0.5 * (np.min(v_all) + np.max(v_all))

    assert abs(u_mid - u_c) < 1e-9
    assert abs(v_mid - v_c) < 1e-9

    c2 = models.build_c2_sep(seed=1)
    c2_verts = np.concatenate([p.vertices for p in c2.parts], axis=0)
    rng = np.random.default_rng(555)

    for _ in range(50):
        z = float(rng.uniform(200.0, 2000.0))
        x = float(rng.uniform(-0.3 * z, 0.3 * z))
        y = float(rng.uniform(-0.3 * z, 0.3 * z))
        pos_cam = np.array([x, y, z], dtype=np.float64)

        r_body = rotations.euler_to_r(
            float(rng.uniform(-math.pi, math.pi)),
            float(rng.uniform(-0.3, 0.3)),
            float(rng.uniform(-0.5, 0.5)),
        )

        u_target, v_target, _ = cam.project_cam(pos_cam)
        v_cam = (r_body @ c2_verts.T).T + pos_cam[None, :]

        u_box = cam.fx * (v_cam[:, 0] / v_cam[:, 2]) + cam.cx
        v_box = cam.fy * (v_cam[:, 1] / v_cam[:, 2]) + cam.cy

        assert np.min(u_box) <= u_target <= np.max(u_box)
        assert np.min(v_box) <= v_target <= np.max(v_box)


def test_g1_horizon_relation_ray_consistency():
    true_rig = Rig.load_nominal().make_true(20260922)
    f_cam = true_rig.camera_model
    rng = np.random.default_rng(777)

    lat, lon, h = 37.62, -122.37, 1000.0
    cam_poses = true_rig.compute_poses(lat, lon, h, 0.0, 0.0, 0.0)
    f_pose = cam_poses["F"]
    p_cam = np.array(f_pose["position"], dtype=np.float64)
    r_cam = np.array(f_pose["r_ecef_cam"], dtype=np.float64)

    for _ in range(100):
        u = float(rng.uniform(10.0, f_cam.width - 10.0))
        roots = horizon.analytic_horizon_v_at_u(p_cam, r_cam, f_cam, u)
        if not roots:
            continue
        v_limb = roots[0]
        delta_v = float(rng.uniform(-50.0, 50.0))
        if abs(delta_v) < 0.5:
            delta_v = 1.0 if delta_v >= 0 else -1.0
        v_test = v_limb + delta_v
        is_earth = horizon.is_earth_at_pixel(p_cam, r_cam, f_cam, u, v_test)
        assert is_earth == (delta_v > 0)
