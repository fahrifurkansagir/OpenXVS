"""Regression tests for the fixes made while closing WP-02 G2 (Architecture Lead, 2026-09-24).

- ROI frustum: Cesium's PerspectiveOffCenterFrustum has no sseDenominator, so the globe's
  screen-space error was NaN and tiles refined without bound (T-LIMB-ROI edge error, renders that
  hit the 600-render cap). setXVSROIFrustum now defines sseDenominator = h / fy.
- setXVSCamera re-installs a symmetric PerspectiveFrustum after a ROI render, so rig and ring
  cameras can share one page.
- export_encounters_jsonl writes LF only, so the file SHA-256 is platform independent.
- The per-camera GT uses the image convention [-0.5, W-0.5] x [-0.5, H-0.5].
"""

from __future__ import annotations

import json
import math
import socket
import threading
from pathlib import Path

import numpy as np
import pytest

from xvs.rig.camera import PinholeCamera
from xvs.rig.rig import Rig
from xvs.sim import horizon
from xvs.sim.renderer import XVSRenderer
from xvs.sim.server import XVSServer
from xvs.traffic import encounter
from xvs.traffic.ring import TrafficRing

REPO_ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_export_encounters_writes_lf_only(tmp_path):
    params = [encounter.EncounterParams(id=f"E{i}", class_id="C2_SEP", model_id="C2_SEP_0001", t_cpa=40.0,
                                        hmd_m=100.0, side="right", vmd_m=0.0, approach_deg=180.0,
                                        gs_mps=60.0, vs_mps=0.0) for i in range(3)]
    p = tmp_path / "enc.jsonl"
    sha = encounter.export_encounters_jsonl(params, p)
    raw = p.read_bytes()
    assert b"\r" not in raw
    assert raw.count(b"\n") == 3
    import hashlib
    assert sha == hashlib.sha256(raw).hexdigest()


def _ray_cast_mask(p_cam, faces, fx, fy, cx, cy, w, h, z_near):
    """Independent point sampling: pixel centre covered iff its ray hits a triangle at depth z >= z_near."""
    jj, ii = np.mgrid[0:h, 0:w]
    d = np.stack([(ii.ravel() - cx) / fx, (jj.ravel() - cy) / fy, np.ones(ii.size)], axis=1)  # ray p = s*d, z = s
    hit = np.zeros(ii.size, dtype=bool)
    for f in faces:
        a, b, c = p_cam[f[0]], p_cam[f[1]], p_cam[f[2]]
        e1, e2 = b - a, c - a
        pv = np.cross(d, e2)
        det = pv @ e1
        ok = np.abs(det) > 1e-15
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = -a  # ray origin is the camera centre
        uu = (pv @ tv) * inv
        qv = np.cross(tv, e1)
        vv = (d * qv).sum(axis=1) * inv
        s = (qv @ e2) * inv
        hit |= ok & (uu >= 0) & (vv >= 0) & (uu + vv <= 1) & (s >= z_near)
    return hit.reshape(h, w)


def test_reference_raster_clips_meshes_crossing_the_camera_plane():
    """G2 finding: a mesh with vertices behind the camera was 'not computable'; it is now clipped at z_near."""
    from xvs.traffic import raster

    rng = np.random.default_rng(20260924)
    fx = fy = 40.0
    cx = cy = 31.5
    w = h = 64
    n_clipped = 0
    for _ in range(20):
        p = np.column_stack([rng.uniform(-3, 3, 9), rng.uniform(-3, 3, 9), rng.uniform(-2.0, 4.0, 9)])
        f = np.arange(9).reshape(3, 3)
        truth = _ray_cast_mask(p, f, fx, fy, cx, cy, w, h, raster.Z_NEAR_M)
        cmp_ = raster.compare_with_mask(truth, p, f, fx, fy, cx, cy, tie_px=1e-6)
        assert cmp_["computable"] is True
        assert cmp_["mismatch_px"] == 0
        assert cmp_["ref_px"] == int(truth.sum())
        n_clipped += cmp_["near_clipped"]
    assert n_clipped >= 15  # the draws do exercise the clipping path


def test_reference_raster_crossing_mesh_far_off_axis_is_empty_and_unclipped_path_unchanged():
    from xvs.traffic import raster

    # Side camera, intruder ~90 deg off axis (as C5_TRANSPORT_earth_2000m, rig R): empty reference, computable.
    p = np.array([[2000.0, -5.0, -8.0], [2010.0, 5.0, 20.0], [1990.0, 0.0, 3.0], [2000.0, 3.0, -2.0]])
    f = np.array([[0, 1, 2], [0, 2, 3], [1, 3, 2]])
    empty = np.zeros((1024, 1024), dtype=bool)
    cmp_ = raster.compare_with_mask(empty, p, f, 429.6, 429.6, 511.5, 511.5)
    assert cmp_["computable"] and cmp_["near_clipped"]
    assert cmp_["ref_px"] == 0 and cmp_["diff_px"] == 0

    # Mesh entirely in front: no clipping, identical to projecting and rasterizing directly.
    q = np.array([[-1.0, -1.0, 10.0], [1.5, -0.5, 11.0], [0.2, 1.2, 9.0]])
    g = np.array([[0, 1, 2]])
    u = 429.6 * q[:, 0] / q[:, 2] + 511.5 - 100
    v = 429.6 * q[:, 1] / q[:, 2] + 511.5 - 50
    direct = raster.rasterize(u, v, g, 1024, 1024)
    cmp2 = raster.compare_with_mask(direct, q, g, 429.6, 429.6, 511.5, 511.5, u0=100, v0=50)
    assert not cmp2["near_clipped"] and cmp2["diff_px"] == 0 and cmp2["ref_px"] == int(direct.sum()) > 0


def test_tie_rule_uses_the_union_boundary_not_interior_edges():
    """R-G2-2: a differing pixel on an interior shared edge, deep inside the silhouette, is a mismatch, not a tie."""
    from xvs.traffic import raster

    # a square split into two triangles along its diagonal; camera at the origin looking along +z
    z = 10.0
    fx = fy = 100.0
    cx = cy = 32.0  # the square's edges (u, v = 12 and 52) and its diagonal (u = v) pass through pixel centres
    w = h = 64
    corners = [(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)]
    p = np.array([[x, y, z] for x, y in corners], float)
    f = np.array([[0, 1, 2], [0, 2, 3]])
    u = fx * p[:, 0] / z + cx
    v = fy * p[:, 1] / z + cy
    ref = raster.rasterize(u, v, f, w, h)
    # the diagonal passes through pixel centres (i, i); remove one of them deep inside the square (a hole)
    gpu = ref.copy()
    gpu[31, 31] = False
    cmp_ = raster.compare_with_mask(gpu, p, f, fx, fy, cx, cy)
    assert cmp_["diff_px"] == 1 and cmp_["tie_px"] == 0 and cmp_["mismatch_px"] == 1
    # a pixel on the outer boundary (the square's edge u = 12) is a genuine tie
    gpu2 = ref.copy()
    ys, xs = np.nonzero(ref)
    edge = xs == xs.min()
    gpu2[ys[edge][0], xs[edge][0]] = False
    cmp2 = raster.compare_with_mask(gpu2, p, f, fx, fy, cx, cy)
    assert cmp2["diff_px"] == 1 and cmp2["tie_px"] == 1 and cmp2["mismatch_px"] == 0


def test_pinhole_in_image_edges_match_gt_convention():
    cam = PinholeCamera(width=2448, height=2048, fx=2319.0, fy=2319.0, cx=1223.5, cy=1023.5)
    assert cam.in_image(2447.5, 2047.5)
    assert not cam.in_image(2447.6, 1000.0)  # the old GT bound (u <= W) accepted this
    assert not cam.in_image(-0.6, 1000.0)


@pytest.fixture(scope="module")
def ref_renderer():
    port = _free_port()
    server = XVSServer(
        server_address=("127.0.0.1", port),
        cesium_build_dir=REPO_ROOT / "sim" / "cesium" / "node_modules" / "cesium" / "Build" / "Cesium",
        page_dir=REPO_ROOT / "sim" / "cesium" / "page",
        cache_dir=REPO_ROOT / "data" / "cache" / "usgs_imagery",
        log_dir=REPO_ROOT / "output" / "xvs" / "test_logs",
        offline=True,
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    r = XVSRenderer(port=port, backend="ref", server=server).start()
    yield r
    r.close()
    server.shutdown()
    server.server_close()


@pytest.mark.render
def test_roi_frustum_sse_denominator_and_rig_reset(ref_renderer):
    r = ref_renderer
    page = r.new_camera_page("fx_test", profile="mask_earth")
    rig = Rig.load_nominal().make_true(20260922)
    f = rig.compute_poses(37.62, -122.37, 600.0, 0.0, 0.0, 0.0)["F"]
    r.set_camera(f["position"], f["direction"], f["up"], f["fov_rad"], f["aspect_ratio"], page=page)
    r.set_roi_frustum(u0=712, v0=512, fx=2319.0, fy=2319.0, cx=1223.5, cy=1023.5, page=page)
    info = page.evaluate("""(() => { const fr = window.xvsWidget.camera.frustum;
        return {offCenter: fr instanceof Cesium.PerspectiveOffCenterFrustum, sse: fr.sseDenominator}; })()""")
    assert info["offCenter"] is True
    assert info["sse"] == pytest.approx(1024.0 / 2319.0, rel=1e-12)
    r.set_camera(f["position"], f["direction"], f["up"], f["fov_rad"], f["aspect_ratio"], page=page)
    info2 = page.evaluate("""(() => { const fr = window.xvsWidget.camera.frustum;
        return {persp: fr instanceof Cesium.PerspectiveFrustum, fov: fr.fov, sse: fr.sseDenominator}; })()""")
    assert info2["persp"] is True
    assert info2["fov"] == pytest.approx(f["fov_rad"], rel=1e-15)
    assert info2["sse"] == pytest.approx(2.0 * math.tan(0.5 * f["fov_rad"]), rel=1e-12)


@pytest.mark.render
def test_ring_roi_limb_at_10km_edge_roi(ref_renderer):
    """T-LIMB-ROI worst case of WP-02 G0 (ROI at the image edge, 10 km): must converge and meet L1."""
    r = ref_renderer
    ring = TrafficRing()
    page = r.new_camera_page("limb_roi_test", profile="mask_earth")
    t0 = ring.compute_poses(37.62, -122.37, 10000.0, 0.0, 0.0, 0.0)["T0"]
    p = np.array(t0["position"], float)
    R = np.array(t0["r_ecef_cam"], float)
    v_c = horizon.analytic_horizon_v_at_u(p, R, ring.camera_model, ring.cx)[0]
    u0, v0 = 0, int(np.clip(np.round(v_c - 512.0), 0, ring.height - 1024))
    r.set_camera(t0["position"], t0["direction"], t0["up"], math.radians(ring.fov_h_deg), 1.0, page=page)
    r.set_roi_frustum(u0=u0, v0=v0, fx=ring.fx, fy=ring.fy, cx=ring.cx, cy=ring.cy, page=page)
    res = r.render_frame(page=page)
    assert not res.incomplete
    assert res.n_renders < 200
    m = res.image[:, :, 0].astype(float) / 255.0
    errs = []
    for col in range(0, 1024, 4):
        roots = horizon.analytic_horizon_v_at_u(p, R, ring.camera_model, float(col + u0))
        if len(roots) != 1:
            continue
        v_edge = (1024 - 0.5) - m[:, col].sum()  # earth below the sky
        errs.append(v_edge - (roots[0] - v0))
    e = np.abs(np.array(errs))
    assert len(e) > 200
    assert np.median(e) <= 0.15
    assert np.percentile(e, 95) <= 0.35
    assert e.max() <= 0.50


@pytest.mark.render
def test_near_clipped_nonempty_mask_matches_reference(ref_renderer):
    """R-G2-4: a traffic mesh crossing the camera plane with a large visible part (C5 lengthwise along rig F's axis,
    10 m to the side): the GPU mask equals the near-plane-clipped reference rasterization."""
    from xvs.traffic import raster

    r = ref_renderer
    page = r.new_camera_page("clip_test", profile="mask_amodal")
    rig = Rig.load_nominal().make_true(20260922)
    f = rig.compute_poses(37.62, -122.37, 1500.0, 0.0, 0.0, 0.0)["F"]
    p_cam = np.array(f["position"], float)
    R = np.array(f["r_ecef_cam"], float)
    m_body = np.column_stack([R[:, 2], R[:, 0], R[:, 1]])  # body x along the camera axis, det +1
    p_model = p_cam + 10.0 * R[:, 0] + 5.0 * R[:, 2]
    m4 = np.eye(4)
    m4[:3, :3] = m_body
    m4[:3, 3] = p_model
    r.clear_models(page=page)
    r.add_model(url=f"http://127.0.0.1:{r.port}/assets/traffic/C5_TRANSPORT/C5_TRANSPORT_0001.glb",
                custom_shader="unlit_white", page=page)
    r.set_model_pose(0, m4.T.flatten().tolist(), page=page)
    r.set_camera(f["position"], f["direction"], f["up"], f["fov_rad"], f["aspect_ratio"], page=page)
    res = r.render_frame(page=page)
    assert not res.incomplete
    V, F = raster.glb_mesh(REPO_ROOT / "assets" / "traffic" / "C5_TRANSPORT" / "C5_TRANSPORT_0001.glb")
    pc = ((p_model[None, :] + V @ m_body.T) - p_cam[None, :]) @ R
    m = rig.camera_model
    cmp_ = raster.compare_with_mask(res.image[:, :, 0] > 127, pc, F, m.fx, m.fy, m.cx, m.cy)
    assert cmp_["near_clipped"] and cmp_["ref_px"] > 1000
    assert cmp_["mismatch_px"] == 0, cmp_


@pytest.mark.render
def test_models_have_no_dynamic_environment_map(ref_renderer):
    """WP02 v1.3 D-B2b: lit model shading must not depend on the page's render history (G3a acceptance item 3)."""
    r = ref_renderer
    page = r.new_camera_page("envmap_test", profile="rgb")
    r.clear_models(page=page)
    r.add_model(url=f"http://127.0.0.1:{r.port}/assets/traffic/C2_SEP/C2_SEP_0001.glb", page=page)
    enabled = page.evaluate("(() => { const m = window.xvsModels[window.xvsModels.length - 1]; "
                            "return m.environmentMapManager ? m.environmentMapManager.enabled : null; })()")
    assert enabled is False
