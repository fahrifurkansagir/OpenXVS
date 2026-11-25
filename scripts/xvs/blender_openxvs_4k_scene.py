"""Headless Blender 4.x Cycles GPU (`OPTIX`/`CUDA` + `OpenImageDenoise`) 4K Cinema Scene Builder.

Reads real-telemetry `outputs/film_4k/trajectories.json` + real USGS `ksfo_3dep_geoid18_v1.bin` +
real USDA NAIP `ksfo_ortho_inner_v1.jpg` / `ksfo_ortho_core2k_v2.jpg` / `ksfo_ortho_outer_v1.jpg`
to render ray-traced photorealistic 4K shots of all 4 OpenXVS showcase chapters
(`ch1_cbdr_activerange`, `ch2_gps_spoof_nav`, `ch3_thermal_fog_landing`,
`ch4_forcedland_35kt_ksfo`).

Pure-Python coordinate conversion (`NED -> Blender Z-up`), `'YXZ'` attitude matrices, heading
unwrapping, `XVS3DEP1` terrain decoding, georeferenced ortho UV projection, runway markings,
and cinematic camera-framing helpers are placed above the lazy `bpy` import so offline unit tests
in `.venv` can import and verify every math & framing invariant without a Blender binary.

Usage (inside Blender 4.x headless on Colab L4 / local GPU):
  blender -b -P scripts/xvs/blender_openxvs_4k_scene.py -- \\
    --trajectories outputs/film_4k/trajectories.json \\
    --out-dir outputs/film_4k/renders \\
    --chapter ch1_cbdr_activerange \\
    --res 3840x2160 --samples 48 --fps 30 --require-gpu
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAJECTORIES = REPO_ROOT / "outputs" / "film_4k" / "trajectories.json"
DEFAULT_TERRAIN_BIN = REPO_ROOT / "apps" / "livedeck" / "public" / "assets" / "ksfo_3dep_geoid18_v1.bin"
DEFAULT_ORTHO_JPG = REPO_ROOT / "apps" / "livedeck" / "public" / "assets" / "ksfo_ortho_core2k_v2.jpg"
DEFAULT_INNER_ORTHO_JPG = REPO_ROOT / "apps" / "livedeck" / "public" / "assets" / "ksfo_ortho_inner_v1.jpg"
DEFAULT_OUTER_ORTHO_JPG = REPO_ROOT / "apps" / "livedeck" / "public" / "assets" / "ksfo_ortho_outer_v1.jpg"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "film_4k" / "renders"

# Exact KSFO 28L threshold (MSL-zero NED) & USGS NAIP georeferencing constants
KSFO_28L_THR_NED: tuple[float, float, float] = (-786.28, 1505.37, -3.96)
KSFO_28L_HDG_RAD: float = math.radians(298.0)
WGS84_A_M: float = 6378137.0
KSFO_ORIGIN_LAT_DEG: float = 37.6188056
KSFO_ORIGIN_LON_DEG: float = -122.3754167
CORE2K_CENTER_LAT_DEG: float = 37.6155
CORE2K_CENTER_LON_DEG: float = -122.3625
CORE2K_EXTENT_KM: float = 2.0

# Blender Euler order: extrinsic Y->X->Z == intrinsic Z(heading)->X(pitch)->Y(roll) == R_z @ R_x @ R_y
AIRCRAFT_ROTATION_MODE: str = "YXZ"


# ---- Pure-Python Coordinate, Terrain, and Camera Helpers (Testable Offline Without `bpy`) ----


def ned_to_blender_xyz(north_m: float, east_m: float, down_m: float) -> tuple[float, float, float]:
    """Convert right-handed NED `(north, east, down)` to right-handed Blender Z-up `(X=east, Y=north, Z=-down)`."""
    n = float(north_m)
    e = float(east_m)
    d = float(down_m)
    if not (math.isfinite(n) and math.isfinite(e) and math.isfinite(d)):
        raise ValueError(f"Non-finite NED coordinates: ({n}, {e}, {d})")
    return (e, n, -d)


def ned_euler_to_blender_xyz_rad(
    heading_rad: float,
    pitch_rad: float,
    roll_rad: float,
) -> tuple[float, float, float]:
    """Map aircraft NED `(heading CW from North, pitch nose-up, roll right-wing-down)` to Blender `'YXZ'` Euler.

    Aircraft mesh convention in Blender local frame:
      - `+Y`: nose forward
      - `+X`: right wing
      - `+Z`: dorsal up
    In Blender's C `eulO_to_mat3`, Euler order `'YXZ'` computes extrinsic `Y -> X -> Z`, i.e.,
    `M = R_z(rot_z) @ R_x(rot_x) @ R_y(rot_y)` (equivalently intrinsic heading `Z` -> pitch `X` -> roll `Y`):
      - `rot_y = roll_rad` (positive right-wing-down rotates local `+X` toward `-Z` around nose `+Y`)
      - `rot_x = pitch_rad` (positive nose-up tilts local `+Y` toward `+Z` around wing `+X`)
      - `rot_z = -heading_rad` (heading=0 -> `+Y` North; heading=+pi/2 -> `+X` East)
    Returns `(rot_x, rot_y, rot_z)` in radians for Blender `obj.rotation_mode = 'YXZ'`.
    """
    hdg = float(heading_rad)
    pit = float(pitch_rad)
    rol = float(roll_rad)
    if not (math.isfinite(hdg) and math.isfinite(pit) and math.isfinite(rol)):
        raise ValueError(f"Non-finite NED Euler angles: ({hdg}, {pit}, {rol})")
    return (pit, rol, -hdg)


def blender_euler_yxz_to_matrix(rot_x: float, rot_y: float, rot_z: float) -> np.ndarray:
    """Compute Blender `'YXZ'` rotation matrix `R_z(rot_z) @ R_x(rot_x) @ R_y(rot_y)` (matches `eulO_to_mat3`)."""
    cx, sx = math.cos(float(rot_x)), math.sin(float(rot_x))
    cy, sy = math.cos(float(rot_y)), math.sin(float(rot_y))
    cz, sz = math.cos(float(rot_z)), math.sin(float(rot_z))
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ rx @ ry


def unwrap_headings_rad(headings_rad: Sequence[float]) -> list[float]:
    """Unwrap heading sequence with `np.unwrap` so `+/-2*pi` wraps never cause motion-blur spin smears."""
    arr = np.asarray(list(headings_rad), dtype=np.float64)
    if arr.size == 0:
        return []
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError("Non-finite heading values cannot be unwrapped")
    return [float(v) for v in np.unwrap(arr)]


def spinner_angular_step_rad(chapter_id: str) -> float:
    """Return engine spinner angular increment per frame (stopped for engine-out `ch4`, fast otherwise)."""
    if chapter_id == "ch4_forcedland_35kt_ksfo":
        return 0.0
    return 1.4


def compute_core2k_bounds_deg(
    *,
    center_lat: float = CORE2K_CENTER_LAT_DEG,
    center_lon: float = CORE2K_CENTER_LON_DEG,
    extent_km: float = CORE2K_EXTENT_KM,
    origin_lat: float = KSFO_ORIGIN_LAT_DEG,
) -> tuple[float, float, float, float]:
    """Compute exact WGS84 `(lat_min, lat_max, lon_min, lon_max)` for `ksfo_ortho_core2k_v2.jpg`."""
    half_m = float(extent_km) * 500.0
    m_per_deg_lat = (math.pi / 180.0) * WGS84_A_M
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(float(origin_lat)))
    d_lat = half_m / m_per_deg_lat
    d_lon = half_m / m_per_deg_lon
    return (
        float(center_lat) - d_lat,
        float(center_lat) + d_lat,
        float(center_lon) - d_lon,
        float(center_lon) + d_lon,
    )


def compute_core2k_uv_footprint_in_inner(inner_pack: dict[str, Any]) -> dict[str, float]:
    """Project `ksfo_ortho_core2k_v2.jpg` (`2x2 km`) onto `inner_ksfo` (`10x10 km`) Blender UV `(u, v)`.

    In Blender UV coordinates:
      - `u = 0 -> 1` spans West (`lon_min`) to East (`lon_max`).
      - `v = 0 -> 1` spans South (`lat_min`) to North (`lat_max`).
    """
    origin_lat = float(inner_pack["origin_lat_deg"])
    c_lat_min, c_lat_max, c_lon_min, c_lon_max = compute_core2k_bounds_deg(origin_lat=origin_lat)
    i_lat_min = float(inner_pack["lat_min"])
    i_lat_max = float(inner_pack["lat_max"])
    i_lon_min = float(inner_pack["lon_min"])
    i_lon_max = float(inner_pack["lon_max"])

    u_min = (c_lon_min - i_lon_min) / (i_lon_max - i_lon_min)
    u_max = (c_lon_max - i_lon_min) / (i_lon_max - i_lon_min)
    v_min = (c_lat_min - i_lat_min) / (i_lat_max - i_lat_min)
    v_max = (c_lat_max - i_lat_min) / (i_lat_max - i_lat_min)
    return {
        "u_min": u_min,
        "u_max": u_max,
        "v_min": v_min,
        "v_max": v_max,
        "u_span": u_max - u_min,
        "v_span": v_max - v_min,
    }


def inner_uv_to_core2k_uv(u: float, v: float, footprint: dict[str, float]) -> tuple[float, float]:
    """Map an `inner_ksfo` UV coordinate `(u, v)` into `ksfo_ortho_core2k_v2.jpg` local UV `[0, 1]^2`."""
    uc = (float(u) - float(footprint["u_min"])) / float(footprint["u_span"])
    vc = (float(v) - float(footprint["v_min"])) / float(footprint["v_span"])
    return (uc, vc)


def core2k_feather_weight(
    u_core: float,
    v_core: float,
    edge_lo: float = 0.02,
    edge_hi: float = 0.12,
) -> float:
    """Smoothstep edge-feather blend weight matching the Blender `UVCore2k` shader node mask."""
    edge = min(float(u_core), 1.0 - float(u_core), float(v_core), 1.0 - float(v_core))
    if edge <= edge_lo:
        return 0.0
    if edge >= edge_hi:
        return 1.0
    t = (edge - edge_lo) / (edge_hi - edge_lo)
    return t * t * (3.0 - 2.0 * t)


def generate_runway_markings_rgba(width: int = 256, height: int = 2048) -> np.ndarray:
    """Generate procedural FAA Part 139 KSFO 28L asphalt + painted markings RGBA texture `(H, W, 4)` float32.

    Row `0` maps to Blender `v = 0` (Runway 28L threshold); row `H - 1` maps to `v = 1` (far 10R end).
    """
    w = int(width)
    h = int(height)
    img = np.zeros((h, w, 4), dtype=np.float32)
    # Dark weathered asphalt base (sRGB-encoded: 0.24..0.27 sRGB ~= 0.047..0.059 linear albedo)
    img[:, :, 0] = 0.24
    img[:, :, 1] = 0.25
    img[:, :, 2] = 0.27
    img[:, :, 3] = 1.0

    # Subtle tire rubber darkening along centerline touchdown zones
    u_lin = np.linspace(0.0, 1.0, w, dtype=np.float32)[np.newaxis, :]
    v_lin = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, np.newaxis]
    center_band = np.exp(-((u_lin - 0.5) / 0.14) ** 2)
    tdz_near = np.exp(-((v_lin - 0.11) / 0.065) ** 2)
    tdz_far = np.exp(-((v_lin - 0.89) / 0.065) ** 2)
    rubber_dark = 1.0 - 0.36 * center_band * np.clip(tdz_near + tdz_far, 0.0, 1.0)
    img[:, :, :3] *= rubber_dark[:, :, np.newaxis]

    paint = np.array([0.90, 0.92, 0.94], dtype=np.float32)

    def _fill_rect(u0: float, u1: float, v0: float, v1: float) -> None:
        x0 = max(0, min(w, int(round(u0 * w))))
        x1 = max(0, min(w, int(round(u1 * w))))
        y0 = max(0, min(h, int(round(v0 * h))))
        y1 = max(0, min(h, int(round(v1 * h))))
        img[y0:y1, x0:x1, :3] = paint

    # Continuous runway side edge stripes
    _fill_rect(0.045, 0.072, 0.004, 0.996)
    _fill_rect(0.928, 0.955, 0.004, 0.996)

    # 10-bar piano-key threshold markings at both ends (28L near v=0, 10R near v=1)
    for v_start, v_end in ((0.006, 0.024), (0.976, 0.994)):
        for bar_i in range(10):
            u0 = 0.105 + bar_i * 0.082
            _fill_rect(u0, u0 + 0.048, v_start, v_end)

    # Aiming point blocks (~300 m from each threshold -> v ~ 0.086 and 0.914)
    for v_aim in (0.080, 0.900):
        _fill_rect(0.18, 0.34, v_aim, v_aim + 0.020)
        _fill_rect(0.66, 0.82, v_aim, v_aim + 0.020)

    # Touchdown zone paired bars
    for v_td in (0.048, 0.118, 0.150, 0.838, 0.870, 0.940):
        _fill_rect(0.22, 0.30, v_td, v_td + 0.012)
        _fill_rect(0.70, 0.78, v_td, v_td + 0.012)

    # Dashed runway centerline (skip threshold piano keys)
    v_cur = 0.042
    while v_cur < 0.948:
        _fill_rect(0.486, 0.514, v_cur, min(0.955, v_cur + 0.012))
        v_cur += 0.024

    return img


def parse_resolution(res_str: str) -> tuple[int, int]:
    """Parse `'WxH'` resolution string (e.g. `'1920x1080'`, `'3840x2160'`) with sanity bounds."""
    parts = res_str.strip().lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid resolution format {res_str!r}; expected 'WIDTHxHEIGHT'")
    w = int(parts[0])
    h = int(parts[1])
    if w < 320 or h < 240 or w > 7680 or h > 4320:
        raise ValueError(f"Resolution out of supported range: {w}x{h}")
    return (w, h)


def parse_xvs3dep_pack(
    bin_path: Path = DEFAULT_TERRAIN_BIN,
    ring_name: str = "inner_ksfo",
) -> dict[str, Any]:
    """Decode `XVS3DEP1` binary heightfield into MSL float32 meters + extent metadata."""
    raw = Path(bin_path).read_bytes()
    if len(raw) < 16 or raw[:8] != b"XVS3DEP1":
        raise ValueError(f"Invalid XVS3DEP1 magic in {bin_path}")
    _, jlen = struct.unpack("<2I", raw[8:16])
    hdr = json.loads(raw[16 : 16 + jlen].decode("utf-8"))
    quant = hdr["quantization"]
    h_min = float(quant["h_min_m"])
    h_step = float(quant["h_step_m"])
    geoid_n = float(hdr["ksfo_28l_geoid_undulation_m"])

    chosen_ring = None
    for ring in hdr["rings"]:
        if str(ring["name"]) == ring_name:
            chosen_ring = ring
            break
    if chosen_ring is None:
        chosen_ring = hdr["rings"][0]

    w = int(chosen_ring["width"])
    h = int(chosen_ring["height"])
    off = int(chosen_ring["byte_offset"])
    length = int(chosen_ring["byte_length"])
    u16 = np.frombuffer(raw[off : off + length], dtype="<u2").reshape((h, w))
    h_wgs84 = h_min + u16.astype(np.float32) * h_step
    h_msl = h_wgs84 - geoid_n

    return {
        "ring_name": str(chosen_ring["name"]),
        "width": w,
        "height": h,
        "extent_m": float(chosen_ring["extent_km"]) * 1000.0,
        "lat_min": float(chosen_ring["lat_min"]),
        "lat_max": float(chosen_ring["lat_max"]),
        "lon_min": float(chosen_ring["lon_min"]),
        "lon_max": float(chosen_ring["lon_max"]),
        "origin_lat_deg": float(hdr["origin_lat_deg"]),
        "origin_lon_deg": float(hdr["origin_lon_deg"]),
        "heights_msl_m": h_msl,
        "runways": hdr["runways"],
    }


def compute_cinematic_camera_pose(
    chapter_id: str,
    frame_data: dict[str, Any],
    t_norm: float,
) -> dict[str, Any]:
    """Compute smooth cinematic Blender Z-up camera location, look-at target, and focal length (mm).

    Dynamic multi-beat framing tailored per chapter:
      - `ch1_cbdr_activerange`: wingman chase transitioning smoothly to dramatic side-angle pass.
      - `ch2_gps_spoof_nav`: elevated tactical three-quarter chase showing spoofed ghost divergence.
      - `ch3_thermal_fog_landing`: over-shoulder cockpit-top approach into glowing KSFO 28L strobes.
      - `ch4_forcedland_35kt_ksfo`: heroic sweeping wingman chase framing curved 3D trochoid ribbon.
    """
    u = max(0.0, min(1.0, float(t_norm)))
    own = frame_data["ownship_ned"]
    ox, oy, oz = ned_to_blender_xyz(own["north_m"], own["east_m"], own["down_m"])
    hdg = float(own["heading_rad"])
    # Forward and right unit vectors in Blender Z-up (X=East, Y=North)
    fx, fy = math.sin(hdg), math.cos(hdg)
    rx, ry = math.cos(hdg), -math.sin(hdg)

    if chapter_id == "ch1_cbdr_activerange":
        # Smooth sweep from rear-right wingman chase into heroic broadside framing both aircraft
        back_m = 26.0 + 10.0 * math.sin(math.pi * u)
        side_m = 13.0 * math.cos(0.75 * math.pi * u) + 8.0
        up_m = 6.5 + 3.5 * math.sin(math.pi * u)
        cam_loc = (ox - back_m * fx + side_m * rx, oy - back_m * fy + side_m * ry, oz + up_m)
        if "intruder_ned" in frame_data:
            intr = frame_data["intruder_ned"]
            ix, iy, iz = ned_to_blender_xyz(intr["north_m"], intr["east_m"], intr["down_m"])
            blend = 0.22 + 0.14 * math.sin(math.pi * u)
            look_at = (
                ox * (1.0 - blend) + ix * blend,
                oy * (1.0 - blend) + iy * blend,
                oz * (1.0 - blend) + iz * blend,
            )
        else:
            look_at = (ox + 18.0 * fx, oy + 18.0 * fy, oz)
        lens_mm = 38.0 + 12.0 * math.sin(math.pi * u)
        rig_name = "wingman_cbdr_sweep"
    elif chapter_id == "ch2_gps_spoof_nav":
        # Elevated oblique tactical chase revealing true path vs diverging red GPS spoof ghost
        back_m = 36.0
        side_m = -22.0 + 8.0 * u
        up_m = 18.0 + 6.0 * u
        cam_loc = (ox - back_m * fx + side_m * rx, oy - back_m * fy + side_m * ry, oz + up_m)
        look_at = (ox + 28.0 * fx, oy + 28.0 * fy, oz - 4.0)
        lens_mm = 34.0
        rig_name = "tactical_spoof_chase"
    elif chapter_id == "ch3_thermal_fog_landing":
        # Tight dorsal chase looking straight down glideslope into glowing KSFO 28L ALSF-2 strobes
        back_m = 22.0
        side_m = 4.5 * math.sin(math.pi * u)
        up_m = 5.5
        cam_loc = (ox - back_m * fx + side_m * rx, oy - back_m * fy + side_m * ry, oz + up_m)
        tx, ty, tz = ned_to_blender_xyz(*KSFO_28L_THR_NED)
        look_at = (ox * 0.45 + tx * 0.55, oy * 0.45 + ty * 0.55, oz * 0.45 + (tz + 2.0) * 0.55)
        lens_mm = 32.0
        rig_name = "glideslope_fog_chase"
    else:
        # ch4_forcedland_35kt_ksfo: sweeping hero wingman chase showcasing 35 kt crab & trochoid ribbon
        back_m = 30.0 - 6.0 * u
        side_m = -16.0 * math.cos(0.5 * math.pi * u) - 6.0
        up_m = 9.0 - 2.5 * u
        cam_loc = (ox - back_m * fx + side_m * rx, oy - back_m * fy + side_m * ry, oz + up_m)
        look_at = (ox + 34.0 * fx, oy + 34.0 * fy, oz - 5.0)
        lens_mm = 35.0
        rig_name = "trochoid_hero_chase"

    return {
        "rig_name": rig_name,
        "location": (round(cam_loc[0], 4), round(cam_loc[1], 4), round(cam_loc[2], 4)),
        "look_at": (round(look_at[0], 4), round(look_at[1], 4), round(look_at[2], 4)),
        "lens_mm": round(float(lens_mm), 2),
    }


# ---- Headless Blender 4.x Cycles GPU Scene Implementation (Lazy `bpy` Import) ----


def _configure_cycles_gpu(
    scene: Any,
    res_w: int,
    res_h: int,
    samples: int,
    fps: int,
    *,
    require_gpu: bool = False,
) -> dict[str, Any]:
    import bpy  # type: ignore[import-not-found]

    scene.render.engine = "CYCLES"
    # Keep BVH/textures resident across animation frames (2.5x-4x speedup on 1,320 frames)
    scene.render.use_persistent_data = True
    scene.cycles.samples = int(samples)
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.02
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 4
    scene.cycles.volume_bounces = 1
    if hasattr(scene.cycles, "volume_step_rate"):
        scene.cycles.volume_step_rate = 2.0
    if hasattr(scene.cycles, "volume_max_steps"):
        scene.cycles.volume_max_steps = 128
    scene.cycles.transparent_max_bounces = 8

    # Motion blur for authentic spinning helicopter rotors & twinjet propellers
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.35

    # Prefer OPTIX -> CUDA strictly GPU-only (disable CPU drag); optional fail-fast via --require-gpu
    prefs = bpy.context.preferences.addons["cycles"].preferences
    device_activated = False
    active_backend = "CPU"
    selected_devices: list[str] = []
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            matched: list[str] = []
            for dev in prefs.devices:
                use_dev = bool(dev.type == backend)
                dev.use = use_dev
                if use_dev:
                    matched.append(f"{dev.type}:{dev.name}")
            if matched:
                scene.cycles.device = "GPU"
                device_activated = True
                active_backend = backend
                selected_devices = matched
                break
        except Exception as exc:
            print(f"[OpenXVS Cycles] Backend {backend} probe failed: {type(exc).__name__}: {exc}", flush=True)
            continue

    if not device_activated:
        if require_gpu:
            raise RuntimeError(
                "Cycles GPU activation failed: no OPTIX or CUDA device found and --require-gpu is set; "
                "refusing silent CPU fallback."
            )
        scene.cycles.device = "CPU"
        print("[OpenXVS Cycles] WARNING: No OPTIX/CUDA GPU found; falling back to CPU.", flush=True)
    else:
        print(
            f"[OpenXVS Cycles] Active GPU backend={active_backend} devices={selected_devices} "
            f"persistent_data={scene.render.use_persistent_data}",
            flush=True,
        )

    # Blender 4.2 GPU OpenImageDenoise (huge 4K speedup on NVIDIA L4 vs 2-vCPU Colab denoising)
    if hasattr(scene.cycles, "denoising_use_gpu"):
        scene.cycles.denoising_use_gpu = bool(device_activated)
        print(f"[OpenXVS Cycles] denoising_use_gpu={scene.cycles.denoising_use_gpu}", flush=True)

    scene.render.resolution_x = int(res_w)
    scene.render.resolution_y = int(res_h)
    scene.render.resolution_percentage = 100
    scene.render.fps = int(fps)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    # Cinematic AgX / Filmic tone mapping
    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Punchy"
    except Exception:
        scene.view_settings.view_transform = "Filmic"
    scene.view_settings.exposure = 0.15

    return {
        "device": str(scene.cycles.device),
        "backend": active_backend,
        "selected_devices": selected_devices,
        "use_persistent_data": bool(scene.render.use_persistent_data),
        "denoising_use_gpu": bool(getattr(scene.cycles, "denoising_use_gpu", False)),
    }


def _setup_compositor_bloom(scene: Any) -> None:
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    rl = tree.nodes.new("CompositorNodeRLayers")
    glare = tree.nodes.new("CompositorNodeGlare")
    glare.glare_type = "FOG_GLOW"
    glare.quality = "HIGH"
    glare.threshold = 1.25
    glare.size = 7
    glare.mix = -0.2
    comp = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(rl.outputs["Image"], glare.inputs["Image"])
    tree.links.new(glare.outputs["Image"], comp.inputs["Image"])


def _setup_nishita_sky_and_clouds(scene: Any, chapter_id: str) -> Any:
    import bpy  # type: ignore[import-not-found]

    world = bpy.data.worlds.new("OpenXVS_World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    sky.sun_disc = True
    sky.sun_size = math.radians(0.65)
    sky.sun_intensity = 0.85 if chapter_id != "ch3_thermal_fog_landing" else 0.45
    sky.sun_elevation = math.radians(8.5)
    sky.sun_rotation = math.radians(245.0)
    sky.air_density = 1.2
    sky.dust_density = 1.35
    sky.ozone_density = 1.8
    bg.inputs["Strength"].default_value = 0.85
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    # High-altitude golden-hour volumetric cloud slab (Principled Volume + World-Meter 3D Noise)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 1050.0))
    cloud_box = bpy.context.active_object
    cloud_box.name = "Volumetric_Clouds"
    cloud_box.scale = (22000.0, 22000.0, 280.0)
    cmat = bpy.data.materials.new("CloudVolumeMat")
    cmat.use_nodes = True
    cnt = cmat.node_tree
    cnt.nodes.clear()
    cout = cnt.nodes.new("ShaderNodeOutputMaterial")
    vol = cnt.nodes.new("ShaderNodeVolumePrincipled")
    vol.inputs["Color"].default_value = (0.98, 0.90, 0.82, 1.0)
    vol.inputs["Anisotropy"].default_value = 0.62
    # World-space meters via ShaderNodeNewGeometry.Position (unscaled TexCoord.Object stays in [-0.5, 0.5]!)
    geom = cnt.nodes.new("ShaderNodeNewGeometry")
    noise = cnt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.00055
    noise.inputs["Detail"].default_value = 7.0
    noise.inputs["Roughness"].default_value = 0.58
    ramp = cnt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.52
    ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    ramp.color_ramp.elements[1].position = 0.78
    ramp.color_ramp.elements[1].color = (0.004, 0.004, 0.004, 1.0)
    cnt.links.new(geom.outputs["Position"], noise.inputs["Vector"])
    cnt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    cnt.links.new(ramp.outputs["Color"], vol.inputs["Density"])
    cnt.links.new(vol.outputs["Volume"], cout.inputs["Volume"])
    cloud_box.data.materials.append(cmat)

    # Optional coastal advection fog volume for ch3 (density animated per frame)
    fog_density_socket = None
    if chapter_id == "ch3_thermal_fog_landing":
        tx, ty, _ = ned_to_blender_xyz(*KSFO_28L_THR_NED)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(tx + 350.0, ty - 200.0, 75.0))
        fog_box = bpy.context.active_object
        fog_box.name = "Coastal_Advection_Fog"
        fog_box.scale = (3600.0, 3600.0, 170.0)
        fmat = bpy.data.materials.new("AdvectionFogMat")
        fmat.use_nodes = True
        fnt = fmat.node_tree
        fnt.nodes.clear()
        fout = fnt.nodes.new("ShaderNodeOutputMaterial")
        fvol = fnt.nodes.new("ShaderNodeVolumePrincipled")
        fvol.inputs["Color"].default_value = (0.86, 0.90, 0.94, 1.0)
        fvol.inputs["Anisotropy"].default_value = 0.45
        fvol.inputs["Density"].default_value = 0.001
        fnt.links.new(fvol.outputs["Volume"], fout.inputs["Volume"])
        fog_box.data.materials.append(fmat)
        fog_density_socket = fvol.inputs["Density"]

    return fog_density_socket


def _resolve_optional_sibling_asset(
    explicit_path: Path | None,
    reference_asset: Path,
    filename: str,
    default_path: Path,
) -> Path | None:
    """Resolve an optional ortho asset from explicit path -> sibling of `reference_asset` -> repo default."""
    if explicit_path is not None and Path(explicit_path).is_file():
        return Path(explicit_path)
    sibling = Path(reference_asset).parent / filename
    if sibling.is_file():
        return sibling
    if Path(default_path).is_file():
        return Path(default_path)
    return None


def _build_ksfo_terrain_and_bay(
    terrain_bin: Path,
    ortho_jpg: Path,
    *,
    inner_ortho_jpg: Path | None = None,
    outer_ortho_jpg: Path | None = None,
) -> None:
    import bmesh  # type: ignore[import-not-found]
    import bpy  # type: ignore[import-not-found]

    resolved_inner = _resolve_optional_sibling_asset(
        inner_ortho_jpg, ortho_jpg, "ksfo_ortho_inner_v1.jpg", DEFAULT_INNER_ORTHO_JPG
    )
    resolved_outer = _resolve_optional_sibling_asset(
        outer_ortho_jpg, ortho_jpg, "ksfo_ortho_outer_v1.jpg", DEFAULT_OUTER_ORTHO_JPG
    )
    resolved_core = Path(ortho_jpg) if Path(ortho_jpg).is_file() else None

    # 1. Inner KSFO 10x10 km terrain mesh (`inner_ksfo`) with dual UV layers:
    #    - `UVMap`: 1:1 with `ksfo_ortho_inner_v1.jpg` (10x10 km)
    #    - `UVCore2k`: georeferenced 2x2 km airport patch for `ksfo_ortho_core2k_v2.jpg`
    pack = parse_xvs3dep_pack(terrain_bin, ring_name="inner_ksfo")
    w = int(pack["width"])
    h = int(pack["height"])
    extent = float(pack["extent_m"])
    heights = pack["heights_msl_m"]
    core_fp = compute_core2k_uv_footprint_in_inner(pack)

    step = 2
    gw = w // step
    gh = h // step
    mesh = bpy.data.meshes.new("KSFO_3DEP_Terrain")
    bm = bmesh.new()
    verts = [[None] * gw for _ in range(gh)]
    uv_layer = bm.loops.layers.uv.new("UVMap")
    uv_core_layer = bm.loops.layers.uv.new("UVCore2k")

    for iy in range(gh):
        v_frac = float(iy) / float(gh - 1)
        north = (0.5 - v_frac) * extent  # top row is North
        for ix in range(gw):
            u_frac = float(ix) / float(gw - 1)
            east = (u_frac - 0.5) * extent
            z_msl = float(heights[iy * step, ix * step])
            verts[iy][ix] = bm.verts.new((east, north, z_msl))

    bm.verts.ensure_lookup_table()
    for iy in range(gh - 1):
        for ix in range(gw - 1):
            f = bm.faces.new((verts[iy][ix], verts[iy][ix + 1], verts[iy + 1][ix + 1], verts[iy + 1][ix]))
            f.smooth = True
            for loop, (lu, lv) in zip(
                f.loops,
                [
                    (ix / (gw - 1), 1.0 - iy / (gh - 1)),
                    ((ix + 1) / (gw - 1), 1.0 - iy / (gh - 1)),
                    ((ix + 1) / (gw - 1), 1.0 - (iy + 1) / (gh - 1)),
                    (ix / (gw - 1), 1.0 - (iy + 1) / (gh - 1)),
                ],
            ):
                loop[uv_layer].uv = (lu, lv)
                loop[uv_core_layer].uv = inner_uv_to_core2k_uv(lu, lv, core_fp)

    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("KSFO_3DEP_Terrain", mesh)
    bpy.context.collection.objects.link(obj)

    mat = bpy.data.materials.new("KSFO_Ortho_Mat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = 0.82

    def _build_core2k_feather_mask(node_tree: Any, uv_core_out: Any) -> Any:
        sep = node_tree.nodes.new("ShaderNodeSeparateXYZ")
        node_tree.links.new(uv_core_out, sep.inputs["Vector"])

        inv_u = node_tree.nodes.new("ShaderNodeMath")
        inv_u.operation = "SUBTRACT"
        inv_u.inputs[0].default_value = 1.0
        node_tree.links.new(sep.outputs["X"], inv_u.inputs[1])

        edge_u = node_tree.nodes.new("ShaderNodeMath")
        edge_u.operation = "MINIMUM"
        node_tree.links.new(sep.outputs["X"], edge_u.inputs[0])
        node_tree.links.new(inv_u.outputs["Value"], edge_u.inputs[1])

        inv_v = node_tree.nodes.new("ShaderNodeMath")
        inv_v.operation = "SUBTRACT"
        inv_v.inputs[0].default_value = 1.0
        node_tree.links.new(sep.outputs["Y"], inv_v.inputs[1])

        edge_v = node_tree.nodes.new("ShaderNodeMath")
        edge_v.operation = "MINIMUM"
        node_tree.links.new(sep.outputs["Y"], edge_v.inputs[0])
        node_tree.links.new(inv_v.outputs["Value"], edge_v.inputs[1])

        edge_min = node_tree.nodes.new("ShaderNodeMath")
        edge_min.operation = "MINIMUM"
        node_tree.links.new(edge_u.outputs["Value"], edge_min.inputs[0])
        node_tree.links.new(edge_v.outputs["Value"], edge_min.inputs[1])

        feather = node_tree.nodes.new("ShaderNodeMapRange")
        feather.clamp = True
        feather.interpolation_type = "SMOOTHSTEP"
        feather.inputs["From Min"].default_value = 0.02
        feather.inputs["From Max"].default_value = 0.12
        feather.inputs["To Min"].default_value = 0.0
        feather.inputs["To Max"].default_value = 1.0
        node_tree.links.new(edge_min.outputs["Value"], feather.inputs["Value"])
        return feather.outputs["Result"]

    if resolved_inner is not None and resolved_core is not None:
        uv_inner_node = nt.nodes.new("ShaderNodeUVMap")
        uv_inner_node.uv_map = "UVMap"
        tex_inner = nt.nodes.new("ShaderNodeTexImage")
        tex_inner.image = bpy.data.images.load(str(resolved_inner))
        tex_inner.extension = "EXTEND"
        nt.links.new(uv_inner_node.outputs["UV"], tex_inner.inputs["Vector"])

        uv_core_node = nt.nodes.new("ShaderNodeUVMap")
        uv_core_node.uv_map = "UVCore2k"
        tex_core = nt.nodes.new("ShaderNodeTexImage")
        tex_core.image = bpy.data.images.load(str(resolved_core))
        tex_core.extension = "CLIP"
        nt.links.new(uv_core_node.outputs["UV"], tex_core.inputs["Vector"])

        feather_out = _build_core2k_feather_mask(nt, uv_core_node.outputs["UV"])
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.clamp_factor = True
        sock_fac = next(s for s in mix.inputs if s.name == "Factor" and s.type == "VALUE")
        sock_a = next(s for s in mix.inputs if s.name == "A" and s.type == "RGBA")
        sock_b = next(s for s in mix.inputs if s.name == "B" and s.type == "RGBA")
        sock_res = next(s for s in mix.outputs if s.name == "Result" and s.type == "RGBA")
        nt.links.new(feather_out, sock_fac)
        nt.links.new(tex_inner.outputs["Color"], sock_a)
        nt.links.new(tex_core.outputs["Color"], sock_b)
        nt.links.new(sock_res, bsdf.inputs["Base Color"])
    elif resolved_inner is not None:
        uv_inner_node = nt.nodes.new("ShaderNodeUVMap")
        uv_inner_node.uv_map = "UVMap"
        tex_inner = nt.nodes.new("ShaderNodeTexImage")
        tex_inner.image = bpy.data.images.load(str(resolved_inner))
        tex_inner.extension = "EXTEND"
        nt.links.new(uv_inner_node.outputs["UV"], tex_inner.inputs["Vector"])
        nt.links.new(tex_inner.outputs["Color"], bsdf.inputs["Base Color"])
    elif resolved_core is not None:
        # Never stretch core2k 5x across 10 km; clip strictly to its 2x2 km georeferenced footprint
        uv_core_node = nt.nodes.new("ShaderNodeUVMap")
        uv_core_node.uv_map = "UVCore2k"
        tex_core = nt.nodes.new("ShaderNodeTexImage")
        tex_core.image = bpy.data.images.load(str(resolved_core))
        tex_core.extension = "CLIP"
        nt.links.new(uv_core_node.outputs["UV"], tex_core.inputs["Vector"])
        feather_out = _build_core2k_feather_mask(nt, uv_core_node.outputs["UV"])
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.clamp_factor = True
        sock_fac = next(s for s in mix.inputs if s.name == "Factor" and s.type == "VALUE")
        sock_a = next(s for s in mix.inputs if s.name == "A" and s.type == "RGBA")
        sock_b = next(s for s in mix.inputs if s.name == "B" and s.type == "RGBA")
        sock_res = next(s for s in mix.outputs if s.name == "Result" and s.type == "RGBA")
        sock_a.default_value = (0.14, 0.18, 0.14, 1.0)
        nt.links.new(feather_out, sock_fac)
        nt.links.new(tex_core.outputs["Color"], sock_b)
        nt.links.new(sock_res, bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (0.14, 0.18, 0.14, 1.0)
    obj.data.materials.append(mat)

    # 2. Outer SF Bay 40x40 km terrain ring (`outer_sf_bay`) eliminating 10 km horizon cliff cutoff
    outer_pack = parse_xvs3dep_pack(terrain_bin, ring_name="outer_sf_bay")
    if outer_pack["ring_name"] == "outer_sf_bay":
        ow = int(outer_pack["width"])
        oh = int(outer_pack["height"])
        oext = float(outer_pack["extent_m"])
        oheights = outer_pack["heights_msl_m"]
        ogw = ow // step
        ogh = oh // step
        omesh = bpy.data.meshes.new("KSFO_3DEP_Outer_Bay")
        obm = bmesh.new()
        overts = [[None] * ogw for _ in range(ogh)]
        ocoords = [[(0.0, 0.0)] * ogw for _ in range(ogh)]
        ouv_layer = obm.loops.layers.uv.new("UVMap")
        inner_cutout_m = 0.46 * extent  # 4600 m cutout inside 5000 m inner perimeter
        inner_skirt_m = 0.52 * extent   # 5200 m overlap skirt tucked -2 m below inner mesh

        for iy in range(ogh):
            v_frac = float(iy) / float(ogh - 1)
            north = (0.5 - v_frac) * oext
            for ix in range(ogw):
                u_frac = float(ix) / float(ogw - 1)
                east = (u_frac - 0.5) * oext
                z_msl = float(oheights[iy * step, ix * step])
                if abs(east) <= inner_skirt_m and abs(north) <= inner_skirt_m:
                    z_msl -= 2.0
                overts[iy][ix] = obm.verts.new((east, north, z_msl))
                ocoords[iy][ix] = (east, north)

        obm.verts.ensure_lookup_table()
        for iy in range(ogh - 1):
            for ix in range(ogw - 1):
                corners = (
                    ocoords[iy][ix],
                    ocoords[iy][ix + 1],
                    ocoords[iy + 1][ix + 1],
                    ocoords[iy + 1][ix],
                )
                if all(abs(ce) < inner_cutout_m and abs(cn) < inner_cutout_m for ce, cn in corners):
                    continue
                f = obm.faces.new(
                    (overts[iy][ix], overts[iy][ix + 1], overts[iy + 1][ix + 1], overts[iy + 1][ix])
                )
                f.smooth = True
                for loop, (lu, lv) in zip(
                    f.loops,
                    [
                        (ix / (ogw - 1), 1.0 - iy / (ogh - 1)),
                        ((ix + 1) / (ogw - 1), 1.0 - iy / (ogh - 1)),
                        ((ix + 1) / (ogw - 1), 1.0 - (iy + 1) / (ogh - 1)),
                        (ix / (ogw - 1), 1.0 - (iy + 1) / (ogh - 1)),
                    ],
                ):
                    loop[ouv_layer].uv = (lu, lv)

        obm.to_mesh(omesh)
        obm.free()
        oobj = bpy.data.objects.new("KSFO_3DEP_Outer_Bay", omesh)
        bpy.context.collection.objects.link(oobj)

        omat = bpy.data.materials.new("KSFO_Outer_Ortho_Mat")
        omat.use_nodes = True
        ont = omat.node_tree
        obsdf = ont.nodes.get("Principled BSDF")
        obsdf.inputs["Roughness"].default_value = 0.85
        if resolved_outer is not None:
            otex = ont.nodes.new("ShaderNodeTexImage")
            otex.image = bpy.data.images.load(str(resolved_outer))
            otex.extension = "EXTEND"
            ont.links.new(otex.outputs["Color"], obsdf.inputs["Base Color"])
        else:
            obsdf.inputs["Base Color"].default_value = (0.14, 0.18, 0.14, 1.0)
        oobj.data.materials.append(omat)

    # 3. Glossy ray-traced SF Bay Ocean plane at Z = +0.35 m MSL (clears +0.034 m 3DEP bay flats cleanly)
    bpy.ops.mesh.primitive_plane_add(size=60000.0, location=(0.0, 0.0, 0.35))
    water = bpy.context.active_object
    water.name = "SF_Bay_Water"
    wmat = bpy.data.materials.new("SFBayWaterMat")
    wmat.use_nodes = True
    wnt = wmat.node_tree
    wbsdf = wnt.nodes.get("Principled BSDF")
    wbsdf.inputs["Base Color"].default_value = (0.012, 0.042, 0.082, 1.0)
    wbsdf.inputs["Roughness"].default_value = 0.08
    wbsdf.inputs["IOR"].default_value = 1.333
    if "Specular IOR Level" in wbsdf.inputs:
        wbsdf.inputs["Specular IOR Level"].default_value = 0.85
    tc = wnt.nodes.new("ShaderNodeTexCoord")
    wn = wnt.nodes.new("ShaderNodeTexNoise")
    wn.inputs["Scale"].default_value = 0.045
    wn.inputs["Detail"].default_value = 8.0
    wn.inputs["Roughness"].default_value = 0.55
    bump = wnt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.14
    wnt.links.new(tc.outputs["Object"], wn.inputs["Vector"])
    wnt.links.new(wn.outputs["Fac"], bump.inputs["Height"])
    wnt.links.new(bump.outputs["Normal"], wbsdf.inputs["Normal"])
    water.data.materials.append(wmat)

    # 4. KSFO 28L Runway asphalt strip with procedural painted centerline/threshold markings + ALSF-2 + PAPI
    tx, ty, tz = ned_to_blender_xyz(*KSFO_28L_THR_NED)
    fx, fy = math.sin(KSFO_28L_HDG_RAD), math.cos(KSFO_28L_HDG_RAD)
    rx, ry = math.cos(KSFO_28L_HDG_RAD), -math.sin(KSFO_28L_HDG_RAD)
    rwy_len = 3468.9
    rwy_wid = 60.96
    cx, cy = tx + 0.5 * rwy_len * fx, ty + 0.5 * rwy_len * fy
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(cx, cy, tz + 0.25))
    rwy = bpy.context.active_object
    rwy.name = "KSFO_Runway_28L"
    rwy.scale = (rwy_wid, rwy_len, 1.0)
    rwy.rotation_euler = (0.0, 0.0, -KSFO_28L_HDG_RAD)
    rmat = bpy.data.materials.new("RunwayAsphaltMat")
    rmat.use_nodes = True
    rnt = rmat.node_tree
    rbsdf = rnt.nodes.get("Principled BSDF")
    rbsdf.inputs["Roughness"].default_value = 0.65
    rwy_rgba = generate_runway_markings_rgba(width=256, height=2048)
    rwy_img = bpy.data.images.new("KSFO_28L_Markings", width=256, height=2048, alpha=True)
    rwy_img.colorspace_settings.name = "sRGB"
    rwy_img.pixels.foreach_set(rwy_rgba.ravel())
    rwy_img.pack()
    rtex = rnt.nodes.new("ShaderNodeTexImage")
    rtex.image = rwy_img
    rtex.extension = "EXTEND"
    rnt.links.new(rtex.outputs["Color"], rbsdf.inputs["Base Color"])
    rwy.data.materials.append(rmat)

    # Emissive material factory
    def _emissive(name: str, rgba: tuple[float, float, float, float], strength: float) -> Any:
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        nt_e = m.node_tree
        nt_e.nodes.clear()
        o = nt_e.nodes.new("ShaderNodeOutputMaterial")
        em = nt_e.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = rgba
        em.inputs["Strength"].default_value = strength
        nt_e.links.new(em.outputs["Emission"], o.inputs["Surface"])
        return m

    mat_white = _emissive("ALSF2_White", (1.0, 0.96, 0.88, 1.0), 32.0)
    mat_green = _emissive("Threshold_Green", (0.12, 1.0, 0.35, 1.0), 26.0)
    mat_red = _emissive("PAPI_Red", (1.0, 0.14, 0.08, 1.0), 26.0)

    # Threshold green bar + 16 ALSF-2 approach light stanchions extending 480 m into Bay
    for lat in (-26.0, -16.0, -6.0, 6.0, 16.0, 26.0):
        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=1,
            radius=0.9,
            location=(tx + lat * rx, ty + lat * ry, tz + 1.2),
        )
        bpy.context.active_object.data.materials.append(mat_green)

    for idx in range(1, 17):
        d_back = idx * 30.0
        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=1,
            radius=1.1,
            location=(tx - d_back * fx, ty - d_back * fy, tz + 1.6),
        )
        bpy.context.active_object.data.materials.append(mat_white)

    # 4-light PAPI left of touchdown zone (2 white, 2 red -> exact 3 deg on-glideslope!)
    for p_idx in range(4):
        lat = -(44.0 + p_idx * 9.0)
        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=1,
            radius=1.1,
            location=(tx + 300.0 * fx + lat * rx, ty + 300.0 * fy + lat * ry, tz + 1.5),
        )
        bpy.context.active_object.data.materials.append(mat_white if p_idx < 2 else mat_red)


def _create_pbr_twinjet_ownship() -> tuple[Any, list[Any]]:
    """Build multi-part smooth-shaded PBR civil twinjet ownship (+Y forward, +X right, +Z up)."""
    import bpy  # type: ignore[import-not-found]

    root = bpy.data.objects.new("Ownship_Twinjet_Root", None)
    root.rotation_mode = AIRCRAFT_ROTATION_MODE
    bpy.context.collection.objects.link(root)

    def _pbr(name: str, color: tuple[float, float, float, float], metallic: float, rough: float, coat: float = 0.0) -> Any:
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        b = m.node_tree.nodes.get("Principled BSDF")
        b.inputs["Base Color"].default_value = color
        b.inputs["Metallic"].default_value = metallic
        b.inputs["Roughness"].default_value = rough
        if coat > 0.0 and "Coat Weight" in b.inputs:
            b.inputs["Coat Weight"].default_value = coat
        return m

    mat_paint = _pbr("Ownship_WhiteMetallic", (0.86, 0.90, 0.95, 1.0), 0.35, 0.20, coat=1.0)
    mat_accent = _pbr("Ownship_CarbonSlate", (0.06, 0.11, 0.18, 1.0), 0.72, 0.28, coat=0.6)
    mat_glass = _pbr("Ownship_CanopyGlass", (0.04, 0.09, 0.15, 1.0), 0.10, 0.04, coat=1.0)
    mat_metal = _pbr("Ownship_Titanium", (0.68, 0.72, 0.78, 1.0), 0.92, 0.24)

    def _attach(obj: Any, mat: Any) -> Any:
        obj.parent = root
        obj.data.materials.append(mat)
        for p in obj.data.polygons:
            p.use_smooth = True
        return obj

    # Fuselage (smooth tapered ellipsoid, 13.5 m long)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1.0, location=(0.0, 0.0, 0.0))
    fuse = bpy.context.active_object
    fuse.name = "Ownship_Fuselage"
    fuse.scale = (0.95, 6.8, 0.92)
    _attach(fuse, mat_paint)

    # Cockpit glass canopy
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=1.0, location=(0.0, 3.2, 0.42))
    canopy = bpy.context.active_object
    canopy.name = "Ownship_Canopy"
    canopy.scale = (0.68, 1.65, 0.56)
    _attach(canopy, mat_glass)

    # Swept airfoil wings (left & right) + winglets
    for side in (-1.0, 1.0):
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(side * 3.8, -0.4, -0.18))
        wing = bpy.context.active_object
        wing.scale = (7.2, 1.65, 0.14)
        wing.rotation_euler = (0.0, side * -0.04, side * -0.22)
        _attach(wing, mat_paint)

        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(side * 7.25, -1.15, 0.35))
        wlet = bpy.context.active_object
        wlet.scale = (0.10, 0.95, 1.05)
        wlet.rotation_euler = (-0.25, side * 0.28, 0.0)
        _attach(wlet, mat_accent)

    # Vertical tail fin + T-tail horizontal stabilizer
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, -5.6, 1.35))
    vtail = bpy.context.active_object
    vtail.scale = (0.14, 2.0, 2.35)
    vtail.rotation_euler = (-0.32, 0.0, 0.0)
    _attach(vtail, mat_accent)

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, -6.15, 2.35))
    htail = bpy.context.active_object
    htail.scale = (4.8, 1.0, 0.10)
    _attach(htail, mat_paint)

    # Under-belly OpenXVS EO/LWIR Sensor Pod
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=10, radius=0.42, location=(0.0, 2.1, -0.95))
    pod = bpy.context.active_object
    pod.name = "OpenXVS_Sensor_Pod"
    pod.scale = (0.85, 1.55, 0.72)
    _attach(pod, mat_accent)

    # Twin nacelles + spinning turbine/propeller disks
    spinners: list[Any] = []
    for side in (-1.0, 1.0):
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=24, radius=0.42, depth=2.1, location=(side * 1.45, -3.5, 0.45), rotation=(math.pi / 2.0, 0.0, 0.0)
        )
        nac = bpy.context.active_object
        _attach(nac, mat_metal)

        bpy.ops.mesh.primitive_cylinder_add(
            vertices=16, radius=0.38, depth=0.06, location=(side * 1.45, -2.42, 0.45), rotation=(math.pi / 2.0, 0.0, 0.0)
        )
        fan = bpy.context.active_object
        _attach(fan, mat_accent)
        spinners.append(fan)

    return root, spinners


def _create_pbr_bell407_intruder() -> tuple[Any, Any]:
    """Build multi-part Bell-407 civil helicopter intruder with spinning main rotor."""
    import bpy  # type: ignore[import-not-found]

    root = bpy.data.objects.new("Intruder_Bell407_Root", None)
    root.rotation_mode = AIRCRAFT_ROTATION_MODE
    bpy.context.collection.objects.link(root)

    mat_body = bpy.data.materials.new("Heli_AmberPaint")
    mat_body.use_nodes = True
    b = mat_body.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (0.88, 0.24, 0.12, 1.0)
    b.inputs["Metallic"].default_value = 0.4
    b.inputs["Roughness"].default_value = 0.22

    mat_dark = bpy.data.materials.new("Heli_DarkCarbon")
    mat_dark.use_nodes = True
    bd = mat_dark.node_tree.nodes.get("Principled BSDF")
    bd.inputs["Base Color"].default_value = (0.04, 0.05, 0.07, 1.0)
    bd.inputs["Roughness"].default_value = 0.35

    def _attach(obj: Any, mat: Any) -> Any:
        obj.parent = root
        obj.data.materials.append(mat)
        for p in obj.data.polygons:
            p.use_smooth = True
        return obj

    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=1.0, location=(0.0, 0.6, 0.0))
    cab = bpy.context.active_object
    cab.scale = (1.05, 2.4, 1.05)
    _attach(cab, mat_body)

    bpy.ops.mesh.primitive_cylinder_add(
        vertices=16, radius=0.22, depth=5.8, location=(0.0, -3.5, 0.22), rotation=(math.pi / 2.0, 0.0, 0.0)
    )
    _attach(bpy.context.active_object, mat_body)

    # Spinning 4-blade main rotor hub (10.7 m diameter)
    rotor_hub = bpy.data.objects.new("Heli_MainRotor_Hub", None)
    rotor_hub.location = (0.0, 0.4, 1.45)
    rotor_hub.parent = root
    bpy.context.collection.objects.link(rotor_hub)
    for angle in (0.0, 0.5 * math.pi):
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
        blade = bpy.context.active_object
        blade.scale = (10.7, 0.32, 0.04)
        blade.rotation_euler = (0.0, 0.0, angle)
        blade.parent = rotor_hub
        blade.data.materials.append(mat_dark)

    return root, rotor_hub


def _create_glowing_manifold_material(name: str, rgba: tuple[float, float, float, float], strength: float) -> Any:
    import bpy  # type: ignore[import-not-found]

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    mix = nt.nodes.new("ShaderNodeMixShader")
    trans = nt.nodes.new("ShaderNodeBsdfTransparent")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = rgba
    em.inputs["Strength"].default_value = strength
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.35
    nt.links.new(lw.outputs["Fresnel"], mix.inputs["Fac"])
    nt.links.new(trans.outputs["BSDF"], mix.inputs[1])
    nt.links.new(em.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return mat


def _disable_hologram_shadows(obj: Any) -> None:
    """Smooth-shade holographic mesh and prevent Cycles shadow casting onto terrain/water."""
    if hasattr(obj, "visible_shadow"):
        obj.visible_shadow = False
    if hasattr(obj, "data") and hasattr(obj.data, "polygons"):
        for poly in obj.data.polygons:
            poly.use_smooth = True


def _build_3d_curve_ribbon(
    name: str,
    waypoints_ned: list[list[float]],
    rgba: tuple[float, float, float, float],
    bevel_m: float = 1.2,
    *,
    z_offset_m: float = 0.0,
) -> Any:
    import bpy  # type: ignore[import-not-found]

    curve_data = bpy.data.curves.new(name=name, type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.bevel_depth = bevel_m
    curve_data.bevel_resolution = 4
    poly = curve_data.splines.new("POLY")
    poly.points.add(len(waypoints_ned) - 1)
    for idx, pt in enumerate(waypoints_ned):
        bx, by, bz = ned_to_blender_xyz(pt[0], pt[1], pt[2])
        poly.points[idx].co = (bx, by, bz + float(z_offset_m), 1.0)
    obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(_create_glowing_manifold_material(f"{name}_Mat", rgba, 6.5))
    _disable_hologram_shadows(obj)
    return obj


def build_and_render_chapter(
    trajectories_path: Path,
    out_dir: Path,
    chapter_id: str,
    *,
    res_w: int = 1920,
    res_h: int = 1080,
    samples: int = 32,
    fps: int = 30,
    frame_start: int | None = None,
    frame_end: int | None = None,
    terrain_bin: Path = DEFAULT_TERRAIN_BIN,
    ortho_jpg: Path = DEFAULT_ORTHO_JPG,
    inner_ortho_jpg: Path | None = None,
    outer_ortho_jpg: Path | None = None,
    require_gpu: bool = False,
) -> None:
    """Build full Cycles GPU scene for one chapter and render PNG frames to `out_dir/chapter_id/`."""
    import bpy  # type: ignore[import-not-found]
    from mathutils import Vector  # type: ignore[import-not-found]

    doc = json.loads(Path(trajectories_path).read_text(encoding="utf-8"))
    if chapter_id not in doc["chapters"]:
        raise KeyError(f"Unknown chapter {chapter_id!r}; available: {list(doc['chapters'].keys())}")
    ch = doc["chapters"][chapter_id]
    frames = ch["frames"]
    total_f = len(frames)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    _configure_cycles_gpu(
        scene,
        res_w=res_w,
        res_h=res_h,
        samples=samples,
        fps=fps,
        require_gpu=bool(require_gpu),
    )
    _setup_compositor_bloom(scene)
    fog_density_socket = _setup_nishita_sky_and_clouds(scene, chapter_id=chapter_id)
    _build_ksfo_terrain_and_bay(
        terrain_bin=terrain_bin,
        ortho_jpg=ortho_jpg,
        inner_ortho_jpg=inner_ortho_jpg,
        outer_ortho_jpg=outer_ortho_jpg,
    )

    ownship, spinners = _create_pbr_twinjet_ownship()

    intruder = None
    rotor_hub = None
    ar_cone = None
    sc3bf_dome = None
    if chapter_id == "ch1_cbdr_activerange":
        intruder, rotor_hub = _create_pbr_bell407_intruder()
        # Glowing Fresnel ActiveRange collapsing range sphere/cone + smooth SC3BF safety dome (zero shadows)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=1.0, location=(0.0, 0.0, 0.0))
        ar_cone = bpy.context.active_object
        ar_cone.name = "ActiveRange_Lock_Manifold"
        ar_cone.data.materials.append(
            _create_glowing_manifold_material("ActiveRange_Emerald", (0.15, 0.96, 0.55, 1.0), 8.0)
        )
        _disable_hologram_shadows(ar_cone)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=152.4, location=(0.0, 0.0, 0.0))
        sc3bf_dome = bpy.context.active_object
        sc3bf_dome.name = "SC3BF_WellClear_Shield"
        sc3bf_dome.data.materials.append(
            _create_glowing_manifold_material("SC3BF_CyanShield", (0.12, 0.78, 1.0, 1.0), 4.5)
        )
        _disable_hologram_shadows(sc3bf_dome)

    if chapter_id == "ch2_gps_spoof_nav":
        true_pts = [
            [f["ownship_ned"]["north_m"], f["ownship_ned"]["east_m"], f["ownship_ned"]["down_m"]]
            for f in frames[::10]
        ]
        spoof_pts = [
            [f["spoofed_gps_ned"]["north_m"], f["spoofed_gps_ned"]["east_m"], f["spoofed_gps_ned"]["down_m"]]
            for f in frames[::10]
        ]
        _build_3d_curve_ribbon(
            "XVS_Nav_True_Path", true_pts, (0.12, 0.95, 0.52, 1.0), bevel_m=1.1, z_offset_m=-4.0
        )
        _build_3d_curve_ribbon(
            "Spoofed_GPS_Ghost", spoof_pts, (1.0, 0.16, 0.22, 1.0), bevel_m=1.0, z_offset_m=-4.0
        )

    if chapter_id == "ch4_forcedland_35kt_ksfo" and "corridor_waypoints_ned" in ch:
        _build_3d_curve_ribbon(
            "ForcedLand_Trochoid_Corridor",
            ch["corridor_waypoints_ned"],
            (0.98, 0.78, 0.18, 1.0),
            bevel_m=1.5,
            z_offset_m=-4.0,
        )

    # Cinematic Camera with smooth look-at orientation per frame
    cam_data = bpy.data.cameras.new("CinemaCam")
    cam_data.clip_start = 0.5
    cam_data.clip_end = 65000.0
    cam_obj = bpy.data.objects.new("CinemaCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # Unwrap heading sequences before keyframing to eliminate +/-2*pi motion-blur spin smears
    own_headings = unwrap_headings_rad([float(f["ownship_ned"]["heading_rad"]) for f in frames])
    intr_indices = [i for i, f in enumerate(frames) if "intruder_ned" in f]
    intr_headings_map: dict[int, float] = {}
    if intr_indices:
        intr_unwrapped = unwrap_headings_rad(
            [float(frames[i]["intruder_ned"]["heading_rad"]) for i in intr_indices]
        )
        intr_headings_map = dict(zip(intr_indices, intr_unwrapped))

    spin_step_rad = spinner_angular_step_rad(chapter_id)
    prev_cam_euler = None

    # Keyframe all frames
    for idx, fdata in enumerate(frames):
        fnum = idx + 1
        own = fdata["ownship_ned"]
        ownship.location = ned_to_blender_xyz(own["north_m"], own["east_m"], own["down_m"])
        ownship.rotation_euler = ned_euler_to_blender_xyz_rad(
            own_headings[idx], own["pitch_rad"], own["roll_rad"]
        )
        ownship.keyframe_insert(data_path="location", frame=fnum)
        ownship.keyframe_insert(data_path="rotation_euler", frame=fnum)

        for sp in spinners:
            sp.rotation_euler = (math.pi / 2.0, float(idx) * spin_step_rad, 0.0)
            sp.keyframe_insert(data_path="rotation_euler", frame=fnum)

        if intruder is not None and "intruder_ned" in fdata:
            intr = fdata["intruder_ned"]
            ix, iy, iz = ned_to_blender_xyz(intr["north_m"], intr["east_m"], intr["down_m"])
            intruder.location = (ix, iy, iz)
            intruder.rotation_euler = ned_euler_to_blender_xyz_rad(
                intr_headings_map[idx], intr["pitch_rad"], intr["roll_rad"]
            )
            intruder.keyframe_insert(data_path="location", frame=fnum)
            intruder.keyframe_insert(data_path="rotation_euler", frame=fnum)
            if rotor_hub is not None:
                rotor_hub.rotation_euler = (0.0, 0.0, float(intr["rotor_phase_rad"]))
                rotor_hub.keyframe_insert(data_path="rotation_euler", frame=fnum)
            if ar_cone is not None:
                sig_vis = max(16.0, min(180.0, float(fdata["telemetry"]["sigma_r_m"]) * 0.22))
                ar_cone.location = (ix, iy, iz)
                ar_cone.scale = (sig_vis, sig_vis, sig_vis)
                ar_cone.keyframe_insert(data_path="location", frame=fnum)
                ar_cone.keyframe_insert(data_path="scale", frame=fnum)
            if sc3bf_dome is not None:
                sc3bf_dome.location = (ix, iy, iz)
                sc3bf_dome.keyframe_insert(data_path="location", frame=fnum)

        if fog_density_socket is not None and "fog_density" in fdata["telemetry"]:
            fog_density_socket.default_value = 0.0005 + 0.0045 * float(fdata["telemetry"]["fog_density"])
            fog_density_socket.keyframe_insert(data_path="default_value", frame=fnum)

        t_norm = float(idx) / float(max(1, total_f - 1))
        cpose = compute_cinematic_camera_pose(chapter_id, fdata, t_norm)
        cam_obj.location = cpose["location"]
        direction = Vector(cpose["look_at"]) - Vector(cpose["location"])
        cam_quat = direction.to_track_quat("-Z", "Y")
        if prev_cam_euler is not None:
            cam_euler = cam_quat.to_euler("XYZ", prev_cam_euler)
        else:
            cam_euler = cam_quat.to_euler("XYZ")
        prev_cam_euler = cam_euler.copy()
        cam_obj.rotation_euler = cam_euler
        cam_data.lens = float(cpose["lens_mm"])
        cam_obj.keyframe_insert(data_path="location", frame=fnum)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=fnum)
        cam_data.keyframe_insert(data_path="lens", frame=fnum)

    s_f = max(1, int(frame_start)) if frame_start is not None else 1
    e_f = min(total_f, int(frame_end)) if frame_end is not None else total_f
    scene.frame_start = s_f
    scene.frame_end = e_f

    ch_out = Path(out_dir) / chapter_id
    ch_out.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(ch_out / "frame_")
    bpy.ops.render.render(animation=True)


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw = sys.argv[1:] if argv is None else list(argv)
    if "--" in raw:
        raw = raw[raw.index("--") + 1 :]
    p = argparse.ArgumentParser(description="Render OpenXVS 4K Cycles GPU cinema shots in Blender 4.x")
    p.add_argument("--trajectories", type=Path, default=DEFAULT_TRAJECTORIES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--chapter", type=str, default="ch1_cbdr_activerange")
    p.add_argument("--res", type=str, default="3840x2160")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--frame-start", type=int, default=None)
    p.add_argument("--frame-end", type=int, default=None)
    p.add_argument("--terrain-bin", type=Path, default=DEFAULT_TERRAIN_BIN)
    p.add_argument("--ortho-jpg", type=Path, default=DEFAULT_ORTHO_JPG)
    p.add_argument("--inner-ortho-jpg", type=Path, default=None)
    p.add_argument("--outer-ortho-jpg", type=Path, default=None)
    p.add_argument("--require-gpu", action="store_true", default=False)
    return p.parse_args(raw)


def main(argv: list[str] | None = None) -> int:
    args = parse_cli_args(argv)
    res_w, res_h = parse_resolution(args.res)
    build_and_render_chapter(
        trajectories_path=args.trajectories,
        out_dir=args.out_dir,
        chapter_id=args.chapter,
        res_w=res_w,
        res_h=res_h,
        samples=int(args.samples),
        fps=int(args.fps),
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        terrain_bin=args.terrain_bin,
        ortho_jpg=args.ortho_jpg,
        inner_ortho_jpg=args.inner_ortho_jpg,
        outer_ortho_jpg=args.outer_ortho_jpg,
        require_gpu=bool(args.require_gpu),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
