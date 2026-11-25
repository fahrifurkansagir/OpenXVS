"""Thin Python 3.14 ctypes wrapper (`XvsCoreFFI`) loading `xvs_core.dll` / `libxvs_core.so`.

Provides zero-PyO3, ABI-strict access to `xvs-rta-core` and `xvs-math-core` for cross-language
parity (`V2-M1-3`) and headless mission cross-checks (`V2-M1-6`).
"""

from __future__ import annotations

import ctypes
from ctypes import (
    POINTER,
    Structure,
    c_double,
    c_int16,
    c_int32,
    c_int64,
    c_size_t,
    c_uint8,
    c_uint16,
    c_uint32,
    c_uint64,
    c_void_p,
)
from pathlib import Path
import sys
from typing import Sequence


class XvsRtaFloatTelemetry(Structure):
    _fields_ = [
        ("horizon_pl_rad", c_double),
        ("horizon_al_rad", c_double),
        ("v_vpl_m", c_double),
        ("v_val_m", c_double),
        ("v_hpl_m", c_double),
        ("v_hal_m", c_double),
        ("epipolar_max_rad", c_double * 6),
        ("signed_horizon_residual_rad", c_double),
        ("runway_width_delta_m", c_double),
        ("runway_width_delta_limit_m", c_double),
        ("cusum_stat", c_double),
        ("cusum_thr", c_double),
        ("ood_global_score", c_double),
        ("ood_global_thr", c_double),
        ("ood_cam_scores", c_double * 6),
        ("ood_cam_thr", c_double),
        ("cam_health_mask", c_uint8),
        ("ml_detector_fault", c_uint8),
        ("transient_dropout", c_uint8),
        ("approach_active", c_uint8),
        ("go_around_ack_reset", c_uint8),
    ]


class XvsRtaTickInput(Structure):
    _fields_ = [
        ("cam_health_mask", c_uint8),
        ("ood_exceeded_mask", c_uint8),
        ("horizon_pl_urad", c_int64),
        ("horizon_al_urad", c_int64),
        ("v_vpl_mm", c_int64),
        ("v_val_mm", c_int64),
        ("v_hpl_mm", c_int64),
        ("v_hal_mm", c_int64),
        ("epipolar_max_urad", c_int64 * 6),
        ("signed_horizon_residual_urad", c_int64),
        ("runway_width_delta_mm", c_int64),
        ("spoof_alarm", c_uint8),
        ("ml_detector_fault", c_uint8),
        ("ood_global_exceeded", c_uint8),
        ("runway_width_fault", c_uint8),
        ("transient_dropout", c_uint8),
        ("approach_active", c_uint8),
        ("go_around_ack_reset", c_uint8),
        ("quantize_fault", c_uint8),
    ]


class XvsRtaTickOutput(Structure):
    _fields_ = [
        ("mode", c_uint8),
        ("raw_target", c_uint8),
        ("healthy_cam_mask", c_uint8),
        ("active_healthy_cams", c_uint8),
        ("consec_dropout_ticks", c_uint32),
        ("window_dropout_20", c_uint8),
        ("recovery_ticks", c_uint32),
        ("dwell_ticks", c_uint32),
        ("mandatory_go_around", c_uint8),
        ("mltl_critical_violation", c_uint8),
        ("blank_conformal_hud", c_uint8),
        ("inhibit_avoidance_turns", c_uint8),
        ("tick_index", c_uint64),
    ]


class XvsFaultPerturbations(Structure):
    _fields_ = [
        ("cam_r_bias_rad", c_double),
        ("gnss_spoof_vel_ned_mps", c_double * 3),
        ("runway_corner_bias_px", c_double),
        ("runway_observed_width_m", c_double),
        ("fog_contrast", c_double),
        ("ood_global_score", c_double),
        ("manual_bank_cmd_rad", c_double),
        ("manual_climb_cmd_mps", c_double),
        ("lwir_efvs_enabled", c_uint8),
        ("transient_dropout", c_uint8),
        ("ml_detector_fault", c_uint8),
        ("go_around_ack_reset", c_uint8),
        ("peer3_active", c_uint8),
        ("manual_override", c_uint8),
        ("foreflight_sim_prefix", c_uint8),
    ]

    @classmethod
    def nominal(cls) -> XvsFaultPerturbations:
        return cls(
            cam_r_bias_rad=0.0,
            gnss_spoof_vel_ned_mps=(c_double * 3)(0.0, 0.0, 0.0),
            runway_corner_bias_px=0.0,
            runway_observed_width_m=61.0,
            fog_contrast=1.0,
            ood_global_score=0.15,
            manual_bank_cmd_rad=0.0,
            manual_climb_cmd_mps=0.0,
            lwir_efvs_enabled=0,
            transient_dropout=0,
            ml_detector_fault=0,
            go_around_ack_reset=0,
            peer3_active=0,
            manual_override=0,
            foreflight_sim_prefix=1,
        )


class XvsTelemetrySnapshot(Structure):
    _fields_ = [
        ("time_s", c_double),
        ("tick_count", c_uint64),
        ("lat_deg", c_double),
        ("lon_deg", c_double),
        ("h_m", c_double),
        ("p_ecef", c_double * 3),
        ("v_ned", c_double * 3),
        ("yaw_rad", c_double),
        ("pitch_rad", c_double),
        ("roll_rad", c_double),
        ("q_ned_body", c_double * 4),
        ("tas_mps", c_double),
        ("epipolar_max_urad", c_double * 6),
        ("fused_attitude_err_urad", c_double),
        ("horizon_pl_urad", c_double),
        ("v_vpl_mm", c_double),
        ("v_hpl_mm", c_double),
        ("cusum_stat", c_double),
        ("nis_stat", c_double),
        ("ss_stat", c_double),
        ("bank_vel_delta_mps", c_double),
        ("ecbf_bank_cmd_rad", c_double),
        ("ecbf_climb_cmd_mps", c_double),
        ("ecbf_slack", c_double),
        ("dtaci_cone_urad", c_double),
        ("intruder_rel_ned", (c_double * 3) * 8),
        ("intruder_vel_ned", (c_double * 3) * 8),
        ("intruder_count", c_uint32),
        ("ghosts_rejected", c_uint32),
        ("mesh_fused_count", c_uint32),
        ("mission_id", c_uint8),
        ("simplex_mode", c_uint8),
        ("raw_target", c_uint8),
        ("healthy_cam_mask", c_uint8),
        ("active_healthy_cams", c_uint8),
        ("active_nav_bank", c_uint8),
        ("mandatory_go_around", c_uint8),
        ("blank_conformal_hud", c_uint8),
        ("inhibit_avoidance_turns", c_uint8),
        ("ecbf_infeasible", c_uint8),
        ("wc1_alert", c_uint8),
        ("dtaci_frozen", c_uint8),
        ("approach_active", c_uint8),
        ("lwir_efvs_enabled", c_uint8),
        ("_pad", c_uint8 * 2),
    ]


class XvsSpoofEvalResult(Structure):
    _fields_ = [
        ("nis_stat", c_double),
        ("cusum_stat", c_double),
        ("ss_stat", c_double),
        ("maneuver_guard_scale", c_double),
        ("spoof_alarm", c_uint8),
        ("active_bank", c_uint8),
    ]


class XvsVraimEvalResult(Structure):
    _fields_ = [
        ("epipolar_max_rad", c_double * 6),
        ("isolated_cam_idx", c_int32),
        ("healthy_cam_mask", c_uint8),
        ("fused_attitude_err_rad", c_double),
        ("horizon_pl_rad", c_double),
        ("v_vpl_m", c_double),
        ("v_hpl_m", c_double),
        ("max_slope_v", c_double),
        ("max_slope_h", c_double),
        ("parity_stat", c_double),
        ("width_delta_m", c_double),
        ("width_fault", c_uint8),
    ]


class XvsBearingTrack(Structure):
    _fields_ = [
        ("time_utc_us", c_uint64),
        ("own_ecef_cm", c_int32 * 3),
        ("bearing_ecef_q15", c_int16 * 3),
        ("bearing_rate_urad_s", c_int16 * 3),
        ("cone_half_angle_urad", c_uint16),
        ("own_pos_std_cm", c_uint16),
        ("track_id", c_uint16),
        ("sender_id", c_uint8),
        ("flags", c_uint8),
        ("cov_diag_packed", c_uint16 * 6),
    ]


class XvsMeshDaaEvalResult(Structure):
    _fields_ = [
        ("num_fused", c_uint32),
        ("ghosts_rejected", c_uint32),
        ("fused_pos_ecef", (c_double * 3) * 8),
        ("triple_ray_confirmed", c_uint8 * 8),
    ]


class XvsIntruderNed(Structure):
    _fields_ = [
        ("pos_ned", c_double * 3),
        ("vel_ned", c_double * 3),
        ("valid", c_uint8),
    ]


class XvsEcbfQpResult(Structure):
    _fields_ = [
        ("bank_cmd_rad", c_double),
        ("climb_rate_cmd_mps", c_double),
        ("slack", c_double),
        ("iterations", c_uint8),
        ("ecbf_infeasible", c_uint8),
        ("wc1_alert", c_uint8),
        ("min_ttc_s", c_double),
    ]


class XvsGdl90Target(Structure):
    _fields_ = [
        ("alert_status", c_uint8),
        ("address_type", c_uint8),
        ("participant_address", c_uint32),
        ("lat_deg", c_double),
        ("lon_deg", c_double),
        ("altitude_ft", c_double),
        ("altitude_valid", c_uint8),
        ("misc_indicators", c_uint8),
        ("nic", c_uint8),
        ("nacp", c_uint8),
        ("horiz_vel_kt", c_double),
        ("horiz_vel_valid", c_uint8),
        ("vert_vel_fpm", c_double),
        ("vert_vel_valid", c_uint8),
        ("track_heading_deg", c_double),
        ("emitter_category", c_uint8),
        ("callsign", c_uint8 * 8),
        ("emergency_code", c_uint8),
    ]

    @classmethod
    def from_values(
        cls,
        *,
        alert_status: int,
        address_type: int,
        participant_address: int,
        lat_deg: float,
        lon_deg: float,
        altitude_ft: float | None,
        misc_indicators: int,
        nic: int,
        nacp: int,
        horiz_vel_kt: float | None,
        vert_vel_fpm: float | None,
        track_heading_deg: float,
        emitter_category: int,
        callsign: str,
        emergency_code: int = 0,
    ) -> XvsGdl90Target:
        cs_bytes = callsign.encode("ascii", errors="ignore")[:8].ljust(8, b" ")
        return cls(
            alert_status=alert_status,
            address_type=address_type,
            participant_address=participant_address,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            altitude_ft=0.0 if altitude_ft is None else float(altitude_ft),
            altitude_valid=0 if altitude_ft is None else 1,
            misc_indicators=misc_indicators,
            nic=nic,
            nacp=nacp,
            horiz_vel_kt=0.0 if horiz_vel_kt is None else float(horiz_vel_kt),
            horiz_vel_valid=0 if horiz_vel_kt is None else 1,
            vert_vel_fpm=0.0 if vert_vel_fpm is None else float(vert_vel_fpm),
            vert_vel_valid=0 if vert_vel_fpm is None else 1,
            track_heading_deg=track_heading_deg,
            emitter_category=emitter_category,
            callsign=(c_uint8 * 8)(*cs_bytes),
            emergency_code=emergency_code,
        )


class XvsMavlinkAdsb246(Structure):
    _fields_ = [
        ("icao_address", c_uint32),
        ("lat_deg_e7", c_int32),
        ("lon_deg_e7", c_int32),
        ("altitude_mm", c_int32),
        ("heading_cdeg", c_uint16),
        ("hor_velocity_cms", c_uint16),
        ("ver_velocity_cms", c_int16),
        ("flags", c_uint16),
        ("squawk", c_uint16),
        ("altitude_type", c_uint8),
        ("callsign", c_uint8 * 9),
        ("emitter_type", c_uint8),
        ("tslc", c_uint8),
        ("seq", c_uint8),
        ("sysid", c_uint8),
        ("compid", c_uint8),
    ]

    @classmethod
    def from_values(
        cls,
        *,
        icao_address: int,
        lat_deg_e7: int,
        lon_deg_e7: int,
        altitude_mm: int,
        heading_cdeg: int,
        hor_velocity_cms: int,
        ver_velocity_cms: int,
        flags: int,
        squawk: int,
        altitude_type: int,
        callsign: str,
        emitter_type: int,
        tslc: int,
        seq: int = 0,
        sysid: int = 1,
        compid: int = 156,
    ) -> XvsMavlinkAdsb246:
        cs_bytes = callsign.encode("ascii", errors="ignore")[:8].ljust(9, b"\x00")
        return cls(
            icao_address=icao_address,
            lat_deg_e7=lat_deg_e7,
            lon_deg_e7=lon_deg_e7,
            altitude_mm=altitude_mm,
            heading_cdeg=heading_cdeg,
            hor_velocity_cms=hor_velocity_cms,
            ver_velocity_cms=ver_velocity_cms,
            flags=flags,
            squawk=squawk,
            altitude_type=altitude_type,
            callsign=(c_uint8 * 9)(*cs_bytes),
            emitter_type=emitter_type,
            tslc=tslc,
            seq=seq,
            sysid=sysid,
            compid=compid,
        )


def default_shared_lib_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    target_dir = repo_root / "xvs-core" / "target" / "release"
    if sys.platform == "win32":
        return target_dir / "xvs_core.dll"
    if sys.platform == "darwin":
        return target_dir / "libxvs_core.dylib"
    return target_dir / "libxvs_core.so"


class XvsCoreFFI:
    """Thin ctypes interface to the compiled `xvs-ffi` shared library."""

    def __init__(self, lib_path: Path | str | None = None) -> None:
        path = Path(lib_path) if lib_path is not None else default_shared_lib_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"Compiled xvs-ffi library not found at {path}; "
                "build via `cargo build --manifest-path xvs-core/Cargo.toml -p xvs-ffi --release`"
            )
        self.lib_path = path
        self._lib = ctypes.CDLL(str(path))
        self._bind_signatures()

    def _bind_signatures(self) -> None:
        lib = self._lib

        lib.xvs_rta_create.argtypes = []
        lib.xvs_rta_create.restype = c_void_p

        lib.xvs_rta_destroy.argtypes = [c_void_p]
        lib.xvs_rta_destroy.restype = None

        lib.xvs_rta_build_input_from_f64.argtypes = [
            POINTER(XvsRtaFloatTelemetry),
            POINTER(XvsRtaTickInput),
        ]
        lib.xvs_rta_build_input_from_f64.restype = c_int32

        lib.xvs_rta_step.argtypes = [
            c_void_p,
            POINTER(XvsRtaTickInput),
            POINTER(XvsRtaTickOutput),
        ]
        lib.xvs_rta_step.restype = c_int32

        lib.xvs_sim_create.argtypes = [c_uint64]
        lib.xvs_sim_create.restype = c_void_p

        lib.xvs_sim_destroy.argtypes = [c_void_p]
        lib.xvs_sim_destroy.restype = None

        lib.xvs_sim_reset_mission.argtypes = [c_void_p, c_uint8, c_uint64]
        lib.xvs_sim_reset_mission.restype = c_int32

        lib.xvs_sim_set_faults.argtypes = [c_void_p, POINTER(XvsFaultPerturbations)]
        lib.xvs_sim_set_faults.restype = c_int32

        lib.xvs_sim_step.argtypes = [c_void_p, c_uint32]
        lib.xvs_sim_step.restype = c_int32

        lib.xvs_sim_step_100hz.argtypes = [c_void_p, c_uint32]
        lib.xvs_sim_step_100hz.restype = c_int32

        lib.xvs_sim_snapshot.argtypes = [c_void_p, POINTER(XvsTelemetrySnapshot)]
        lib.xvs_sim_snapshot.restype = c_int32

        lib.xvs_sim_drain_tx_packets.argtypes = [
            c_void_p,
            POINTER(c_uint8),
            c_size_t,
            POINTER(c_uint8),
        ]
        lib.xvs_sim_drain_tx_packets.restype = c_int32

        lib.xvs_geo_geodetic_to_ecef.argtypes = [c_double, c_double, c_double, POINTER(c_double)]
        lib.xvs_geo_geodetic_to_ecef.restype = c_int32

        lib.xvs_geo_ecef_to_geodetic.argtypes = [
            c_double,
            c_double,
            c_double,
            POINTER(c_double),
            POINTER(c_double),
            POINTER(c_double),
        ]
        lib.xvs_geo_ecef_to_geodetic.restype = c_int32

        lib.xvs_geo_r_ecef_ned.argtypes = [c_double, c_double, POINTER(c_double)]
        lib.xvs_geo_r_ecef_ned.restype = c_int32

        lib.xvs_geo_euler_zyx_to_quat.argtypes = [c_double, c_double, c_double, POINTER(c_double)]
        lib.xvs_geo_euler_zyx_to_quat.restype = c_int32

        lib.xvs_rig_project_ecef_to_cam.argtypes = [
            c_uint8,
            POINTER(c_double),
            POINTER(c_double),
            POINTER(c_double),
            POINTER(c_double),
            POINTER(c_double),
        ]
        lib.xvs_rig_project_ecef_to_cam.restype = c_int32

        lib.xvs_es_ekf_step_cusum_eval.argtypes = [
            POINTER(c_double),
            POINTER(c_double),
            c_size_t,
            c_double,
            POINTER(c_double),
            POINTER(c_double),
            POINTER(c_double),
            POINTER(c_double),
            POINTER(XvsSpoofEvalResult),
        ]
        lib.xvs_es_ekf_step_cusum_eval.restype = c_int32

        lib.xvs_vraim_eval.argtypes = [
            POINTER(c_double),
            c_uint8,
            c_double,
            c_double,
            c_double,
            POINTER(XvsVraimEvalResult),
        ]
        lib.xvs_vraim_eval.restype = c_int32

        lib.xvs_mesh_daa_eval.argtypes = [
            POINTER(XvsBearingTrack),
            c_size_t,
            POINTER(XvsMeshDaaEvalResult),
        ]
        lib.xvs_mesh_daa_eval.restype = c_int32

        lib.xvs_fci_fuse_3d.argtypes = [
            POINTER(c_double),
            POINTER(c_double),
            POINTER(c_double),
            POINTER(c_double),
            POINTER(c_double),
            POINTER(c_double),
        ]
        lib.xvs_fci_fuse_3d.restype = c_int32

        lib.xvs_ecbf_qp_solve.argtypes = [
            POINTER(c_double),
            POINTER(c_double),
            c_double,
            c_double,
            POINTER(XvsIntruderNed),
            c_size_t,
            POINTER(XvsEcbfQpResult),
        ]
        lib.xvs_ecbf_qp_solve.restype = c_int32

        lib.xvs_encode_gdl90_heartbeat.argtypes = [
            c_uint32,
            c_uint8,
            c_uint8,
            c_uint8,
            c_uint8,
            c_uint16,
            POINTER(c_uint8),
            c_size_t,
        ]
        lib.xvs_encode_gdl90_heartbeat.restype = c_int32

        lib.xvs_encode_gdl90_ownship_geo_alt.argtypes = [
            c_double,
            c_double,
            c_uint8,
            POINTER(c_uint8),
            c_size_t,
        ]
        lib.xvs_encode_gdl90_ownship_geo_alt.restype = c_int32

        lib.xvs_encode_gdl90_ownship.argtypes = [
            POINTER(XvsGdl90Target),
            POINTER(c_uint8),
            c_size_t,
        ]
        lib.xvs_encode_gdl90_ownship.restype = c_int32

        lib.xvs_encode_gdl90_traffic.argtypes = [
            POINTER(XvsGdl90Target),
            c_uint8,
            POINTER(c_uint8),
            c_size_t,
        ]
        lib.xvs_encode_gdl90_traffic.restype = c_int32

        lib.xvs_encode_mavlink_adsb_246.argtypes = [
            POINTER(XvsMavlinkAdsb246),
            POINTER(c_uint8),
            c_size_t,
        ]
        lib.xvs_encode_mavlink_adsb_246.restype = c_int32

        lib.xvs_encode_mavlink_bearing_track.argtypes = [
            POINTER(XvsBearingTrack),
            c_uint8,
            c_uint8,
            c_uint8,
            c_uint8,
            c_uint64,
            POINTER(c_uint8),
            POINTER(c_uint8),
            c_size_t,
        ]
        lib.xvs_encode_mavlink_bearing_track.restype = c_int32

    # --- Simplex RTA ---
    def rta_create(self) -> int:
        ptr = self._lib.xvs_rta_create()
        if not ptr:
            raise RuntimeError("xvs_rta_create returned NULL")
        return int(ptr)

    def rta_destroy(self, handle: int) -> None:
        self._lib.xvs_rta_destroy(c_void_p(handle))

    def rta_build_input_from_f64(self, telemetry: XvsRtaFloatTelemetry) -> XvsRtaTickInput:
        out = XvsRtaTickInput()
        rc = self._lib.xvs_rta_build_input_from_f64(ctypes.byref(telemetry), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(f"xvs_rta_build_input_from_f64 failed rc={rc}")
        return out

    def rta_step(self, handle: int, tick_input: XvsRtaTickInput) -> XvsRtaTickOutput:
        out = XvsRtaTickOutput()
        rc = self._lib.xvs_rta_step(c_void_p(handle), ctypes.byref(tick_input), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(f"xvs_rta_step failed rc={rc}")
        return out

    # --- LiveDeckEngine ---
    def sim_create(self, seed: int) -> int:
        ptr = self._lib.xvs_sim_create(c_uint64(seed))
        if not ptr:
            raise RuntimeError("xvs_sim_create returned NULL")
        return int(ptr)

    def sim_destroy(self, handle: int) -> None:
        self._lib.xvs_sim_destroy(c_void_p(handle))

    def sim_reset_mission(self, handle: int, mission_id: int, seed: int) -> None:
        rc = self._lib.xvs_sim_reset_mission(c_void_p(handle), c_uint8(mission_id), c_uint64(seed))
        if rc != 0:
            raise RuntimeError(f"xvs_sim_reset_mission failed rc={rc}")

    def sim_set_faults(self, handle: int, faults: XvsFaultPerturbations) -> None:
        rc = self._lib.xvs_sim_set_faults(c_void_p(handle), ctypes.byref(faults))
        if rc != 0:
            raise RuntimeError(f"xvs_sim_set_faults failed rc={rc}")

    def sim_step(self, handle: int, rta_ticks: int = 1) -> None:
        rc = self._lib.xvs_sim_step(c_void_p(handle), c_uint32(rta_ticks))
        if rc != 0:
            raise RuntimeError(f"xvs_sim_step failed rc={rc}")

    def sim_step_100hz(self, handle: int, substeps: int = 1) -> None:
        rc = self._lib.xvs_sim_step_100hz(c_void_p(handle), c_uint32(substeps))
        if rc != 0:
            raise RuntimeError(f"xvs_sim_step_100hz failed rc={rc}")

    def sim_snapshot(self, handle: int) -> XvsTelemetrySnapshot:
        out = XvsTelemetrySnapshot()
        rc = self._lib.xvs_sim_snapshot(c_void_p(handle), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(f"xvs_sim_snapshot failed rc={rc}")
        return out

    def sim_drain_all_tx_packets(self, handle: int) -> list[tuple[int, bytes]]:
        pkts: list[tuple[int, bytes]] = []
        buf = (c_uint8 * 128)()
        proto = c_uint8(0)
        while True:
            n = self._lib.xvs_sim_drain_tx_packets(
                c_void_p(handle), buf, 128, ctypes.byref(proto)
            )
            if n <= 0:
                break
            pkts.append((int(proto.value), bytes(buf[:n])))
        return pkts

    # --- Geo & Rig ---
    def geodetic_to_ecef(self, lat_deg: float, lon_deg: float, h_m: float) -> tuple[float, float, float]:
        out = (c_double * 3)()
        self._lib.xvs_geo_geodetic_to_ecef(lat_deg, lon_deg, h_m, out)
        return (float(out[0]), float(out[1]), float(out[2]))

    def ecef_to_geodetic(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        la, lo, hm = c_double(), c_double(), c_double()
        self._lib.xvs_geo_ecef_to_geodetic(x, y, z, ctypes.byref(la), ctypes.byref(lo), ctypes.byref(hm))
        return (float(la.value), float(lo.value), float(hm.value))

    def r_ecef_ned(self, lat_deg: float, lon_deg: float) -> tuple[tuple[float, ...], ...]:
        out = (c_double * 9)()
        self._lib.xvs_geo_r_ecef_ned(lat_deg, lon_deg, out)
        return (
            (float(out[0]), float(out[1]), float(out[2])),
            (float(out[3]), float(out[4]), float(out[5])),
            (float(out[6]), float(out[7]), float(out[8])),
        )

    def euler_zyx_to_quat(
        self, yaw_rad: float, pitch_rad: float, roll_rad: float
    ) -> tuple[float, float, float, float]:
        out = (c_double * 4)()
        self._lib.xvs_geo_euler_zyx_to_quat(yaw_rad, pitch_rad, roll_rad, out)
        return (float(out[0]), float(out[1]), float(out[2]), float(out[3]))

    def project_ecef_to_cam(
        self,
        rig_kind: int,
        p_ecef: Sequence[float],
        p_cam: Sequence[float],
        r_ecef_cam_flat9: Sequence[float],
    ) -> tuple[float, float] | None:
        pe = (c_double * 3)(*p_ecef)
        pc = (c_double * 3)(*p_cam)
        rm = (c_double * 9)(*r_ecef_cam_flat9)
        u, v = c_double(), c_double()
        rc = self._lib.xvs_rig_project_ecef_to_cam(
            c_uint8(rig_kind), pe, pc, rm, ctypes.byref(u), ctypes.byref(v)
        )
        if rc == 1:
            return (float(u.value), float(v.value))
        return None

    # --- Math Parity Submodules ---
    def es_ekf_step_cusum_eval(
        self,
        p_init: Sequence[float],
        v_init: Sequence[float],
        dt: float,
        accel_ned_steps: Sequence[Sequence[float]],
        gnss_vel_steps: Sequence[Sequence[float]],
        bank_rad_steps: Sequence[float],
        omega_steps: Sequence[float],
    ) -> XvsSpoofEvalResult:
        n = len(accel_ned_steps)
        pi = (c_double * 3)(*p_init)
        vi = (c_double * 3)(*v_init)
        accs = (c_double * (3 * n))(*[v for row in accel_ned_steps for v in row])
        gvs = (c_double * (3 * n))(*[v for row in gnss_vel_steps for v in row])
        banks = (c_double * n)(*bank_rad_steps)
        omegas = (c_double * n)(*omega_steps)
        out = XvsSpoofEvalResult()
        rc = self._lib.xvs_es_ekf_step_cusum_eval(
            pi, vi, c_size_t(n), dt, accs, gvs, banks, omegas, ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError(f"xvs_es_ekf_step_cusum_eval failed rc={rc}")
        return out

    def vraim_eval(
        self,
        cam_biases_rad: Sequence[float],
        cam_mask: int,
        corner_bias_px: float,
        obs_width_m: float,
        slant_range_m: float,
    ) -> XvsVraimEvalResult:
        cb = (c_double * 6)(*cam_biases_rad)
        out = XvsVraimEvalResult()
        rc = self._lib.xvs_vraim_eval(
            cb, c_uint8(cam_mask), corner_bias_px, obs_width_m, slant_range_m, ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError(f"xvs_vraim_eval failed rc={rc}")
        return out

    def mesh_daa_eval(self, tracks: Sequence[XvsBearingTrack]) -> XvsMeshDaaEvalResult:
        n = len(tracks)
        arr = (XvsBearingTrack * n)(*tracks)
        out = XvsMeshDaaEvalResult()
        rc = self._lib.xvs_mesh_daa_eval(arr, c_size_t(n), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(f"xvs_mesh_daa_eval failed rc={rc}")
        return out

    def fci_fuse_3d(
        self,
        xa: Sequence[float],
        pa_flat9: Sequence[float],
        xb: Sequence[float],
        pb_flat9: Sequence[float],
    ) -> tuple[tuple[float, float, float], tuple[tuple[float, ...], ...]]:
        xa_c = (c_double * 3)(*xa)
        pa_c = (c_double * 9)(*pa_flat9)
        xb_c = (c_double * 3)(*xb)
        pb_c = (c_double * 9)(*pb_flat9)
        ox = (c_double * 3)()
        op = (c_double * 9)()
        rc = self._lib.xvs_fci_fuse_3d(xa_c, pa_c, xb_c, pb_c, ox, op)
        if rc != 0:
            raise RuntimeError(f"xvs_fci_fuse_3d failed rc={rc}")
        x_f = (float(ox[0]), float(ox[1]), float(ox[2]))
        p_f = (
            (float(op[0]), float(op[1]), float(op[2])),
            (float(op[3]), float(op[4]), float(op[5])),
            (float(op[6]), float(op[7]), float(op[8])),
        )
        return x_f, p_f

    def ecbf_qp_solve(
        self,
        p_own: Sequence[float],
        v_own: Sequence[float],
        nom_bank_rad: float,
        nom_climb_mps: float,
        intruders: Sequence[XvsIntruderNed],
    ) -> XvsEcbfQpResult:
        po = (c_double * 3)(*p_own)
        vo = (c_double * 3)(*v_own)
        n = len(intruders)
        arr = (XvsIntruderNed * max(1, n))(*intruders) if n > 0 else (XvsIntruderNed * 1)()
        out = XvsEcbfQpResult()
        rc = self._lib.xvs_ecbf_qp_solve(
            po, vo, nom_bank_rad, nom_climb_mps, arr, c_size_t(n), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError(f"xvs_ecbf_qp_solve failed rc={rc}")
        return out

    # --- Avionics Codecs ---
    def encode_gdl90_heartbeat(
        self,
        *,
        utc_time_s: int = 0,
        gps_pos_valid: bool = True,
        maint_reqd: bool = False,
        utc_ok: bool = True,
        uplink_count: int = 0,
        basic_and_long_count: int = 0,
    ) -> bytes:
        buf = (c_uint8 * 64)()
        n = self._lib.xvs_encode_gdl90_heartbeat(
            c_uint32(utc_time_s),
            c_uint8(1 if gps_pos_valid else 0),
            c_uint8(1 if maint_reqd else 0),
            c_uint8(1 if utc_ok else 0),
            c_uint8(uplink_count),
            c_uint16(basic_and_long_count),
            buf,
            64,
        )
        if n <= 0:
            raise RuntimeError("xvs_encode_gdl90_heartbeat failed")
        return bytes(buf[:n])

    def encode_gdl90_ownship_geo_alt(
        self, *, alt_ft: float, vfom_m: float = 3.0, vertical_warning: bool = False
    ) -> bytes:
        buf = (c_uint8 * 64)()
        n = self._lib.xvs_encode_gdl90_ownship_geo_alt(
            alt_ft, vfom_m, c_uint8(1 if vertical_warning else 0), buf, 64
        )
        if n <= 0:
            raise RuntimeError("xvs_encode_gdl90_ownship_geo_alt failed")
        return bytes(buf[:n])

    def encode_gdl90_ownship(self, target: XvsGdl90Target) -> bytes:
        buf = (c_uint8 * 96)()
        n = self._lib.xvs_encode_gdl90_ownship(ctypes.byref(target), buf, 96)
        if n <= 0:
            raise RuntimeError("xvs_encode_gdl90_ownship failed")
        return bytes(buf[:n])

    def encode_gdl90_traffic(self, target: XvsGdl90Target, *, sim_prefix: bool = True) -> bytes:
        buf = (c_uint8 * 96)()
        n = self._lib.xvs_encode_gdl90_traffic(
            ctypes.byref(target), c_uint8(1 if sim_prefix else 0), buf, 96
        )
        if n <= 0:
            raise RuntimeError("xvs_encode_gdl90_traffic failed")
        return bytes(buf[:n])

    def encode_mavlink_adsb_246(self, params: XvsMavlinkAdsb246) -> bytes:
        buf = (c_uint8 * 64)()
        n = self._lib.xvs_encode_mavlink_adsb_246(ctypes.byref(params), buf, 64)
        if n != 50:
            raise RuntimeError(f"xvs_encode_mavlink_adsb_246 failed n={n}")
        return bytes(buf[:50])

    def encode_mavlink_bearing_track(
        self,
        track: XvsBearingTrack,
        *,
        seq: int,
        sysid: int,
        compid: int,
        link_id: int,
        timestamp_48: int,
        secret_key: bytes,
    ) -> bytes:
        if len(secret_key) != 32:
            raise ValueError("MAVLink v2 secret_key must be exactly 32 bytes")
        key_arr = (c_uint8 * 32)(*secret_key)
        buf = (c_uint8 * 96)()
        n = self._lib.xvs_encode_mavlink_bearing_track(
            ctypes.byref(track),
            c_uint8(seq),
            c_uint8(sysid),
            c_uint8(compid),
            c_uint8(link_id),
            c_uint64(timestamp_48),
            key_arr,
            buf,
            96,
        )
        if n != 77:
            raise RuntimeError(f"xvs_encode_mavlink_bearing_track failed n={n}")
        return bytes(buf[:77])
