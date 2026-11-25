# -*- coding: utf-8 -*-
"""Kinematic aircraft trajectory models T1, T2, T3 and aircraft state representations.

Conforms strictly to docs/xvs/WP01_SIM_RIG.md v1.1 Section2.2 and docs/xvs/CONVENTIONS.md.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable
import numpy as np

from xvs.geo import wgs84, rotations
from xvs.sim import airports


@dataclass
class AircraftState:
    t_s: float
    lat_deg: float
    lon_deg: float
    h_m: float
    yaw_rad: float
    pitch_rad: float
    roll_rad: float
    p_ecef_m: np.ndarray
    v_ned_mps: np.ndarray
    ypr_deg: tuple[float, float, float]
    q_ned_body_wxyz: np.ndarray
    r_ecef_body: np.ndarray


_RK4_HISTORY: dict[tuple[Any, ...], list[tuple[float, float, float]]] = {}

_T1_TRACK_RAD = math.radians(315.0)
_T1_SPEED = 120.0
_T1_V_N = _T1_SPEED * math.cos(_T1_TRACK_RAD)
_T1_V_E = _T1_SPEED * math.sin(_T1_TRACK_RAD)


def _t1_vel_func(_: float) -> tuple[float, float, float]:
    return (_T1_V_N, _T1_V_E, 0.0)


def _integrate_rk4_trajectory(
    t_target: float,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
    vel_func: Callable[[float], tuple[float, float, float]],
    dt: float = 0.01,
) -> tuple[float, float, float]:
    """Integrate kinematic position on the WGS84 ellipsoid via RK4 with step dt = 0.01 s.

    Uses prefix-history caching so repeated or sequential evaluations do not re-integrate from 0.
    """
    if t_target <= 0.0:
        return lat0_rad, lon0_rad, h0_m

    n_full_steps = int(t_target // dt)
    remainder = t_target - n_full_steps * dt

    def deriv(lat_k: float, lon_k: float, h_k: float, t_k: float) -> tuple[float, float, float]:
        v_n, v_e, v_d = vel_func(t_k)
        m = wgs84.radius_m(lat_k)
        n = wgs84.radius_n(lat_k)
        dlat = v_n / (m + h_k)
        dlon = v_e / ((n + h_k) * math.cos(lat_k))
        dh = -v_d
        return dlat, dlon, dh

    cache_key = (lat0_rad, lon0_rad, h0_m, dt, vel_func)
    if cache_key not in _RK4_HISTORY:
        _RK4_HISTORY[cache_key] = [(lat0_rad, lon0_rad, h0_m)]

    history = _RK4_HISTORY[cache_key]
    curr_len = len(history)

    if curr_len <= n_full_steps:
        lat, lon, h = history[-1]
        for i in range(curr_len - 1, n_full_steps):
            t_k = i * dt
            k1_lat, k1_lon, k1_h = deriv(lat, lon, h, t_k)
            k2_lat, k2_lon, k2_h = deriv(lat + 0.5 * dt * k1_lat, lon + 0.5 * dt * k1_lon, h + 0.5 * dt * k1_h, t_k + 0.5 * dt)
            k3_lat, k3_lon, k3_h = deriv(lat + 0.5 * dt * k2_lat, lon + 0.5 * dt * k2_lon, h + 0.5 * dt * k2_h, t_k + 0.5 * dt)
            k4_lat, k4_lon, k4_h = deriv(lat + dt * k3_lat, lon + dt * k3_lon, h + dt * k3_h, t_k + dt)

            lat += (dt / 6.0) * (k1_lat + 2.0 * k2_lat + 2.0 * k3_lat + k4_lat)
            lon += (dt / 6.0) * (k1_lon + 2.0 * k2_lon + 2.0 * k3_lon + k4_lon)
            h += (dt / 6.0) * (k1_h + 2.0 * k2_h + 2.0 * k3_h + k4_h)
            history.append((lat, lon, h))

    lat, lon, h = history[n_full_steps]

    if remainder > 1e-12:
        t_k = n_full_steps * dt
        dt_rem = remainder
        k1_lat, k1_lon, k1_h = deriv(lat, lon, h, t_k)
        k2_lat, k2_lon, k2_h = deriv(lat + 0.5 * dt_rem * k1_lat, lon + 0.5 * dt_rem * k1_lon, h + 0.5 * dt_rem * k1_h, t_k + 0.5 * dt_rem)
        k3_lat, k3_lon, k3_h = deriv(lat + 0.5 * dt_rem * k2_lat, lon + 0.5 * dt_rem * k2_lon, h + 0.5 * dt_rem * k2_h, t_k + 0.5 * dt_rem)
        k4_lat, k4_lon, k4_h = deriv(lat + dt_rem * k3_lat, lon + dt_rem * k3_lon, h + dt_rem * k3_h, t_k + dt_rem)

        lat += (dt_rem / 6.0) * (k1_lat + 2.0 * k2_lat + 2.0 * k3_lat + k4_lat)
        lon += (dt_rem / 6.0) * (k1_lon + 2.0 * k2_lon + 2.0 * k3_lon + k4_lon)
        h += (dt_rem / 6.0) * (k1_h + 2.0 * k2_h + 2.0 * k3_h + k4_h)

    return lat, lon, h


def t1_cruise_state(t_s: float) -> AircraftState:
    """Evaluate T1 level cruise at time t_s.

    Conforms to docs/xvs/WP01_SIM_RIG.md v1.1 Section2.2:
      start (37.050000 deg, -120.850000 deg), h = 3000 m, 60 s
      ground track 315 deg, ground speed 120 m/s, v_D = 0
      attitude: psi = 315 deg, theta = 2.0 deg + 1.0 deg * sin(2*pi*t/15 s), phi = 5.0 deg * sin(2*pi*t/20 s)
    """
    track_deg = 315.0
    v_n = _T1_V_N
    v_e = _T1_V_E
    v_d = 0.0

    lat0 = math.radians(37.050000)
    lon0 = math.radians(-120.850000)
    h0 = 3000.0

    lat_rad, lon_rad, h_m = _integrate_rk4_trajectory(
        t_target=t_s,
        lat0_rad=lat0,
        lon0_rad=lon0,
        h0_m=h0,
        vel_func=_t1_vel_func,
        dt=0.01,
    )

    lat_deg = math.degrees(lat_rad)
    lon_deg = math.degrees(lon_rad)

    yaw_deg = 315.0
    pitch_deg = 2.0 + 1.0 * math.sin(2.0 * math.pi * t_s / 15.0)
    roll_deg = 5.0 * math.sin(2.0 * math.pi * t_s / 20.0)

    yaw_rad = math.radians(yaw_deg)
    pitch_rad = math.radians(pitch_deg)
    roll_rad = math.radians(roll_deg)

    p_ecef = wgs84.geodetic_to_ecef(lat_deg, lon_deg, h_m)
    r_ned_body = rotations.euler_to_r(yaw_rad, pitch_rad, roll_rad)
    r_ecef_ned_mat = wgs84.r_ecef_ned(lat_deg, lon_deg)
    r_ecef_body = r_ecef_ned_mat @ r_ned_body
    q_ned_body = rotations.r_to_quat(r_ned_body)
    v_ned = np.array([v_n, v_e, v_d], dtype=np.float64)

    return AircraftState(
        t_s=t_s,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        h_m=h_m,
        yaw_rad=yaw_rad,
        pitch_rad=pitch_rad,
        roll_rad=roll_rad,
        p_ecef_m=p_ecef,
        v_ned_mps=v_ned,
        ypr_deg=(yaw_deg, pitch_deg, roll_deg),
        q_ned_body_wxyz=q_ned_body,
        r_ecef_body=r_ecef_body,
    )


def t2_ksfo28l_state(
    t_s: float,
    thr: airports.RunwayThreshold | None = None,
) -> AircraftState:
    """Evaluate T2 straight-in approach to KSFO runway 28L at time t_s.

    Conforms strictly to docs/xvs/WP01_SIM_RIG.md v1.1 Section2.2:
      d(t) = 70.0 * (60.0 - t)
      offset toward he_heading_degT + 180 deg using exact tangent-plane construction:
        wgs84.ned_offset_to_geodetic(thr.lat, thr.lon, offset_north, offset_east)
      h(t) = h_THR + 15.0 + d(t) * tan(3.0 deg)
      v_ground = 70 m/s along runway heading; v_D = 70 * tan(3 deg) computed
      Attitude: psi = he_heading_degT, theta = +2.5 deg, phi = 1.5 deg * sin(2*pi*t/12)
    """
    if thr is None:
        thr = airports.load_ksfo_28l_threshold()

    d_t = 70.0 * (60.0 - t_s)
    back_bearing_deg = (thr.heading_deg + 180.0) % 360.0
    back_bearing_rad = math.radians(back_bearing_deg)

    # Offset in tangent (NED) plane at THR via exact tangent-plane construction
    offset_north = d_t * math.cos(back_bearing_rad)
    offset_east = d_t * math.sin(back_bearing_rad)

    lat_deg, lon_deg, _ = wgs84.ned_offset_to_geodetic(
        thr.lat_deg, thr.lon_deg, offset_north, offset_east
    )

    # Height in ellipsoid mode
    h_m = thr.h_m + 15.0 + d_t * math.tan(math.radians(3.0))

    # Velocity: v_D = 70.0 * tan(3 deg) computed, not hardcoded
    head_rad = math.radians(thr.heading_deg)
    v_n = 70.0 * math.cos(head_rad)
    v_e = 70.0 * math.sin(head_rad)
    v_d = 70.0 * math.tan(math.radians(3.0))
    v_ned = np.array([v_n, v_e, v_d], dtype=np.float64)

    # Attitude
    yaw_deg = thr.heading_deg
    pitch_deg = 2.5
    roll_deg = 1.5 * math.sin(2.0 * math.pi * t_s / 12.0)

    yaw_rad = math.radians(yaw_deg)
    pitch_rad = math.radians(pitch_deg)
    roll_rad = math.radians(roll_deg)

    p_ecef = wgs84.geodetic_to_ecef(lat_deg, lon_deg, h_m)
    r_ned_body = rotations.euler_to_r(yaw_rad, pitch_rad, roll_rad)
    r_ecef_ned_mat = wgs84.r_ecef_ned(lat_deg, lon_deg)
    r_ecef_body = r_ecef_ned_mat @ r_ned_body
    q_ned_body = rotations.r_to_quat(r_ned_body)

    return AircraftState(
        t_s=t_s,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        h_m=h_m,
        yaw_rad=yaw_rad,
        pitch_rad=pitch_rad,
        roll_rad=roll_rad,
        p_ecef_m=p_ecef,
        v_ned_mps=v_ned,
        ypr_deg=(yaw_deg, pitch_deg, roll_deg),
        q_ned_body_wxyz=q_ned_body,
        r_ecef_body=r_ecef_body,
    )


_T3_G = 9.80665
_T3_V_TOTAL = 100.0
_T3_V_D = -5.0
_T3_V_H = math.sqrt(_T3_V_TOTAL * _T3_V_TOTAL - _T3_V_D * _T3_V_D)
_T3_BANK_RAD = math.radians(25.0)
_T3_OMEGA = _T3_G * math.tan(_T3_BANK_RAD) / _T3_V_H


def _t3_vel_func(t_k: float) -> tuple[float, float, float]:
    psi_k = _T3_OMEGA * t_k
    return (_T3_V_H * math.cos(psi_k), _T3_V_H * math.sin(psi_k), _T3_V_D)


def t3_turn_climb_state(t_s: float) -> AircraftState:
    """Evaluate T3 coordinated right turn with climb at time t_s.

    Conforms to docs/xvs/WP01_SIM_RIG.md v1.1 Section2.2:
      start (37.250000 deg, -120.900000 deg), h0 = 1500 m, 60 s
      airspeed V = 100 m/s, climb v_D = -5.0 m/s, horizontal speed V_h = sqrt(100^2 - 5^2) = 99.8749 m/s
      bank phi = +25 deg (constant), turn rate omega = g * tan(phi) / V_h with g = 9.80665
      heading psi(t) = 0 deg + omega * t; v_N = V_h * cos(psi), v_E = V_h * sin(psi)
      theta = asin(5/100) + 2.0 deg = 4.8660 deg
    """
    omega = _T3_OMEGA

    lat0 = math.radians(37.250000)
    lon0 = math.radians(-120.900000)
    h0 = 1500.0

    lat_rad, lon_rad, h_m = _integrate_rk4_trajectory(
        t_target=t_s,
        lat0_rad=lat0,
        lon0_rad=lon0,
        h0_m=h0,
        vel_func=_t3_vel_func,
        dt=0.01,
    )

    lat_deg = math.degrees(lat_rad)
    lon_deg = math.degrees(lon_rad)

    yaw_rad = (omega * t_s) % (2.0 * math.pi)
    yaw_deg = math.degrees(yaw_rad)

    pitch_rad = math.asin(5.0 / 100.0) + math.radians(2.0)
    pitch_deg = math.degrees(pitch_rad)

    roll_deg = 25.0
    roll_rad = _T3_BANK_RAD

    v_n, v_e, _ = _t3_vel_func(t_s)
    v_ned = np.array([v_n, v_e, _T3_V_D], dtype=np.float64)

    p_ecef = wgs84.geodetic_to_ecef(lat_deg, lon_deg, h_m)
    r_ned_body = rotations.euler_to_r(yaw_rad, pitch_rad, roll_rad)
    r_ecef_ned_mat = wgs84.r_ecef_ned(lat_deg, lon_deg)
    r_ecef_body = r_ecef_ned_mat @ r_ned_body
    q_ned_body = rotations.r_to_quat(r_ned_body)

    return AircraftState(
        t_s=t_s,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        h_m=h_m,
        yaw_rad=yaw_rad,
        pitch_rad=pitch_rad,
        roll_rad=roll_rad,
        p_ecef_m=p_ecef,
        v_ned_mps=v_ned,
        ypr_deg=(yaw_deg, pitch_deg, roll_deg),
        q_ned_body_wxyz=q_ned_body,
        r_ecef_body=r_ecef_body,
    )
