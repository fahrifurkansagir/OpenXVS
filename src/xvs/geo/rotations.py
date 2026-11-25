"""Rotations, Euler angles, Quaternions, and so(3) Lie algebra.

Conforms strictly to docs/xvs/CONVENTIONS.md v1.0 §4.
"""

from __future__ import annotations
import math
import numpy as np


def rx(phi: float) -> np.ndarray:
    """Elementary rotation about x axis by angle phi (radians)."""
    c = math.cos(phi)
    s = math.sin(phi)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c]
    ], dtype=np.float64)


def ry(theta: float) -> np.ndarray:
    """Elementary rotation about y axis by angle theta (radians)."""
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c]
    ], dtype=np.float64)


def rz(psi: float) -> np.ndarray:
    """Elementary rotation about z axis by angle psi (radians)."""
    c = math.cos(psi)
    s = math.sin(psi)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def euler_to_r(yaw_rad: float, pitch_rad: float, roll_rad: float) -> np.ndarray:
    """Convert intrinsic Z-Y-X Euler angles (yaw psi, pitch theta, roll phi) to R_ned_body.

    R_ned_body = Rz(psi) @ Ry(theta) @ Rx(phi)
    v_ned = R_ned_body @ v_body
    """
    return rz(yaw_rad) @ ry(pitch_rad) @ rx(roll_rad)


def r_to_euler(r: np.ndarray) -> tuple[float, float, float]:
    """Extract intrinsic Z-Y-X Euler angles (yaw, pitch, roll in radians) from rotation matrix.

    Assumes pitch is not near gimbal lock (|pitch| < 89.9 deg).
    """
    # R[2, 0] = -sin(theta)
    sin_theta = -r[2, 0]
    sin_theta = max(-1.0, min(1.0, sin_theta))
    theta = math.asin(sin_theta)

    cos_theta = math.cos(theta)
    if abs(cos_theta) > 1e-12:
        psi = math.atan2(r[1, 0], r[0, 0])
        phi = math.atan2(r[2, 1], r[2, 2])
    else:
        # Gimbal lock fallback
        psi = math.atan2(-r[0, 1], r[1, 1])
        phi = 0.0
    return psi, theta, phi


def r_to_quat(r: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to Hamilton quaternion (w, x, y, z) with canonical w >= 0."""
    tr = r[0, 0] + r[1, 1] + r[2, 2]
    if tr > 0.0:
        s = 0.5 / math.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (r[2, 1] - r[1, 2]) * s
        y = (r[0, 2] - r[2, 0]) * s
        z = (r[1, 0] - r[0, 1]) * s
    elif (r[0, 0] > r[1, 1]) and (r[0, 0] > r[2, 2]):
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm > 0:
        q /= norm
    if q[0] < 0.0:
        q = -q
    return q


def quat_to_r(q: np.ndarray) -> np.ndarray:
    """Convert Hamilton quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)]
    ], dtype=np.float64)


def so3_exp(omega: np.ndarray) -> np.ndarray:
    """Exponential map from so(3) vector (radians) to SO(3) rotation matrix."""
    theta = float(np.linalg.norm(omega))
    wx, wy, wz = omega
    hat = np.array([
        [0.0, -wz, wy],
        [wz, 0.0, -wx],
        [-wy, wx, 0.0]
    ], dtype=np.float64)

    if theta < 1e-12:
        return np.eye(3, dtype=np.float64) + hat

    a = math.sin(theta) / theta
    b = (1.0 - math.cos(theta)) / (theta * theta)
    return np.eye(3, dtype=np.float64) + a * hat + b * (hat @ hat)


def so3_log(r: np.ndarray) -> np.ndarray:
    """Logarithmic map from SO(3) rotation matrix to so(3) vector (radians)."""
    tr = float(np.trace(r))
    cos_theta = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    theta = math.acos(cos_theta)
    hat = np.array([
        r[2, 1] - r[1, 2],
        r[0, 2] - r[2, 0],
        r[1, 0] - r[0, 1]
    ], dtype=np.float64)

    if theta < 1e-12:
        return 0.5 * hat

    return (theta / (2.0 * math.sin(theta))) * hat
