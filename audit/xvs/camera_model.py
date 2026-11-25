"""Independent reference camera model (numpy only; no Cesium code or constants used).

Conventions (from the rig spec):
  * WGS84 ellipsoid; geodetic (lat, lon, h) -> ECEF.
  * NED at (lat, lon); body FRD; R_ned_body = Rz(yaw) Ry(pitch) Rx(roll).
  * Camera frame OpenCV (x right, y down, z forward); R_body_cam columns = camera axes in body.
  * R_ecef_cam = R_ecef_ned @ R_ned_body @ R_body_cam.
  * Pinhole: fx = fy = 512 / tan(50 deg), cx = cy = 511.5 (OpenCV pixel-centre convention).
"""
import numpy as np

A = 6378137.0
F = 1.0 / 298.257223563
B = A * (1.0 - F)
E2 = F * (2.0 - F)

W = H = 1024
FOV_DEG = 100.0
FX = FY = 512.0 / np.tan(np.radians(FOV_DEG / 2.0))
CX = CY = 511.5


def geodetic_to_ecef(lat_deg, lon_deg, h):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = A / np.sqrt(1.0 - E2 * np.sin(lat) ** 2)
    return np.array([
        (n + h) * np.cos(lat) * np.cos(lon),
        (n + h) * np.cos(lat) * np.sin(lon),
        (n * (1.0 - E2) + h) * np.sin(lat),
    ])


def r_ecef_ned(lat_deg, lon_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    n = [-sl * co, -sl * so, cl]
    e = [-so, co, 0.0]
    d = [-cl * co, -cl * so, -sl]
    return np.column_stack([n, e, d])


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def r_ned_body(yaw_deg, pitch_deg, roll_deg):
    return rz(np.radians(yaw_deg)) @ ry(np.radians(pitch_deg)) @ rx(np.radians(roll_deg))


# camera axes (x_cam, y_cam, z_cam) expressed in body FRD coordinates
CAMS = {
    "F": ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    "R": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    "B": ((0, -1, 0), (0, 0, 1), (-1, 0, 0)),
    "L": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
}


def r_body_cam(name):
    x, y, z = CAMS[name]
    return np.column_stack([x, y, z]).astype(float)


class Pose:
    def __init__(self, lat, lon, h, yaw, pitch, roll, cam):
        self.lat, self.lon, self.h = lat, lon, h
        self.yaw, self.pitch, self.roll, self.cam = yaw, pitch, roll, cam
        self.C = geodetic_to_ecef(lat, lon, h)
        self.R = r_ecef_ned(lat, lon) @ r_ned_body(yaw, pitch, roll) @ r_body_cam(cam)  # R_ecef_cam

    def cesium_view(self):
        """Values handed to Cesium camera.setView + PerspectiveFrustum."""
        return {
            "pos": self.C.tolist(),
            "dir": self.R[:, 2].tolist(),
            "up": (-self.R[:, 1]).tolist(),
            "fov": float(np.radians(FOV_DEG)),
        }

    def to_cam(self, P):
        P = np.atleast_2d(P)
        return (P - self.C) @ self.R  # rows: R^T (P - C)

    def project(self, P):
        pc = self.to_cam(P)
        u = FX * pc[:, 0] / pc[:, 2] + CX
        v = FY * pc[:, 1] / pc[:, 2] + CY
        return u, v, pc[:, 2]

    def backproject(self, u, v, rng):
        """ECEF point at range `rng` (metres, along the ray) through pixel (u, v)."""
        d = np.array([(u - CX) / FX, (v - CY) / FY, 1.0])
        d /= np.linalg.norm(d)
        return self.C + self.R @ (rng * d)

    def as_dict(self):
        return dict(lat=self.lat, lon=self.lon, h=self.h, yaw=self.yaw, pitch=self.pitch, roll=self.roll, cam=self.cam)


def limb_points(C, n=400000):
    """Analytic limb (tangent-point circle) of the WGS84 ellipsoid seen from C (ECEF)."""
    Cp = np.array([C[0] / A, C[1] / A, C[2] / B])
    k = np.linalg.norm(Cp)
    ch = Cp / k
    centre = ch / k
    rad = np.sqrt(1.0 - 1.0 / k ** 2)
    # orthonormal basis perpendicular to ch
    t = np.array([0.0, 0.0, 1.0]) if abs(ch[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(ch, t); e1 /= np.linalg.norm(e1)
    e2 = np.cross(ch, e1)
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    Y = centre[None, :] + rad * (np.cos(th)[:, None] * e1[None, :] + np.sin(th)[:, None] * e2[None, :])
    X = np.column_stack([A * Y[:, 0], A * Y[:, 1], B * Y[:, 2]])
    return X


def limb_curve_v_at_columns(pose, n=400000):
    """For each column centre u=0..W-1 return analytic limb row v (NaN if absent or ambiguous).

    Uses the densely sampled tangent circle; finds all crossings of each column
    between consecutive in-front samples, linear interpolation in u.
    """
    X = limb_points(pose.C, n)
    u, v, z = pose.project(X)
    vcol = np.full(W, np.nan)
    ncross = np.zeros(W, dtype=int)
    u2 = np.roll(u, -1); v2 = np.roll(v, -1); z2 = np.roll(z, -1)
    ok = (z > 0) & (z2 > 0) & (np.abs(u2 - u) < 50)  # skip wrap through infinity
    ua, ub, va, vb = u[ok], u2[ok], v[ok], v2[ok]
    lo, hi = np.minimum(ua, ub), np.maximum(ua, ub)
    # candidate integer columns inside [lo, hi)
    c0 = np.ceil(lo).astype(int)
    c1 = np.ceil(hi).astype(int) - 1
    has = c1 >= c0
    for a_, b_, va_, vb_, s, e in zip(ua[has], ub[has], va[has], vb[has], c0[has], c1[has]):
        for col in range(max(s, 0), min(e, W - 1) + 1):
            t = (col - a_) / (b_ - a_)
            vv = va_ + t * (vb_ - va_)
            if -0.5 <= vv <= H - 0.5:
                if ncross[col] == 0:
                    vcol[col] = vv
                ncross[col] += 1
    vcol[ncross != 1] = np.nan
    return vcol, ncross


def horizontal_plane_v_at_columns(pose):
    """Vanishing line of the local-level plane through the camera (no dip): v(u) at column centres."""
    lat, lon = np.radians(pose.lat), np.radians(pose.lon)
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])  # geodetic normal
    nc = pose.R.T @ up
    u = np.arange(W, dtype=float)
    # nc . [ (u-cx)/fx, (v-cy)/fy, 1 ] = 0
    return CY - FY * (nc[2] + nc[0] * (u - CX) / FX) / nc[1]
