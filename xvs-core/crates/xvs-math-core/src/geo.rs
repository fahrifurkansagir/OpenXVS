//! WGS-84 ellipsoid geometry, local NED frames, Hamilton quaternions, Z-Y-X Euler angles,
//! and `SO(3)` Lie algebra (`Exp`/`Log`), conforming strictly to `docs/xvs/CONVENTIONS.md` v1.0.

pub use crate::det_f64::{
    mat3_mul, mat3_mul_vec, mat3_t_mul_vec, mat3_transpose, Mat3, Vec3,
};
use crate::det_f64::{acos, asin, atan2, clamp, cos, fabs, hypot, mat3_eye, norm3, sin, sqrt};

pub const WGS84_A: f64 = 6_378_137.0;
pub const WGS84_F: f64 = 1.0 / 298.257_223_563;
pub const WGS84_B: f64 = WGS84_A * (1.0 - WGS84_F);
pub const WGS84_E2: f64 = WGS84_F * (2.0 - WGS84_F);
pub const WGS84_E_PRIME2: f64 = (WGS84_A * WGS84_A - WGS84_B * WGS84_B) / (WGS84_B * WGS84_B);

const DEG_TO_RAD: f64 = core::f64::consts::PI / 180.0;
const RAD_TO_DEG: f64 = 180.0 / core::f64::consts::PI;

#[inline]
pub fn deg_to_rad(deg: f64) -> f64 {
    deg * DEG_TO_RAD
}

#[inline]
pub fn rad_to_deg(rad: f64) -> f64 {
    rad * RAD_TO_DEG
}

/// Prime vertical radius of curvature `N(phi)` in metres.
#[inline]
pub fn radius_n(lat_rad: f64) -> f64 {
    let s = sin(lat_rad);
    WGS84_A / sqrt(1.0 - WGS84_E2 * s * s)
}

/// Meridional radius of curvature `M(phi)` in metres.
#[inline]
pub fn radius_m(lat_rad: f64) -> f64 {
    let s = sin(lat_rad);
    let denom = 1.0 - WGS84_E2 * s * s;
    WGS84_A * (1.0 - WGS84_E2) / (denom * sqrt(denom))
}

/// Convert geodetic `(lat_deg, lon_deg, h_m)` to WGS-84 ECEF `[X, Y, Z]` in metres.
pub fn geodetic_to_ecef(lat_deg: f64, lon_deg: f64, h_m: f64) -> Vec3 {
    let lat_rad = deg_to_rad(lat_deg);
    let lon_rad = deg_to_rad(lon_deg);
    let cos_lat = cos(lat_rad);
    let sin_lat = sin(lat_rad);
    let cos_lon = cos(lon_rad);
    let sin_lon = sin(lon_rad);
    let n = radius_n(lat_rad);
    let x = (n + h_m) * cos_lat * cos_lon;
    let y = (n + h_m) * cos_lat * sin_lon;
    let z = (n * (1.0 - WGS84_E2) + h_m) * sin_lat;
    [x, y, z]
}

/// Convert ECEF `(x, y, z)` in metres to geodetic `(lat_deg, lon_deg, h_m)` via iterated Bowring.
pub fn ecef_to_geodetic(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let p = hypot(x, y);
    if p < 1e-10 {
        let lat_rad = if z >= 0.0 {
            core::f64::consts::FRAC_PI_2
        } else {
            -core::f64::consts::FRAC_PI_2
        };
        let h_m = fabs(z) - WGS84_B;
        return (rad_to_deg(lat_rad), 0.0, h_m);
    }

    let lon_rad = atan2(y, x);
    let theta = atan2(z * WGS84_A, p * WGS84_B);
    let sin_t = sin(theta);
    let cos_t = cos(theta);
    let mut phi = atan2(
        z + WGS84_E_PRIME2 * WGS84_B * sin_t * sin_t * sin_t,
        p - WGS84_E2 * WGS84_A * cos_t * cos_t * cos_t,
    );

    for _ in 0..6 {
        let sin_phi = sin(phi);
        let cos_phi = cos(phi);
        let n = WGS84_A / sqrt(1.0 - WGS84_E2 * sin_phi * sin_phi);
        let h = if fabs(cos_phi) > 1e-10 {
            p / cos_phi - n
        } else {
            fabs(z) / fabs(sin_phi) - n * (1.0 - WGS84_E2)
        };
        let phi_next = atan2(z, p * (1.0 - WGS84_E2 * n / (n + h)));
        if fabs(phi_next - phi) < 1e-14 {
            phi = phi_next;
            break;
        }
        phi = phi_next;
    }

    let sin_phi = sin(phi);
    let cos_phi = cos(phi);
    let n = WGS84_A / sqrt(1.0 - WGS84_E2 * sin_phi * sin_phi);
    let h_m = if fabs(cos_phi) > 1e-10 {
        p / cos_phi - n
    } else {
        fabs(z) / fabs(sin_phi) - n * (1.0 - WGS84_E2)
    };
    (rad_to_deg(phi), rad_to_deg(lon_rad), h_m)
}

/// Local NED basis unit vectors `[n, e, d]` expressed in ECEF (`CONVENTIONS.md` §3).
pub fn ned_basis(lat_deg: f64, lon_deg: f64) -> (Vec3, Vec3, Vec3) {
    let lat_rad = deg_to_rad(lat_deg);
    let lon_rad = deg_to_rad(lon_deg);
    let sin_lat = sin(lat_rad);
    let cos_lat = cos(lat_rad);
    let sin_lon = sin(lon_rad);
    let cos_lon = cos(lon_rad);
    let n = [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat];
    let e = [-sin_lon, cos_lon, 0.0];
    let d = [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat];
    (n, e, d)
}

/// Rotation matrix `R_ecef_ned` (`v_ecef = R_ecef_ned @ v_ned`, columns are `[n, e, d]`).
pub fn r_ecef_ned(lat_deg: f64, lon_deg: f64) -> Mat3 {
    let (n, e, d) = ned_basis(lat_deg, lon_deg);
    [
        [n[0], e[0], d[0]],
        [n[1], e[1], d[1]],
        [n[2], e[2], d[2]],
    ]
}

/// Exact tangent-plane geodetic point offset by `(dn_m, de_m)` in local NED.
pub fn ned_offset_to_geodetic(lat_deg: f64, lon_deg: f64, dn_m: f64, de_m: f64) -> (f64, f64, f64) {
    let p_ref = geodetic_to_ecef(lat_deg, lon_deg, 0.0);
    let r_mat = r_ecef_ned(lat_deg, lon_deg);
    let off = mat3_mul_vec(&r_mat, [dn_m, de_m, 0.0]);
    ecef_to_geodetic(p_ref[0] + off[0], p_ref[1] + off[1], p_ref[2] + off[2])
}

#[inline]
pub fn rx(phi: f64) -> Mat3 {
    let c = cos(phi);
    let s = sin(phi);
    [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
}

#[inline]
pub fn ry(theta: f64) -> Mat3 {
    let c = cos(theta);
    let s = sin(theta);
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}

#[inline]
pub fn rz(psi: f64) -> Mat3 {
    let c = cos(psi);
    let s = sin(psi);
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

/// Intrinsic `Z-Y-X` Euler `(yaw psi, pitch theta, roll phi)` to `R_ned_body = Rz(psi) Ry(theta) Rx(phi)`.
pub fn euler_to_r(yaw_rad: f64, pitch_rad: f64, roll_rad: f64) -> Mat3 {
    let rzy = mat3_mul(&rz(yaw_rad), &ry(pitch_rad));
    mat3_mul(&rzy, &rx(roll_rad))
}

/// Extract intrinsic `Z-Y-X` Euler `(yaw, pitch, roll)` in radians from rotation matrix.
pub fn r_to_euler(r: &Mat3) -> (f64, f64, f64) {
    let sin_theta = clamp(-r[2][0], -1.0, 1.0);
    let theta = asin(sin_theta);
    let cos_theta = cos(theta);
    if fabs(cos_theta) > 1e-12 {
        let psi = atan2(r[1][0], r[0][0]);
        let phi = atan2(r[2][1], r[2][2]);
        (psi, theta, phi)
    } else {
        let psi = atan2(-r[0][1], r[1][1]);
        (psi, theta, 0.0)
    }
}

/// Normalize Hamilton quaternion `(w, x, y, z)` enforcing canonical `w >= 0.0`.
pub fn quat_normalize(q: [f64; 4]) -> [f64; 4] {
    let n = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if n <= 1e-15 || !n.is_finite() {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let sign = if q[0] < 0.0 { -1.0 } else { 1.0 };
    let inv = sign / n;
    [q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv]
}

/// Hamilton quaternion product `a ⊗ b` with canonical `w >= 0`.
pub fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    quat_normalize([
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ])
}

/// Convert `3x3` rotation matrix to canonical Hamilton quaternion `(w, x, y, z)` (`w >= 0`).
pub fn r_to_quat(r: &Mat3) -> [f64; 4] {
    let tr = r[0][0] + r[1][1] + r[2][2];
    let (w, x, y, z) = if tr > 0.0 {
        let s = 0.5 / sqrt(tr + 1.0);
        (
            0.25 / s,
            (r[2][1] - r[1][2]) * s,
            (r[0][2] - r[2][0]) * s,
            (r[1][0] - r[0][1]) * s,
        )
    } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
        let s = 2.0 * sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]);
        (
            (r[2][1] - r[1][2]) / s,
            0.25 * s,
            (r[0][1] + r[1][0]) / s,
            (r[0][2] + r[2][0]) / s,
        )
    } else if r[1][1] > r[2][2] {
        let s = 2.0 * sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]);
        (
            (r[0][2] - r[2][0]) / s,
            (r[0][1] + r[1][0]) / s,
            0.25 * s,
            (r[1][2] + r[2][1]) / s,
        )
    } else {
        let s = 2.0 * sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]);
        (
            (r[1][0] - r[0][1]) / s,
            (r[0][2] + r[2][0]) / s,
            (r[1][2] + r[2][1]) / s,
            0.25 * s,
        )
    };
    quat_normalize([w, x, y, z])
}

/// Convert intrinsic `Z-Y-X` Euler `(yaw, pitch, roll)` directly to canonical Hamilton `(w, x, y, z)`.
pub fn euler_zyx_to_quat(yaw_rad: f64, pitch_rad: f64, roll_rad: f64) -> [f64; 4] {
    let (cy, sy) = (cos(0.5 * yaw_rad), sin(0.5 * yaw_rad));
    let (cp, sp) = (cos(0.5 * pitch_rad), sin(0.5 * pitch_rad));
    let (cr, sr) = (cos(0.5 * roll_rad), sin(0.5 * roll_rad));
    quat_normalize([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])
}

/// Convert Hamilton quaternion `(w, x, y, z)` to `3x3` rotation matrix.
pub fn quat_to_r(q: [f64; 4]) -> Mat3 {
    let [w, x, y, z] = q;
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// Skew-symmetric `so(3)` hat matrix `[omega]_x`.
#[inline]
pub const fn so3_hat(omega: Vec3) -> Mat3 {
    [
        [0.0, -omega[2], omega[1]],
        [omega[2], 0.0, -omega[0]],
        [-omega[1], omega[0], 0.0],
    ]
}

/// Exponential map `so(3) -> SO(3)` via Rodrigues formula.
pub fn so3_exp(omega: Vec3) -> Mat3 {
    let theta = norm3(omega);
    let hat = so3_hat(omega);
    let eye = mat3_eye();
    if theta < 1e-12 {
        return [
            [eye[0][0] + hat[0][0], eye[0][1] + hat[0][1], eye[0][2] + hat[0][2]],
            [eye[1][0] + hat[1][0], eye[1][1] + hat[1][1], eye[1][2] + hat[1][2]],
            [eye[2][0] + hat[2][0], eye[2][1] + hat[2][1], eye[2][2] + hat[2][2]],
        ];
    }
    let a = sin(theta) / theta;
    let b = (1.0 - cos(theta)) / (theta * theta);
    let hat2 = mat3_mul(&hat, &hat);
    let mut out = eye;
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] += a * hat[i][j] + b * hat2[i][j];
        }
    }
    out
}

/// Logarithmic map `SO(3) -> so(3)` (radians).
pub fn so3_log(r: &Mat3) -> Vec3 {
    let tr = r[0][0] + r[1][1] + r[2][2];
    let cos_theta = clamp((tr - 1.0) * 0.5, -1.0, 1.0);
    let theta = acos(cos_theta);
    let vee = [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]];
    if theta < 1e-12 {
        return [0.5 * vee[0], 0.5 * vee[1], 0.5 * vee[2]];
    }
    let s = theta / (2.0 * sin(theta));
    [s * vee[0], s * vee[1], s * vee[2]]
}
