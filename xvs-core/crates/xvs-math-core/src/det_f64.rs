//! Deterministic IEEE-754 `f64` math wrappers routing every transcendental strictly through
//! pure-Rust `libm`, canonical quiet NaN sanitization, `SplitMix64` PRNG, and stack-only 3D math.

pub const CANONICAL_QNAN_BITS: u64 = 0x7FF8_0000_0000_0000;

pub type Vec3 = [f64; 3];
pub type Mat3 = [[f64; 3]; 3];

/// Deterministic `f64` wrapper guaranteeing canonical NaN payloads and `-0.0` normalization.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct DetF64(pub f64);

impl DetF64 {
    #[inline]
    pub fn new(v: f64) -> Self {
        Self(sanitize_nan(v))
    }

    #[inline]
    pub const fn raw(self) -> f64 {
        self.0
    }
}

/// Replace any NaN bit-pattern with the canonical quiet NaN (`0x7FF8_0000_0000_0000`).
#[inline]
pub fn sanitize_nan(v: f64) -> f64 {
    if v.is_nan() {
        f64::from_bits(CANONICAL_QNAN_BITS)
    } else if v == 0.0 {
        0.0
    } else {
        v
    }
}

#[inline]
pub fn sin(x: f64) -> f64 {
    sanitize_nan(libm::sin(x))
}

#[inline]
pub fn cos(x: f64) -> f64 {
    sanitize_nan(libm::cos(x))
}

#[inline]
pub fn tan(x: f64) -> f64 {
    sanitize_nan(libm::tan(x))
}

#[inline]
pub fn asin(x: f64) -> f64 {
    sanitize_nan(libm::asin(x))
}

#[inline]
pub fn acos(x: f64) -> f64 {
    sanitize_nan(libm::acos(x))
}

#[inline]
pub fn atan2(y: f64, x: f64) -> f64 {
    sanitize_nan(libm::atan2(y, x))
}

#[inline]
pub fn sqrt(x: f64) -> f64 {
    sanitize_nan(libm::sqrt(x))
}

#[inline]
pub fn hypot(x: f64, y: f64) -> f64 {
    sanitize_nan(libm::hypot(x, y))
}

#[inline]
pub fn exp(x: f64) -> f64 {
    sanitize_nan(libm::exp(x))
}

#[inline]
pub fn ln(x: f64) -> f64 {
    sanitize_nan(libm::log(x))
}

#[inline]
pub fn fabs(x: f64) -> f64 {
    sanitize_nan(libm::fabs(x))
}

#[inline]
pub fn floor(x: f64) -> f64 {
    sanitize_nan(libm::floor(x))
}

#[inline]
pub fn ceil(x: f64) -> f64 {
    sanitize_nan(libm::ceil(x))
}

#[inline]
pub fn round(x: f64) -> f64 {
    sanitize_nan(libm::round(x))
}

/// IEEE-754 round-to-nearest ties-to-even (`libm::rint` / Python 3 `round()` parity).
#[inline]
pub fn round_ties_even(x: f64) -> f64 {
    if !x.is_finite() {
        return sanitize_nan(x);
    }
    let f = libm::floor(x);
    let diff = x - f;
    if diff < 0.5 {
        f
    } else if diff > 0.5 {
        f + 1.0
    } else {
        // Exact tie: choose even integer
        let half = libm::floor(f * 0.5);
        if (f - 2.0 * half) == 0.0 {
            f
        } else {
            f + 1.0
        }
    }
}

#[inline]
pub fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    if !x.is_finite() {
        return lo;
    }
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// NaN/Inf-preserving maximum: returns canonical `NaN` whenever either operand is non-finite.
/// Prevents IEEE-754 `f64::max(NaN, 0.0) == 0.0` fail-open swallowing in safety monitors.
#[inline]
pub fn max_preserve_nan(a: f64, b: f64) -> f64 {
    if !a.is_finite() || !b.is_finite() {
        f64::from_bits(CANONICAL_QNAN_BITS)
    } else if a >= b {
        a
    } else {
        b
    }
}

/// NaN/Inf-preserving minimum: returns canonical `NaN` whenever either operand is non-finite.
#[inline]
pub fn min_preserve_nan(a: f64, b: f64) -> f64 {
    if !a.is_finite() || !b.is_finite() {
        f64::from_bits(CANONICAL_QNAN_BITS)
    } else if a <= b {
        a
    } else {
        b
    }
}

/// NaN/Inf-preserving clamp: returns canonical `NaN` whenever any operand is non-finite.
#[inline]
pub fn clamp_preserve_nan(x: f64, lo: f64, hi: f64) -> f64 {
    if !x.is_finite() || !lo.is_finite() || !hi.is_finite() {
        f64::from_bits(CANONICAL_QNAN_BITS)
    } else if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

#[inline]
pub fn dot3(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub fn cross3(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub fn add3(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub fn sub3(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub fn scale3(a: Vec3, s: f64) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub fn norm3(a: Vec3) -> f64 {
    sqrt(dot3(a, a))
}

#[inline]
pub fn normalize3(a: Vec3) -> Vec3 {
    let n = norm3(a);
    if n > 1e-15 && n.is_finite() {
        [a[0] / n, a[1] / n, a[2] / n]
    } else {
        [1.0, 0.0, 0.0]
    }
}

#[inline]
pub const fn mat3_eye() -> Mat3 {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

#[inline]
pub const fn mat3_zero() -> Mat3 {
    [[0.0; 3]; 3]
}

#[inline]
pub fn mat3_transpose(m: &Mat3) -> Mat3 {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

#[inline]
pub fn mat3_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

#[inline]
pub fn mat3_mul_vec(m: &Mat3, v: Vec3) -> Vec3 {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

#[inline]
pub fn mat3_t_mul_vec(m: &Mat3, v: Vec3) -> Vec3 {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

#[inline]
pub fn mat3_add(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][j] + b[i][j];
        }
    }
    out
}

#[inline]
pub fn mat3_scale(a: &Mat3, s: f64) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][j] * s;
        }
    }
    out
}

#[inline]
pub fn mat3_trace(a: &Mat3) -> f64 {
    a[0][0] + a[1][1] + a[2][2]
}

/// Invert a 3x3 matrix if non-singular (`|det| > 1e-18`).
pub fn mat3_inv(m: &Mat3) -> Option<Mat3> {
    let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let c01 = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]);
    let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
    if !det.is_finite() || fabs(det) < 1e-18 {
        return None;
    }
    let inv_det = 1.0 / det;
    let c10 = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]);
    let c11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    let c12 = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]);
    let c20 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    let c21 = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]);
    let c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    Some([
        [c00 * inv_det, c10 * inv_det, c20 * inv_det],
        [c01 * inv_det, c11 * inv_det, c21 * inv_det],
        [c02 * inv_det, c12 * inv_det, c22 * inv_det],
    ])
}

/// Lower-triangular Cholesky decomposition `A = L * L^T` for 3x3 SPD matrix `A`.
pub fn mat3_cholesky_lower(a: &Mat3) -> Option<Mat3> {
    let mut l = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if !sum.is_finite() || sum <= 1e-18 {
                    return None;
                }
                l[i][j] = sqrt(sum);
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Some(l)
}

/// Solve `L * x = b` where `L` is 3x3 lower-triangular.
pub fn mat3_solve_lower(l: &Mat3, b: Vec3) -> Vec3 {
    let x0 = b[0] / l[0][0];
    let x1 = (b[1] - l[1][0] * x0) / l[1][1];
    let x2 = (b[2] - l[2][0] * x0 - l[2][1] * x1) / l[2][2];
    [x0, x1, x2]
}

/// Deterministic `SplitMix64` PRNG for seeded turbulence and sensor simulation.
#[derive(Debug, Clone, Copy)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform float in `(0.0, 1.0)`.
    pub fn next_f64_unit(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        ((bits as f64) + 0.5) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Standard Gaussian pair `(z0, z1)` via deterministic `libm` Box-Muller transform.
    pub fn next_normal_pair(&mut self) -> (f64, f64) {
        let u1 = self.next_f64_unit().max(1e-15);
        let u2 = self.next_f64_unit();
        let r = sqrt(-2.0 * ln(u1));
        let theta = 2.0 * core::f64::consts::PI * u2;
        (r * cos(theta), r * sin(theta))
    }
}
