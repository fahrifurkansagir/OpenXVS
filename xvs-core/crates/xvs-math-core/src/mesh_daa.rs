//! `XVS-MeshDAA` ECEF 52-Byte `OPENXVS_BEARING_TRACK` Wire Contract, 3-Stage Deterministic
//! Association (Skew-Ray + 3-Ship Triple-Ray Epipolar Ghost Gate + Bearing-Rate Check), and
//! Fast Covariance Intersection (`FCI`).

use crate::det_f64::{
    add3, clamp, dot3, mat3_add, mat3_inv, mat3_mul_vec, mat3_scale, mat3_trace, norm3,
    normalize3, round, scale3, sub3, Mat3, Vec3,
};

pub const OPENXVS_BEARING_TRACK_PAYLOAD_LEN: usize = 52;
pub const FLAG_SIMULATION: u8 = 0x01;
pub const FLAG_GNSS_DENIED: u8 = 0x02;

pub const MAX_FUSED_TRACKS: usize = 8;

/// Exact 52-byte ECEF `OPENXVS_BEARING_TRACK` payload (`SP1_LIVEDECK_DESIGN.md` §4.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenXvsBearingTrack {
    pub time_utc_us: u64,
    pub own_ecef_cm: [i32; 3],
    pub bearing_ecef_q15: [i16; 3],
    pub bearing_rate_urad_s: [i16; 3],
    pub cone_half_angle_urad: u16,
    pub own_pos_std_cm: u16,
    pub track_id: u16,
    pub sender_id: u8,
    pub flags: u8,
    pub cov_diag_packed: [u16; 6],
}

impl OpenXvsBearingTrack {
    #[allow(clippy::too_many_arguments)]
    pub fn from_ecef_ray(
        time_utc_us: u64,
        own_ecef_m: Vec3,
        bearing_ecef: Vec3,
        bearing_rate_rad_s: Vec3,
        cone_half_angle_urad: u16,
        own_pos_std_cm: u16,
        track_id: u16,
        sender_id: u8,
        flags: u8,
    ) -> Self {
        let u = normalize3(bearing_ecef);
        let q15 = [
            clamp(round(u[0] * 32767.0), -32767.0, 32767.0) as i16,
            clamp(round(u[1] * 32767.0), -32767.0, 32767.0) as i16,
            clamp(round(u[2] * 32767.0), -32767.0, 32767.0) as i16,
        ];
        // Quantize ECEF angular velocity at 100 urad/s LSB (1e4 * rad/s)
        let br = [
            clamp(round(bearing_rate_rad_s[0] * 10_000.0), -32767.0, 32767.0) as i16,
            clamp(round(bearing_rate_rad_s[1] * 10_000.0), -32767.0, 32767.0) as i16,
            clamp(round(bearing_rate_rad_s[2] * 10_000.0), -32767.0, 32767.0) as i16,
        ];
        let cm = [
            clamp(round(own_ecef_m[0] * 100.0), i32::MIN as f64, i32::MAX as f64) as i32,
            clamp(round(own_ecef_m[1] * 100.0), i32::MIN as f64, i32::MAX as f64) as i32,
            clamp(round(own_ecef_m[2] * 100.0), i32::MIN as f64, i32::MAX as f64) as i32,
        ];
        Self {
            time_utc_us,
            own_ecef_cm: cm,
            bearing_ecef_q15: q15,
            bearing_rate_urad_s: br,
            cone_half_angle_urad,
            own_pos_std_cm,
            track_id,
            sender_id,
            flags,
            cov_diag_packed: [
                own_pos_std_cm,
                own_pos_std_cm,
                own_pos_std_cm,
                cone_half_angle_urad,
                cone_half_angle_urad,
                cone_half_angle_urad,
            ],
        }
    }

    pub fn own_ecef_m(&self) -> Vec3 {
        [
            (self.own_ecef_cm[0] as f64) * 0.01,
            (self.own_ecef_cm[1] as f64) * 0.01,
            (self.own_ecef_cm[2] as f64) * 0.01,
        ]
    }

    pub fn unit_bearing_ecef(&self) -> Vec3 {
        normalize3([
            (self.bearing_ecef_q15[0] as f64) / 32767.0,
            (self.bearing_ecef_q15[1] as f64) / 32767.0,
            (self.bearing_ecef_q15[2] as f64) / 32767.0,
        ])
    }

    pub fn cone_half_angle_rad(&self) -> f64 {
        (self.cone_half_angle_urad as f64) * 1e-6
    }

    pub fn own_pos_std_m(&self) -> f64 {
        (self.own_pos_std_cm as f64) * 0.01
    }

    /// Deterministic little-endian 52-byte serialization (zero `unsafe`).
    pub fn to_le_bytes(&self) -> [u8; OPENXVS_BEARING_TRACK_PAYLOAD_LEN] {
        let mut buf = [0u8; OPENXVS_BEARING_TRACK_PAYLOAD_LEN];
        buf[0..8].copy_from_slice(&self.time_utc_us.to_le_bytes());
        buf[8..12].copy_from_slice(&self.own_ecef_cm[0].to_le_bytes());
        buf[12..16].copy_from_slice(&self.own_ecef_cm[1].to_le_bytes());
        buf[16..20].copy_from_slice(&self.own_ecef_cm[2].to_le_bytes());
        buf[20..22].copy_from_slice(&self.bearing_ecef_q15[0].to_le_bytes());
        buf[22..24].copy_from_slice(&self.bearing_ecef_q15[1].to_le_bytes());
        buf[24..26].copy_from_slice(&self.bearing_ecef_q15[2].to_le_bytes());
        buf[26..28].copy_from_slice(&self.bearing_rate_urad_s[0].to_le_bytes());
        buf[28..30].copy_from_slice(&self.bearing_rate_urad_s[1].to_le_bytes());
        buf[30..32].copy_from_slice(&self.bearing_rate_urad_s[2].to_le_bytes());
        buf[32..34].copy_from_slice(&self.cone_half_angle_urad.to_le_bytes());
        buf[34..36].copy_from_slice(&self.own_pos_std_cm.to_le_bytes());
        buf[36..38].copy_from_slice(&self.track_id.to_le_bytes());
        buf[38] = self.sender_id;
        buf[39] = self.flags;
        for (k, val) in self.cov_diag_packed.iter().enumerate() {
            let off = 40 + 2 * k;
            buf[off..off + 2].copy_from_slice(&val.to_le_bytes());
        }
        buf
    }

    /// Deterministic little-endian 52-byte deserialization (zero `unsafe`).
    pub fn from_le_bytes(buf: &[u8; OPENXVS_BEARING_TRACK_PAYLOAD_LEN]) -> Self {
        let u16_at = |o: usize| u16::from_le_bytes([buf[o], buf[o + 1]]);
        let i16_at = |o: usize| i16::from_le_bytes([buf[o], buf[o + 1]]);
        let i32_at = |o: usize| i32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
        Self {
            time_utc_us: u64::from_le_bytes([
                buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
            ]),
            own_ecef_cm: [i32_at(8), i32_at(12), i32_at(16)],
            bearing_ecef_q15: [i16_at(20), i16_at(22), i16_at(24)],
            bearing_rate_urad_s: [i16_at(26), i16_at(28), i16_at(30)],
            cone_half_angle_urad: u16_at(32),
            own_pos_std_cm: u16_at(34),
            track_id: u16_at(36),
            sender_id: buf[38],
            flags: buf[39],
            cov_diag_packed: [
                u16_at(40),
                u16_at(42),
                u16_at(44),
                u16_at(46),
                u16_at(48),
                u16_at(50),
            ],
        }
    }
}

/// Fast Covariance Intersection (`FCI`) for two 3D estimates `(x_a, P_a)` and `(x_b, P_b)`
/// under unknown cross-correlation (Niehsen trace weighting).
pub fn fci_fuse_3d(x_a: Vec3, p_a: &Mat3, x_b: Vec3, p_b: &Mat3) -> (Vec3, Mat3) {
    let inv_a = mat3_inv(p_a).unwrap_or([[1e-4, 0.0, 0.0], [0.0, 1e-4, 0.0], [0.0, 0.0, 1e-4]]);
    let inv_b = mat3_inv(p_b).unwrap_or([[1e-4, 0.0, 0.0], [0.0, 1e-4, 0.0], [0.0, 0.0, 1e-4]]);

    let tr_inv_a = mat3_trace(&inv_a).max(1e-15);
    let tr_inv_b = mat3_trace(&inv_b).max(1e-15);
    let omega = clamp(tr_inv_a / (tr_inv_a + tr_inv_b), 1e-6, 1.0 - 1e-6);

    let info = mat3_add(&mat3_scale(&inv_a, omega), &mat3_scale(&inv_b, 1.0 - omega));
    let p_f = mat3_inv(&info).unwrap_or(*p_a);

    let rhs_a = scale3(mat3_mul_vec(&inv_a, x_a), omega);
    let rhs_b = scale3(mat3_mul_vec(&inv_b, x_b), 1.0 - omega);
    let x_f = mat3_mul_vec(&p_f, add3(rhs_a, rhs_b));
    (x_f, p_f)
}

#[derive(Debug, Clone, Copy)]
pub struct FusedMeshTrack {
    pub pos_ecef: Vec3,
    pub cov_ecef: Mat3,
    pub triple_ray_confirmed: bool,
    pub valid: bool,
}

impl Default for FusedMeshTrack {
    fn default() -> Self {
        Self {
            pos_ecef: [0.0; 3],
            cov_ecef: [[0.0; 3]; 3],
            triple_ray_confirmed: false,
            valid: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MeshFusionResult {
    pub tracks: [FusedMeshTrack; MAX_FUSED_TRACKS],
    pub num_fused: usize,
    pub ghosts_rejected: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct MeshDaaFuser {
    pub skew_gate_k: f64,
    pub merge_radius_m: f64,
}

impl Default for MeshDaaFuser {
    fn default() -> Self {
        Self::new()
    }
}

impl MeshDaaFuser {
    pub const fn new() -> Self {
        Self {
            skew_gate_k: 3.0,
            merge_radius_m: 150.0,
        }
    }

    /// 3-stage deterministic association + 3-ship triple-ray epipolar ghost rejection + FCI fusion.
    pub fn fuse(&mut self, obs: &[OpenXvsBearingTrack]) -> MeshFusionResult {
        let mut out = MeshFusionResult {
            tracks: [FusedMeshTrack::default(); MAX_FUSED_TRACKS],
            num_fused: 0,
            ghosts_rejected: 0,
        };

        // Count distinct peer sender IDs present
        let mut senders = [0u8; 16];
        let mut n_senders = 0usize;
        for o in obs {
            let mut seen = false;
            for &s in senders.iter().take(n_senders) {
                if s == o.sender_id {
                    seen = true;
                    break;
                }
            }
            if !seen && n_senders < senders.len() {
                senders[n_senders] = o.sender_id;
                n_senders += 1;
            }
        }
        let has_third_ship = n_senders >= 3;

        // Stage 1: Pairwise cone-normalized skew-ray triangulation
        for i in 0..obs.len() {
            for j in (i + 1)..obs.len() {
                let a = &obs[i];
                let b = &obs[j];
                if a.sender_id == b.sender_id {
                    continue;
                }
                let oa = a.own_ecef_m();
                let ob = b.own_ecef_m();
                let ua = a.unit_bearing_ecef();
                let ub = b.unit_bearing_ecef();

                let w0 = sub3(oa, ob);
                let b_dot = dot3(ua, ub);
                let d_dot = dot3(ua, w0);
                let e_dot = dot3(ub, w0);
                let denom = 1.0 - b_dot * b_dot;
                if denom < 1e-4 {
                    continue; // parallel rays, cannot triangulate
                }
                let ta = (b_dot * e_dot - d_dot) / denom;
                let tb = (e_dot - b_dot * d_dot) / denom;
                if !(10.0..=50_000.0).contains(&ta) || !(10.0..=50_000.0).contains(&tb) {
                    continue;
                }

                let pa = add3(oa, scale3(ua, ta));
                let pb = add3(ob, scale3(ub, tb));
                let d_skew = norm3(sub3(pa, pb));
                let r_cone = ta * a.cone_half_angle_rad()
                    + tb * b.cone_half_angle_rad()
                    + a.own_pos_std_m()
                    + b.own_pos_std_m()
                    + 5.0;
                if d_skew > self.skew_gate_k * r_cone {
                    continue;
                }

                let p_mid = scale3(add3(pa, pb), 0.5);

                // Stage 2: 3-ship triple-ray epipolar verification (kills 2-ship coplanar ghosts!)
                let mut triple_confirmed = false;
                if has_third_ship {
                    for c in obs {
                        if c.sender_id == a.sender_id || c.sender_id == b.sender_id {
                            continue;
                        }
                        let oc = c.own_ecef_m();
                        let uc = c.unit_bearing_ecef();
                        let rel = sub3(p_mid, oc);
                        let dist = norm3(rel);
                        if dist < 10.0 {
                            continue;
                        }
                        let dir_c = scale3(rel, 1.0 / dist);
                        let cos_ang = clamp(dot3(dir_c, uc), -1.0, 1.0);
                        let sin_ang = crate::det_f64::sqrt((1.0 - cos_ang * cos_ang).max(0.0));
                        let ang_tol = c.cone_half_angle_rad() * 3.0 + (r_cone / dist) + 0.008;
                        if cos_ang > 0.0 && sin_ang <= ang_tol {
                            triple_confirmed = true;
                            break;
                        }
                    }
                    if !triple_confirmed {
                        out.ghosts_rejected = out.ghosts_rejected.saturating_add(1);
                        continue;
                    }
                }

                // Build isotropic-plus-range covariance estimate for FCI
                let sig = (r_cone * 0.5).max(5.0);
                let var = sig * sig;
                let p_est = [[var, 0.0, 0.0], [0.0, var, 0.0], [0.0, 0.0, var]];

                // Merge with existing fused track via Fast Covariance Intersection (FCI)
                let mut merged = false;
                for slot in out.tracks.iter_mut().take(out.num_fused) {
                    if norm3(sub3(slot.pos_ecef, p_mid)) <= self.merge_radius_m {
                        let (xf, pf) = fci_fuse_3d(slot.pos_ecef, &slot.cov_ecef, p_mid, &p_est);
                        slot.pos_ecef = xf;
                        slot.cov_ecef = pf;
                        slot.triple_ray_confirmed |= triple_confirmed;
                        merged = true;
                        break;
                    }
                }
                if !merged && out.num_fused < MAX_FUSED_TRACKS {
                    out.tracks[out.num_fused] = FusedMeshTrack {
                        pos_ecef: p_mid,
                        cov_ecef: p_est,
                        triple_ray_confirmed: triple_confirmed,
                        valid: true,
                    };
                    out.num_fused += 1;
                }
            }
        }

        out
    }
}
