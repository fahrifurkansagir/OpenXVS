//! Exact `XVS6-CUBE-v1` (`docs/xvs/spec/rig_nominal.json`) and `XVS-TRAFFIC-RING-8-v1`
//! (`docs/xvs/spec/traffic_ring_v1.json`) pinhole camera rig definitions and ECEF projection.

use crate::det_f64::{add3, mat3_mul, mat3_mul_vec, mat3_t_mul_vec, normalize3, sub3, Mat3, Vec3};

/// Ideal pinhole camera intrinsics conforming to `CONVENTIONS.md` §5.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PinholeCamera {
    pub width: u32,
    pub height: u32,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

impl PinholeCamera {
    /// Project a 3D point in OpenCV camera frame (`+x` right, `+y` down, `+z` forward) to `(u, v)`.
    /// Strictly returns `None` whenever `p_cam[2] <= 0.0` or any coordinate is non-finite.
    #[inline]
    pub fn project_cam(&self, p_cam: Vec3) -> Option<(f64, f64)> {
        let [x, y, z] = p_cam;
        if !x.is_finite() || !y.is_finite() || !z.is_finite() || z <= 0.0 {
            return None;
        }
        let u = self.fx * (x / z) + self.cx;
        let v = self.fy * (y / z) + self.cy;
        if u.is_finite() && v.is_finite() {
            Some((u, v))
        } else {
            None
        }
    }

    /// Project a 3D ECEF point into pixel coordinates `(u, v)` given camera ECEF pose.
    /// Returns `None` if point is behind camera (`v_cam.z <= 0.0`).
    #[inline]
    pub fn project_ecef(
        &self,
        p_ecef: Vec3,
        p_ecef_cam: Vec3,
        r_ecef_cam: &Mat3,
    ) -> Option<(f64, f64)> {
        let dp = sub3(p_ecef, p_ecef_cam);
        let p_cam = mat3_t_mul_vec(r_ecef_cam, dp);
        self.project_cam(p_cam)
    }

    /// Back-project pixel `(u, v)` to unit ray in OpenCV camera frame.
    #[inline]
    pub fn unproject(&self, u: f64, v: f64) -> Vec3 {
        normalize3([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])
    }

    /// Check whether `(u, v)` lies within pixel-center image bounds `[-0.5, W-0.5] x [-0.5, H-0.5]`.
    #[inline]
    pub fn in_image(&self, u: f64, v: f64) -> bool {
        u.is_finite()
            && v.is_finite()
            && u >= -0.5
            && u <= (self.width as f64) - 0.5
            && v >= -0.5
            && v <= (self.height as f64) - 0.5
    }
}

/// Free-function helper matching Task 2 interface: returns `None` when `v_cam.z <= 0.0`.
#[inline]
pub fn project_ecef_to_cam(
    cam: &PinholeCamera,
    p_ecef: Vec3,
    p_ecef_cam: Vec3,
    r_ecef_cam: &Mat3,
) -> Option<(f64, f64)> {
    cam.project_ecef(p_ecef, p_ecef_cam, r_ecef_cam)
}

/// Single camera mount definition in aircraft FRD body frame (`v_body = R_body_cam @ v_cam`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigCamera {
    pub id: [u8; 2],
    pub lever_arm_m: Vec3,
    pub r_body_cam: Mat3,
    pub boresight_body: Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Xvs6CubeRig {
    pub intrinsics: PinholeCamera,
    pub cameras: [RigCamera; 6],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrafficRingRig {
    pub intrinsics: PinholeCamera,
    pub cameras: [RigCamera; 8],
}

/// Exact `XVS6-CUBE-v1` rig from `docs/xvs/spec/rig_nominal.json` (order: `F, R, B, L, U, D`).
pub const XVS6_CUBE_V1: Xvs6CubeRig = Xvs6CubeRig {
    intrinsics: PinholeCamera {
        width: 1024,
        height: 1024,
        fx: 429.619_011_162_767_4,
        fy: 429.619_011_162_767_4,
        cx: 511.5,
        cy: 511.5,
    },
    cameras: [
        // 0: F (forward)
        RigCamera {
            id: *b"F ",
            lever_arm_m: [13.2, 0.0, 0.6],
            r_body_cam: [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            boresight_body: [1.0, 0.0, 0.0],
        },
        // 1: R (right)
        RigCamera {
            id: *b"R ",
            lever_arm_m: [2.5, 1.55, -0.3],
            r_body_cam: [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            boresight_body: [0.0, 1.0, 0.0],
        },
        // 2: B (back)
        RigCamera {
            id: *b"B ",
            lever_arm_m: [-14.8, 0.0, -0.9],
            r_body_cam: [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            boresight_body: [-1.0, 0.0, 0.0],
        },
        // 3: L (left)
        RigCamera {
            id: *b"L ",
            lever_arm_m: [2.5, -1.55, -0.3],
            r_body_cam: [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            boresight_body: [0.0, -1.0, 0.0],
        },
        // 4: U (up)
        RigCamera {
            id: *b"U ",
            lever_arm_m: [-2.0, 0.0, -1.75],
            r_body_cam: [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
            boresight_body: [0.0, 0.0, -1.0],
        },
        // 5: D (down)
        RigCamera {
            id: *b"D ",
            lever_arm_m: [1.0, 0.0, 1.75],
            r_body_cam: [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            boresight_body: [0.0, 0.0, 1.0],
        },
    ],
};

const SQRT_HALF: f64 = core::f64::consts::FRAC_1_SQRT_2;

/// Exact `XVS-TRAFFIC-RING-8-v1` rig from `docs/xvs/spec/traffic_ring_v1.json` (`T0..T7`).
pub const TRAFFIC_RING_V1: TrafficRingRig = TrafficRingRig {
    intrinsics: PinholeCamera {
        width: 2448,
        height: 2048,
        fx: 2319.0,
        fy: 2319.0,
        cx: 1223.5,
        cy: 1023.5,
    },
    cameras: [
        RigCamera {
            id: *b"T0",
            lever_arm_m: [0.0, 0.0, -0.5],
            r_body_cam: [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            boresight_body: [1.0, 0.0, 0.0],
        },
        RigCamera {
            id: *b"T1",
            lever_arm_m: [0.0, 0.0, -0.5],
            r_body_cam: [[-SQRT_HALF, 0.0, SQRT_HALF], [SQRT_HALF, 0.0, SQRT_HALF], [0.0, 1.0, 0.0]],
            boresight_body: [SQRT_HALF, SQRT_HALF, 0.0],
        },
        RigCamera {
            id: *b"T2",
            lever_arm_m: [0.0, 0.0, -0.5],
            r_body_cam: [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            boresight_body: [0.0, 1.0, 0.0],
        },
        RigCamera {
            id: *b"T3",
            lever_arm_m: [0.0, 0.0, -0.5],
            r_body_cam: [[-SQRT_HALF, 0.0, -SQRT_HALF], [-SQRT_HALF, 0.0, SQRT_HALF], [0.0, 1.0, 0.0]],
            boresight_body: [-SQRT_HALF, SQRT_HALF, 0.0],
        },
        RigCamera {
            id: *b"T4",
            lever_arm_m: [0.0, 0.0, -0.5],
            r_body_cam: [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            boresight_body: [-1.0, 0.0, 0.0],
        },
        RigCamera {
            id: *b"T5",
            lever_arm_m: [0.0, 0.0, -0.5],
            r_body_cam: [[SQRT_HALF, 0.0, -SQRT_HALF], [-SQRT_HALF, 0.0, -SQRT_HALF], [0.0, 1.0, 0.0]],
            boresight_body: [-SQRT_HALF, -SQRT_HALF, 0.0],
        },
        RigCamera {
            id: *b"T6",
            lever_arm_m: [0.0, 0.0, -0.5],
            r_body_cam: [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            boresight_body: [0.0, -1.0, 0.0],
        },
        RigCamera {
            id: *b"T7",
            lever_arm_m: [0.0, 0.0, -0.5],
            r_body_cam: [[SQRT_HALF, 0.0, SQRT_HALF], [SQRT_HALF, 0.0, -SQRT_HALF], [0.0, 1.0, 0.0]],
            boresight_body: [SQRT_HALF, -SQRT_HALF, 0.0],
        },
    ],
};

/// Compute camera optical-center position and orientation `(p_ecef_cam, R_ecef_cam)` in ECEF.
#[inline]
pub fn camera_pose_ecef(p_own_ecef: Vec3, r_ecef_body: &Mat3, cam: &RigCamera) -> (Vec3, Mat3) {
    let lever_ecef = mat3_mul_vec(r_ecef_body, cam.lever_arm_m);
    let p_cam = add3(p_own_ecef, lever_ecef);
    let r_ec = mat3_mul(r_ecef_body, &cam.r_body_cam);
    (p_cam, r_ec)
}
