/**
 * Exact OpenXVS CONVENTIONS.md v1.0 (§1–§5) Coordinate & Projection Module.
 *
 * - SI units internally (metres, seconds, radians).
 * - Frames strictly: ECEF, NED, BODY (FRD), CAM (OpenCV: +x right, +y down, +z forward).
 * - Zero internal ENU.
 * - Hamilton quaternions (w, x, y, z), unit norm, canonical w >= 0.
 * - Intrinsic Z-Y-X Euler: R_ned_body = Rz(yaw) @ Ry(pitch) @ Rx(roll).
 * - Integer pixel centers (u, v); image spans [-0.5, W-0.5] x [-0.5, H-0.5].
 * - WebGPU NDC maps pixel center (u + 0.5, v + 0.5):
 *     x_ndc = 2.0 * (u + 0.5) / W - 1.0
 *     y_ndc = 1.0 - 2.0 * (v + 0.5) / H
 *     z_ndc = (zFar / (zFar - zNear)) * (1.0 - zNear / v_cam.z)  in [0, 1]
 */

export const WGS84_A = 6378137.0;
export const WGS84_F = 1.0 / 298.257223563;
export const WGS84_B = WGS84_A * (1.0 - WGS84_F);
export const WGS84_E2 = WGS84_F * (2.0 - WGS84_F);

export const DEG_TO_RAD = Math.PI / 180.0;
export const RAD_TO_DEG = 180.0 / Math.PI;

export type Vec3 = readonly [number, number, number];
/** Row-major 3x3 matrix stored as flat 9-element tuple. */
export type Mat3RowMajor = readonly [
  number, number, number,
  number, number, number,
  number, number, number,
];
/** Hamilton quaternion (w, x, y, z) with w >= 0. */
export type QuatHamilton = readonly [number, number, number, number];

export type RigCameraId = "F" | "R" | "B" | "L" | "U" | "D";

export const RIG_CAMERA_ORDER: readonly RigCameraId[] = ["F", "R", "B", "L", "U", "D"] as const;

export interface PinholeIntrinsics {
  readonly width: number;
  readonly height: number;
  readonly fx: number;
  readonly fy: number;
  readonly cx: number;
  readonly cy: number;
}

export interface ViewportSpec {
  readonly width: number;
  readonly height: number;
  readonly zNear?: number;
  readonly zFar?: number;
}

export interface RigCameraSpec {
  readonly id: RigCameraId;
  readonly name: string;
  readonly leverArmM: Vec3;
  /** Row-major 3x3 R_body_cam: v_body = R_body_cam @ v_cam. */
  readonly rBodyCam: Mat3RowMajor;
  readonly boresightBody: Vec3;
}

export interface ProjectionResult {
  readonly xNdc: number;
  readonly yNdc: number;
  readonly zNdc: number;
  readonly uPx: number;
  readonly vPx: number;
  readonly vCam: Vec3;
  /** True when v_cam.z > 0. */
  readonly inFront: boolean;
  /** True when inFront and (uPx, vPx) lies within [-0.5, W-0.5] x [-0.5, H-0.5]. */
  readonly visible: boolean;
}

/** Exact XVS6-CUBE-v1 intrinsics from docs/xvs/spec/rig_nominal.json. */
export const XVS6_CUBE_V1_INTRINSICS: PinholeIntrinsics = {
  width: 1024,
  height: 1024,
  fx: 429.6190111627674,
  fy: 429.6190111627674,
  cx: 511.5,
  cy: 511.5,
} as const;

/** Exact XVS6-CUBE-v1 6-camera rig strictly in F, R, B, L, U, D order. */
export const XVS6_CUBE_V1_CAMERAS: readonly RigCameraSpec[] = [
  {
    id: "F",
    name: "forward",
    leverArmM: [13.2, 0.0, 0.6],
    rBodyCam: [0, 0, 1, 1, 0, 0, 0, 1, 0],
    boresightBody: [1, 0, 0],
  },
  {
    id: "R",
    name: "right",
    leverArmM: [2.5, 1.55, -0.3],
    rBodyCam: [-1, 0, 0, 0, 0, 1, 0, 1, 0],
    boresightBody: [0, 1, 0],
  },
  {
    id: "B",
    name: "back",
    leverArmM: [-14.8, 0.0, -0.9],
    rBodyCam: [0, 0, -1, -1, 0, 0, 0, 1, 0],
    boresightBody: [-1, 0, 0],
  },
  {
    id: "L",
    name: "left",
    leverArmM: [2.5, -1.55, -0.3],
    rBodyCam: [1, 0, 0, 0, 0, -1, 0, 1, 0],
    boresightBody: [0, -1, 0],
  },
  {
    id: "U",
    name: "up",
    leverArmM: [-2.0, 0.0, -1.75],
    rBodyCam: [0, -1, 0, -1, 0, 0, 0, 0, -1],
    boresightBody: [0, 0, -1],
  },
  {
    id: "D",
    name: "down",
    leverArmM: [1.0, 0.0, 1.75],
    rBodyCam: [0, -1, 0, 1, 0, 0, 0, 0, 1],
    boresightBody: [0, 0, 1],
  },
] as const;

/** Exact XVS-TRAFFIC-RING-8-v1 intrinsics from docs/xvs/spec/traffic_ring_v1.json. */
export const TRAFFIC_RING_V1_INTRINSICS: PinholeIntrinsics = {
  width: 2448,
  height: 2048,
  fx: 2319.0,
  fy: 2319.0,
  cx: 1223.5,
  cy: 1023.5,
} as const;

export const TRAFFIC_RING_V1_IFOV_DEG_PER_PX = RAD_TO_DEG / TRAFFIC_RING_V1_INTRINSICS.fx;

/** Build an ideal square-pixel pinhole intrinsics object for arbitrary viewport & horizontal FOV. */
export function buildPinholeIntrinsics(
  width: number,
  height: number,
  fovHDeg: number,
): PinholeIntrinsics {
  const f = (width * 0.5) / Math.tan((fovHDeg * DEG_TO_RAD) * 0.5);
  return {
    width,
    height,
    fx: f,
    fy: f,
    cx: (width - 1) * 0.5,
    cy: (height - 1) * 0.5,
  };
}

/** Build shifted intrinsics for a sub-crop ROI starting at top-left (roiU0, roiV0). */
export function buildRoiIntrinsics(
  fullIntr: PinholeIntrinsics,
  roiU0: number,
  roiV0: number,
  roiWidth: number,
  roiHeight: number,
): PinholeIntrinsics {
  return {
    width: roiWidth,
    height: roiHeight,
    fx: fullIntr.fx,
    fy: fullIntr.fy,
    cx: fullIntr.cx - roiU0,
    cy: fullIntr.cy - roiV0,
  };
}

/** Geodetic (latRad, lonRad, hM above WGS-84 ellipsoid) -> ECEF metres. */
export function geodeticToEcef(latRad: number, lonRad: number, hM: number): [number, number, number] {
  const sinLat = Math.sin(latRad);
  const cosLat = Math.cos(latRad);
  const sinLon = Math.sin(lonRad);
  const cosLon = Math.cos(lonRad);
  const N = WGS84_A / Math.sqrt(1.0 - WGS84_E2 * sinLat * sinLat);
  const x = (N + hM) * cosLat * cosLon;
  const y = (N + hM) * cosLat * sinLon;
  const z = (N * (1.0 - WGS84_E2) + hM) * sinLat;
  return [x, y, z];
}

/** Iterative Bowring ECEF -> geodetic (latRad, lonRad, hM), round-trip < 1e-6 m. */
export function ecefToGeodetic(x: number, y: number, z: number): {
  latRad: number;
  lonRad: number;
  hM: number;
} {
  const lonRad = Math.atan2(y, x);
  const p = Math.hypot(x, y);
  let latRad = Math.atan2(z, p * (1.0 - WGS84_E2));
  for (let i = 0; i < 8; i++) {
    const sinLat = Math.sin(latRad);
    const N = WGS84_A / Math.sqrt(1.0 - WGS84_E2 * sinLat * sinLat);
    const nextLat = Math.atan2(z + WGS84_E2 * N * sinLat, p);
    if (Math.abs(nextLat - latRad) < 1e-15) {
      latRad = nextLat;
      break;
    }
    latRad = nextLat;
  }
  const sinLat = Math.sin(latRad);
  const cosLat = Math.cos(latRad);
  const N = WGS84_A / Math.sqrt(1.0 - WGS84_E2 * sinLat * sinLat);
  const hM = Math.abs(cosLat) > 1e-10 ? p / cosLat - N : Math.abs(z) / Math.abs(sinLat) - N * (1.0 - WGS84_E2);
  return { latRad, lonRad, hM };
}

/** R_ecef_ned = [n e d] columns in row-major 3x3 (CONVENTIONS.md §3). */
export function rEcefNed(latRad: number, lonRad: number): Mat3RowMajor {
  const sLat = Math.sin(latRad);
  const cLat = Math.cos(latRad);
  const sLon = Math.sin(lonRad);
  const cLon = Math.cos(lonRad);
  return [
    -sLat * cLon, -sLon, -cLat * cLon,
    -sLat * sLon,  cLon, -cLat * sLon,
     cLat,         0.0,  -sLat,
  ];
}

/** Intrinsic Z-Y-X Euler (yaw, pitch, roll) -> R_ned_body (CONVENTIONS.md §4). */
export function eulerZyxToRotMat(yawRad: number, pitchRad: number, rollRad: number): Mat3RowMajor {
  const cy = Math.cos(yawRad);
  const sy = Math.sin(yawRad);
  const cp = Math.cos(pitchRad);
  const sp = Math.sin(pitchRad);
  const cr = Math.cos(rollRad);
  const sr = Math.sin(rollRad);

  return [
    cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
    sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
    -sp,     cp * sr,                cp * cr,
  ];
}

/** Intrinsic Z-Y-X Euler -> canonical Hamilton quaternion (w, x, y, z) with w >= 0. */
export function eulerZyxToQuat(yawRad: number, pitchRad: number, rollRad: number): QuatHamilton {
  const cy = Math.cos(yawRad * 0.5);
  const sy = Math.sin(yawRad * 0.5);
  const cp = Math.cos(pitchRad * 0.5);
  const sp = Math.sin(pitchRad * 0.5);
  const cr = Math.cos(rollRad * 0.5);
  const sr = Math.sin(rollRad * 0.5);

  let w = cr * cp * cy + sr * sp * sy;
  let x = sr * cp * cy - cr * sp * sy;
  let y = cr * sp * cy + sr * cp * sy;
  let z = cr * cp * sy - sr * sp * cy;
  const norm = Math.hypot(w, x, y, z) || 1.0;
  w /= norm;
  x /= norm;
  y /= norm;
  z /= norm;
  if (w < 0.0) {
    return [-w, -x, -y, -z];
  }
  return [w, x, y, z];
}

/** Hamilton quaternion (w, x, y, z) -> row-major 3x3 rotation matrix. */
export function quatToRotMat(q: QuatHamilton): Mat3RowMajor {
  const [w, x, y, z] = q;
  const xx = x * x;
  const yy = y * y;
  const zz = z * z;
  const xy = x * y;
  const xz = x * z;
  const yz = y * z;
  const wx = w * x;
  const wy = w * y;
  const wz = w * z;

  return [
    1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy),
    2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
    2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy),
  ];
}

/** Row-major 3x3 matrix multiplication A @ B. */
export function mat3Mul(a: Mat3RowMajor, b: Mat3RowMajor): Mat3RowMajor {
  return [
    a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
    a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
    a[0] * b[2] + a[1] * b[5] + a[2] * b[8],

    a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
    a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
    a[3] * b[2] + a[4] * b[5] + a[5] * b[8],

    a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
    a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
    a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
  ];
}

/** Row-major 3x3 transpose. */
export function mat3Transpose(m: Mat3RowMajor): Mat3RowMajor {
  return [
    m[0], m[3], m[6],
    m[1], m[4], m[7],
    m[2], m[5], m[8],
  ];
}

/** Row-major 3x3 matrix times 3-vector: R @ v. */
export function mat3MulVec(m: Mat3RowMajor, v: Vec3): [number, number, number] {
  return [
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
    m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
  ];
}

/** Row-major 3x3 transpose times 3-vector: R^T @ v. */
export function mat3TransposeMulVec(m: Mat3RowMajor, v: Vec3): [number, number, number] {
  return [
    m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
    m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
    m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
  ];
}

/** SO(3) exponential map Exp(omega) via Rodrigues formula (CONVENTIONS.md §4). */
export function so3Exp(omega: Vec3): Mat3RowMajor {
  const [wx, wy, wz] = omega;
  const theta = Math.hypot(wx, wy, wz);
  if (theta < 1e-12) {
    return [
      1.0, -wz, wy,
      wz, 1.0, -wx,
      -wy, wx, 1.0,
    ];
  }
  const s = Math.sin(theta) / theta;
  const c = (1.0 - Math.cos(theta)) / (theta * theta);
  const wx2 = wx * wx;
  const wy2 = wy * wy;
  const wz2 = wz * wz;
  const wxy = wx * wy;
  const wxz = wx * wz;
  const wyz = wy * wz;
  return [
    1.0 - c * (wy2 + wz2), c * wxy - s * wz,      c * wxz + s * wy,
    c * wxy + s * wz,      1.0 - c * (wx2 + wz2), c * wyz - s * wx,
    c * wxz - s * wy,      c * wyz + s * wx,      1.0 - c * (wx2 + wy2),
  ];
}

/**
 * Compute camera ECEF pose from aircraft body ECEF pose and rig camera spec:
 *   R_ecef_cam = R_ecef_body @ R_body_cam
 *   p_ecef_cam = p_ecef_body + R_ecef_body @ lever_arm_body
 */
export function computeRigCameraPose(
  pEcefBody: Vec3,
  rEcefBody: Mat3RowMajor,
  camSpec: RigCameraSpec,
  perturbationCamRad?: Vec3,
): { pEcefCam: [number, number, number]; rEcefCam: Mat3RowMajor } {
  const leverEcef = mat3MulVec(rEcefBody, camSpec.leverArmM);
  const pEcefCam: [number, number, number] = [
    pEcefBody[0] + leverEcef[0],
    pEcefBody[1] + leverEcef[1],
    pEcefBody[2] + leverEcef[2],
  ];
  let rBodyCam = camSpec.rBodyCam;
  if (perturbationCamRad) {
    rBodyCam = mat3Mul(rBodyCam, so3Exp(perturbationCamRad));
  }
  const rEcefCam = mat3Mul(rEcefBody, rBodyCam);
  return { pEcefCam, rEcefCam };
}

/**
 * Build a free-look head-tracked camera orientation in BODY frame from pan (yaw right, rad)
 * and tilt (pitch up, rad), starting from Forward camera boresight (+x BODY).
 */
export function buildFreeLookRBodyCam(panRad: number, tiltRad: number): Mat3RowMajor {
  const rBodyCamF = XVS6_CUBE_V1_CAMERAS[0].rBodyCam;
  // Pan right = +yaw around BODY z (down); tilt up = +pitch around BODY y (right)
  const rHead = eulerZyxToRotMat(panRad, tiltRad, 0.0);
  return mat3Mul(rHead, rBodyCamF);
}

/**
 * Exact OpenCV CAM -> WebGPU NDC projection in f64 RTE (SP1_LIVEDECK_DESIGN.md §4.4).
 *
 * 1. v_cam = R_ecef_cam^T @ (p_ecef_obj - p_ecef_cam)  [f64 RTE]
 * 2. Clip v_cam.z <= 0 (returns visible=false, inFront=false, NaN coords).
 * 3. CONVENTIONS.md §5 pinhole pixels (integer pixel centers):
 *      u = fx * (v_cam.x / v_cam.z) + cx
 *      v = fy * (v_cam.y / v_cam.z) + cy
 * 4. WebGPU NDC accounting for pixel center (u + 0.5, v + 0.5):
 *      x_ndc = 2.0 * (u + 0.5) / W - 1.0
 *      y_ndc = 1.0 - 2.0 * (v + 0.5) / H
 *      z_ndc = (zFar / (zFar - zNear)) * (1.0 - zNear / v_cam.z)
 */
export function projectEcefToNdc(
  pEcefObj: Vec3,
  pEcefCam: Vec3,
  rEcefCam: Mat3RowMajor,
  intr: PinholeIntrinsics,
  viewport: ViewportSpec,
): ProjectionResult {
  const dx = pEcefObj[0] - pEcefCam[0];
  const dy = pEcefObj[1] - pEcefCam[1];
  const dz = pEcefObj[2] - pEcefCam[2];

  const vCam = mat3TransposeMulVec(rEcefCam, [dx, dy, dz]);
  const z = vCam[2];
  if (!Number.isFinite(z) || z <= 0.0) {
    return {
      xNdc: Number.NaN,
      yNdc: Number.NaN,
      zNdc: Number.NaN,
      uPx: Number.NaN,
      vPx: Number.NaN,
      vCam,
      inFront: false,
      visible: false,
    };
  }

  const uPx = intr.fx * (vCam[0] / z) + intr.cx;
  const vPx = intr.fy * (vCam[1] / z) + intr.cy;

  const W = viewport.width;
  const H = viewport.height;
  const zNear = viewport.zNear ?? 1.0;
  const zFar = viewport.zFar ?? 200000.0;

  const xNdc = (2.0 * (uPx + 0.5)) / W - 1.0;
  const yNdc = 1.0 - (2.0 * (vPx + 0.5)) / H;
  const zNdc = (zFar / (zFar - zNear)) * (1.0 - zNear / z);

  const visible =
    uPx >= -0.5 &&
    uPx <= W - 0.5 &&
    vPx >= -0.5 &&
    vPx <= H - 0.5 &&
    Number.isFinite(xNdc) &&
    Number.isFinite(yNdc);

  return {
    xNdc,
    yNdc,
    zNdc,
    uPx,
    vPx,
    vCam,
    inFront: true,
    visible,
  };
}

/**
 * Convert WebGPU NDC back into CONVENTIONS.md §5 integer-center pixel coordinates (u, v).
 */
export function ndcToPixelCenter(
  xNdc: number,
  yNdc: number,
  viewport: ViewportSpec,
): { uPx: number; vPx: number } {
  const uPx = (xNdc + 1.0) * 0.5 * viewport.width - 0.5;
  const vPx = (1.0 - yNdc) * 0.5 * viewport.height - 0.5;
  return { uPx, vPx };
}

/**
 * Build column-major 4x4 View-Projection matrix mapping RTE ECEF vector
 * `d_ecef = p_ecef_obj - p_ecef_cam` directly into WebGPU clip space `(x, y, z, w)`
 * where `clip.xyz / clip.w` matches `projectEcefToNdc` identically (z in [0, 1]).
 *
 * Naga's GLSL ES 3.00 transpiler automatically adjusts clip space for WebGL2.
 */
export function buildCamViewProjMatrixF64(
  rEcefCam: Mat3RowMajor,
  intr: PinholeIntrinsics,
  viewport: ViewportSpec,
): Float64Array {
  const W = viewport.width;
  const H = viewport.height;
  const zNear = viewport.zNear ?? 1.0;
  const zFar = viewport.zFar ?? 200000.0;

  // OpenCV CAM -> WebGPU clip space:
  //   x_clip = a00 * vx + a02 * vz
  //   y_clip = a11 * vy + a12 * vz
  //   z_clip = a22 * vz + a23 * 1
  //   w_clip = vz
  const a00 = (2.0 * intr.fx) / W;
  const a02 = (2.0 * (intr.cx + 0.5)) / W - 1.0;
  const a11 = (-2.0 * intr.fy) / H;
  const a12 = 1.0 - (2.0 * (intr.cy + 0.5)) / H;
  const a22 = zFar / (zFar - zNear);
  const a23 = -(zFar * zNear) / (zFar - zNear);

  // V = R_ecef_cam^T (rows of V are columns of R_ecef_cam)
  // Row 0 of V: [r0, r3, r6]  -> vx
  // Row 1 of V: [r1, r4, r7]  -> vy
  // Row 2 of V: [r2, r5, r8]  -> vz
  const r = rEcefCam;

  // M = P * V, stored in column-major order for WGSL mat4x4<f32> / wgpu-matrix
  const out = new Float64Array(16);

  // Column 0 (multiplies d_ecef.x)
  out[0] = a00 * r[0] + a02 * r[2];
  out[1] = a11 * r[1] + a12 * r[2];
  out[2] = a22 * r[2];
  out[3] = r[2];

  // Column 1 (multiplies d_ecef.y)
  out[4] = a00 * r[3] + a02 * r[5];
  out[5] = a11 * r[4] + a12 * r[5];
  out[6] = a22 * r[5];
  out[7] = r[5];

  // Column 2 (multiplies d_ecef.z)
  out[8] = a00 * r[6] + a02 * r[8];
  out[9] = a11 * r[7] + a12 * r[8];
  out[10] = a22 * r[8];
  out[11] = r[8];

  // Column 3 (translation / constant term)
  out[12] = 0.0;
  out[13] = 0.0;
  out[14] = a23;
  out[15] = 0.0;

  return out;
}

/** Float32 version of `buildCamViewProjMatrixF64` ready for GPU uniform upload. */
export function buildCamViewProjMatrix(
  rEcefCam: Mat3RowMajor,
  intr: PinholeIntrinsics,
  viewport: ViewportSpec,
): Float32Array {
  const f64 = buildCamViewProjMatrixF64(rEcefCam, intr, viewport);
  return new Float32Array(f64);
}

/** Multiply column-major 4x4 matrix by [x, y, z, 1] -> [cx, cy, cz, cw]. */
export function mulMat4ColMajorVec3(
  m: Float32Array | Float64Array,
  v: Vec3,
): [number, number, number, number] {
  const [x, y, z] = v;
  return [
    m[0] * x + m[4] * y + m[8] * z + m[12],
    m[1] * x + m[5] * y + m[9] * z + m[13],
    m[2] * x + m[6] * y + m[10] * z + m[14],
    m[3] * x + m[7] * y + m[11] * z + m[15],
  ];
}
