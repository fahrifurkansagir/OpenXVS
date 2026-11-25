// Pass 4: Sensor-Matched D1-KF Narrow-FOV (traffic_ring_v1, fx=fy=2319.0) WebGPU Compute Pipeline.
//
// Exact match to:
//   - src/xvs/sensor/srgb.py (IEC 61966-2-1 sRGB EOTF + Rec.709 weights 0.2126, 0.7152, 0.0722)
//   - configs/xvs/sensors/ring_mono_v0_nodist.json (full_well_e=10500, sigma_r_e=2.4, k_dn_per_e=0.35, bits=12)
//   - src/xvs/detection/warp.py (2-layer 6-DoF WGS-84 + sky warp, overlap mask M, M*(I_t - I_warped))
//   - src/xvs/detection/d1_model.py:285-289 (single FP32 normalize_input with std0 clamped >= 1.0, out clamped [-20, 20])
//
// Manual FP32 bilinear interpolation via textureLoad (NEVER hardware fixed-point sampler filtering!).

struct Pass4Params {
  // Intrinsics & ROI geometry
  intr: vec4<f32>,          // x=fx (2319), y=fy (2319), z=cx (1223.5), w=cy (1023.5)
  roiCurr: vec4<f32>,       // x=u0_curr, y=v0_curr, z=width (256), w=height (256)
  roiPrev1: vec4<f32>,      // x=u0_p1, y=v0_p1, z=u0_p2, w=v0_p2
  sensorCfg: vec4<f32>,     // x=fullWellE (10500), y=sigmaRE (2.4), z=kDnPerE (0.35), w=blackLevelDn (64)
  flags: vec4<f32>,         // x=noiseOff (1=off,0=on), y=applyPsf (1=on,0=off), z=frameSeed, w=darkCurrentE (0.005)
  // Precomputed f64 WGS-84 RTE scalars to eliminate f32 catastrophic cancellation
  cPrimeAndK: vec4<f32>,    // xyz = p_cam_curr / [a, a, b], w = dot_f64(cPrime, cPrime) - 1.0
  deltaP1: vec4<f32>,       // xyz = p_cam_curr - p_cam_prev1 (ECEF m), w = 0
  deltaP2: vec4<f32>,       // xyz = p_cam_curr - p_cam_prev2 (ECEF m), w = 0
  // Row-major 3x3 rotations stored as 3 vec4 rows
  rCurr0: vec4<f32>,
  rCurr1: vec4<f32>,
  rCurr2: vec4<f32>,
  rPrev1_0: vec4<f32>,
  rPrev1_1: vec4<f32>,
  rPrev1_2: vec4<f32>,
  rPrev2_0: vec4<f32>,
  rPrev2_1: vec4<f32>,
  rPrev2_2: vec4<f32>,
};

@group(0) @binding(0) var<uniform> p: Pass4Params;
@group(0) @binding(1) var texRoiRgba: texture_2d<f32>;
@group(0) @binding(2) var texPrev1: texture_2d<f32>;
@group(0) @binding(3) var texPrev2: texture_2d<f32>;
@group(0) @binding(4) var<storage, read_write> currElectrons: array<f32>;
@group(0) @binding(5) var<storage, read_write> rawStack: array<f32>;      // [5 * 65536]
@group(0) @binding(6) var<storage, read_write> statsBuf: array<f32>;      // [8]: mean[0..4], std0, pad, pad
@group(0) @binding(7) var<storage, read_write> normOut: array<f32>;       // [5 * 65536]

const WGS84_INV_A: f32 = 1.567855942887398e-7;
const WGS84_INV_B: f32 = 1.573130351105623e-7;
const TILE_W: u32 = 256u;
const TILE_H: u32 = 256u;
const TILE_N: u32 = 65536u;

fn srgbChannelToLinear(c: f32) -> f32 {
  let x = max(c, 0.0);
  if (x <= 0.04045) {
    return x / 12.92;
  }
  return pow((x + 0.055) / 1.055, 2.4);
}

fn sampleLinearLuma(ix: i32, iy: i32) -> f32 {
  let cx = clamp(ix, 0, i32(TILE_W) - 1);
  let cy = clamp(iy, 0, i32(TILE_H) - 1);
  let rgba = textureLoad(texRoiRgba, vec2<i32>(cx, cy), 0);
  let r = srgbChannelToLinear(rgba.r);
  let g = srgbChannelToLinear(rgba.g);
  let b = srgbChannelToLinear(rgba.b);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

fn psfWeight5x5(dx: i32, dy: i32) -> f32 {
  let ax = abs(dx);
  let ay = abs(dy);
  let hi = max(ax, ay);
  let lo = min(ax, ay);
  if (hi == 0 && lo == 0) { return 0.64551891; }
  if (hi == 1 && lo == 0) { return 0.06989676; }
  if (hi == 1 && lo == 1) { return 0.01243128; }
  if (hi == 2 && lo == 0) { return 0.00235286; }
  if (hi == 2 && lo == 1) { return 0.00160273; }
  return 0.00073390;
}

fn pcgHash(vIn: u32) -> u32 {
  let state = vIn * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn boxMullerNormal(seedA: u32, seedB: u32) -> f32 {
  let u1 = max(f32(pcgHash(seedA)) / 4294967296.0, 1.0e-7);
  let u2 = f32(pcgHash(seedB)) / 4294967296.0;
  return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

// Stage 1: sRGB EOTF -> Rec.709 -> optional 5x5 PSF -> * full_well_e -> optional EMVA-1288 noise
@compute @workgroup_size(16, 16, 1)
fn cs_srgb_to_electrons(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= TILE_W || gid.y >= TILE_H) {
    return;
  }
  let ix = i32(gid.x);
  let iy = i32(gid.y);

  var luma: f32 = 0.0;
  if (p.flags.y > 0.5) {
    for (var dy: i32 = -2; dy <= 2; dy = dy + 1) {
      for (var dx: i32 = -2; dx <= 2; dx = dx + 1) {
        luma = luma + sampleLinearLuma(ix + dx, iy + dy) * psfWeight5x5(dx, dy);
      }
    }
  } else {
    luma = sampleLinearLuma(ix, iy);
  }

  let fullWell = p.sensorCfg.x;
  var electrons = luma * fullWell;

  // Optional live EMVA-1288 shot + read noise + 12-bit ADC when noiseOff == 0
  if (p.flags.x < 0.5) {
    let seedBase = u32(p.flags.z) * 1000003u + gid.y * 1009u + gid.x * 37u + 17u;
    let zShot = boxMullerNormal(seedBase, seedBase ^ 0x9e3779b9u);
    let zRead = boxMullerNormal(seedBase ^ 0x85ebca6bu, seedBase ^ 0xc2b2ae35u);
    let meanE = max(electrons + p.flags.w, 0.0);
    let noisyE = clamp(meanE + sqrt(meanE) * zShot + p.sensorCfg.y * zRead, 0.0, fullWell);
    let rawDn = clamp(floor(noisyE * p.sensorCfg.z + p.sensorCfg.w + 0.5), 0.0, 4095.0);
    electrons = (rawDn - p.sensorCfg.w) / p.sensorCfg.z;
  }

  currElectrons[gid.y * TILE_W + gid.x] = electrons;
}

fn mulRowMat3(r0: vec3<f32>, r1: vec3<f32>, r2: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(dot(r0, v), dot(r1, v), dot(r2, v));
}

fn mulRowMat3T(r0: vec3<f32>, r1: vec3<f32>, r2: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    r0.x * v.x + r1.x * v.y + r2.x * v.z,
    r0.y * v.x + r1.y * v.y + r2.y * v.z,
    r0.z * v.x + r1.z * v.y + r2.z * v.z
  );
}

fn manualBilinearPrev1(uLocal: f32, vLocal: f32) -> vec2<f32> {
  let maxU = f32(TILE_W - 1u);
  let maxV = f32(TILE_H - 1u);
  let eps = 1.0e-4;
  if (uLocal < -eps || uLocal > maxU + eps || vLocal < -eps || vLocal > maxV + eps) {
    return vec2<f32>(0.0, 0.0);
  }
  let uc = clamp(uLocal, 0.0, maxU);
  let vc = clamp(vLocal, 0.0, maxV);
  let u0 = i32(floor(uc));
  let v0 = i32(floor(vc));
  let u1 = min(u0 + 1, i32(TILE_W) - 1);
  let v1 = min(v0 + 1, i32(TILE_H) - 1);
  let fx = uc - f32(u0);
  let fy = vc - f32(v0);

  let p00 = textureLoad(texPrev1, vec2<i32>(u0, v0), 0).r;
  let p10 = textureLoad(texPrev1, vec2<i32>(u1, v0), 0).r;
  let p01 = textureLoad(texPrev1, vec2<i32>(u0, v1), 0).r;
  let p11 = textureLoad(texPrev1, vec2<i32>(u1, v1), 0).r;

  let top = p00 + (p10 - p00) * fx;
  let bot = p01 + (p11 - p01) * fx;
  return vec2<f32>(top + (bot - top) * fy, 1.0);
}

fn manualBilinearPrev2(uLocal: f32, vLocal: f32) -> vec2<f32> {
  let maxU = f32(TILE_W - 1u);
  let maxV = f32(TILE_H - 1u);
  let eps = 1.0e-4;
  if (uLocal < -eps || uLocal > maxU + eps || vLocal < -eps || vLocal > maxV + eps) {
    return vec2<f32>(0.0, 0.0);
  }
  let uc = clamp(uLocal, 0.0, maxU);
  let vc = clamp(vLocal, 0.0, maxV);
  let u0 = i32(floor(uc));
  let v0 = i32(floor(vc));
  let u1 = min(u0 + 1, i32(TILE_W) - 1);
  let v1 = min(v0 + 1, i32(TILE_H) - 1);
  let fx = uc - f32(u0);
  let fy = vc - f32(v0);

  let p00 = textureLoad(texPrev2, vec2<i32>(u0, v0), 0).r;
  let p10 = textureLoad(texPrev2, vec2<i32>(u1, v0), 0).r;
  let p01 = textureLoad(texPrev2, vec2<i32>(u0, v1), 0).r;
  let p11 = textureLoad(texPrev2, vec2<i32>(u1, v1), 0).r;

  let top = p00 + (p10 - p00) * fx;
  let bot = p01 + (p11 - p01) * fx;
  return vec2<f32>(top + (bot - top) * fy, 1.0);
}

fn projectRayToPrev(
  dEcef: vec3<f32>,
  hitsEarth: bool,
  tHit: f32,
  deltaP: vec3<f32>,
  rPrev0: vec3<f32>,
  rPrev1: vec3<f32>,
  rPrev2: vec3<f32>,
  roiU0Prev: f32,
  roiV0Prev: f32
) -> vec2<f32> {
  var vCamDst: vec3<f32>;
  if (hitsEarth) {
    let relEcef = deltaP + dEcef * tHit;
    vCamDst = mulRowMat3T(rPrev0, rPrev1, rPrev2, relEcef);
  } else {
    vCamDst = mulRowMat3T(rPrev0, rPrev1, rPrev2, dEcef);
  }
  let zSafe = select(1.0e-12, vCamDst.z, abs(vCamDst.z) > 1.0e-12);
  let uFull = p.intr.x * (vCamDst.x / zSafe) + p.intr.z;
  let vFull = p.intr.y * (vCamDst.y / zSafe) + p.intr.w;
  return vec2<f32>(uFull - roiU0Prev, vFull - roiV0Prev);
}

// Stage 2: 2-layer 6-DoF WGS-84 + sky warp with manual FP32 bilinear textureLoad & overlap mask M
@compute @workgroup_size(16, 16, 1)
fn cs_warp_and_stack(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= TILE_W || gid.y >= TILE_H) {
    return;
  }
  let idx = gid.y * TILE_W + gid.x;
  let iCurr = currElectrons[idx];

  let uFullCurr = f32(gid.x) + p.roiCurr.x;
  let vFullCurr = f32(gid.y) + p.roiCurr.y;
  let dCamCurr = vec3<f32>(
    (uFullCurr - p.intr.z) / p.intr.x,
    (vFullCurr - p.intr.w) / p.intr.y,
    1.0
  );
  let dEcef = mulRowMat3(p.rCurr0.xyz, p.rCurr1.xyz, p.rCurr2.xyz, dCamCurr);

  // WGS-84 ellipsoid ray intersection using precomputed f64 cPrime and bigK
  let cPrime = p.cPrimeAndK.xyz;
  let bigK = p.cPrimeAndK.w;
  let dPrime = vec3<f32>(dEcef.x * WGS84_INV_A, dEcef.y * WGS84_INV_A, dEcef.z * WGS84_INV_B);
  let cDotD = dot(cPrime, dPrime);
  let d2 = dot(dPrime, dPrime);
  let disc = cDotD * cDotD - d2 * bigK;
  let hitsEarth = (disc > 0.0) && (cDotD < 0.0);
  let tHit = select(0.0, (-cDotD - sqrt(max(disc, 0.0))) / max(d2, 1.0e-20), hitsEarth);

  let uv1 = projectRayToPrev(
    dEcef, hitsEarth, tHit, p.deltaP1.xyz,
    p.rPrev1_0.xyz, p.rPrev1_1.xyz, p.rPrev1_2.xyz,
    p.roiPrev1.x, p.roiPrev1.y
  );
  let w1m1 = manualBilinearPrev1(uv1.x, uv1.y);
  let warped1 = w1m1.x * w1m1.y;
  let diff1 = w1m1.y * (iCurr - warped1);

  let uv2 = projectRayToPrev(
    dEcef, hitsEarth, tHit, p.deltaP2.xyz,
    p.rPrev2_0.xyz, p.rPrev2_1.xyz, p.rPrev2_2.xyz,
    p.roiPrev1.z, p.roiPrev1.w
  );
  let w2m2 = manualBilinearPrev2(uv2.x, uv2.y);
  let warped2 = w2m2.x * w2m2.y;
  let diff2 = w2m2.y * (iCurr - warped2);

  rawStack[0u * TILE_N + idx] = iCurr;
  rawStack[1u * TILE_N + idx] = warped1;
  rawStack[2u * TILE_N + idx] = diff1;
  rawStack[3u * TILE_N + idx] = warped2;
  rawStack[4u * TILE_N + idx] = diff2;
}

// Stage 3a: Exact FP32 per-channel mean and channel-0 population std (clamped >= 1.0)
var<workgroup> wgSum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn cs_compute_stats(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let invN = 1.0 / f32(TILE_N);

  // Per-channel shifted-anchor means for c = 0..4 (eliminates 65536-sample f32 summation ULP drift!)
  for (var c: u32 = 0u; c < 5u; c = c + 1u) {
    let base = c * TILE_N;
    let anchor = rawStack[base];
    var acc: f32 = 0.0;
    for (var i: u32 = tid; i < TILE_N; i = i + 256u) {
      acc = acc + (rawStack[base + i] - anchor);
    }
    wgSum[tid] = acc;
    workgroupBarrier();

    var stride: u32 = 128u;
    loop {
      if (stride == 0u) { break; }
      if (tid < stride) {
        wgSum[tid] = wgSum[tid] + wgSum[tid + stride];
      }
      workgroupBarrier();
      stride = stride >> 1u;
    }
    if (tid == 0u) {
      statsBuf[c] = anchor + wgSum[0] * invN;
    }
    workgroupBarrier();
  }

  // Two-pass population variance of channel 0 (unbiased=False) to avoid catastrophic cancellation
  let mean0 = statsBuf[0];
  var varAcc: f32 = 0.0;
  for (var i: u32 = tid; i < TILE_N; i = i + 256u) {
    let d = rawStack[i] - mean0;
    varAcc = varAcc + d * d;
  }
  wgSum[tid] = varAcc;
  workgroupBarrier();

  var stride2: u32 = 128u;
  loop {
    if (stride2 == 0u) { break; }
    if (tid < stride2) {
      wgSum[tid] = wgSum[tid] + wgSum[tid + stride2];
    }
    workgroupBarrier();
    stride2 = stride2 >> 1u;
  }
  if (tid == 0u) {
    let std0 = max(sqrt(max(wgSum[0] * invN, 0.0)), 1.0);
    statsBuf[5] = std0;
  }
}

// Stage 3b: Single FP32 normalize_input matching d1_model.py:285-289 clamped [-20, 20]
@compute @workgroup_size(256, 1, 1)
fn cs_normalize_input(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= TILE_N) {
    return;
  }
  let invStd0 = 1.0 / max(statsBuf[5], 1.0);
  for (var c: u32 = 0u; c < 5u; c = c + 1u) {
    let meanC = statsBuf[c];
    let val = rawStack[c * TILE_N + idx];
    normOut[c * TILE_N + idx] = clamp((val - meanC) * invStd0, -20.0, 20.0);
  }
}
