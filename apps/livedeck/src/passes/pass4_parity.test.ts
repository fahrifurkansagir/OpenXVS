import { describe, expect, it } from "vitest";
import {
  DEG_TO_RAD,
  XVS6_CUBE_V1_CAMERAS,
  eulerZyxToRotMat,
  geodeticToEcef,
  mat3Mul,
  rEcefNed,
} from "../rhi/coords.ts";
import {
  PASS4_SENSOR_DEFAULTS,
  PASS4_TENSOR_ELEMENTS,
  PASS4_TILE_H,
  PASS4_TILE_N,
  PASS4_TILE_W,
  computePass4NormalizedCpu,
  manualBilinearSampleF32,
  normalizeD1StackCpu,
  srgbChannelToLinear,
  srgbRgbaToElectronsCpu,
  warpElectrons6DofWgs84Cpu,
  type Pass4CameraPose,
} from "./pass4_ai_roi.ts";

function makePose(latDeg: number, lonDeg: number, hM: number, yawDeg: number, pitchDeg: number): Pass4CameraPose {
  const latRad = latDeg * DEG_TO_RAD;
  const lonRad = lonDeg * DEG_TO_RAD;
  const pCamEcef = geodeticToEcef(latRad, lonRad, hM);
  const rNed = rEcefNed(latRad, lonRad);
  const rBody = mat3Mul(rNed, eulerZyxToRotMat(yawDeg * DEG_TO_RAD, pitchDeg * DEG_TO_RAD, 0));
  const rEcefCam = mat3Mul(rBody, XVS6_CUBE_V1_CAMERAS[0].rBodyCam);
  return { pCamEcef, rEcefCam };
}

describe("Pass 4 D1-KF sensor-matched narrow-FOV FP32 parity (Review Focus #4)", () => {
  it("handles flat zero-contrast fog tiles with std0 clamped to 1.0 and ZERO NaN/Inf", () => {
    const flatRgba = new Uint8Array(PASS4_TILE_N * 4);
    for (let i = 0; i < PASS4_TILE_N; i++) {
      flatRgba[i * 4 + 0] = 160;
      flatRgba[i * 4 + 1] = 160;
      flatRgba[i * 4 + 2] = 160;
      flatRgba[i * 4 + 3] = 255;
    }

    const pose = makePose(37.6188, -122.3754, 600.0, 298.0, -3.0);
    const prevE = srgbRgbaToElectronsCpu(flatRgba, { noiseOff: true, applyPsf: false });
    const out = computePass4NormalizedCpu({
      currRgba: flatRgba,
      prev1Electrons: prevE,
      prev2Electrons: prevE,
      poseCurr: pose,
      posePrev1: pose,
      posePrev2: pose,
      roiCurr: [1096, 896],
      roiPrev1: [1096, 896],
      roiPrev2: [1096, 896],
      noiseOff: true,
      applyPsf: false,
    });

    expect(out.std0).toBe(1.0);
    expect(out.normalized.length).toBe(PASS4_TENSOR_ELEMENTS);
    let nonFiniteCount = 0;
    let maxAbs = 0.0;
    for (let i = 0; i < out.normalized.length; i++) {
      const v = out.normalized[i];
      if (!Number.isFinite(v)) {
        nonFiniteCount++;
      } else {
        const av = Math.abs(v);
        if (av > maxAbs) maxAbs = av;
      }
    }
    expect(nonFiniteCount).toBe(0);
    expect(maxAbs).toBeLessThan(1e-5);

    // All-zero raw stack directly into normalizeD1StackCpu
    const zeros = new Float32Array(PASS4_TENSOR_ELEMENTS);
    const normZeros = normalizeD1StackCpu(zeros);
    expect(normZeros.std0).toBe(1.0);
    expect(normZeros.normalized.every((v) => v === 0 && Number.isFinite(v))).toBe(true);
  });

  it("matches exact IEC 61966-2-1 sRGB EOTF -> Rec.709 -> * 10500.0 electron scaling", () => {
    const rgba = new Uint8Array(PASS4_TILE_N * 4);
    // Pixel 0: pure linear-segment sRGB (10/255 <= 0.04045)
    rgba[0] = 10; rgba[1] = 10; rgba[2] = 10; rgba[3] = 255;
    // Pixel 1: mid-tone RGB (128, 192, 64)
    rgba[4] = 128; rgba[5] = 192; rgba[6] = 64; rgba[7] = 255;
    // Pixel 2: peak white (255, 255, 255) -> full_well_e = 10500.0
    rgba[8] = 255; rgba[9] = 255; rgba[10] = 255; rgba[11] = 255;

    const e = srgbRgbaToElectronsCpu(rgba, { noiseOff: true, applyPsf: false });

    const exp0 = ((10 / 255) / 12.92) * PASS4_SENSOR_DEFAULTS.fullWellE;
    const exp1 =
      (0.2126 * srgbChannelToLinear(128 / 255) +
        0.7152 * srgbChannelToLinear(192 / 255) +
        0.0722 * srgbChannelToLinear(64 / 255)) *
      PASS4_SENSOR_DEFAULTS.fullWellE;

    expect(Math.abs(e[0] - exp0)).toBeLessThan(1e-3);
    expect(Math.abs(e[1] - exp1)).toBeLessThan(1e-2);
    expect(Math.abs(e[2] - 10500.0)).toBeLessThan(1e-3);
  });

  it("enforces explicit overlap mask M * (I_t - I_warped) and manual FP32 bilinear interpolation", () => {
    // 4x4 synthetic ramp checking manual bilinear vs analytical exact interpolation
    const mini = new Float32Array([
      100, 200, 300, 400,
      500, 600, 700, 800,
      900, 1000, 1100, 1200,
      1300, 1400, 1500, 1600,
    ]);
    const sIn = manualBilinearSampleF32(mini, 1.25, 0.5, 4, 4);
    expect(sIn.overlap).toBe(1.0);
    // At x=1.25: row0 = 225, row1 = 625 -> y=0.5 => 425 exact
    expect(Math.abs(sIn.value - 425.0)).toBeLessThan(1e-5);

    const sOut = manualBilinearSampleF32(mini, -0.01, 1.0, 4, 4);
    expect(sOut.overlap).toBe(0.0);
    expect(sOut.value).toBe(0.0);

    // 6-DoF yaw shift between prev and curr creates a known non-overlapping border region
    const poseCurr = makePose(37.6188, -122.3754, 600.0, 298.0, 0.0);
    const posePrev = makePose(37.6188, -122.3754, 600.0, 299.5, 0.0); // 1.5 deg yaw shift (~60 px at fx=2319)

    const prevE = new Float32Array(PASS4_TILE_N).fill(4200.0);
    const currRgba = new Uint8Array(PASS4_TILE_N * 4);
    for (let i = 0; i < PASS4_TILE_N; i++) {
      currRgba[i * 4 + 0] = 180;
      currRgba[i * 4 + 1] = 180;
      currRgba[i * 4 + 2] = 180;
      currRgba[i * 4 + 3] = 255;
    }

    const { overlapMask } = warpElectrons6DofWgs84Cpu(
      prevE,
      poseCurr,
      posePrev,
      [1096, 896],
      [1096, 896],
    );

    const out = computePass4NormalizedCpu({
      currRgba,
      prev1Electrons: prevE,
      prev2Electrons: prevE,
      poseCurr,
      posePrev1: posePrev,
      posePrev2: posePrev,
      roiCurr: [1096, 896],
      roiPrev1: [1096, 896],
      roiPrev2: [1096, 896],
      noiseOff: true,
      applyPsf: false,
    });

    let zeroMaskPixels = 0;
    let oneMaskPixels = 0;
    let nonZeroOutsideMask = 0;
    for (let i = 0; i < PASS4_TILE_N; i++) {
      if (overlapMask[i] === 0.0) {
        zeroMaskPixels++;
        // Outside overlap: both warped prev (ch 1) and residual M*(I_t - I_warped) (ch 2) MUST be exact 0.0!
        if (out.rawStack[1 * PASS4_TILE_N + i] !== 0.0 || out.rawStack[2 * PASS4_TILE_N + i] !== 0.0) {
          nonZeroOutsideMask++;
        }
      } else {
        oneMaskPixels++;
      }
    }
    expect(nonZeroOutsideMask).toBe(0);
    expect(zeroMaskPixels).toBeGreaterThan(PASS4_TILE_H * 40);
    expect(oneMaskPixels).toBeGreaterThan(PASS4_TILE_H * 150);
  });

  it("clamps extreme outliers to [-20, 20] and matches EMVA-1288 noise mean/variance within +/-5% over 120 frames", () => {
    // 1. Outlier clamp check
    const raw = new Float32Array(PASS4_TENSOR_ELEMENTS).fill(5000.0);
    raw[0] = 5000.0 + 10000.0; // huge spike
    const norm = normalizeD1StackCpu(raw);
    expect(norm.normalized[0]).toBeLessThanOrEqual(20.0);
    expect(norm.normalized[0]).toBeGreaterThanOrEqual(-20.0);

    // 2. Live EMVA-1288 noise statistics across 120 deterministic frames at fixed pixel
    const flatRgba = new Uint8Array(PASS4_TILE_N * 4);
    for (let i = 0; i < PASS4_TILE_N; i++) {
      flatRgba[i * 4 + 0] = 140;
      flatRgba[i * 4 + 1] = 140;
      flatRgba[i * 4 + 2] = 140;
      flatRgba[i * 4 + 3] = 255;
    }
    const cleanE = srgbRgbaToElectronsCpu(flatRgba, { noiseOff: true, applyPsf: false })[0];
    const samples: number[] = [];
    // Measure at pixel (128, 128) across 120 frames (plus 16 spatial neighbors = 1920 independent draws)
    for (let f = 0; f < 120; f++) {
      const noisy = srgbRgbaToElectronsCpu(flatRgba, { noiseOff: false, applyPsf: false, frameSeed: f + 1 });
      for (let k = 0; k < 16; k++) {
        samples.push(noisy[128 * PASS4_TILE_W + 120 + k]);
      }
    }
    const sampleMean = samples.reduce((a, b) => a + b, 0) / samples.length;
    const sampleVar =
      samples.reduce((a, b) => a + (b - sampleMean) ** 2, 0) / (samples.length - 1);
    const analyticalVar =
      cleanE +
      PASS4_SENSOR_DEFAULTS.sigmaRE ** 2 +
      1.0 / (12.0 * PASS4_SENSOR_DEFAULTS.kDnPerE ** 2);

    expect(Math.abs(sampleMean - cleanE) / cleanE).toBeLessThan(0.02);
    expect(Math.abs(sampleVar - analyticalVar) / analyticalVar).toBeLessThan(0.05);
  });
});
