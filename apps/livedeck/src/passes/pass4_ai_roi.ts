/**
 * Pass 4: Sensor-Matched D1-KF Narrow-FOV Pipeline (traffic_ring_v1, fx=fy=2319.0, IFOV=0.0247 deg/px).
 *
 * - Runs natively on WebGPU compute (`pass4_ai_roi.wgsl`) with a 3-slot async Staging Ring Buffer
 *   (`mapAsync` is NEVER awaited inside `requestAnimationFrame`).
 * - Provides an explicit deterministic FP32 TypeScript mirror (`computePass4NormalizedCpu`) used
 *   both in WebGL2 mode and in Vitest/Playwright numerical parity harnesses (`V2-M1-4`).
 * - Exact match to:
 *     - `src/xvs/sensor/srgb.py` (IEC 61966-2-1 EOTF + Rec.709)
 *     - `configs/xvs/sensors/ring_mono_v0_nodist.json` (`full_well_e=10500`, `sigma_r_e=2.4`, `k=0.35`, `12-bit`)
 *     - `src/xvs/detection/warp.py` (2-layer 6-DoF WGS-84 + sky warp, manual FP32 bilinear, overlap mask M)
 *     - `src/xvs/detection/d1_model.py:285-289` (single FP32 `normalize_input`, `std0 = max(std, 1.0)`, clamp `[-20, 20]`).
 */

import {
  TRAFFIC_RING_V1_INTRINSICS,
  WGS84_A,
  WGS84_B,
  XVS6_CUBE_V1_CAMERAS,
  buildRoiIntrinsics,
  mat3Mul,
  mat3MulVec,
  mat3TransposeMulVec,
  projectEcefToNdc,
  type Mat3RowMajor,
  type Vec3,
} from "../rhi/coords.ts";
import type { RhiContext } from "../rhi/types.ts";
import pass4Wgsl from "../shaders/pass4_ai_roi.wgsl?raw";
import type {
  ActiveCameraView,
  CockpitFrameState,
  Pass4RenderOptions,
  Pass4TensorPacket,
} from "./types.ts";

export const PASS4_TILE_W = 256;
export const PASS4_TILE_H = 256;
export const PASS4_TILE_N = PASS4_TILE_W * PASS4_TILE_H; // 65536
export const PASS4_CHANNELS = 5;
export const PASS4_TENSOR_ELEMENTS = PASS4_CHANNELS * PASS4_TILE_N; // 327680
export const PASS4_TENSOR_BYTES = PASS4_TENSOR_ELEMENTS * 4; // 1,310,720 bytes

export const PASS4_SENSOR_DEFAULTS = {
  fullWellE: 10500.0,
  sigmaRE: 2.4,
  darkCurrentE: 5.0 * 0.001, // 0.005 e- per 1 ms exposure
  kDnPerE: 0.35,
  blackLevelDn: 64.0,
  maxDn: 4095.0,
  keptFrameStepDtS: 1.0,
} as const;

export const REC709_R = 0.2126;
export const REC709_G = 0.7152;
export const REC709_B = 0.0722;

/** Symmetric 5x5 matched optical PSF kernel from `ring_mono_v0_nodist.json` (f/2.8, 550nm, 3.45um, sigma=0.30). */
export function psfWeight5x5(dx: number, dy: number): number {
  const ax = Math.abs(dx);
  const ay = Math.abs(dy);
  const hi = Math.max(ax, ay);
  const lo = Math.min(ax, ay);
  if (hi === 0 && lo === 0) return 0.64551891;
  if (hi === 1 && lo === 0) return 0.06989676;
  if (hi === 1 && lo === 1) return 0.01243128;
  if (hi === 2 && lo === 0) return 0.00235286;
  if (hi === 2 && lo === 1) return 0.00160273;
  return 0.0007339;
}

export interface Pass4CameraPose {
  readonly pCamEcef: Vec3;
  readonly rEcefCam: Mat3RowMajor;
}

export interface Pass4CpuInput {
  readonly currRgba: Uint8Array;
  readonly prev1Electrons: Float32Array;
  readonly prev2Electrons: Float32Array;
  readonly poseCurr: Pass4CameraPose;
  readonly posePrev1: Pass4CameraPose;
  readonly posePrev2: Pass4CameraPose;
  readonly roiCurr: readonly [number, number];
  readonly roiPrev1: readonly [number, number];
  readonly roiPrev2: readonly [number, number];
  readonly noiseOff: boolean;
  readonly applyPsf: boolean;
  readonly frameSeed?: number;
}

/** Exact IEC 61966-2-1 sRGB EOTF matching `src/xvs/sensor/srgb.py:17-46`. */
export function srgbChannelToLinear(c: number): number {
  const x = Math.max(c, 0.0);
  if (x <= 0.04045) {
    return x / 12.92;
  }
  return Math.pow((x + 0.055) / 1.055, 2.4);
}

function pcgHash(vIn: number): number {
  const state = (Math.imul(vIn >>> 0, 747796405) + 2891336453) >>> 0;
  const shift = ((state >>> 28) + 4) >>> 0;
  const word = Math.imul(((state >>> shift) ^ state) >>> 0, 277803737) >>> 0;
  return ((word >>> 22) ^ word) >>> 0;
}

function boxMullerNormal(seedA: number, seedB: number): number {
  const u1 = Math.max(pcgHash(seedA) / 4294967296.0, 1.0e-7);
  const u2 = pcgHash(seedB) / 4294967296.0;
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(6.283185307179586 * u2);
}

/**
 * Convert 256x256 sRGB RGBA8 tile to linear electrons (`Float32Array[65536]`) using
 * IEC 61966-2-1 EOTF -> Rec.709 -> optional 5x5 PSF -> `* 10500.0` -> optional EMVA-1288 noise.
 */
export function srgbRgbaToElectronsCpu(
  rgba: Uint8Array,
  options: { noiseOff: boolean; applyPsf: boolean; frameSeed?: number },
): Float32Array {
  const luma = new Float32Array(PASS4_TILE_N);
  for (let i = 0; i < PASS4_TILE_N; i++) {
    const r = srgbChannelToLinear(rgba[i * 4 + 0] / 255.0);
    const g = srgbChannelToLinear(rgba[i * 4 + 1] / 255.0);
    const b = srgbChannelToLinear(rgba[i * 4 + 2] / 255.0);
    luma[i] = Math.fround(REC709_R * r + REC709_G * g + REC709_B * b);
  }

  const out = new Float32Array(PASS4_TILE_N);
  const seedFrame = (options.frameSeed ?? 0) >>> 0;

  for (let y = 0; y < PASS4_TILE_H; y++) {
    for (let x = 0; x < PASS4_TILE_W; x++) {
      let val = 0.0;
      if (options.applyPsf) {
        for (let dy = -2; dy <= 2; dy++) {
          const cy = Math.max(0, Math.min(PASS4_TILE_H - 1, y + dy));
          for (let dx = -2; dx <= 2; dx++) {
            const cx = Math.max(0, Math.min(PASS4_TILE_W - 1, x + dx));
            val += luma[cy * PASS4_TILE_W + cx] * psfWeight5x5(dx, dy);
          }
        }
      } else {
        val = luma[y * PASS4_TILE_W + x];
      }

      let electrons = val * PASS4_SENSOR_DEFAULTS.fullWellE;
      if (!options.noiseOff) {
        const seedBase =
          (Math.imul(seedFrame, 1000003) + Math.imul(y, 1009) + Math.imul(x, 37) + 17) >>> 0;
        const zShot = boxMullerNormal(seedBase, (seedBase ^ 0x9e3779b9) >>> 0);
        const zRead = boxMullerNormal((seedBase ^ 0x85ebca6b) >>> 0, (seedBase ^ 0xc2b2ae35) >>> 0);
        const meanE = Math.max(electrons + PASS4_SENSOR_DEFAULTS.darkCurrentE, 0.0);
        const noisyE = Math.max(
          0.0,
          Math.min(
            PASS4_SENSOR_DEFAULTS.fullWellE,
            meanE + Math.sqrt(meanE) * zShot + PASS4_SENSOR_DEFAULTS.sigmaRE * zRead,
          ),
        );
        const rawDn = Math.max(
          0.0,
          Math.min(
            PASS4_SENSOR_DEFAULTS.maxDn,
            Math.floor(noisyE * PASS4_SENSOR_DEFAULTS.kDnPerE + PASS4_SENSOR_DEFAULTS.blackLevelDn + 0.5),
          ),
        );
        electrons = (rawDn - PASS4_SENSOR_DEFAULTS.blackLevelDn) / PASS4_SENSOR_DEFAULTS.kDnPerE;
      }

      out[y * PASS4_TILE_W + x] = Math.fround(electrons);
    }
  }
  return out;
}

/** Manual FP32 bilinear sampling matching WGSL `textureLoad` bilinear (zero fixed-point error). */
export function manualBilinearSampleF32(
  img: Float32Array,
  uLocal: number,
  vLocal: number,
  width = PASS4_TILE_W,
  height = PASS4_TILE_H,
): { value: number; overlap: number } {
  const maxU = width - 1;
  const maxV = height - 1;
  const eps = 1.0e-4;
  if (
    !Number.isFinite(uLocal) ||
    !Number.isFinite(vLocal) ||
    uLocal < -eps ||
    uLocal > maxU + eps ||
    vLocal < -eps ||
    vLocal > maxV + eps
  ) {
    return { value: 0.0, overlap: 0.0 };
  }
  const uClamped = Math.max(0.0, Math.min(maxU, uLocal));
  const vClamped = Math.max(0.0, Math.min(maxV, vLocal));
  const u0 = Math.floor(uClamped);
  const v0 = Math.floor(vClamped);
  const u1 = Math.min(u0 + 1, maxU);
  const v1 = Math.min(v0 + 1, maxV);
  const fx = Math.fround(uClamped - u0);
  const fy = Math.fround(vClamped - v0);

  const p00 = img[v0 * width + u0];
  const p10 = img[v0 * width + u1];
  const p01 = img[v1 * width + u0];
  const p11 = img[v1 * width + u1];

  const top = Math.fround(p00 + (p10 - p00) * fx);
  const bot = Math.fround(p01 + (p11 - p01) * fx);
  return { value: Math.fround(top + (bot - top) * fy), overlap: 1.0 };
}

/**
 * 2-layer 6-DoF WGS-84 ellipsoid + sky-at-infinity registration matching `src/xvs/detection/warp.py:130-270`.
 */
export function warpElectrons6DofWgs84Cpu(
  prevElectrons: Float32Array,
  poseCurr: Pass4CameraPose,
  posePrev: Pass4CameraPose,
  roiCurr: readonly [number, number],
  roiPrev: readonly [number, number],
): { warped: Float32Array; overlapMask: Float32Array } {
  const warped = new Float32Array(PASS4_TILE_N);
  const overlapMask = new Float32Array(PASS4_TILE_N);

  const invA = 1.0 / WGS84_A;
  const invB = 1.0 / WGS84_B;
  const cPrime: Vec3 = [
    poseCurr.pCamEcef[0] * invA,
    poseCurr.pCamEcef[1] * invA,
    poseCurr.pCamEcef[2] * invB,
  ];
  const bigK =
    cPrime[0] * cPrime[0] + cPrime[1] * cPrime[1] + cPrime[2] * cPrime[2] - 1.0;

  // RTE position delta in f64 (eliminates 6.378e6 cancellation)
  const deltaP: Vec3 = [
    poseCurr.pCamEcef[0] - posePrev.pCamEcef[0],
    poseCurr.pCamEcef[1] - posePrev.pCamEcef[1],
    poseCurr.pCamEcef[2] - posePrev.pCamEcef[2],
  ];

  const { fx, fy, cx, cy } = TRAFFIC_RING_V1_INTRINSICS;
  const [u0Curr, v0Curr] = roiCurr;
  const [u0Prev, v0Prev] = roiPrev;

  for (let y = 0; y < PASS4_TILE_H; y++) {
    const vFullCurr = y + v0Curr;
    const dyCam = (vFullCurr - cy) / fy;
    for (let x = 0; x < PASS4_TILE_W; x++) {
      const uFullCurr = x + u0Curr;
      const dxCam = (uFullCurr - cx) / fx;
      const dEcef = mat3MulVec(poseCurr.rEcefCam, [dxCam, dyCam, 1.0]);

      const dpx = dEcef[0] * invA;
      const dpy = dEcef[1] * invA;
      const dpz = dEcef[2] * invB;
      const cDotD = cPrime[0] * dpx + cPrime[1] * dpy + cPrime[2] * dpz;
      const d2 = dpx * dpx + dpy * dpy + dpz * dpz;
      const disc = cDotD * cDotD - d2 * bigK;
      const hitsEarth = disc > 0.0 && cDotD < 0.0;

      let vCamDst: [number, number, number];
      if (hitsEarth) {
        const tHit = (-cDotD - Math.sqrt(Math.max(0.0, disc))) / Math.max(d2, 1e-20);
        const relEcef: Vec3 = [
          deltaP[0] + dEcef[0] * tHit,
          deltaP[1] + dEcef[1] * tHit,
          deltaP[2] + dEcef[2] * tHit,
        ];
        vCamDst = mat3TransposeMulVec(posePrev.rEcefCam, relEcef);
      } else {
        vCamDst = mat3TransposeMulVec(posePrev.rEcefCam, dEcef);
      }

      const zSafe = Math.abs(vCamDst[2]) > 1e-12 ? vCamDst[2] : 1e-12;
      const uLocalPrev = fx * (vCamDst[0] / zSafe) + cx - u0Prev;
      const vLocalPrev = fy * (vCamDst[1] / zSafe) + cy - v0Prev;

      const s = manualBilinearSampleF32(prevElectrons, uLocalPrev, vLocalPrev);
      const idx = y * PASS4_TILE_W + x;
      warped[idx] = Math.fround(s.value * s.overlap);
      overlapMask[idx] = s.overlap;
    }
  }

  return { warped, overlapMask };
}

/**
 * Exact single FP32 `normalize_input()` matching `src/xvs/detection/d1_model.py:285-289`:
 *   mean[c] = mean(x[c])
 *   std0    = max(std(x[0], unbiased=False), 1.0)
 *   out[c]  = clamp((x[c] - mean[c]) / std0, -20.0, 20.0)
 *
 * Guarantees zero NaN on flat zero-contrast fog tiles (`std0 == 0 -> clamp 1.0`).
 */
export function normalizeD1StackCpu(rawStack: Float32Array): {
  normalized: Float32Array;
  means: [number, number, number, number, number];
  std0: number;
} {
  if (rawStack.length !== PASS4_TENSOR_ELEMENTS) {
    throw new Error(`Expected rawStack length ${PASS4_TENSOR_ELEMENTS}, got ${rawStack.length}`);
  }
  const means: [number, number, number, number, number] = [0, 0, 0, 0, 0];
  for (let c = 0; c < PASS4_CHANNELS; c++) {
    const base = c * PASS4_TILE_N;
    let sum = 0.0;
    for (let i = 0; i < PASS4_TILE_N; i++) {
      sum += rawStack[base + i];
    }
    means[c] = sum / PASS4_TILE_N;
  }

  // Two-pass population variance of channel 0 (unbiased=False)
  const mean0 = means[0];
  let varSum = 0.0;
  for (let i = 0; i < PASS4_TILE_N; i++) {
    const d = rawStack[i] - mean0;
    varSum += d * d;
  }
  const std0 = Math.max(Math.sqrt(Math.max(varSum / PASS4_TILE_N, 0.0)), 1.0);
  const invStd0 = 1.0 / std0;

  const normalized = new Float32Array(PASS4_TENSOR_ELEMENTS);
  for (let c = 0; c < PASS4_CHANNELS; c++) {
    const base = c * PASS4_TILE_N;
    const meanC = means[c];
    for (let i = 0; i < PASS4_TILE_N; i++) {
      const z = (rawStack[base + i] - meanC) * invStd0;
      const clamped = z < -20.0 ? -20.0 : z > 20.0 ? 20.0 : z;
      normalized[base + i] = Math.fround(clamped);
    }
  }

  return { normalized, means, std0 };
}

/**
 * Full deterministic CPU FP32 Pass-4 pipeline mirroring `pass4_ai_roi.wgsl` end-to-end.
 */
export function computePass4NormalizedCpu(input: Pass4CpuInput): {
  currElectrons: Float32Array;
  rawStack: Float32Array;
  normalized: Float32Array;
  means: [number, number, number, number, number];
  std0: number;
} {
  const currElectrons = srgbRgbaToElectronsCpu(input.currRgba, {
    noiseOff: input.noiseOff,
    applyPsf: input.applyPsf,
    frameSeed: input.frameSeed,
  });

  const w1 = warpElectrons6DofWgs84Cpu(
    input.prev1Electrons,
    input.poseCurr,
    input.posePrev1,
    input.roiCurr,
    input.roiPrev1,
  );
  const w2 = warpElectrons6DofWgs84Cpu(
    input.prev2Electrons,
    input.poseCurr,
    input.posePrev2,
    input.roiCurr,
    input.roiPrev2,
  );

  const rawStack = new Float32Array(PASS4_TENSOR_ELEMENTS);
  rawStack.set(currElectrons, 0 * PASS4_TILE_N);
  rawStack.set(w1.warped, 1 * PASS4_TILE_N);
  rawStack.set(w2.warped, 3 * PASS4_TILE_N);

  for (let i = 0; i < PASS4_TILE_N; i++) {
    // Explicit overlap masking: M * (I_t - I_warped) matching warp.py:449
    rawStack[2 * PASS4_TILE_N + i] = Math.fround(w1.overlapMask[i] * (currElectrons[i] - w1.warped[i]));
    rawStack[4 * PASS4_TILE_N + i] = Math.fround(w2.overlapMask[i] * (currElectrons[i] - w2.warped[i]));
  }

  const { normalized, means, std0 } = normalizeD1StackCpu(rawStack);
  return { currElectrons, rawStack, normalized, means, std0 };
}

interface KeptRingFrame {
  simTimeS: number;
  pose: Pass4CameraPose;
  roiOrigin: [number, number];
  electronsCpu: Float32Array;
}

type StagingSlotState = "idle" | "mapping";

interface StagingSlot {
  buffer: GPUBuffer;
  statsBuffer: GPUBuffer;
  state: StagingSlotState;
}

export interface Pass4RingDiagnostics {
  readonly computePath: "webgpu-compute" | "cpu-fp32-fallback";
  readonly completedReadbacks: number;
  readonly skippedReadbacks: number;
  readonly inFlightSlots: number;
  readonly keptRingDepth: number;
}

export class Pass4AiRoi {
  private readonly rhi: RhiContext;
  private readonly device: GPUDevice | null;

  // WebGPU compute resources (null in WebGL2 mode)
  private paramsBuf: GPUBuffer | null = null;
  private roiRgbaTex: GPUTexture | null = null;
  private texPrev1: GPUTexture | null = null;
  private texPrev2: GPUTexture | null = null;
  private currElectronsBuf: GPUBuffer | null = null;
  private rawStackBuf: GPUBuffer | null = null;
  private statsBuf: GPUBuffer | null = null;
  private normOutBuf: GPUBuffer | null = null;
  private stagingRing: StagingSlot[] = [];
  private nextStagingIdx = 0;

  private pipeSrgb: GPUComputePipeline | null = null;
  private pipeWarp: GPUComputePipeline | null = null;
  private pipeStats: GPUComputePipeline | null = null;
  private pipeNorm: GPUComputePipeline | null = null;
  private bindGroup: GPUBindGroup | null = null;

  private readonly paramsScratch = new Float32Array(68); // 17 x vec4<f32> = 272 bytes
  private readonly roiRgbaScratch = new Uint8Array(PASS4_TILE_N * 4);
  private readonly latestNormCopy = new Float32Array(PASS4_TENSOR_ELEMENTS);

  private keptRing: KeptRingFrame[] = [];
  private latestPacket: Pass4TensorPacket | null = null;
  private latestMeans: [number, number, number, number, number] = [0, 0, 0, 0, 0];
  private latestStd0 = 1.0;

  private frameCounter = 0;
  private completedReadbacks = 0;
  private skippedReadbacks = 0;
  private disposed = false;

  constructor(rhi: RhiContext) {
    this.rhi = rhi;
    this.device = rhi.gpuDevice;

    if (this.device) {
      this.initWebGpuCompute(this.device);
    }
  }

  private initWebGpuCompute(device: GPUDevice): void {
    const module = device.createShaderModule({
      label: "pass4-ai-roi-wgsl",
      code: pass4Wgsl,
    });

    const bgl = device.createBindGroupLayout({
      label: "pass4-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const pl = device.createPipelineLayout({ label: "pass4-pl", bindGroupLayouts: [bgl] });

    this.pipeSrgb = device.createComputePipeline({
      label: "pass4-cs-srgb",
      layout: pl,
      compute: { module, entryPoint: "cs_srgb_to_electrons" },
    });
    this.pipeWarp = device.createComputePipeline({
      label: "pass4-cs-warp",
      layout: pl,
      compute: { module, entryPoint: "cs_warp_and_stack" },
    });
    this.pipeStats = device.createComputePipeline({
      label: "pass4-cs-stats",
      layout: pl,
      compute: { module, entryPoint: "cs_compute_stats" },
    });
    this.pipeNorm = device.createComputePipeline({
      label: "pass4-cs-norm",
      layout: pl,
      compute: { module, entryPoint: "cs_normalize_input" },
    });

    this.paramsBuf = device.createBuffer({
      label: "pass4-params-ubo",
      size: 288, // 17 * 16 = 272 -> align up to 288
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.roiRgbaTex = device.createTexture({
      label: "pass4-roi-rgba8",
      size: { width: PASS4_TILE_W, height: PASS4_TILE_H, depthOrArrayLayers: 1 },
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    const mkR32Tex = (label: string) =>
      device.createTexture({
        label,
        size: { width: PASS4_TILE_W, height: PASS4_TILE_H, depthOrArrayLayers: 1 },
        format: "r32float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });

    this.texPrev1 = mkR32Tex("pass4-prev1-r32f");
    this.texPrev2 = mkR32Tex("pass4-prev2-r32f");

    this.currElectronsBuf = device.createBuffer({
      label: "pass4-curr-electrons",
      size: PASS4_TILE_N * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.rawStackBuf = device.createBuffer({
      label: "pass4-raw-stack",
      size: PASS4_TENSOR_BYTES,
      usage: GPUBufferUsage.STORAGE,
    });
    this.statsBuf = device.createBuffer({
      label: "pass4-stats",
      size: 32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.normOutBuf = device.createBuffer({
      label: "pass4-norm-out",
      size: PASS4_TENSOR_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // 3-slot async Staging Ring Buffer (paired tensor + stats buffers; never awaited inside rAF!)
    this.stagingRing = [0, 1, 2].map((slotIdx) => ({
      buffer: device.createBuffer({
        label: `pass4-staging-slot-${slotIdx}`,
        size: PASS4_TENSOR_BYTES,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      }),
      statsBuffer: device.createBuffer({
        label: `pass4-stats-slot-${slotIdx}`,
        size: 32,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      }),
      state: "idle",
    }));

    this.bindGroup = device.createBindGroup({
      label: "pass4-bg",
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuf } },
        { binding: 1, resource: this.roiRgbaTex.createView() },
        { binding: 2, resource: this.texPrev1.createView() },
        { binding: 3, resource: this.texPrev2.createView() },
        { binding: 4, resource: { buffer: this.currElectronsBuf } },
        { binding: 5, resource: { buffer: this.rawStackBuf } },
        { binding: 6, resource: { buffer: this.statsBuf } },
        { binding: 7, resource: { buffer: this.normOutBuf } },
      ],
    });
  }

  /** Reset the Delta_t = 1.0 s 3-frame kept ring (e.g., when switching missions). */
  resetRing(): void {
    this.keptRing = [];
    this.latestPacket = null;
  }

  getLatestNormalizedTensor(): Pass4TensorPacket | null {
    return this.latestPacket;
  }

  getDiagnostics(): Pass4RingDiagnostics {
    const inFlightSlots = this.stagingRing.filter((s) => s.state === "mapping").length;
    return {
      computePath: this.device ? "webgpu-compute" : "cpu-fp32-fallback",
      completedReadbacks: this.completedReadbacks,
      skippedReadbacks: this.skippedReadbacks,
      inFlightSlots,
      keptRingDepth: this.keptRing.length,
    };
  }

  /**
   * Synthesize narrow-FOV 256x256 `traffic_ring_v1` (fx=fy=2319.0) ROI sRGB crop
   * steered onto the forward/tracked encounter sector.
   */
  private synthesizeNarrowFovRoiRgba(
    state: CockpitFrameState,
    view: ActiveCameraView,
    poseCurr: Pass4CameraPose,
    roiOrigin: readonly [number, number],
  ): Uint8Array {
    const out = this.roiRgbaScratch;
    const [u0, v0] = roiOrigin;
    const roiIntr = buildRoiIntrinsics(
      TRAFFIC_RING_V1_INTRINSICS,
      u0,
      v0,
      PASS4_TILE_W,
      PASS4_TILE_H,
    );

    // Subtle sky-to-haze vertical gradient matching scene fogContrast
    const fog = Math.max(0.08, Math.min(1.0, state.fogContrast));
    const baseSkyR = Math.round((95 + (1 - fog) * 75));
    const baseSkyG = Math.round((132 + (1 - fog) * 48));
    const baseSkyB = Math.round((186 + (1 - fog) * 14));

    for (let y = 0; y < PASS4_TILE_H; y++) {
      const t = y / (PASS4_TILE_H - 1);
      const r = Math.min(255, Math.round(baseSkyR + t * 38));
      const g = Math.min(255, Math.round(baseSkyG + t * 34));
      const b = Math.min(255, Math.round(baseSkyB + t * 24));
      for (let x = 0; x < PASS4_TILE_W; x++) {
        const p = (y * PASS4_TILE_W + x) * 4;
        out[p + 0] = r;
        out[p + 1] = g;
        out[p + 2] = b;
        out[p + 3] = 255;
      }
    }

    // Stamp real 3D intruders at exact `traffic_ring_v1` sub-pixel scale (s_50 ~= 1.02 px at long range)
    const vp = { width: PASS4_TILE_W, height: PASS4_TILE_H, zNear: 1.0, zFar: 200000.0 };
    for (const tr of state.intruders) {
      if (tr.isGhost && tr.ghostRejected) continue;
      const proj = projectEcefToNdc(
        tr.posEcef,
        poseCurr.pCamEcef,
        poseCurr.rEcefCam,
        roiIntr,
        vp,
      );
      if (!proj.visible) continue;

      // Angular wingspan (~12 m GA/jet) -> exact pixel radius in traffic_ring_v1
      const wingspanPx = (12.0 / Math.max(tr.rangeM, 50.0)) * roiIntr.fx;
      const sigma = Math.max(0.85, Math.min(4.0, wingspanPx * 0.5));
      const rad = Math.ceil(sigma * 3.0);
      const cx = Math.round(proj.uPx);
      const cy = Math.round(proj.vPx);

      for (let dy = -rad; dy <= rad; dy++) {
        const py = cy + dy;
        if (py < 0 || py >= PASS4_TILE_H) continue;
        for (let dx = -rad; dx <= rad; dx++) {
          const px = cx + dx;
          if (px < 0 || px >= PASS4_TILE_W) continue;
          const dist2 = (px - proj.uPx) ** 2 + (py - proj.vPx) ** 2;
          const w = Math.exp(-dist2 / (2.0 * sigma * sigma));
          if (w < 0.02) continue;
          const idx = (py * PASS4_TILE_W + px) * 4;
          // Dark aircraft fuselage contrast against sky (scaled by fogContrast)
          const darkFactor = 1.0 - w * 0.72 * fog;
          out[idx + 0] = Math.max(0, Math.round(out[idx + 0] * darkFactor));
          out[idx + 1] = Math.max(0, Math.round(out[idx + 1] * darkFactor));
          out[idx + 2] = Math.max(0, Math.round(out[idx + 2] * darkFactor));
        }
      }
    }

    // Keep view referenced cleanly (used if extending to horizon limb projection)
    void view;
    return out;
  }

  private packUniforms(
    poseCurr: Pass4CameraPose,
    posePrev1: Pass4CameraPose,
    posePrev2: Pass4CameraPose,
    roiCurr: readonly [number, number],
    roiPrev1: readonly [number, number],
    roiPrev2: readonly [number, number],
    noiseOff: boolean,
    applyPsf: boolean,
    frameSeed: number,
  ): void {
    const u = this.paramsScratch;
    const { fx, fy, cx, cy } = TRAFFIC_RING_V1_INTRINSICS;
    u[0] = fx; u[1] = fy; u[2] = cx; u[3] = cy;
    u[4] = roiCurr[0]; u[5] = roiCurr[1]; u[6] = PASS4_TILE_W; u[7] = PASS4_TILE_H;
    u[8] = roiPrev1[0]; u[9] = roiPrev1[1]; u[10] = roiPrev2[0]; u[11] = roiPrev2[1];
    u[12] = PASS4_SENSOR_DEFAULTS.fullWellE;
    u[13] = PASS4_SENSOR_DEFAULTS.sigmaRE;
    u[14] = PASS4_SENSOR_DEFAULTS.kDnPerE;
    u[15] = PASS4_SENSOR_DEFAULTS.blackLevelDn;
    u[16] = noiseOff ? 1.0 : 0.0;
    u[17] = applyPsf ? 1.0 : 0.0;
    u[18] = frameSeed;
    u[19] = PASS4_SENSOR_DEFAULTS.darkCurrentE;

    // Precompute f64 WGS-84 cPrime, bigK, and RTE deltas to prevent f32 cancellation
    const cpx = poseCurr.pCamEcef[0] / WGS84_A;
    const cpy = poseCurr.pCamEcef[1] / WGS84_A;
    const cpz = poseCurr.pCamEcef[2] / WGS84_B;
    const bigK = cpx * cpx + cpy * cpy + cpz * cpz - 1.0;

    u[20] = cpx; u[21] = cpy; u[22] = cpz; u[23] = bigK;
    u[24] = poseCurr.pCamEcef[0] - posePrev1.pCamEcef[0];
    u[25] = poseCurr.pCamEcef[1] - posePrev1.pCamEcef[1];
    u[26] = poseCurr.pCamEcef[2] - posePrev1.pCamEcef[2];
    u[27] = 0.0;
    u[28] = poseCurr.pCamEcef[0] - posePrev2.pCamEcef[0];
    u[29] = poseCurr.pCamEcef[1] - posePrev2.pCamEcef[1];
    u[30] = poseCurr.pCamEcef[2] - posePrev2.pCamEcef[2];
    u[31] = 0.0;

    const writeMat3Rows = (offset: number, m: Mat3RowMajor) => {
      u[offset + 0] = m[0]; u[offset + 1] = m[1]; u[offset + 2] = m[2]; u[offset + 3] = 0;
      u[offset + 4] = m[3]; u[offset + 5] = m[4]; u[offset + 6] = m[5]; u[offset + 7] = 0;
      u[offset + 8] = m[6]; u[offset + 9] = m[7]; u[offset + 10] = m[8]; u[offset + 11] = 0;
    };
    writeMat3Rows(32, poseCurr.rEcefCam);
    writeMat3Rows(44, posePrev1.rEcefCam);
    writeMat3Rows(56, posePrev2.rEcefCam);
  }

  /**
   * Execute Pass 4 for the current cockpit frame.
   * Never blocks or awaits inside `requestAnimationFrame`!
   */
  step(
    state: CockpitFrameState,
    view: ActiveCameraView,
    options: Pass4RenderOptions = {},
  ): Pass4TensorPacket | null {
    if (this.disposed) return this.latestPacket;

    const noiseOff = options.noiseOff ?? false;
    const applyPsf = options.applyPsf ?? true;
    const frameIdx = this.frameCounter++;

    // Forward traffic_ring_v1 T0 camera pose (`lever_arm_m = [0, 0, -0.5]`, forward boresight)
    const leverEcef = mat3MulVec(view.rEcefBody, [0.0, 0.0, -0.5]);
    const pCamCurr: [number, number, number] = [
      view.pEcefBody[0] + leverEcef[0],
      view.pEcefBody[1] + leverEcef[1],
      view.pEcefBody[2] + leverEcef[2],
    ];
    const rEcefCamCurr = mat3Mul(view.rEcefBody, XVS6_CUBE_V1_CAMERAS[0].rBodyCam);
    const poseCurr: Pass4CameraPose = { pCamEcef: pCamCurr, rEcefCam: rEcefCamCurr };

    // Steer 256x256 ROI onto primary intruder if in front of T0, else centered on forward boresight
    let roiU0 = options.roiOrigin ? options.roiOrigin[0] : 1096;
    let roiV0 = options.roiOrigin ? options.roiOrigin[1] : 896;
    if (!options.roiOrigin && state.intruders.length > 0) {
      const target = state.intruders.find((t) => !(t.isGhost && t.ghostRejected)) ?? state.intruders[0];
      const fullProj = projectEcefToNdc(
        target.posEcef,
        pCamCurr,
        rEcefCamCurr,
        TRAFFIC_RING_V1_INTRINSICS,
        { width: TRAFFIC_RING_V1_INTRINSICS.width, height: TRAFFIC_RING_V1_INTRINSICS.height },
      );
      if (fullProj.visible) {
        roiU0 = Math.max(
          0,
          Math.min(TRAFFIC_RING_V1_INTRINSICS.width - PASS4_TILE_W, Math.round(fullProj.uPx - PASS4_TILE_W * 0.5)),
        );
        roiV0 = Math.max(
          0,
          Math.min(TRAFFIC_RING_V1_INTRINSICS.height - PASS4_TILE_H, Math.round(fullProj.vPx - PASS4_TILE_H * 0.5)),
        );
      }
    }
    const roiCurr: [number, number] = [roiU0, roiV0];

    const currRgba = this.synthesizeNarrowFovRoiRgba(state, view, poseCurr, roiCurr);

    // Resolve t-1.0s and t-2.0s kept ring frames
    const shouldKeep =
      this.keptRing.length === 0 ||
      state.simTimeS - this.keptRing[this.keptRing.length - 1].simTimeS >=
        PASS4_SENSOR_DEFAULTS.keptFrameStepDtS - 1e-6;

    const prev1Entry: KeptRingFrame =
      this.keptRing.length >= 1
        ? this.keptRing[this.keptRing.length - 1]
        : {
            simTimeS: state.simTimeS,
            pose: poseCurr,
            roiOrigin: roiCurr,
            electronsCpu: srgbRgbaToElectronsCpu(currRgba, { noiseOff, applyPsf, frameSeed: frameIdx }),
          };
    const prev2Entry: KeptRingFrame =
      this.keptRing.length >= 2 ? this.keptRing[this.keptRing.length - 2] : prev1Entry;

    if (this.device && this.bindGroup && this.paramsBuf && this.roiRgbaTex && this.texPrev1 && this.texPrev2) {
      this.dispatchWebGpuCompute(
        this.device,
        currRgba,
        poseCurr,
        prev1Entry,
        prev2Entry,
        roiCurr,
        state.simTimeS,
        frameIdx,
        noiseOff,
        applyPsf,
      );
      if (shouldKeep) {
        // Keep compact CPU mirror at 1 Hz so prev1/prev2 r32float textures stay exact on Delta_t = 1.0 s cadence
        const keptElectrons = srgbRgbaToElectronsCpu(currRgba, {
          noiseOff,
          applyPsf,
          frameSeed: frameIdx,
        });
        this.pushKeptFrame({
          simTimeS: state.simTimeS,
          pose: poseCurr,
          roiOrigin: roiCurr,
          electronsCpu: keptElectrons,
        });
      }
      return this.latestPacket;
    }

    // Explicit deterministic FP32 CPU fallback path (WebGL2 mode)
    const res = computePass4NormalizedCpu({
      currRgba,
      prev1Electrons: prev1Entry.electronsCpu,
      prev2Electrons: prev2Entry.electronsCpu,
      poseCurr,
      posePrev1: prev1Entry.pose,
      posePrev2: prev2Entry.pose,
      roiCurr,
      roiPrev1: prev1Entry.roiOrigin,
      roiPrev2: prev2Entry.roiOrigin,
      noiseOff,
      applyPsf,
      frameSeed: frameIdx,
    });

    if (shouldKeep) {
      this.pushKeptFrame({
        simTimeS: state.simTimeS,
        pose: poseCurr,
        roiOrigin: roiCurr,
        electronsCpu: res.currElectrons,
      });
    }

    this.latestNormCopy.set(res.normalized);
    this.latestMeans = res.means;
    this.latestStd0 = res.std0;
    this.completedReadbacks++;
    this.latestPacket = {
      data: this.latestNormCopy,
      shape: [1, 5, 256, 256],
      simTimeS: state.simTimeS,
      frameIndex: frameIdx,
      roiU0,
      roiV0,
      noiseOff,
      computePath: "cpu-fp32-fallback",
      stats: { means: res.means, std0: res.std0 },
    };
    return this.latestPacket;
  }

  private pushKeptFrame(entry: KeptRingFrame): void {
    this.keptRing.push(entry);
    while (this.keptRing.length > 3) {
      this.keptRing.shift();
    }
  }

  private dispatchWebGpuCompute(
    device: GPUDevice,
    currRgba: Uint8Array,
    poseCurr: Pass4CameraPose,
    prev1: KeptRingFrame,
    prev2: KeptRingFrame,
    roiCurr: readonly [number, number],
    simTimeS: number,
    frameIdx: number,
    noiseOff: boolean,
    applyPsf: boolean,
  ): void {
    this.packUniforms(
      poseCurr,
      prev1.pose,
      prev2.pose,
      roiCurr,
      prev1.roiOrigin,
      prev2.roiOrigin,
      noiseOff,
      applyPsf,
      frameIdx,
    );

    device.queue.writeBuffer(this.paramsBuf!, 0, this.paramsScratch);
    device.queue.writeTexture(
      { texture: this.roiRgbaTex! },
      currRgba as Uint8Array<ArrayBuffer>,
      { bytesPerRow: PASS4_TILE_W * 4, rowsPerImage: PASS4_TILE_H },
      { width: PASS4_TILE_W, height: PASS4_TILE_H, depthOrArrayLayers: 1 },
    );
    device.queue.writeTexture(
      { texture: this.texPrev1! },
      prev1.electronsCpu as Float32Array<ArrayBuffer>,
      { bytesPerRow: PASS4_TILE_W * 4, rowsPerImage: PASS4_TILE_H },
      { width: PASS4_TILE_W, height: PASS4_TILE_H, depthOrArrayLayers: 1 },
    );
    device.queue.writeTexture(
      { texture: this.texPrev2! },
      prev2.electronsCpu as Float32Array<ArrayBuffer>,
      { bytesPerRow: PASS4_TILE_W * 4, rowsPerImage: PASS4_TILE_H },
      { width: PASS4_TILE_W, height: PASS4_TILE_H, depthOrArrayLayers: 1 },
    );

    const encoder = device.createCommandEncoder({ label: "pass4-compute-encoder" });
    const pass = encoder.beginComputePass({ label: "pass4-compute" });
    pass.setBindGroup(0, this.bindGroup!);

    // 1. sRGB -> Rec.709 -> PSF -> electrons (+ optional EMVA-1288 noise)
    pass.setPipeline(this.pipeSrgb!);
    pass.dispatchWorkgroups(PASS4_TILE_W / 16, PASS4_TILE_H / 16, 1);

    // 2. 2-layer 6-DoF WGS-84 + sky warp with manual FP32 textureLoad bilinear & overlap M
    pass.setPipeline(this.pipeWarp!);
    pass.dispatchWorkgroups(PASS4_TILE_W / 16, PASS4_TILE_H / 16, 1);

    // 3a. Exact FP32 per-channel mean and two-pass channel-0 std (clamped >= 1.0)
    pass.setPipeline(this.pipeStats!);
    pass.dispatchWorkgroups(1, 1, 1);

    // 3b. Single FP32 normalize_input clamped [-20, 20]
    pass.setPipeline(this.pipeNorm!);
    pass.dispatchWorkgroups(PASS4_TILE_N / 256, 1, 1);
    pass.end();

    // Select an idle, strictly unmapped slot from the 3-slot async Staging Ring Buffer
    let selectedSlot: StagingSlot | null = null;
    for (let k = 0; k < this.stagingRing.length; k++) {
      const idx = (this.nextStagingIdx + k) % this.stagingRing.length;
      const cand = this.stagingRing[idx];
      if (
        cand.state === "idle" &&
        cand.buffer.mapState === "unmapped" &&
        cand.statsBuffer.mapState === "unmapped"
      ) {
        selectedSlot = cand;
        this.nextStagingIdx = (idx + 1) % this.stagingRing.length;
        break;
      }
    }

    if (selectedSlot) {
      encoder.copyBufferToBuffer(
        this.normOutBuf!,
        0,
        selectedSlot.buffer,
        0,
        PASS4_TENSOR_BYTES,
      );
      encoder.copyBufferToBuffer(this.statsBuf!, 0, selectedSlot.statsBuffer, 0, 32);
    } else {
      this.skippedReadbacks++;
    }

    device.queue.submit([encoder.finish()]);

    // Non-blocking async mapAsync (NEVER awaited inside requestAnimationFrame!)
    if (selectedSlot) {
      const slot = selectedSlot;
      slot.state = "mapping";
      void Promise.all([
        slot.buffer.mapAsync(GPUMapMode.READ),
        slot.statsBuffer.mapAsync(GPUMapMode.READ),
      ])
        .then(() => {
          try {
            if (!this.disposed) {
              const mapped = new Float32Array(slot.buffer.getMappedRange());
              const st = new Float32Array(slot.statsBuffer.getMappedRange());
              this.latestNormCopy.set(mapped);
              this.latestMeans = [st[0], st[1], st[2], st[3], st[4]];
              this.latestStd0 = Math.max(st[5], 1.0);
              this.completedReadbacks++;
              this.latestPacket = {
                data: this.latestNormCopy,
                shape: [1, 5, 256, 256],
                simTimeS,
                frameIndex: frameIdx,
                roiU0: roiCurr[0],
                roiV0: roiCurr[1],
                noiseOff,
                computePath: "webgpu-compute",
                stats: {
                  means: [...this.latestMeans] as [number, number, number, number, number],
                  std0: this.latestStd0,
                },
              };
            }
          } finally {
            try {
              if (slot.buffer.mapState === "mapped") slot.buffer.unmap();
            } catch {
              // ignore unmap error during teardown
            }
            try {
              if (slot.statsBuffer.mapState === "mapped") slot.statsBuffer.unmap();
            } catch {
              // ignore unmap error during teardown
            }
            slot.state = "idle";
          }
        })
        .catch(() => {
          try {
            if (slot.buffer.mapState === "mapped") slot.buffer.unmap();
          } catch {
            // ignore
          }
          try {
            if (slot.statsBuffer.mapState === "mapped") slot.statsBuffer.unmap();
          } catch {
            // ignore
          }
          slot.state = "idle";
        });
    }
  }

  /**
   * Explicit async parity helper for V2-M1-4 Playwright E2E harness:
   * runs either real WebGPU compute (when available) or CPU FP32 fallback on caller-supplied
   * 256x256 RGBA + previous electron frames + 6-DoF poses and resolves the normalized `[1,5,256,256]` tensor.
   */
  async runExplicitParityAsync(input: Pass4CpuInput): Promise<Pass4TensorPacket> {
    if (!this.device || !this.bindGroup) {
      const cpu = computePass4NormalizedCpu(input);
      return {
        data: cpu.normalized,
        shape: [1, 5, 256, 256],
        simTimeS: 0,
        frameIndex: input.frameSeed ?? 0,
        roiU0: input.roiCurr[0],
        roiV0: input.roiCurr[1],
        noiseOff: input.noiseOff,
        computePath: "cpu-fp32-fallback",
        stats: { means: cpu.means, std0: cpu.std0 },
      };
    }

    const device = this.device;
    this.packUniforms(
      input.poseCurr,
      input.posePrev1,
      input.posePrev2,
      input.roiCurr,
      input.roiPrev1,
      input.roiPrev2,
      input.noiseOff,
      input.applyPsf,
      input.frameSeed ?? 0,
    );

    device.queue.writeBuffer(this.paramsBuf!, 0, this.paramsScratch);
    device.queue.writeTexture(
      { texture: this.roiRgbaTex! },
      input.currRgba as Uint8Array<ArrayBuffer>,
      { bytesPerRow: PASS4_TILE_W * 4, rowsPerImage: PASS4_TILE_H },
      { width: PASS4_TILE_W, height: PASS4_TILE_H, depthOrArrayLayers: 1 },
    );
    device.queue.writeTexture(
      { texture: this.texPrev1! },
      input.prev1Electrons as Float32Array<ArrayBuffer>,
      { bytesPerRow: PASS4_TILE_W * 4, rowsPerImage: PASS4_TILE_H },
      { width: PASS4_TILE_W, height: PASS4_TILE_H, depthOrArrayLayers: 1 },
    );
    device.queue.writeTexture(
      { texture: this.texPrev2! },
      input.prev2Electrons as Float32Array<ArrayBuffer>,
      { bytesPerRow: PASS4_TILE_W * 4, rowsPerImage: PASS4_TILE_H },
      { width: PASS4_TILE_W, height: PASS4_TILE_H, depthOrArrayLayers: 1 },
    );

    const readNorm = device.createBuffer({
      size: PASS4_TENSOR_BYTES,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const readStats = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const enc = device.createCommandEncoder({ label: "pass4-parity-encoder" });
    const pass = enc.beginComputePass();
    pass.setBindGroup(0, this.bindGroup!);
    pass.setPipeline(this.pipeSrgb!);
    pass.dispatchWorkgroups(PASS4_TILE_W / 16, PASS4_TILE_H / 16, 1);
    pass.setPipeline(this.pipeWarp!);
    pass.dispatchWorkgroups(PASS4_TILE_W / 16, PASS4_TILE_H / 16, 1);
    pass.setPipeline(this.pipeStats!);
    pass.dispatchWorkgroups(1, 1, 1);
    pass.setPipeline(this.pipeNorm!);
    pass.dispatchWorkgroups(PASS4_TILE_N / 256, 1, 1);
    pass.end();

    enc.copyBufferToBuffer(this.normOutBuf!, 0, readNorm, 0, PASS4_TENSOR_BYTES);
    enc.copyBufferToBuffer(this.statsBuf!, 0, readStats, 0, 32);
    device.queue.submit([enc.finish()]);

    await Promise.all([readNorm.mapAsync(GPUMapMode.READ), readStats.mapAsync(GPUMapMode.READ)]);
    const data = new Float32Array(readNorm.getMappedRange()).slice();
    const st = new Float32Array(readStats.getMappedRange()).slice();
    readNorm.unmap();
    readStats.unmap();
    readNorm.destroy();
    readStats.destroy();

    return {
      data,
      shape: [1, 5, 256, 256],
      simTimeS: 0,
      frameIndex: input.frameSeed ?? 0,
      roiU0: input.roiCurr[0],
      roiV0: input.roiCurr[1],
      noiseOff: input.noiseOff,
      computePath: "webgpu-compute",
      stats: { means: [st[0], st[1], st[2], st[3], st[4]], std0: Math.max(st[5], 1.0) },
    };
  }

  dispose(): void {
    this.disposed = true;
    this.paramsBuf?.destroy();
    this.roiRgbaTex?.destroy();
    this.texPrev1?.destroy();
    this.texPrev2?.destroy();
    this.currElectronsBuf?.destroy();
    this.rawStackBuf?.destroy();
    this.statsBuf?.destroy();
    this.normOutBuf?.destroy();
    for (const s of this.stagingRing) {
      s.buffer.destroy();
      s.statsBuffer.destroy();
    }
    this.stagingRing = [];
    void this.rhi;
  }
}
