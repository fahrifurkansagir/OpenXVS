/**
 * OpenXVS Live Deck Stage-1 ONNX Runtime WebGPU / WASM-SIMD Worker & Resident Runner.
 *
 * Enforces SP1_LIVEDECK_DESIGN.md §4.4 & Review Focus #4:
 *   - Dynamically imports verified non-JSEP `onnxruntime-web/webgpu` (Stage-1 async hydration)
 *   - `enableGraphCapture: false` (prevents 300-frame GPU heap/graph capture instability)
 *   - Single resident `InferenceSession` created once and reused across all ticks
 *   - 3-slot async `StagingRingBuffer` decoupling `requestAnimationFrame` from inference
 *   - Explicit visible backend hierarchy (Charter §7.5 — ZERO silent fallbacks, NEVER ORT WebGL EP):
 *       1. WebGPU + shader-f16 -> `d1_kf_k3_fp16.onnx` (`AI: ORT-WebGPU-FP16`)
 *       2. WebGPU without f16  -> `d1_kf_k3_fp32.onnx` (`AI: ORT-WebGPU-FP32 (NO F16)`)
 *       3. WebGL2 / no WebGPU / after `device.lost` -> `d1_kf_k3_fp32.onnx` (`AI: ORT-WASM-SIMD`)
 */

import {
  getActiveModelEntry,
  getXvsD2ModelEntry,
  loadModelRegistry,
  loadVerifiedModelBytes,
  type ActiveModelEntry,
  type CacheSource,
  type ModelRegistryDocument,
  type XvsD2ModelEntry,
} from "./model_registry.ts";

export type AiBackendKind = "webgpu-fp16" | "webgpu-fp32" | "wasm-simd";
export type DetectorMode = "d0" | "d1_kf" | "xvs_d2";

export interface AiBackendSelection {
  readonly kind: AiBackendKind;
  readonly executionProvider: "webgpu" | "wasm";
  readonly precision: "fp16" | "fp32";
  readonly aiBadge:
    | "AI: ORT-WebGPU-FP16"
    | "AI: ORT-WebGPU-FP32 (NO F16)"
    | "AI: ORT-WASM-SIMD";
  readonly degradedReason: string | null;
}

export interface DetectionPeak {
  readonly u: number;
  readonly v: number;
  readonly score: number;
  readonly logW: number;
  readonly logH: number;
}

export interface LiveDetectionBox {
  readonly u: number;
  readonly v: number;
  readonly widthPx: number;
  readonly heightPx: number;
  readonly score: number;
  readonly sigmaPx: number;
  readonly modelId: string;
}

export interface DetectorInferenceResult {
  readonly frameIndex: number;
  readonly backendKind: AiBackendKind;
  readonly aiBadge: AiBackendSelection["aiBadge"];
  readonly inferenceMs: number;
  readonly allFinite: boolean;
  readonly maxHeatmapScore: number;
  readonly peaks: readonly DetectionPeak[];
  readonly detections?: readonly LiveDetectionBox[];
  readonly modelId?: string;
  readonly detectorMode?: DetectorMode;
}

export type OrtInferenceResult = DetectorInferenceResult;

export interface AiInitStatus {
  readonly ready: boolean;
  readonly modelId: string;
  readonly modelSha256: string;
  readonly backend: AiBackendSelection;
  readonly cacheSource: CacheSource;
  readonly opfsWarning: string | null;
  readonly enableGraphCapture: false;
  readonly sessionReusedCount: number;
  readonly xvsD2Available?: boolean;
  readonly xvsD2Status?: string | null;
}

export const D1_INPUT_SHAPE = [1, 5, 256, 256] as const;
export const D1_INPUT_FLOATS = 1 * 5 * 256 * 256;
export const D2_INPUT_SHAPE = [1, 6, 256, 256] as const;
export const D2_INPUT_FLOATS = 1 * 6 * 256 * 256;
export const D2_HIDDEN_SHAPE = [1, 24, 64, 64] as const;
export const D2_HIDDEN_FLOATS = 1 * 24 * 64 * 64;
export const D2_FRAME_FLOATS = 256 * 256;
export const D2_OUT_W = 128;
export const D2_OUT_H = 128;
export const STAGING_RING_SLOTS = 3;

export function selectAiBackend(opts: {
  hasWebGpu: boolean;
  hasShaderF16: boolean;
  forceWasmSimd?: boolean;
  fallbackReason?: string | null;
}): AiBackendSelection {
  if (!opts.forceWasmSimd && opts.hasWebGpu) {
    if (opts.hasShaderF16) {
      return {
        kind: "webgpu-fp16",
        executionProvider: "webgpu",
        precision: "fp16",
        aiBadge: "AI: ORT-WebGPU-FP16",
        degradedReason: null,
      };
    }
    return {
      kind: "webgpu-fp32",
      executionProvider: "webgpu",
      precision: "fp32",
      aiBadge: "AI: ORT-WebGPU-FP32 (NO F16)",
      degradedReason: "WebGPU adapter lacks shader-f16; using FP32 D1-KF ONNX",
    };
  }
  return {
    kind: "wasm-simd",
    executionProvider: "wasm",
    precision: "fp32",
    aiBadge: "AI: ORT-WASM-SIMD",
    degradedReason:
      opts.fallbackReason ?? "WebGPU unavailable or WebGL2 mode active; using ORT WASM-SIMD EP",
  };
}

/**
 * 3-slot asynchronous Staging Ring Buffer preventing `requestAnimationFrame` stalls.
 */
export class StagingRingBuffer {
  private readonly buffers: Float32Array[];
  private readonly states: ("free" | "ready" | "in-flight")[];
  private writeCursor = 0;
  private readCursor = 0;

  constructor(slotCount = STAGING_RING_SLOTS, elementCount = D1_INPUT_FLOATS) {
    this.buffers = Array.from({ length: slotCount }, () => new Float32Array(elementCount));
    this.states = Array.from({ length: slotCount }, () => "free");
  }

  acquireWriteSlot(): { slotIndex: number; buffer: Float32Array } | null {
    for (let i = 0; i < this.buffers.length; i++) {
      const idx = (this.writeCursor + i) % this.buffers.length;
      if (this.states[idx] === "free") {
        this.writeCursor = (idx + 1) % this.buffers.length;
        return { slotIndex: idx, buffer: this.buffers[idx] };
      }
    }
    return null;
  }

  commitWriteSlot(slotIndex: number): void {
    this.states[slotIndex] = "ready";
  }

  acquireReadySlot(): { slotIndex: number; buffer: Float32Array } | null {
    for (let i = 0; i < this.buffers.length; i++) {
      const idx = (this.readCursor + i) % this.buffers.length;
      if (this.states[idx] === "ready") {
        this.states[idx] = "in-flight";
        this.readCursor = (idx + 1) % this.buffers.length;
        return { slotIndex: idx, buffer: this.buffers[idx] };
      }
    }
    return null;
  }

  releaseSlot(slotIndex: number): void {
    this.states[slotIndex] = "free";
  }

  get slotCount(): number {
    return this.buffers.length;
  }
}

/**
 * 3x3 max-pool peak extraction matching `src/xvs/detection/d1_model.py:extract_d1_detections`.
 * Enforces `Number.isFinite` on every heatmap/offset/log_scale element (zero NaN/Inf leak).
 */
export function extractDetectionPeaks(
  heatmap: Float32Array,
  offset: Float32Array,
  logScale: Float32Array,
  width = 256,
  height = 256,
  scoreThreshold = 0.35,
  maxPeaks = 32,
): { allFinite: boolean; maxHeatmapScore: number; peaks: DetectionPeak[] } {
  const hw = width * height;
  let allFinite = true;
  let maxScore = -Infinity;

  for (let i = 0; i < hw; i++) {
    const v = heatmap[i];
    if (!Number.isFinite(v)) {
      allFinite = false;
      continue;
    }
    if (v > maxScore) {
      maxScore = v;
    }
  }
  for (let i = 0; i < 2 * hw; i++) {
    if (!Number.isFinite(offset[i]) || !Number.isFinite(logScale[i])) {
      allFinite = false;
      break;
    }
  }
  if (!Number.isFinite(maxScore)) {
    maxScore = 0;
  }

  const candidates: DetectionPeak[] = [];
  // Ignore 2-px boundary border artifacts matching d1_model.py:488-493
  for (let y = 2; y < height - 2; y++) {
    const rowOff = y * width;
    for (let x = 2; x < width - 2; x++) {
      const idx = rowOff + x;
      const sc = heatmap[idx];
      if (!Number.isFinite(sc) || sc < scoreThreshold) {
        continue;
      }
      // 3x3 max-pool NMS
      let isPeak = true;
      for (let dy = -1; dy <= 1 && isPeak; dy++) {
        const nRow = (y + dy) * width;
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          if (heatmap[nRow + (x + dx)] > sc) {
            isPeak = false;
            break;
          }
        }
      }
      if (!isPeak) continue;

      const du = offset[idx];
      const dv = offset[hw + idx];
      const logW = logScale[idx];
      const logH = logScale[hw + idx];
      if (
        Number.isFinite(du) &&
        Number.isFinite(dv) &&
        Number.isFinite(logW) &&
        Number.isFinite(logH)
      ) {
        candidates.push({
          u: x + du,
          v: y + dv,
          score: sc,
          logW,
          logH,
        });
      }
    }
  }

  candidates.sort((a, b) => b.score - a.score);
  return {
    allFinite,
    maxHeatmapScore: maxScore,
    peaks: candidates.slice(0, maxPeaks),
  };
}

function sigmoidSafe(z: number): number {
  if (z >= 0) {
    const e = Math.exp(-z);
    return 1 / (1 + e);
  }
  const e = Math.exp(z);
  return e / (1 + e);
}

/**
 * Decode XVS-D2 stride-2 heads (`hm_logits`, `offset`, `log_scale`, `log_var`, `peak_mask`)
 * into `LiveDetectionBox[]` in `256x256` sensor pixel coordinates with strict finiteness guard.
 */
export function extractXvsD2Detections(
  hmLogits: Float32Array,
  offset: Float32Array,
  logScale: Float32Array,
  logVar: Float32Array,
  peakMask: Float32Array,
  width = D2_OUT_W,
  height = D2_OUT_H,
  stride = 2,
  scoreThreshold = 0.25,
  maxDetections = 16,
  modelId = "xvs_d2_web_fp16",
): {
  allFinite: boolean;
  maxHeatmapScore: number;
  peaks: DetectionPeak[];
  detections: LiveDetectionBox[];
} {
  const hw = width * height;
  let allFinite = true;
  let maxScore = 0;

  for (let i = 0; i < hw; i++) {
    const z = hmLogits[i];
    const pm = peakMask[i];
    if (!Number.isFinite(z) || !Number.isFinite(pm)) {
      allFinite = false;
      continue;
    }
    const p = sigmoidSafe(z);
    if (p > maxScore) {
      maxScore = p;
    }
  }
  for (let i = 0; i < 2 * hw; i++) {
    if (!Number.isFinite(offset[i]) || !Number.isFinite(logScale[i])) {
      allFinite = false;
      break;
    }
  }
  for (let i = 0; i < 3 * hw; i++) {
    if (!Number.isFinite(logVar[i])) {
      allFinite = false;
      break;
    }
  }

  const boxes: LiveDetectionBox[] = [];
  const peaks: DetectionPeak[] = [];

  for (let y = 1; y < height - 1; y++) {
    const rowOff = y * width;
    for (let x = 1; x < width - 1; x++) {
      const idx = rowOff + x;
      if (peakMask[idx] <= 0.5) {
        continue;
      }
      const z = hmLogits[idx];
      if (!Number.isFinite(z)) {
        continue;
      }
      const score = sigmoidSafe(z);
      if (score < scoreThreshold) {
        continue;
      }

      const du = offset[idx];
      const dv = offset[hw + idx];
      const lnS = Math.max(-2.0, Math.min(5.0, logScale[idx]));
      const lnR = Math.max(-2.0, Math.min(2.0, logScale[hw + idx]));
      const lu = Math.max(-6.0, Math.min(4.0, logVar[idx]));
      const lv = Math.max(-6.0, Math.min(4.0, logVar[hw + idx]));

      const u = (x + du + 0.5) * stride;
      const v = (y + dv + 0.5) * stride;
      const scale = Math.exp(lnS);
      const aspect = Math.exp(0.5 * lnR);
      const widthPx = Math.max(2.0, scale * aspect);
      const heightPx = Math.max(2.0, scale / aspect);
      const sigmaPx = Math.exp(0.25 * (lu + lv)) * stride;

      if (
        Number.isFinite(u) &&
        Number.isFinite(v) &&
        Number.isFinite(widthPx) &&
        Number.isFinite(heightPx) &&
        Number.isFinite(sigmaPx)
      ) {
        boxes.push({
          u,
          v,
          widthPx,
          heightPx,
          score,
          sigmaPx,
          modelId,
        });
        peaks.push({
          u,
          v,
          score,
          logW: Math.log(widthPx),
          logH: Math.log(heightPx),
        });
      }
    }
  }

  boxes.sort((a, b) => b.score - a.score);
  peaks.sort((a, b) => b.score - a.score);
  return {
    allFinite,
    maxHeatmapScore: maxScore,
    peaks: peaks.slice(0, maxDetections),
    detections: boxes.slice(0, maxDetections),
  };
}

function normalizeFrame256(src: Float32Array): Float32Array {
  const out = new Float32Array(D2_FRAME_FLOATS);
  let sum = 0;
  let maxAbs = 0;
  for (let i = 0; i < D2_FRAME_FLOATS; i++) {
    const val = Number.isFinite(src[i]) ? src[i] : 0;
    sum += val;
    const a = Math.abs(val);
    if (a > maxAbs) maxAbs = a;
  }
  // If already scene-normalized (e.g. from D1-KF channel 0), clamp cleanly to [-5, +5]
  if (maxAbs <= 25.0) {
    for (let i = 0; i < D2_FRAME_FLOATS; i++) {
      const v = Number.isFinite(src[i]) ? src[i] : 0;
      out[i] = Math.max(-5.0, Math.min(5.0, v));
    }
    return out;
  }
  const mean = sum / D2_FRAME_FLOATS;
  let sq = 0;
  for (let i = 0; i < D2_FRAME_FLOATS; i++) {
    const d = (Number.isFinite(src[i]) ? src[i] : mean) - mean;
    sq += d * d;
  }
  const std = Math.max(8.0, Math.sqrt(sq / D2_FRAME_FLOATS));
  for (let i = 0; i < D2_FRAME_FLOATS; i++) {
    const v = ((Number.isFinite(src[i]) ? src[i] : mean) - mean) / std;
    out[i] = Math.max(-5.0, Math.min(5.0, v));
  }
  return out;
}

/**
 * Convert an RGBA ImageData buffer (`w x h`) to Rec.709 luminance (`256 x 256` Float32Array).
 * Exact 1:1 pixel copy when `w === 256 && h === 256` (zero downscale shrinkage on native AOT crops).
 */
export function rgbaToGray256(rgba: Uint8ClampedArray, w: number, h: number): Float32Array {
  const out = new Float32Array(D2_FRAME_FLOATS);
  if (w <= 0 || h <= 0 || rgba.length < w * h * 4) {
    return out;
  }
  if (w === 256 && h === 256) {
    for (let i = 0; i < D2_FRAME_FLOATS; i++) {
      const o = i * 4;
      out[i] = 0.2126 * rgba[o] + 0.7152 * rgba[o + 1] + 0.0722 * rgba[o + 2];
    }
    return out;
  }
  for (let y = 0; y < 256; y++) {
    const sy = Math.min(h - 1, Math.floor(((y + 0.5) * h) / 256));
    const rowOff = sy * w;
    const dstRow = y * 256;
    for (let x = 0; x < 256; x++) {
      const sx = Math.min(w - 1, Math.floor(((x + 0.5) * w) / 256));
      const o = (rowOff + sx) * 4;
      out[dstRow + x] = 0.2126 * rgba[o] + 0.7152 * rgba[o + 1] + 0.0722 * rgba[o + 2];
    }
  }
  return out;
}

/** 4x4 box-averaged downsample from 256x256 to 64x64 for fast integer SAD GMC. */
export function downsample256To64(plane256: Float32Array): Float32Array {
  const out = new Float32Array(64 * 64);
  for (let y = 0; y < 64; y++) {
    const yBase = y * 4;
    const outRow = y * 64;
    for (let x = 0; x < 64; x++) {
      const xBase = x * 4;
      let acc = 0;
      for (let dy = 0; dy < 4; dy++) {
        const r = (yBase + dy) * 256 + xBase;
        acc += plane256[r] + plane256[r + 1] + plane256[r + 2] + plane256[r + 3];
      }
      out[outRow + x] = acc * 0.0625;
    }
  }
  return out;
}

/**
 * Fast 64x64 SAD block translation estimator between consecutive frames.
 * Returns `(dxPx, dyPx)` in 256x256 pixel coordinates such that shifting `prev` by
 * `(dxPx, dyPx)` aligns background sky/clouds onto `curr`.
 */
export function estimateSadTranslation64(
  prev64: Float32Array,
  curr64: Float32Array,
  maxShift64 = 6,
): { readonly dxPx: number; readonly dyPx: number } {
  const r = Math.max(1, Math.min(12, Math.floor(maxShift64)));
  const yStart = r;
  const yEnd = 64 - r;
  const xStart = r;
  const xEnd = 64 - r;

  let bestDx = 0;
  let bestDy = 0;
  let bestSad = Number.POSITIVE_INFINITY;

  for (let dy = -r; dy <= r; dy++) {
    for (let dx = -r; dx <= r; dx++) {
      let sad = 0;
      for (let y = yStart; y < yEnd; y++) {
        const currRow = y * 64;
        const prevRow = (y - dy) * 64;
        for (let x = xStart; x < xEnd; x++) {
          sad += Math.abs(curr64[currRow + x] - prev64[prevRow + (x - dx)]);
        }
      }
      // Small tie-break toward zero shift on flat skies
      const score = sad + (Math.abs(dx) + Math.abs(dy)) * 1e-4;
      if (score < bestSad) {
        bestSad = score;
        bestDx = dx;
        bestDy = dy;
      }
    }
  }

  return { dxPx: bestDx * 4, dyPx: bestDy * 4 };
}

/** Integer-pixel translation warp of a 256x256 plane with replicate border clamp. */
export function shiftPlane256(src: Float32Array, dxPx: number, dyPx: number): Float32Array {
  const idx = Math.round(dxPx);
  const idy = Math.round(dyPx);
  if (idx === 0 && idy === 0) {
    return src;
  }
  const out = new Float32Array(D2_FRAME_FLOATS);
  for (let y = 0; y < 256; y++) {
    const sy = Math.max(0, Math.min(255, y - idy));
    const srcRow = sy * 256;
    const dstRow = y * 256;
    for (let x = 0; x < 256; x++) {
      const sx = Math.max(0, Math.min(255, x - idx));
      out[dstRow + x] = src[srcRow + sx];
    }
  }
  return out;
}

/** Integer-pixel translation warp of XVS-D2 ConvGRU hidden state `[1, 24, 64, 64]`. */
export function shiftHiddenState64(h: Float32Array, dxPx: number, dyPx: number): Float32Array {
  const sdx = Math.round(dxPx / 4);
  const sdy = Math.round(dyPx / 4);
  if (sdx === 0 && sdy === 0) {
    return h;
  }
  const out = new Float32Array(D2_HIDDEN_FLOATS);
  const plane = 64 * 64;
  for (let c = 0; c < 24; c++) {
    const base = c * plane;
    for (let y = 0; y < 64; y++) {
      const sy = Math.max(0, Math.min(63, y - sdy));
      const srcRow = base + sy * 64;
      const dstRow = base + y * 64;
      for (let x = 0; x < 64; x++) {
        const sx = Math.max(0, Math.min(63, x - sdx));
        out[dstRow + x] = h[srcRow + sx];
      }
    }
  }
  return out;
}

type OrtModule = typeof import("onnxruntime-web/webgpu");
type OrtSession = import("onnxruntime-web/webgpu").InferenceSession;

/**
 * Single resident ORT session manager supporting D1-KF (default M1) + lazy XVS-D2 (6-ch + ConvGRU)
 * with explicit WebGPU -> WASM-SIMD failover on `device.lost`.
 */
export class OrtDetectorRunner {
  private ortMod: OrtModule | null = null;
  private registry: ModelRegistryDocument | null = null;
  private session: OrtSession | null = null;
  private d2Session: OrtSession | null = null;
  private activeModel: ActiveModelEntry | null = null;
  private d2Model: XvsD2ModelEntry | null = null;
  private d2Backend: AiBackendSelection | null = null;
  private status: AiInitStatus | null = null;
  private baseUrl = "./";
  private inferenceCount = 0;
  private readonly d2FrameRing: Float32Array[] = [];
  private readonly d2InputScratch = new Float32Array(D2_INPUT_FLOATS);
  private readonly hPrevWarped = new Float32Array(D2_HIDDEN_FLOATS);
  private d2Prev64: Float32Array | null = null;
  private d2ResetCount = 0;
  private d2LastTimestampS: number | null = null;
  /** Strict single-flight serialization lock (ORT WebGPU session.run() is non-reentrant). */
  private opLock: Promise<void> = Promise.resolve();

  private _serialize<T>(fn: () => Promise<T>): Promise<T> {
    const next = this.opLock.then(fn, fn);
    this.opLock = next.then(
      () => undefined,
      () => undefined,
    );
    return next;
  }

  resetXvsD2State(): void {
    this.hPrevWarped.fill(0);
    this.d2FrameRing.length = 0;
    this.d2Prev64 = null;
    this.d2LastTimestampS = null;
    this.d2ResetCount += 1;
  }

  /** Explicit Tab 3 temporal state reset when switching Real-Sky clips or user videos. */
  resetXvsD2TemporalState(): void {
    this.resetXvsD2State();
  }

  getXvsD2TemporalStateSnapshot(): {
    readonly ringLength: number;
    readonly hiddenNorm: number;
    readonly hasPrev64: boolean;
    readonly resetCount: number;
  } {
    let sumSq = 0;
    for (let i = 0; i < this.hPrevWarped.length; i++) {
      const v = this.hPrevWarped[i];
      sumSq += v * v;
    }
    return {
      ringLength: this.d2FrameRing.length,
      hiddenNorm: Math.sqrt(sumSq),
      hasPrev64: this.d2Prev64 !== null,
      resetCount: this.d2ResetCount,
    };
  }

  async init(opts: {
    baseUrl?: string;
    hasWebGpu: boolean;
    hasShaderF16: boolean;
    forceWasmSimd?: boolean;
    fallbackReason?: string | null;
    wasmPaths?: string;
  }): Promise<AiInitStatus> {
    this.baseUrl = opts.baseUrl ?? "./";
    const registry = await loadModelRegistry(this.baseUrl);
    this.registry = registry;
    const active = getActiveModelEntry(registry);
    this.activeModel = active;
    this.d2Model = getXvsD2ModelEntry(registry);

    const desired = selectAiBackend({
      hasWebGpu: opts.hasWebGpu,
      hasShaderF16: opts.hasShaderF16,
      forceWasmSimd: opts.forceWasmSimd,
      fallbackReason: opts.fallbackReason,
    });

    return this._createResidentSession(active, desired, opts.wasmPaths);
  }

  private async _createResidentSession(
    active: ActiveModelEntry,
    selection: AiBackendSelection,
    wasmPaths?: string,
  ): Promise<AiInitStatus> {
    if (!this.ortMod) {
      // Dynamic Stage-1 import of verified non-JSEP WebGPU/WASM bundle (never ORT WebGL EP)
      this.ortMod = await import("onnxruntime-web/webgpu");
      if (wasmPaths) {
        this.ortMod.env.wasm.wasmPaths = wasmPaths;
      }
      this.ortMod.env.wasm.simd = true;
      // Avoid Emscripten pthread spin-join deadlock on main thread under crossOriginIsolated
      this.ortMod.env.wasm.numThreads = 1;
      this.ortMod.env.wasm.proxy = false;
    }

    if (this.session) {
      await this.session.release();
      this.session = null;
    }
    if (this.d2Session) {
      await this.d2Session.release();
      this.d2Session = null;
      this.d2Backend = null;
      this.resetXvsD2State();
    }

    const artifact =
      selection.precision === "fp16" ? active.artifacts.fp16 : active.artifacts.fp32;
    const verified = await loadVerifiedModelBytes(artifact, this.baseUrl);

    try {
      this.session = await this.ortMod.InferenceSession.create(verified.bytes, {
        executionProviders: [selection.executionProvider],
        // Hard lock (SP1_LIVEDECK_DESIGN.md §4.4 & Review Focus #4): NEVER enable graph capture
        enableGraphCapture: false,
      });

      this.status = {
        ready: true,
        modelId: active.id,
        modelSha256: verified.sha256,
        backend: selection,
        cacheSource: verified.cacheSource,
        opfsWarning: verified.opfsWarning,
        enableGraphCapture: false,
        sessionReusedCount: 0,
        xvsD2Available: this.d2Model !== null,
        xvsD2Status: this.d2Model?.status ?? null,
      };
      return this.status;
    } catch (err) {
      // Explicit visible fallback to WASM-SIMD if WebGPU session creation fails (NEVER silent, NEVER WebGL EP)
      if (selection.executionProvider === "webgpu") {
        const reason = err instanceof Error ? err.message : String(err);
        const wasmFallback = selectAiBackend({
          hasWebGpu: false,
          hasShaderF16: false,
          forceWasmSimd: true,
          fallbackReason: `WebGPU ORT session init failed (${reason}); fell back to ORT-WASM-SIMD`,
        });
        return this._createResidentSession(active, wasmFallback, wasmPaths);
      }
      throw err;
    }
  }

  private async _ensureXvsD2Session(): Promise<OrtSession> {
    if (this.d2Session) {
      return this.d2Session;
    }
    if (!this.ortMod || !this.status) {
      throw new Error("OrtDetectorRunner._ensureXvsD2Session called before init()");
    }
    if (!this.d2Model && this.registry) {
      this.d2Model = getXvsD2ModelEntry(this.registry);
    }
    if (!this.d2Model) {
      throw new Error("XVS-D2 model entry ('xvs_d2_web_fp16') not registered in registry.json");
    }

    const verified = await loadVerifiedModelBytes(this.d2Model.artifacts.fp16, this.baseUrl);
    const ep = this.status.backend.executionProvider;
    try {
      this.d2Session = await this.ortMod.InferenceSession.create(verified.bytes, {
        executionProviders: [ep],
        enableGraphCapture: false,
      });
      this.d2Backend = this.status.backend;
    } catch (err) {
      if (ep !== "webgpu") {
        throw err;
      }
      // Explicit visible Charter §7.5 fallback to WASM-SIMD (ZERO silent fallback)
      const reason = err instanceof Error ? err.message : String(err);
      const wasmFallback = selectAiBackend({
        hasWebGpu: false,
        hasShaderF16: false,
        forceWasmSimd: true,
        fallbackReason: `XVS-D2 WebGPU init failed (${reason}); explicit WASM-SIMD fallback`,
      });
      this.d2Session = await this.ortMod.InferenceSession.create(verified.bytes, {
        executionProviders: ["wasm"],
        enableGraphCapture: false,
      });
      this.d2Backend = wasmFallback;
      this.status = {
        ...this.status,
        backend: wasmFallback,
      };
    }
    return this.d2Session;
  }

  async handleDeviceLost(reason: string): Promise<AiInitStatus> {
    return this._serialize(async () => {
      if (!this.activeModel) {
        throw new Error("OrtDetectorRunner not initialized before handleDeviceLost");
      }
      const fallback = selectAiBackend({
        hasWebGpu: false,
        hasShaderF16: false,
        forceWasmSimd: true,
        fallbackReason: `WebGPU device.lost (${reason}); transitioned visibly to ORT-WASM-SIMD`,
      });
      return this._createResidentSession(this.activeModel, fallback);
    });
  }

  async runNormalized(
    xNorm: Float32Array,
    frameIndex: number,
    scoreThreshold = 0.35,
  ): Promise<DetectorInferenceResult> {
    return this._serialize(async () => {
      if (!this.session || !this.ortMod || !this.status) {
        throw new Error("OrtDetectorRunner.runNormalized called before init()");
      }
      if (xNorm.length !== D1_INPUT_FLOATS) {
        throw new Error(`Expected ${D1_INPUT_FLOATS} floats for x_norm, got ${xNorm.length}`);
      }

      const t0 = performance.now();
      const inputTensor = new this.ortMod.Tensor("float32", xNorm, [1, 5, 256, 256]);
      let outputs: Record<string, import("onnxruntime-web/webgpu").Tensor> | null = null;
      try {
        outputs = await this.session.run({ x_norm: inputTensor });
        const hmData = outputs["heatmap"].data as Float32Array;
        const offData = outputs["offset"].data as Float32Array;
        const lscData = outputs["log_scale"].data as Float32Array;

        const extracted = extractDetectionPeaks(
          hmData,
          offData,
          lscData,
          256,
          256,
          scoreThreshold,
        );
        const inferenceMs = performance.now() - t0;

        this.inferenceCount += 1;
        this.status = {
          ...this.status,
          sessionReusedCount: this.inferenceCount,
        };

        const modelId = this.status.modelId;
        const detections: LiveDetectionBox[] = extracted.peaks.map((pk) => ({
          u: pk.u,
          v: pk.v,
          widthPx: Math.max(2.0, Math.exp(Math.max(-2.0, Math.min(5.0, pk.logW)))),
          heightPx: Math.max(2.0, Math.exp(Math.max(-2.0, Math.min(5.0, pk.logH)))),
          score: pk.score,
          sigmaPx: 1.5,
          modelId,
        }));

        return {
          frameIndex,
          backendKind: this.status.backend.kind,
          aiBadge: this.status.backend.aiBadge,
          inferenceMs,
          allFinite: extracted.allFinite,
          maxHeatmapScore: extracted.maxHeatmapScore,
          peaks: extracted.peaks,
          detections,
          modelId,
          detectorMode: "d1_kf",
        };
      } finally {
        // Dispose ORT tensors promptly even on error to avoid GPU/WASM leaks across 300+ frames
        inputTensor.dispose();
        if (outputs) {
          outputs["heatmap"]?.dispose();
          outputs["offset"]?.dispose();
          outputs["log_scale"]?.dispose();
        }
      }
    });
  }

  /**
   * Run lazy-loaded XVS-D2 (6-ch dual-interval + persistent ConvGRU hidden state) on either
   * a `256x256` single frame (`65536` floats) or the `[1, 5, 256, 256]` staging buffer (channel 0 = `I_t`).
   */
  async runXvsD2(
    frameOrXNorm: Float32Array,
    frameIndex: number,
    scoreThreshold = 0.25,
    timestampS?: number,
  ): Promise<DetectorInferenceResult> {
    return this._serialize(async () => {
      if (!this.ortMod || !this.status) {
        throw new Error("OrtDetectorRunner.runXvsD2 called before init()");
      }
      if (
        frameOrXNorm.length !== D2_FRAME_FLOATS &&
        frameOrXNorm.length !== D1_INPUT_FLOATS
      ) {
        throw new Error(
          `Expected ${D2_FRAME_FLOATS} or ${D1_INPUT_FLOATS} floats for XVS-D2 frame, got ${frameOrXNorm.length}`,
        );
      }

      if (
        typeof timestampS === "number" &&
        Number.isFinite(timestampS) &&
        this.d2LastTimestampS !== null &&
        (timestampS < this.d2LastTimestampS || timestampS - this.d2LastTimestampS > 0.5)
      ) {
        this.resetXvsD2State();
      }
      if (typeof timestampS === "number" && Number.isFinite(timestampS)) {
        this.d2LastTimestampS = timestampS;
      }

      const d2Sess = await this._ensureXvsD2Session();
      const rawPlane =
        frameOrXNorm.length === D2_FRAME_FLOATS
          ? frameOrXNorm
          : frameOrXNorm.subarray(0, D2_FRAME_FLOATS);
      const currNorm = normalizeFrame256(rawPlane);

      this.d2FrameRing.push(currNorm);
      if (this.d2FrameRing.length > 6) {
        this.d2FrameRing.shift();
      }
      const nRing = this.d2FrameRing.length;
      const iT = this.d2FrameRing[nRing - 1];
      const iTm1 = this.d2FrameRing[Math.max(0, nRing - 2)];
      const iTm5 = this.d2FrameRing[Math.max(0, nRing - 6)];

      const xBuf = this.d2InputScratch;
      const plane = D2_FRAME_FLOATS;
      for (let i = 0; i < plane; i++) {
        const c0 = iT[i];
        const c1 = iTm1[i];
        const c2 = iTm5[i];
        xBuf[i] = c0;
        xBuf[plane + i] = c1;
        xBuf[2 * plane + i] = c2;
        xBuf[3 * plane + i] = c0 - c1;
        xBuf[4 * plane + i] = c0 - c2;
        xBuf[5 * plane + i] = 1.0;
      }

      const t0 = performance.now();
      const xTensor = new this.ortMod.Tensor("float32", xBuf, [1, 6, 256, 256]);
      const hTensor = new this.ortMod.Tensor("float32", this.hPrevWarped, [1, 24, 64, 64]);
      let outputs: Record<string, import("onnxruntime-web/webgpu").Tensor> | null = null;

      try {
        outputs = await d2Sess.run({ x_t: xTensor, h_prev_warped: hTensor });
        const hmLogits = outputs["hm_logits"].data as Float32Array;
        const offset = outputs["offset"].data as Float32Array;
        const logScale = outputs["log_scale"].data as Float32Array;
        const logVar = outputs["log_var"].data as Float32Array;
        const peakMask = outputs["peak_mask"].data as Float32Array;
        const hNext = outputs["h_next"].data as Float32Array;

        let hFinite = hNext.length === D2_HIDDEN_FLOATS;
        if (hFinite) {
          for (let i = 0; i < D2_HIDDEN_FLOATS; i++) {
            if (!Number.isFinite(hNext[i])) {
              hFinite = false;
              break;
            }
          }
        }
        if (hFinite) {
          this.hPrevWarped.set(hNext);
        } else {
          this.hPrevWarped.fill(0);
        }

        const modelId = this.d2Model?.id ?? "xvs_d2_web_fp16";
        const decoded = extractXvsD2Detections(
          hmLogits,
          offset,
          logScale,
          logVar,
          peakMask,
          D2_OUT_W,
          D2_OUT_H,
          2,
          scoreThreshold,
          16,
          modelId,
        );
        const inferenceMs = performance.now() - t0;

        const effectiveBackend = this.d2Backend ?? this.status.backend;
        this.inferenceCount += 1;
        this.status = {
          ...this.status,
          backend: effectiveBackend,
          sessionReusedCount: this.inferenceCount,
        };

        return {
          frameIndex,
          backendKind: effectiveBackend.kind,
          aiBadge: effectiveBackend.aiBadge,
          inferenceMs,
          allFinite: decoded.allFinite && hFinite,
          maxHeatmapScore: decoded.maxHeatmapScore,
          peaks: decoded.peaks,
          detections: decoded.detections,
          modelId,
          detectorMode: "xvs_d2",
        };
      } finally {
        xTensor.dispose();
        hTensor.dispose();
        if (outputs) {
          outputs["hm_logits"]?.dispose();
          outputs["offset"]?.dispose();
          outputs["log_scale"]?.dispose();
          outputs["log_var"]?.dispose();
          outputs["peak_mask"]?.dispose();
          outputs["h_next"]?.dispose();
        }
      }
    });
  }

  /**
   * Run resident XVS-D2 directly on RGBA ImageData (e.g. Tab 3 Real-Sky AOT 1:1 native crops
   * or user-dropped videos) with explicit `(shiftDx, shiftDy)` translation GMC or automatic
   * fast 64x64 SAD translation compensation when shifts are omitted.
   */
  async runXvsD2ImageData(
    rgba: Uint8ClampedArray,
    w: number,
    h: number,
    shiftDx?: number,
    shiftDy?: number,
    frameIndex = 0,
    scoreThreshold = 0.25,
  ): Promise<DetectorInferenceResult> {
    return this._serialize(async () => {
      if (!this.ortMod || !this.status) {
        throw new Error("OrtDetectorRunner.runXvsD2ImageData called before init()");
      }
      if (w <= 0 || h <= 0 || rgba.length < w * h * 4) {
        throw new Error(`Invalid RGBA frame dimensions (${w}x${h}, ${rgba.length} bytes)`);
      }

      const d2Sess = await this._ensureXvsD2Session();
      const gray256 = rgbaToGray256(rgba, w, h);
      const currNorm = normalizeFrame256(gray256);
      const curr64 = downsample256To64(currNorm);

      let dxPx = 0;
      let dyPx = 0;
      if (
        typeof shiftDx === "number" &&
        Number.isFinite(shiftDx) &&
        typeof shiftDy === "number" &&
        Number.isFinite(shiftDy)
      ) {
        dxPx = Math.round(shiftDx);
        dyPx = Math.round(shiftDy);
      } else if (this.d2Prev64 !== null) {
        const est = estimateSadTranslation64(this.d2Prev64, curr64);
        dxPx = est.dxPx;
        dyPx = est.dyPx;
      }
      this.d2Prev64 = curr64;

      // Warp previous ring frames & ConvGRU hidden state into current frame coordinates
      if (dxPx !== 0 || dyPx !== 0) {
        for (let r = 0; r < this.d2FrameRing.length; r++) {
          this.d2FrameRing[r] = shiftPlane256(this.d2FrameRing[r], dxPx, dyPx);
        }
        const hShifted = shiftHiddenState64(this.hPrevWarped, dxPx, dyPx);
        this.hPrevWarped.set(hShifted);
      }

      this.d2FrameRing.push(currNorm);
      if (this.d2FrameRing.length > 6) {
        this.d2FrameRing.shift();
      }
      const nRing = this.d2FrameRing.length;
      const iT = this.d2FrameRing[nRing - 1];
      const iTm1 = this.d2FrameRing[Math.max(0, nRing - 2)];
      const iTm5 = this.d2FrameRing[Math.max(0, nRing - 6)];

      const xBuf = this.d2InputScratch;
      const plane = D2_FRAME_FLOATS;
      for (let i = 0; i < plane; i++) {
        const c0 = iT[i];
        const c1 = iTm1[i];
        const c2 = iTm5[i];
        xBuf[i] = c0;
        xBuf[plane + i] = c1;
        xBuf[2 * plane + i] = c2;
        xBuf[3 * plane + i] = c0 - c1;
        xBuf[4 * plane + i] = c0 - c2;
        xBuf[5 * plane + i] = 1.0;
      }

      const t0 = performance.now();
      const xTensor = new this.ortMod.Tensor("float32", xBuf, [1, 6, 256, 256]);
      const hTensor = new this.ortMod.Tensor("float32", this.hPrevWarped, [1, 24, 64, 64]);
      let outputs: Record<string, import("onnxruntime-web/webgpu").Tensor> | null = null;

      try {
        outputs = await d2Sess.run({ x_t: xTensor, h_prev_warped: hTensor });
        const hmLogits = outputs["hm_logits"].data as Float32Array;
        const offset = outputs["offset"].data as Float32Array;
        const logScale = outputs["log_scale"].data as Float32Array;
        const logVar = outputs["log_var"].data as Float32Array;
        const peakMask = outputs["peak_mask"].data as Float32Array;
        const hNext = outputs["h_next"].data as Float32Array;

        let hFinite = hNext.length === D2_HIDDEN_FLOATS;
        if (hFinite) {
          for (let i = 0; i < D2_HIDDEN_FLOATS; i++) {
            if (!Number.isFinite(hNext[i])) {
              hFinite = false;
              break;
            }
          }
        }
        if (hFinite) {
          this.hPrevWarped.set(hNext);
        } else {
          this.hPrevWarped.fill(0);
        }

        const modelId = this.d2Model?.id ?? "xvs_d2_web_fp16";
        const decoded = extractXvsD2Detections(
          hmLogits,
          offset,
          logScale,
          logVar,
          peakMask,
          D2_OUT_W,
          D2_OUT_H,
          2,
          scoreThreshold,
          16,
          modelId,
        );
        const inferenceMs = performance.now() - t0;

        const effectiveBackend = this.d2Backend ?? this.status.backend;
        this.inferenceCount += 1;
        this.status = {
          ...this.status,
          backend: effectiveBackend,
          sessionReusedCount: this.inferenceCount,
        };

        return {
          frameIndex,
          backendKind: effectiveBackend.kind,
          aiBadge: effectiveBackend.aiBadge,
          inferenceMs,
          allFinite: decoded.allFinite && hFinite,
          maxHeatmapScore: decoded.maxHeatmapScore,
          peaks: decoded.peaks,
          detections: decoded.detections,
          modelId,
          detectorMode: "xvs_d2",
        };
      } finally {
        xTensor.dispose();
        hTensor.dispose();
        if (outputs) {
          outputs["hm_logits"]?.dispose();
          outputs["offset"]?.dispose();
          outputs["log_scale"]?.dispose();
          outputs["log_var"]?.dispose();
          outputs["peak_mask"]?.dispose();
          outputs["h_next"]?.dispose();
        }
      }
    });
  }

  getStatus(): AiInitStatus | null {
    return this.status;
  }

  async dispose(): Promise<void> {
    return this._serialize(async () => {
      if (this.session) {
        await this.session.release();
        this.session = null;
      }
      if (this.d2Session) {
        await this.d2Session.release();
        this.d2Session = null;
      }
      this.d2Backend = null;
      this.resetXvsD2State();
    });
  }
}

// ---------------------------------------------------------------------------
// Module Web Worker Message Protocol (activated only inside a Worker context)
// ---------------------------------------------------------------------------

export type OrtWorkerRequest =
  | {
      type: "init";
      baseUrl?: string;
      hasWebGpu: boolean;
      hasShaderF16: boolean;
      forceWasmSimd?: boolean;
      fallbackReason?: string | null;
      wasmPaths?: string;
    }
  | {
      type: "infer";
      frameIndex: number;
      slotIndex: number;
      xNormBuffer: ArrayBuffer;
      scoreThreshold?: number;
      detectorMode?: DetectorMode;
      timestampS?: number;
    }
  | {
      type: "reset-d2-state";
    }
  | {
      type: "infer-d2-rgba";
      frameIndex: number;
      rgbaBuffer: ArrayBuffer;
      width: number;
      height: number;
      shiftDx?: number;
      shiftDy?: number;
      scoreThreshold?: number;
    }
  | {
      type: "device-lost";
      reason: string;
    };

export type OrtWorkerResponse =
  | { type: "init-ok"; status: AiInitStatus }
  | {
      type: "infer-ok";
      slotIndex: number;
      xNormBuffer: ArrayBuffer;
      result: DetectorInferenceResult;
    }
  | { type: "status-update"; status: AiInitStatus }
  | { type: "error"; message: string; slotIndex?: number; xNormBuffer?: ArrayBuffer };

interface WorkerGlobalLike {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((ev: MessageEvent<OrtWorkerRequest>) => void) | null;
}

const isWorkerRuntime =
  typeof globalThis !== "undefined" &&
  typeof (globalThis as { window?: unknown }).window === "undefined" &&
  typeof (globalThis as { postMessage?: unknown }).postMessage === "function";

if (isWorkerRuntime) {
  const workerSelf = globalThis as unknown as WorkerGlobalLike;
  const runner = new OrtDetectorRunner();

  workerSelf.onmessage = async (ev: MessageEvent<OrtWorkerRequest>) => {
    const msg = ev.data;
    try {
      if (msg.type === "init") {
        const status = await runner.init({
          baseUrl: msg.baseUrl,
          hasWebGpu: msg.hasWebGpu,
          hasShaderF16: msg.hasShaderF16,
          forceWasmSimd: msg.forceWasmSimd,
          fallbackReason: msg.fallbackReason,
          wasmPaths: msg.wasmPaths,
        });
        const resp: OrtWorkerResponse = { type: "init-ok", status };
        workerSelf.postMessage(resp);
      } else if (msg.type === "reset-d2-state") {
        runner.resetXvsD2TemporalState();
      } else if (msg.type === "infer-d2-rgba") {
        const rgba = new Uint8ClampedArray(msg.rgbaBuffer);
        const result = await runner.runXvsD2ImageData(
          rgba,
          msg.width,
          msg.height,
          msg.shiftDx,
          msg.shiftDy,
          msg.frameIndex,
          msg.scoreThreshold,
        );
        const resp: OrtWorkerResponse = {
          type: "infer-ok",
          slotIndex: -1,
          xNormBuffer: rgba.buffer,
          result,
        };
        workerSelf.postMessage(resp, [rgba.buffer]);
      } else if (msg.type === "device-lost") {
        const status = await runner.handleDeviceLost(msg.reason);
        const resp: OrtWorkerResponse = { type: "status-update", status };
        workerSelf.postMessage(resp);
      } else if (msg.type === "infer") {
        const xNorm = new Float32Array(msg.xNormBuffer);
        const result =
          msg.detectorMode === "xvs_d2"
            ? await runner.runXvsD2(xNorm, msg.frameIndex, msg.scoreThreshold, msg.timestampS)
            : await runner.runNormalized(xNorm, msg.frameIndex, msg.scoreThreshold);
        const resp: OrtWorkerResponse = {
          type: "infer-ok",
          slotIndex: msg.slotIndex,
          xNormBuffer: xNorm.buffer,
          result,
        };
        workerSelf.postMessage(resp, [xNorm.buffer]);
      }
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      const resp: OrtWorkerResponse =
        msg.type === "infer"
          ? {
              type: "error",
              message: errMsg,
              slotIndex: msg.slotIndex,
              xNormBuffer: msg.xNormBuffer,
            }
          : { type: "error", message: errMsg };
      if (resp.type === "error" && resp.xNormBuffer) {
        workerSelf.postMessage(resp, [resp.xNormBuffer]);
      } else {
        workerSelf.postMessage(resp);
      }
    }
  };
}
