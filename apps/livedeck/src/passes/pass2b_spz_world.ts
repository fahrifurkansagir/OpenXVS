/**
 * Pass 2b: Pure-TS Niantic SPZ v2 (`0x5053474E` "NGSP") 3D Gaussian Splatting Parser
 * & Dual-Backend WebGPU/WebGL2 Splat Overlay (`VISUAL_STRESS_ONLY`).
 *
 * Matches `src/xvs/v2/xvs_world/train_g4.py` (`write_spz_v2` / `read_spz_v2_header`) byte-for-byte:
 *   - 16-byte little-endian header: `<IIIBBBB` (magic, version=2, numPoints, shDegree, fractionalBits, flags, reserved)
 *   - Planar payload: 24-bit signed LE positions in local KSFO 28L NED, uint8 alphas, uint8 SH0 colors,
 *     uint8 log-scales, uint8 quaternion xyz, plus optional SH1..SH3 coefficients.
 *   - Zero npm dependencies (uses native `DecompressionStream("gzip")` in browser/Node).
 *   - Reuses Pass 3's compiled WGSL + Naga GLSL ES 3.00 pipeline for 100% WebGPU + WebGL2 parity.
 */

import { DEG_TO_RAD, geodeticToEcef, mat3MulVec, rEcefNed } from "../rhi/coords.ts";
import type { RhiBufferHandle, RhiContext, RhiPipelineHandle } from "../rhi/types.ts";
import pass3Wgsl from "../shaders/pass3_airframe_traffic.wgsl?raw";
import pass3GlVert from "../shaders/generated_gl/pass3_airframe_traffic.vert.glsl?raw";
import pass3GlFrag from "../shaders/generated_gl/pass3_airframe_traffic.frag.glsl?raw";
import type { ActiveCameraView, CockpitFrameState, TerrainPackData } from "./types.ts";

export const SPZ_MAGIC_U32 = 0x5053474e; // Little-endian "NGSP"
export const SPZ_VERSION = 2;
export const SPZ_HEADER_BYTES = 16;
export const SPZ_SH_C0 = 0.28209479177387814;
export const SPZ_MAX_CHROMA_SPREAD = 0.35;
export const MAX_RENDER_SPLATS = 16384;

const FLOATS_PER_VERTEX = 9; // posRte(3) + color(4) + dashMeta(2)
const VERTS_PER_SPLAT = 6;   // 2 triangles forming a soft camera-facing diamond billboard
const SH_EXTRA_COEFFS: Readonly<Record<number, number>> = {
  0: 0,
  1: 3,
  2: 8,
  3: 15,
};

export type SpzMode = "off" | "preview" | "custom";

export interface SpzHeader {
  readonly magicU32: number;
  readonly version: number;
  readonly numPoints: number;
  readonly shDegree: number;
  readonly fractionalBits: number;
  readonly flags: number;
  readonly reserved: number;
}

export interface SpzSplatCloud {
  readonly gtTier: "VISUAL_STRESS_ONLY";
  readonly sourceLabel: string;
  readonly totalFileSplats: number;
  readonly renderSplats: number;
  readonly stride: number;
  readonly header: SpzHeader;
  /** Packed [N, E, D] metres relative to KSFO 28L threshold (length = renderSplats * 3). */
  readonly positionsNed: Float32Array;
  /** Packed [R, G, B] in [0, 1] (length = renderSplats * 3). */
  readonly colorsRgb: Float32Array;
  /** Opacity in [0, 1] (length = renderSplats). */
  readonly alphas: Float32Array;
  /** Geometric-mean splat visual radius in metres (length = renderSplats). */
  readonly radiiM: Float32Array;
}

export interface SpzDiagnostics {
  readonly mode: SpzMode;
  readonly gtTier: "VISUAL_STRESS_ONLY";
  readonly sourceLabel: string;
  readonly totalFileSplats: number;
  readonly renderSplats: number;
  readonly lastError: string | null;
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/**
 * Clamp RGB chroma spread (`max(r,g,b) - min(r,g,b) <= maxSpread`) while strictly preserving
 * Rec.709 luminance (`0.2126 R + 0.7152 G + 0.0722 B`). Suppresses rainbow SH0 outlier splats.
 */
export function clampSpzChromaRgb(
  r: number,
  g: number,
  b: number,
  maxSpread = SPZ_MAX_CHROMA_SPREAD,
): readonly [number, number, number] {
  const cr = clamp(r, 0.0, 1.0);
  const cg = clamp(g, 0.0, 1.0);
  const cb = clamp(b, 0.0, 1.0);
  const cMax = Math.max(cr, cg, cb);
  const cMin = Math.min(cr, cg, cb);
  const spread = cMax - cMin;
  if (spread <= maxSpread || spread <= 1e-8) {
    return [cr, cg, cb];
  }
  const luma = 0.2126 * cr + 0.7152 * cg + 0.0722 * cb;
  const scale = Math.max(0.0, maxSpread) / spread;
  return [
    clamp(luma + (cr - luma) * scale, 0.0, 1.0),
    clamp(luma + (cg - luma) * scale, 0.0, 1.0),
    clamp(luma + (cb - luma) * scale, 0.0, 1.0),
  ];
}

/** Decode signed 24-bit little-endian integer into signed JS int32. */
function readSigned24LE(bytes: Uint8Array, offset: number): number {
  const raw = bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16);
  return (raw << 8) >> 8;
}

/**
 * Parse uncompressed Niantic SPZ v2 (`0x5053474E` "NGSP", `version=2`) bytes fail-closed,
 * stride-subsampling cleanly to `<= maxRenderSplats` for 60+ FPS real-time rendering.
 */
export function parseSpzV2Raw(
  raw: Uint8Array,
  sourceLabel = "custom.spz",
  maxRenderSplats = MAX_RENDER_SPLATS,
  maxChromaSpread?: number,
): SpzSplatCloud {
  if (raw.byteLength < SPZ_HEADER_BYTES) {
    throw new Error(`SPZ v2 payload truncated (${raw.byteLength} B < ${SPZ_HEADER_BYTES} B header)`);
  }

  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const magicU32 = dv.getUint32(0, true);
  const version = dv.getUint32(4, true);
  const numPoints = dv.getUint32(8, true);
  const shDegree = dv.getUint8(12);
  const fractionalBits = dv.getUint8(13);
  const flags = dv.getUint8(14);
  const reserved = dv.getUint8(15);

  if (magicU32 !== SPZ_MAGIC_U32) {
    throw new Error(
      `Invalid SPZ magic 0x${magicU32.toString(16).toUpperCase().padStart(8, "0")}; expected 0x5053474E ("NGSP")`,
    );
  }
  if (version !== SPZ_VERSION) {
    throw new Error(`Unsupported SPZ version ${version}; expected ${SPZ_VERSION}`);
  }
  if (numPoints <= 0) {
    throw new Error(`Invalid SPZ numPoints=${numPoints}; must be > 0`);
  }
  const shExtra = SH_EXTRA_COEFFS[shDegree];
  if (shExtra === undefined) {
    throw new Error(`Unsupported SPZ shDegree=${shDegree}; expected 0..3`);
  }
  if (fractionalBits < 1 || fractionalBits > 20) {
    throw new Error(`Invalid SPZ fractionalBits=${fractionalBits}; expected 1..20`);
  }

  const bytesPerPoint = 9 + 1 + 3 + 3 + 3 + 3 * shExtra;
  const expectedBytes = SPZ_HEADER_BYTES + numPoints * bytesPerPoint;
  if (raw.byteLength !== expectedBytes) {
    throw new Error(
      `SPZ v2 uncompressed length mismatch: got ${raw.byteLength} B, expected ${expectedBytes} B`,
    );
  }

  const posBase = SPZ_HEADER_BYTES;
  const alphaBase = posBase + numPoints * 9;
  const colorBase = alphaBase + numPoints;
  const scaleBase = colorBase + numPoints * 3;

  const cap = Math.max(1, Math.min(MAX_RENDER_SPLATS, Math.floor(maxRenderSplats)));
  const stride = Math.max(1, Math.ceil(numPoints / cap));
  const renderSplats = Math.ceil(numPoints / stride);

  const positionsNed = new Float32Array(renderSplats * 3);
  const colorsRgb = new Float32Array(renderSplats * 3);
  const alphas = new Float32Array(renderSplats);
  const radiiM = new Float32Array(renderSplats);

  const invPosScale = 1.0 / (1 << fractionalBits);
  const invColorScale = 1.0 / (0.15 * 255.0);

  let outIdx = 0;
  for (let i = 0; i < numPoints; i += stride) {
    const pOff = posBase + i * 9;
    const pn = readSigned24LE(raw, pOff + 0) * invPosScale;
    const pe = readSigned24LE(raw, pOff + 3) * invPosScale;
    const pd = readSigned24LE(raw, pOff + 6) * invPosScale;
    positionsNed[outIdx * 3 + 0] = pn;
    positionsNed[outIdx * 3 + 1] = pe;
    positionsNed[outIdx * 3 + 2] = pd;

    alphas[outIdx] = raw[alphaBase + i] / 255.0;

    const cOff = colorBase + i * 3;
    const rawR = clamp((raw[cOff + 0] - 127.5) * invColorScale * SPZ_SH_C0 + 0.5, 0.0, 1.0);
    const rawG = clamp((raw[cOff + 1] - 127.5) * invColorScale * SPZ_SH_C0 + 0.5, 0.0, 1.0);
    const rawB = clamp((raw[cOff + 2] - 127.5) * invColorScale * SPZ_SH_C0 + 0.5, 0.0, 1.0);
    if (maxChromaSpread !== undefined) {
      const [cr, cg, cb] = clampSpzChromaRgb(rawR, rawG, rawB, maxChromaSpread);
      colorsRgb[outIdx * 3 + 0] = cr;
      colorsRgb[outIdx * 3 + 1] = cg;
      colorsRgb[outIdx * 3 + 2] = cb;
    } else {
      colorsRgb[outIdx * 3 + 0] = rawR;
      colorsRgb[outIdx * 3 + 1] = rawG;
      colorsRgb[outIdx * 3 + 2] = rawB;
    }

    const sOff = scaleBase + i * 3;
    const ls0 = clamp(raw[sOff + 0] / 16.0 - 10.0, -6.0, 5.0);
    const ls1 = clamp(raw[sOff + 1] / 16.0 - 10.0, -6.0, 5.0);
    const ls2 = clamp(raw[sOff + 2] / 16.0 - 10.0, -6.0, 5.0);
    const rGeom = Math.exp((ls0 + ls1 + ls2) / 3.0);
    radiiM[outIdx] = clamp(rGeom, 0.45, 28.0);

    outIdx++;
  }

  return {
    gtTier: "VISUAL_STRESS_ONLY",
    sourceLabel,
    totalFileSplats: numPoints,
    renderSplats: outIdx,
    stride,
    header: {
      magicU32,
      version,
      numPoints,
      shDegree,
      fractionalBits,
      flags,
      reserved,
    },
    positionsNed,
    colorsRgb,
    alphas,
    radiiM,
  };
}

/**
 * Decompress a gzip-compressed `.spz` file via zero-dependency native `DecompressionStream("gzip")`
 * (also accepts already-uncompressed `NGSP` bytes cleanly) and decode via `parseSpzV2Raw`.
 */
export async function parseSpzV2Gzip(
  compressed: ArrayBuffer | Uint8Array,
  sourceLabel = "custom.spz",
  maxRenderSplats = MAX_RENDER_SPLATS,
  maxChromaSpread?: number,
): Promise<SpzSplatCloud> {
  const bytes = compressed instanceof Uint8Array ? compressed : new Uint8Array(compressed);
  if (bytes.byteLength < 4) {
    throw new Error(`SPZ buffer too small (${bytes.byteLength} B)`);
  }

  // Check if uncompressed NGSP payload was passed directly
  if (
    bytes[0] === 0x4e &&
    bytes[1] === 0x47 &&
    bytes[2] === 0x53 &&
    bytes[3] === 0x50
  ) {
    return parseSpzV2Raw(bytes, sourceLabel, maxRenderSplats, maxChromaSpread);
  }

  // Verify gzip header magic (0x1F 0x8B)
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) {
    throw new Error("Invalid SPZ file: expected gzip header (0x1F 0x8B) or NGSP magic");
  }
  if (typeof DecompressionStream === "undefined") {
    throw new Error("Native DecompressionStream('gzip') unavailable in this runtime");
  }

  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  const stream = new Blob([copy.buffer])
    .stream()
    .pipeThrough(new DecompressionStream("gzip"));
  const decompressedBuf = await new Response(stream).arrayBuffer();
  return parseSpzV2Raw(new Uint8Array(decompressedBuf), sourceLabel, maxRenderSplats, maxChromaSpread);
}

/**
 * Deterministic ~6,000-splat KSFO 28L 1x1 km 3DGS visual-stress preview cloud anchored in strict
 * local KSFO 28L NED coordinates (`gtTier = "VISUAL_STRESS_ONLY"`).
 */
export function buildKsfo28lPreviewSpzCloud(): SpzSplatCloud {
  const hdgRad = 297.7 * DEG_TO_RAD;
  const alongN = Math.cos(hdgRad);
  const alongE = Math.sin(hdgRad);
  const rightN = -Math.sin(hdgRad);
  const rightE = Math.cos(hdgRad);

  const posList: number[] = [];
  const colList: number[] = [];
  const alphaList: number[] = [];
  const radList: number[] = [];

  // Deterministic LCG PRNG for crisp organic 3DGS splat clustering
  let seed = 20261121;
  const rand = (): number => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };

  const pushSplat = (
    alongM: number,
    acrossM: number,
    upM: number,
    r: number,
    g: number,
    b: number,
    alpha: number,
    radiusM: number,
  ) => {
    const n = alongN * alongM + rightN * acrossM;
    const e = alongE * alongM + rightE * acrossM;
    const d = -upM; // Strict NED: Up is negative Down
    posList.push(n, e, d);
    colList.push(clamp(r, 0, 1), clamp(g, 0, 1), clamp(b, 0, 1));
    alphaList.push(clamp(alpha, 0.05, 0.98));
    radList.push(clamp(radiusM, 0.8, 22.0));
  };

  // 1. KSFO 28L & 28R high-density runway threshold, aiming-point & tarmac splats (~2,100 splats)
  for (const rwyOffsetAcross of [0.0, 228.0]) {
    const is28L = rwyOffsetAcross === 0.0;
    for (let s = 0; s <= 760; s += 16) {
      for (let c = -28; c <= 28; c += 8) {
        const jitterS = (rand() - 0.5) * 6.0;
        const jitterC = (rand() - 0.5) * 3.0;
        const isPiano = s >= 10 && s <= 56 && Math.abs(c) >= 4 && Math.abs(c) <= 24;
        const isAim = s >= 285 && s <= 345 && Math.abs(c) >= 7 && Math.abs(c) <= 17;
        const isCenter = Math.abs(c) < 3.5 && s % 64 < 36;
        const bright = isPiano || isAim || isCenter;
        const tint = bright ? 0.94 : is28L ? 0.22 + rand() * 0.06 : 0.19 + rand() * 0.05;
        pushSplat(
          s + jitterS,
          rwyOffsetAcross + c + jitterC,
          2.2 + rand() * 0.5,
          tint,
          tint + (bright ? 0.03 : 0.01),
          tint + (bright ? 0.05 : 0.03),
          bright ? 0.78 : 0.42,
          bright ? 4.4 : 7.2,
        );
      }
    }
  }

  // 2. ALSF-2 Approach Light Stanchions & SF Bay Causeway Pilings extending -500 m aft of 28L (~700 splats)
  for (let d = 25; d <= 520; d += 25) {
    const hasCrossbar = d === 300 || d === 150;
    const span = hasCrossbar ? 26 : 6;
    for (let c = -span; c <= span; c += 4) {
      // Causeway piling pier column + warm incandescent approach lamp halo
      pushSplat(-d, c, 1.5, 0.36, 0.39, 0.43, 0.65, 3.2);
      pushSplat(-d, c, 4.2, 1.0, 0.92, 0.68, 0.88, 2.9);
    }
  }

  // 3. KSFO International Terminal Concourse A/G volumetric 3DGS structure south-west of 28L (~1,600 splats)
  for (let i = 0; i < 1600; i++) {
    const along = 320 + rand() * 440;
    const across = -220 - rand() * 240;
    const isRoof = rand() > 0.35;
    const up = isRoof ? 18.0 + rand() * 8.0 : 3.0 + rand() * 16.0;
    const glass = !isRoof && rand() > 0.45;
    pushSplat(
      along,
      across,
      up,
      glass ? 0.32 + rand() * 0.2 : 0.72 + rand() * 0.16,
      glass ? 0.64 + rand() * 0.2 : 0.76 + rand() * 0.14,
      glass ? 0.86 + rand() * 0.12 : 0.80 + rand() * 0.12,
      glass ? 0.76 : 0.64,
      glass ? 4.8 : 7.5,
    );
  }

  // 4. KSFO Air Traffic Control Tower volumetric spire & illuminated cab (~420 splats)
  // Located near KSFO_28L_TOWER (roughly along=+240 m, across=-175 m relative to 28L threshold)
  for (let i = 0; i < 420; i++) {
    const hFrac = rand();
    const up = 2.0 + hFrac * 48.0;
    const rad = hFrac > 0.82 ? 9.0 + rand() * 4.0 : 4.5 + (1.0 - hFrac) * 4.0;
    const ang = rand() * Math.PI * 2.0;
    const isCab = hFrac > 0.82;
    pushSplat(
      245 + Math.cos(ang) * rad,
      -175 + Math.sin(ang) * rad,
      up,
      isCab ? 0.42 : 0.86,
      isCab ? 0.92 : 0.89,
      isCab ? 1.0 : 0.92,
      isCab ? 0.88 : 0.74,
      isCab ? 3.4 : 4.2,
    );
  }

  // 5. SF Bay shoreline riprap seawall & maritime channel marker splats (~1,180 splats)
  for (let i = 0; i < 1180; i++) {
    const along = -480 + rand() * 1100;
    const across = 320 + rand() * 240;
    const nearEdge = across < 365;
    pushSplat(
      along,
      across,
      nearEdge ? 1.8 + rand() * 1.6 : 0.6,
      nearEdge ? 0.48 + rand() * 0.12 : 0.08 + rand() * 0.06,
      nearEdge ? 0.46 + rand() * 0.12 : 0.24 + rand() * 0.08,
      nearEdge ? 0.40 + rand() * 0.10 : 0.34 + rand() * 0.10,
      nearEdge ? 0.62 : 0.36,
      nearEdge ? 5.6 : 11.5,
    );
  }

  const numPoints = alphaList.length;
  return {
    gtTier: "VISUAL_STRESS_ONLY",
    sourceLabel: "KSFO 28L 1x1km Preview (VISUAL_STRESS_ONLY)",
    totalFileSplats: numPoints,
    renderSplats: numPoints,
    stride: 1,
    header: {
      magicU32: SPZ_MAGIC_U32,
      version: SPZ_VERSION,
      numPoints,
      shDegree: 0,
      fractionalBits: 12,
      flags: 0,
      reserved: 0,
    },
    positionsNed: new Float32Array(posList),
    colorsRgb: new Float32Array(colList),
    alphas: new Float32Array(alphaList),
    radiiM: new Float32Array(radList),
  };
}

/**
 * Dual-backend WebGPU/WebGL2 3DGS Splat Overlay pass reusing Pass 3's compiled WGSL/GLSL pipeline.
 * Allocates splat GPU vertex memory lazily ONLY when mode !== "off" so Stage-0 boot cost is 0 B.
 */
export class Pass2bSpzWorld {
  private readonly rhi: RhiContext;
  private readonly pipeline: RhiPipelineHandle;
  private readonly ubo: RhiBufferHandle;
  private readonly uniformData = new Float32Array(24); // mat4 (16) + 2 x vec4 (8) = 96 B

  private vb: RhiBufferHandle | null = null;
  private scratch: Float32Array | null = null;
  private mode: SpzMode = "off";
  private cloud: SpzSplatCloud | null = null;
  private lastError: string | null = null;

  constructor(rhi: RhiContext) {
    this.rhi = rhi;
    this.ubo = rhi.createBuffer("uniform", this.uniformData.byteLength, "pass2b-spz-ubo");
    this.pipeline = rhi.createRenderPipeline({
      label: "pass2b-spz-world",
      wgsl: pass3Wgsl,
      glslVert: pass3GlVert,
      glslFrag: pass3GlFrag,
      topology: "triangle-list",
      depthTest: true,
      depthWrite: false,
      blendAlpha: true,
      vertexLayouts: [
        {
          arrayStride: FLOATS_PER_VERTEX * 4,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x4" },
            { shaderLocation: 2, offset: 28, format: "float32x2" },
          ],
        },
      ],
      uniformBindings: [0],
    });
  }

  private ensureScratch(splatCount: number): void {
    const neededFloats = Math.min(MAX_RENDER_SPLATS, splatCount) * VERTS_PER_SPLAT * FLOATS_PER_VERTEX;
    if (!this.scratch || this.scratch.length < neededFloats) {
      this.scratch = new Float32Array(neededFloats);
    }
    if (!this.vb || this.vb.byteLength < neededFloats * 4) {
      if (this.vb) {
        this.rhi.destroyBuffer(this.vb);
      }
      this.vb = this.rhi.createBuffer("vertex", neededFloats * 4, "pass2b-spz-vb");
    }
  }

  setMode(mode: "off" | "preview"): void {
    this.lastError = null;
    if (mode === "off") {
      this.mode = "off";
      return;
    }
    this.cloud = buildKsfo28lPreviewSpzCloud();
    this.mode = "preview";
    this.ensureScratch(this.cloud.renderSplats);
  }

  setCustomCloud(cloud: SpzSplatCloud): void {
    this.lastError = null;
    this.cloud = cloud;
    this.mode = "custom";
    this.ensureScratch(cloud.renderSplats);
  }

  async loadSpzBuffer(buffer: ArrayBuffer | Uint8Array, sourceLabel = "custom.spz"): Promise<SpzDiagnostics> {
    try {
      const parsed = await parseSpzV2Gzip(
        buffer,
        sourceLabel,
        MAX_RENDER_SPLATS,
        SPZ_MAX_CHROMA_SPREAD,
      );
      this.setCustomCloud(parsed);
      return this.getDiagnostics();
    } catch (err) {
      this.lastError = err instanceof Error ? err.message : String(err);
      throw err;
    }
  }

  getMode(): SpzMode {
    return this.mode;
  }

  getDiagnostics(): SpzDiagnostics {
    const isOff = this.mode === "off";
    return {
      mode: this.mode,
      gtTier: "VISUAL_STRESS_ONLY",
      sourceLabel: isOff ? "OFF" : (this.cloud?.sourceLabel ?? "OFF"),
      totalFileSplats: isOff ? 0 : (this.cloud?.totalFileSplats ?? 0),
      renderSplats: isOff ? 0 : (this.cloud?.renderSplats ?? 0),
      lastError: this.lastError,
    };
  }

  encode(state: CockpitFrameState, view: ActiveCameraView, pack: TerrainPackData): void {
    // Strictly gated behind explicit `manifoldToggles?.spzStress === true` (OFF by default -> zero confetti)
    if (state.manifoldToggles?.spzStress !== true) return;
    if (this.mode === "off" || !this.cloud || !this.vb || !this.scratch) return;

    const cloud = this.cloud;
    const n = Math.min(MAX_RENDER_SPLATS, cloud.renderSplats);
    if (n <= 0) return;

    const thrLatRad = pack.ksfo28lThrLatDeg * DEG_TO_RAD;
    const thrLonRad = pack.ksfo28lThrLonDeg * DEG_TO_RAD;
    const pThr = geodeticToEcef(thrLatRad, thrLonRad, pack.ksfo28lThrWgs84HM);
    const rNedThr = rEcefNed(thrLatRad, thrLonRad);
    const pCam = view.pEcefCam;

    const thrRteX = pThr[0] - pCam[0];
    const thrRteY = pThr[1] - pCam[1];
    const thrRteZ = pThr[2] - pCam[2];

    // Camera right (+X_cam) and up (-Y_cam) columns of R_ecef_cam for camera-facing splat billboards
    const camRx = view.rEcefCam[0];
    const camRy = view.rEcefCam[3];
    const camRz = view.rEcefCam[6];
    const camUx = -view.rEcefCam[1];
    const camUy = -view.rEcefCam[4];
    const camUz = -view.rEcefCam[7];

    const v = this.scratch;
    const pos = cloud.positionsNed;
    const col = cloud.colorsRgb;
    const alp = cloud.alphas;
    const rad = cloud.radiiM;

    let outV = 0;
    for (let i = 0; i < n; i++) {
      const pn = pos[i * 3 + 0];
      const pe = pos[i * 3 + 1];
      const pd = pos[i * 3 + 2];
      const dEcef = mat3MulVec(rNedThr, [pn, pe, pd]);
      const cx = thrRteX + dEcef[0];
      const cy = thrRteY + dEcef[1];
      const cz = thrRteZ + dEcef[2];

      const rM = rad[i];
      const [cr, cg, cb] = clampSpzChromaRgb(
        col[i * 3 + 0],
        col[i * 3 + 1],
        col[i * 3 + 2],
        SPZ_MAX_CHROMA_SPREAD,
      );
      const ca = alp[i] * 0.72;

      const rx = camRx * rM;
      const ry = camRy * rM;
      const rz = camRz * rM;
      const ux = camUx * rM;
      const uy = camUy * rM;
      const uz = camUz * rM;

      // Full camera-facing quad with billboard UV in [-1, 1]^2 for soft radial Gaussian falloff
      const blX = cx - rx - ux, blY = cy - ry - uy, blZ = cz - rz - uz;
      const brX = cx + rx - ux, brY = cy + ry - uy, brZ = cz + rz - uz;
      const trX = cx + rx + ux, trY = cy + ry + uy, trZ = cz + rz + uz;
      const tlX = cx - rx + ux, tlY = cy - ry + uy, tlZ = cz - rz + uz;

      const writeVert = (px: number, py: number, pz: number, uCoord: number, vCoord: number) => {
        const o = outV * FLOATS_PER_VERTEX;
        v[o + 0] = px; v[o + 1] = py; v[o + 2] = pz;
        v[o + 3] = cr; v[o + 4] = cg; v[o + 5] = cb; v[o + 6] = ca;
        v[o + 7] = uCoord; v[o + 8] = vCoord;
        outV++;
      };

      writeVert(blX, blY, blZ, -1.0, -1.0);
      writeVert(brX, brY, brZ, 1.0, -1.0);
      writeVert(trX, trY, trZ, 1.0, 1.0);
      writeVert(blX, blY, blZ, -1.0, -1.0);
      writeVert(trX, trY, trZ, 1.0, 1.0);
      writeVert(tlX, tlY, tlZ, -1.0, 1.0);
    }

    this.uniformData.set(view.viewProj, 0);
    this.uniformData[16] = state.simTimeS;
    this.uniformData[17] = state.efvsEnabled ? 1.0 : 0.0;
    this.uniformData[18] = state.fogContrast;
    this.uniformData[19] = 1.0;
    // modeParams: billboardMode = 1 (soft radial Gaussian splat), sigma = 3.5, discard < 0.04
    this.uniformData[20] = 1.0;
    this.uniformData[21] = 3.5;
    this.uniformData[22] = 0.04;
    this.uniformData[23] = 0.0;

    this.rhi.writeBuffer(this.ubo, this.uniformData);
    this.rhi.writeBuffer(this.vb, v.subarray(0, outV * FLOATS_PER_VERTEX));
    this.rhi.draw({
      pipeline: this.pipeline,
      uniformBuffers: [{ binding: 0, buffer: this.ubo }],
      vertexBuffers: [this.vb],
      vertexCount: outV,
    });
  }

  dispose(): void {
    if (this.vb) {
      this.rhi.destroyBuffer(this.vb);
      this.vb = null;
    }
    this.scratch = null;
    this.rhi.destroyBuffer(this.ubo);
  }
}
