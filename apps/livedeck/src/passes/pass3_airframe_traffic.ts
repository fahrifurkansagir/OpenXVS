/**
 * Pass 3: Translucent See-Through Airframe Wireframe, Solid 3D Approaching Aircraft + WCV Cones,
 * XVS-MeshDAA 3-Ship Sight-Lines (verified green / rejected ghost dashed red),
 * 3-deg KSFO 28L HITS Tunnel, and High-Visibility KSFO 28L / 28R / Taxiway C Asphalt,
 * Threshold Piano Keys, Dashed Centerline, Runway Edge Lights & Live 4-Light PAPI Bar.
 */

import {
  DEG_TO_RAD,
  XVS6_CUBE_V1_CAMERAS,
  geodeticToEcef,
  mat3MulVec,
  rEcefNed,
  type Vec3,
} from "../rhi/coords.ts";
import type { RhiBufferHandle, RhiContext, RhiPipelineHandle } from "../rhi/types.ts";
import { SF_BAY_SUN_DIR_NED } from "./pass1_das_efvs.ts";
import pass3Wgsl from "../shaders/pass3_airframe_traffic.wgsl?raw";
import pass3GlVert from "../shaders/generated_gl/pass3_airframe_traffic.vert.glsl?raw";
import pass3GlFrag from "../shaders/generated_gl/pass3_airframe_traffic.frag.glsl?raw";
import type {
  ActiveCameraView,
  CockpitFrameState,
  RunwayGeometry,
  TerrainPackData,
} from "./types.ts";

export interface ActiveRangeOverlayState {
  readonly ownshipEcef?: Vec3;
  readonly targetPosEcef?: Vec3;
  readonly bearingEcef?: Vec3;
  readonly rHatM: number;
  readonly sigmaRM: number;
  readonly rSafeLowerM: number;
  readonly minSc3bfMargin?: number;
  readonly sc3bfRadiusM?: number;
  readonly mode?: number;
}

export interface ForcedLandFunnelWaypoint {
  readonly posEcef: Vec3;
  readonly halfWidthM?: number;
  readonly halfHeightM?: number;
}

export interface ForcedLandFunnelState {
  readonly waypoints?: readonly ForcedLandFunnelWaypoint[];
  readonly crosswindKt?: number;
  readonly targetEcef?: Vec3;
  readonly active?: boolean;
}

export interface ManifoldToggleState {
  readonly activeRangeCone?: boolean;
  readonly sc3bfTube?: boolean;
  readonly forcedLandFunnel?: boolean;
  readonly spzStress?: boolean;
}

declare module "./types.ts" {
  interface CockpitFrameState {
    readonly activeRangeOverlay?: ActiveRangeOverlayState;
    readonly forcedLandFunnel?: ForcedLandFunnelState;
    readonly manifoldToggles?: ManifoldToggleState;
  }
}

export interface PapiLightState {
  readonly index: number;
  readonly transitionDeg: number;
  readonly isWhite: boolean;
  readonly rgba: readonly [number, number, number, number];
}

export interface ManifoldGeometryOutput {
  readonly vertices: Float32Array;
  readonly vertexCount: number;
  readonly meanColorRgb: readonly [number, number, number];
}

export interface ActiveRangeConeGeometry extends ManifoldGeometryOutput {
  readonly axialSpanM: number;
  readonly rNearM: number;
  readonly rFarM: number;
}

export interface ActiveRangeConeParams {
  readonly originRte: Vec3;
  readonly bearingDir: Vec3;
  readonly upRef: Vec3;
  readonly rHatM: number;
  readonly sigmaRM: number;
  readonly rSafeLowerM: number;
}

export interface Sc3bfBarrierTubeParams {
  readonly centerRte: Vec3;
  readonly axisDir: Vec3;
  readonly upRef: Vec3;
  readonly radiusM: number;
  readonly halfLengthM: number;
  readonly safe: boolean;
}

export interface ForcedLandFunnelParams {
  readonly originRte: Vec3;
  readonly targetRte: Vec3;
  readonly upRef: Vec3;
  readonly crosswindKt?: number;
  readonly waypointsRte?: ReadonlyArray<{
    readonly posRte: Vec3;
    readonly halfWidthM?: number;
    readonly halfHeightM?: number;
  }>;
}

const FLOATS_PER_VERTEX = 9; // posRte(3) + color(4) + dashMeta(2)
const MAX_LINE_VERTICES = 28672;
const MAX_SOLID_TRI_VERTICES = 32768;
const MAX_TRANSLUCENT_VERTICES = 16384;
const MAX_HALO_VERTICES = 12288;
export const MAX_MANIFOLD_VERTICES = 3500;
const MAX_TRAIL_POINTS = 120;

const PAPI_TRANSITION_ANGLES_DEG: readonly [number, number, number, number] = [
  3.5,
  3.17,
  2.83,
  2.5,
];

type Rgba = readonly [number, number, number, number];

function clampNum(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function normalize3(v: Vec3): Vec3 {
  const len = Math.hypot(v[0], v[1], v[2]);
  if (len < 1e-6) return [1, 0, 0];
  return [v[0] / len, v[1] / len, v[2] / len];
}

function cross3(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function dot3(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function buildOrthonormalBasis(axis: Vec3, upRef: Vec3): {
  readonly fwd: Vec3;
  readonly right: Vec3;
  readonly up: Vec3;
} {
  const fwd = normalize3(axis);
  let rRaw = cross3(fwd, upRef);
  if (Math.hypot(rRaw[0], rRaw[1], rRaw[2]) < 1e-4) {
    const alt: Vec3 = Math.abs(fwd[0]) < 0.8 ? [1, 0, 0] : [0, 1, 0];
    rRaw = cross3(fwd, alt);
  }
  const right = normalize3(rRaw);
  const up = normalize3(cross3(right, fwd));
  return { fwd, right, up };
}

/**
 * Compute standard 4-light PAPI lamp states from glideslope angle (deg):
 *   - > 3.5 deg: 4 white (high)
 *   - 3.0 deg: 2 white / 2 red (on 3.0-deg glideslope)
 *   - < 2.5 deg: 4 red (low)
 */
export function computePapiLightStates(glideslopeDeg: number): readonly PapiLightState[] {
  const gs = Number.isFinite(glideslopeDeg) ? glideslopeDeg : 3.0;
  return PAPI_TRANSITION_ANGLES_DEG.map((transitionDeg, index) => {
    const isWhite = gs >= transitionDeg;
    const rgba: Rgba = isWhite ? [1.0, 0.98, 0.92, 0.99] : [1.0, 0.16, 0.14, 0.99];
    return { index, transitionDeg, isWhite, rgba };
  });
}

function packTriangleList(
  tris: ReadonlyArray<{
    readonly a: Vec3;
    readonly b: Vec3;
    readonly c: Vec3;
    readonly col: Rgba;
    readonly pulseMode?: number;
    readonly phaseOffset?: number;
  }>,
): ManifoldGeometryOutput {
  const vertexCount = tris.length * 3;
  const vertices = new Float32Array(vertexCount * FLOATS_PER_VERTEX);
  let rSum = 0;
  let gSum = 0;
  let bSum = 0;
  let outIdx = 0;

  for (const t of tris) {
    const pm = t.pulseMode ?? 0.0;
    const po = t.phaseOffset ?? 0.0;
    const col = t.col;
    rSum += col[0];
    gSum += col[1];
    bSum += col[2];
    for (const p of [t.a, t.b, t.c]) {
      const o = outIdx * FLOATS_PER_VERTEX;
      vertices[o + 0] = p[0];
      vertices[o + 1] = p[1];
      vertices[o + 2] = p[2];
      vertices[o + 3] = col[0];
      vertices[o + 4] = col[1];
      vertices[o + 5] = col[2];
      vertices[o + 6] = col[3];
      vertices[o + 7] = po;
      vertices[o + 8] = pm;
      outIdx++;
    }
  }

  const inv = tris.length > 0 ? 1.0 / tris.length : 1.0;
  return {
    vertices,
    vertexCount,
    meanColorRgb: [rSum * inv, gSum * inv, bSum * inv],
  };
}

/**
 * Pure geometry builder for Manifold 1: XVS-ActiveRange Collapsing 3D CRLB/Conformal Range Cone.
 * Spans [rSafeLowerM, rHatM + 2*sigmaRM] along the intruder bearing; collapses axially and shifts
 * from warm amber-red (high sigma_r) to glowing emerald lock capsule (low sigma_r).
 */
export function buildActiveRangeConeGeometry(
  params: ActiveRangeConeParams,
): ActiveRangeConeGeometry {
  const { originRte, bearingDir, upRef } = params;
  const rHat = Math.max(40.0, Number.isFinite(params.rHatM) ? params.rHatM : 1000.0);
  const sigmaR = Math.max(5.0, Number.isFinite(params.sigmaRM) ? params.sigmaRM : 200.0);
  const rNear = clampNum(
    Number.isFinite(params.rSafeLowerM) ? params.rSafeLowerM : rHat - 1.65 * sigmaR,
    25.0,
    rHat,
  );
  const rFar = Math.max(rNear + 24.0, rHat + 2.0 * sigmaR);
  const axialSpanM = rFar - rNear;

  // Uncertainty norm: 0 = locked (<= 40 m -> emerald), 1 = wide uncertain (>= 600 m -> amber-red)
  const uNorm = clampNum((sigmaR - 40.0) / 560.0, 0.0, 1.0);
  const rCol = 0.12 + (1.0 - 0.12) * uNorm;
  const gCol = 0.96 + (0.40 - 0.96) * uNorm;
  const bCol = 0.56 + (0.12 - 0.56) * uNorm;

  const { fwd, right, up } = buildOrthonormalBasis(bearingDir, upRef);
  const ringPt = (sM: number, radM: number, angleRad: number): Vec3 => {
    const ca = Math.cos(angleRad) * radM;
    const sa = Math.sin(angleRad) * radM;
    return [
      originRte[0] + fwd[0] * sM + right[0] * ca + up[0] * sa,
      originRte[1] + fwd[1] * sM + right[1] * ca + up[1] * sa,
      originRte[2] + fwd[2] * sM + right[2] * ca + up[2] * sa,
    ];
  };

  // Cone angular radius tightens visually as sigma collapses
  const tanHalfAngle = 0.032 + 0.048 * uNorm;
  const segs = 16;
  const rMid = clampNum(rHat, rNear + axialSpanM * 0.2, rFar - axialSpanM * 0.2);
  const rings: readonly { readonly s: number; readonly rad: number; readonly alpha: number }[] = [
    { s: rNear, rad: Math.max(16.0, rNear * tanHalfAngle * 0.85), alpha: 0.24 + 0.10 * (1.0 - uNorm) },
    { s: rMid, rad: Math.max(22.0, rMid * tanHalfAngle * (0.95 + 0.15 * uNorm)), alpha: 0.18 },
    { s: rFar, rad: Math.max(18.0, rFar * tanHalfAngle * (0.75 + 0.30 * uNorm)), alpha: 0.11 },
  ];

  const tris: Array<{
    readonly a: Vec3;
    readonly b: Vec3;
    readonly c: Vec3;
    readonly col: Rgba;
    readonly pulseMode?: number;
    readonly phaseOffset?: number;
  }> = [];

  // Frustum side shells (2 longitudinal bands x 16 segs x 2 tris = 64 tris = 192 verts)
  for (let rIdx = 0; rIdx < rings.length - 1; rIdx++) {
    const r0 = rings[rIdx];
    const r1 = rings[rIdx + 1];
    const bandAlpha = 0.5 * (r0.alpha + r1.alpha);
    const bandCol: Rgba = [rCol, gCol, bCol, bandAlpha];
    for (let k = 0; k < segs; k++) {
      const a0 = (k / segs) * Math.PI * 2.0;
      const a1 = ((k + 1) / segs) * Math.PI * 2.0;
      const p00 = ringPt(r0.s, r0.rad, a0);
      const p01 = ringPt(r0.s, r0.rad, a1);
      const p10 = ringPt(r1.s, r1.rad, a0);
      const p11 = ringPt(r1.s, r1.rad, a1);
      tris.push({ a: p00, b: p01, c: p11, col: bandCol });
      tris.push({ a: p00, b: p11, c: p10, col: bandCol });
    }
  }

  // Glowing conformal lower-bound safety floor collar at rSafeLowerM (16 tris = 48 verts)
  const nearCenter = ringPt(rNear, 0.0, 0.0);
  const floorCol: Rgba = [
    clampNum(rCol + 0.08, 0.0, 1.0),
    clampNum(gCol + 0.08, 0.0, 1.0),
    clampNum(bCol + 0.08, 0.0, 1.0),
    0.26,
  ];
  for (let k = 0; k < segs; k++) {
    const a0 = (k / segs) * Math.PI * 2.0;
    const a1 = ((k + 1) / segs) * Math.PI * 2.0;
    tris.push({
      a: nearCenter,
      b: ringPt(rNear, rings[0].rad, a0),
      c: ringPt(rNear, rings[0].rad, a1),
      col: floorCol,
    });
  }

  // Glowing lock-core octahedron centered at rHat (8 tris = 24 verts; intensifies as sigma collapses)
  const coreR = clampNum(18.0 + sigmaR * 0.06, 16.0, 48.0);
  const cF = ringPt(rHat + coreR, 0.0, 0.0);
  const cB = ringPt(rHat - coreR, 0.0, 0.0);
  const c0 = ringPt(rHat, coreR, 0.0);
  const c1 = ringPt(rHat, coreR, Math.PI * 0.5);
  const c2 = ringPt(rHat, coreR, Math.PI);
  const c3 = ringPt(rHat, coreR, Math.PI * 1.5);
  const coreCol: Rgba = [
    clampNum(rCol + 0.12, 0.0, 1.0),
    clampNum(gCol + 0.12, 0.0, 1.0),
    clampNum(bCol + 0.12, 0.0, 1.0),
    0.34 + 0.22 * (1.0 - uNorm),
  ];
  const eq = [c0, c1, c2, c3];
  for (let k = 0; k < 4; k++) {
    const n = eq[(k + 1) % 4];
    tris.push({ a: cF, b: eq[k], c: n, col: coreCol, pulseMode: uNorm < 0.2 ? -1.0 : 0.0 });
    tris.push({ a: cB, b: n, c: eq[k], col: coreCol, pulseMode: uNorm < 0.2 ? -1.0 : 0.0 });
  }

  const packed = packTriangleList(tris);
  return {
    ...packed,
    axialSpanM,
    rNearM: rNear,
    rFarM: rFar,
  };
}

/**
 * Pure geometry builder for Manifold 2: SC3BF Shifted Collision-Cone Barrier Tube.
 * Translucent cyan sheath (safe: cbf_margin >= 0) or pulsing rose-crimson sheath (violation).
 */
export function buildSc3bfBarrierTubeGeometry(
  params: Sc3bfBarrierTubeParams,
): ManifoldGeometryOutput {
  const { centerRte, axisDir, upRef, safe } = params;
  const radiusM = Math.max(20.0, Number.isFinite(params.radiusM) ? params.radiusM : 152.4);
  const halfLen = Math.max(30.0, Number.isFinite(params.halfLengthM) ? params.halfLengthM : 210.0);
  const { fwd, right, up } = buildOrthonormalBasis(axisDir, upRef);

  const ringPt = (sM: number, radM: number, angleRad: number): Vec3 => {
    const ca = Math.cos(angleRad) * radM;
    const sa = Math.sin(angleRad) * radM;
    return [
      centerRte[0] + fwd[0] * sM + right[0] * ca + up[0] * sa,
      centerRte[1] + fwd[1] * sM + right[1] * ca + up[1] * sa,
      centerRte[2] + fwd[2] * sM + right[2] * ca + up[2] * sa,
    ];
  };

  const baseCol: Rgba = safe ? [0.14, 0.88, 0.98, 0.16] : [1.0, 0.18, 0.42, 0.28];
  const collarCol: Rgba = safe ? [0.22, 0.96, 1.0, 0.24] : [1.0, 0.24, 0.48, 0.36];
  const pulse = safe ? 0.0 : -1.0;

  const stations: readonly { readonly s: number; readonly r: number; readonly col: Rgba }[] = [
    { s: -halfLen, r: radiusM * 0.72, col: collarCol },
    { s: -halfLen * 0.55, r: radiusM, col: baseCol },
    { s: halfLen * 0.55, r: radiusM, col: baseCol },
    { s: halfLen, r: radiusM * 0.72, col: collarCol },
  ];

  const segs = 16;
  const tris: Array<{
    readonly a: Vec3;
    readonly b: Vec3;
    readonly c: Vec3;
    readonly col: Rgba;
    readonly pulseMode?: number;
  }> = [];

  for (let i = 0; i < stations.length - 1; i++) {
    const st0 = stations[i];
    const st1 = stations[i + 1];
    const segCol = i === 1 ? baseCol : collarCol;
    for (let k = 0; k < segs; k++) {
      const a0 = (k / segs) * Math.PI * 2.0;
      const a1 = ((k + 1) / segs) * Math.PI * 2.0;
      const p00 = ringPt(st0.s, st0.r, a0);
      const p01 = ringPt(st0.s, st0.r, a1);
      const p10 = ringPt(st1.s, st1.r, a0);
      const p11 = ringPt(st1.s, st1.r, a1);
      tris.push({ a: p00, b: p01, c: p11, col: segCol, pulseMode: pulse });
      tris.push({ a: p00, b: p11, c: p10, col: segCol, pulseMode: pulse });
    }
  }

  return packTriangleList(tris);
}

/**
 * Pure geometry builder for Manifold 3: XVS-ForcedLand 3D Wind-Warped Trochoidal Glide Funnel.
 * Golden-emerald volumetric descent corridor curving smoothly with live crosswind down to KSFO 28L.
 */
export function buildForcedLandFunnelGeometry(
  params: ForcedLandFunnelParams,
): ManifoldGeometryOutput {
  const { originRte, targetRte, upRef } = params;
  const crosswindKt = Number.isFinite(params.crosswindKt ?? 0) ? (params.crosswindKt ?? 0) : 0;

  let samples: Array<{ readonly pos: Vec3; readonly hw: number; readonly hh: number }> = [];
  if (params.waypointsRte && params.waypointsRte.length >= 2) {
    const raw = params.waypointsRte;
    const maxSegs = 16;
    const step = Math.max(1, Math.ceil((raw.length - 1) / maxSegs));
    for (let i = 0; i < raw.length; i += step) {
      const t = i / Math.max(1, raw.length - 1);
      samples.push({
        pos: raw[i].posRte,
        hw: raw[i].halfWidthM ?? (68.0 - 44.0 * t),
        hh: raw[i].halfHeightM ?? (30.0 - 19.0 * t),
      });
    }
    const last = raw[raw.length - 1];
    if (samples[samples.length - 1].pos !== last.posRte) {
      samples.push({
        pos: last.posRte,
        hw: last.halfWidthM ?? 24.0,
        hh: last.halfHeightM ?? 11.0,
      });
    }
  } else {
    const chord: Vec3 = [
      targetRte[0] - originRte[0],
      targetRte[1] - originRte[1],
      targetRte[2] - originRte[2],
    ];
    const chordLen = Math.max(100.0, Math.hypot(chord[0], chord[1], chord[2]));
    const { right, up } = buildOrthonormalBasis(chord, upRef);
    const windBowM = (crosswindKt / 35.0) * Math.min(340.0, chordLen * 0.14);
    const segs = 14;
    samples = [];
    for (let i = 0; i <= segs; i++) {
      const t = i / segs;
      // Trochoidal wind-warped bow peaking mid-glide and aligning cleanly at runway threshold
      const latWarp = Math.sin(Math.PI * t) * Math.pow(1.0 - 0.35 * t, 1.2) * windBowM;
      const vertSag = Math.sin(Math.PI * t) * Math.min(45.0, chordLen * 0.015);
      const pos: Vec3 = [
        originRte[0] + chord[0] * t + right[0] * latWarp + up[0] * vertSag,
        originRte[1] + chord[1] * t + right[1] * latWarp + up[1] * vertSag,
        originRte[2] + chord[2] * t + right[2] * latWarp + up[2] * vertSag,
      ];
      samples.push({
        pos,
        hw: 66.0 - 42.0 * t,
        hh: 28.0 - 17.0 * t,
      });
    }
  }

  // Clip near-camera segments (< 65 m from ownship origin) so cockpit camera never slices through a near-plane quad
  const MIN_FUNNEL_START_M = 65.0;
  let clippedSamples: Array<{ readonly pos: Vec3; readonly hw: number; readonly hh: number }> = [];
  let accumDist = 0.0;
  for (let i = 0; i < samples.length; i++) {
    if (i > 0) {
      const prev = samples[i - 1];
      const cur = samples[i];
      const segLen = Math.hypot(
        cur.pos[0] - prev.pos[0],
        cur.pos[1] - prev.pos[1],
        cur.pos[2] - prev.pos[2],
      );
      if (accumDist < MIN_FUNNEL_START_M && accumDist + segLen >= MIN_FUNNEL_START_M && segLen > 1e-3) {
        const f = (MIN_FUNNEL_START_M - accumDist) / segLen;
        clippedSamples.push({
          pos: [
            prev.pos[0] + (cur.pos[0] - prev.pos[0]) * f,
            prev.pos[1] + (cur.pos[1] - prev.pos[1]) * f,
            prev.pos[2] + (cur.pos[2] - prev.pos[2]) * f,
          ],
          hw: prev.hw + (cur.hw - prev.hw) * f,
          hh: prev.hh + (cur.hh - prev.hh) * f,
        });
      }
      accumDist += segLen;
    }
    if (accumDist >= MIN_FUNNEL_START_M) {
      clippedSamples.push(samples[i]);
    }
  }
  if (clippedSamples.length < 2) {
    clippedSamples = samples;
  }

  const tris: Array<{
    readonly a: Vec3;
    readonly b: Vec3;
    readonly c: Vec3;
    readonly col: Rgba;
  }> = [];

  for (let i = 0; i < clippedSamples.length - 1; i++) {
    const s0 = clippedSamples[i];
    const s1 = clippedSamples[i + 1];
    const segDir: Vec3 = [
      s1.pos[0] - s0.pos[0],
      s1.pos[1] - s0.pos[1],
      s1.pos[2] - s0.pos[2],
    ];
    if (Math.hypot(segDir[0], segDir[1], segDir[2]) < 1.0) continue;
    const { right, up } = buildOrthonormalBasis(segDir, upRef);

    const t = i / Math.max(1, clippedSamples.length - 2);
    // Golden high-altitude glide transitioning smoothly to luminous emerald at touchdown
    const rC = 0.96 + (0.16 - 0.96) * t;
    const gC = 0.80 + (0.98 - 0.80) * t;
    const bC = 0.22 + (0.58 - 0.22) * t;
    const wallCol: Rgba = [rC, gC, bC, 0.11 + 0.07 * t];
    const floorCol: Rgba = [rC, gC, bC, 0.24 + 0.12 * t];

    const corner = (c: Vec3, dx: number, dy: number): Vec3 => [
      c[0] + right[0] * dx + up[0] * dy,
      c[1] + right[1] * dx + up[1] * dy,
      c[2] + right[2] * dx + up[2] * dy,
    ];

    const bL0 = corner(s0.pos, -s0.hw, -s0.hh);
    const bR0 = corner(s0.pos, s0.hw, -s0.hh);
    const tR0 = corner(s0.pos, s0.hw, s0.hh);
    const tL0 = corner(s0.pos, -s0.hw, s0.hh);

    const bL1 = corner(s1.pos, -s1.hw, -s1.hh);
    const bR1 = corner(s1.pos, s1.hw, -s1.hh);
    const tR1 = corner(s1.pos, s1.hw, s1.hh);
    const tL1 = corner(s1.pos, -s1.hw, s1.hh);

    // Floor ribbon (brightest guidance surface)
    tris.push({ a: bL0, b: bR0, c: bR1, col: floorCol });
    tris.push({ a: bL0, b: bR1, c: bL1, col: floorCol });
    // Left & right corridor side walls
    tris.push({ a: bL0, b: tL0, c: tL1, col: wallCol });
    tris.push({ a: bL0, b: tL1, c: bL1, col: wallCol });
    tris.push({ a: bR0, b: bR1, c: tR1, col: wallCol });
    tris.push({ a: bR0, b: tR1, c: tR0, col: wallCol });
  }

  return packTriangleList(tris);
}

export class Pass3AirframeTraffic {
  private readonly rhi: RhiContext;
  private readonly linePipeline: RhiPipelineHandle;
  private readonly solidTriPipeline: RhiPipelineHandle;
  private readonly translucentVolPipeline: RhiPipelineHandle;
  private readonly lineVb: RhiBufferHandle;
  private readonly solidVb: RhiBufferHandle;
  private readonly translucentVb: RhiBufferHandle;
  private readonly haloVb: RhiBufferHandle;
  private readonly ubo: RhiBufferHandle;
  private readonly haloUbo: RhiBufferHandle;
  private readonly uniformData = new Float32Array(24); // mat4 (16) + 2 x vec4 (8) = 96 B
  private readonly haloUniformData = new Float32Array(24);
  private readonly lineScratch = new Float32Array(MAX_LINE_VERTICES * FLOATS_PER_VERTEX);
  private readonly solidScratch = new Float32Array(MAX_SOLID_TRI_VERTICES * FLOATS_PER_VERTEX);
  private readonly translucentScratch = new Float32Array(MAX_TRANSLUCENT_VERTICES * FLOATS_PER_VERTEX);
  private readonly haloScratch = new Float32Array(MAX_HALO_VERTICES * FLOATS_PER_VERTEX);
  private lineVertexCount = 0;
  private solidVertexCount = 0;
  private translucentVertexCount = 0;
  private haloVertexCount = 0;
  private readonly ownshipTrail: Vec3[] = [];
  private readonly intruderTrail: Vec3[] = [];
  private lastTrailSimTimeS = -1.0;

  constructor(rhi: RhiContext) {
    this.rhi = rhi;
    this.lineVb = rhi.createBuffer("vertex", this.lineScratch.byteLength, "pass3-lines-vb");
    this.solidVb = rhi.createBuffer("vertex", this.solidScratch.byteLength, "pass3-solid-tris-vb");
    this.translucentVb = rhi.createBuffer(
      "vertex",
      this.translucentScratch.byteLength,
      "pass3-translucent-vol-vb",
    );
    this.haloVb = rhi.createBuffer("vertex", this.haloScratch.byteLength, "pass3-halo-vb");
    this.ubo = rhi.createBuffer("uniform", this.uniformData.byteLength, "pass3-ubo");
    this.haloUbo = rhi.createBuffer("uniform", this.haloUniformData.byteLength, "pass3-halo-ubo");

    const vertexLayouts = [
      {
        arrayStride: FLOATS_PER_VERTEX * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" as const },
          { shaderLocation: 1, offset: 12, format: "float32x4" as const },
          { shaderLocation: 2, offset: 28, format: "float32x2" as const },
        ],
      },
    ];

    this.solidTriPipeline = rhi.createRenderPipeline({
      label: "pass3-airframe-traffic-solid-tris",
      wgsl: pass3Wgsl,
      glslVert: pass3GlVert,
      glslFrag: pass3GlFrag,
      topology: "triangle-list",
      depthTest: true,
      depthWrite: true,
      blendAlpha: false,
      vertexLayouts,
      uniformBindings: [0],
    });

    this.translucentVolPipeline = rhi.createRenderPipeline({
      label: "pass3-airframe-traffic-translucent-vol",
      wgsl: pass3Wgsl,
      glslVert: pass3GlVert,
      glslFrag: pass3GlFrag,
      topology: "triangle-list",
      depthTest: true,
      depthWrite: false,
      blendAlpha: true,
      vertexLayouts,
      uniformBindings: [0],
    });

    this.linePipeline = rhi.createRenderPipeline({
      label: "pass3-airframe-traffic-lines",
      wgsl: pass3Wgsl,
      glslVert: pass3GlVert,
      glslFrag: pass3GlFrag,
      topology: "line-list",
      depthTest: true,
      depthWrite: false,
      blendAlpha: true,
      vertexLayouts,
      uniformBindings: [0],
    });
  }

  private pushLineRte(a: Vec3, b: Vec3, col: Rgba, dashPeriodM = 0.0): void {
    if (this.lineVertexCount + 2 > MAX_LINE_VERTICES) return;
    const dist = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    const o0 = this.lineVertexCount * FLOATS_PER_VERTEX;
    const v = this.lineScratch;
    v[o0 + 0] = a[0]; v[o0 + 1] = a[1]; v[o0 + 2] = a[2];
    v[o0 + 3] = col[0]; v[o0 + 4] = col[1]; v[o0 + 5] = col[2]; v[o0 + 6] = col[3];
    v[o0 + 7] = 0.0; v[o0 + 8] = dashPeriodM;

    const o1 = o0 + FLOATS_PER_VERTEX;
    v[o1 + 0] = b[0]; v[o1 + 1] = b[1]; v[o1 + 2] = b[2];
    v[o1 + 3] = col[0]; v[o1 + 4] = col[1]; v[o1 + 5] = col[2]; v[o1 + 6] = col[3];
    v[o1 + 7] = dist; v[o1 + 8] = dashPeriodM;

    this.lineVertexCount += 2;
  }

  private pushLineEcef(a: Vec3, b: Vec3, pCam: Vec3, col: Rgba, dashPeriodM = 0.0): void {
    this.pushLineRte(
      [a[0] - pCam[0], a[1] - pCam[1], a[2] - pCam[2]],
      [b[0] - pCam[0], b[1] - pCam[1], b[2] - pCam[2]],
      col,
      dashPeriodM,
    );
  }

  private pushSolidTriRte(
    a: Vec3,
    b: Vec3,
    c: Vec3,
    col: Rgba,
    pulseMode = 0.0,
    phaseOffset = 0.0,
  ): void {
    if (this.solidVertexCount + 3 > MAX_SOLID_TRI_VERTICES) return;
    const v = this.solidScratch;
    const write = (idx: number, p: Vec3) => {
      const o = idx * FLOATS_PER_VERTEX;
      v[o + 0] = p[0]; v[o + 1] = p[1]; v[o + 2] = p[2];
      v[o + 3] = col[0]; v[o + 4] = col[1]; v[o + 5] = col[2]; v[o + 6] = 1.0;
      v[o + 7] = phaseOffset; v[o + 8] = pulseMode;
    };
    const base = this.solidVertexCount;
    write(base + 0, a);
    write(base + 1, b);
    write(base + 2, c);
    this.solidVertexCount += 3;
  }

  private pushTranslucentTriRte(
    a: Vec3,
    b: Vec3,
    c: Vec3,
    col: Rgba,
    pulseMode = 0.0,
    phaseOffset = 0.0,
  ): void {
    if (this.translucentVertexCount + 3 > MAX_TRANSLUCENT_VERTICES) return;
    const v = this.translucentScratch;
    const write = (idx: number, p: Vec3) => {
      const o = idx * FLOATS_PER_VERTEX;
      v[o + 0] = p[0]; v[o + 1] = p[1]; v[o + 2] = p[2];
      v[o + 3] = col[0]; v[o + 4] = col[1]; v[o + 5] = col[2]; v[o + 6] = col[3];
      v[o + 7] = phaseOffset; v[o + 8] = pulseMode;
    };
    const base = this.translucentVertexCount;
    write(base + 0, a);
    write(base + 1, b);
    write(base + 2, c);
    this.translucentVertexCount += 3;
  }

  private appendTranslucentPacked(verts: Float32Array, vertexCount: number): void {
    const avail = MAX_TRANSLUCENT_VERTICES - this.translucentVertexCount;
    const count = Math.min(avail, vertexCount);
    if (count <= 0) return;
    this.translucentScratch.set(
      verts.subarray(0, count * FLOATS_PER_VERTEX),
      this.translucentVertexCount * FLOATS_PER_VERTEX,
    );
    this.translucentVertexCount += count;
  }

  private pushSolidQuadRte(
    a: Vec3,
    b: Vec3,
    c: Vec3,
    d: Vec3,
    col: Rgba,
    pulseMode = 0.0,
    phaseOffset = 0.0,
  ): void {
    this.pushSolidTriRte(a, b, c, col, pulseMode, phaseOffset);
    this.pushSolidTriRte(a, c, d, col, pulseMode, phaseOffset);
  }

  private pushTranslucentQuadEcef(
    a: Vec3,
    b: Vec3,
    c: Vec3,
    d: Vec3,
    pCam: Vec3,
    col: Rgba,
    pulseMode = 0.0,
    phaseOffset = 0.0,
  ): void {
    const aR: Vec3 = [a[0] - pCam[0], a[1] - pCam[1], a[2] - pCam[2]];
    const bR: Vec3 = [b[0] - pCam[0], b[1] - pCam[1], b[2] - pCam[2]];
    const cR: Vec3 = [c[0] - pCam[0], c[1] - pCam[1], c[2] - pCam[2]];
    const dR: Vec3 = [d[0] - pCam[0], d[1] - pCam[1], d[2] - pCam[2]];
    this.pushTranslucentTriRte(aR, bR, cR, col, pulseMode, phaseOffset);
    this.pushTranslucentTriRte(aR, cR, dR, col, pulseMode, phaseOffset);
  }

  private pushQuadEcef(
    a: Vec3,
    b: Vec3,
    c: Vec3,
    d: Vec3,
    pCam: Vec3,
    col: Rgba,
    pulseMode = 0.0,
    phaseOffset = 0.0,
  ): void {
    this.pushSolidQuadRte(
      [a[0] - pCam[0], a[1] - pCam[1], a[2] - pCam[2]],
      [b[0] - pCam[0], b[1] - pCam[1], b[2] - pCam[2]],
      [c[0] - pCam[0], c[1] - pCam[1], c[2] - pCam[2]],
      [d[0] - pCam[0], d[1] - pCam[1], d[2] - pCam[2]],
      col,
      pulseMode,
      phaseOffset,
    );
  }

  /**
   * Camera-facing emissive radial-halo billboard in ECEF (for 4-light PAPI, ALSF-2,
   * runway edge lights, nav lights & pulsing anti-collision strobes).
   * Passes UV in [-1, 1]^2 via dashMeta.xy for smooth Gaussian halo falloff.
   */
  private pushBeaconEcef(
    center: Vec3,
    radiusM: number,
    view: ActiveCameraView,
    col: Rgba,
    pulse = false,
    phaseOffset = 0.0,
    simTimeS = 0.0,
  ): void {
    if (this.haloVertexCount + 6 > MAX_HALO_VERTICES) return;
    const pCam = view.pEcefCam;
    const cRte: Vec3 = [center[0] - pCam[0], center[1] - pCam[1], center[2] - pCam[2]];
    const camR: Vec3 = [view.rEcefCam[0], view.rEcefCam[3], view.rEcefCam[6]];
    const camU: Vec3 = [-view.rEcefCam[1], -view.rEcefCam[4], -view.rEcefCam[7]];

    let rC = col[0];
    let gC = col[1];
    let bC = col[2];
    let aC = col[3];
    let rEff = radiusM;
    if (pulse) {
      const strobe = 0.35 + 0.65 * Math.pow(Math.max(Math.sin(simTimeS * 9.0 + phaseOffset), 0.0), 4.0);
      rC = clampNum(rC * (0.75 + 0.55 * strobe), 0.0, 1.0);
      gC = clampNum(gC * (0.75 + 0.55 * strobe), 0.0, 1.0);
      bC = clampNum(bC * (0.75 + 0.55 * strobe), 0.0, 1.0);
      aC = clampNum(aC * (0.45 + 0.55 * strobe), 0.0, 1.0);
      rEff = radiusM * (0.85 + 0.35 * strobe);
    }

    const rx = camR[0] * rEff, ry = camR[1] * rEff, rz = camR[2] * rEff;
    const ux = camU[0] * rEff, uy = camU[1] * rEff, uz = camU[2] * rEff;
    const bl: Vec3 = [cRte[0] - rx - ux, cRte[1] - ry - uy, cRte[2] - rz - uz];
    const br: Vec3 = [cRte[0] + rx - ux, cRte[1] + ry - uy, cRte[2] + rz - uz];
    const tr: Vec3 = [cRte[0] + rx + ux, cRte[1] + ry + uy, cRte[2] + rz + uz];
    const tl: Vec3 = [cRte[0] - rx + ux, cRte[1] - ry + uy, cRte[2] - rz + uz];

    const v = this.haloScratch;
    const write = (idx: number, p: Vec3, uCoord: number, vCoord: number) => {
      const o = idx * FLOATS_PER_VERTEX;
      v[o + 0] = p[0]; v[o + 1] = p[1]; v[o + 2] = p[2];
      v[o + 3] = rC; v[o + 4] = gC; v[o + 5] = bC; v[o + 6] = aC;
      v[o + 7] = uCoord; v[o + 8] = vCoord;
    };
    const base = this.haloVertexCount;
    write(base + 0, bl, -1.0, -1.0);
    write(base + 1, br, 1.0, -1.0);
    write(base + 2, tr, 1.0, 1.0);
    write(base + 3, bl, -1.0, -1.0);
    write(base + 4, tr, 1.0, 1.0);
    write(base + 5, tl, -1.0, 1.0);
    this.haloVertexCount += 6;
  }

  private addAirframeWireframe(view: ActiveCameraView): void {
    // Translucent see-through twin-jet cockpit bow & glareshield wireframe in BODY FRD (metres from ARP)
    const bodyCol: Rgba = [0.22, 0.85, 0.92, 0.24];
    const noseCol: Rgba = [0.35, 0.96, 0.80, 0.30];
    const bodyOffsetRte: Vec3 = [
      view.pEcefBody[0] - view.pEcefCam[0],
      view.pEcefBody[1] - view.pEcefCam[1],
      view.pEcefBody[2] - view.pEcefCam[2],
    ];
    const toRte = (b: Vec3): Vec3 => {
      const e = mat3MulVec(view.rEcefBody, b);
      return [bodyOffsetRte[0] + e[0], bodyOffsetRte[1] + e[1], bodyOffsetRte[2] + e[2]];
    };

    // Nose cone & cockpit glareshield struts
    const nose = toRte([15.4, 0.0, 0.35]);
    const noseL = toRte([12.6, -1.65, 0.35]);
    const noseR = toRte([12.6, 1.65, 0.35]);
    const noseTop = toRte([12.2, 0.0, -1.15]);
    const noseBot = toRte([12.2, 0.0, 1.25]);
    this.pushLineRte(nose, noseL, noseCol);
    this.pushLineRte(nose, noseR, noseCol);
    this.pushLineRte(nose, noseTop, noseCol);
    this.pushLineRte(nose, noseBot, noseCol);
    this.pushLineRte(noseL, noseTop, bodyCol);
    this.pushLineRte(noseTop, noseR, bodyCol);
    this.pushLineRte(noseR, noseBot, bodyCol);
    this.pushLineRte(noseBot, noseL, bodyCol);

    // Wing leading/trailing wireframe (swept civil twin-jet)
    const wRootF = toRte([4.0, 0.0, 0.6]);
    const wTipL = toRte([-4.5, -17.0, 0.0]);
    const wTipR = toRte([-4.5, 17.0, 0.0]);
    const wRootA = toRte([-2.0, 0.0, 0.6]);
    this.pushLineRte(wRootF, wTipL, bodyCol);
    this.pushLineRte(wTipL, wRootA, bodyCol);
    this.pushLineRte(wRootF, wTipR, bodyCol);
    this.pushLineRte(wTipR, wRootA, bodyCol);
  }

  private addRunwaysAndHitsTunnel(
    state: CockpitFrameState,
    view: ActiveCameraView,
    pack: TerrainPackData,
  ): void {
    const pCam = view.pEcefCam;

    // 1. High-Visibility Solid Asphalt Runways (KSFO 28L, 28R, Taxiway C) + Markings + Edge Lights + PAPI
    for (const rwy of pack.runways) {
      const is28L = rwy.id === "28L";
      const isTaxi = rwy.kind === "taxiway" || rwy.id.toLowerCase().includes("taxiway");

      const thrLatRad = rwy.thresholdLatDeg * DEG_TO_RAD;
      const thrLonRad = rwy.thresholdLonDeg * DEG_TO_RAD;
      const pThr = geodeticToEcef(thrLatRad, thrLonRad, rwy.thresholdWgs84HM + 1.8);
      const rNed = rEcefNed(thrLatRad, thrLonRad);
      const hdgRad = rwy.trueHeadingDeg * DEG_TO_RAD;
      const fwdNed: Vec3 = [Math.cos(hdgRad), Math.sin(hdgRad), 0];
      const rightNed: Vec3 = [-Math.sin(hdgRad), Math.cos(hdgRad), 0];
      const upNed: Vec3 = [0, 0, -1];
      const fwdE = mat3MulVec(rNed, fwdNed);
      const rightE = mat3MulVec(rNed, rightNed);
      const upE = mat3MulVec(rNed, upNed);

      const halfW = Math.max(12.0, (rwy.widthM || (isTaxi ? 23.0 : 61.0)) * 0.5);
      const lenM = Math.max(400.0, rwy.lengthM || 3200.0);

      const rwyPt = (alongM: number, acrossM: number, liftM = 0.0): Vec3 => [
        pThr[0] + fwdE[0] * alongM + rightE[0] * acrossM + upE[0] * liftM,
        pThr[1] + fwdE[1] * alongM + rightE[1] * acrossM + upE[1] * liftM,
        pThr[2] + fwdE[2] * alongM + rightE[2] * acrossM + upE[2] * liftM,
      ];

      // Solid dark asphalt slab (warm tarmac signature in LWIR EFVS, crisp dark slate in daylight)
      const asphaltCol: Rgba = is28L
        ? [0.11, 0.14, 0.17, 0.96]
        : isTaxi
          ? [0.16, 0.14, 0.12, 0.90]
          : [0.10, 0.13, 0.16, 0.92];
      const c0 = rwyPt(0, -halfW, 0.0);
      const c1 = rwyPt(0, halfW, 0.0);
      const c2 = rwyPt(lenM, halfW, 0.0);
      const c3 = rwyPt(lenM, -halfW, 0.0);
      this.pushQuadEcef(c0, c1, c2, c3, pCam, asphaltCol);

      // Perimeter outline
      const outlineCol: Rgba = is28L
        ? [0.22, 1.0, 0.60, 0.95]
        : isTaxi
          ? [1.0, 0.56, 0.16, 0.82]
          : [0.38, 0.86, 1.0, 0.78];
      const dash = isTaxi ? 45.0 : 0.0;
      this.pushLineEcef(c0, c1, pCam, is28L ? [0.45, 1.0, 0.72, 1.0] : outlineCol, 0.0);
      this.pushLineEcef(c1, c2, pCam, outlineCol, dash);
      this.pushLineEcef(c2, c3, pCam, outlineCol, dash);
      this.pushLineEcef(c3, c0, pCam, outlineCol, dash);

      if (!isTaxi) {
        const markCol: Rgba = [0.95, 0.98, 1.0, 0.96];
        const sideStripeW = 1.6;
        // Left & right white runway edge stripes
        this.pushQuadEcef(
          rwyPt(0, -halfW + 1.0, 0.25),
          rwyPt(0, -halfW + 1.0 + sideStripeW, 0.25),
          rwyPt(lenM, -halfW + 1.0 + sideStripeW, 0.25),
          rwyPt(lenM, -halfW + 1.0, 0.25),
          pCam,
          markCol,
        );
        this.pushQuadEcef(
          rwyPt(0, halfW - 1.0 - sideStripeW, 0.25),
          rwyPt(0, halfW - 1.0, 0.25),
          rwyPt(lenM, halfW - 1.0, 0.25),
          rwyPt(lenM, halfW - 1.0 - sideStripeW, 0.25),
          pCam,
          markCol,
        );

        // Threshold piano keys (8 bars across runway width)
        const numKeys = 8;
        const keyStartM = 12.0;
        const keyEndM = 58.0;
        const usableW = (halfW - 4.0) * 2.0;
        const slotW = usableW / numKeys;
        for (let k = 0; k < numKeys; k++) {
          const yL = -halfW + 4.0 + k * slotW + slotW * 0.18;
          const yR = yL + slotW * 0.64;
          this.pushQuadEcef(
            rwyPt(keyStartM, yL, 0.30),
            rwyPt(keyStartM, yR, 0.30),
            rwyPt(keyEndM, yR, 0.30),
            rwyPt(keyEndM, yL, 0.30),
            pCam,
            markCol,
          );
        }

        // Aiming point blocks (300 m past threshold) + Touchdown-zone pairs
        for (const tdz of [
          { start: 150, end: 195, inner: 7.5, outer: 14.0 },
          { start: 290, end: 345, inner: 6.5, outer: 16.5 }, // Bold aiming point
          { start: 450, end: 495, inner: 7.5, outer: 13.5 },
          { start: 600, end: 640, inner: 7.5, outer: 12.5 },
        ]) {
          if (tdz.end < lenM) {
            this.pushQuadEcef(
              rwyPt(tdz.start, -tdz.outer, 0.30),
              rwyPt(tdz.start, -tdz.inner, 0.30),
              rwyPt(tdz.end, -tdz.inner, 0.30),
              rwyPt(tdz.end, -tdz.outer, 0.30),
              pCam,
              markCol,
            );
            this.pushQuadEcef(
              rwyPt(tdz.start, tdz.inner, 0.30),
              rwyPt(tdz.start, tdz.outer, 0.30),
              rwyPt(tdz.end, tdz.outer, 0.30),
              rwyPt(tdz.end, tdz.inner, 0.30),
              pCam,
              markCol,
            );
          }
        }

        // Dashed white runway centerline stripes
        for (let d = 80; d < lenM - 60; d += 72) {
          const dEnd = Math.min(d + 38, lenM - 40);
          this.pushQuadEcef(
            rwyPt(d, -1.2, 0.32),
            rwyPt(d, 1.2, 0.32),
            rwyPt(dEnd, 1.2, 0.32),
            rwyPt(dEnd, -1.2, 0.32),
            pCam,
            markCol,
          );
        }

        // High-intensity runway edge lights & green threshold lights (especially bright on 28L)
        const lightStep = is28L ? 120 : 240;
        const lightRad = is28L ? 5.2 : 3.6;
        for (let d = 0; d <= lenM; d += lightStep) {
          const isCautionZone = d > lenM - 600;
          const edgeCol: Rgba = isCautionZone
            ? [1.0, 0.78, 0.22, 0.96]
            : [0.96, 0.98, 1.0, 0.95];
          this.pushBeaconEcef(rwyPt(d, -halfW - 2.5, 1.8), lightRad, view, edgeCol, false);
          this.pushBeaconEcef(rwyPt(d, halfW + 2.5, 1.8), lightRad, view, edgeCol, false);
        }

        if (is28L) {
          // Green threshold light bar across the approach end of KSFO 28L
          const thrGreen: Rgba = [0.16, 1.0, 0.50, 0.98];
          for (let y = -halfW; y <= halfW; y += 10.0) {
            this.pushBeaconEcef(rwyPt(-3.0, y, 2.0), 5.6, view, thrGreen, false);
          }

          // Approach Lighting System (ALSF-2) lead-in centerline strobes extending 900 m before 28L threshold
          for (let d = 60; d <= 900; d += 75) {
            const isRabbit = d >= 300;
            this.pushBeaconEcef(
              rwyPt(-d, 0, 2.2),
              isRabbit ? 6.4 : 4.8,
              view,
              [0.98, 0.99, 1.0, 0.96],
              isRabbit,
              d * 0.015,
              state.simTimeS,
            );
            // Crossbar at 300 m (1000 ft decision bar)
            if (d === 300) {
              for (const cy of [-22, -14, -6, 6, 14, 22]) {
                this.pushBeaconEcef(
                  rwyPt(-d, cy, 2.2),
                  5.2,
                  view,
                  [1.0, 0.22, 0.22, 0.95],
                  false,
                  0.0,
                  state.simTimeS,
                );
              }
            }
          }

          // Live 4-Light PAPI Bar on the left side of KSFO 28L (~310 m past threshold)
          const papiCenter = rwyPt(310, -halfW - 22.0, 2.4);
          const ownToPapi: Vec3 = [
            view.pEcefBody[0] - papiCenter[0],
            view.pEcefBody[1] - papiCenter[1],
            view.pEcefBody[2] - papiCenter[2],
          ];
          const deltaUpM = dot3(ownToPapi, upE);
          const distAlongBackM = Math.max(40.0, -dot3(ownToPapi, fwdE));
          const glideslopeDeg = Math.atan2(Math.max(0.0, deltaUpM), distAlongBackM) * (180.0 / Math.PI);

          // Dark housing bar behind the 4 PAPI units
          this.pushQuadEcef(
            rwyPt(304, -halfW - 46.0, 1.0),
            rwyPt(304, -halfW - 8.0, 1.0),
            rwyPt(316, -halfW - 8.0, 1.0),
            rwyPt(316, -halfW - 46.0, 1.0),
            pCam,
            [0.05, 0.06, 0.08, 1.0],
          );

          const papiLights = computePapiLightStates(glideslopeDeg);
          const papiOffsets = [-halfW - 40.0, -halfW - 31.0, -halfW - 22.0, -halfW - 13.0];
          for (let p = 0; p < papiLights.length; p++) {
            this.pushBeaconEcef(
              rwyPt(310, papiOffsets[p], 3.4),
              8.6,
              view,
              papiLights[p].rgba,
              false,
              0.0,
              state.simTimeS,
            );
          }
        }
      }
    }

    // 2. 3-deg KSFO 28L Highway-in-the-Sky (HITS) Tunnel
    // Hard Safety Lock: Blanked immediately when simplexMode === 0 (INVALID_INHIBIT) or mandatoryGoAround is latched!
    if (state.approachActive && state.simplexMode !== 0 && !state.mandatoryGoAround) {
      const rwy28l: RunwayGeometry | undefined = pack.runways.find((r) => r.id === "28L");
      const thrLatRad = (rwy28l?.thresholdLatDeg ?? pack.ksfo28lThrLatDeg) * DEG_TO_RAD;
      const thrLonRad = (rwy28l?.thresholdLonDeg ?? pack.ksfo28lThrLonDeg) * DEG_TO_RAD;
      const thrHM = rwy28l?.thresholdWgs84HM ?? pack.ksfo28lThrWgs84HM;
      const hdgRad = (rwy28l?.trueHeadingDeg ?? 297.7) * DEG_TO_RAD;

      const pThr = geodeticToEcef(thrLatRad, thrLonRad, thrHM);
      const rNed = rEcefNed(thrLatRad, thrLonRad);
      const backN = -Math.cos(hdgRad);
      const backE = -Math.sin(hdgRad);
      const rightN = -Math.sin(hdgRad);
      const rightE = Math.cos(hdgRad);
      const tan3Deg = Math.tan(3.0 * DEG_TO_RAD);

      const gateCol: Rgba = state.efvsEnabled
        ? [0.65, 1.0, 0.84, 0.85]
        : [0.18, 1.0, 0.64, 0.78];
      const railCol: Rgba = [0.22, 0.92, 0.72, 0.36];
      const numGates = 10;
      let prevCorners: Vec3[] | null = null;

      for (let g = 1; g <= numGates; g++) {
        const distM = g * 500.0; // 500 m .. 5000 m out on final
        const hGate = distM * tan3Deg + 15.0; // 3-deg glideslope + 15 m TCH
        const halfW = 38.0 + distM * 0.012;
        const halfH = 18.0 + distM * 0.005;

        const cornersNed: Vec3[] = [
          [backN * distM - rightN * halfW, backE * distM - rightE * halfW, -(hGate + halfH)],
          [backN * distM + rightN * halfW, backE * distM + rightE * halfW, -(hGate + halfH)],
          [backN * distM + rightN * halfW, backE * distM + rightE * halfW, -(hGate - halfH)],
          [backN * distM - rightN * halfW, backE * distM - rightE * halfW, -(hGate - halfH)],
        ];
        const cornersEcef = cornersNed.map((c) => {
          const d = mat3MulVec(rNed, c);
          return [pThr[0] + d[0], pThr[1] + d[1], pThr[2] + d[2]] as Vec3;
        });

        for (let k = 0; k < 4; k++) {
          this.pushLineEcef(cornersEcef[k], cornersEcef[(k + 1) % 4], pCam, gateCol, 0.0);
        }
        // Bottom center glideslope tick chevron
        const botMid: Vec3 = [
          (cornersEcef[2][0] + cornersEcef[3][0]) * 0.5,
          (cornersEcef[2][1] + cornersEcef[3][1]) * 0.5,
          (cornersEcef[2][2] + cornersEcef[3][2]) * 0.5,
        ];
        this.pushBeaconEcef(botMid, 4.5, view, gateCol, false);

        // Longitudinal corridor guide rails connecting consecutive gates
        if (prevCorners) {
          this.pushLineEcef(prevCorners[2], cornersEcef[2], pCam, railCol, 35.0);
          this.pushLineEcef(prevCorners[3], cornersEcef[3], pCam, railCol, 35.0);
        }
        prevCorners = cornersEcef;
      }
    }
  }

  /**
   * Builds a solid shaded 3D twin-jet intruder aircraft silhouette (fuselage, swept wings,
   * horizontal stabilizer, tall vertical tailfin, twin underwing engines, glowing nav lights
   * & pulsing white anti-collision strobes) oriented along its real velocity / approach vector.
   */
  private addSolid3DAircraft(
    pT: Vec3,
    fwd: Vec3,
    right: Vec3,
    up: Vec3,
    scale: number,
    alertCol: Rgba,
    isRejectedGhost: boolean,
    view: ActiveCameraView,
    isOwnship = false,
    simTimeS = 0.0,
  ): void {
    const pCam = view.pEcefCam;
    const sunEcef = normalize3(mat3MulVec(view.rEcefNed, [
      SF_BAY_SUN_DIR_NED[0],
      SF_BAY_SUN_DIR_NED[1],
      SF_BAY_SUN_DIR_NED[2],
    ]));
    const upLocal: Vec3 = [-view.rEcefNed[2], -view.rEcefNed[5], -view.rEcefNed[8]];
    const pt = (xF: number, yR: number, zU: number): Vec3 => [
      pT[0] + (fwd[0] * xF + right[0] * yR + up[0] * zU) * scale,
      pT[1] + (fwd[1] * xF + right[1] * yR + up[1] * zU) * scale,
      pT[2] + (fwd[2] * xF + right[2] * yR + up[2] * zU) * scale,
    ];
    const toRte = (p: Vec3): Vec3 => [p[0] - pCam[0], p[1] - pCam[1], p[2] - pCam[2]];
    const pushFacetTri = (aE: Vec3, bE: Vec3, cE: Vec3, base: Rgba) => {
      const e1: Vec3 = [bE[0] - aE[0], bE[1] - aE[1], bE[2] - aE[2]];
      const e2: Vec3 = [cE[0] - aE[0], cE[1] - aE[1], cE[2] - aE[2]];
      const n = normalize3(cross3(e1, e2));
      const nDotL = Math.max(dot3(n, sunEcef), 0.0);
      const hemi = clampNum(0.5 + 0.5 * dot3(n, upLocal), 0.0, 1.0);
      const lightR = 0.42 + 0.24 * hemi + 0.48 * nDotL;
      const lightG = 0.44 + 0.26 * hemi + 0.45 * nDotL;
      const lightB = 0.48 + 0.30 * hemi + 0.39 * nDotL;
      const shaded: Rgba = [
        clampNum(base[0] * lightR, 0.0, 1.0),
        clampNum(base[1] * lightG, 0.0, 1.0),
        clampNum(base[2] * lightB, 0.0, 1.0),
        1.0,
      ];
      this.pushSolidTriRte(toRte(aE), toRte(bE), toRte(cE), shaded);
    };
    const pushFacetQuad = (aE: Vec3, bE: Vec3, cE: Vec3, dE: Vec3, base: Rgba) => {
      pushFacetTri(aE, bE, cE, base);
      pushFacetTri(aE, cE, dE, base);
    };

    // Nominal civil narrow-body jet geometry (metres, centered near mid-fuselage)
    const nose = pt(19.0, 0.0, -0.4);
    const fTopF = pt(12.5, 0.0, 2.2);
    const fTopA = pt(-12.5, 0.0, 2.1);
    const fBotF = pt(12.0, 0.0, -2.0);
    const fBotA = pt(-12.0, 0.0, -1.6);
    const fLeftF = pt(12.5, -2.1, 0.1);
    const fLeftA = pt(-12.5, -1.8, 0.2);
    const fRightF = pt(12.5, 2.1, 0.1);
    const fRightA = pt(-12.5, 1.8, 0.2);
    const tailTip = pt(-19.5, 0.0, 0.8);

    // Swept main wings (38 m wingspan)
    const wRootF_L = pt(5.5, -1.8, -0.4);
    const wRootA_L = pt(-3.5, -1.8, -0.4);
    const wTipF_L = pt(-5.8, -19.0, 1.0);
    const wTipA_L = pt(-9.5, -19.0, 1.0);

    const wRootF_R = pt(5.5, 1.8, -0.4);
    const wRootA_R = pt(-3.5, 1.8, -0.4);
    const wTipF_R = pt(-5.8, 19.0, 1.0);
    const wTipA_R = pt(-9.5, 19.0, 1.0);

    // Horizontal stabilizer
    const hRootF_L = pt(-13.2, -1.0, 0.8);
    const hRootA_L = pt(-18.0, -1.0, 0.8);
    const hTipF_L = pt(-16.8, -7.2, 1.2);
    const hTipA_L = pt(-19.2, -7.2, 1.2);

    const hRootF_R = pt(-13.2, 1.0, 0.8);
    const hRootA_R = pt(-18.0, 1.0, 0.8);
    const hTipF_R = pt(-16.8, 7.2, 1.2);
    const hTipA_R = pt(-19.2, 7.2, 1.2);

    // Tall vertical tailfin
    const finBaseF = pt(-10.5, 0.0, 2.0);
    const finBaseA = pt(-18.2, 0.0, 1.1);
    const finTopF = pt(-16.2, 0.0, 9.6);
    const finTopA = pt(-19.6, 0.0, 9.2);

    // Shaded solid surfaces (skipped for rejected ghosts so ghosts stay unmistakably ghostly!)
    if (!isRejectedGhost) {
      const topCol: Rgba = [0.95, 0.97, 1.0, 1.0];
      const sideCol: Rgba = [0.84, 0.89, 0.95, 1.0];
      const bellyCol: Rgba = [0.58, 0.65, 0.74, 1.0];
      const wingTopCol: Rgba = [0.90, 0.94, 0.98, 1.0];
      const tailAccentCol: Rgba = [
        alertCol[0] * 0.85 + 0.15,
        alertCol[1] * 0.85 + 0.15,
        alertCol[2] * 0.85 + 0.15,
        1.0,
      ];
      const cockpitCol: Rgba = [0.06, 0.14, 0.24, 1.0];

      // Nose cone facets (directional sun-shaded)
      pushFacetTri(nose, fTopF, fLeftF, topCol);
      pushFacetTri(nose, fRightF, fTopF, topCol);
      pushFacetTri(nose, fBotF, fRightF, bellyCol);
      pushFacetTri(nose, fLeftF, fBotF, bellyCol);

      // Cockpit windshield strip
      pushFacetQuad(
        pt(15.2, -1.25, 0.95),
        pt(15.2, 1.25, 0.95),
        pt(13.0, 1.55, 1.75),
        pt(13.0, -1.55, 1.75),
        cockpitCol,
      );

      // Fuselage upper & lower hull quads
      pushFacetQuad(fTopF, fRightF, fRightA, fTopA, topCol);
      pushFacetQuad(fLeftF, fTopF, fTopA, fLeftA, sideCol);
      pushFacetQuad(fRightF, fBotF, fBotA, fRightA, bellyCol);
      pushFacetQuad(fBotF, fLeftF, fLeftA, fBotA, bellyCol);

      // Tail cone facets
      pushFacetTri(tailTip, fLeftA, fTopA, topCol);
      pushFacetTri(tailTip, fTopA, fRightA, topCol);
      pushFacetTri(tailTip, fRightA, fBotA, bellyCol);
      pushFacetTri(tailTip, fBotA, fLeftA, bellyCol);

      // Swept main wings + upward swept winglets
      pushFacetQuad(wRootF_L, wTipF_L, wTipA_L, wRootA_L, wingTopCol);
      pushFacetQuad(wRootF_R, wRootA_R, wTipA_R, wTipF_R, wingTopCol);
      pushFacetQuad(pt(-5.8, -19.0, 1.0), pt(-7.6, -19.5, 3.2), pt(-10.0, -19.5, 3.0), pt(-9.5, -19.0, 1.0), tailAccentCol);
      pushFacetQuad(pt(-5.8, 19.0, 1.0), pt(-9.5, 19.0, 1.0), pt(-10.0, 19.5, 3.0), pt(-7.6, 19.5, 3.2), tailAccentCol);

      // Horizontal stabilizers
      pushFacetQuad(hRootF_L, hTipF_L, hTipA_L, hRootA_L, sideCol);
      pushFacetQuad(hRootF_R, hRootA_R, hTipA_R, hTipF_R, sideCol);

      // Vertical tailfin painted with alert-tier livery color
      pushFacetQuad(finBaseF, finTopF, finTopA, finBaseA, tailAccentCol);

      // Twin underwing turbofan engines (hot thermal cores glow bright in LWIR EFVS!)
      const engCol: Rgba = [1.0, 0.72, 0.30, 0.98];
      const engR = isOwnship ? 0.85 : 2.3 * scale;
      this.pushBeaconEcef(pt(1.5, -6.4, -1.6), engR, view, engCol, false, 0.0, simTimeS);
      this.pushBeaconEcef(pt(1.5, 6.4, -1.6), engR, view, engCol, false, 0.0, simTimeS);

      // Aviation Navigation & Anti-Collision Strobe Lights (emissive radial halos)
      const beaconR = isOwnship ? 0.85 : Math.max(3.0, 2.5 * scale);
      this.pushBeaconEcef(wTipF_L, beaconR, view, [1.0, 0.14, 0.14, 0.98], false, 0.0, simTimeS);
      this.pushBeaconEcef(wTipF_R, beaconR, view, [0.14, 1.0, 0.44, 0.98], false, 0.0, simTimeS);
      this.pushBeaconEcef(wTipA_L, beaconR * 1.25, view, [1.0, 1.0, 1.0, 0.98], true, 0.0, simTimeS);
      this.pushBeaconEcef(wTipA_R, beaconR * 1.25, view, [1.0, 1.0, 1.0, 0.98], true, 0.25, simTimeS);
      this.pushBeaconEcef(pt(17.5, 0.0, -1.0), beaconR * 1.15, view, [1.0, 0.97, 0.85, 0.96], false, 0.0, simTimeS);
      this.pushBeaconEcef(pt(2.0, 0.0, 2.6), beaconR * 1.05, view, [1.0, 0.18, 0.18, 0.98], true, 1.2, simTimeS);
    }

    // Crisp wireframe outline around silhouette (dashed crimson for rejected ghost, sharp alert tint for real traffic)
    const wireCol: Rgba = isRejectedGhost
      ? [1.0, 0.22, 0.22, 0.92]
      : [alertCol[0], alertCol[1], alertCol[2], 0.92];
    const dash = isRejectedGhost ? 14.0 : 0.0;
    this.pushLineEcef(nose, tailTip, pCam, wireCol, dash);
    this.pushLineEcef(wRootF_L, wTipF_L, pCam, wireCol, dash);
    this.pushLineEcef(wTipF_L, wTipA_L, pCam, wireCol, dash);
    this.pushLineEcef(wTipA_L, wRootA_L, pCam, wireCol, dash);
    this.pushLineEcef(wRootF_R, wTipF_R, pCam, wireCol, dash);
    this.pushLineEcef(wTipF_R, wTipA_R, pCam, wireCol, dash);
    this.pushLineEcef(wTipA_R, wRootA_R, pCam, wireCol, dash);
    this.pushLineEcef(hTipF_L, hTipF_R, pCam, wireCol, dash);
    this.pushLineEcef(finBaseF, finTopF, pCam, wireCol, dash);
    this.pushLineEcef(finTopF, finTopA, pCam, wireCol, dash);
    this.pushLineEcef(finTopA, finBaseA, pCam, wireCol, dash);
  }

  private addIntrudersAndMeshDaa(state: CockpitFrameState, view: ActiveCameraView): void {
    const pCam = view.pEcefCam;
    const nVec: Vec3 = [view.rEcefNed[0], view.rEcefNed[3], view.rEcefNed[6]];
    const eVec: Vec3 = [view.rEcefNed[1], view.rEcefNed[4], view.rEcefNed[7]];
    const dVec: Vec3 = [view.rEcefNed[2], view.rEcefNed[5], view.rEcefNed[8]];
    const upLocal: Vec3 = [-dVec[0], -dVec[1], -dVec[2]];

    for (const tr of state.intruders) {
      const pT = tr.posEcef;
      const isRejectedGhost = Boolean(tr.isGhost && tr.ghostRejected);

      const col: Rgba = isRejectedGhost
        ? [1.0, 0.22, 0.22, 0.90]
        : tr.alertTier >= 3
          ? [1.0, 0.20, 0.20, 0.98]
          : tr.alertTier === 2
            ? [1.0, 0.74, 0.14, 0.95]
            : tr.meshVerified
              ? [0.16, 1.0, 0.54, 0.95]
              : [0.35, 0.90, 1.0, 0.90];

      // Orient 3D aircraft along true velocity vector (or head-on toward ownship if stationary)
      let rawFwd: Vec3;
      if (tr.velEcef && Math.hypot(tr.velEcef[0], tr.velEcef[1], tr.velEcef[2]) > 2.0) {
        rawFwd = normalize3(tr.velEcef);
      } else {
        rawFwd = normalize3([pCam[0] - pT[0], pCam[1] - pT[1], pCam[2] - pT[2]]);
      }
      // Project forward perpendicular to local up so wings remain wings-level unless climbing/descending gently
      const rightAxis = normalize3(cross3(rawFwd, upLocal));
      const upAxis = normalize3(cross3(rightAxis, rawFwd));

      // Gentle perspective visibility boost at multi-km range so approaching jet silhouette is unmistakable
      const visualScale = Math.max(1.0, Math.min(3.6, tr.rangeM / 620.0));
      this.addSolid3DAircraft(
        pT,
        rawFwd,
        rightAxis,
        upAxis,
        visualScale,
        col,
        isRejectedGhost,
        view,
        false,
        state.simTimeS,
      );

      // 2. Well-Clear Volume (WCV) cylinder / ring (skipped for rejected ghost)
      if (!isRejectedGhost) {
        const rWcv = tr.wcvRadiusM ?? 160.0;
        const hWcv = 65.0;
        const ringSegs = 16;
        const wcvCol: Rgba = [col[0], col[1], col[2], 0.42];
        for (let k = 0; k < ringSegs; k++) {
          const a0 = (k / ringSegs) * Math.PI * 2.0;
          const a1 = ((k + 1) / ringSegs) * Math.PI * 2.0;
          const c0 = Math.cos(a0) * rWcv;
          const s0 = Math.sin(a0) * rWcv;
          const c1 = Math.cos(a1) * rWcv;
          const s1 = Math.sin(a1) * rWcv;
          const u0: Vec3 = [
            pT[0] + nVec[0] * c0 + eVec[0] * s0 - dVec[0] * hWcv,
            pT[1] + nVec[1] * c0 + eVec[1] * s0 - dVec[1] * hWcv,
            pT[2] + nVec[2] * c0 + eVec[2] * s0 - dVec[2] * hWcv,
          ];
          const u1: Vec3 = [
            pT[0] + nVec[0] * c1 + eVec[0] * s1 - dVec[0] * hWcv,
            pT[1] + nVec[1] * c1 + eVec[1] * s1 - dVec[1] * hWcv,
            pT[2] + nVec[2] * c1 + eVec[2] * s1 - dVec[2] * hWcv,
          ];
          const b0: Vec3 = [u0[0] + dVec[0] * 2 * hWcv, u0[1] + dVec[1] * 2 * hWcv, u0[2] + dVec[2] * 2 * hWcv];
          const b1: Vec3 = [u1[0] + dVec[0] * 2 * hWcv, u1[1] + dVec[1] * 2 * hWcv, u1[2] + dVec[2] * 2 * hWcv];
          this.pushLineEcef(u0, u1, pCam, wcvCol);
          this.pushLineEcef(b0, b1, pCam, wcvCol);
          if (k % 4 === 0) {
            this.pushLineEcef(u0, b0, pCam, wcvCol);
          }
        }
      }

      // 3. XVS-MeshDAA 3-ship sight-lines (verified green solid / rejected ghost dashed red)
      if (tr.peerOriginsEcef && tr.peerOriginsEcef.length > 0) {
        const rayCol: Rgba = isRejectedGhost
          ? [1.0, 0.22, 0.22, 0.78]
          : tr.meshVerified
            ? [0.18, 1.0, 0.54, 0.72]
            : [0.98, 0.80, 0.24, 0.56];
        const rayDash = isRejectedGhost ? 70.0 : 0.0;
        for (const peerEcef of tr.peerOriginsEcef) {
          this.pushLineEcef(peerEcef, pT, pCam, rayCol, rayDash);
        }
      }
    }
  }

  resetTrails(): void {
    this.ownshipTrail.length = 0;
    this.intruderTrail.length = 0;
    this.lastTrailSimTimeS = -1.0;
  }

  private recordFlightTrails(state: CockpitFrameState, view: ActiveCameraView): void {
    if (this.lastTrailSimTimeS >= 0 && state.simTimeS + 1e-4 < this.lastTrailSimTimeS) {
      this.resetTrails();
    }
    const shouldSample =
      this.ownshipTrail.length === 0 || state.simTimeS - this.lastTrailSimTimeS >= 0.25 - 1e-6;
    if (!shouldSample) return;

    this.lastTrailSimTimeS = state.simTimeS;
    this.ownshipTrail.push([view.pEcefBody[0], view.pEcefBody[1], view.pEcefBody[2]]);
    if (this.ownshipTrail.length > MAX_TRAIL_POINTS) {
      this.ownshipTrail.shift();
    }

    const primaryIntr = state.intruders.find((tr) => !tr.isGhost);
    if (primaryIntr) {
      this.intruderTrail.push([
        primaryIntr.posEcef[0],
        primaryIntr.posEcef[1],
        primaryIntr.posEcef[2],
      ]);
      if (this.intruderTrail.length > MAX_TRAIL_POINTS) {
        this.intruderTrail.shift();
      }
    }
  }

  /**
   * External 3D Chase / Orbit Ownship: sculpted solid twin-jet (winglets, twin underwing nacelles,
   * OpenXVS livery tailfin, glowing nav/strobe beacons) + 6-Camera 360-deg Sensor Frustum Cones
   * (`F, R, B, L, U, D` — cyan/emerald when healthy, pulsing bright crimson when isolated/faulted).
   */
  private addOwnship3DAndSensorFrustums(state: CockpitFrameState, view: ActiveCameraView): void {
    const pCam = view.pEcefCam;
    const pBody = view.pEcefBody;
    const rBody = view.rEcefBody;
    // BODY FRD columns of R_ecef_body: col0 = +Fwd, col1 = +Right, -col2 = +Up
    const fwd: Vec3 = [rBody[0], rBody[3], rBody[6]];
    const right: Vec3 = [rBody[1], rBody[4], rBody[7]];
    const up: Vec3 = [-rBody[2], -rBody[5], -rBody[8]];

    // 1. Solid 3D Ownship Twin-Jet (OpenXVS cyan/emerald livery accent)
    const ownAccent: Rgba =
      state.simplexMode === 0
        ? [1.0, 0.22, 0.22, 1.0]
        : state.simplexMode <= 2
          ? [1.0, 0.72, 0.14, 1.0]
          : [0.16, 0.88, 0.96, 1.0];
    this.addSolid3DAircraft(
      pBody,
      fwd,
      right,
      up,
      1.0,
      ownAccent,
      false,
      view,
      true,
      state.simTimeS,
    );

    // Extra sculpted upward winglets + twin nacelle barrels for unmistakable 3D close-up fidelity
    const pt = (xF: number, yR: number, zU: number): Vec3 => [
      pBody[0] + fwd[0] * xF + right[0] * yR + up[0] * zU,
      pBody[1] + fwd[1] * xF + right[1] * yR + up[1] * zU,
      pBody[2] + fwd[2] * xF + right[2] * yR + up[2] * zU,
    ];
    // Port & starboard swept upward winglets
    this.pushQuadEcef(
      pt(-5.8, -19.0, 1.0),
      pt(-7.8, -19.6, 3.6),
      pt(-10.2, -19.6, 3.4),
      pt(-9.5, -19.0, 1.0),
      pCam,
      ownAccent,
    );
    this.pushQuadEcef(
      pt(-5.8, 19.0, 1.0),
      pt(-9.5, 19.0, 1.0),
      pt(-10.2, 19.6, 3.4),
      pt(-7.8, 19.6, 3.6),
      pCam,
      ownAccent,
    );
    // Twin underwing turbofan nacelle cowlings
    const nacelleCol: Rgba = [0.62, 0.70, 0.78, 1.0];
    for (const side of [-1, 1]) {
      const yIn = side * 5.2;
      const yOut = side * 7.6;
      this.pushQuadEcef(
        pt(4.2, yIn, -0.8),
        pt(4.2, yOut, -0.8),
        pt(-0.8, yOut, -0.8),
        pt(-0.8, yIn, -0.8),
        pCam,
        nacelleCol,
      );
      this.pushQuadEcef(
        pt(4.2, yIn, -2.5),
        pt(-0.8, yIn, -2.5),
        pt(-0.8, yOut, -2.5),
        pt(4.2, yOut, -2.5),
        pCam,
        [0.36, 0.44, 0.52, 1.0],
      );
    }

    // 2. 6-Camera 360-deg Hemispherical Sensor Frustum Cones strictly ordered F, R, B, L, U, D
    const frustumDepthM = 36.0;
    const halfSpanM = 24.0;
    for (let i = 0; i < XVS6_CUBE_V1_CAMERAS.length; i++) {
      const spec = XVS6_CUBE_V1_CAMERAS[i];
      const healthy = ((state.camHealthMask >> i) & 1) !== 0;

      const origBody = mat3MulVec(rBody, spec.leverArmM);
      const origin: Vec3 = [
        pBody[0] + origBody[0],
        pBody[1] + origBody[1],
        pBody[2] + origBody[2],
      ];

      // 4 far corners of camera pyramid in OpenCV CAM frame -> BODY -> ECEF
      const cornersCam: readonly Vec3[] = [
        [-halfSpanM, -halfSpanM, frustumDepthM],
        [halfSpanM, -halfSpanM, frustumDepthM],
        [halfSpanM, halfSpanM, frustumDepthM],
        [-halfSpanM, halfSpanM, frustumDepthM],
      ];
      const cornersEcef = cornersCam.map((c) => {
        const cBody = mat3MulVec(spec.rBodyCam, c);
        const cEcef = mat3MulVec(rBody, cBody);
        return [origin[0] + cEcef[0], origin[1] + cEcef[1], origin[2] + cEcef[2]] as Vec3;
      });

      const edgeCol: Rgba = healthy
        ? [0.18, 0.92, 0.78, 0.42]
        : [1.0, 0.16, 0.18, 0.96];
      const faceCol: Rgba = healthy
        ? [0.14, 0.85, 0.80, 0.045]
        : [1.0, 0.16, 0.18, 0.14];
      const pulseMode = healthy ? 0.0 : -1.0;

      // Wireframe frustum pyramid edges
      for (let k = 0; k < 4; k++) {
        this.pushLineEcef(origin, cornersEcef[k], pCam, edgeCol, healthy ? 0.0 : 4.5);
        this.pushLineEcef(cornersEcef[k], cornersEcef[(k + 1) % 4], pCam, edgeCol, 0.0);
      }
      // Subtle translucent pyramid faces (pulsing crimson when faulted, e.g. Cam-R in M1!)
      const oRte: Vec3 = [origin[0] - pCam[0], origin[1] - pCam[1], origin[2] - pCam[2]];
      for (let k = 0; k < 4; k++) {
        const c0 = cornersEcef[k];
        const c1 = cornersEcef[(k + 1) % 4];
        this.pushTranslucentTriRte(
          oRte,
          [c0[0] - pCam[0], c0[1] - pCam[1], c0[2] - pCam[2]],
          [c1[0] - pCam[0], c1[1] - pCam[1], c1[2] - pCam[2]],
          faceCol,
          pulseMode,
          i * 0.7,
        );
      }
    }

    // 3. DR-14 7th-Sector 140-deg Rear Equidistant Fisheye Passive Optical Guard Cone
    // (1x Sony IMX296LLR + pingRX Pro EAR99, 102.0 g / 2.7 W — 222-deg fwd + 138-deg rear = 360-deg zero blind spot)
    const rearOriginBody = mat3MulVec(rBody, [-16.2, 0.0, -0.6]);
    const rearOrigin: Vec3 = [
      pBody[0] + rearOriginBody[0],
      pBody[1] + rearOriginBody[1],
      pBody[2] + rearOriginBody[2],
    ];
    const rearOriginRte: Vec3 = [
      rearOrigin[0] - pCam[0],
      rearOrigin[1] - pCam[1],
      rearOrigin[2] - pCam[2],
    ];
    const rearRadiusM = 36.0;
    const halfH = 70.0 * DEG_TO_RAD; // +-70 deg horizontal = 140 deg FOV
    const halfV = 52.0 * DEG_TO_RAD; // +-52 deg vertical
    const octSides = 8;
    const rearRimEcef: Vec3[] = [];
    for (let k = 0; k < octSides; k++) {
      const phi = (k / octSides) * Math.PI * 2.0;
      const alpha = halfH * Math.cos(phi);
      const beta = halfV * Math.sin(phi);
      const ang = Math.min(72.0 * DEG_TO_RAD, Math.hypot(alpha, beta));
      const sinc = ang > 1e-6 ? Math.sin(ang) / ang : 1.0;
      // BODY FRD: -X aft, +Y right, +Z down
      const dirBody: Vec3 = [-Math.cos(ang), sinc * alpha, sinc * beta];
      const dirEcef = mat3MulVec(rBody, dirBody);
      rearRimEcef.push([
        rearOrigin[0] + dirEcef[0] * rearRadiusM,
        rearOrigin[1] + dirEcef[1] * rearRadiusM,
        rearOrigin[2] + dirEcef[2] * rearRadiusM,
      ]);
    }
    const rearEdgeCol: Rgba = [0.96, 0.78, 0.28, 0.56];
    const rearFillCol: Rgba = [0.94, 0.76, 0.24, 0.035];
    for (let k = 0; k < octSides; k++) {
      const r0 = rearRimEcef[k];
      const r1 = rearRimEcef[(k + 1) % octSides];
      this.pushLineEcef(rearOrigin, r0, pCam, rearEdgeCol, 0.0);
      this.pushLineEcef(r0, r1, pCam, rearEdgeCol, 0.0);
      this.pushTranslucentTriRte(
        rearOriginRte,
        [r0[0] - pCam[0], r0[1] - pCam[1], r0[2] - pCam[2]],
        [r1[0] - pCam[0], r1[1] - pCam[1], r1[2] - pCam[2]],
        rearFillCol,
        0.0,
        0.0,
      );
    }
  }

  private addFlightTrailRibbons(view: ActiveCameraView): void {
    const pCam = view.pEcefCam;
    const upLocal: Vec3 = [-view.rEcefNed[2], -view.rEcefNed[5], -view.rEcefNed[8]];

    const drawRibbon = (trail: readonly Vec3[], rgb: readonly [number, number, number], halfW: number) => {
      const n = trail.length;
      if (n < 2) return;
      for (let i = 0; i < n - 1; i++) {
        const p0 = trail[i];
        const p1 = trail[i + 1];
        const segLen = Math.hypot(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]);
        if (segLen < 0.5 || segLen > 2500.0) continue;

        const t = (i + 1) / n; // fades gracefully toward older tail
        const alphaLine = 0.14 + 0.76 * t;
        const alphaFill = 0.05 + 0.30 * t;
        const dir = normalize3([p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]]);
        const side = normalize3(cross3(dir, upLocal));
        const hw = halfW * (0.35 + 0.65 * t);

        const a0: Vec3 = [p0[0] - side[0] * hw, p0[1] - side[1] * hw, p0[2] - side[2] * hw];
        const b0: Vec3 = [p0[0] + side[0] * hw, p0[1] + side[1] * hw, p0[2] + side[2] * hw];
        const b1: Vec3 = [p1[0] + side[0] * hw, p1[1] + side[1] * hw, p1[2] + side[2] * hw];
        const a1: Vec3 = [p1[0] - side[0] * hw, p1[1] - side[1] * hw, p1[2] - side[2] * hw];

        this.pushTranslucentQuadEcef(a0, b0, b1, a1, pCam, [rgb[0], rgb[1], rgb[2], alphaFill]);
        this.pushLineEcef(p0, p1, pCam, [rgb[0], rgb[1], rgb[2], alphaLine], 0.0);
      }
    };

    drawRibbon(this.ownshipTrail, [0.24, 0.90, 1.0], 5.5);
    drawRibbon(this.intruderTrail, [1.0, 0.42, 0.18], 4.8);
  }

  /**
   * Renders the 3 live glowing 3D mathematical manifolds on `translucentVolPipeline`
   * (strictly <= 3,500 combined manifold vertices):
   *   1. XVS-ActiveRange Collapsing 3D CRLB/Conformal Range Cone
   *   2. SC3BF Shifted Collision-Cone Barrier Tube
   *   3. XVS-ForcedLand 3D Wind-Warped Trochoidal Glide Funnel
   */
  private addSafetyManifolds(
    state: CockpitFrameState,
    view: ActiveCameraView,
    pack: TerrainPackData,
  ): void {
    const pCam = view.pEcefCam;
    const upLocal: Vec3 = [-view.rEcefNed[2], -view.rEcefNed[5], -view.rEcefNed[8]];
    const toggles = state.manifoldToggles;
    const overlay = state.activeRangeOverlay;
    const primaryIntr = state.intruders.find((tr) => !tr.isGhost && !tr.ghostRejected);

    let manifoldVertsUsed = 0;
    const appendBounded = (geom: ManifoldGeometryOutput) => {
      const rem = MAX_MANIFOLD_VERTICES - manifoldVertsUsed;
      const take = Math.min(rem, geom.vertexCount);
      if (take > 0) {
        this.appendTranslucentPacked(geom.vertices, take);
        manifoldVertsUsed += take;
      }
    };

    // 1. XVS-ActiveRange Collapsing 3D CRLB/Conformal Range Cone
    const showCone = toggles?.activeRangeCone ?? Boolean(overlay);
    if (showCone && (overlay || primaryIntr)) {
      const ownEcef = overlay?.ownshipEcef ?? view.pEcefBody;
      const originRte: Vec3 = [
        ownEcef[0] - pCam[0],
        ownEcef[1] - pCam[1],
        ownEcef[2] - pCam[2],
      ];
      let bearingDir: Vec3;
      if (overlay?.bearingEcef) {
        bearingDir = normalize3(overlay.bearingEcef);
      } else {
        const tgt = overlay?.targetPosEcef ?? primaryIntr?.posEcef;
        bearingDir = tgt
          ? normalize3([tgt[0] - ownEcef[0], tgt[1] - ownEcef[1], tgt[2] - ownEcef[2]])
          : [view.rEcefBody[0], view.rEcefBody[3], view.rEcefBody[6]];
      }
      const rHatM = overlay?.rHatM ?? primaryIntr?.rangeM ?? 1200.0;
      const sigmaRM = overlay?.sigmaRM ?? Math.max(85.0, rHatM * 0.22);
      const rSafeLowerM = overlay?.rSafeLowerM ?? Math.max(40.0, rHatM - 1.65 * sigmaRM);

      const cone = buildActiveRangeConeGeometry({
        originRte,
        bearingDir,
        upRef: upLocal,
        rHatM,
        sigmaRM,
        rSafeLowerM,
      });
      appendBounded(cone);
    }

    // 2. SC3BF Shifted Collision-Cone Barrier Tube around intruder
    const showSc3bf = toggles?.sc3bfTube ?? Boolean(overlay);
    const tubeTarget = overlay?.targetPosEcef ?? primaryIntr?.posEcef;
    if (showSc3bf && tubeTarget) {
      const centerRte: Vec3 = [
        tubeTarget[0] - pCam[0],
        tubeTarget[1] - pCam[1],
        tubeTarget[2] - pCam[2],
      ];
      const ownEcef = overlay?.ownshipEcef ?? view.pEcefBody;
      const axisDir = normalize3([
        ownEcef[0] - tubeTarget[0],
        ownEcef[1] - tubeTarget[1],
        ownEcef[2] - tubeTarget[2],
      ]);
      const radiusM = overlay?.sc3bfRadiusM ?? primaryIntr?.wcvRadiusM ?? 152.4;
      const safe = (overlay?.minSc3bfMargin ?? 1.0) >= 0.0;

      const tube = buildSc3bfBarrierTubeGeometry({
        centerRte,
        axisDir,
        upRef: upLocal,
        radiusM,
        halfLengthM: Math.max(160.0, radiusM * 1.45),
        safe,
      });
      appendBounded(tube);
    }

    // 3. XVS-ForcedLand 3D Wind-Warped Trochoidal Glide Funnel to KSFO 28L
    const fl = state.forcedLandFunnel;
    const showFunnel = toggles?.forcedLandFunnel ?? Boolean(fl?.active);
    if (showFunnel) {
      const originRte: Vec3 = [
        view.pEcefBody[0] - pCam[0],
        view.pEcefBody[1] - pCam[1],
        view.pEcefBody[2] - pCam[2],
      ];
      let targetEcef = fl?.targetEcef;
      if (!targetEcef) {
        const rwy28l = pack.runways.find((r) => r.id === "28L");
        const thrLat = (rwy28l?.thresholdLatDeg ?? pack.ksfo28lThrLatDeg) * DEG_TO_RAD;
        const thrLon = (rwy28l?.thresholdLonDeg ?? pack.ksfo28lThrLonDeg) * DEG_TO_RAD;
        const thrH = (rwy28l?.thresholdWgs84HM ?? pack.ksfo28lThrWgs84HM) + 12.0;
        targetEcef = geodeticToEcef(thrLat, thrLon, thrH);
      }
      const targetRte: Vec3 = [
        targetEcef[0] - pCam[0],
        targetEcef[1] - pCam[1],
        targetEcef[2] - pCam[2],
      ];
      const waypointsRte = fl?.waypoints?.map((wp) => ({
        posRte: [
          wp.posEcef[0] - pCam[0],
          wp.posEcef[1] - pCam[1],
          wp.posEcef[2] - pCam[2],
        ] as Vec3,
        halfWidthM: wp.halfWidthM,
        halfHeightM: wp.halfHeightM,
      }));
      const funnel = buildForcedLandFunnelGeometry({
        originRte,
        targetRte,
        upRef: upLocal,
        crosswindKt: fl?.crosswindKt ?? 35.0,
        waypointsRte,
      });
      appendBounded(funnel);
    }
  }

  encode(state: CockpitFrameState, view: ActiveCameraView, pack: TerrainPackData): void {
    this.lineVertexCount = 0;
    this.solidVertexCount = 0;
    this.translucentVertexCount = 0;
    this.haloVertexCount = 0;

    this.recordFlightTrails(state, view);

    if (view.cameraMode === "cockpit") {
      this.addAirframeWireframe(view);
    } else {
      this.addOwnship3DAndSensorFrustums(state, view);
      this.addFlightTrailRibbons(view);
    }
    this.addRunwaysAndHitsTunnel(state, view, pack);
    this.addIntrudersAndMeshDaa(state, view);
    this.addSafetyManifolds(state, view, pack);

    if (
      this.solidVertexCount === 0 &&
      this.translucentVertexCount === 0 &&
      this.haloVertexCount === 0 &&
      this.lineVertexCount === 0
    ) {
      return;
    }

    // Standard mode UBO (modeParams.x = 0: solid aircraft, runways, translucent manifolds & vector lines)
    this.uniformData.set(view.viewProj, 0);
    this.uniformData[16] = state.simTimeS;
    this.uniformData[17] = state.efvsEnabled ? 1.0 : 0.0;
    this.uniformData[18] = state.fogContrast;
    this.uniformData[19] = 1.0;
    this.uniformData[20] = 0.0;
    this.uniformData[21] = 3.5;
    this.uniformData[22] = 0.04;
    this.uniformData[23] = 0.0;
    this.rhi.writeBuffer(this.ubo, this.uniformData);

    // 1. Opaque depth-writing solid 3D aircraft, asphalt runways & markings
    if (this.solidVertexCount > 0) {
      this.rhi.writeBuffer(
        this.solidVb,
        this.solidScratch.subarray(0, this.solidVertexCount * FLOATS_PER_VERTEX),
      );
      this.rhi.draw({
        pipeline: this.solidTriPipeline,
        uniformBuffers: [{ binding: 0, buffer: this.ubo }],
        vertexBuffers: [this.solidVb],
        vertexCount: this.solidVertexCount,
      });
    }

    // 2. Depth-tested, non-depth-writing glowing 3D mathematical manifolds, sensor frustums & trail ribbons
    if (this.translucentVertexCount > 0) {
      this.rhi.writeBuffer(
        this.translucentVb,
        this.translucentScratch.subarray(0, this.translucentVertexCount * FLOATS_PER_VERTEX),
      );
      this.rhi.draw({
        pipeline: this.translucentVolPipeline,
        uniformBuffers: [{ binding: 0, buffer: this.ubo }],
        vertexBuffers: [this.translucentVb],
        vertexCount: this.translucentVertexCount,
      });
    }

    // 3. Emissive radial-halo beacons (4-light PAPI, ALSF-2, runway edge lights, aircraft nav & strobes)
    if (this.haloVertexCount > 0) {
      this.haloUniformData.set(view.viewProj, 0);
      this.haloUniformData[16] = state.simTimeS;
      this.haloUniformData[17] = state.efvsEnabled ? 1.0 : 0.0;
      this.haloUniformData[18] = state.fogContrast;
      this.haloUniformData[19] = 1.0;
      // modeParams.x = 2 (emissive radial halo), sigma = 4.2, discard < 0.04
      this.haloUniformData[20] = 2.0;
      this.haloUniformData[21] = 4.2;
      this.haloUniformData[22] = 0.04;
      this.haloUniformData[23] = 0.0;
      this.rhi.writeBuffer(this.haloUbo, this.haloUniformData);

      this.rhi.writeBuffer(
        this.haloVb,
        this.haloScratch.subarray(0, this.haloVertexCount * FLOATS_PER_VERTEX),
      );
      this.rhi.draw({
        pipeline: this.translucentVolPipeline,
        uniformBuffers: [{ binding: 0, buffer: this.haloUbo }],
        vertexBuffers: [this.haloVb],
        vertexCount: this.haloVertexCount,
      });
    }

    // 4. Crisp vector wireframes, HITS gates, WCV rings & MeshDAA rays over solids
    if (this.lineVertexCount > 0) {
      this.rhi.writeBuffer(
        this.lineVb,
        this.lineScratch.subarray(0, this.lineVertexCount * FLOATS_PER_VERTEX),
      );
      this.rhi.draw({
        pipeline: this.linePipeline,
        uniformBuffers: [{ binding: 0, buffer: this.ubo }],
        vertexBuffers: [this.lineVb],
        vertexCount: this.lineVertexCount,
      });
    }
  }

  dispose(): void {
    this.rhi.destroyBuffer(this.lineVb);
    this.rhi.destroyBuffer(this.solidVb);
    this.rhi.destroyBuffer(this.translucentVb);
    this.rhi.destroyBuffer(this.haloVb);
    this.rhi.destroyBuffer(this.ubo);
    this.rhi.destroyBuffer(this.haloUbo);
  }
}
