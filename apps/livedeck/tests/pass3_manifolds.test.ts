import { describe, expect, it } from "vitest";
import {
  MAX_MANIFOLD_VERTICES,
  buildActiveRangeConeGeometry,
  buildForcedLandFunnelGeometry,
  buildSc3bfBarrierTubeGeometry,
  computePapiLightStates,
} from "../src/passes/pass3_airframe_traffic.ts";
import {
  SPZ_HEADER_BYTES,
  SPZ_MAGIC_U32,
  SPZ_MAX_CHROMA_SPREAD,
  SPZ_SH_C0,
  SPZ_VERSION,
  clampSpzChromaRgb,
  parseSpzV2Raw,
} from "../src/passes/pass2b_spz_world.ts";

function rec709Luma(rgb: readonly [number, number, number]): number {
  return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
}

function encodeSingleRainbowSpz(rgb: readonly [number, number, number]): Uint8Array {
  const out = new Uint8Array(SPZ_HEADER_BYTES + 19);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, SPZ_MAGIC_U32, true);
  dv.setUint32(4, SPZ_VERSION, true);
  dv.setUint32(8, 1, true);
  dv.setUint8(12, 0);
  dv.setUint8(13, 12);
  // Alpha
  out[SPZ_HEADER_BYTES + 9] = 230;
  // SH0 RGB
  for (let c = 0; c < 3; c++) {
    const sh0 = (rgb[c] - 0.5) / SPZ_SH_C0;
    out[SPZ_HEADER_BYTES + 10 + c] = Math.max(
      0,
      Math.min(255, Math.round(sh0 * (0.15 * 255.0) + 127.5)),
    );
  }
  // Log-scales + identity quat
  out[SPZ_HEADER_BYTES + 13] = 176;
  out[SPZ_HEADER_BYTES + 14] = 176;
  out[SPZ_HEADER_BYTES + 15] = 176;
  out[SPZ_HEADER_BYTES + 16] = 128;
  out[SPZ_HEADER_BYTES + 17] = 128;
  out[SPZ_HEADER_BYTES + 18] = 128;
  return out;
}

describe("Pass 3 Split Solid/Translucent 3D Safety Manifolds, PAPI & Soft SPZ", () => {
  it("test_active_range_cone_shrinks_as_sigma_collapses", () => {
    const originRte: [number, number, number] = [0, 0, 0];
    const bearing: [number, number, number] = [1, 0, 0];
    const upRef: [number, number, number] = [0, 0, 1];

    const wide = buildActiveRangeConeGeometry({
      originRte,
      bearingDir: bearing,
      upRef,
      rHatM: 1200,
      sigmaRM: 600,
      rSafeLowerM: 480,
    });

    const locked = buildActiveRangeConeGeometry({
      originRte,
      bearingDir: bearing,
      upRef,
      rHatM: 1200,
      sigmaRM: 40,
      rSafeLowerM: 1120,
    });

    expect(wide.vertexCount).toBeGreaterThan(24);
    expect(locked.vertexCount).toBeGreaterThan(24);
    expect(wide.vertexCount).toBeLessThanOrEqual(MAX_MANIFOLD_VERTICES);

    // Measure actual axial span along bearingDir (+X) from raw emitted vertices (9 floats/vert)
    const measureAxialSpan = (verts: Float32Array, count: number): number => {
      let minX = Number.POSITIVE_INFINITY;
      let maxX = Number.NEGATIVE_INFINITY;
      for (let i = 0; i < count; i++) {
        const x = verts[i * 9 + 0];
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
      }
      return maxX - minX;
    };

    const wideSpan = measureAxialSpan(wide.vertices, wide.vertexCount);
    const lockedSpan = measureAxialSpan(locked.vertices, locked.vertexCount);
    expect(wideSpan).toBeGreaterThan(1500);
    expect(lockedSpan).toBeLessThan(wideSpan * 0.25);
    expect(locked.axialSpanM).toBeLessThan(wide.axialSpanM * 0.25);

    // Color shifts from amber-red (R > G) at high uncertainty to emerald (G > R) at lock
    expect(wide.meanColorRgb[0]).toBeGreaterThan(wide.meanColorRgb[1]);
    expect(locked.meanColorRgb[1]).toBeGreaterThan(locked.meanColorRgb[0]);
  });

  it("test_papi_lights_reflect_glideslope_high_on_low", () => {
    const high = computePapiLightStates(3.9); // > 3.5 deg => 4 white
    expect(high).toHaveLength(4);
    expect(high.filter((l) => l.isWhite)).toHaveLength(4);
    expect(high.filter((l) => !l.isWhite)).toHaveLength(0);

    const onPath = computePapiLightStates(3.0); // 3.0 deg => 2 white, 2 red
    expect(onPath).toHaveLength(4);
    expect(onPath.filter((l) => l.isWhite)).toHaveLength(2);
    expect(onPath.filter((l) => !l.isWhite)).toHaveLength(2);

    const low = computePapiLightStates(2.1); // < 2.5 deg => 4 red
    expect(low).toHaveLength(4);
    expect(low.filter((l) => l.isWhite)).toHaveLength(0);
    expect(low.filter((l) => !l.isWhite)).toHaveLength(4);
  });

  it("test_spz_chroma_clamp_suppresses_rainbow_outliers", () => {
    const outliers: ReadonlyArray<readonly [number, number, number]> = [
      [1.0, 0.0, 0.0],
      [0.0, 0.98, 0.92],
      [0.95, 0.08, 0.94],
      [0.12, 0.95, 0.10],
    ];

    for (const rgb of outliers) {
      const origLuma = rec709Luma(rgb);
      const clamped = clampSpzChromaRgb(rgb[0], rgb[1], rgb[2], SPZ_MAX_CHROMA_SPREAD);
      const spread = Math.max(...clamped) - Math.min(...clamped);
      expect(spread).toBeLessThanOrEqual(SPZ_MAX_CHROMA_SPREAD + 1e-6);
      expect(Math.abs(rec709Luma(clamped) - origLuma)).toBeLessThan(1e-5);
      for (const ch of clamped) {
        expect(ch).toBeGreaterThanOrEqual(0.0);
        expect(ch).toBeLessThanOrEqual(1.0);
      }
    }

    // Natural low-chroma asphalt/concrete stays unchanged
    const neutral = clampSpzChromaRgb(0.42, 0.44, 0.47, SPZ_MAX_CHROMA_SPREAD);
    expect(Math.abs(neutral[0] - 0.42)).toBeLessThan(1e-6);
    expect(Math.abs(neutral[1] - 0.44)).toBeLessThan(1e-6);
    expect(Math.abs(neutral[2] - 0.47)).toBeLessThan(1e-6);

    // End-to-end SPZ decode with chroma clamp enforces <= 0.35 spread
    const raw = encodeSingleRainbowSpz([0.96, 0.06, 0.92]);
    const cloud = parseSpzV2Raw(raw, "rainbow_test.spz", 16, SPZ_MAX_CHROMA_SPREAD);
    const r = cloud.colorsRgb[0];
    const g = cloud.colorsRgb[1];
    const b = cloud.colorsRgb[2];
    expect(Math.max(r, g, b) - Math.min(r, g, b)).toBeLessThanOrEqual(SPZ_MAX_CHROMA_SPREAD + 1e-5);
  });

  it("keeps combined 3-manifold geometry strictly within <= 3,500 vertex budget", () => {
    const cone = buildActiveRangeConeGeometry({
      originRte: [0, 0, 0],
      bearingDir: [1, 0, 0],
      upRef: [0, 0, 1],
      rHatM: 1200,
      sigmaRM: 320,
      rSafeLowerM: 650,
    });
    const tube = buildSc3bfBarrierTubeGeometry({
      centerRte: [1200, 0, 0],
      axisDir: [-1, 0, 0],
      upRef: [0, 0, 1],
      radiusM: 152.4,
      halfLengthM: 220,
      safe: true,
    });
    const funnel = buildForcedLandFunnelGeometry({
      originRte: [0, 0, 0],
      targetRte: [2800, 120, -340],
      upRef: [0, 0, 1],
      crosswindKt: 35,
    });

    const totalVerts = cone.vertexCount + tube.vertexCount + funnel.vertexCount;
    expect(cone.vertexCount).toBeGreaterThan(0);
    expect(tube.vertexCount).toBeGreaterThan(0);
    expect(funnel.vertexCount).toBeGreaterThan(0);
    expect(totalVerts).toBeLessThanOrEqual(MAX_MANIFOLD_VERTICES);
  });
});
