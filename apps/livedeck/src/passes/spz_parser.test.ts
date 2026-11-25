import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import {
  MAX_RENDER_SPLATS,
  SPZ_HEADER_BYTES,
  SPZ_MAGIC_U32,
  SPZ_SH_C0,
  SPZ_VERSION,
  buildKsfo28lPreviewSpzCloud,
  parseSpzV2Gzip,
  parseSpzV2Raw,
} from "./pass2b_spz_world.ts";

interface SyntheticSplat {
  readonly posNed: readonly [number, number, number];
  readonly rgb: readonly [number, number, number];
  readonly alpha: number;
  readonly logScale: readonly [number, number, number];
}

/** Encode synthetic splats using the exact Notebook-3 `write_spz_v2` byte layout (`<IIIBBBB`). */
function encodeSpzV2Raw(
  splats: readonly SyntheticSplat[],
  fractionalBits = 12,
  shDegree = 0,
): Uint8Array {
  const n = splats.length;
  const shExtra = { 0: 0, 1: 3, 2: 8, 3: 15 }[shDegree] ?? 0;
  const totalBytes = SPZ_HEADER_BYTES + n * (9 + 1 + 3 + 3 + 3 + 3 * shExtra);
  const out = new Uint8Array(totalBytes);
  const dv = new DataView(out.buffer);

  dv.setUint32(0, SPZ_MAGIC_U32, true);
  dv.setUint32(4, SPZ_VERSION, true);
  dv.setUint32(8, n, true);
  dv.setUint8(12, shDegree);
  dv.setUint8(13, fractionalBits);
  dv.setUint8(14, 0);
  dv.setUint8(15, 0);

  const posBase = SPZ_HEADER_BYTES;
  const alphaBase = posBase + n * 9;
  const colorBase = alphaBase + n;
  const scaleBase = colorBase + n * 3;
  const rotBase = scaleBase + n * 3;

  const posScale = 1 << fractionalBits;
  for (let i = 0; i < n; i++) {
    const sp = splats[i];
    // 24-bit signed LE positions
    for (let c = 0; c < 3; c++) {
      const q = Math.round(sp.posNed[c] * posScale);
      const u24 = q & 0xffffff;
      const off = posBase + i * 9 + c * 3;
      out[off + 0] = u24 & 0xff;
      out[off + 1] = (u24 >> 8) & 0xff;
      out[off + 2] = (u24 >> 16) & 0xff;
    }
    // Alpha uint8
    out[alphaBase + i] = Math.max(0, Math.min(255, Math.round(sp.alpha * 255.0)));
    // SH0 RGB uint8
    for (let c = 0; c < 3; c++) {
      const sh0 = (sp.rgb[c] - 0.5) / SPZ_SH_C0;
      out[colorBase + i * 3 + c] = Math.max(
        0,
        Math.min(255, Math.round(sh0 * (0.15 * 255.0) + 127.5)),
      );
    }
    // Log-scales uint8
    for (let c = 0; c < 3; c++) {
      out[scaleBase + i * 3 + c] = Math.max(
        0,
        Math.min(255, Math.round((sp.logScale[c] + 10.0) * 16.0)),
      );
    }
    // Identity Hamilton quaternion xyz = (0, 0, 0) -> uint8 128
    out[rotBase + i * 3 + 0] = 128;
    out[rotBase + i * 3 + 1] = 128;
    out[rotBase + i * 3 + 2] = 128;
  }
  return out;
}

describe("Niantic SPZ v2 (NGSP) 3DGS Binary Parser & KSFO Preview Cloud", () => {
  it("decodes raw & gzipped SPZ v2 round-trip positions, RGB colors, opacity & scale within quantization bounds", async () => {
    const splats: SyntheticSplat[] = [
      { posNed: [125.5, -42.25, -18.75], rgb: [0.88, 0.42, 0.16], alpha: 0.92, logScale: [1.5, 1.5, 0.5] },
      { posNed: [-310.125, 214.0, 2.5], rgb: [0.15, 0.78, 0.91], alpha: 0.55, logScale: [2.0, 2.0, 1.0] },
      { posNed: [0.0, 0.0, 0.0], rgb: [0.95, 0.95, 0.95], alpha: 1.0, logScale: [0.8, 0.8, 0.2] },
    ];
    const raw = encodeSpzV2Raw(splats, 12, 0);
    const gz = gzipSync(raw);

    const parsedRaw = parseSpzV2Raw(raw, "ksfo_test_raw.spz");
    const parsedGz = await parseSpzV2Gzip(new Uint8Array(gz), "ksfo_test_gz.spz");

    for (const cloud of [parsedRaw, parsedGz]) {
      expect(cloud.gtTier).toBe("VISUAL_STRESS_ONLY");
      expect(cloud.totalFileSplats).toBe(3);
      expect(cloud.renderSplats).toBe(3);
      expect(cloud.stride).toBe(1);
      expect(cloud.header.magicU32).toBe(SPZ_MAGIC_U32);
      expect(cloud.header.version).toBe(SPZ_VERSION);

      for (let i = 0; i < splats.length; i++) {
        for (let c = 0; c < 3; c++) {
          expect(Math.abs(cloud.positionsNed[i * 3 + c] - splats[i].posNed[c])).toBeLessThan(1e-3);
          expect(Math.abs(cloud.colorsRgb[i * 3 + c] - splats[i].rgb[c])).toBeLessThan(0.02);
        }
        expect(Math.abs(cloud.alphas[i] - splats[i].alpha)).toBeLessThan(0.01);
        const expectedR = Math.exp(
          (splats[i].logScale[0] + splats[i].logScale[1] + splats[i].logScale[2]) / 3.0,
        );
        expect(Math.abs(cloud.radiiM[i] - expectedR)).toBeLessThan(0.35);
      }
    }
  });

  it("rejects invalid magic, unsupported version, truncated header, and length mismatches fail-closed", async () => {
    // Truncated header
    expect(() => parseSpzV2Raw(new Uint8Array(8))).toThrow(/truncated/i);

    // Invalid magic
    const badMagic = encodeSpzV2Raw(
      [{ posNed: [1, 2, 3], rgb: [0.5, 0.5, 0.5], alpha: 0.8, logScale: [1, 1, 1] }],
    );
    badMagic[0] = 0x58; // 'X'
    expect(() => parseSpzV2Raw(badMagic)).toThrow(/magic/i);

    // Unsupported version
    const badVer = encodeSpzV2Raw(
      [{ posNed: [1, 2, 3], rgb: [0.5, 0.5, 0.5], alpha: 0.8, logScale: [1, 1, 1] }],
    );
    new DataView(badVer.buffer).setUint32(4, 99, true);
    expect(() => parseSpzV2Raw(badVer)).toThrow(/version/i);

    // Truncated payload body
    const good = encodeSpzV2Raw(
      [{ posNed: [1, 2, 3], rgb: [0.5, 0.5, 0.5], alpha: 0.8, logScale: [1, 1, 1] }],
    );
    expect(() => parseSpzV2Raw(good.subarray(0, good.byteLength - 4))).toThrow(/mismatch/i);

    // Non-gzip / non-NGSP bytes in gzip loader
    await expect(parseSpzV2Gzip(new Uint8Array([0x01, 0x02, 0x03, 0x04]))).rejects.toThrow(/gzip|NGSP/i);
  });

  it("stride-subsamples large SPZ clouds cleanly to <= MAX_RENDER_SPLATS", () => {
    const n = MAX_RENDER_SPLATS + 5000;
    const many: SyntheticSplat[] = Array.from({ length: n }, (_, idx) => ({
      posNed: [(idx % 100) - 50, Math.floor(idx / 100) - 50, -5],
      rgb: [0.4, 0.6, 0.8],
      alpha: 0.8,
      logScale: [1.0, 1.0, 0.5],
    }));
    const raw = encodeSpzV2Raw(many, 12, 0);
    const cloud = parseSpzV2Raw(raw, "large_5m_sim.spz", MAX_RENDER_SPLATS);
    expect(cloud.totalFileSplats).toBe(n);
    expect(cloud.stride).toBe(2);
    expect(cloud.renderSplats).toBeLessThanOrEqual(MAX_RENDER_SPLATS);
    expect(cloud.renderSplats).toBe(Math.ceil(n / 2));
  });

  it("builds deterministic VISUAL_STRESS_ONLY KSFO 28L preview cloud with all finite values", () => {
    const preview = buildKsfo28lPreviewSpzCloud();
    expect(preview.gtTier).toBe("VISUAL_STRESS_ONLY");
    expect(preview.renderSplats).toBeGreaterThan(4000);
    expect(preview.renderSplats).toBeLessThanOrEqual(MAX_RENDER_SPLATS);
    expect(Array.from(preview.positionsNed).every(Number.isFinite)).toBe(true);
    expect(Array.from(preview.colorsRgb).every((c) => Number.isFinite(c) && c >= 0 && c <= 1)).toBe(true);
    expect(Array.from(preview.alphas).every((a) => Number.isFinite(a) && a >= 0 && a <= 1)).toBe(true);
    expect(Array.from(preview.radiiM).every((r) => Number.isFinite(r) && r > 0)).toBe(true);
  });

  it("decodes bundled Colab-trained ksfo_28l_1km_visual_stress.spz (9,216 splats, VISUAL_STRESS_ONLY)", async () => {
    const spzPath = resolve(
      dirname(fileURLToPath(import.meta.url)),
      "../../public/assets/ksfo_28l_1km_visual_stress.spz",
    );
    const bytes = readFileSync(spzPath);
    expect(bytes.byteLength).toBe(90450);

    const cloud = await parseSpzV2Gzip(new Uint8Array(bytes), "ksfo_28l_1km_visual_stress.spz");
    const magicBytes = new Uint8Array(4);
    new DataView(magicBytes.buffer).setUint32(0, cloud.header.magicU32, true);
    const magicStr = String.fromCharCode(...magicBytes);

    expect(magicStr).toBe("NGSP");
    expect(cloud.header.magicU32).toBe(SPZ_MAGIC_U32);
    expect(cloud.header.version).toBe(2);
    expect(cloud.header.numPoints).toBe(9216);
    expect(cloud.totalFileSplats).toBe(9216);
    expect(cloud.renderSplats).toBe(9216);
    expect(cloud.stride).toBe(1);
    expect(cloud.gtTier).toBe("VISUAL_STRESS_ONLY");
    expect(cloud.sourceLabel).toBe("ksfo_28l_1km_visual_stress.spz");

    expect(Array.from(cloud.positionsNed).every(Number.isFinite)).toBe(true);
    expect(Array.from(cloud.colorsRgb).every((c) => Number.isFinite(c) && c >= 0 && c <= 1)).toBe(true);
    expect(Array.from(cloud.alphas).every((a) => Number.isFinite(a) && a >= 0 && a <= 1)).toBe(true);
    expect(Array.from(cloud.radiiM).every((r) => Number.isFinite(r) && r > 0)).toBe(true);
  });
});
