/**
 * Pass 2: Dual-Ring 16-bit USGS 3DEP (GEOID18 -> WGS-84) CDLOD Terrain Clipmap
 * with TSO-C151d TAWS FLTA caution/warning shading and explicit procedural-ci mode.
 */

import { DEG_TO_RAD, WGS84_A, geodeticToEcef, rEcefNed } from "../rhi/coords.ts";
import type {
  RhiBufferHandle,
  RhiContext,
  RhiPipelineHandle,
  RhiTextureHandle,
} from "../rhi/types.ts";
import { SF_BAY_SUN_DIR_NED } from "./pass1_das_efvs.ts";
import pass2Wgsl from "../shaders/pass2_terrain_cdlod.wgsl?raw";
import pass2GlVert from "../shaders/generated_gl/pass2_terrain_cdlod.vert.glsl?raw";
import pass2GlFrag from "../shaders/generated_gl/pass2_terrain_cdlod.frag.glsl?raw";
import type {
  ActiveCameraView,
  CockpitFrameState,
  RunwayGeometry,
  TerrainPackData,
  TerrainRingData,
} from "./types.ts";

const KSFO_DEFAULT_LAT_DEG = 37.6188056;
const KSFO_DEFAULT_LON_DEG = -122.3754167;
// Exact geographic bounds matching `scripts/xvs/build_ksfo_naip_ortho_pack.py --core2k` (2048x2048, 2x2 km)
export const KSFO_CORE2K_CENTER_LAT_DEG = 37.6155;
export const KSFO_CORE2K_CENTER_LON_DEG = -122.3625;
export const KSFO_CORE2K_HALF_EXTENT_M = 1000.0;
const MAGIC_XVS3DEP1 = "XVS3DEP1";

interface RawRingHeader {
  name: string;
  extent_km: number;
  width: number;
  height: number;
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
  min_wgs84_h_m: number;
  max_wgs84_h_m: number;
  byte_offset: number;
  byte_length: number;
}

interface RawRunwayHeader {
  id: string;
  kind: string;
  width_m: number;
  length_m: number;
  true_heading_deg: number;
  threshold_lat_deg: number;
  threshold_lon_deg: number;
  threshold_wgs84_h_m: number;
  opposite_end_lat_deg: number;
  opposite_end_lon_deg: number;
  opposite_end_wgs84_h_m: number;
  polygon?: Array<[number, number, number]>;
}

interface RawPackHeader {
  terrain_mode: string;
  geoid_model: string;
  datum: string;
  origin_lat_deg: number;
  origin_lon_deg: number;
  ksfo_28l_thr_lat_deg: number;
  ksfo_28l_thr_lon_deg: number;
  ksfo_28l_thr_wgs84_h_m: number;
  quantization: {
    h_min_m: number;
    h_step_m: number;
    dtype: string;
  };
  rings: RawRingHeader[];
  runways: RawRunwayHeader[];
}

/**
 * Parse binary `ksfo_3dep_geoid18_v1.bin` (`magic = "XVS3DEP1"`, version 1) produced by
 * `scripts/xvs/build_ksfo_3dep_pack.py`.
 */
export function parseKsfoTerrainPack(buffer: ArrayBuffer): TerrainPackData {
  if (buffer.byteLength < 16) {
    throw new Error("KSFO terrain pack truncated (< 16 byte preamble)");
  }
  const bytes = new Uint8Array(buffer);
  const magic = String.fromCharCode(...bytes.subarray(0, 8));
  if (magic !== MAGIC_XVS3DEP1) {
    throw new Error(`Invalid terrain pack magic "${magic}" (expected "${MAGIC_XVS3DEP1}")`);
  }
  const dv = new DataView(buffer);
  const version = dv.getUint32(8, true);
  if (version !== 1) {
    throw new Error(`Unsupported XVS3DEP1 version ${version}`);
  }
  const hdrLen = dv.getUint32(12, true);
  if (16 + hdrLen > buffer.byteLength) {
    throw new Error("KSFO terrain pack JSON header out of bounds");
  }

  const hdrText = new TextDecoder("utf-8").decode(bytes.subarray(16, 16 + hdrLen));
  const hdr = JSON.parse(hdrText) as RawPackHeader;

  const parseRing = (rh: RawRingHeader): TerrainRingData => {
    const count = rh.width * rh.height;
    if (rh.byte_offset + count * 2 > buffer.byteLength) {
      throw new Error(`Terrain ring "${rh.name}" data out of bounds`);
    }
    const u16 = new Uint16Array(count);
    for (let i = 0; i < count; i++) {
      u16[i] = dv.getUint16(rh.byte_offset + i * 2, true);
    }
    return {
      name: rh.name,
      extentKm: rh.extent_km,
      width: rh.width,
      height: rh.height,
      latMin: rh.lat_min,
      latMax: rh.lat_max,
      lonMin: rh.lon_min,
      lonMax: rh.lon_max,
      minWgs84HM: rh.min_wgs84_h_m,
      maxWgs84HM: rh.max_wgs84_h_m,
      heightsU16: u16,
    };
  };

  if (hdr.rings.length < 2) {
    throw new Error(`Expected 2 terrain rings (outer + inner), found ${hdr.rings.length}`);
  }
  const outerHdr = hdr.rings.find((r) => r.name.includes("outer")) ?? hdr.rings[0];
  const innerHdr = hdr.rings.find((r) => r.name.includes("inner")) ?? hdr.rings[1];

  const runways: RunwayGeometry[] = (hdr.runways ?? []).map((r) => ({
    id: r.id,
    kind: r.kind,
    widthM: r.width_m,
    lengthM: r.length_m,
    trueHeadingDeg: r.true_heading_deg,
    thresholdLatDeg: r.threshold_lat_deg,
    thresholdLonDeg: r.threshold_lon_deg,
    thresholdWgs84HM: r.threshold_wgs84_h_m,
    oppositeEndLatDeg: r.opposite_end_lat_deg,
    oppositeEndLonDeg: r.opposite_end_lon_deg,
    oppositeEndWgs84HM: r.opposite_end_wgs84_h_m,
    polygon: r.polygon,
  }));

  return {
    terrainMode: hdr.terrain_mode,
    geoidModel: hdr.geoid_model,
    datum: hdr.datum,
    originLatDeg: hdr.origin_lat_deg,
    originLonDeg: hdr.origin_lon_deg,
    ksfo28lThrLatDeg: hdr.ksfo_28l_thr_lat_deg,
    ksfo28lThrLonDeg: hdr.ksfo_28l_thr_lon_deg,
    ksfo28lThrWgs84HM: hdr.ksfo_28l_thr_wgs84_h_m,
    hMinM: hdr.quantization.h_min_m,
    hStepM: hdr.quantization.h_step_m,
    outerRing: parseRing(outerHdr),
    innerRing: parseRing(innerHdr),
    runways,
  };
}

/**
 * Explicit hermetic CI / offline terrain pack tagged `terrainMode = "procedural_ci"`.
 * NEVER silently substituted for USGS 3DEP — triggers unmistakable amber HUD badge.
 */
export function createProceduralCiTerrainPack(gridSize = 256): TerrainPackData {
  const geoidConst = -32.25;
  const hMinM = -100.0;
  const hStepM = 0.1;

  const buildRing = (name: string, extentKm: number): TerrainRingData => {
    const halfM = extentKm * 500.0;
    const dLatDeg = (halfM / WGS84_A) * (180.0 / Math.PI);
    const dLonDeg = dLatDeg / Math.cos(KSFO_DEFAULT_LAT_DEG * DEG_TO_RAD);
    const latMin = KSFO_DEFAULT_LAT_DEG - dLatDeg;
    const latMax = KSFO_DEFAULT_LAT_DEG + dLatDeg;
    const lonMin = KSFO_DEFAULT_LON_DEG - dLonDeg;
    const lonMax = KSFO_DEFAULT_LON_DEG + dLonDeg;

    const heightsU16 = new Uint16Array(gridSize * gridSize);
    let minH = Infinity;
    let maxH = -Infinity;

    for (let row = 0; row < gridSize; row++) {
      const lat = latMax - (row / (gridSize - 1)) * (latMax - latMin);
      for (let col = 0; col < gridSize; col++) {
        const lon = lonMin + (col / (gridSize - 1)) * (lonMax - lonMin);
        const ridge =
          Math.max(0.0, (-122.41 - lon) * 3500.0) *
          Math.exp(-Math.pow((lat - 37.62) / 0.14, 2));
        const hWgs84 = Math.min(550.0, Math.max(0.0, ridge)) + geoidConst;
        minH = Math.min(minH, hWgs84);
        maxH = Math.max(maxH, hWgs84);
        heightsU16[row * gridSize + col] = Math.max(
          0,
          Math.min(65535, Math.round((hWgs84 - hMinM) / hStepM)),
        );
      }
    }

    return {
      name,
      extentKm,
      width: gridSize,
      height: gridSize,
      latMin,
      latMax,
      lonMin,
      lonMax,
      minWgs84HM: minH,
      maxWgs84HM: maxH,
      heightsU16,
    };
  };

  return {
    terrainMode: "procedural_ci",
    geoidModel: "procedural_ci_constant",
    datum: "WGS84_ELLIPSOIDAL",
    originLatDeg: KSFO_DEFAULT_LAT_DEG,
    originLonDeg: KSFO_DEFAULT_LON_DEG,
    ksfo28lThrLatDeg: 37.61172,
    ksfo28lThrLonDeg: -122.358367,
    ksfo28lThrWgs84HM: 3.9624 + geoidConst,
    hMinM,
    hStepM,
    outerRing: buildRing("outer_sf_bay", 40.0),
    innerRing: buildRing("inner_ksfo", 10.0),
    runways: [
      {
        id: "28L",
        kind: "runway",
        widthM: 60.96,
        lengthM: 3471.0,
        trueHeadingDeg: 297.7,
        thresholdLatDeg: 37.61172,
        thresholdLonDeg: -122.358367,
        thresholdWgs84HM: 3.9624 + geoidConst,
        oppositeEndLatDeg: 37.6263,
        oppositeEndLonDeg: -122.3932,
        oppositeEndWgs84HM: 3.9624 + geoidConst,
      },
      {
        id: "28R",
        kind: "runway",
        widthM: 60.96,
        lengthM: 3618.0,
        trueHeadingDeg: 297.7,
        thresholdLatDeg: 37.6135,
        thresholdLonDeg: -122.3571,
        thresholdWgs84HM: 3.9624 + geoidConst,
        oppositeEndLatDeg: 37.6287,
        oppositeEndLonDeg: -122.3934,
        oppositeEndWgs84HM: 3.9624 + geoidConst,
      },
      {
        id: "Taxiway C",
        kind: "taxiway",
        widthM: 22.86,
        lengthM: 3471.0,
        trueHeadingDeg: 297.7,
        thresholdLatDeg: 37.6104,
        thresholdLonDeg: -122.3592,
        thresholdWgs84HM: 3.9624 + geoidConst,
        oppositeEndLatDeg: 37.6249,
        oppositeEndLonDeg: -122.3941,
        oppositeEndWgs84HM: 3.9624 + geoidConst,
      },
    ],
  };
}

/** Encode uint16 heightfield into RGBA8 (R=hi byte, G=lo byte) for universal WebGPU+WebGL2 compatibility. */
function packU16HeightToRgba8(ring: TerrainRingData): Uint8Array {
  const n = ring.width * ring.height;
  const rgba = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    const q = ring.heightsU16[i];
    rgba[i * 4 + 0] = (q >> 8) & 0xff;
    rgba[i * 4 + 1] = q & 0xff;
    rgba[i * 4 + 2] = 0;
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}

/** Build dual-ring CDLOD clipmap mesh (inner 10 km full grid + outer 40 km hollow ring). */
function buildDualRingCdlodMesh(pack: TerrainPackData, segs = 64): {
  vertices: Float32Array;
  indices: Uint32Array;
} {
  const originLatRad = pack.originLatDeg * DEG_TO_RAD;
  const mPerDegLat = (Math.PI / 180.0) * WGS84_A;
  const mPerDegLon = mPerDegLat * Math.cos(originLatRad);

  const verts: number[] = [];
  const inds: number[] = [];

  const appendRing = (ring: TerrainRingData, ringFlag: number, cutHoleKm: number) => {
    const baseVertex = verts.length / 5;
    const cols = segs + 1;
    const rows = segs + 1;

    for (let r = 0; r < rows; r++) {
      const v = r / segs; // 0 = latMax (north), 1 = latMin (south)
      const latDeg = ring.latMax - v * (ring.latMax - ring.latMin);
      const northM = (latDeg - pack.originLatDeg) * mPerDegLat;
      for (let c = 0; c < cols; c++) {
        const u = c / segs; // 0 = lonMin (west), 1 = lonMax (east)
        const lonDeg = ring.lonMin + u * (ring.lonMax - ring.lonMin);
        const eastM = (lonDeg - pack.originLonDeg) * mPerDegLon;
        verts.push(northM, eastM, u, v, ringFlag);
      }
    }

    const halfHoleM = cutHoleKm * 485.0; // slight overlap avoids T-junction cracks
    for (let r = 0; r < segs; r++) {
      for (let c = 0; c < segs; c++) {
        const i00 = baseVertex + r * cols + c;
        const i10 = i00 + 1;
        const i01 = i00 + cols;
        const i11 = i01 + 1;

        if (cutHoleKm > 0) {
          const nMid = 0.5 * (verts[i00 * 5] + verts[i11 * 5]);
          const eMid = 0.5 * (verts[i00 * 5 + 1] + verts[i11 * 5 + 1]);
          if (Math.abs(nMid) < halfHoleM && Math.abs(eMid) < halfHoleM) {
            continue;
          }
        }
        inds.push(i00, i01, i10, i10, i01, i11);
      }
    }
  };

  appendRing(pack.innerRing, 0.0, 0.0);
  appendRing(pack.outerRing, 1.0, pack.innerRing.extentKm);

  return {
    vertices: new Float32Array(verts),
    indices: new Uint32Array(inds),
  };
}

async function decodeOrthoImageUrlToRgba(
  url: string,
): Promise<{ rgba: Uint8Array; width: number; height: number }> {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Failed to fetch ortho ${url} (HTTP ${resp.status})`);
  }
  const blob = await resp.blob();
  const bitmap = await createImageBitmap(blob);
  try {
    if (typeof OffscreenCanvas !== "undefined") {
      const off = new OffscreenCanvas(bitmap.width, bitmap.height);
      const ctx = off.getContext("2d");
      if (ctx) {
        ctx.drawImage(bitmap, 0, 0);
        const img = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
        return {
          rgba: new Uint8Array(img.data.buffer),
          width: img.width,
          height: img.height,
        };
      }
    }
    const el = document.createElement("canvas");
    el.width = bitmap.width;
    el.height = bitmap.height;
    const ctx = el.getContext("2d");
    if (!ctx) {
      throw new Error("2D canvas context unavailable for ortho decode");
    }
    ctx.drawImage(bitmap, 0, 0);
    const img = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
    return {
      rgba: new Uint8Array(img.data.buffer),
      width: img.width,
      height: img.height,
    };
  } finally {
    bitmap.close();
  }
}

export class Pass2TerrainCdlod {
  private readonly rhi: RhiContext;
  private readonly pipeline: RhiPipelineHandle;
  private readonly ubo: RhiBufferHandle;
  private readonly uniformData = new Float32Array(52); // mat4 (16) + 9 x vec4 (36) = 52 floats (208 B)

  private vb: RhiBufferHandle;
  private ib: RhiBufferHandle;
  private indexCount: number;
  private texInner: RhiTextureHandle;
  private texOuter: RhiTextureHandle;
  private texOrthoInner: RhiTextureHandle;
  private texOrthoOuter: RhiTextureHandle;
  private texOrthoCore: RhiTextureHandle;
  private orthoLoaded = false;
  private orthoCoreLoaded = false;
  private pack: TerrainPackData;
  private originEcef: [number, number, number];
  private originREcefNed: ReturnType<typeof rEcefNed>;
  private coreCenterN = 0;
  private coreCenterE = 0;

  constructor(rhi: RhiContext, initialPack?: TerrainPackData) {
    this.rhi = rhi;
    this.pack = initialPack ?? createProceduralCiTerrainPack();
    this.originEcef = geodeticToEcef(
      this.pack.originLatDeg * DEG_TO_RAD,
      this.pack.originLonDeg * DEG_TO_RAD,
      0.0,
    );
    this.originREcefNed = rEcefNed(
      this.pack.originLatDeg * DEG_TO_RAD,
      this.pack.originLonDeg * DEG_TO_RAD,
    );
    this.recomputeCoreBoundsNe();

    this.ubo = rhi.createBuffer("uniform", this.uniformData.byteLength, "pass2-ubo");
    this.pipeline = rhi.createRenderPipeline({
      label: "pass2-terrain-cdlod",
      wgsl: pass2Wgsl,
      glslVert: pass2GlVert,
      glslFrag: pass2GlFrag,
      topology: "triangle-list",
      depthTest: true,
      depthWrite: true,
      blendAlpha: false,
      vertexLayouts: [
        {
          arrayStride: 20, // vec2 neCoord + vec3 uvRing
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x2" },
            { shaderLocation: 1, offset: 8, format: "float32x3" },
          ],
        },
      ],
      uniformBindings: [0],
      textureBindings: [
        { textureBinding: 1 },
        { textureBinding: 2 },
        { textureBinding: 3, samplerBinding: 5 },
        { textureBinding: 4 },
        { textureBinding: 6 },
      ],
    });

    const mesh = buildDualRingCdlodMesh(this.pack, 64);
    this.vb = rhi.createBuffer("vertex", mesh.vertices, "pass2-cdlod-vb");
    this.ib = rhi.createBuffer("index", mesh.indices, "pass2-cdlod-ib");
    this.indexCount = mesh.indices.length;

    this.texInner = rhi.createTextureRGBA8(
      this.pack.innerRing.width,
      this.pack.innerRing.height,
      packU16HeightToRgba8(this.pack.innerRing),
      "pass2-inner-3dep",
    );
    this.texOuter = rhi.createTextureRGBA8(
      this.pack.outerRing.width,
      this.pack.outerRing.height,
      packU16HeightToRgba8(this.pack.outerRing),
      "pass2-outer-3dep",
    );
    // Tiny 1x1 Stage-0 placeholders until Stage-1 real USGS/USDA satellite ortho JPEGs hydrate
    const placeholder = new Uint8Array([32, 56, 72, 255]);
    this.texOrthoInner = rhi.createTextureRGBA8(1, 1, placeholder, "pass2-ortho-inner-init");
    this.texOrthoOuter = rhi.createTextureRGBA8(1, 1, placeholder, "pass2-ortho-outer-init");
    this.texOrthoCore = rhi.createTextureRGBA8(1, 1, placeholder, "pass2-ortho-core2k-init");
  }

  private recomputeCoreBoundsNe(): void {
    const mPerDegLat = (Math.PI / 180.0) * WGS84_A;
    const mPerDegLon = mPerDegLat * Math.cos(this.pack.originLatDeg * DEG_TO_RAD);
    this.coreCenterN = (KSFO_CORE2K_CENTER_LAT_DEG - this.pack.originLatDeg) * mPerDegLat;
    this.coreCenterE = (KSFO_CORE2K_CENTER_LON_DEG - this.pack.originLonDeg) * mPerDegLon;
  }

  getTerrainPack(): TerrainPackData {
    return this.pack;
  }

  isOrthoLoaded(): boolean {
    return this.orthoLoaded;
  }

  isOrthoCoreLoaded(): boolean {
    return this.orthoCoreLoaded;
  }

  setOrthoTextures(
    inner: { rgba: Uint8Array; width: number; height: number },
    outer: { rgba: Uint8Array; width: number; height: number },
  ): void {
    this.rhi.destroyTexture(this.texOrthoInner);
    this.rhi.destroyTexture(this.texOrthoOuter);
    this.texOrthoInner = this.rhi.createTextureRGBA8(
      inner.width,
      inner.height,
      inner.rgba,
      "pass2-ortho-inner-usgs",
    );
    this.texOrthoOuter = this.rhi.createTextureRGBA8(
      outer.width,
      outer.height,
      outer.rgba,
      "pass2-ortho-outer-usgs",
    );
    this.orthoLoaded = true;
  }

  setOrthoCoreTexture(core: { rgba: Uint8Array; width: number; height: number }): void {
    this.rhi.destroyTexture(this.texOrthoCore);
    this.texOrthoCore = this.rhi.createTextureRGBA8(
      core.width,
      core.height,
      core.rgba,
      "pass2-ortho-core2k-usgs",
    );
    this.orthoCoreLoaded = true;
  }

  async loadOrthoCoreTextureFromUrl(
    coreUrl = "./assets/ksfo_ortho_core2k_v2.jpg",
  ): Promise<boolean> {
    const core = await decodeOrthoImageUrlToRgba(coreUrl);
    this.setOrthoCoreTexture(core);
    return true;
  }

  async loadOrthoTexturesFromUrls(
    innerUrl = "./assets/ksfo_ortho_inner_v1.jpg",
    outerUrl = "./assets/ksfo_ortho_outer_v1.jpg",
    coreUrl = "./assets/ksfo_ortho_core2k_v2.jpg",
  ): Promise<boolean> {
    const corePromise = decodeOrthoImageUrlToRgba(coreUrl).catch(() => null);
    const [inner, outer, core] = await Promise.all([
      decodeOrthoImageUrlToRgba(innerUrl),
      decodeOrthoImageUrlToRgba(outerUrl),
      corePromise,
    ]);
    this.setOrthoTextures(inner, outer);
    if (core) {
      this.setOrthoCoreTexture(core);
    }
    return true;
  }

  setTerrainPack(pack: TerrainPackData): void {
    this.pack = pack;
    this.originEcef = geodeticToEcef(
      pack.originLatDeg * DEG_TO_RAD,
      pack.originLonDeg * DEG_TO_RAD,
      0.0,
    );
    this.originREcefNed = rEcefNed(
      pack.originLatDeg * DEG_TO_RAD,
      pack.originLonDeg * DEG_TO_RAD,
    );
    this.recomputeCoreBoundsNe();

    this.rhi.destroyBuffer(this.vb);
    this.rhi.destroyBuffer(this.ib);
    this.rhi.destroyTexture(this.texInner);
    this.rhi.destroyTexture(this.texOuter);

    const mesh = buildDualRingCdlodMesh(pack, 64);
    this.vb = this.rhi.createBuffer("vertex", mesh.vertices, "pass2-cdlod-vb");
    this.ib = this.rhi.createBuffer("index", mesh.indices, "pass2-cdlod-ib");
    this.indexCount = mesh.indices.length;

    this.texInner = this.rhi.createTextureRGBA8(
      pack.innerRing.width,
      pack.innerRing.height,
      packU16HeightToRgba8(pack.innerRing),
      "pass2-inner-3dep",
    );
    this.texOuter = this.rhi.createTextureRGBA8(
      pack.outerRing.width,
      pack.outerRing.height,
      packU16HeightToRgba8(pack.outerRing),
      "pass2-outer-3dep",
    );
  }

  encode(state: CockpitFrameState, view: ActiveCameraView): void {
    const u = this.uniformData;
    // viewProj (16 floats)
    u.set(view.viewProj, 0);

    // originRteEcef = p_origin_ecef - p_cam_ecef in f64 RTE
    const rteX = this.originEcef[0] - view.pEcefCam[0];
    const rteY = this.originEcef[1] - view.pEcefCam[1];
    const rteZ = this.originEcef[2] - view.pEcefCam[2];
    u[16] = rteX;
    u[17] = rteY;
    u[18] = rteZ;
    u[19] = this.pack.hMinM;

    // Columns of R_ecef_ned at KSFO origin
    const r = this.originREcefNed;
    u[20] = r[0]; u[21] = r[3]; u[22] = r[6]; u[23] = this.pack.hStepM;
    u[24] = r[1]; u[25] = r[4]; u[26] = r[7]; u[27] = state.altWgs84M;
    u[28] = r[2]; u[29] = r[5]; u[30] = r[8]; u[31] = state.fogContrast;

    // renderParams
    const isProcCi = this.pack.terrainMode === "procedural_ci";
    u[32] = state.efvsEnabled ? 1.0 : 0.0;
    u[33] = isProcCi ? 1.0 : 0.0;
    u[34] = state.simTimeS;
    u[35] = this.pack.innerRing.width;

    // orthoParams: real USGS/USDA satellite ortho active only after Stage-1 hydration & outside procedural_ci
    const orthoActive = this.orthoLoaded && !isProcCi;
    u[36] = orthoActive ? 1.0 : 0.0;
    u[37] = orthoActive ? 1.0 : 0.0;
    u[38] = this.pack.innerRing.extentKm * 1000.0;
    u[39] = this.pack.outerRing.extentKm * 1000.0;

    // sunDirNed (matches Pass 1) + coreOrthoEnabled
    u[40] = SF_BAY_SUN_DIR_NED[0];
    u[41] = SF_BAY_SUN_DIR_NED[1];
    u[42] = SF_BAY_SUN_DIR_NED[2];
    u[43] = this.orthoCoreLoaded && !isProcCi ? 1.0 : 0.0;

    // coreBoundsNe: 2x2 km KSFO core ortho bounds in local NED meters
    u[44] = this.coreCenterN;
    u[45] = this.coreCenterE;
    u[46] = KSFO_CORE2K_HALF_EXTENT_M;
    u[47] = 0.0;

    // camPosNeDown: camera position in KSFO origin NED for accurate water Fresnel & GGX specular
    const dx = -rteX;
    const dy = -rteY;
    const dz = -rteZ;
    u[48] = r[0] * dx + r[3] * dy + r[6] * dz;
    u[49] = r[1] * dx + r[4] * dy + r[7] * dz;
    u[50] = r[2] * dx + r[5] * dy + r[8] * dz;
    u[51] = 0.0;

    this.rhi.writeBuffer(this.ubo, this.uniformData);
    this.rhi.draw({
      pipeline: this.pipeline,
      uniformBuffers: [{ binding: 0, buffer: this.ubo }],
      textures: [
        { textureBinding: 1, texture: this.texInner },
        { textureBinding: 2, texture: this.texOuter },
        { textureBinding: 3, samplerBinding: 5, texture: this.texOrthoInner },
        { textureBinding: 4, texture: this.texOrthoOuter },
        { textureBinding: 6, texture: this.texOrthoCore },
      ],
      vertexBuffers: [this.vb],
      indexBuffer: this.ib,
      indexCount: this.indexCount,
    });
  }

  dispose(): void {
    this.rhi.destroyBuffer(this.vb);
    this.rhi.destroyBuffer(this.ib);
    this.rhi.destroyBuffer(this.ubo);
    this.rhi.destroyTexture(this.texInner);
    this.rhi.destroyTexture(this.texOuter);
    this.rhi.destroyTexture(this.texOrthoInner);
    this.rhi.destroyTexture(this.texOrthoOuter);
    this.rhi.destroyTexture(this.texOrthoCore);
  }
}
