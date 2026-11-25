/**
 * Binary USGS 3DEP (GEOID18 -> WGS-84) Heightfield Parser & KSFO-28L Local Meter Sampler.
 *
 * Reads the real dual-ring 256x256 16-bit heightfield `ksfo_3dep_geoid18_v1.bin` (`XVS3DEP1`)
 * and converts coordinates cleanly into KSFO 28L threshold local ENU/Three.js meter space:
 *   +X = East (m), +Y = Up above KSFO 28L runway datum (m), -Z = North (m).
 */

export const KSFO_28L_ORIGIN = {
  latDeg: 37.61356,
  lonDeg: -122.35714,
  runwayWgs84HM: -28.6714, // NAVD88 3.96m + GEOID18 -32.63m
  runwayNavd88M: 3.9624,
  trueHeadingDeg: 297.7,
} as const;

export const METERS_PER_DEG_LAT = 110_996.0;
export const METERS_PER_DEG_LON = 111_319.5 * Math.cos((KSFO_28L_ORIGIN.latDeg * Math.PI) / 180.0);

export interface KsfoTerrainRing {
  readonly name: string;
  readonly extentKm: number;
  readonly width: number;
  readonly height: number;
  readonly latMin: number;
  readonly latMax: number;
  readonly lonMin: number;
  readonly lonMax: number;
  readonly minWgs84HM: number;
  readonly maxWgs84HM: number;
  readonly heightsU16: Uint16Array;
}

export interface KsfoTerrainPack {
  readonly terrainMode: string;
  readonly hMinM: number;
  readonly hStepM: number;
  readonly ksfo28lWgs84HM: number;
  readonly outer: KsfoTerrainRing;
  readonly inner: KsfoTerrainRing;
}

interface RawRingHeader {
  readonly name: string;
  readonly extent_km: number;
  readonly width: number;
  readonly height: number;
  readonly lat_min: number;
  readonly lat_max: number;
  readonly lon_min: number;
  readonly lon_max: number;
  readonly min_wgs84_h_m: number;
  readonly max_wgs84_h_m: number;
  readonly byte_offset: number;
  readonly byte_length: number;
}

interface RawPackHeader {
  readonly terrain_mode: string;
  readonly ksfo_28l_thr_wgs84_h_m: number;
  readonly quantization: {
    readonly h_min_m: number;
    readonly h_step_m: number;
  };
  readonly rings: RawRingHeader[];
}

export function latLonToKsfoLocalEn(latDeg: number, lonDeg: number): readonly [number, number] {
  const eastM = (lonDeg - KSFO_28L_ORIGIN.lonDeg) * METERS_PER_DEG_LON;
  const northM = (latDeg - KSFO_28L_ORIGIN.latDeg) * METERS_PER_DEG_LAT;
  return [eastM, northM];
}

export function ksfoLocalEnToLatLon(eastM: number, northM: number): readonly [number, number] {
  const latDeg = KSFO_28L_ORIGIN.latDeg + northM / METERS_PER_DEG_LAT;
  const lonDeg = KSFO_28L_ORIGIN.lonDeg + eastM / METERS_PER_DEG_LON;
  return [latDeg, lonDeg];
}

export function parseKsfoTerrainPack(buffer: ArrayBuffer): KsfoTerrainPack {
  if (buffer.byteLength < 16) {
    throw new Error("KSFO terrain pack truncated (< 16 bytes)");
  }
  const bytes = new Uint8Array(buffer);
  const magic = String.fromCharCode(...bytes.subarray(0, 8));
  if (magic !== "XVS3DEP1") {
    throw new Error(`Invalid KSFO terrain magic "${magic}"`);
  }
  const dv = new DataView(buffer);
  const version = dv.getUint32(8, true);
  if (version !== 1) {
    throw new Error(`Unsupported XVS3DEP1 version ${version}`);
  }
  const hdrLen = dv.getUint32(12, true);
  const hdrJson = new TextDecoder("utf-8").decode(bytes.subarray(16, 16 + hdrLen));
  const hdr = JSON.parse(hdrJson) as RawPackHeader;

  const parseRing = (rh: RawRingHeader): KsfoTerrainRing => {
    const count = rh.width * rh.height;
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

  const outerHdr = hdr.rings.find((r) => r.name.includes("outer")) ?? hdr.rings[0];
  const innerHdr = hdr.rings.find((r) => r.name.includes("inner")) ?? hdr.rings[1];

  return {
    terrainMode: hdr.terrain_mode,
    hMinM: hdr.quantization.h_min_m,
    hStepM: hdr.quantization.h_step_m,
    ksfo28lWgs84HM: hdr.ksfo_28l_thr_wgs84_h_m,
    outer: parseRing(outerHdr),
    inner: parseRing(innerHdr),
  };
}

function sampleRingBilinear(
  ring: KsfoTerrainRing,
  latDeg: number,
  lonDeg: number,
  hMinM: number,
  hStepM: number,
): number | null {
  if (latDeg < ring.latMin || latDeg > ring.latMax || lonDeg < ring.lonMin || lonDeg > ring.lonMax) {
    return null;
  }
  const u = (lonDeg - ring.lonMin) / Math.max(1e-9, ring.lonMax - ring.lonMin);
  // Row 0 in XVS3DEP1 is latMax (North-top raster)
  const v = (ring.latMax - latDeg) / Math.max(1e-9, ring.latMax - ring.latMin);
  const fx = Math.max(0, Math.min(ring.width - 1.0001, u * (ring.width - 1)));
  const fy = Math.max(0, Math.min(ring.height - 1.0001, v * (ring.height - 1)));
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const x1 = Math.min(ring.width - 1, x0 + 1);
  const y1 = Math.min(ring.height - 1, y0 + 1);
  const tx = fx - x0;
  const ty = fy - y0;

  const h00 = ring.heightsU16[y0 * ring.width + x0];
  const h10 = ring.heightsU16[y0 * ring.width + x1];
  const h01 = ring.heightsU16[y1 * ring.width + x0];
  const h11 = ring.heightsU16[y1 * ring.width + x1];

  const top = h00 + (h10 - h00) * tx;
  const bot = h01 + (h11 - h01) * tx;
  const q = top + (bot - top) * ty;
  return hMinM + q * hStepM;
}

/**
 * Samples real elevation in meters relative to KSFO 28L runway datum (y = 0 at 28L threshold,
 * bay water level at ~ -3.9 m). Falls back cleanly to a sculpted San Bruno + Bay ridge model
 * if the binary pack is unavailable.
 */
export function sampleKsfoElevationM(
  pack: KsfoTerrainPack | null,
  eastM: number,
  northM: number,
): number {
  if (pack) {
    const [latDeg, lonDeg] = ksfoLocalEnToLatLon(eastM, northM);
    const wgs84H =
      sampleRingBilinear(pack.inner, latDeg, lonDeg, pack.hMinM, pack.hStepM) ??
      sampleRingBilinear(pack.outer, latDeg, lonDeg, pack.hMinM, pack.hStepM);
    if (wgs84H !== null && Number.isFinite(wgs84H)) {
      return wgs84H - pack.ksfo28lWgs84HM;
    }
  }
  // Procedural San Francisco Peninsula & San Bruno Mountain fallback (offline/unit-test safe)
  const bayShoreEast = -1400.0 + northM * 0.25;
  if (eastM > bayShoreEast + 1800.0 && Math.abs(northM) < 4200.0) {
    return -4.2;
  }
  const sanBruno = 395.0 * Math.exp(-(((eastM + 5200.0) ** 2 + (northM - 7800.0) ** 2) / (2 * 2400.0 ** 2)));
  const peninsulaHills =
    260.0 * Math.exp(-(((eastM + 6800.0) ** 2) / (2 * 2800.0 ** 2))) *
    (0.65 + 0.35 * Math.sin(northM * 0.00055));
  const eastBayHills =
    320.0 * Math.exp(-(((eastM - 15500.0) ** 2) / (2 * 3200.0 ** 2)));
  const airportPad =
    Math.exp(-(((eastM + 1100.0) ** 2 + (northM - 550.0) ** 2) / (2 * 1600.0 ** 2)));
  const base = Math.max(-4.2, sanBruno + peninsulaHills + eastBayHills - 12.0);
  return base * (1.0 - 0.92 * airportPad);
}
