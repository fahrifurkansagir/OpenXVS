/**
 * Bit-Exact `f64` TypeScript Mirror of `#![no_std]` Rust `xvs-active-range` (Spec §4.1.1 & §4.2.3).
 *
 * Locked to `tests/golden/active_range_vectors.json` (`max_abs_err < 1e-5`, exact `mode`,
 * `activeBankIdx`, `flags`, and bit-exact 64-byte LE wire frame + IEEE-802.3 CRC32).
 * Pure NED coordinates (`North, East, Down`), zero external dependencies.
 */

export const NUM_BANKS = 6;
export const R_FLOOR_M = 30.0;
export const R_CEIL_M = 6000.0;
export const SIGMA_FLOOR_M = 1.0;

export const R_INIT_M: readonly [number, number, number, number, number, number] = [
  200.0,
  348.2202,
  606.2866,
  1055.6063,
  1837.9174,
  3200.0,
];
export const W_FLOOR = 1.0e-4;
export const W_ACTIVE = 0.02;

export const G_ACCEL_MPS2 = 9.80665;
export const R_SEP_M = 152.4;
export const EVADE_BANK_RAD = 0.4363323129985824;
export const A_MAX_EVADE_MPS2 = 4.572915995845716;
export const K_ALLOW = 0.5;
export const ALPHA_CBF = 0.35;
export const DT_LOOK_S = 1.0;
export const R_EVADE_TRIGGER_M = 220.0;
export const TAU_EVADE_TRIGGER_S = 4.5;

export const PHI_CAND_RAD: readonly number[] = [
  -0.20943951023931956,
  -0.15707963267948966,
  -0.10471975511965978,
  -0.05235987755982989,
  0.0,
  0.05235987755982989,
  0.10471975511965978,
  0.15707963267948966,
  0.20943951023931956,
];
export const PHI_CAND_TAN: readonly number[] = [
  -0.21255656167002213,
  -0.15838444032453627,
  -0.10510423526567647,
  -0.05240777928304121,
  0.0,
  0.05240777928304121,
  0.10510423526567647,
  0.15838444032453627,
  0.21255656167002213,
];

const FOV_LIMIT_RAD = (55.0 - 8.0) * (Math.PI / 180.0);
const TAU_MIN_SIZE_PX = 6.0;
const TAU_MIN_TRACK_AGE = 5;
const KOSCH_MAX_SIZE_PX = 8.0;
const KOSCH_MAX_LOG_DELTA = 0.15;
const TRIGGER_REL_SIGMA = 0.10;
const LOCK_REL_SIGMA = 0.08;
const CLOSING_RHO_DOT_THRESH = -0.015;

export const FLAG_TAU_FUSED = 1 << 0;
export const FLAG_KOSCH_FUSED = 1 << 1;
export const FLAG_SC3BF_ALL_SAFE = 1 << 2;
export const FLAG_FOV_SAFE = 1 << 3;
export const WIRE_SIZE_BYTES = 64;

export const enum ActiveRangeModeTs {
  NominalHold = 0,
  ActiveSwerve = 1,
  Locked = 2,
  ImmediateEvade = 3,
}

export interface ActiveRangeConfigTs {
  readonly rInitM?: readonly [number, number, number, number, number, number];
  readonly sigmaBearingRad?: number;
  readonly qHatConformal?: number;
}

export interface OwnshipStateNedTs {
  readonly vNed: readonly [number, number, number];
  readonly aNed: readonly [number, number, number];
  readonly headingRad: number;
  readonly tasMps: number;
  readonly vStallMps?: number;
}

export interface ActiveRangeMeasurementTs {
  readonly alphaRad: number;
  readonly betaRad: number;
  readonly sizePx: number;
  readonly sizeRatePxS: number;
  readonly contrastRatio?: number;
  readonly skyBetaExt?: number;
  readonly daytimeClear?: boolean;
  readonly trackAge?: number;
  readonly dtS?: number;
}

export interface ActiveRangeTelemetryTs {
  readonly seq: number;
  readonly mode: ActiveRangeModeTs;
  readonly activeBankIdx: number;
  readonly flags: number;
  readonly rHatM: number;
  readonly sigmaRM: number;
  readonly rSafeLowerM: number;
  readonly rhoDotInvS: number;
  readonly cmdBankRad: number;
  readonly minSc3bfMargin: number;
  readonly fimGain: number;
  readonly weights: readonly [number, number, number, number, number, number];
  readonly crc32: number;
}

// IEEE-802.3 CRC32 lookup table (polynomial 0xEDB88320)
const CRC32_TABLE = new Uint32Array(256);
for (let i = 0; i < 256; i++) {
  let c = i;
  for (let b = 0; b < 8; b++) {
    c = (c & 1) !== 0 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  }
  CRC32_TABLE[i] = c >>> 0;
}

export function ieeeCrc32(bytes: Uint8Array, length = bytes.byteLength): number {
  let crc = 0xffffffff;
  for (let i = 0; i < length; i++) {
    crc = CRC32_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

export function bytesToHex(bytes: Uint8Array): string {
  let s = "";
  for (let i = 0; i < bytes.byteLength; i++) {
    s += bytes[i].toString(16).padStart(2, "0");
  }
  return s;
}

function clampF64(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function wrapPi(angleRad: number): number {
  if (!Number.isFinite(angleRad)) return 0.0;
  const twoPi = 2.0 * Math.PI;
  let a = (angleRad + Math.PI) % twoPi;
  if (a < 0.0) a += twoPi;
  return a - Math.PI;
}

export function computeConformalLowerBound(
  rHat: number,
  sigmaR: number,
  qHat: number,
): number {
  if (!Number.isFinite(rHat)) return R_FLOOR_M;
  const rClamped = Math.min(R_CEIL_M, Math.max(R_FLOOR_M, rHat));
  const sigEff = !Number.isFinite(sigmaR) || sigmaR < 0.0 ? SIGMA_FLOOR_M : Math.max(SIGMA_FLOOR_M, sigmaR);
  const qEff = !Number.isFinite(qHat) || qHat < 0.0 ? 0.0 : qHat;
  const lower = rClamped - qEff * sigEff;
  if (!Number.isFinite(lower)) return R_FLOOR_M;
  return Math.min(rClamped, Math.max(R_FLOOR_M, lower));
}

export function encodeWire(pkt: ActiveRangeTelemetryTs): Uint8Array {
  const buf = new Uint8Array(WIRE_SIZE_BYTES);
  const dv = new DataView(buf.buffer);
  dv.setUint32(0, pkt.seq >>> 0, true);
  dv.setUint8(4, pkt.mode & 0xff);
  dv.setUint8(5, pkt.activeBankIdx & 0xff);
  dv.setUint16(6, pkt.flags & 0xffff, true);
  dv.setFloat32(8, pkt.rHatM, true);
  dv.setFloat32(12, pkt.sigmaRM, true);
  dv.setFloat32(16, pkt.rSafeLowerM, true);
  dv.setFloat32(20, pkt.rhoDotInvS, true);
  dv.setFloat32(24, pkt.cmdBankRad, true);
  dv.setFloat32(28, pkt.minSc3bfMargin, true);
  dv.setFloat32(32, pkt.fimGain, true);
  for (let m = 0; m < NUM_BANKS; m++) {
    dv.setFloat32(36 + m * 4, pkt.weights[m], true);
  }
  const crc = ieeeCrc32(buf, 60);
  dv.setUint32(60, crc, true);
  return buf;
}

export function decodeWire(raw: Uint8Array): ActiveRangeTelemetryTs | null {
  if (raw.byteLength !== WIRE_SIZE_BYTES) return null;
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const crc = dv.getUint32(60, true);
  if (crc !== ieeeCrc32(raw, 60)) return null;

  const seq = dv.getUint32(0, true);
  const modeRaw = dv.getUint8(4);
  const activeBankIdx = dv.getUint8(5);
  const flags = dv.getUint16(6, true);
  if (modeRaw > 3 || activeBankIdx >= NUM_BANKS) return null;

  const f7 = [
    dv.getFloat32(8, true),
    dv.getFloat32(12, true),
    dv.getFloat32(16, true),
    dv.getFloat32(20, true),
    dv.getFloat32(24, true),
    dv.getFloat32(28, true),
    dv.getFloat32(32, true),
  ];
  const w6: [number, number, number, number, number, number] = [
    dv.getFloat32(36, true),
    dv.getFloat32(40, true),
    dv.getFloat32(44, true),
    dv.getFloat32(48, true),
    dv.getFloat32(52, true),
    dv.getFloat32(56, true),
  ];
  if (!f7.every(Number.isFinite) || !w6.every(Number.isFinite)) return null;

  return {
    seq,
    mode: modeRaw as ActiveRangeModeTs,
    activeBankIdx,
    flags,
    rHatM: f7[0],
    sigmaRM: f7[1],
    rSafeLowerM: f7[2],
    rhoDotInvS: f7[3],
    cmdBankRad: f7[4],
    minSc3bfMargin: f7[5],
    fimGain: f7[6],
    weights: w6,
    crc32: crc,
  };
}

function projectAccelToLos(
  aNed: readonly [number, number, number],
  alpha: number,
  beta: number,
): readonly [number, number, number] {
  const sa = Math.sin(alpha);
  const ca = Math.cos(alpha);
  const sb = Math.sin(beta);
  const cb = Math.cos(beta);
  const aLos = aNed[0] * (cb * ca) + aNed[1] * (cb * sa) + aNed[2] * -sb;
  const aAz = aNed[0] * -sa + aNed[1] * ca;
  const aEl = aNed[0] * (-sb * ca) + aNed[1] * (-sb * sa) + aNed[2] * -cb;
  return [aLos, aAz, aEl];
}

function mul6x6(a: Float64Array, b: Float64Array): Float64Array {
  const out = new Float64Array(36);
  for (let r = 0; r < 6; r++) {
    const rOff = r * 6;
    for (let c = 0; c < 6; c++) {
      let s = 0.0;
      for (let k = 0; k < 6; k++) {
        s += a[rOff + k] * b[k * 6 + c];
      }
      out[rOff + c] = s;
    }
  }
  return out;
}

function mul6x6T(a: Float64Array, b: Float64Array): Float64Array {
  const out = new Float64Array(36);
  for (let r = 0; r < 6; r++) {
    const rOff = r * 6;
    for (let c = 0; c < 6; c++) {
      const cOff = c * 6;
      let s = 0.0;
      for (let k = 0; k < 6; k++) {
        s += a[rOff + k] * b[cOff + k];
      }
      out[rOff + c] = s;
    }
  }
  return out;
}

function normalizeWeightsAndMoments(
  logLik: readonly number[],
  rBanks: readonly number[],
  varRBanks: readonly number[],
  prevW: readonly number[],
): {
  readonly weights: [number, number, number, number, number, number];
  readonly rHat: number;
  readonly sigmaR: number;
} {
  let maxRawLl = -1.0e300;
  for (let m = 0; m < NUM_BANKS; m++) {
    const llv = logLik[m];
    if (Number.isFinite(llv) && llv > maxRawLl) maxRawLl = llv;
  }

  const logW = new Float64Array(NUM_BANKS);
  let maxLog = -1.0e300;
  for (let m = 0; m < NUM_BANKS; m++) {
    const pw = prevW[m];
    const wp = Number.isFinite(pw) && pw >= W_FLOOR ? (pw > 1.0 ? 1.0 : pw) : W_FLOOR;
    const llRaw = logLik[m];
    const ll = Number.isFinite(llRaw) && maxRawLl > -1.0e299 ? clampF64(llRaw - maxRawLl, -25.0, 0.0) : -25.0;
    const lw = Math.log(wp) + ll;
    logW[m] = lw;
    if (lw > maxLog) maxLog = lw;
  }

  const unnorm = new Float64Array(NUM_BANKS);
  let sumUnnorm = 0.0;
  for (let m = 0; m < NUM_BANKS; m++) {
    const diff = clampF64(logW[m] - maxLog, -50.0, 0.0);
    const u = Math.exp(diff);
    unnorm[m] = u;
    sumUnnorm += u;
  }

  const scale = 1.0 - NUM_BANKS * W_FLOOR;
  const wOut: [number, number, number, number, number, number] = [
    1 / 6,
    1 / 6,
    1 / 6,
    1 / 6,
    1 / 6,
    1 / 6,
  ];
  if (Number.isFinite(sumUnnorm) && sumUnnorm > 0.0) {
    for (let m = 0; m < NUM_BANKS; m++) {
      wOut[m] = scale * (unnorm[m] / sumUnnorm) + W_FLOOR;
    }
  }

  const rClean = new Float64Array(NUM_BANKS);
  const vClean = new Float64Array(NUM_BANKS);
  let rHatRaw = 0.0;
  for (let m = 0; m < NUM_BANKS; m++) {
    const rmRaw = rBanks[m];
    const rm = Number.isFinite(rmRaw) ? clampF64(rmRaw, R_FLOOR_M, R_CEIL_M) : R_INIT_M[m];
    const vmRaw = varRBanks[m];
    const vm = Number.isFinite(vmRaw) && vmRaw >= 0.0 ? clampF64(vmRaw, 0.0, 1.0e8) : (0.28 * rm) ** 2;
    rClean[m] = rm;
    vClean[m] = vm;
    rHatRaw += wOut[m] * rm;
  }

  const rHat = clampF64(rHatRaw, R_FLOOR_M, R_CEIL_M);
  let varTot = 0.0;
  for (let m = 0; m < NUM_BANKS; m++) {
    const dr = rClean[m] - rHat;
    varTot += wOut[m] * (vClean[m] + dr * dr);
  }
  const sigmaR = Number.isFinite(varTot) && varTot > 0.0 ? Math.sqrt(varTot) : 0.0;
  return { weights: wOut, rHat, sigmaR };
}

class MscBankTs {
  readonly x: Float64Array = new Float64Array(6);
  p: Float64Array = new Float64Array(36);

  constructor(alphaRad: number, betaRad: number, rInitM: number) {
    const r0 = clampF64(rInitM, R_FLOOR_M, R_CEIL_M);
    const rho0 = 1.0 / r0;
    this.x[0] = alphaRad;
    this.x[1] = betaRad;
    this.x[4] = rho0;
    this.p[0] = 0.002 * 0.002;
    this.p[7] = 0.002 * 0.002;
    this.p[14] = 0.0025 * 0.0025;
    this.p[21] = 0.0025 * 0.0025;
    const sigRho = 0.28 * rho0;
    this.p[28] = sigRho * sigRho;
    this.p[35] = 0.15 * 0.15;
  }

  rangeAndVariance(): readonly [number, number] {
    const rho = clampF64(this.x[4], 1.0 / R_CEIL_M, 1.0 / R_FLOOR_M);
    const rM = clampF64(1.0 / rho, R_FLOOR_M, R_CEIL_M);
    const rho2 = rho * rho;
    const rho4 = rho2 * rho2;
    const p44 = this.p[28];
    const rawVar = Number.isFinite(p44) && p44 >= 0.0 ? p44 / rho4 : (0.28 * rM) ** 2;
    const varR = clampF64(rawVar, 1.0, (0.5 * rM) ** 2);
    return [rM, varR];
  }

  private static rhs(x: Float64Array, aNed: readonly [number, number, number]): Float64Array {
    const alpha = x[0];
    const beta = clampF64(x[1], -1.35, 1.35);
    const aDot = x[2];
    const bDot = x[3];
    const rho = clampF64(x[4], 1.0 / R_CEIL_M, 1.0 / R_FLOOR_M);
    const rhoDot = clampF64(x[5], -2.0, 2.0);
    const [aLos, aAz, aEl] = projectAccelToLos(aNed, alpha, beta);
    const sb = Math.sin(beta);
    const cb = Math.cos(beta);
    const cbSafe = cb > 0.15 ? cb : 0.15;
    const tb = sb / cbSafe;

    const out = new Float64Array(6);
    out[0] = aDot;
    out[1] = bDot;
    out[2] = -2.0 * aDot * rhoDot + 2.0 * aDot * bDot * tb - (rho / cbSafe) * aAz;
    out[3] = -2.0 * bDot * rhoDot - aDot * aDot * sb * cb - rho * aEl;
    out[4] = -rho * rhoDot;
    out[5] = aDot * aDot * cb * cb + bDot * bDot - rhoDot * rhoDot - rho * aLos;
    return out;
  }

  private static jacobian(x: Float64Array, aNed: readonly [number, number, number]): Float64Array {
    const alpha = x[0];
    const beta = clampF64(x[1], -1.35, 1.35);
    const aDot = x[2];
    const bDot = x[3];
    const rho = clampF64(x[4], 1.0 / R_CEIL_M, 1.0 / R_FLOOR_M);
    const rhoDot = clampF64(x[5], -2.0, 2.0);
    const [aLos, aAz, aEl] = projectAccelToLos(aNed, alpha, beta);
    const sb = Math.sin(beta);
    const cb = Math.cos(beta);
    const cbSafe = cb > 0.15 ? cb : 0.15;
    const tb = sb / cbSafe;
    const sec2 = 1.0 / (cbSafe * cbSafe);

    const j = new Float64Array(36);
    j[0 * 6 + 2] = 1.0;
    j[1 * 6 + 3] = 1.0;
    j[2 * 6 + 1] = 2.0 * aDot * bDot * sec2 - rho * aAz * sb * sec2;
    j[2 * 6 + 2] = -2.0 * rhoDot + 2.0 * bDot * tb;
    j[2 * 6 + 3] = 2.0 * aDot * tb;
    j[2 * 6 + 4] = -aAz / cbSafe;
    j[2 * 6 + 5] = -2.0 * aDot;
    j[3 * 6 + 1] = -aDot * aDot * (cb * cb - sb * sb);
    j[3 * 6 + 2] = -2.0 * aDot * sb * cb;
    j[3 * 6 + 3] = -2.0 * rhoDot;
    j[3 * 6 + 4] = -aEl;
    j[3 * 6 + 5] = -2.0 * bDot;
    j[4 * 6 + 4] = -rhoDot;
    j[4 * 6 + 5] = -rho;
    j[5 * 6 + 1] = -2.0 * aDot * aDot * cb * sb;
    j[5 * 6 + 2] = 2.0 * aDot * cb * cb;
    j[5 * 6 + 3] = 2.0 * bDot;
    j[5 * 6 + 4] = -aLos;
    j[5 * 6 + 5] = -2.0 * rhoDot;
    return j;
  }

  propagate(aNed: readonly [number, number, number], dtS: number): void {
    const dt = clampF64(dtS, 1.0e-3, 0.5);
    const k1 = MscBankTs.rhs(this.x, aNed);
    const xMid = new Float64Array(6);
    for (let i = 0; i < 6; i++) xMid[i] = this.x[i] + 0.5 * dt * k1[i];
    xMid[0] = wrapPi(xMid[0]);
    xMid[1] = clampF64(xMid[1], -1.35, 1.35);
    xMid[4] = clampF64(xMid[4], 1.0 / R_CEIL_M, 1.0 / R_FLOOR_M);
    xMid[5] = clampF64(xMid[5], -2.0, 2.0);

    const k2 = MscBankTs.rhs(xMid, aNed);
    for (let i = 0; i < 6; i++) this.x[i] = this.x[i] + dt * k2[i];
    this.x[0] = wrapPi(this.x[0]);
    this.x[1] = clampF64(this.x[1], -1.35, 1.35);
    this.x[4] = clampF64(this.x[4], 1.0 / R_CEIL_M, 1.0 / R_FLOOR_M);
    this.x[5] = clampF64(this.x[5], -2.0, 2.0);

    const j = MscBankTs.jacobian(xMid, aNed);
    const j2 = mul6x6(j, j);
    const fMat = new Float64Array(36);
    const halfDt2 = 0.5 * dt * dt;
    for (let i = 0; i < 36; i++) fMat[i] = dt * j[i] + halfDt2 * j2[i];
    for (let d = 0; d < 6; d++) fMat[d * 6 + d] += 1.0;

    const pNext = mul6x6T(mul6x6(fMat, this.p), fMat);
    const qAngAcc = 0.0003 * 0.0003;
    const qAngPos = 0.25 * qAngAcc * dt ** 4 + 1.0e-10 * dt;
    const qAngVel = qAngAcc * dt ** 2 + 1.0e-9 * dt;
    const qRho = (1.0e-6 * dt) ** 2;
    const qRhodot = (0.01 * dt) ** 2 + 1.0e-6 * dt;

    pNext[0] += qAngPos;
    pNext[7] += qAngPos;
    pNext[14] += qAngVel;
    pNext[21] += qAngVel;
    pNext[28] += qRho;
    pNext[35] += qRhodot;

    for (let r = 0; r < 6; r++) {
      for (let c = r; c < 6; c++) {
        const avg = 0.5 * (pNext[r * 6 + c] + pNext[c * 6 + r]);
        const clean = Number.isFinite(avg) ? avg : 0.0;
        pNext[r * 6 + c] = clean;
        pNext[c * 6 + r] = clean;
      }
      if (pNext[r * 6 + r] < 1.0e-14) pNext[r * 6 + r] = 1.0e-14;
    }
    this.p = pNext;
  }

  updateBearing(alphaMeas: number, betaMeas: number, sigmaBearingRad: number): number {
    const rVar = sigmaBearingRad * sigmaBearingRad;
    const nu0 = wrapPi(alphaMeas - this.x[0]);
    const nu1 = clampF64(betaMeas, -1.35, 1.35) - this.x[1];

    const s00 = this.p[0] + rVar;
    const s01 = this.p[1];
    const s11 = this.p[7] + rVar;
    const detS = s00 * s11 - s01 * s01;
    if (!Number.isFinite(detS) || detS <= 1.0e-20) return -25.0;

    const invDet = 1.0 / detS;
    const inv00 = s11 * invDet;
    const inv01 = -s01 * invDet;
    const inv11 = s00 * invDet;

    const maha = nu0 * (inv00 * nu0 + inv01 * nu1) + nu1 * (inv01 * nu0 + inv11 * nu1);
    const logLik = -0.5 * (maha + Math.log(detS));

    const k0 = new Float64Array(6);
    const k1 = new Float64Array(6);
    for (let r = 0; r < 6; r++) {
      k0[r] = this.p[r * 6 + 0] * inv00 + this.p[r * 6 + 1] * inv01;
      k1[r] = this.p[r * 6 + 0] * inv01 + this.p[r * 6 + 1] * inv11;
      this.x[r] += k0[r] * nu0 + k1[r] * nu1;
    }
    this.x[0] = wrapPi(this.x[0]);
    this.x[1] = clampF64(this.x[1], -1.35, 1.35);
    this.x[4] = clampF64(this.x[4], 1.0 / R_CEIL_M, 1.0 / R_FLOOR_M);
    this.x[5] = clampF64(this.x[5], -2.0, 2.0);

    const ikh = new Float64Array(36);
    for (let d = 0; d < 6; d++) ikh[d * 6 + d] = 1.0;
    for (let r = 0; r < 6; r++) {
      ikh[r * 6 + 0] -= k0[r];
      ikh[r * 6 + 1] -= k1[r];
    }
    const pNew = mul6x6T(mul6x6(ikh, this.p), ikh);
    for (let r = 0; r < 6; r++) {
      for (let c = r; c < 6; c++) {
        const sym = 0.5 * (pNew[r * 6 + c] + pNew[c * 6 + r]) + rVar * (k0[r] * k0[c] + k1[r] * k1[c]);
        const clean = Number.isFinite(sym) ? sym : 0.0;
        pNew[r * 6 + c] = clean;
        pNew[c * 6 + r] = clean;
      }
      if (pNew[r * 6 + r] < 1.0e-14) pNew[r * 6 + r] = 1.0e-14;
    }
    this.p = pNew;
    return Number.isFinite(logLik) ? logLik : -1.0e6;
  }

  updateTau(zTau: number, rTau: number): void {
    const nu = zTau - this.x[5];
    const s = this.p[35] + rTau;
    if (!Number.isFinite(s) || s <= 1.0e-16) return;

    const k = new Float64Array(6);
    for (let r = 0; r < 6; r++) {
      k[r] = this.p[r * 6 + 5] / s;
      this.x[r] += k[r] * nu;
    }
    this.x[0] = wrapPi(this.x[0]);
    this.x[1] = clampF64(this.x[1], -1.35, 1.35);
    this.x[4] = clampF64(this.x[4], 1.0 / R_CEIL_M, 1.0 / R_FLOOR_M);
    this.x[5] = clampF64(this.x[5], -2.0, 2.0);

    const ikh = new Float64Array(36);
    for (let d = 0; d < 6; d++) ikh[d * 6 + d] = 1.0;
    for (let r = 0; r < 6; r++) ikh[r * 6 + 5] -= k[r];
    const pNew = mul6x6T(mul6x6(ikh, this.p), ikh);
    for (let r = 0; r < 6; r++) {
      for (let c = r; c < 6; c++) {
        const sym = 0.5 * (pNew[r * 6 + c] + pNew[c * 6 + r]) + rTau * k[r] * k[c];
        const clean = Number.isFinite(sym) ? sym : 0.0;
        pNew[r * 6 + c] = clean;
        pNew[c * 6 + r] = clean;
      }
      if (pNew[r * 6 + r] < 1.0e-14) pNew[r * 6 + r] = 1.0e-14;
    }
    this.p = pNew;
  }
}

export interface SwerveDecisionTs {
  readonly mode: ActiveRangeModeTs;
  readonly cmdBankRad: number;
  readonly minSc3bfMargin: number;
  readonly fimGain: number;
  readonly sc3bfAllSafe: boolean;
  readonly fovSafe: boolean;
  readonly nowLocked: boolean;
}

export function selectActiveSwerveTs(args: {
  readonly rBanks: readonly number[];
  readonly weights: readonly number[];
  readonly rHatM: number;
  readonly sigmaRM: number;
  readonly rSafeLowerM: number;
  readonly rhoDotInvS: number;
  readonly alphaDot: number;
  readonly betaDot: number;
  readonly alphaBodyRad: number;
  readonly alphaRad: number;
  readonly betaRad: number;
  readonly headingRad: number;
  readonly tasMps: number;
  readonly vStallMps: number;
  readonly sigmaBearingRad: number;
  readonly cbdrProbeActive: boolean;
  readonly wasLocked: boolean;
  readonly preferredSwerveSign: number;
}): SwerveDecisionTs {
  const sa = Math.sin(args.alphaRad);
  const ca = Math.cos(args.alphaRad);
  const sb = Math.sin(args.betaRad);
  const cb = Math.cos(args.betaRad);
  const uR: readonly [number, number, number] = [cb * ca, cb * sa, -sb];
  const uA: readonly [number, number, number] = [-sa, ca, 0.0];
  const uB: readonly [number, number, number] = [-sb * ca, -sb * sa, -cb];

  const sh = Math.sin(args.headingRad);
  const ch = Math.cos(args.headingRad);
  const ownFwd: readonly [number, number, number] = [ch, sh, 0.0];
  const ownRight: readonly [number, number, number] = [-sh, ch, 0.0];

  const deltaVOwn = (tanPhi: number): readonly [number, number, number] => {
    const vEff = args.tasMps > 1.0 ? args.tasMps : 1.0;
    const dpsi = clampF64((G_ACCEL_MPS2 * tanPhi / vEff) * DT_LOOK_S, -0.5, 0.5);
    const dpsi2 = dpsi * dpsi;
    const cMinus1 = -0.5 * dpsi2 + (dpsi2 * dpsi2) / 24.0;
    const sApprox = dpsi - (dpsi2 * dpsi) / 6.0;
    return [
      vEff * (cMinus1 * ownFwd[0] + sApprox * ownRight[0]),
      vEff * (cMinus1 * ownFwd[1] + sApprox * ownRight[1]),
      vEff * (cMinus1 * ownFwd[2] + sApprox * ownRight[2]),
    ];
  };

  const sc3bfCosThetaC = (rH: number): number => {
    if (rH <= R_SEP_M) return 0.0;
    const denom = rH > R_SEP_M + 1.0 ? rH : R_SEP_M + 1.0;
    const ratio = R_SEP_M / denom;
    const arg = 1.0 - ratio * ratio;
    return arg > 0.0 ? Math.sqrt(arg) : 0.0;
  };

  const sc3bfEta = (rH: number, vRelNorm: number): number => {
    const excess = rH > R_SEP_M ? rH - R_SEP_M : 0.0;
    const vEff = vRelNorm > 1.0 ? vRelNorm : 1.0;
    return (K_ALLOW * excess * excess * A_MAX_EVADE_MPS2) / vEff;
  };

  const evalH = (rH: number, tanPhi: number): number => {
    const r = rH > R_FLOOR_M ? rH : R_FLOOR_M;
    const p0 = r * uR[0];
    const p1 = r * uR[1];
    const p2 = r * uR[2];
    const vR = r * args.rhoDotInvS;
    const vA = r * args.alphaDot * cb;
    const vB = r * args.betaDot;
    const dv = deltaVOwn(tanPhi);
    const vr0 = vR * uR[0] + vA * uA[0] + vB * uB[0] - dv[0];
    const vr1 = vR * uR[1] + vA * uA[1] + vB * uB[1] - dv[1];
    const vr2 = vR * uR[2] + vA * uA[2] + vB * uB[2] - dv[2];
    const vNorm = Math.sqrt(vr0 * vr0 + vr1 * vr1 + vr2 * vr2);
    const pv = p0 * vr0 + p1 * vr1 + p2 * vr2;
    return pv + r * vNorm * sc3bfCosThetaC(r) + sc3bfEta(r, vNorm);
  };

  const hypMarginSafe = (rH: number, tanPhi: number): readonly [number, boolean] => {
    const h0 = evalH(rH, 0.0);
    const hPhi = evalH(rH, tanPhi);
    if (h0 >= 0.0) {
      const margin = hPhi - (1.0 - ALPHA_CBF * DT_LOOK_S) * h0;
      return [margin, Number.isFinite(margin) && margin >= 0.0];
    }
    // D16-P1-M1: inside a violated collision cone (h0 < 0), require strict healing (hPhi > h0)
    // so straight flight (phi = 0, margin == 0) is NEVER falsely marked safe.
    const margin = hPhi - h0;
    return [margin, Number.isFinite(margin) && margin > 0.0];
  };

  const fovSafe = (tanPhi: number): boolean => {
    const vEff = args.tasMps > 1.0 ? args.tasMps : 1.0;
    const dpsi = clampF64((G_ACCEL_MPS2 * tanPhi / vEff) * DT_LOOK_S, -0.5, 0.5);
    const predAb = args.alphaBodyRad - dpsi + args.alphaDot * DT_LOOK_S;
    return Number.isFinite(predAb) && Math.abs(predAb) <= FOV_LIMIT_RAD;
  };

  const universal = (tanPhi: number): readonly [number, boolean] => {
    let minM = 1.0e300;
    let allOk = true;
    for (let m = 0; m < NUM_BANKS; m++) {
      if (args.weights[m] >= W_ACTIVE) {
        const [mg, ok] = hypMarginSafe(args.rBanks[m], tanPhi);
        if (mg < minM) minM = mg;
        if (!ok) allOk = false;
      }
    }
    const [mgLow, okLow] = hypMarginSafe(args.rSafeLowerM, tanPhi);
    if (mgLow < minM) minM = mgLow;
    if (!okLow) allOk = false;
    if (!Number.isFinite(minM)) return [-1.0e6, false];
    return [minM, allOk];
  };

  const evadeBank = args.alphaBodyRad > 0.01 ? -EVADE_BANK_RAD : EVADE_BANK_RAD;
  const rHatEff = args.rHatM > R_FLOOR_M ? args.rHatM : R_FLOOR_M;
  const relSigma = Number.isFinite(args.sigmaRM) && args.sigmaRM >= 0.0 ? args.sigmaRM / rHatEff : 1.0;
  const nowLocked = args.wasLocked ? relSigma <= TRIGGER_REL_SIGMA : relSigma <= LOCK_REL_SIGMA;

  const isClosing = args.rhoDotInvS < CLOSING_RHO_DOT_THRESH;
  const tauTerminal = args.rhoDotInvS <= -(1.0 / TAU_EVADE_TRIGGER_S);
  const hLowZero = evalH(args.rSafeLowerM, 0.0);
  const rangeTerminal =
    args.rHatM <= R_EVADE_TRIGGER_M || (args.rSafeLowerM <= R_EVADE_TRIGGER_M && relSigma <= 0.35);

  if (isClosing && (tauTerminal || rangeTerminal) && hLowZero < 0.0) {
    const [marg0] = universal(0.0);
    return {
      mode: ActiveRangeModeTs.ImmediateEvade,
      cmdBankRad: evadeBank,
      minSc3bfMargin: marg0,
      fimGain: 0.0,
      sc3bfAllSafe: false,
      fovSafe: fovSafe(0.0),
      nowLocked,
    };
  }

  const envelopeOk =
    Number.isFinite(args.tasMps) && Number.isFinite(args.vStallMps) && args.tasMps >= 1.25 * args.vStallMps;
  const [marg0, safe0] = universal(0.0);
  const fov0 = fovSafe(0.0);

  if (!(isClosing || args.cbdrProbeActive)) {
    return {
      mode: nowLocked ? ActiveRangeModeTs.Locked : ActiveRangeModeTs.NominalHold,
      cmdBankRad: 0.0,
      minSc3bfMargin: marg0,
      fimGain: 0.0,
      sc3bfAllSafe: safe0,
      fovSafe: fov0,
      nowLocked,
    };
  }

  if (nowLocked) {
    return {
      mode: ActiveRangeModeTs.Locked,
      cmdBankRad: 0.0,
      minSc3bfMargin: marg0,
      fimGain: 0.0,
      sc3bfAllSafe: safe0,
      fovSafe: fov0,
      nowLocked: true,
    };
  }

  if (!envelopeOk) {
    return {
      mode: ActiveRangeModeTs.NominalHold,
      cmdBankRad: 0.0,
      minSc3bfMargin: marg0,
      fimGain: 0.0,
      sc3bfAllSafe: safe0,
      fovSafe: fov0,
      nowLocked: false,
    };
  }

  const sigA = args.sigmaBearingRad > 1.0e-5 ? args.sigmaBearingRad : 1.0e-5;
  const t4Over12s2 = (2.5 ** 4) / (12.0 * sigA * sigA);
  const lambdaU = 1.0e-2;
  const lambdaMargin = 1.0e-4;
  const rightDotUalpha = ownRight[0] * uA[0] + ownRight[1] * uA[1] + ownRight[2] * uA[2];

  let foundSwerve = false;
  let bestK = 4;
  let bestScore = -1.0e300;
  let bestMargin = marg0;
  let bestFim = 0.0;

  for (let k = 0; k < PHI_CAND_RAD.length; k++) {
    const phi = PHI_CAND_RAD[k];
    const tanPhi = PHI_CAND_TAN[k];
    const fovOk = fovSafe(tanPhi);
    const [minMarg, allSafe] = universal(tanPhi);
    if (fovOk && allSafe) {
      const aAzCand = G_ACCEL_MPS2 * tanPhi * rightDotUalpha;
      const aAz2 = aAzCand * aAzCand;
      let fimGain = 0.0;
      for (let m = 0; m < NUM_BANKS; m++) {
        const rm = args.rBanks[m] > R_FLOOR_M ? args.rBanks[m] : R_FLOOR_M;
        fimGain += args.weights[m] * (aAz2 / (rm * rm)) * t4Over12s2;
      }
      const prefSign = args.preferredSwerveSign < 0.0 ? -1.0 : 1.0;
      const rowTiebreak =
        Math.abs(args.alphaBodyRad) < 0.07
          ? 1.0e-3 * prefSign * phi
          : args.alphaBodyRad > 0.0
            ? -1.0e-3 * phi
            : 1.0e-3 * phi;
      const score = fimGain - lambdaU * phi * phi + lambdaMargin * minMarg + rowTiebreak;
      if (Number.isFinite(score) && (!foundSwerve || score > bestScore)) {
        foundSwerve = true;
        bestK = k;
        bestScore = score;
        bestMargin = minMarg;
        bestFim = fimGain;
      }
    }
  }

  if (!foundSwerve) {
    return {
      mode: ActiveRangeModeTs.ImmediateEvade,
      cmdBankRad: evadeBank,
      minSc3bfMargin: marg0,
      fimGain: 0.0,
      sc3bfAllSafe: false,
      fovSafe: fov0,
      nowLocked: false,
    };
  }

  if (bestK === 4) {
    return {
      mode: ActiveRangeModeTs.NominalHold,
      cmdBankRad: 0.0,
      minSc3bfMargin: bestMargin,
      fimGain: bestFim,
      sc3bfAllSafe: true,
      fovSafe: true,
      nowLocked: false,
    };
  }

  return {
    mode: ActiveRangeModeTs.ActiveSwerve,
    cmdBankRad: PHI_CAND_RAD[bestK],
    minSc3bfMargin: bestMargin,
    fimGain: bestFim,
    sc3bfAllSafe: true,
    fovSafe: true,
    nowLocked: false,
  };
}

export class ActiveRangeControllerTs {
  private readonly rInitM: readonly [number, number, number, number, number, number];
  private readonly sigmaBearingRad: number;
  private readonly qHatConformal: number;
  private initialized = false;
  private banks: MscBankTs[];
  private weights: [number, number, number, number, number, number] = [
    1 / 6,
    1 / 6,
    1 / 6,
    1 / 6,
    1 / 6,
    1 / 6,
  ];
  private seq = 0;
  private swerveTicks = 0;
  private wasLocked = false;

  constructor(cfg: ActiveRangeConfigTs = {}) {
    this.rInitM = cfg.rInitM ?? R_INIT_M;
    this.sigmaBearingRad = cfg.sigmaBearingRad ?? 0.00055;
    this.qHatConformal = cfg.qHatConformal ?? 1.65;
    this.banks = Array.from({ length: NUM_BANKS }, (_, m) => new MscBankTs(0.0, 0.0, this.rInitM[m]));
  }

  reset(): void {
    this.initialized = false;
    this.banks = Array.from({ length: NUM_BANKS }, (_, m) => new MscBankTs(0.0, 0.0, this.rInitM[m]));
    this.weights = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6];
    this.seq = 0;
    this.swerveTicks = 0;
    this.wasLocked = false;
  }

  private currentMoments(): {
    readonly rHatM: number;
    readonly sigmaRM: number;
    readonly activeIdx: number;
    readonly rhoDot: number;
  } {
    const rBanks = new Float64Array(NUM_BANKS);
    const vBanks = new Float64Array(NUM_BANKS);
    let rHat = 0.0;
    let rhoDot = 0.0;
    let activeIdx = 0;
    let maxW = -1.0;
    for (let m = 0; m < NUM_BANKS; m++) {
      const [rm, vm] = this.banks[m].rangeAndVariance();
      rBanks[m] = rm;
      vBanks[m] = vm;
      const wm = this.weights[m];
      if (wm > maxW) {
        maxW = wm;
        activeIdx = m;
      }
      rHat += wm * rm;
      rhoDot += wm * this.banks[m].x[5];
    }
    rHat = clampF64(rHat, R_FLOOR_M, R_CEIL_M);
    let varTot = 0.0;
    for (let m = 0; m < NUM_BANKS; m++) {
      const dr = rBanks[m] - rHat;
      varTot += this.weights[m] * (vBanks[m] + dr * dr);
    }
    const sigmaRM = Number.isFinite(varTot) && varTot > 0.0 ? Math.sqrt(varTot) : 0.0;
    return { rHatM: rHat, sigmaRM, activeIdx, rhoDot: clampF64(rhoDot, -2.0, 2.0) };
  }

  private packTelemetry(fields: {
    readonly mode: ActiveRangeModeTs;
    readonly activeBankIdx: number;
    readonly flags: number;
    readonly rHatM: number;
    readonly sigmaRM: number;
    readonly rSafeLowerM: number;
    readonly rhoDotInvS: number;
    readonly cmdBankRad: number;
    readonly minSc3bfMargin: number;
    readonly fimGain: number;
    readonly weights: readonly [number, number, number, number, number, number];
  }): ActiveRangeTelemetryTs {
    const w32: [number, number, number, number, number, number] = [
      Math.fround(fields.weights[0]),
      Math.fround(fields.weights[1]),
      Math.fround(fields.weights[2]),
      Math.fround(fields.weights[3]),
      Math.fround(fields.weights[4]),
      Math.fround(fields.weights[5]),
    ];
    const draft: ActiveRangeTelemetryTs = {
      seq: this.seq,
      mode: fields.mode,
      activeBankIdx: fields.activeBankIdx,
      flags: fields.flags,
      rHatM: Math.fround(fields.rHatM),
      sigmaRM: Math.fround(fields.sigmaRM),
      rSafeLowerM: Math.fround(fields.rSafeLowerM),
      rhoDotInvS: Math.fround(fields.rhoDotInvS),
      cmdBankRad: Math.fround(fields.cmdBankRad),
      minSc3bfMargin: Math.fround(fields.minSc3bfMargin),
      fimGain: Math.fround(fields.fimGain),
      weights: w32,
      crc32: 0,
    };
    const wire = encodeWire(draft);
    const crc = new DataView(wire.buffer).getUint32(60, true);
    return { ...draft, crc32: crc };
  }

  step(own: OwnshipStateNedTs, meas: ActiveRangeMeasurementTs): ActiveRangeTelemetryTs {
    this.seq = (this.seq + 1) >>> 0;
    const clean = (v: number, fb: number): number => (Number.isFinite(v) ? v : fb);
    const aNed: readonly [number, number, number] = [
      clampF64(clean(own.aNed[0], 0.0), -25.0, 25.0),
      clampF64(clean(own.aNed[1], 0.0), -25.0, 25.0),
      clampF64(clean(own.aNed[2], 0.0), -25.0, 25.0),
    ];
    const headingRad = wrapPi(clean(own.headingRad, 0.0));
    const tasMps = clampF64(clean(own.tasMps, 25.0), 1.0, 250.0);
    const vStallMps = clampF64(clean(own.vStallMps ?? 14.5, 14.5), 1.0, 100.0);

    const dtIn = meas.dtS ?? 0.1;
    const inputsValid =
      Number.isFinite(meas.alphaRad) &&
      Number.isFinite(meas.betaRad) &&
      Number.isFinite(dtIn) &&
      dtIn > 0.0;
    if (!inputsValid) {
      const m = this.currentMoments();
      const rSafe = computeConformalLowerBound(m.rHatM, m.sigmaRM, this.qHatConformal);
      return this.packTelemetry({
        mode: ActiveRangeModeTs.NominalHold,
        activeBankIdx: m.activeIdx,
        flags: 0,
        rHatM: m.rHatM,
        sigmaRM: m.sigmaRM,
        rSafeLowerM: rSafe,
        rhoDotInvS: m.rhoDot,
        cmdBankRad: 0.0,
        minSc3bfMargin: 0.0,
        fimGain: 0.0,
        weights: this.weights,
      });
    }

    const alphaM = wrapPi(meas.alphaRad);
    const betaM = clampF64(meas.betaRad, -1.35, 1.35);
    const dtS = clampF64(dtIn, 1.0e-3, 0.5);

    const logLik = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let flags = 0;

    if (!this.initialized) {
      for (let m = 0; m < NUM_BANKS; m++) {
        this.banks[m] = new MscBankTs(alphaM, betaM, this.rInitM[m]);
      }
      this.weights = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6];
      this.initialized = true;
    } else {
      let llMax = -1.0e300;
      for (let m = 0; m < NUM_BANKS; m++) {
        this.banks[m].propagate(aNed, dtS);
        logLik[m] = this.banks[m].updateBearing(alphaM, betaM, this.sigmaBearingRad);
        if (logLik[m] > llMax) llMax = logLik[m];
      }
      for (let m = 0; m < NUM_BANKS; m++) {
        logLik[m] = clampF64(logLik[m] - llMax, -25.0, 0.0);
      }
    }

    const trackAge = Math.trunc(meas.trackAge ?? 1);
    if (
      Number.isFinite(meas.sizePx) &&
      Number.isFinite(meas.sizeRatePxS) &&
      meas.sizePx >= TAU_MIN_SIZE_PX &&
      trackAge >= TAU_MIN_TRACK_AGE
    ) {
      const zTau = -meas.sizeRatePxS / meas.sizePx;
      if (Number.isFinite(zTau) && Math.abs(zTau) <= 1.0) {
        const s2 = meas.sizePx * meas.sizePx;
        const rTau = (2.0 * 0.64) / (s2 * 0.25) + 0.015 * 0.015;
        for (let m = 0; m < NUM_BANKS; m++) {
          this.banks[m].updateTau(zTau, rTau);
        }
        flags |= FLAG_TAU_FUSED;
      }
    }

    const [, aAz] = projectAccelToLos(aNed, alphaM, betaM);
    const skyBetaExt = meas.skyBetaExt ?? 0.0;
    const contrastRatio = meas.contrastRatio ?? 0.5;
    if (
      Boolean(meas.daytimeClear) &&
      Number.isFinite(meas.sizePx) &&
      meas.sizePx < KOSCH_MAX_SIZE_PX &&
      betaM > 0.0 &&
      Number.isFinite(skyBetaExt) &&
      skyBetaExt >= 5.0e-5 &&
      skyBetaExt <= 2.0e-3 &&
      Number.isFinite(contrastRatio) &&
      contrastRatio > 0.01 &&
      contrastRatio < 0.99 &&
      Math.abs(aAz) < 0.25
    ) {
      const maxDelta = KOSCH_MAX_LOG_DELTA * dtS;
      for (let m = 0; m < NUM_BANKS; m++) {
        const [rM] = this.banks[m].rangeAndVariance();
        const cPred = Math.exp(-skyBetaExt * rM);
        if (contrastRatio > cPred) {
          const diff = (contrastRatio - cPred) / 0.25;
          logLik[m] += clampF64(-0.5 * diff * diff * dtS, -maxDelta, 0.0);
        }
      }
      flags |= FLAG_KOSCH_FUSED;
    }

    const rBanksL = [0, 0, 0, 0, 0, 0];
    const varRL = [0, 0, 0, 0, 0, 0];
    for (let m = 0; m < NUM_BANKS; m++) {
      const [rm, vm] = this.banks[m].rangeAndVariance();
      rBanksL[m] = rm;
      varRL[m] = vm;
    }

    const norm = normalizeWeightsAndMoments(logLik, rBanksL, varRL, this.weights);
    this.weights = norm.weights;

    let activeIdx = 0;
    let maxW = -1.0;
    let rhoDotMix = 0.0;
    let aDotMix = 0.0;
    let bDotMix = 0.0;
    let alphaMixS = 0.0;
    let alphaMixC = 0.0;
    let betaMix = 0.0;
    for (let m = 0; m < NUM_BANKS; m++) {
      const wm = this.weights[m];
      if (wm > maxW) {
        maxW = wm;
        activeIdx = m;
      }
      rhoDotMix += wm * this.banks[m].x[5];
      aDotMix += wm * this.banks[m].x[2];
      bDotMix += wm * this.banks[m].x[3];
      alphaMixS += wm * Math.sin(this.banks[m].x[0]);
      alphaMixC += wm * Math.cos(this.banks[m].x[0]);
      betaMix += wm * this.banks[m].x[1];
    }

    const rhoDotInvS = clampF64(rhoDotMix, -2.0, 2.0);
    const alphaMix = Math.atan2(alphaMixS, alphaMixC);
    const betaMixClamped = clampF64(betaMix, -1.35, 1.35);
    const rSafeLowerM = computeConformalLowerBound(norm.rHat, norm.sigmaR, this.qHatConformal);
    const alphaBodyRad = wrapPi(alphaMix - headingRad);

    const cbdrProbeActive =
      trackAge >= TAU_MIN_TRACK_AGE &&
      Number.isFinite(meas.sizePx) &&
      meas.sizePx < TAU_MIN_SIZE_PX &&
      Number.isFinite(meas.sizeRatePxS) &&
      meas.sizeRatePxS > 0.0 &&
      Math.abs(aDotMix) < 0.004 &&
      Math.abs(bDotMix) < 0.004;

    const preferredSwerveSign = Math.floor(this.swerveTicks / 12) % 2 === 0 ? 1.0 : -1.0;

    const decision = selectActiveSwerveTs({
      rBanks: rBanksL,
      weights: this.weights,
      rHatM: norm.rHat,
      sigmaRM: norm.sigmaR,
      rSafeLowerM,
      rhoDotInvS,
      alphaDot: aDotMix,
      betaDot: bDotMix,
      alphaBodyRad,
      alphaRad: alphaMix,
      betaRad: betaMixClamped,
      headingRad,
      tasMps,
      vStallMps,
      sigmaBearingRad: this.sigmaBearingRad,
      cbdrProbeActive,
      wasLocked: this.wasLocked,
      preferredSwerveSign,
    });

    this.wasLocked = decision.nowLocked;
    if (decision.mode === ActiveRangeModeTs.ActiveSwerve) {
      this.swerveTicks = (this.swerveTicks + 1) >>> 0;
    } else {
      this.swerveTicks = 0;
    }

    if (decision.sc3bfAllSafe) flags |= FLAG_SC3BF_ALL_SAFE;
    if (decision.fovSafe) flags |= FLAG_FOV_SAFE;

    return this.packTelemetry({
      mode: decision.mode,
      activeBankIdx: activeIdx,
      flags,
      rHatM: norm.rHat,
      sigmaRM: norm.sigmaR,
      rSafeLowerM,
      rhoDotInvS,
      cmdBankRad: decision.cmdBankRad,
      minSc3bfMargin: decision.minSc3bfMargin,
      fimGain: decision.fimGain,
      weights: this.weights,
    });
  }
}
