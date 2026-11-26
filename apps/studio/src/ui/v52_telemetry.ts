/**
 * Pure, Fail-Closed v5.2 Avionics Telemetry & Formal Assurance Formatters for OpenXVS Studio v5.2.
 *
 * Mirrors frozen Rust `apps/studio/src-tauri/src/rust_engine.rs` v5.2.0 constants and semantics:
 *   - 12-Azimuth-Sector EO Optical Quality (`sector_q_eo`, `min_sector_q_eo`, `degraded_sector_mask`, thr = 0.50)
 *   - LWIR Hardware FFC (`lwir_ffc_imminent`, `lwir_ffc_in_progress`, alias `lwir_ffc_flag`) &
 *     O(N) histogram `p95-p05` Contrast-Collapse (`< 10.0 DN`, `lwir_contrast_collapsed`)
 *   - Dual-Tier TimestampSkewGuard (`<= 1.0 ms` FSYNC precision tier + `> 200.0 ms` hard breaker)
 *   - Zero-Allocation M-of-N (`3-of-5`) Bearing Track-Before-Confirm (`bearing_track_confirmed`,
 *     `bearing_track_hits_in_window`, optional alias `bearing_tbc_confirmed_tracks`)
 *   - Formal & Bare-Metal Assurance Badges: `26/26 Kani CBMC (5,228 checks)`, `Apalache N=2/N=3`,
 *     `Cortex-M7 thumbv7em (88 KB Flash / 0 B RAM)`, `Sequence-VPR (0 HMI)`
 *
 * Backward Compatibility & Fail-Closed Guarantee:
 *   - Absent optional v5.2 fields fall back gracefully to legacy v5.0 fields / standby indicators.
 *   - Present but malformed (`NaN`, `±Infinity`, out-of-range, negative, non-integer bitmask) fields
 *     strictly fail closed to `"alert"` / degraded rather than displaying misleading nominal telemetry.
 */

import type { NativeSensorHealthPayload } from "./bridge_client.ts";
import type { StudioLocale } from "./i18n.ts";

export type HudBadgeState = "nominal" | "caution" | "alert" | "standby";

export interface HudBadgeDisplay {
  readonly text: string;
  readonly title: string;
  readonly state: HudBadgeState;
}

export type EoSectorBand = "nominal" | "mid" | "caution" | "alert" | "standby";

export interface SectorOpticalHealthDisplay extends HudBadgeDisplay {
  readonly minSectorQ: number | null;
  readonly degradedCount: number;
  readonly degradedMask: number;
  readonly sectors: readonly (number | null)[];
  readonly sectorBands: readonly EoSectorBand[];
  readonly failedClosed: boolean;
  readonly legacyFallback: boolean;
}

export interface LwirFfcContrastDisplay extends HudBadgeDisplay {
  readonly ffcImminent: boolean;
  readonly ffcInProgress: boolean;
  readonly contrastCollapsed: boolean;
  readonly dynamicRangeDn: number | null;
  readonly failedClosed: boolean;
}

export interface FsyncPrecisionDisplay extends HudBadgeDisplay {
  readonly skewMs: number | null;
  readonly fsyncPrecisionOk: boolean;
  readonly hardSkewFault: boolean;
  readonly failedClosed: boolean;
}

export interface BearingTbcDisplay extends HudBadgeDisplay {
  readonly confirmed: boolean;
  readonly hitsInWindow: number | null;
  readonly confirmedTracks: number | null;
  readonly failedClosed: boolean;
}

export interface FormalAssuranceBadge {
  readonly id: "kani" | "apalache" | "mcu" | "seq_vpr";
  readonly text: string;
  readonly title: string;
  readonly tone: "emerald" | "cyan";
}

export const EO_SECTOR_COUNT = 12 as const;
export const EO_SECTOR_MASK_12 = 0x0fff as const;
export const EO_SECTOR_DEGRADED_THR = 0.5 as const;
export const LWIR_CONTRAST_COLLAPSE_P95_P05_THR = 10.0 as const;
export const LWIR_FFC_BIT_IMMINENT = 1 << 0;
export const LWIR_FFC_BIT_IN_PROGRESS = 1 << 1;
export const FSYNC_PRECISION_MAX_SKEW_MS = 1.0 as const;
export const MAX_TIMESTAMP_SKEW_MS = 200.0 as const;
export const BEARING_TBC_CONFIRM_M = 3 as const;
export const BEARING_TBC_WINDOW_N = 5 as const;

function escHtml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function isValidUnitQuality(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v) && v >= 0.0 && v <= 1.0;
}

/** Counts set bits in the lower 12 azimuth-sector bits (`0x0FFF`). Non-finite/invalid fails closed to 12. */
export function popcount12(mask: number): number {
  if (!Number.isFinite(mask) || !Number.isInteger(mask) || mask < 0 || mask > 0xffff) {
    return EO_SECTOR_COUNT;
  }
  let m = mask & EO_SECTOR_MASK_12;
  let count = 0;
  while (m !== 0) {
    m &= m - 1;
    count++;
  }
  return count;
}

function sectorQualityToBand(q: number | null, degraded: boolean): EoSectorBand {
  if (q === null) return "standby";
  if (!Number.isFinite(q) || q < 0.25) return "alert";
  if (degraded || q < EO_SECTOR_DEGRADED_THR) return "caution";
  if (q < 0.75) return "mid";
  return "nominal";
}

/**
 * Formats 12-azimuth-sector EO optical health (`min_sector_q_eo` + `degraded_sector_mask` + `sector_q_eo`).
 * Backward-compatible with v5.0 payloads lacking sector arrays; strictly fail-closed on NaN/Infinity/invalid data.
 */
export function formatSectorOpticalHealth(
  health: NativeSensorHealthPayload | null,
  locale: StudioLocale,
): SectorOpticalHealthDisplay {
  const tr = locale === "tr";
  if (!health) {
    return {
      text: tr ? "12-SEK EO · BEKLEME" : "12-SEC EO · STANDBY",
      title: tr
        ? "12-Azimut Sektör EO Optik Sağlık Haritası (yerel/canlı telemetri bekleniyor)"
        : "12-Azimuth-Sector EO Optical Quality Map (awaiting native/live telemetry)",
      state: "standby",
      minSectorQ: null,
      degradedCount: 0,
      degradedMask: 0,
      sectors: Array<null>(EO_SECTOR_COUNT).fill(null),
      sectorBands: Array<EoSectorBand>(EO_SECTOR_COUNT).fill("standby"),
      failedClosed: false,
      legacyFallback: false,
    };
  }

  const hasV52Fields =
    health.sector_q_eo !== undefined ||
    health.min_sector_q_eo !== undefined ||
    health.degraded_sector_mask !== undefined;

  if (!hasV52Fields) {
    if (!isValidUnitQuality(health.q_eo_effective)) {
      return {
        text: tr ? "12-SEK EO · HATALI (0%)" : "12-SEC EO · FAULT (0%)",
        title: "12-Sector EO fail-closed: invalid/non-finite q_eo_effective",
        state: "alert",
        minSectorQ: 0.0,
        degradedCount: EO_SECTOR_COUNT,
        degradedMask: EO_SECTOR_MASK_12,
        sectors: Array<number>(EO_SECTOR_COUNT).fill(0.0),
        sectorBands: Array<EoSectorBand>(EO_SECTOR_COUNT).fill("alert"),
        failedClosed: true,
        legacyFallback: true,
      };
    }
    const q = health.q_eo_effective;
    const degraded = q < EO_SECTOR_DEGRADED_THR;
    const mask = degraded ? EO_SECTOR_MASK_12 : 0;
    const degradedCount = degraded ? EO_SECTOR_COUNT : 0;
    const pct = Math.round(q * 100);
    const band = sectorQualityToBand(q, degraded);
    const state: HudBadgeState = q < 0.25 ? "alert" : degraded ? "caution" : "nominal";
    return {
      text: tr
        ? `12-SEK EO · min %${pct} · ${degradedCount}/12`
        : `12-SEC EO · min ${pct}% · ${degradedCount}/12`,
      title: `12-Sector EO (legacy global q_eo fallback): q=${q.toFixed(2)}, degraded=${degradedCount}/12`,
      state,
      minSectorQ: q,
      degradedCount,
      degradedMask: mask,
      sectors: Array<number>(EO_SECTOR_COUNT).fill(q),
      sectorBands: Array<EoSectorBand>(EO_SECTOR_COUNT).fill(band),
      failedClosed: false,
      legacyFallback: true,
    };
  }

  let failedClosed = false;
  let mask = 0;

  if (health.degraded_sector_mask !== undefined) {
    const rawMask = health.degraded_sector_mask;
    if (
      typeof rawMask !== "number" ||
      !Number.isFinite(rawMask) ||
      !Number.isInteger(rawMask) ||
      rawMask < 0 ||
      rawMask > 0xffff
    ) {
      failedClosed = true;
      mask = EO_SECTOR_MASK_12;
    } else {
      mask = rawMask & EO_SECTOR_MASK_12;
    }
  }

  let explicitMinQ: number | null = null;
  if (health.min_sector_q_eo !== undefined) {
    if (!isValidUnitQuality(health.min_sector_q_eo)) {
      failedClosed = true;
      explicitMinQ = 0.0;
    } else {
      explicitMinQ = health.min_sector_q_eo;
    }
  }

  const sectors: number[] = Array<number>(EO_SECTOR_COUNT).fill(0.0);
  let computedMinQ: number | null = null;

  if (health.sector_q_eo !== undefined) {
    const arr = health.sector_q_eo;
    if (!Array.isArray(arr) || arr.length !== EO_SECTOR_COUNT) {
      failedClosed = true;
    }
    let minVal = 1.0;
    for (let i = 0; i < EO_SECTOR_COUNT; i++) {
      const val = Array.isArray(arr) && i < arr.length ? arr[i] : Number.NaN;
      if (isValidUnitQuality(val)) {
        sectors[i] = val;
        if (val < minVal) minVal = val;
        if (val < EO_SECTOR_DEGRADED_THR) {
          mask |= 1 << i;
        }
      } else {
        failedClosed = true;
        sectors[i] = 0.0;
        minVal = 0.0;
        mask |= 1 << i;
      }
    }
    computedMinQ = minVal;
  } else {
    const qBaseValid = isValidUnitQuality(health.q_eo_effective);
    if (!qBaseValid) {
      failedClosed = true;
    }
    const qBase = qBaseValid ? health.q_eo_effective : 0.0;
    const minFallback = explicitMinQ ?? (mask === 0 ? qBase : Math.min(qBase, 0.49));
    for (let i = 0; i < EO_SECTOR_COUNT; i++) {
      const bitDegraded = ((mask >> i) & 1) !== 0;
      if (bitDegraded) {
        sectors[i] = Math.min(minFallback, 0.49);
      } else if (mask === 0 && explicitMinQ !== null && explicitMinQ < EO_SECTOR_DEGRADED_THR) {
        sectors[i] = explicitMinQ;
      } else {
        sectors[i] = qBase;
      }
    }
  }

  let minSectorQ: number;
  if (explicitMinQ !== null && computedMinQ !== null) {
    minSectorQ = Math.min(explicitMinQ, computedMinQ);
  } else if (computedMinQ !== null) {
    minSectorQ = computedMinQ;
  } else if (explicitMinQ !== null) {
    minSectorQ = explicitMinQ;
  } else {
    minSectorQ = failedClosed
      ? 0.0
      : mask === 0
        ? (isValidUnitQuality(health.q_eo_effective) ? health.q_eo_effective : 0.0)
        : 0.0;
  }

  if (failedClosed) {
    minSectorQ = 0.0;
    if (mask === 0) mask = EO_SECTOR_MASK_12;
  }

  const degradedCount = popcount12(mask);
  const sectorBands: EoSectorBand[] = sectors.map((q, i) =>
    sectorQualityToBand(q, ((mask >> i) & 1) !== 0),
  );

  const state: HudBadgeState =
    failedClosed || minSectorQ < 0.25 || degradedCount >= 6
      ? "alert"
      : degradedCount > 0 || minSectorQ < EO_SECTOR_DEGRADED_THR
        ? "caution"
        : "nominal";

  const pct = Math.round(minSectorQ * 100);
  const maskHex = `0x${mask.toString(16).toUpperCase().padStart(3, "0")}`;

  let text: string;
  if (failedClosed) {
    text = tr
      ? `12-SEK EO · HATALI (${degradedCount}/12)`
      : `12-SEC EO · FAULT (${degradedCount}/12)`;
  } else if (degradedCount > 0) {
    text = tr
      ? `12-SEK EO · min %${pct} · ${degradedCount}/12 BOZUK`
      : `12-SEC EO · min ${pct}% · ${degradedCount}/12 DEG`;
  } else {
    text = tr ? `12-SEK EO · min %${pct} · 0/12` : `12-SEC EO · min ${pct}% · 0/12`;
  }

  const title =
    `12-Sector EO Optical Health: min=${minSectorQ.toFixed(2)}, ` +
    `degraded=${degradedCount}/12 (mask=${maskHex}, thr=${EO_SECTOR_DEGRADED_THR.toFixed(2)})` +
    (failedClosed ? " [FAIL-CLOSED: INVALID TELEMETRY]" : "");

  return {
    text,
    title,
    state,
    minSectorQ,
    degradedCount,
    degradedMask: mask,
    sectors,
    sectorBands,
    failedClosed,
    legacyFallback: false,
  };
}

/**
 * Formats LWIR Hardware FFC (Flat-Field Correction) & `p95-p05` Contrast-Collapse backstop badge.
 * Fails closed on invalid/NaN dynamic range, malformed bitmasks, or non-finite LWIR quality.
 */
export function formatLwirFfcContrastBadge(
  health: NativeSensorHealthPayload | null,
  locale: StudioLocale,
): LwirFfcContrastDisplay {
  const tr = locale === "tr";
  if (!health) {
    return {
      text: tr ? "LWIR FFC · BEKLEME" : "LWIR FFC · STANDBY",
      title: tr
        ? "LWIR Donanım FFC (Flat-Field Correction) & p95-p05 Kontrast Çökme Koruması"
        : "LWIR Hardware FFC (Flat-Field Correction) & p95-p05 Contrast-Collapse Guard (< 10 DN)",
      state: "standby",
      ffcImminent: false,
      ffcInProgress: false,
      contrastCollapsed: false,
      dynamicRangeDn: null,
      failedClosed: false,
    };
  }

  let failedClosed = false;
  let aliasImminent = false;
  let aliasInProgress = false;

  if (health.lwir_ffc_flag !== undefined) {
    if (typeof health.lwir_ffc_flag === "boolean") {
      aliasInProgress = health.lwir_ffc_flag;
    } else if (typeof health.lwir_ffc_flag === "number") {
      if (
        !Number.isFinite(health.lwir_ffc_flag) ||
        !Number.isInteger(health.lwir_ffc_flag) ||
        health.lwir_ffc_flag < 0
      ) {
        failedClosed = true;
        aliasInProgress = true;
      } else {
        aliasImminent = (health.lwir_ffc_flag & LWIR_FFC_BIT_IMMINENT) !== 0;
        aliasInProgress = (health.lwir_ffc_flag & LWIR_FFC_BIT_IN_PROGRESS) !== 0;
      }
    } else {
      failedClosed = true;
      aliasInProgress = true;
    }
  }

  const ffcImminent = health.lwir_ffc_imminent === true || aliasImminent;
  const ffcInProgress = health.lwir_ffc_in_progress === true || aliasInProgress;

  let dynamicRangeDn: number | null = null;
  let drCollapsed = false;
  if (health.lwir_dynamic_range_p95_p05 !== undefined) {
    const dr = health.lwir_dynamic_range_p95_p05;
    if (typeof dr !== "number" || !Number.isFinite(dr) || dr < 0) {
      failedClosed = true;
      drCollapsed = true;
    } else {
      dynamicRangeDn = dr;
      drCollapsed = dr < LWIR_CONTRAST_COLLAPSE_P95_P05_THR;
    }
  }

  const qLwirValid = isValidUnitQuality(health.q_lwir_effective);
  if (!qLwirValid) {
    failedClosed = true;
  }

  const contrastCollapsed = health.lwir_contrast_collapsed === true || drCollapsed || failedClosed;
  const drLabel = dynamicRangeDn !== null ? `Δ${Math.round(dynamicRangeDn)} DN` : "NaN DN";
  const detail =
    `LWIR FFC imminent=${ffcImminent}, in_progress=${ffcInProgress} | ` +
    `Contrast collapsed=${contrastCollapsed} (p95-p05=${dynamicRangeDn !== null ? dynamicRangeDn.toFixed(1) : "n/a"} DN, thr=${LWIR_CONTRAST_COLLAPSE_P95_P05_THR} DN) | ` +
    `NUC frozen=${health.lwir_nuc_frozen} (${health.lwir_freeze_ticks}t)` +
    (failedClosed ? " [FAIL-CLOSED]" : "");

  if (health.lwir_nuc_frozen) {
    const ticks = Number.isFinite(health.lwir_freeze_ticks) ? Math.max(0, Math.round(health.lwir_freeze_ticks)) : 0;
    return {
      text: tr ? `LWIR NUC DONDU (${ticks}t)` : `LWIR NUC FREEZE (${ticks}t)`,
      title: detail,
      state: "alert",
      ffcImminent,
      ffcInProgress,
      contrastCollapsed,
      dynamicRangeDn,
      failedClosed,
    };
  }

  if (ffcInProgress) {
    return {
      text: tr ? "LWIR FFC AKTİF" : "LWIR FFC ACTIVE",
      title: detail,
      state: "alert",
      ffcImminent,
      ffcInProgress,
      contrastCollapsed,
      dynamicRangeDn,
      failedClosed,
    };
  }

  if (contrastCollapsed) {
    return {
      text: tr ? `LWIR KONTRAST · ${drLabel}` : `LWIR CONTRAST · ${drLabel}`,
      title: detail,
      state: "alert",
      ffcImminent,
      ffcInProgress,
      contrastCollapsed,
      dynamicRangeDn,
      failedClosed,
    };
  }

  if (ffcImminent) {
    return {
      text: tr ? "LWIR FFC YAKIN" : "LWIR FFC IMMINENT",
      title: detail,
      state: "caution",
      ffcImminent,
      ffcInProgress,
      contrastCollapsed,
      dynamicRangeDn,
      failedClosed,
    };
  }

  if (health.q_lwir_effective < 0.5) {
    return {
      text: tr
        ? `LWIR DÜŞÜK · %${Math.round(health.q_lwir_effective * 100)}`
        : `LWIR DEG · ${Math.round(health.q_lwir_effective * 100)}%`,
      title: detail,
      state: "caution",
      ffcImminent,
      ffcInProgress,
      contrastCollapsed,
      dynamicRangeDn,
      failedClosed,
    };
  }

  return {
    text: dynamicRangeDn !== null ? `LWIR FFC OK · ${drLabel}` : "LWIR FFC OK",
    title: detail,
    state: "nominal",
    ffcImminent,
    ffcInProgress,
    contrastCollapsed,
    dynamicRangeDn,
    failedClosed,
  };
}

/**
 * Formats Dual-Tier `TimestampSkewGuard` (`<= 1.0 ms` FSYNC precision tier + `> 200.0 ms` hard breaker).
 * Fails closed if `timestamp_skew_ms` is non-finite/negative or conflicts with `fsync_precision_ok`.
 */
export function formatFsyncPrecisionBadge(
  health: NativeSensorHealthPayload | null,
  locale: StudioLocale,
): FsyncPrecisionDisplay {
  const tr = locale === "tr";
  if (!health) {
    return {
      text: tr ? "FSYNC <=1ms · BEKLEME" : "FSYNC <=1ms · STANDBY",
      title: tr
        ? "Çift-Kademeli Zaman Kayması Koruması: <=1.0ms FSYNC hassasiyet + >200ms sert kesici"
        : "Dual-Tier TimestampSkewGuard: <=1.0ms FSYNC precision tier + >200ms hard breaker",
      state: "standby",
      skewMs: null,
      fsyncPrecisionOk: false,
      hardSkewFault: false,
      failedClosed: false,
    };
  }

  const rawSkew = health.timestamp_skew_ms;
  if (typeof rawSkew !== "number" || !Number.isFinite(rawSkew) || rawSkew < 0) {
    return {
      text: tr ? "FSYNC HATA · GEÇERSİZ" : "FSYNC FAULT · INVALID",
      title: "Dual-Tier TimestampSkewGuard fail-closed: non-finite or negative timestamp_skew_ms",
      state: "alert",
      skewMs: null,
      fsyncPrecisionOk: false,
      hardSkewFault: true,
      failedClosed: true,
    };
  }

  const hardSkewFault =
    rawSkew > MAX_TIMESTAMP_SKEW_MS ||
    health.timestamp_skew_fault === true ||
    health.timestamp_monotonic_ok === false;
  const withinPrecision = rawSkew <= FSYNC_PRECISION_MAX_SKEW_MS;
  const fsyncPrecisionOk =
    !hardSkewFault && withinPrecision && health.fsync_precision_ok !== false;

  const detail =
    `Dual-Tier FSYNC Guard: skew=${rawSkew.toFixed(2)}ms | ` +
    `FSYNC <=1.0ms=${fsyncPrecisionOk ? "OK" : "WARN"} | ` +
    `Hard <=200ms=${hardSkewFault ? "FAULT" : "OK"} | ` +
    `Monotonic=${health.timestamp_monotonic_ok !== false ? "OK" : "FAULT"}`;

  if (hardSkewFault) {
    return {
      text: tr ? `FSYNC KESİCİ · ${Math.round(rawSkew)}ms` : `FSYNC FAULT · ${Math.round(rawSkew)}ms`,
      title: detail,
      state: "alert",
      skewMs: rawSkew,
      fsyncPrecisionOk: false,
      hardSkewFault: true,
      failedClosed: false,
    };
  }

  if (fsyncPrecisionOk) {
    return {
      text: `FSYNC <=1ms · ${rawSkew.toFixed(2)}ms`,
      title: detail,
      state: "nominal",
      skewMs: rawSkew,
      fsyncPrecisionOk: true,
      hardSkewFault: false,
      failedClosed: false,
    };
  }

  return {
    text: tr ? `FSYNC >1ms · ${rawSkew.toFixed(1)}ms` : `FSYNC >1ms · ${rawSkew.toFixed(1)}ms`,
    title: detail,
    state: "caution",
    skewMs: rawSkew,
    fsyncPrecisionOk: false,
    hardSkewFault: false,
    failedClosed: false,
  };
}

/**
 * Formats zero-allocation M-of-N (`3-of-5`) Bearing Track-Before-Confirm (`TBC`) status badge.
 * Backward-compatible with legacy payloads; fails closed on NaN/negative hit or track counts.
 */
export function formatBearingTbcBadge(
  health: NativeSensorHealthPayload | null,
  locale: StudioLocale,
): BearingTbcDisplay {
  const tr = locale === "tr";
  if (!health) {
    return {
      text: tr ? "M/N TBC 3/5 · BEKLEME" : "M/N TBC 3/5 · STANDBY",
      title: tr
        ? "Sıfır-Ayırmalı 3-of-5 Açısal İz Onayı (0.035 rad geçit, 5-kayıp düşürme)"
        : "Zero-alloc 3-of-5 Bearing Track-Before-Confirm (0.035 rad angular gate, 5-miss drop)",
      state: "standby",
      confirmed: false,
      hitsInWindow: null,
      confirmedTracks: null,
      failedClosed: false,
    };
  }

  const hasTbcFields =
    health.bearing_track_confirmed !== undefined ||
    health.bearing_track_hits_in_window !== undefined ||
    health.bearing_tbc_confirmed_tracks !== undefined;

  if (!hasTbcFields) {
    return {
      text: tr ? "M/N TBC 3/5 · HAZIR" : "M/N TBC 3/5 · READY",
      title: tr
        ? "M-of-N (3/5) TBC İzleyici Hazır (eski telemetri uyumlu)"
        : "M-of-N (3/5) Bearing TBC Tracker Ready (legacy telemetry fallback)",
      state: "standby",
      confirmed: false,
      hitsInWindow: null,
      confirmedTracks: null,
      failedClosed: false,
    };
  }

  let hits: number | null = null;
  if (health.bearing_track_hits_in_window !== undefined) {
    const raw = health.bearing_track_hits_in_window;
    if (typeof raw !== "number" || !Number.isFinite(raw) || !Number.isInteger(raw) || raw < 0 || raw > 16) {
      return {
        text: tr ? "M/N TBC 3/5 · HATALI" : "M/N TBC 3/5 · FAULT",
        title: "Bearing TBC fail-closed: invalid bearing_track_hits_in_window",
        state: "alert",
        confirmed: false,
        hitsInWindow: null,
        confirmedTracks: null,
        failedClosed: true,
      };
    }
    hits = raw;
  }

  let confirmedTracks: number | null = null;
  if (health.bearing_tbc_confirmed_tracks !== undefined) {
    const raw = health.bearing_tbc_confirmed_tracks;
    if (typeof raw !== "number" || !Number.isFinite(raw) || !Number.isInteger(raw) || raw < 0) {
      return {
        text: tr ? "M/N TBC 3/5 · HATALI" : "M/N TBC 3/5 · FAULT",
        title: "Bearing TBC fail-closed: invalid bearing_tbc_confirmed_tracks",
        state: "alert",
        confirmed: false,
        hitsInWindow: hits,
        confirmedTracks: null,
        failedClosed: true,
      };
    }
    confirmedTracks = raw;
  }

  const confirmed =
    health.bearing_track_confirmed === true || (confirmedTracks !== null && confirmedTracks > 0);
  const windowLabel =
    hits !== null ? `${Math.min(hits, BEARING_TBC_WINDOW_N)}/${BEARING_TBC_WINDOW_N}` : "";
  const detail =
    `BearingTrackBeforeConfirm (${BEARING_TBC_CONFIRM_M}-of-${BEARING_TBC_WINDOW_N}): ` +
    `confirmed=${confirmed}, hits=${hits ?? "n/a"}/${BEARING_TBC_WINDOW_N}` +
    (confirmedTracks !== null ? `, confirmed_tracks=${confirmedTracks}` : "");

  if (confirmed) {
    const suffix =
      confirmedTracks !== null && confirmedTracks > 0
        ? tr
          ? `${confirmedTracks} ONAYLI`
          : `${confirmedTracks} CONF`
        : windowLabel
          ? tr
            ? `ONAYLI ${windowLabel}`
            : `CONF ${windowLabel}`
          : tr
            ? "ONAYLI"
            : "CONFIRMED";
    return {
      text: `TBC 3/5 · ${suffix}`,
      title: detail,
      state: "nominal",
      confirmed: true,
      hitsInWindow: hits,
      confirmedTracks,
      failedClosed: false,
    };
  }

  if (hits !== null && hits > 0) {
    return {
      text: tr ? `TBC 3/5 · ADAY ${windowLabel}` : `TBC 3/5 · TENT ${windowLabel}`,
      title: detail,
      state: "caution",
      confirmed: false,
      hitsInWindow: hits,
      confirmedTracks,
      failedClosed: false,
    };
  }

  return {
    text: tr
      ? `TBC 3/5 · İZ YOK${windowLabel ? ` ${windowLabel}` : ""}`
      : `TBC 3/5 · IDLE${windowLabel ? ` ${windowLabel}` : ""}`,
    title: detail,
    state: "standby",
    confirmed: false,
    hitsInWindow: hits,
    confirmedTracks,
    failedClosed: false,
  };
}

/** Verified static Formal & Bare-Metal Assurance badges for OpenXVS Studio v5.2 (bilingual TR/EN). */
export function getFormalAssuranceBadges(locale: StudioLocale): readonly FormalAssuranceBadge[] {
  const tr = locale === "tr";
  return [
    {
      id: "kani",
      text: tr ? "26/26 Kani CBMC · 5.228" : "26/26 Kani CBMC · 5,228",
      title: tr
        ? "26/26 Kani CBMC formal kanıt demeti doğrulandı (5.228 çoklu-uçak kontrolü, 0 hata)"
        : "26/26 Kani CBMC formal proof harnesses verified (5,228 multiship checks, 0 failures)",
      tone: "emerald",
    },
    {
      id: "apalache",
      text: "Apalache N=2/N=3",
      title: tr
        ? "N=2 ve N=3 çoklu-uçak koordinasyonu için Apalache sembolik TLA+ model kontrolü"
        : "Apalache symbolic TLA+ bounded model checking verified for N=2 and N=3 multi-ship coordination",
      tone: "emerald",
    },
    {
      id: "mcu",
      text: "Cortex-M7 thumbv7em · 88KB/0B",
      title: tr
        ? "Bare-metal ARM Cortex-M7 (thumbv7em-none-eabi) Ring-0 kanıtı: 88 KB Flash (%4,3 STM32H7), 0 B statik RAM, 0 heap"
        : "Bare-metal ARM Cortex-M7 (thumbv7em-none-eabi) Ring-0 proof: 88 KB Flash (4.3% of 2 MiB STM32H7), 0 B static RAM, 0 heap",
      tone: "cyan",
    },
    {
      id: "seq_vpr",
      text: "Sequence-VPR · 0 HMI",
      title: tr
        ? "Sequence-VPR zamansal görsel yerelleştirme: 0 HMI (Tehlikeli Yanıltıcı Bilgi)"
        : "Sequence-VPR temporal visual localization: 0 HMI (Hazardously Misleading Information)",
      tone: "emerald",
    },
  ];
}

/** Pure HTML badge strip renderer for the Workbench Eval/Assurance tab (directly unit-testable). */
export function renderFormalAssuranceStripHtml(locale: StudioLocale): string {
  const chips = getFormalAssuranceBadges(locale)
    .map(
      (b) =>
        `<span class="studio-assurance-chip" data-tone="${b.tone}" title="${escHtml(b.title)}">${escHtml(b.text)}</span>`,
    )
    .join("");
  return `<div class="bench-assurance-strip" role="list">${chips}</div>`;
}
