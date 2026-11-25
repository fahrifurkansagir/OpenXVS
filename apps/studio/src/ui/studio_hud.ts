/**
 * OpenXVS Studio v5.2 Glassmorphic HUD + Persistent 4-Channel Sensor Drawer + Weather Popover
 * + Live v5.2 Avionics Telemetry (12-Sector EO, LWIR FFC/Contrast-Collapse, Dual-Tier FSYNC <=1.0ms,
 *   M-of-N Bearing TBC) + Formal & Bare-Metal Assurance Badges + Multi-Ship Coord (F1/F4).
 *
 * Preserves `D16-W19-m3`: mounts the 4-channel drawer DOM ONCE and updates labels/active attributes
 * in place so live/uploaded `<img>` previews are NEVER destroyed when clicking HUD buttons,
 * and displays exact short `checkpoint_sha256` badges (`SHA 2c9578f3`, etc.) on every channel.
 */

import type { VisualWorldMode } from "../scene/atmosphere_world.ts";
import type { FlightControlMode } from "../sim/manual_flight_controller.ts";
import {
  ActiveRangeModeTs,
  DEFAULT_WEATHER_STATE,
  type StudioScenarioId,
  type StudioSimSnapshot,
  type WeatherState,
} from "../sim/studio_sim.ts";
import {
  renderFallbackSensorCanvas,
  type BridgeChannelResult,
  type NativeSensorHealthPayload,
  type SensorChannelId,
  type StudioBridgeClient,
} from "./bridge_client.ts";
import { getStudioStrings, type StudioCameraPreset, type StudioLocale } from "./i18n.ts";
import {
  EO_SECTOR_COUNT,
  formatBearingTbcBadge,
  formatFsyncPrecisionBadge,
  formatLwirFfcContrastBadge,
  formatSectorOpticalHealth,
  getFormalAssuranceBadges,
  type HudBadgeDisplay,
} from "./v52_telemetry.ts";

export {
  formatBearingTbcBadge,
  formatFsyncPrecisionBadge,
  formatLwirFfcContrastBadge,
  formatSectorOpticalHealth,
  getFormalAssuranceBadges,
  type HudBadgeDisplay,
  type HudBadgeState,
} from "./v52_telemetry.ts";

export interface StudioHudCallbacks {
  readonly onScenarioChange: (scenario: StudioScenarioId) => void;
  readonly onCameraChange: (preset: StudioCameraPreset) => void;
  readonly onVisualModeChange: (mode: VisualWorldMode) => void;
  readonly onControlModeChange: (mode: FlightControlMode) => void;
  readonly onShieldToggle: (enabled: boolean) => void;
  readonly onWeatherChange: (partial: Partial<WeatherState>) => void;
  readonly onToggleEngineOut: () => void;
  readonly onResetApproach: () => void;
  readonly onDomeToggle: (visible: boolean) => void;
  readonly onOpenVideoBench: () => void;
}

const SCENARIO_ORDER: readonly StudioScenarioId[] = ["cbdr", "forcedland", "thermal_fog", "gps_spoof"];
const CAMERA_ORDER: readonly StudioCameraPreset[] = ["chase", "wingman", "overview", "cockpit"];
const CHANNEL_ORDER: readonly SensorChannelId[] = ["d2", "land", "thermal", "nav"];

function esc(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function setTextIfChanged(el: HTMLElement | null | undefined, next: string): void {
  if (el && el.textContent !== next) {
    el.textContent = next;
  }
}

function setAttrIfChanged(el: HTMLElement | null | undefined, attr: string, next: string): void {
  if (el && el.getAttribute(attr) !== next) {
    el.setAttribute(attr, next);
  }
}

/** Pure formatter for the live Sensor Health HUD badge (directly unit-testable in Vitest). */
export function formatSensorHealthBadge(
  health: NativeSensorHealthPayload | null,
  locale: StudioLocale,
): HudBadgeDisplay {
  const hs = getStudioStrings(locale).hudStatus;
  if (!health) {
    return {
      text: hs.sensorStandby,
      title:
        "SensorHealthMonitor v5.2: 12-Sector EO, LWIR FFC/Contrast-Collapse, Dual-Tier FSYNC <=1ms & M-of-N TBC (awaiting telemetry)",
      state: "standby",
    };
  }

  const sectorInfo = formatSectorOpticalHealth(health, locale);
  const lwirInfo = formatLwirFfcContrastBadge(health, locale);
  const fsyncInfo = formatFsyncPrecisionBadge(health, locale);

  const qEoSafe = Number.isFinite(health.q_eo_effective) ? health.q_eo_effective : Number.NaN;
  const qLwirSafe = Number.isFinite(health.q_lwir_effective) ? health.q_lwir_effective : Number.NaN;
  const blurStr = Number.isFinite(health.eo_blur_score) ? health.eo_blur_score.toFixed(1) : "NaN";
  const glareStr = Number.isFinite(health.eo_glare_ratio)
    ? `${(health.eo_glare_ratio * 100).toFixed(1)}%`
    : "NaN";
  const skewStr = Number.isFinite(health.timestamp_skew_ms)
    ? `${health.timestamp_skew_ms.toFixed(1)}ms`
    : "NaN";

  const diag =
    `Mode: ${health.thermal_mode} | EO q=${Number.isFinite(qEoSafe) ? qEoSafe.toFixed(2) : "NaN"} ` +
    `(blur=${blurStr}, glare=${glareStr}, 12-sec min=${sectorInfo.minSectorQ !== null ? sectorInfo.minSectorQ.toFixed(2) : "n/a"}, deg=${sectorInfo.degradedCount}/12) | ` +
    `LWIR q=${Number.isFinite(qLwirSafe) ? qLwirSafe.toFixed(2) : "NaN"} ` +
    `(NUC frozen=${health.lwir_nuc_frozen}, ticks=${health.lwir_freeze_ticks}, FFC=${lwirInfo.ffcInProgress ? "ACT" : lwirInfo.ffcImminent ? "IMM" : "OK"}, contrast_drop=${lwirInfo.contrastCollapsed}) | ` +
    `Skew=${skewStr} (FSYNC<=1ms=${fsyncInfo.fsyncPrecisionOk ? "OK" : "WARN"})`;

  if (health.lwir_nuc_frozen) {
    const ticks = Number.isFinite(health.lwir_freeze_ticks)
      ? Math.max(0, Math.round(health.lwir_freeze_ticks))
      : 0;
    return {
      text: `${hs.sensorNucFreeze} (${ticks}t)`,
      title: diag,
      state: "alert",
    };
  }
  if (fsyncInfo.failedClosed || fsyncInfo.hardSkewFault) {
    const skewLabel =
      fsyncInfo.skewMs !== null ? `${Math.round(fsyncInfo.skewMs)}ms` : "FAULT";
    return {
      text: `${hs.sensorSkew} ${skewLabel}`,
      title: diag,
      state: "alert",
    };
  }
  if (lwirInfo.ffcInProgress) {
    return {
      text: hs.sensorLwirFfc,
      title: diag,
      state: "alert",
    };
  }
  if (lwirInfo.contrastCollapsed) {
    return {
      text: hs.sensorLwirContrast,
      title: diag,
      state: "alert",
    };
  }
  if (sectorInfo.failedClosed || !Number.isFinite(qEoSafe) || !Number.isFinite(qLwirSafe)) {
    return {
      text: `${hs.sensorDegraded} · FAULT`,
      title: diag,
      state: "alert",
    };
  }
  if (
    qEoSafe < 0.5 ||
    qLwirSafe < 0.5 ||
    health.thermal_mode === "DUAL_DVE_SVS_FALLBACK"
  ) {
    return {
      text: `${hs.sensorDegraded} · EO ${Math.round(qEoSafe * 100)}%`,
      title: diag,
      state: "caution",
    };
  }
  if (sectorInfo.degradedCount > 0 || (sectorInfo.minSectorQ !== null && sectorInfo.minSectorQ < 0.5)) {
    return {
      text: `${hs.sensorSectorDegraded} (${sectorInfo.degradedCount}/12)`,
      title: diag,
      state: "caution",
    };
  }
  if (lwirInfo.ffcImminent) {
    return {
      text: hs.sensorLwirFfc,
      title: diag,
      state: "caution",
    };
  }
  return {
    text: `${hs.sensorOk} · ${Math.round(health.timestamp_skew_ms)}ms`,
    title: diag,
    state: "nominal",
  };
}

/** Pure formatter for the live Multi-Ship Coord HUD badge (directly unit-testable in Vitest). */
export function formatMultiShipCoordBadge(
  health: NativeSensorHealthPayload | null,
  locale: StudioLocale,
): HudBadgeDisplay {
  const hs = getStudioStrings(locale).hudStatus;
  if (!health) {
    return {
      text: hs.coordStandby,
      title:
        "xvs-multiship-coord: F1 FAR 91.113 non-reversing right-turn asymmetry + F4 500ms sticky triple-ray ghost hysteresis",
      state: "standby",
    };
  }
  const detail =
    `MultiShipCoord=${health.multiship_coord_enabled ? "ON" : "OFF"} | ` +
    `F1 FAR 91.113=${health.f1_far91113_active ? "ACTIVE" : "IDLE"} | ` +
    `F4 Ghost Hysteresis=${health.f4_ghost_hysteresis_active ? "ACTIVE" : "IDLE"}`;

  if (!health.multiship_coord_enabled) {
    return {
      text: locale === "tr" ? "COORD: KAPALI" : "COORD: OFF",
      title: detail,
      state: "caution",
    };
  }
  if (health.f1_far91113_active && health.f4_ghost_hysteresis_active) {
    return {
      text: hs.coordActive,
      title: detail,
      state: "nominal",
    };
  }
  return {
    text: `COORD: ${health.f1_far91113_active ? "F1 91.113" : "F1 IDLE"} · ${health.f4_ghost_hysteresis_active ? "F4 GHOST" : "F4 IDLE"}`,
    title: detail,
    state: "caution",
  };
}

export class StudioHud {
  private readonly host: HTMLElement;
  private readonly callbacks: StudioHudCallbacks;
  private readonly bridge: StudioBridgeClient;

  private readonly topbarEl: HTMLElement;
  private readonly statusStripEl: HTMLDivElement;
  private readonly alertEl: HTMLDivElement;
  private readonly weatherEl: HTMLDivElement;
  private readonly bottomDockEl: HTMLDivElement;
  private readonly drawerEl: HTMLElement;

  private locale: StudioLocale = "tr";
  private scenario: StudioScenarioId = "cbdr";
  private cameraPreset: StudioCameraPreset = "chase";
  private visualMode: VisualWorldMode = "photoreal";
  private controlMode: FlightControlMode = "autopilot";
  private shieldEnabled = true;
  private weather: WeatherState = { ...DEFAULT_WEATHER_STATE };
  private weatherOpen = false;
  private domeVisible = false;
  private drawerOpen = typeof window !== "undefined" && window.innerWidth >= 1280;
  private drawerBeforeCockpit: boolean | null = null;
  private uploadChannel: SensorChannelId = "d2";
  private sensorHealth: NativeSensorHealthPayload | null = null;

  private readonly channelTitleEls = new Map<SensorChannelId, HTMLElement>();
  private readonly channelCanvases = new Map<SensorChannelId, HTMLCanvasElement>();
  private readonly channelImgs = new Map<SensorChannelId, HTMLImageElement>();
  private readonly channelLatEls = new Map<SensorChannelId, HTMLElement>();
  private readonly channelShaEls = new Map<SensorChannelId, HTMLElement>();
  private readonly channelHasCustomImage = new Set<SensorChannelId>();
  private readonly sectorCellEls: HTMLElement[] = [];
  private readonly assuranceChipEls = new Map<string, HTMLElement>();

  constructor(host: HTMLElement, bridge: StudioBridgeClient, callbacks: StudioHudCallbacks) {
    this.host = host;
    this.bridge = bridge;
    this.callbacks = callbacks;

    this.topbarEl = document.createElement("header");
    this.topbarEl.className = "studio-topbar";

    this.statusStripEl = document.createElement("div");
    this.statusStripEl.className = "studio-status-strip";
    this.statusStripEl.innerHTML =
      `<span id="studio-sensor-health-badge" class="studio-status-badge" data-state="standby"></span>` +
      `<span id="studio-multiship-badge" class="studio-status-badge" data-state="standby"></span>`;

    this.alertEl = document.createElement("div");
    this.alertEl.className = "studio-flight-alert";
    this.alertEl.dataset.visible = "false";

    this.weatherEl = document.createElement("div");
    this.weatherEl.className = "studio-weather-popover";
    this.weatherEl.dataset.open = "false";

    this.bottomDockEl = document.createElement("div");
    this.bottomDockEl.className = "studio-bottom-dock";

    this.drawerEl = document.createElement("aside");
    this.drawerEl.className = "studio-sensor-drawer";
    this.drawerEl.dataset.open = String(this.drawerOpen);

    this.host.appendChild(this.topbarEl);
    this.host.appendChild(this.statusStripEl);
    this.host.appendChild(this.alertEl);
    this.host.appendChild(this.weatherEl);
    this.host.appendChild(this.bottomDockEl);
    this.host.appendChild(this.drawerEl);

    this.mountDrawerOnce();
    this.mountWeatherOnce();
    this.renderBars();
    this.syncDrawerText();
    this.syncWeatherUi();
    this.syncStatusBadges();
  }

  setLocale(next: StudioLocale): void {
    this.locale = next;
    this.renderBars();
    this.syncDrawerText();
    this.syncWeatherUi();
    this.syncStatusBadges();
  }

  getLocale(): StudioLocale {
    return this.locale;
  }

  setScenario(next: StudioScenarioId): void {
    this.scenario = next;
    this.renderBars();
  }

  setCameraPreset(next: StudioCameraPreset): void {
    const prev = this.cameraPreset;
    this.cameraPreset = next;
    // Auto-collapse 4-Channel AI drawer in Cockpit view so both G1000 displays are 100% unobstructed;
    // restore previous drawer state automatically when returning to external cameras!
    if (next === "cockpit" && prev !== "cockpit") {
      this.drawerBeforeCockpit = this.drawerOpen;
      this.setDrawerOpen(false);
    } else if (next !== "cockpit" && prev === "cockpit" && this.drawerBeforeCockpit !== null) {
      this.setDrawerOpen(this.drawerBeforeCockpit);
      this.drawerBeforeCockpit = null;
    }
    this.renderBars();
  }

  setVisualMode(next: VisualWorldMode): void {
    this.visualMode = next;
    this.renderBars();
  }

  setControlMode(next: FlightControlMode): void {
    this.controlMode = next;
    this.renderBars();
  }

  setShieldEnabled(enabled: boolean): void {
    this.shieldEnabled = enabled;
    this.renderBars();
  }

  setWeather(next: WeatherState): void {
    this.weather = { ...next };
    this.syncWeatherUi();
  }

  setDomeVisible(visible: boolean): void {
    this.domeVisible = visible;
    this.renderBars();
  }

  setDrawerOpen(open: boolean): void {
    this.drawerOpen = open;
    this.drawerEl.dataset.open = String(open);
    const btn = this.topbarEl.querySelector<HTMLButtonElement>("#studio-sensors-btn");
    if (btn) btn.dataset.active = String(open);
  }

  isDrawerOpen(): boolean {
    return this.drawerOpen;
  }

  applySensorHealth(health: NativeSensorHealthPayload | null): void {
    this.sensorHealth = health;
    this.syncStatusBadges();
  }

  private syncStatusBadges(): void {
    const shEl = this.statusStripEl.querySelector<HTMLElement>("#studio-sensor-health-badge");
    const msEl = this.statusStripEl.querySelector<HTMLElement>("#studio-multiship-badge");
    if (shEl) {
      const info = formatSensorHealthBadge(this.sensorHealth, this.locale);
      setTextIfChanged(shEl, info.text);
      setAttrIfChanged(shEl, "title", info.title);
      setAttrIfChanged(shEl, "data-state", info.state);
    }
    if (msEl) {
      const info = formatMultiShipCoordBadge(this.sensorHealth, this.locale);
      setTextIfChanged(msEl, info.text);
      setAttrIfChanged(msEl, "title", info.title);
      setAttrIfChanged(msEl, "data-state", info.state);
    }
    this.syncV52AvionicsPanel();
  }

  private syncV52AvionicsPanel(): void {
    const sectorInfo = formatSectorOpticalHealth(this.sensorHealth, this.locale);
    const sectorBadge = this.drawerEl.querySelector<HTMLElement>("#studio-v52-sector-badge");
    setTextIfChanged(sectorBadge, sectorInfo.text);
    setAttrIfChanged(sectorBadge, "title", sectorInfo.title);
    setAttrIfChanged(sectorBadge, "data-state", sectorInfo.state);

    const sectorBar = this.drawerEl.querySelector<HTMLElement>("#studio-v52-sector-bar");
    setAttrIfChanged(sectorBar, "title", sectorInfo.title);
    for (let i = 0; i < this.sectorCellEls.length; i++) {
      const cell = this.sectorCellEls[i];
      const band = sectorInfo.sectorBands[i] ?? "standby";
      const q = sectorInfo.sectors[i];
      setAttrIfChanged(cell, "data-band", band);
      setAttrIfChanged(
        cell,
        "title",
        q !== null && q !== undefined ? `S${i + 1}: ${Math.round(q * 100)}%` : `S${i + 1}: --`,
      );
    }

    const lwirInfo = formatLwirFfcContrastBadge(this.sensorHealth, this.locale);
    const lwirEl = this.drawerEl.querySelector<HTMLElement>("#studio-v52-lwir-badge");
    setTextIfChanged(lwirEl, lwirInfo.text);
    setAttrIfChanged(lwirEl, "title", lwirInfo.title);
    setAttrIfChanged(lwirEl, "data-state", lwirInfo.state);

    const fsyncInfo = formatFsyncPrecisionBadge(this.sensorHealth, this.locale);
    const fsyncEl = this.drawerEl.querySelector<HTMLElement>("#studio-v52-fsync-badge");
    setTextIfChanged(fsyncEl, fsyncInfo.text);
    setAttrIfChanged(fsyncEl, "title", fsyncInfo.title);
    setAttrIfChanged(fsyncEl, "data-state", fsyncInfo.state);

    const tbcInfo = formatBearingTbcBadge(this.sensorHealth, this.locale);
    const tbcEl = this.drawerEl.querySelector<HTMLElement>("#studio-v52-tbc-badge");
    setTextIfChanged(tbcEl, tbcInfo.text);
    setAttrIfChanged(tbcEl, "title", tbcInfo.title);
    setAttrIfChanged(tbcEl, "data-state", tbcInfo.state);
  }

  syncBridgeBadge(): void {
    const badge = this.drawerEl.querySelector<HTMLElement>("#studio-bridge-badge");
    if (!badge) return;
    const s = getStudioStrings(this.locale);
    const online = this.bridge.isOnline();
    badge.dataset.online = String(online);
    if (online && this.bridge.getTransport() === "tauri-ipc") {
      badge.textContent = s.drawer.bridgeNative;
      badge.title = s.stackV5Spec;
    } else {
      badge.textContent = online ? s.drawer.bridgeOnline : s.drawer.bridgeFallback;
      badge.title = online ? s.stackV5Spec : s.drawer.bridgeFallback;
    }
  }

  applyBridgeChannels(
    channels: Partial<Record<SensorChannelId, BridgeChannelResult>>,
    fromManualUpload = false,
  ): void {
    for (const id of CHANNEL_ORDER) {
      const res = channels[id];
      if (!res) continue;
      if (fromManualUpload) {
        this.channelHasCustomImage.add(id);
      } else if (this.channelHasCustomImage.has(id)) {
        // Keep user's uploaded image visible in this slot instead of overwriting immediately
        continue;
      }
      const img = this.channelImgs.get(id);
      const canvas = this.channelCanvases.get(id);
      const latEl = this.channelLatEls.get(id);
      const shaEl = this.channelShaEls.get(id);

      if (res.imageDataUrl && img && canvas) {
        img.src = res.imageDataUrl;
        img.classList.remove("hidden");
        canvas.style.display = "none";
      }
      if (latEl) {
        if (id === "nav" && res.metrics && typeof res.metrics.xfeat_keypoints_count === "number") {
          const msStr = res.latencyMs !== null ? `${res.latencyMs.toFixed(1)}ms` : "LIVE";
          latEl.textContent = `${msStr} · XF ${res.metrics.xfeat_keypoints_count}`;
        } else {
          latEl.textContent = res.latencyMs !== null ? `${res.latencyMs.toFixed(1)} ms` : res.summary;
        }
      }
      if (shaEl) {
        if (res.checkpointShaShort) {
          shaEl.textContent = `SHA ${res.checkpointShaShort}`;
          shaEl.classList.remove("hidden");
        } else {
          shaEl.classList.add("hidden");
        }
      }
    }
    this.syncBridgeBadge();
  }

  updateTelemetry(snap: StudioSimSnapshot, fps: number): void {
    const s = getStudioStrings(this.locale);

    const fpsEl = this.topbarEl.querySelector<HTMLElement>("#studio-fps-pill");
    if (fpsEl) fpsEl.textContent = `${Math.round(fps)} FPS`;

    const rangeEl = this.bottomDockEl.querySelector<HTMLElement>("#telem-range");
    if (rangeEl) {
      const locked =
        snap.activeRange.mode === ActiveRangeModeTs.Locked || snap.activeRange.sigmaRM <= 95;
      rangeEl.textContent = `${Math.round(snap.activeRange.rHatM)}m ±${Math.round(snap.activeRange.sigmaRM)}m ${locked ? s.telemetry.locked : s.telemetry.swerve}`;
      rangeEl.dataset.state = locked ? "nominal" : "caution";
    }

    const scEl = this.bottomDockEl.querySelector<HTMLElement>("#telem-sc3bf");
    if (scEl) {
      scEl.textContent = snap.sc3bfSafe ? s.telemetry.safe : s.telemetry.evade;
      scEl.dataset.state = snap.sc3bfSafe ? "nominal" : "alert";
    }

    const bankEl = this.bottomDockEl.querySelector<HTMLElement>("#telem-bank");
    if (bankEl) {
      const bankDeg = (snap.ownship.rollRad * 180) / Math.PI;
      bankEl.textContent = `${bankDeg >= 0 ? "+" : ""}${bankDeg.toFixed(1)}°`;
    }

    const spdEl = this.bottomDockEl.querySelector<HTMLElement>("#telem-speed");
    if (spdEl) {
      spdEl.textContent = `${Math.round(snap.ownship.tasMps * 1.94384)} kt`;
    }

    // Floating alert pill for Shield Override / Envelope Breach / Runway Touchdown
    if (snap.shieldOverrideActive) {
      this.alertEl.dataset.visible = "true";
      this.alertEl.dataset.kind = "override";
      this.alertEl.textContent = `${s.telemetry.shieldOverridePrefix}: ${snap.shieldReason}`;
    } else if (snap.envelopeBreach) {
      this.alertEl.dataset.visible = "true";
      this.alertEl.dataset.kind = "breach";
      this.alertEl.textContent = `${s.telemetry.breachPrefix} (${snap.breachReason ?? "SHIELD OFF"})`;
    } else if (snap.touchdown.landed) {
      this.alertEl.dataset.visible = "true";
      this.alertEl.dataset.kind = "touchdown";
      this.alertEl.textContent = `${s.telemetry.touchdownPrefix}: ${snap.touchdown.grade} · ${Math.round(snap.touchdown.sinkRateFpm)} fpm · ${snap.touchdown.centerlineOffsetM >= 0 ? "+" : ""}${snap.touchdown.centerlineOffsetM.toFixed(1)}m`;
    } else {
      this.alertEl.dataset.visible = "false";
    }

    // Fallback synthetic canvases only for slots without a live/uploaded image
    if (this.drawerOpen && !this.bridge.isOnline()) {
      for (const ch of CHANNEL_ORDER) {
        if (this.channelHasCustomImage.has(ch)) continue;
        const canvas = this.channelCanvases.get(ch);
        const img = this.channelImgs.get(ch);
        if (canvas) {
          canvas.style.display = "block";
          renderFallbackSensorCanvas(canvas, ch, snap);
        }
        if (img) img.classList.add("hidden");
      }
      this.syncBridgeBadge();
    }
  }

  /** Builds the 4-channel sensor drawer DOM ONCE so `<img>` previews persist across UI clicks (D16-W19-m3). */
  private mountDrawerOnce(): void {
    const channelCards = CHANNEL_ORDER.map(
      (id) =>
        `<div class="studio-channel-card">` +
        `<div class="studio-channel-meta">` +
        `<span data-ch-title="${id}"></span>` +
        `<span class="studio-channel-badges">` +
        `<span class="studio-channel-sha hidden" data-ch-sha="${id}"></span>` +
        `<span class="studio-channel-lat" data-ch-lat="${id}">LIVE</span>` +
        `</span>` +
        `</div>` +
        `<div class="studio-channel-media">` +
        `<canvas width="288" height="100" data-ch-canvas="${id}"></canvas>` +
        `<img class="hidden" alt="${esc(id)}" data-ch-img="${id}" />` +
        `</div>` +
        `</div>`,
    ).join("");

    const sectorCells = Array.from(
      { length: EO_SECTOR_COUNT },
      (_, i) => `<span class="studio-sector-cell" data-eo-sector="${i}" data-band="standby"></span>`,
    ).join("");

    const assuranceChips = getFormalAssuranceBadges(this.locale)
      .map(
        (b) =>
          `<span class="studio-assurance-chip" data-assurance-id="${b.id}" data-tone="${b.tone}" title="${esc(b.title)}">${esc(b.text)}</span>`,
      )
      .join("");

    this.drawerEl.innerHTML =
      `<div class="studio-drawer-head">` +
      `<span id="studio-drawer-title" class="studio-drawer-title"></span>` +
      `<span id="studio-bridge-badge" class="studio-bridge-badge" data-online="false"></span>` +
      `</div>` +
      `<div class="studio-channel-grid">${channelCards}</div>` +
      `<section id="studio-v52-avionics-panel" class="studio-v52-panel" aria-label="v5.2 Avionics Health and Assurance">` +
      `<div class="studio-v52-head">` +
      `<span id="studio-v52-panel-title" class="studio-v52-title"></span>` +
      `</div>` +
      `<span id="studio-v52-sector-badge" class="studio-v52-badge studio-v52-sector-badge" data-state="standby"></span>` +
      `<div id="studio-v52-sector-bar" class="studio-sector-bar" role="img" aria-label="12-Sector EO Optical Quality">${sectorCells}</div>` +
      `<div class="studio-v52-guards-grid">` +
      `<span id="studio-v52-lwir-badge" class="studio-v52-badge" data-state="standby"></span>` +
      `<span id="studio-v52-fsync-badge" class="studio-v52-badge" data-state="standby"></span>` +
      `</div>` +
      `<div id="studio-v52-assurance-title" class="studio-v52-subhead"></div>` +
      `<div class="studio-assurance-chips">` +
      assuranceChips +
      `<span id="studio-v52-tbc-badge" class="studio-v52-badge studio-v52-tbc-chip" data-state="standby"></span>` +
      `</div>` +
      `</section>` +
      `<div id="studio-nav-v5-spec" class="studio-nav-v5-spec"></div>` +
      `<div class="studio-upload-zone" id="studio-upload-zone">` +
      `<div id="studio-upload-hint" class="studio-upload-hint"></div>` +
      `<div class="studio-upload-row">` +
      `<select id="studio-upload-ch" class="studio-select">` +
      `<option value="d2" selected>XVS-D2</option>` +
      `<option value="land">XVS-Land</option>` +
      `<option value="thermal">XVS-Thermal</option>` +
      `<option value="nav">XVS-Nav v5.0</option>` +
      `</select>` +
      `<input id="studio-upload-input" type="file" accept="image/*" style="display:none" />` +
      `<button id="studio-upload-btn" class="studio-btn" style="flex:1"></button>` +
      `</div>` +
      `</div>`;

    for (const id of CHANNEL_ORDER) {
      const t = this.drawerEl.querySelector<HTMLElement>(`[data-ch-title="${id}"]`);
      const c = this.drawerEl.querySelector<HTMLCanvasElement>(`[data-ch-canvas="${id}"]`);
      const img = this.drawerEl.querySelector<HTMLImageElement>(`[data-ch-img="${id}"]`);
      const lat = this.drawerEl.querySelector<HTMLElement>(`[data-ch-lat="${id}"]`);
      const sha = this.drawerEl.querySelector<HTMLElement>(`[data-ch-sha="${id}"]`);
      if (t) this.channelTitleEls.set(id, t);
      if (c) this.channelCanvases.set(id, c);
      if (img) this.channelImgs.set(id, img);
      if (lat) this.channelLatEls.set(id, lat);
      if (sha) this.channelShaEls.set(id, sha);
    }

    this.sectorCellEls.length = 0;
    for (let i = 0; i < EO_SECTOR_COUNT; i++) {
      const cell = this.drawerEl.querySelector<HTMLElement>(`[data-eo-sector="${i}"]`);
      if (cell) this.sectorCellEls.push(cell);
    }
    this.assuranceChipEls.clear();
    this.drawerEl.querySelectorAll<HTMLElement>("[data-assurance-id]").forEach((el) => {
      const key = el.dataset.assuranceId;
      if (key) this.assuranceChipEls.set(key, el);
    });

    const chSel = this.drawerEl.querySelector<HTMLSelectElement>("#studio-upload-ch");
    chSel?.addEventListener("change", () => {
      this.uploadChannel = chSel.value as SensorChannelId;
    });

    const fileInput = this.drawerEl.querySelector<HTMLInputElement>("#studio-upload-input");
    this.drawerEl.querySelector<HTMLButtonElement>("#studio-upload-btn")?.addEventListener("click", () => {
      fileInput?.click();
    });

    const handleFile = async (file: File) => {
      const buf = await file.arrayBuffer();
      const res = await this.bridge.analyzeImage(this.uploadChannel, buf, file.type || "image/jpeg");
      if (res) {
        this.applyBridgeChannels({ [this.uploadChannel]: res }, true);
      }
    };

    fileInput?.addEventListener("change", () => {
      const f = fileInput.files?.[0];
      if (f) void handleFile(f);
    });

    const dropZone = this.drawerEl.querySelector<HTMLElement>("#studio-upload-zone");
    dropZone?.addEventListener("dragover", (e) => e.preventDefault());
    dropZone?.addEventListener("drop", (e) => {
      e.preventDefault();
      const f = e.dataTransfer?.files?.[0];
      if (f) void handleFile(f);
    });
  }

  private syncDrawerText(): void {
    const full = getStudioStrings(this.locale);
    const s = full.drawer;
    setTextIfChanged(this.drawerEl.querySelector<HTMLElement>("#studio-drawer-title"), s.title);
    setTextIfChanged(
      this.drawerEl.querySelector<HTMLElement>("#studio-v52-panel-title"),
      s.avionicsTitle,
    );
    setTextIfChanged(
      this.drawerEl.querySelector<HTMLElement>("#studio-v52-assurance-title"),
      s.assuranceTitle,
    );
    for (const badge of getFormalAssuranceBadges(this.locale)) {
      const chip = this.assuranceChipEls.get(badge.id);
      setTextIfChanged(chip, badge.text);
      setAttrIfChanged(chip, "title", badge.title);
    }
    setTextIfChanged(
      this.drawerEl.querySelector<HTMLElement>("#studio-nav-v5-spec"),
      full.navV5Spec,
    );
    setTextIfChanged(
      this.drawerEl.querySelector<HTMLElement>("#studio-upload-hint"),
      s.uploadHint,
    );
    setTextIfChanged(
      this.drawerEl.querySelector<HTMLElement>("#studio-upload-btn"),
      s.uploadBtn,
    );
    for (const id of CHANNEL_ORDER) {
      setTextIfChanged(this.channelTitleEls.get(id), s.channels[id]);
    }
    this.syncV52AvionicsPanel();
    this.syncBridgeBadge();
  }

  private mountWeatherOnce(): void {
    this.weatherEl.innerHTML =
      `<div id="w-title" class="studio-weather-title"></div>` +
      `<label class="studio-slider-row"><div class="studio-slider-head"><span id="w-lbl-tod"></span><b id="w-val-tod">16.5h</b></div><input id="w-tod" type="range" min="5" max="22" step="0.25" value="16.5" /></label>` +
      `<label class="studio-slider-row"><div class="studio-slider-head"><span id="w-lbl-vis"></span><b id="w-val-vis">18 km</b></div><input id="w-vis" type="range" min="0.5" max="25" step="0.5" value="18" /></label>` +
      `<label class="studio-slider-row"><div class="studio-slider-head"><span id="w-lbl-wind"></span><b id="w-val-wind">12 kt</b></div><input id="w-wind" type="range" min="0" max="45" step="1" value="12" /></label>` +
      `<label class="studio-slider-row"><div class="studio-slider-head"><span id="w-lbl-cloud"></span><b id="w-val-cloud">42%</b></div><input id="w-cloud" type="range" min="0" max="1" step="0.05" value="0.42" /></label>` +
      `<label class="studio-slider-row"><div class="studio-slider-head"><span id="w-lbl-rain"></span><b id="w-val-rain">0%</b></div><input id="w-rain" type="range" min="0" max="1" step="0.05" value="0" /></label>` +
      `<div class="studio-weather-btns">` +
      `<button id="w-eng-btn" class="studio-btn" style="flex:1"></button>` +
      `<button id="w-rst-btn" class="studio-btn"></button>` +
      `</div>`;

    const bindSlider = (id: string, handler: (v: number) => void) => {
      const el = this.weatherEl.querySelector<HTMLInputElement>(`#${id}`);
      el?.addEventListener("input", () => handler(Number(el.value)));
    };
    bindSlider("w-tod", (v) => {
      this.weather = { ...this.weather, timeOfDayH: v };
      this.syncWeatherUi();
      this.callbacks.onWeatherChange({ timeOfDayH: v });
    });
    bindSlider("w-vis", (v) => {
      this.weather = { ...this.weather, visibilityKm: v };
      this.syncWeatherUi();
      this.callbacks.onWeatherChange({ visibilityKm: v });
    });
    bindSlider("w-wind", (v) => {
      this.weather = { ...this.weather, crosswindKt: v };
      this.syncWeatherUi();
      this.callbacks.onWeatherChange({ crosswindKt: v });
    });
    bindSlider("w-cloud", (v) => {
      this.weather = { ...this.weather, cloudCover: v };
      this.syncWeatherUi();
      this.callbacks.onWeatherChange({ cloudCover: v });
    });
    bindSlider("w-rain", (v) => {
      this.weather = { ...this.weather, rainIntensity: v };
      this.syncWeatherUi();
      this.callbacks.onWeatherChange({ rainIntensity: v });
    });

    this.weatherEl.querySelector<HTMLButtonElement>("#w-eng-btn")?.addEventListener("click", () => {
      this.callbacks.onToggleEngineOut();
    });
    this.weatherEl.querySelector<HTMLButtonElement>("#w-rst-btn")?.addEventListener("click", () => {
      this.callbacks.onResetApproach();
    });
  }

  private syncWeatherUi(): void {
    const w = getStudioStrings(this.locale).weather;
    const setTxt = (id: string, txt: string) => {
      const el = this.weatherEl.querySelector<HTMLElement>(`#${id}`);
      if (el) el.textContent = txt;
    };
    const setVal = (id: string, val: number) => {
      const el = this.weatherEl.querySelector<HTMLInputElement>(`#${id}`);
      if (el) el.value = String(val);
    };
    setTxt("w-title", w.title);
    setTxt("w-lbl-tod", w.timeOfDay);
    setTxt("w-lbl-vis", w.visibility);
    setTxt("w-lbl-wind", w.crosswind);
    setTxt("w-lbl-cloud", w.cloudCover);
    setTxt("w-lbl-rain", w.rain);
    setTxt("w-eng-btn", w.engineOutBtn);
    setTxt("w-rst-btn", w.resetBtn);

    const h = Math.floor(this.weather.timeOfDayH);
    const m = Math.round((this.weather.timeOfDayH - h) * 60);
    setTxt("w-val-tod", `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`);
    setTxt("w-val-vis", `${this.weather.visibilityKm.toFixed(1)} km`);
    setTxt("w-val-wind", `${Math.round(this.weather.crosswindKt)} kt`);
    setTxt("w-val-cloud", `${Math.round(this.weather.cloudCover * 100)}%`);
    setTxt("w-val-rain", `${Math.round(this.weather.rainIntensity * 100)}%`);

    setVal("w-tod", this.weather.timeOfDayH);
    setVal("w-vis", this.weather.visibilityKm);
    setVal("w-wind", this.weather.crosswindKt);
    setVal("w-cloud", this.weather.cloudCover);
    setVal("w-rain", this.weather.rainIntensity);
  }

  private renderBars(): void {
    const s = getStudioStrings(this.locale);
    const modeLabel = this.visualMode === "photoreal" ? s.modePhotoreal : s.modeObsidian;
    const ctrlLabel = this.controlMode === "manual" ? s.controlManual : s.controlAutopilot;
    const shieldLabel = this.shieldEnabled ? s.shieldOn : s.shieldOff;
    const bannerText = s.scenarios[this.scenario].banner;

    this.topbarEl.innerHTML =
      `<div class="studio-brand-wrap">` +
      `<span class="studio-dot"></span>` +
      `<span class="studio-brand">${esc(s.brandTitle)}</span>` +
      `<span class="studio-ver">${esc(s.brandVersion)}</span>` +
      `</div>` +
      `<div class="studio-scenario-banner" title="${esc(bannerText)}">${esc(bannerText)}</div>` +
      `<div class="studio-top-actions">` +
      `<span class="studio-pill studio-pill-emerald" title="${esc(s.brandBadgeTitle)}">${esc(s.brandBadge)}</span>` +
      `<button id="studio-mode-btn" class="studio-btn" data-active="${this.visualMode === "photoreal"}">${esc(modeLabel)}</button>` +
      `<button id="studio-ctrl-btn" class="studio-btn" data-active="${this.controlMode === "manual"}">${esc(ctrlLabel)}</button>` +
      `<button id="studio-shield-btn" class="studio-btn" data-tone="${this.shieldEnabled ? "emerald" : "rose"}">${esc(shieldLabel)}</button>` +
      `<button id="studio-weather-btn" class="studio-btn" data-active="${this.weatherOpen}">${esc(s.weatherBtn)}</button>` +
      `<button id="studio-bench-btn" class="studio-btn" data-active="true">${esc(s.videoLabBtn)}</button>` +
      `<button id="studio-sensors-btn" class="studio-btn" data-active="${this.drawerOpen}">${esc(s.sensorsToggle)}</button>` +
      `<button id="studio-locale-btn" class="studio-btn">${esc(s.localeToggle)}</button>` +
      `<span id="studio-fps-pill" class="studio-pill studio-pill-cyan">60 FPS</span>` +
      `</div>`;

    const camButtons = CAMERA_ORDER.map(
      (id) =>
        `<button class="studio-btn" data-studio-cam="${id}" data-active="${this.cameraPreset === id}">${esc(s.cameras[id])}</button>`,
    ).join("");

    const scenButtons = SCENARIO_ORDER.map(
      (id) =>
        `<button class="studio-btn" data-studio-scen="${id}" data-active="${this.scenario === id}">${esc(s.scenarios[id].label)}</button>`,
    ).join("");

    this.bottomDockEl.innerHTML =
      `<div class="studio-manual-hint${this.controlMode === "manual" ? "" : " hidden"}">${esc(s.manualHint)}</div>` +
      `<div class="studio-dock-strip">` +
      `<div class="studio-telem-chip"><span class="studio-telem-k">${esc(s.telemetry.range)}</span><span id="telem-range" class="studio-telem-v" data-state="caution">—</span></div>` +
      `<div class="studio-telem-chip"><span class="studio-telem-k">${esc(s.telemetry.sc3bf)}</span><span id="telem-sc3bf" class="studio-telem-v" data-state="nominal">6/6</span></div>` +
      `<div class="studio-telem-chip"><span class="studio-telem-k">${esc(s.telemetry.bank)}</span><span id="telem-bank" class="studio-telem-v">0.0°</span></div>` +
      `<div class="studio-telem-chip"><span class="studio-telem-k">${esc(s.telemetry.speed)}</span><span id="telem-speed" class="studio-telem-v">101 kt</span></div>` +
      `<span class="studio-sep"></span>` +
      camButtons +
      `<span class="studio-sep"></span>` +
      `<button id="studio-dome-btn" class="studio-btn" data-active="${this.domeVisible}">${esc(s.domeToggle)}</button>` +
      `</div>` +
      `<div class="studio-dock-strip">${scenButtons}</div>`;

    this.wireBarEvents();
  }

  private wireBarEvents(): void {
    this.topbarEl.querySelector<HTMLButtonElement>("#studio-locale-btn")?.addEventListener("click", () => {
      this.setLocale(this.locale === "tr" ? "en" : "tr");
    });

    this.topbarEl.querySelector<HTMLButtonElement>("#studio-mode-btn")?.addEventListener("click", () => {
      const next: VisualWorldMode = this.visualMode === "photoreal" ? "obsidian" : "photoreal";
      this.setVisualMode(next);
      this.callbacks.onVisualModeChange(next);
    });

    this.topbarEl.querySelector<HTMLButtonElement>("#studio-ctrl-btn")?.addEventListener("click", () => {
      const next: FlightControlMode = this.controlMode === "autopilot" ? "manual" : "autopilot";
      this.setControlMode(next);
      this.callbacks.onControlModeChange(next);
    });

    this.topbarEl.querySelector<HTMLButtonElement>("#studio-shield-btn")?.addEventListener("click", () => {
      const next = !this.shieldEnabled;
      this.setShieldEnabled(next);
      this.callbacks.onShieldToggle(next);
    });

    this.topbarEl.querySelector<HTMLButtonElement>("#studio-weather-btn")?.addEventListener("click", () => {
      this.weatherOpen = !this.weatherOpen;
      this.weatherEl.dataset.open = String(this.weatherOpen);
      this.renderBars();
    });

    this.topbarEl.querySelector<HTMLButtonElement>("#studio-bench-btn")?.addEventListener("click", () => {
      this.callbacks.onOpenVideoBench();
    });

    this.topbarEl.querySelector<HTMLButtonElement>("#studio-sensors-btn")?.addEventListener("click", () => {
      this.setDrawerOpen(!this.drawerOpen);
    });

    this.bottomDockEl.querySelector<HTMLButtonElement>("#studio-dome-btn")?.addEventListener("click", () => {
      this.setDomeVisible(!this.domeVisible);
      this.callbacks.onDomeToggle(this.domeVisible);
    });

    this.bottomDockEl.querySelectorAll<HTMLButtonElement>("[data-studio-cam]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const preset = btn.dataset.studioCam as StudioCameraPreset;
        this.setCameraPreset(preset);
        this.callbacks.onCameraChange(preset);
      });
    });

    this.bottomDockEl.querySelectorAll<HTMLButtonElement>("[data-studio-scen]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const scen = btn.dataset.studioScen as StudioScenarioId;
        this.setScenario(scen);
        this.callbacks.onScenarioChange(scen);
      });
    });
  }
}
