/**
 * OpenXVS Studio Glassmorphic HUD + Persistent 4-Channel Sensor Drawer + Weather Popover
 * + Live Avionics Health & Formal Assurance Sub-Tab + Multi-Ship Coord (F1/F4).
 *
 * Engineering Guarantees:
 *   - Mounts topbar, bottom dock, weather popover, and right sensor drawer DOM ONCE on boot;
 *     every click updates text/attributes in place (`document.activeElement` never destroyed).
 *   - Caches all HUD DOM element refs on mount; throttles telemetry text updates to 10 Hz
 *     and synthetic fallback canvas draws to 8 Hz with diff-guarded writes (`setTextIfChanged`).
 *   - Right drawer splits `4-Ch Feeds` and `Avionics & Assurance` into two clean sub-tabs so
 *     all channel cards + Upload CTA fit with 0 px vertical overflow at 1280x800 and 1440x900.
 *   - Updates `document.documentElement.lang` on every locale switch (eliminates dotted `İ` bug).
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
  formatFallbackChannelCaption,
  renderFallbackSensorCanvas,
  type BridgeChannelResult,
  type NativeSensorHealthPayload,
  type SensorChannelId,
  type StudioBridgeClient,
} from "./bridge_client.ts";
import {
  STUDIO_ASSURANCE_V52_SPEC,
  formatBreachReason,
  formatShieldReason,
  formatTouchdownGrade,
  getStudioStrings,
  type StudioCameraPreset,
  type StudioLocale,
} from "./i18n.ts";
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

export type DrawerTabId = "feeds" | "avionics";

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
  readonly onDrawerToggle?: (open: boolean) => void;
  readonly onLocaleChange?: (locale: StudioLocale) => void;
  readonly onRetryBridge?: () => Promise<boolean> | void;
}

const SCENARIO_ORDER: readonly StudioScenarioId[] = ["cbdr", "forcedland", "thermal_fog", "gps_spoof"];
const CAMERA_ORDER: readonly StudioCameraPreset[] = ["chase", "wingman", "overview", "cockpit"];
const CHANNEL_ORDER: readonly SensorChannelId[] = ["d2", "land", "thermal", "nav"];

const TELEMETRY_TEXT_INTERVAL_MS = 100; // 10 Hz
const FALLBACK_CANVAS_INTERVAL_MS = 125; // 8 Hz

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
        "SensorHealthMonitor: 12-Sector EO, LWIR FFC/Contrast-Collapse, Dual-Tier FSYNC <=1ms & M-of-N TBC (awaiting telemetry)",
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
  private readonly abortCtrl = new AbortController();

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
  private benchOpen = false;
  private domeVisible = false;
  private drawerOpen = typeof window !== "undefined" && window.innerWidth >= 1280;
  private drawerTab: DrawerTabId = "feeds";
  private drawerBeforeCockpit: boolean | null = null;
  private uploadChannel: SensorChannelId = "d2";
  private uploadBusy = false;
  private uploadFeedbackKey: "none" | "offline" | "failed" | "custom" = "none";
  private uploadCustomMsg = "";
  private retryBridgeBusy = false;
  private sensorHealth: NativeSensorHealthPayload | null = null;

  private lastTelemetryTextMs = -Infinity;
  private lastFallbackCanvasMs = -Infinity;
  private lastSyncedBridgeOnline: boolean | null = null;
  private lastSyncedBridgeTransport: string | null = null;

  // Cached Topbar & Status Strip DOM refs (zero per-frame querySelector)
  private brandTitleEl: HTMLElement | null = null;
  private brandVerEl: HTMLElement | null = null;
  private bannerEl: HTMLElement | null = null;
  private brandBadgeEl: HTMLElement | null = null;
  private modeBtnEl: HTMLButtonElement | null = null;
  private ctrlBtnEl: HTMLButtonElement | null = null;
  private shieldBtnEl: HTMLButtonElement | null = null;
  private weatherBtnEl: HTMLButtonElement | null = null;
  private benchBtnEl: HTMLButtonElement | null = null;
  private sensorsBtnEl: HTMLButtonElement | null = null;
  private localeBtnEl: HTMLButtonElement | null = null;
  private fpsPillEl: HTMLElement | null = null;
  private sensorHealthBadgeEl: HTMLElement | null = null;
  private multishipBadgeEl: HTMLElement | null = null;

  // Cached Bottom Dock DOM refs
  private manualHintEl: HTMLElement | null = null;
  private telemKeyRangeEl: HTMLElement | null = null;
  private telemKeySc3bfEl: HTMLElement | null = null;
  private telemKeyBankEl: HTMLElement | null = null;
  private telemKeySpeedEl: HTMLElement | null = null;
  private telemRangeEl: HTMLElement | null = null;
  private telemSc3bfEl: HTMLElement | null = null;
  private telemBankEl: HTMLElement | null = null;
  private telemSpeedEl: HTMLElement | null = null;
  private domeBtnEl: HTMLButtonElement | null = null;
  private readonly camBtnEls = new Map<StudioCameraPreset, HTMLButtonElement>();
  private readonly scenBtnEls = new Map<StudioScenarioId, HTMLButtonElement>();

  // Cached Drawer DOM refs
  private drawerTitleEl: HTMLElement | null = null;
  private bridgeBadgeEl: HTMLElement | null = null;
  private retryBridgeBtnEl: HTMLButtonElement | null = null;
  private tabFeedsBtnEl: HTMLButtonElement | null = null;
  private tabAvionicsBtnEl: HTMLButtonElement | null = null;
  private paneFeedsEl: HTMLElement | null = null;
  private paneAvionicsEl: HTMLElement | null = null;
  private avionicsTitleEl: HTMLElement | null = null;
  private assuranceTitleEl: HTMLElement | null = null;
  private specsSummaryEl: HTMLElement | null = null;
  private navSpecEl: HTMLElement | null = null;
  private assuranceSpecEl: HTMLElement | null = null;
  private sectorBadgeEl: HTMLElement | null = null;
  private sectorBarEl: HTMLElement | null = null;
  private lwirBadgeEl: HTMLElement | null = null;
  private fsyncBadgeEl: HTMLElement | null = null;
  private tbcBadgeEl: HTMLElement | null = null;
  private uploadHintEl: HTMLElement | null = null;
  private uploadChSelectEl: HTMLSelectElement | null = null;
  private uploadBtnEl: HTMLButtonElement | null = null;
  private uploadStatusEl: HTMLElement | null = null;

  // Cached Weather DOM refs
  private readonly weatherTextEls = new Map<string, HTMLElement>();
  private readonly weatherInputEls = new Map<string, HTMLInputElement>();

  private readonly channelTitleEls = new Map<SensorChannelId, HTMLElement>();
  private readonly channelCaptionEls = new Map<SensorChannelId, HTMLElement>();
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

    if (typeof document !== "undefined") {
      document.documentElement.lang = this.locale;
    }

    this.topbarEl = document.createElement("header");
    this.topbarEl.className = "studio-topbar";

    this.statusStripEl = document.createElement("div");
    this.statusStripEl.className = "studio-status-strip";
    this.statusStripEl.innerHTML =
      `<span id="studio-sensor-health-badge" class="studio-status-badge" data-state="standby"></span>` +
      `<span id="studio-multiship-badge" class="studio-status-badge" data-state="standby"></span>`;
    this.sensorHealthBadgeEl = this.statusStripEl.querySelector("#studio-sensor-health-badge");
    this.multishipBadgeEl = this.statusStripEl.querySelector("#studio-multiship-badge");

    this.alertEl = document.createElement("div");
    this.alertEl.id = "studio-flight-alert";
    this.alertEl.className = "studio-flight-alert";
    this.alertEl.setAttribute("role", "status");
    this.alertEl.setAttribute("aria-live", "polite");
    this.alertEl.dataset.visible = "false";

    this.weatherEl = document.createElement("div");
    this.weatherEl.id = "studio-weather-popover";
    this.weatherEl.className = "studio-weather-popover";
    this.weatherEl.setAttribute("role", "dialog");
    this.weatherEl.dataset.open = "false";

    this.bottomDockEl = document.createElement("div");
    this.bottomDockEl.className = "studio-bottom-dock";

    this.drawerEl = document.createElement("aside");
    this.drawerEl.id = "studio-sensor-drawer";
    this.drawerEl.className = "studio-sensor-drawer";
    this.drawerEl.dataset.open = String(this.drawerOpen);
    this.applyDrawerInert(this.drawerOpen);

    this.host.appendChild(this.topbarEl);
    this.host.appendChild(this.statusStripEl);
    this.host.appendChild(this.alertEl);
    this.host.appendChild(this.weatherEl);
    this.host.appendChild(this.bottomDockEl);
    this.host.appendChild(this.drawerEl);

    this.mountTopbarOnce();
    this.mountBottomDockOnce();
    this.mountDrawerOnce();
    this.mountWeatherOnce();
    this.bindGlobalPopoverDismiss();

    this.syncBars();
    this.syncDrawerText();
    this.syncWeatherUi();
    this.syncStatusBadges();
    this.syncBridgeBadge(true);
  }

  dispose(): void {
    this.abortCtrl.abort();
  }

  private applyDrawerInert(open: boolean): void {
    this.drawerEl.inert = !open;
    setAttrIfChanged(this.drawerEl, "aria-hidden", String(!open));
  }

  private bindGlobalPopoverDismiss(): void {
    if (typeof document === "undefined") return;
    document.addEventListener(
      "pointerdown",
      (ev) => {
        if (!this.weatherOpen) return;
        const target = ev.target as Node | null;
        if (!target) return;
        if (this.weatherEl.contains(target) || this.weatherBtnEl?.contains(target)) {
          return;
        }
        this.setWeatherOpen(false, false);
      },
      { signal: this.abortCtrl.signal },
    );
  }

  setLocale(next: StudioLocale): void {
    this.locale = next;
    if (typeof document !== "undefined") {
      document.documentElement.lang = next;
    }
    this.syncBars();
    this.syncDrawerText();
    this.syncWeatherUi();
    this.syncStatusBadges();
    this.syncBridgeBadge(true);
    this.syncUploadStatusText();
    this.callbacks.onLocaleChange?.(next);
  }

  getLocale(): StudioLocale {
    return this.locale;
  }

  setScenario(next: StudioScenarioId): void {
    this.scenario = next;
    this.syncBars();
  }

  setCameraPreset(next: StudioCameraPreset): void {
    const prev = this.cameraPreset;
    this.cameraPreset = next;
    // Auto-collapse right drawer in Cockpit view so both G1000 displays stay 100% unobstructed;
    // restore previous drawer state automatically when returning to external cameras.
    if (next === "cockpit" && prev !== "cockpit") {
      this.drawerBeforeCockpit = this.drawerOpen;
      this.setDrawerOpen(false);
    } else if (next !== "cockpit" && prev === "cockpit" && this.drawerBeforeCockpit !== null) {
      this.setDrawerOpen(this.drawerBeforeCockpit);
      this.drawerBeforeCockpit = null;
    }
    this.syncBars();
  }

  setVisualMode(next: VisualWorldMode): void {
    this.visualMode = next;
    this.syncBars();
  }

  setControlMode(next: FlightControlMode): void {
    this.controlMode = next;
    this.syncBars();
  }

  setShieldEnabled(enabled: boolean): void {
    this.shieldEnabled = enabled;
    this.syncBars();
  }

  setWeather(next: WeatherState): void {
    this.weather = { ...next };
    this.syncWeatherUi();
  }

  setWeatherOpen(open: boolean, restoreFocus = false): void {
    if (this.weatherOpen === open) return;
    this.weatherOpen = open;
    this.weatherEl.dataset.open = String(open);
    setAttrIfChanged(this.weatherBtnEl, "data-active", String(open));
    setAttrIfChanged(this.weatherBtnEl, "aria-expanded", String(open));
    if (!open && restoreFocus) {
      this.weatherBtnEl?.focus();
    }
  }

  isWeatherOpen(): boolean {
    return this.weatherOpen;
  }

  setBenchOpen(open: boolean): void {
    this.benchOpen = open;
    setAttrIfChanged(this.benchBtnEl, "data-active", String(open));
    setAttrIfChanged(this.benchBtnEl, "aria-expanded", String(open));
  }

  setDomeVisible(visible: boolean): void {
    this.domeVisible = visible;
    this.syncBars();
  }

  setDrawerOpen(open: boolean): void {
    const changed = this.drawerOpen !== open;
    this.drawerOpen = open;
    this.drawerEl.dataset.open = String(open);
    this.applyDrawerInert(open);
    setAttrIfChanged(this.sensorsBtnEl, "data-active", String(open));
    setAttrIfChanged(this.sensorsBtnEl, "aria-expanded", String(open));
    if (changed) {
      this.callbacks.onDrawerToggle?.(open);
    }
  }

  isDrawerOpen(): boolean {
    return this.drawerOpen;
  }

  setDrawerTab(tab: DrawerTabId): void {
    this.drawerTab = tab;
    const isFeeds = tab === "feeds";
    setAttrIfChanged(this.tabFeedsBtnEl, "aria-selected", String(isFeeds));
    setAttrIfChanged(this.tabFeedsBtnEl, "data-active", String(isFeeds));
    setAttrIfChanged(this.tabAvionicsBtnEl, "aria-selected", String(!isFeeds));
    setAttrIfChanged(this.tabAvionicsBtnEl, "data-active", String(!isFeeds));
    this.paneFeedsEl?.classList.toggle("hidden", !isFeeds);
    this.paneAvionicsEl?.classList.toggle("hidden", isFeeds);
  }

  getDrawerTab(): DrawerTabId {
    return this.drawerTab;
  }

  applySensorHealth(health: NativeSensorHealthPayload | null): void {
    this.sensorHealth = health;
    this.syncStatusBadges();
  }

  private syncStatusBadges(): void {
    if (this.sensorHealthBadgeEl) {
      const info = formatSensorHealthBadge(this.sensorHealth, this.locale);
      setTextIfChanged(this.sensorHealthBadgeEl, info.text);
      setAttrIfChanged(this.sensorHealthBadgeEl, "title", info.title);
      setAttrIfChanged(this.sensorHealthBadgeEl, "data-state", info.state);
    }
    if (this.multishipBadgeEl) {
      const info = formatMultiShipCoordBadge(this.sensorHealth, this.locale);
      setTextIfChanged(this.multishipBadgeEl, info.text);
      setAttrIfChanged(this.multishipBadgeEl, "title", info.title);
      setAttrIfChanged(this.multishipBadgeEl, "data-state", info.state);
    }
    this.syncAvionicsPanel();
  }

  private syncAvionicsPanel(): void {
    const sectorInfo = formatSectorOpticalHealth(this.sensorHealth, this.locale);
    setTextIfChanged(this.sectorBadgeEl, sectorInfo.text);
    setAttrIfChanged(this.sectorBadgeEl, "title", sectorInfo.title);
    setAttrIfChanged(this.sectorBadgeEl, "data-state", sectorInfo.state);

    setAttrIfChanged(this.sectorBarEl, "title", sectorInfo.title);
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
    setTextIfChanged(this.lwirBadgeEl, lwirInfo.text);
    setAttrIfChanged(this.lwirBadgeEl, "title", lwirInfo.title);
    setAttrIfChanged(this.lwirBadgeEl, "data-state", lwirInfo.state);

    const fsyncInfo = formatFsyncPrecisionBadge(this.sensorHealth, this.locale);
    setTextIfChanged(this.fsyncBadgeEl, fsyncInfo.text);
    setAttrIfChanged(this.fsyncBadgeEl, "title", fsyncInfo.title);
    setAttrIfChanged(this.fsyncBadgeEl, "data-state", fsyncInfo.state);

    const tbcInfo = formatBearingTbcBadge(this.sensorHealth, this.locale);
    setTextIfChanged(this.tbcBadgeEl, tbcInfo.text);
    setAttrIfChanged(this.tbcBadgeEl, "title", tbcInfo.title);
    setAttrIfChanged(this.tbcBadgeEl, "data-state", tbcInfo.state);
  }

  syncBridgeBadge(force = false): void {
    const online = this.bridge.isOnline();
    const transport = this.bridge.getTransport();
    if (
      !force &&
      this.lastSyncedBridgeOnline === online &&
      this.lastSyncedBridgeTransport === transport
    ) {
      return;
    }
    this.lastSyncedBridgeOnline = online;
    this.lastSyncedBridgeTransport = transport;

    const s = getStudioStrings(this.locale);
    setAttrIfChanged(this.bridgeBadgeEl, "data-online", String(online));
    if (online && transport === "tauri-ipc") {
      setTextIfChanged(this.bridgeBadgeEl, s.drawer.bridgeNative);
      setAttrIfChanged(this.bridgeBadgeEl, "title", s.stackV5Spec);
    } else {
      setTextIfChanged(
        this.bridgeBadgeEl,
        online ? s.drawer.bridgeOnline : s.drawer.bridgeFallback,
      );
      setAttrIfChanged(
        this.bridgeBadgeEl,
        "title",
        online ? s.stackV5Spec : s.drawer.bridgeFallback,
      );
    }

    if (this.retryBridgeBtnEl) {
      const showRetry = !online && !this.bridge.isNativeTauriIpc();
      this.retryBridgeBtnEl.classList.toggle("hidden", !showRetry);
      this.retryBridgeBtnEl.disabled = this.retryBridgeBusy;
      setTextIfChanged(
        this.retryBridgeBtnEl,
        this.retryBridgeBusy ? s.drawer.retryingBridge : s.drawer.retryBridgeBtn,
      );
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
      const caption = this.channelCaptionEls.get(id);
      const latEl = this.channelLatEls.get(id);
      const shaEl = this.channelShaEls.get(id);

      if (res.imageDataUrl && img && canvas) {
        img.src = res.imageDataUrl;
        img.classList.remove("hidden");
        canvas.style.display = "none";
      }
      if (caption) {
        setTextIfChanged(caption, res.summary);
      }
      if (latEl) {
        if (id === "nav" && res.metrics && typeof res.metrics.xfeat_keypoints_count === "number") {
          const msStr = res.latencyMs !== null ? `${res.latencyMs.toFixed(1)}ms` : "LIVE";
          setTextIfChanged(latEl, `${msStr} · XF ${res.metrics.xfeat_keypoints_count}`);
        } else {
          setTextIfChanged(
            latEl,
            res.latencyMs !== null ? `${res.latencyMs.toFixed(1)} ms` : res.summary,
          );
        }
      }
      if (shaEl) {
        if (res.checkpointShaShort) {
          setTextIfChanged(shaEl, `SHA ${res.checkpointShaShort}`);
          shaEl.classList.remove("hidden");
        } else {
          shaEl.classList.add("hidden");
        }
      }
    }
    this.syncBridgeBadge();
  }

  updateTelemetry(snap: StudioSimSnapshot, fps: number, force = false): void {
    const nowMs = typeof performance !== "undefined" ? performance.now() : Date.now();

    if (force || nowMs - this.lastTelemetryTextMs >= TELEMETRY_TEXT_INTERVAL_MS) {
      this.lastTelemetryTextMs = nowMs;
      const s = getStudioStrings(this.locale);

      setTextIfChanged(this.fpsPillEl, `${Math.round(fps)} FPS`);

      if (this.telemRangeEl) {
        const locked =
          snap.activeRange.mode === ActiveRangeModeTs.Locked || snap.activeRange.sigmaRM <= 95;
        setTextIfChanged(
          this.telemRangeEl,
          `${Math.round(snap.activeRange.rHatM)}m ±${Math.round(snap.activeRange.sigmaRM)}m ${locked ? s.telemetry.locked : s.telemetry.swerve}`,
        );
        setAttrIfChanged(this.telemRangeEl, "data-state", locked ? "nominal" : "caution");
      }

      if (this.telemSc3bfEl) {
        setTextIfChanged(
          this.telemSc3bfEl,
          snap.sc3bfSafe ? s.telemetry.safe : s.telemetry.evade,
        );
        setAttrIfChanged(this.telemSc3bfEl, "data-state", snap.sc3bfSafe ? "nominal" : "alert");
      }

      if (this.telemBankEl) {
        const bankDeg = (snap.ownship.rollRad * 180) / Math.PI;
        setTextIfChanged(this.telemBankEl, `${bankDeg >= 0 ? "+" : ""}${bankDeg.toFixed(1)}°`);
      }

      if (this.telemSpeedEl) {
        setTextIfChanged(this.telemSpeedEl, `${Math.round(snap.ownship.tasMps * 1.94384)} kt`);
      }

      // Floating alert pill for Shield Override / Envelope Breach / Runway Touchdown (fully localized)
      if (snap.shieldOverrideActive) {
        setAttrIfChanged(this.alertEl, "data-visible", "true");
        setAttrIfChanged(this.alertEl, "data-kind", "override");
        setTextIfChanged(
          this.alertEl,
          `${s.telemetry.shieldOverridePrefix}: ${formatShieldReason(snap.shieldReason, this.locale)}`,
        );
      } else if (snap.envelopeBreach) {
        setAttrIfChanged(this.alertEl, "data-visible", "true");
        setAttrIfChanged(this.alertEl, "data-kind", "breach");
        setTextIfChanged(
          this.alertEl,
          `${s.telemetry.breachPrefix} (${formatBreachReason(snap.breachReason, this.locale)})`,
        );
      } else if (snap.touchdown.landed) {
        setAttrIfChanged(this.alertEl, "data-visible", "true");
        setAttrIfChanged(this.alertEl, "data-kind", "touchdown");
        setTextIfChanged(
          this.alertEl,
          `${s.telemetry.touchdownPrefix}: ${formatTouchdownGrade(snap.touchdown.grade, this.locale)} · ${Math.round(snap.touchdown.sinkRateFpm)} fpm · ${snap.touchdown.centerlineOffsetM >= 0 ? "+" : ""}${snap.touchdown.centerlineOffsetM.toFixed(1)}m`,
        );
      } else {
        setAttrIfChanged(this.alertEl, "data-visible", "false");
      }

      // Sync non-cropped HTML telemetry captions for fallback channels at 10 Hz
      if (this.drawerOpen && this.drawerTab === "feeds" && !this.bridge.isOnline()) {
        for (const ch of CHANNEL_ORDER) {
          if (this.channelHasCustomImage.has(ch)) continue;
          const cap = this.channelCaptionEls.get(ch);
          setTextIfChanged(cap, formatFallbackChannelCaption(ch, snap));
        }
      }
    }

    // Fallback synthetic canvases throttled to 8 Hz and only when Feeds tab is visible & offline
    if (
      this.drawerOpen &&
      this.drawerTab === "feeds" &&
      !this.bridge.isOnline() &&
      (force || nowMs - this.lastFallbackCanvasMs >= FALLBACK_CANVAS_INTERVAL_MS)
    ) {
      this.lastFallbackCanvasMs = nowMs;
      for (const ch of CHANNEL_ORDER) {
        if (this.channelHasCustomImage.has(ch)) continue;
        const canvas = this.channelCanvases.get(ch);
        const img = this.channelImgs.get(ch);
        if (canvas) {
          if (canvas.style.display !== "block") canvas.style.display = "block";
          renderFallbackSensorCanvas(canvas, ch, snap);
        }
        if (img && !img.classList.contains("hidden")) {
          img.classList.add("hidden");
        }
      }
      this.syncBridgeBadge();
    }
  }

  /** Mounts topbar DOM ONCE so clicking buttons NEVER destroys keyboard focus (`document.activeElement`). */
  private mountTopbarOnce(): void {
    this.topbarEl.innerHTML =
      `<div class="studio-brand-wrap">` +
      `<span class="studio-dot" aria-hidden="true"></span>` +
      `<span id="studio-brand-title" class="studio-brand"></span>` +
      `<span id="studio-brand-ver" class="studio-ver"></span>` +
      `</div>` +
      `<div id="studio-scenario-banner" class="studio-scenario-banner"></div>` +
      `<div class="studio-top-actions">` +
      `<span id="studio-brand-badge" class="studio-pill studio-pill-emerald studio-brand-badge"></span>` +
      `<button type="button" id="studio-mode-btn" class="studio-btn"></button>` +
      `<button type="button" id="studio-ctrl-btn" class="studio-btn"></button>` +
      `<button type="button" id="studio-shield-btn" class="studio-btn"></button>` +
      `<button type="button" id="studio-weather-btn" class="studio-btn" aria-controls="studio-weather-popover" aria-expanded="false"></button>` +
      `<button type="button" id="studio-bench-btn" class="studio-btn" aria-haspopup="dialog" aria-expanded="false" data-active="false"></button>` +
      `<button type="button" id="studio-sensors-btn" class="studio-btn" aria-controls="studio-sensor-drawer"></button>` +
      `<button type="button" id="studio-locale-btn" class="studio-btn"></button>` +
      `<span id="studio-fps-pill" class="studio-pill studio-pill-cyan">-- FPS</span>` +
      `</div>`;

    this.brandTitleEl = this.topbarEl.querySelector("#studio-brand-title");
    this.brandVerEl = this.topbarEl.querySelector("#studio-brand-ver");
    this.bannerEl = this.topbarEl.querySelector("#studio-scenario-banner");
    this.brandBadgeEl = this.topbarEl.querySelector("#studio-brand-badge");
    this.modeBtnEl = this.topbarEl.querySelector("#studio-mode-btn");
    this.ctrlBtnEl = this.topbarEl.querySelector("#studio-ctrl-btn");
    this.shieldBtnEl = this.topbarEl.querySelector("#studio-shield-btn");
    this.weatherBtnEl = this.topbarEl.querySelector("#studio-weather-btn");
    this.benchBtnEl = this.topbarEl.querySelector("#studio-bench-btn");
    this.sensorsBtnEl = this.topbarEl.querySelector("#studio-sensors-btn");
    this.localeBtnEl = this.topbarEl.querySelector("#studio-locale-btn");
    this.fpsPillEl = this.topbarEl.querySelector("#studio-fps-pill");

    const sig = this.abortCtrl.signal;
    this.localeBtnEl?.addEventListener(
      "click",
      () => this.setLocale(this.locale === "tr" ? "en" : "tr"),
      { signal: sig },
    );
    this.modeBtnEl?.addEventListener(
      "click",
      () => {
        const next: VisualWorldMode = this.visualMode === "photoreal" ? "obsidian" : "photoreal";
        this.setVisualMode(next);
        this.callbacks.onVisualModeChange(next);
      },
      { signal: sig },
    );
    this.ctrlBtnEl?.addEventListener(
      "click",
      () => {
        const next: FlightControlMode = this.controlMode === "autopilot" ? "manual" : "autopilot";
        this.setControlMode(next);
        this.callbacks.onControlModeChange(next);
      },
      { signal: sig },
    );
    this.shieldBtnEl?.addEventListener(
      "click",
      () => {
        const next = !this.shieldEnabled;
        this.setShieldEnabled(next);
        this.callbacks.onShieldToggle(next);
      },
      { signal: sig },
    );
    this.weatherBtnEl?.addEventListener(
      "click",
      () => this.setWeatherOpen(!this.weatherOpen, false),
      { signal: sig },
    );
    this.benchBtnEl?.addEventListener(
      "click",
      () => this.callbacks.onOpenVideoBench(),
      { signal: sig },
    );
    this.sensorsBtnEl?.addEventListener(
      "click",
      () => this.setDrawerOpen(!this.drawerOpen),
      { signal: sig },
    );
  }

  /** Mounts bottom telemetry/camera/scenario dock DOM ONCE (preserves focus across clicks). */
  private mountBottomDockOnce(): void {
    const camButtons = CAMERA_ORDER.map(
      (id) => `<button type="button" class="studio-btn" data-studio-cam="${id}"></button>`,
    ).join("");
    const scenButtons = SCENARIO_ORDER.map(
      (id) => `<button type="button" class="studio-btn" data-studio-scen="${id}"></button>`,
    ).join("");

    this.bottomDockEl.innerHTML =
      `<div id="studio-manual-hint" class="studio-manual-hint hidden"></div>` +
      `<div class="studio-dock-strip">` +
      `<div class="studio-telem-chip"><span id="telem-k-range" class="studio-telem-k"></span><span id="telem-range" class="studio-telem-v" data-state="caution">—</span></div>` +
      `<div class="studio-telem-chip"><span id="telem-k-sc3bf" class="studio-telem-k"></span><span id="telem-sc3bf" class="studio-telem-v" data-state="nominal">6/6</span></div>` +
      `<div class="studio-telem-chip"><span id="telem-k-bank" class="studio-telem-k"></span><span id="telem-bank" class="studio-telem-v">0.0°</span></div>` +
      `<div class="studio-telem-chip"><span id="telem-k-speed" class="studio-telem-k"></span><span id="telem-speed" class="studio-telem-v">101 kt</span></div>` +
      `<span class="studio-sep" aria-hidden="true"></span>` +
      camButtons +
      `<span class="studio-sep" aria-hidden="true"></span>` +
      `<button type="button" id="studio-dome-btn" class="studio-btn"></button>` +
      `</div>` +
      `<div class="studio-dock-strip">${scenButtons}</div>`;

    this.manualHintEl = this.bottomDockEl.querySelector("#studio-manual-hint");
    this.telemKeyRangeEl = this.bottomDockEl.querySelector("#telem-k-range");
    this.telemKeySc3bfEl = this.bottomDockEl.querySelector("#telem-k-sc3bf");
    this.telemKeyBankEl = this.bottomDockEl.querySelector("#telem-k-bank");
    this.telemKeySpeedEl = this.bottomDockEl.querySelector("#telem-k-speed");
    this.telemRangeEl = this.bottomDockEl.querySelector("#telem-range");
    this.telemSc3bfEl = this.bottomDockEl.querySelector("#telem-sc3bf");
    this.telemBankEl = this.bottomDockEl.querySelector("#telem-bank");
    this.telemSpeedEl = this.bottomDockEl.querySelector("#telem-speed");
    this.domeBtnEl = this.bottomDockEl.querySelector("#studio-dome-btn");

    const sig = this.abortCtrl.signal;
    this.domeBtnEl?.addEventListener(
      "click",
      () => {
        this.setDomeVisible(!this.domeVisible);
        this.callbacks.onDomeToggle(this.domeVisible);
      },
      { signal: sig },
    );

    for (const id of CAMERA_ORDER) {
      const btn = this.bottomDockEl.querySelector<HTMLButtonElement>(`[data-studio-cam="${id}"]`);
      if (btn) {
        this.camBtnEls.set(id, btn);
        btn.addEventListener(
          "click",
          () => {
            this.setCameraPreset(id);
            this.callbacks.onCameraChange(id);
          },
          { signal: sig },
        );
      }
    }

    for (const id of SCENARIO_ORDER) {
      const btn = this.bottomDockEl.querySelector<HTMLButtonElement>(`[data-studio-scen="${id}"]`);
      if (btn) {
        this.scenBtnEls.set(id, btn);
        btn.addEventListener(
          "click",
          () => {
            this.setScenario(id);
            this.callbacks.onScenarioChange(id);
          },
          { signal: sig },
        );
      }
    }
  }

  /** Patches topbar & bottom-dock labels and ARIA states in place without wiping innerHTML. */
  private syncBars(): void {
    const s = getStudioStrings(this.locale);
    const modeLabel = this.visualMode === "photoreal" ? s.modePhotoreal : s.modeObsidian;
    const ctrlLabel = this.controlMode === "manual" ? s.controlManual : s.controlAutopilot;
    const shieldLabel = this.shieldEnabled ? s.shieldOn : s.shieldOff;
    const bannerText = s.scenarios[this.scenario].banner;

    setTextIfChanged(this.brandTitleEl, s.brandTitle);
    setTextIfChanged(this.brandVerEl, s.brandVersion);
    setTextIfChanged(this.bannerEl, bannerText);
    setAttrIfChanged(this.bannerEl, "title", bannerText);

    setTextIfChanged(this.brandBadgeEl, s.brandBadge);
    setAttrIfChanged(this.brandBadgeEl, "title", s.brandBadgeTitle);

    setTextIfChanged(this.modeBtnEl, modeLabel);
    setAttrIfChanged(this.modeBtnEl, "data-active", String(this.visualMode === "photoreal"));
    setAttrIfChanged(this.modeBtnEl, "aria-pressed", String(this.visualMode === "photoreal"));

    setTextIfChanged(this.ctrlBtnEl, ctrlLabel);
    setAttrIfChanged(this.ctrlBtnEl, "data-active", String(this.controlMode === "manual"));
    setAttrIfChanged(this.ctrlBtnEl, "aria-pressed", String(this.controlMode === "manual"));

    setTextIfChanged(this.shieldBtnEl, shieldLabel);
    setAttrIfChanged(this.shieldBtnEl, "data-tone", this.shieldEnabled ? "emerald" : "rose");
    setAttrIfChanged(this.shieldBtnEl, "aria-pressed", String(this.shieldEnabled));

    setTextIfChanged(this.weatherBtnEl, s.weatherBtn);
    setAttrIfChanged(this.weatherBtnEl, "data-active", String(this.weatherOpen));
    setAttrIfChanged(this.weatherBtnEl, "aria-expanded", String(this.weatherOpen));

    setTextIfChanged(this.benchBtnEl, s.videoLabBtn);
    setAttrIfChanged(this.benchBtnEl, "data-active", String(this.benchOpen));
    setAttrIfChanged(this.benchBtnEl, "aria-expanded", String(this.benchOpen));

    setTextIfChanged(this.sensorsBtnEl, s.sensorsToggle);
    setAttrIfChanged(this.sensorsBtnEl, "data-active", String(this.drawerOpen));
    setAttrIfChanged(this.sensorsBtnEl, "aria-expanded", String(this.drawerOpen));

    setTextIfChanged(this.localeBtnEl, s.localeToggle);

    setTextIfChanged(this.manualHintEl, s.manualHint);
    this.manualHintEl?.classList.toggle("hidden", this.controlMode !== "manual");

    setTextIfChanged(this.telemKeyRangeEl, s.telemetry.range);
    setTextIfChanged(this.telemKeySc3bfEl, s.telemetry.sc3bf);
    setTextIfChanged(this.telemKeyBankEl, s.telemetry.bank);
    setTextIfChanged(this.telemKeySpeedEl, s.telemetry.speed);

    for (const id of CAMERA_ORDER) {
      const btn = this.camBtnEls.get(id);
      const active = this.cameraPreset === id;
      setTextIfChanged(btn, s.cameras[id]);
      setAttrIfChanged(btn, "data-active", String(active));
      setAttrIfChanged(btn, "aria-pressed", String(active));
    }

    for (const id of SCENARIO_ORDER) {
      const btn = this.scenBtnEls.get(id);
      const active = this.scenario === id;
      setTextIfChanged(btn, s.scenarios[id].label);
      setAttrIfChanged(btn, "data-active", String(active));
      setAttrIfChanged(btn, "aria-pressed", String(active));
    }

    setTextIfChanged(this.domeBtnEl, s.domeToggle);
    setAttrIfChanged(this.domeBtnEl, "data-active", String(this.domeVisible));
    setAttrIfChanged(this.domeBtnEl, "aria-pressed", String(this.domeVisible));
  }

  /** Builds the 2-tab sensor drawer DOM ONCE so `<img>` previews persist across UI clicks. */
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
        `<canvas width="288" height="76" data-ch-canvas="${id}"></canvas>` +
        `<img class="hidden" alt="${esc(id)}" data-ch-img="${id}" />` +
        `<div class="studio-channel-caption" data-ch-caption="${id}"></div>` +
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
      `<div class="studio-drawer-head-right">` +
      `<span id="studio-bridge-badge" class="studio-bridge-badge" data-online="false"></span>` +
      `<button type="button" id="studio-bridge-retry-btn" class="studio-btn studio-btn-compact hidden"></button>` +
      `</div>` +
      `</div>` +
      `<div class="studio-drawer-tabs" role="tablist" aria-label="Sensor Drawer Views">` +
      `<button type="button" id="studio-drawer-tab-feeds" class="studio-drawer-tab" role="tab" aria-selected="true" aria-controls="studio-drawer-pane-feeds" data-active="true"></button>` +
      `<button type="button" id="studio-drawer-tab-avionics" class="studio-drawer-tab" role="tab" aria-selected="false" aria-controls="studio-drawer-pane-avionics" data-active="false"></button>` +
      `</div>` +
      `<div id="studio-drawer-pane-feeds" class="studio-drawer-pane" role="tabpanel" aria-labelledby="studio-drawer-tab-feeds">` +
      `<div class="studio-channel-grid">${channelCards}</div>` +
      `<div class="studio-upload-zone" id="studio-upload-zone">` +
      `<div id="studio-upload-hint" class="studio-upload-hint"></div>` +
      `<div class="studio-upload-row">` +
      `<select id="studio-upload-ch" class="studio-select">` +
      `<option value="d2" selected>XVS-D2</option>` +
      `<option value="land">XVS-Land</option>` +
      `<option value="thermal">XVS-Thermal</option>` +
      `<option value="nav">XVS-Nav</option>` +
      `</select>` +
      `<input id="studio-upload-input" type="file" accept="image/*" style="display:none" />` +
      `<button type="button" id="studio-upload-btn" class="studio-btn" style="flex:1"></button>` +
      `</div>` +
      `<div id="studio-upload-status" class="studio-upload-status hidden" role="status" aria-live="polite"></div>` +
      `</div>` +
      `</div>` +
      `<div id="studio-drawer-pane-avionics" class="studio-drawer-pane hidden" role="tabpanel" aria-labelledby="studio-drawer-tab-avionics">` +
      `<section id="studio-v52-avionics-panel" class="studio-v52-panel" aria-labelledby="studio-v52-panel-title">` +
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
      `<details class="studio-specs-disclosure" open>` +
      `<summary id="studio-specs-summary"></summary>` +
      `<div id="studio-nav-v5-spec" class="studio-nav-v5-spec"></div>` +
      `<div id="studio-assurance-spec" class="studio-nav-v5-spec"></div>` +
      `</details>` +
      `</div>`;

    this.drawerTitleEl = this.drawerEl.querySelector("#studio-drawer-title");
    this.bridgeBadgeEl = this.drawerEl.querySelector("#studio-bridge-badge");
    this.retryBridgeBtnEl = this.drawerEl.querySelector("#studio-bridge-retry-btn");
    this.tabFeedsBtnEl = this.drawerEl.querySelector("#studio-drawer-tab-feeds");
    this.tabAvionicsBtnEl = this.drawerEl.querySelector("#studio-drawer-tab-avionics");
    this.paneFeedsEl = this.drawerEl.querySelector("#studio-drawer-pane-feeds");
    this.paneAvionicsEl = this.drawerEl.querySelector("#studio-drawer-pane-avionics");
    this.avionicsTitleEl = this.drawerEl.querySelector("#studio-v52-panel-title");
    this.assuranceTitleEl = this.drawerEl.querySelector("#studio-v52-assurance-title");
    this.specsSummaryEl = this.drawerEl.querySelector("#studio-specs-summary");
    this.navSpecEl = this.drawerEl.querySelector("#studio-nav-v5-spec");
    this.assuranceSpecEl = this.drawerEl.querySelector("#studio-assurance-spec");
    this.sectorBadgeEl = this.drawerEl.querySelector("#studio-v52-sector-badge");
    this.sectorBarEl = this.drawerEl.querySelector("#studio-v52-sector-bar");
    this.lwirBadgeEl = this.drawerEl.querySelector("#studio-v52-lwir-badge");
    this.fsyncBadgeEl = this.drawerEl.querySelector("#studio-v52-fsync-badge");
    this.tbcBadgeEl = this.drawerEl.querySelector("#studio-v52-tbc-badge");
    this.uploadHintEl = this.drawerEl.querySelector("#studio-upload-hint");
    this.uploadChSelectEl = this.drawerEl.querySelector("#studio-upload-ch");
    this.uploadBtnEl = this.drawerEl.querySelector("#studio-upload-btn");
    this.uploadStatusEl = this.drawerEl.querySelector("#studio-upload-status");

    for (const id of CHANNEL_ORDER) {
      const t = this.drawerEl.querySelector<HTMLElement>(`[data-ch-title="${id}"]`);
      const cap = this.drawerEl.querySelector<HTMLElement>(`[data-ch-caption="${id}"]`);
      const c = this.drawerEl.querySelector<HTMLCanvasElement>(`[data-ch-canvas="${id}"]`);
      const img = this.drawerEl.querySelector<HTMLImageElement>(`[data-ch-img="${id}"]`);
      const lat = this.drawerEl.querySelector<HTMLElement>(`[data-ch-lat="${id}"]`);
      const sha = this.drawerEl.querySelector<HTMLElement>(`[data-ch-sha="${id}"]`);
      if (t) this.channelTitleEls.set(id, t);
      if (cap) this.channelCaptionEls.set(id, cap);
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

    const sig = this.abortCtrl.signal;

    this.tabFeedsBtnEl?.addEventListener("click", () => this.setDrawerTab("feeds"), { signal: sig });
    this.tabAvionicsBtnEl?.addEventListener("click", () => this.setDrawerTab("avionics"), {
      signal: sig,
    });

    this.retryBridgeBtnEl?.addEventListener(
      "click",
      () => {
        if (this.retryBridgeBusy) return;
        this.retryBridgeBusy = true;
        this.syncBridgeBadge(true);
        void Promise.resolve(this.callbacks.onRetryBridge?.()).finally(() => {
          this.retryBridgeBusy = false;
          this.syncBridgeBadge(true);
        });
      },
      { signal: sig },
    );

    this.uploadChSelectEl?.addEventListener(
      "change",
      () => {
        if (this.uploadChSelectEl) {
          this.uploadChannel = this.uploadChSelectEl.value as SensorChannelId;
        }
      },
      { signal: sig },
    );

    const fileInput = this.drawerEl.querySelector<HTMLInputElement>("#studio-upload-input");
    this.uploadBtnEl?.addEventListener(
      "click",
      () => {
        if (this.uploadBusy) return;
        if (!this.bridge.isOnline() && !this.bridge.isNativeTauriIpc()) {
          // Probe once or allow user to pick file? Show explicit offline state immediately if offline,
          // while still letting user select file if they want — wait: showing immediate offline feedback
          // makes clicking Upload while offline clearly actionable instead of opening a silent file picker!
          this.setUploadFeedback("offline", "");
          return;
        }
        fileInput?.click();
      },
      { signal: sig },
    );

    const handleFile = async (file: File) => {
      if (this.uploadBusy) return;
      this.uploadBusy = true;
      this.setUploadFeedback("none", "");
      this.syncUploadButtonState();
      try {
        if (!this.bridge.isOnline()) {
          await this.bridge.probeHealth();
          this.syncBridgeBadge(true);
        }
        if (!this.bridge.isOnline()) {
          this.setUploadFeedback("offline", "");
          return;
        }
        const buf = await file.arrayBuffer();
        const res = await this.bridge.analyzeImage(
          this.uploadChannel,
          buf,
          file.type || "image/jpeg",
        );
        if (res) {
          this.applyBridgeChannels({ [this.uploadChannel]: res }, true);
          const latTxt = res.latencyMs !== null ? `${res.latencyMs.toFixed(1)} ms` : "OK";
          const shaTxt = res.checkpointShaShort ? ` · SHA ${res.checkpointShaShort}` : "";
          this.setUploadFeedback("custom", `${this.uploadChannel.toUpperCase()} · ${latTxt}${shaTxt}`);
        } else {
          this.setUploadFeedback("failed", "");
        }
      } catch {
        this.setUploadFeedback("failed", "");
      } finally {
        this.uploadBusy = false;
        if (fileInput) fileInput.value = "";
        this.syncUploadButtonState();
      }
    };

    fileInput?.addEventListener(
      "change",
      () => {
        const f = fileInput.files?.[0];
        if (f) void handleFile(f);
      },
      { signal: sig },
    );

    const dropZone = this.drawerEl.querySelector<HTMLElement>("#studio-upload-zone");
    dropZone?.addEventListener("dragover", (e) => e.preventDefault(), { signal: sig });
    dropZone?.addEventListener(
      "drop",
      (e) => {
        e.preventDefault();
        const f = e.dataTransfer?.files?.[0];
        if (f) void handleFile(f);
      },
      { signal: sig },
    );
  }

  private setUploadFeedback(
    kind: "none" | "offline" | "failed" | "custom",
    customMsg: string,
  ): void {
    this.uploadFeedbackKey = kind;
    this.uploadCustomMsg = customMsg;
    this.syncUploadStatusText();
  }

  private syncUploadButtonState(): void {
    const d = getStudioStrings(this.locale).drawer;
    if (this.uploadBtnEl) {
      this.uploadBtnEl.disabled = this.uploadBusy;
      setTextIfChanged(this.uploadBtnEl, this.uploadBusy ? d.uploadAnalyzing : d.uploadBtn);
    }
  }

  private syncUploadStatusText(): void {
    if (!this.uploadStatusEl) return;
    const d = getStudioStrings(this.locale).drawer;
    if (this.uploadFeedbackKey === "none") {
      this.uploadStatusEl.classList.add("hidden");
      setTextIfChanged(this.uploadStatusEl, "");
      return;
    }
    this.uploadStatusEl.classList.remove("hidden");
    if (this.uploadFeedbackKey === "offline") {
      setAttrIfChanged(this.uploadStatusEl, "data-tone", "error");
      setTextIfChanged(this.uploadStatusEl, d.uploadOffline);
    } else if (this.uploadFeedbackKey === "failed") {
      setAttrIfChanged(this.uploadStatusEl, "data-tone", "error");
      setTextIfChanged(this.uploadStatusEl, d.uploadFailed);
    } else {
      setAttrIfChanged(this.uploadStatusEl, "data-tone", "ok");
      setTextIfChanged(this.uploadStatusEl, this.uploadCustomMsg);
    }
  }

  private syncDrawerText(): void {
    const full = getStudioStrings(this.locale);
    const s = full.drawer;
    setTextIfChanged(this.drawerTitleEl, s.title);
    setTextIfChanged(this.tabFeedsBtnEl, s.tabFeeds);
    setTextIfChanged(this.tabAvionicsBtnEl, s.tabAvionics);
    setTextIfChanged(this.avionicsTitleEl, s.avionicsTitle);
    setTextIfChanged(this.assuranceTitleEl, s.assuranceTitle);
    setTextIfChanged(this.specsSummaryEl, s.specsSummary);
    for (const badge of getFormalAssuranceBadges(this.locale)) {
      const chip = this.assuranceChipEls.get(badge.id);
      setTextIfChanged(chip, badge.text);
      setAttrIfChanged(chip, "title", badge.title);
    }
    setTextIfChanged(this.navSpecEl, full.navV5Spec);
    setTextIfChanged(this.assuranceSpecEl, STUDIO_ASSURANCE_V52_SPEC);
    setTextIfChanged(this.uploadHintEl, s.uploadHint);
    setAttrIfChanged(this.uploadChSelectEl, "aria-label", s.uploadSelectAria);
    this.syncUploadButtonState();
    for (const id of CHANNEL_ORDER) {
      setTextIfChanged(this.channelTitleEls.get(id), s.channels[id]);
    }
    this.syncAvionicsPanel();
    this.syncBridgeBadge(true);
  }

  private mountWeatherOnce(): void {
    this.weatherEl.innerHTML =
      `<div id="w-title" class="studio-weather-title"></div>` +
      `<label class="studio-slider-row"><div class="studio-slider-head"><span id="w-lbl-tod"></span><b id="w-val-tod">16:30</b></div><input id="w-tod" type="range" min="5" max="22" step="0.25" value="16.5" /></label>` +
      `<label class="studio-slider-row"><div class="studio-slider-head"><span id="w-lbl-vis"></span><b id="w-val-vis">18.0 km</b></div><input id="w-vis" type="range" min="0.5" max="25" step="0.5" value="18" /></label>` +
      `<label class="studio-slider-row"><div class="studio-slider-head"><span id="w-lbl-wind"></span><b id="w-val-wind">12 kt</b></div><input id="w-wind" type="range" min="0" max="45" step="1" value="12" /></label>` +
      `<label class="studio-slider-row"><div class="studio-slider-head"><span id="w-lbl-cloud"></span><b id="w-val-cloud">42%</b></div><input id="w-cloud" type="range" min="0" max="1" step="0.05" value="0.42" /></label>` +
      `<label class="studio-slider-row"><div class="studio-slider-head"><span id="w-lbl-rain"></span><b id="w-val-rain">0%</b></div><input id="w-rain" type="range" min="0" max="1" step="0.05" value="0" /></label>` +
      `<div class="studio-weather-btns">` +
      `<button type="button" id="w-eng-btn" class="studio-btn" style="flex:1"></button>` +
      `<button type="button" id="w-rst-btn" class="studio-btn"></button>` +
      `</div>`;

    for (const id of [
      "w-title",
      "w-lbl-tod",
      "w-lbl-vis",
      "w-lbl-wind",
      "w-lbl-cloud",
      "w-lbl-rain",
      "w-val-tod",
      "w-val-vis",
      "w-val-wind",
      "w-val-cloud",
      "w-val-rain",
      "w-eng-btn",
      "w-rst-btn",
    ]) {
      const el = this.weatherEl.querySelector<HTMLElement>(`#${id}`);
      if (el) this.weatherTextEls.set(id, el);
    }

    const sig = this.abortCtrl.signal;
    const bindSlider = (id: string, handler: (v: number) => void) => {
      const el = this.weatherEl.querySelector<HTMLInputElement>(`#${id}`);
      if (el) {
        this.weatherInputEls.set(id, el);
        el.addEventListener("input", () => handler(Number(el.value)), { signal: sig });
      }
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

    this.weatherEl
      .querySelector<HTMLButtonElement>("#w-eng-btn")
      ?.addEventListener("click", () => this.callbacks.onToggleEngineOut(), { signal: sig });
    this.weatherEl
      .querySelector<HTMLButtonElement>("#w-rst-btn")
      ?.addEventListener("click", () => this.callbacks.onResetApproach(), { signal: sig });
  }

  private syncWeatherUi(): void {
    const w = getStudioStrings(this.locale).weather;
    setAttrIfChanged(this.weatherEl, "aria-label", w.title);
    const setTxt = (id: string, txt: string) => setTextIfChanged(this.weatherTextEls.get(id), txt);
    const setVal = (id: string, val: number) => {
      const el = this.weatherInputEls.get(id);
      const str = String(val);
      if (el && el.value !== str) el.value = str;
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
}
