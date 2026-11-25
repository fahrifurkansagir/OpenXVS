/**
 * OpenXVS v3.0 Observatory 3-Tab Bilingual (`TR / EN`) Shell.
 *
 * Keeps Tab 1's 3D WebGPU viewport completely unobstructed while presenting:
 *   - Top Bar: 3 Tabs (`3D Scene` | `Live Architecture` | `Real-Sky AI Lab`), instant `TR/EN` toggle,
 *     FPS / WebGPU / `18/18 Kani` proof pill, and collapsible `Technical Telemetry` drawer toggle.
 *   - Tab 1 Bottom Compact Dock: Plain-language status cards, 4 Adversarial Red-Team Stress buttons
 *     (defaulting to `cbdr` ActiveRange), and 4 3D Manifold toggles (`spzStress` OFF by default).
 *   - Tab 2 Overlay: Live-pulsing 8-node `ArchMapView`.
 *   - Tab 3 Overlay: Stage-1 lazy-loaded `RealSkyLabView` (zero Stage-0 bytes!).
 */

import type { ManifoldToggleState } from "../passes/pass3_airframe_traffic.ts";
import { ArchMapView, type ArchLiveMetrics, type TrainedSuiteProvenance } from "./arch_map_view.ts";
import {
  DEFAULT_LOCALE,
  getObservatoryStrings,
  type ManifoldId,
  type ObservatoryLocale,
  type ObservatoryTabId,
  type StressPresetId,
} from "./i18n.ts";

export const DEFAULT_STRESS_PRESET: StressPresetId = "cbdr";

export const DEFAULT_MANIFOLD_TOGGLES: Required<ManifoldToggleState> = {
  activeRangeCone: true,
  sc3bfTube: true,
  forcedLandFunnel: false,
  spzStress: false,
};

export interface ObservatoryLiveStatus {
  readonly fps: number;
  readonly backendLabel: string;
  readonly rangeSigmaM: number | null;
  readonly rangeHatM: number | null;
  readonly sc3bfSafe: boolean;
  readonly kaniPassed: number;
  readonly kaniTotal: number;
  readonly activeRangeModeLabel: string;
  readonly detectorHz: number;
  readonly detectorMs: number;
  readonly halHealthy: boolean;
  readonly forcedLandActive: boolean;
}

export interface ObservatoryShellCallbacks {
  readonly onTabChange?: (tab: ObservatoryTabId) => void;
  readonly onLocaleChange?: (locale: ObservatoryLocale) => void;
  readonly onStressPresetChange?: (preset: StressPresetId) => void;
  readonly onManifoldToggleChange?: (toggles: Required<ManifoldToggleState>) => void;
  readonly onTechDrawerToggle?: (expanded: boolean) => void;
  readonly onCameraToggle?: (nextMode: "cockpit" | "orbit") => void;
}

/** Pure state store (unit-testable in Node without DOM). */
export class ObservatoryStateStore {
  private locale: ObservatoryLocale = DEFAULT_LOCALE;
  private activeTab: ObservatoryTabId = "scene";
  private stressPreset: StressPresetId = DEFAULT_STRESS_PRESET;
  private techDrawerOpen = false;
  private manifolds: Required<ManifoldToggleState> = { ...DEFAULT_MANIFOLD_TOGGLES };

  getLocale(): ObservatoryLocale {
    return this.locale;
  }

  toggleLocale(): ObservatoryLocale {
    this.locale = this.locale === "tr" ? "en" : "tr";
    return this.locale;
  }

  getActiveTab(): ObservatoryTabId {
    return this.activeTab;
  }

  setActiveTab(tab: ObservatoryTabId): void {
    this.activeTab = tab;
  }

  getStressPreset(): StressPresetId {
    return this.stressPreset;
  }

  setStressPreset(preset: StressPresetId): void {
    this.stressPreset = preset;
    if (preset === "cbdr") {
      this.manifolds = { ...this.manifolds, activeRangeCone: true, sc3bfTube: true, forcedLandFunnel: false };
    } else if (preset === "forcedland") {
      this.manifolds = { ...this.manifolds, forcedLandFunnel: true };
    } else if (preset === "normal" || preset === "spoof") {
      this.manifolds = { ...this.manifolds, forcedLandFunnel: false };
    }
  }

  getManifoldToggles(): Required<ManifoldToggleState> {
    return { ...this.manifolds };
  }

  setManifoldToggle(id: ManifoldId, value: boolean): Required<ManifoldToggleState> {
    this.manifolds = { ...this.manifolds, [id]: value };
    return this.getManifoldToggles();
  }

  isTechDrawerOpen(): boolean {
    return this.techDrawerOpen;
  }

  setTechDrawerOpen(open: boolean): void {
    this.techDrawerOpen = open;
  }
}

function esc(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

/** Browser DOM Controller for the OpenXVS v3.0 Observatory 3-Tab Shell. */
export class ObservatoryShell {
  readonly store = new ObservatoryStateStore();
  private readonly host: HTMLElement;
  private readonly callbacks: ObservatoryShellCallbacks;
  private readonly topbarEl: HTMLDivElement;
  private readonly dockEl: HTMLDivElement;
  private readonly archOverlayEl: HTMLDivElement;
  private readonly realskyOverlayEl: HTMLDivElement;
  private readonly archMapView: ArchMapView;
  private lastArchUpdateMs = 0;
  private lastArchMetrics: ArchLiveMetrics | null = null;
  private isOrbitCam = false;

  constructor(host: HTMLElement, callbacks: ObservatoryShellCallbacks = {}) {
    this.host = host;
    this.callbacks = callbacks;

    this.topbarEl = document.createElement("div");
    this.topbarEl.id = "xvs-observatory-topbar";
    this.topbarEl.className = "obs-topbar";

    this.dockEl = document.createElement("div");
    this.dockEl.id = "xvs-observatory-dock";
    this.dockEl.className = "obs-dock";

    this.archOverlayEl = document.createElement("div");
    this.archOverlayEl.id = "xvs-tab-arch";
    this.archOverlayEl.className = "obs-tab-panel obs-hidden";

    this.realskyOverlayEl = document.createElement("div");
    this.realskyOverlayEl.id = "xvs-tab-realsky";
    this.realskyOverlayEl.className = "obs-tab-panel obs-hidden";

    this.host.appendChild(this.topbarEl);
    this.host.appendChild(this.dockEl);
    this.host.appendChild(this.archOverlayEl);
    this.host.appendChild(this.realskyOverlayEl);

    this.archMapView = new ArchMapView(this.archOverlayEl, this.store.getLocale());
    this.renderChrome();
    this.applyTechDrawerVisibility();
  }

  getRealSkyContainer(): HTMLElement {
    return this.realskyOverlayEl;
  }

  setTrainedProvenance(partial: Partial<TrainedSuiteProvenance>): void {
    this.archMapView.setTrainedProvenance(partial);
  }

  applyTechDrawerVisibility(): void {
    const open = this.store.isTechDrawerOpen();
    this.host.classList.toggle("obs-tech-collapsed", !open);
  }

  setStressPreset(preset: StressPresetId): void {
    this.store.setStressPreset(preset);
    this.renderChrome();
    this.callbacks.onStressPresetChange?.(preset);
    this.callbacks.onManifoldToggleChange?.(this.store.getManifoldToggles());
  }

  setCameraMode(mode: "cockpit" | "chase" | "orbit"): void {
    const nextOrbit = mode !== "cockpit";
    if (this.isOrbitCam !== nextOrbit) {
      this.isOrbitCam = nextOrbit;
      this.renderChrome();
    }
  }

  setActiveTab(tab: ObservatoryTabId): void {
    this.store.setActiveTab(tab);
    this.archOverlayEl.classList.toggle("obs-hidden", tab !== "arch");
    this.realskyOverlayEl.classList.toggle("obs-hidden", tab !== "realsky");
    this.dockEl.classList.toggle("obs-hidden", tab !== "scene");
    if (tab === "arch" && this.lastArchMetrics) {
      this.archMapView.updateMetrics(this.lastArchMetrics);
    }
    this.renderChrome();
    this.callbacks.onTabChange?.(tab);
  }

  updateLiveStatus(st: ObservatoryLiveStatus): void {
    const fpsEl = this.topbarEl.querySelector<HTMLElement>("#obs-fps-pill");
    if (fpsEl) {
      fpsEl.textContent = `${Math.round(st.fps)} FPS · ${st.backendLabel}`;
    }
    const kaniEl = this.topbarEl.querySelector<HTMLElement>("#obs-kani-pill");
    if (kaniEl) {
      kaniEl.textContent = `${st.kaniPassed}/${st.kaniTotal} KANI`;
    }
    const rngVal = this.dockEl.querySelector<HTMLElement>("#obs-card-range-val");
    if (rngVal) {
      if (st.rangeHatM !== null && st.rangeSigmaM !== null) {
        const locked = st.activeRangeModeLabel === "LOCKED" || st.rangeSigmaM <= 95;
        const lockTag = locked
          ? this.store.getLocale() === "tr"
            ? " · KİLİTLENDİ"
            : " · LOCKED"
          : "";
        rngVal.textContent = `${Math.round(st.rangeHatM)} m (±${Math.round(st.rangeSigmaM)} m${lockTag})`;
        rngVal.dataset.severity = locked ? "nominal" : st.rangeSigmaM > 250 ? "caution" : "info";
      } else {
        rngVal.textContent = "—";
        rngVal.dataset.severity = "info";
      }
    }
    const scVal = this.dockEl.querySelector<HTMLElement>("#obs-card-sc3bf-val");
    if (scVal) {
      scVal.textContent = st.sc3bfSafe ? "6/6 SAFE" : "EVADE";
      scVal.dataset.severity = st.sc3bfSafe ? "nominal" : "alert";
    }

    const metrics: ArchLiveMetrics = {
      fps: st.fps,
      detectorHz: st.detectorHz,
      detectorMs: st.detectorMs,
      kaniPassed: st.kaniPassed,
      kaniTotal: st.kaniTotal,
      activeRangeMode: st.activeRangeModeLabel,
      forcedLandActive: st.forcedLandActive,
      halHealthy: st.halHealthy,
    };
    this.lastArchMetrics = metrics;

    const now = performance.now();
    if (this.store.getActiveTab() === "arch" && now - this.lastArchUpdateMs >= 250) {
      this.lastArchUpdateMs = now;
      this.archMapView.updateMetrics(metrics);
    }
  }

  private renderChrome(): void {
    const locale = this.store.getLocale();
    const s = getObservatoryStrings(locale);
    const activeTab = this.store.getActiveTab();
    const activePreset = this.store.getStressPreset();
    const toggles = this.store.getManifoldToggles();

    const tabsHtml = (["scene", "arch", "realsky"] as const)
      .map(
        (id, idx) =>
          `<button class="obs-tab-btn" data-obs-tab="${id}" data-active="${activeTab === id}">[${idx + 1}] ${esc(s.tabs[id])}</button>`,
      )
      .join("");

    this.topbarEl.innerHTML =
      `<div class="obs-brand">${esc(s.appTitle)}</div>` +
      `<div class="obs-tabs" role="tablist">${tabsHtml}</div>` +
      `<div class="obs-top-right">` +
      `<span id="obs-fps-pill" class="obs-pill obs-pill-info">60 FPS · WebGPU</span>` +
      `<span id="obs-kani-pill" class="obs-pill obs-pill-nominal">18/18 KANI</span>` +
      `<button id="obs-tech-btn" class="obs-btn" data-active="${this.store.isTechDrawerOpen()}">${esc(s.statusCards.techDrawerToggle)}</button>` +
      `<button id="obs-locale-btn" class="obs-btn obs-locale-btn">${esc(s.localeToggleLabel)}</button>` +
      `</div>`;

    const presetHtml = (["cbdr", "forcedland", "spoof", "normal"] as const)
      .map((id) => {
        const p = s.stressPresets[id];
        return `<button class="obs-preset-btn" data-obs-preset="${id}" data-active="${activePreset === id}" title="${esc(p.hint)}">${esc(p.label)}</button>`;
      })
      .join("");

    const manifoldHtml = (["activeRangeCone", "sc3bfTube", "forcedLandFunnel", "spzStress"] as const)
      .map(
        (id) =>
          `<label class="obs-toggle-chip" data-active="${toggles[id]}">` +
          `<input type="checkbox" data-obs-manifold="${id}"${toggles[id] ? " checked" : ""} />` +
          `<span>${esc(s.manifolds[id])}</span>` +
          `</label>`,
      )
      .join("");

    const camToggleLabel =
      locale === "tr"
        ? this.isOrbitCam
          ? "Kamera: 3B Orbit → Kokpit"
          : "Kamera: Kokpit → 3B Orbit"
        : this.isOrbitCam
          ? "View: 3D Orbit → Cockpit"
          : "View: Cockpit → 3D Orbit";

    this.dockEl.innerHTML =
      `<div class="obs-dock-row obs-status-strip">` +
      `<div class="obs-card"><span class="obs-card-k">${esc(s.statusCards.rangeUncertainty)}</span><strong id="obs-card-range-val" class="obs-card-v">—</strong></div>` +
      `<div class="obs-card"><span class="obs-card-k">${esc(s.statusCards.sc3bfSafety)}</span><strong id="obs-card-sc3bf-val" class="obs-card-v" data-severity="nominal">6/6 SAFE</strong></div>` +
      `<div class="obs-card"><span class="obs-card-k">${esc(s.statusCards.kaniProof)}</span><strong class="obs-card-v" data-severity="nominal">VERIFIED</strong></div>` +
      `<button id="obs-cam-toggle-btn" class="obs-btn" data-active="${this.isOrbitCam}">${esc(camToggleLabel)}</button>` +
      `</div>` +
      `<div class="obs-dock-row obs-presets-row">${presetHtml}</div>` +
      `<div class="obs-dock-row obs-manifolds-row">${manifoldHtml}</div>`;

    this.dockEl.querySelector<HTMLButtonElement>("#obs-cam-toggle-btn")?.addEventListener("click", () => {
      const next: "cockpit" | "orbit" = this.isOrbitCam ? "cockpit" : "orbit";
      this.isOrbitCam = next === "orbit";
      this.renderChrome();
      this.callbacks.onCameraToggle?.(next);
    });

    this.topbarEl.querySelectorAll<HTMLButtonElement>("[data-obs-tab]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const tab = btn.dataset.obsTab as ObservatoryTabId;
        this.setActiveTab(tab);
      });
    });

    this.topbarEl.querySelector<HTMLButtonElement>("#obs-locale-btn")?.addEventListener("click", () => {
      const next = this.store.toggleLocale();
      this.archMapView.setLocale(next);
      this.renderChrome();
      this.callbacks.onLocaleChange?.(next);
    });

    this.topbarEl.querySelector<HTMLButtonElement>("#obs-tech-btn")?.addEventListener("click", () => {
      const next = !this.store.isTechDrawerOpen();
      this.store.setTechDrawerOpen(next);
      this.applyTechDrawerVisibility();
      this.renderChrome();
      this.callbacks.onTechDrawerToggle?.(next);
    });

    this.dockEl.querySelectorAll<HTMLButtonElement>("[data-obs-preset]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const preset = btn.dataset.obsPreset as StressPresetId;
        this.setStressPreset(preset);
      });
    });

    this.dockEl.querySelectorAll<HTMLInputElement>("[data-obs-manifold]").forEach((inp) => {
      inp.addEventListener("change", () => {
        const id = inp.dataset.obsManifold as ManifoldId;
        const next = this.store.setManifoldToggle(id, inp.checked);
        this.renderChrome();
        this.callbacks.onManifoldToggleChange?.(next);
      });
    });
  }
}
