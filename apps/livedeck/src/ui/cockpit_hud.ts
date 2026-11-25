/**
 * OpenXVS Live Deck Cockpit HUD Overlay & Guided Mission Control Deck.
 *
 * Enforces Charter §7.5 (ZERO silent fallbacks):
 *   - Persistent non-dismissable Status Badge Bar for RENDER, AI, TERRAIN, SIMPLEX, and BRIDGE
 *   - Unmistakable amber degraded banner whenever WebGL2, NO F16, ORT-WASM-SIMD, or PROCEDURAL-CI is active
 *   - Dynamic top-center avionics annunciator across M1..M4 + mandatory go-around / spoof / Cam-R / traffic alerts
 *   - Conformal full-viewport SVG overlay (TD Box, DO-365B diamond, AI detection boxes, KSFO 28L bracket)
 *     strictly blanked when `simplexMode === 0` (`INVALID_INHIBIT`)
 *   - Compact bottom-center Horizontal Situation / Traffic Radar Inset (heading-up compass, runway, traffic)
 *   - One-click Red-Team Guided Mission selector (M1..M4) + manual physical fault-injection controls
 *   - 3-way Detector selector: D0 (Classical Baseline) | D1-KF (Gate-G1 ONNX) | XVS-D2 (v2.0 6-ch ConvGRU)
 *   - Embedded collapsible `TelemetryDrawer` + one-click Mission Proof JSON exporter
 */

import {
  ALL_MISSIONS,
  createNominalPerturbations,
  type DetectorMode,
  type MissionId,
  type PhysicalPerturbationState,
} from "../missions/index.ts";
import type { BridgeTelemetry } from "../net/bridge_client.ts";
import type { SpzDiagnostics } from "../passes/pass2b_spz_world.ts";
import type { CameraViewMode, OrbitTargetMode } from "../passes/types.ts";
import { TelemetryDrawer, type TelemetryDrawerSnapshot } from "./telemetry_drawer.ts";

export type SimplexModeCode = 0 | 1 | 2 | 3 | 4;
export type UiDetectorMode = "d0" | "d1_kf" | "xvs_d2";

export const SIMPLEX_LABELS: Readonly<Record<SimplexModeCode, string>> = {
  0: "INVALID_INHIBIT",
  1: "FALLBACK_D0_SAFE",
  2: "DEGRADED_COAST",
  3: "ISOLATED_5CAM",
  4: "NOMINAL_ML",
};

export interface CockpitBadgeState {
  readonly renderBadge: "RENDER: WebGPU" | "RENDER: WebGL2 (DEGRADED)";
  readonly aiBadge:
    | "AI: ORT-WebGPU-FP16"
    | "AI: ORT-WebGPU-FP32 (NO F16)"
    | "AI: ORT-WASM-SIMD"
    | "AI: LOADING STAGE-1";
  readonly terrainBadge: "TERRAIN: USGS-3DEP-GEOID18" | "TERRAIN: PROCEDURAL-CI";
  readonly orthoLoaded?: boolean;
  readonly simplexMode: SimplexModeCode;
  readonly degradedReasons: readonly string[];
  readonly bridge: BridgeTelemetry;
}

export interface ConformalTargetOverlay {
  readonly id: string;
  readonly screenX: number;
  readonly screenY: number;
  readonly visible: boolean;
  readonly boxHalfPx: number;
  readonly rangeM: number;
  readonly ttcS: number;
  readonly alertTier: 0 | 1 | 2 | 3;
  readonly isGhost: boolean;
  readonly ghostRejected: boolean;
  readonly meshVerified: boolean;
  /** Bearing relative to ownship nose in radians (+right, -left) for off-screen chevron cues. */
  readonly relBearingRad: number;
}

export interface ConformalRunwayOverlay {
  readonly visible: boolean;
  readonly screenX: number;
  readonly screenY: number;
  readonly halfWidthPx: number;
  readonly halfHeightPx: number;
  readonly vVplM: number;
  readonly glideslopeDeg: number;
  readonly papiWhiteCount: number;
  readonly distM: number;
}

export interface ConformalAiBoxOverlay {
  readonly u: number;
  readonly v: number;
  readonly widthPx: number;
  readonly heightPx: number;
  readonly score: number;
  readonly sigmaPx: number;
  readonly modelId: string;
}

export interface HsiRadarTrack {
  readonly id: string;
  readonly relNorthM: number;
  readonly relEastM: number;
  readonly velNorthMps: number;
  readonly velEastMps: number;
  readonly rangeM: number;
  readonly alertTier: 0 | 1 | 2 | 3;
  readonly isGhost: boolean;
  readonly ghostRejected: boolean;
  readonly meshVerified: boolean;
}

export interface ConformalAndHsiFrameData {
  readonly viewportWidth: number;
  readonly viewportHeight: number;
  readonly cameraMode?: CameraViewMode;
  readonly conformalBlanked: boolean;
  readonly efvsEnabled: boolean;
  readonly detectorMode: UiDetectorMode;
  readonly yawRad: number;
  readonly airspeedMps: number;
  readonly altWgs84M: number;
  readonly targets: readonly ConformalTargetOverlay[];
  readonly runway28L: ConformalRunwayOverlay | null;
  readonly aiDetections: readonly ConformalAiBoxOverlay[];
  readonly hsiTracks: readonly HsiRadarTrack[];
  /** KSFO 28L threshold relative to ownship in NED (North, East metres). */
  readonly hsiRunway28LRelNe: readonly [number, number] | null;
  /** KSFO 28L true heading in radians. */
  readonly hsiRunway28LHdgRad: number;
  /** Optional 3-ship MeshDAA peer relative positions in NED (North, East metres). */
  readonly hsiPeersRelNe?: ReadonlyArray<readonly [number, number]>;
}

export interface CockpitHudCallbacks {
  readonly onSelectMission: (missionId: MissionId) => void;
  readonly onTogglePlayPause: () => void;
  readonly onResetMission: () => void;
  readonly onManualPerturbationsChange: (next: PhysicalPerturbationState) => void;
  readonly onDetectorModeChange?: (mode: UiDetectorMode) => void;
  readonly onCameraModeChange?: (mode: CameraViewMode) => void;
  readonly onOrbitTargetChange?: (target: OrbitTargetMode) => void;
  readonly onSpzModeChange?: (mode: "off" | "preview") => void;
  readonly onSpzFileSelected?: (file: File) => void;
  readonly onToggleBridgeConnect: () => void;
  readonly onDownloadProof: () => void;
}

function escapeXml(raw: string): string {
  return raw
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function toLegacyDetectorMode(uiMode: UiDetectorMode): DetectorMode {
  return uiMode === "d0" ? "D0" : "D1_KF";
}

function allFinite(...vals: number[]): boolean {
  for (let i = 0; i < vals.length; i++) {
    if (!Number.isFinite(vals[i])) return false;
  }
  return true;
}

function formatPapiPattern(whiteCount: number): string {
  const w = Math.max(0, Math.min(4, Math.round(whiteCount)));
  return `${w}W ${4 - w}R`;
}

export class CockpitHudOverlay {
  private readonly root: HTMLElement;
  private readonly conformalSvg: SVGSVGElement;
  private readonly hsiWrapEl: HTMLElement;
  private readonly hsiSvg: SVGSVGElement;
  private readonly renderBadgeEl: HTMLElement;
  private readonly aiBadgeEl: HTMLElement;
  private readonly terrainBadgeEl: HTMLElement;
  private readonly orthoBadgeEl: HTMLElement;
  private readonly simplexBadgeEl: HTMLElement;
  private readonly bridgeBadgeEl: HTMLElement;
  private readonly cameraBtns = new Map<CameraViewMode, HTMLButtonElement>();
  private readonly orbitTargetBtns = new Map<OrbitTargetMode, HTMLButtonElement>();
  private readonly oddPillRain: HTMLElement;
  private readonly oddPillIce: HTMLElement;
  private readonly oddPillGlare: HTMLElement;
  private readonly oddPillFog: HTMLElement;
  private readonly oddPillLwir: HTMLElement;
  private lastGlareStrength = 0.0;
  private readonly vlmGateStatusEl: HTMLElement;
  private readonly degradedBannerEl: HTMLElement;
  private readonly alertCalloutEl: HTMLElement;
  private readonly missionBtns = new Map<MissionId, HTMLButtonElement>();
  private readonly playPauseBtn: HTMLButtonElement;
  private readonly missionStatusEl: HTMLElement;

  // Manual fault-injection controls
  private readonly camRSlider: HTMLInputElement;
  private readonly camRValEl: HTMLElement;
  private readonly spoofSlider: HTMLInputElement;
  private readonly spoofValEl: HTMLElement;
  private readonly fogSlider: HTMLInputElement;
  private readonly fogValEl: HTMLElement;
  private readonly cornerSlider: HTMLInputElement;
  private readonly cornerValEl: HTMLElement;
  private readonly efvsCheck: HTMLInputElement;
  private readonly peer3Check: HTMLInputElement;
  private readonly detectorSelect: HTMLSelectElement;
  private readonly spzOffBtn: HTMLButtonElement;
  private readonly spzPreviewBtn: HTMLButtonElement;
  private readonly spzLoadBtn: HTMLButtonElement;
  private readonly spzStatusEl: HTMLElement;

  readonly drawer: TelemetryDrawer;
  private currentPerturbations: PhysicalPerturbationState = createNominalPerturbations();
  private currentUiDetectorMode: UiDetectorMode = "xvs_d2";
  private readonly callbacks: CockpitHudCallbacks;

  constructor(mountParent: HTMLElement, callbacks: CockpitHudCallbacks) {
    this.callbacks = callbacks;
    this.root = document.createElement("div");
    this.root.className = "xvs-hud-root";

    // 0. Full-viewport Conformal SVG Layer (TD Box, AI Detection Overlay, KSFO 28L Bracket)
    this.conformalSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    this.conformalSvg.setAttribute("class", "xvs-conformal-svg");
    this.conformalSvg.setAttribute("data-testid", "conformal-svg-overlay");
    this.conformalSvg.setAttribute("preserveAspectRatio", "none");
    this.root.appendChild(this.conformalSvg);

    // 1. Persistent Top Status Badge Bar (Charter §7.5 — always visible)
    const topBar = document.createElement("header");
    topBar.className = "xvs-status-bar";
    topBar.setAttribute("data-testid", "status-badge-bar");

    const brand = document.createElement("span");
    brand.className = "xvs-brand";
    brand.textContent = "OPENXVS LIVE DECK v2.5";

    this.renderBadgeEl = this._makeBadge("render-badge", "RENDER: WebGPU", "nominal");
    this.aiBadgeEl = this._makeBadge("ai-badge", "AI: LOADING STAGE-1", "info");
    this.terrainBadgeEl = this._makeBadge(
      "terrain-badge",
      "TERRAIN: USGS-3DEP-GEOID18",
      "nominal",
    );
    this.orthoBadgeEl = this._makeBadge("ortho-badge", "USDA/USGS Ortho", "info");
    this.orthoBadgeEl.style.display = "none";
    this.orthoBadgeEl.title = "Real U.S. Public Domain USDA/USGS National Map Orthoimagery draped over 3DEP elevation";
    this.simplexBadgeEl = this._makeBadge("simplex-badge", "SIMPLEX: NOMINAL_ML", "nominal");
    this.bridgeBadgeEl = this._makeBadge("bridge-badge", "BRIDGE: OFFLINE", "muted");
    this.bridgeBadgeEl.style.cursor = "pointer";
    this.bridgeBadgeEl.title = "Click to connect/disconnect local ws://127.0.0.1:8765 bridge";
    this.bridgeBadgeEl.addEventListener("click", () => callbacks.onToggleBridgeConnect());

    // Compact 3D Camera Suite Switcher ([1] Cockpit HUD | [2] 3D Chase | [3] 3D Orbit)
    const camGroup = document.createElement("div");
    camGroup.className = "xvs-cam-group";
    camGroup.setAttribute("data-testid", "camera-mode-bar");
    const camDefs: ReadonlyArray<{ mode: CameraViewMode; label: string; title: string }> = [
      { mode: "cockpit", label: "1: COCKPIT", title: "Pilot-eye conformal HUD camera (Key 1)" },
      { mode: "chase", label: "2: 3D CHASE", title: "External 3D chase camera behind ownship (Key 2)" },
      { mode: "orbit", label: "3: 3D ORBIT", title: "Free 360-deg 3D orbit camera — drag to orbit, scroll to zoom (Key 3)" },
    ];
    for (const def of camDefs) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "xvs-btn xvs-cam-btn";
      btn.setAttribute("data-testid", `cam-btn-${def.mode}`);
      btn.dataset.active = def.mode === "cockpit" ? "true" : "false";
      btn.textContent = def.label;
      btn.title = def.title;
      btn.addEventListener("click", () => {
        this.setCameraMode(def.mode);
        callbacks.onCameraModeChange?.(def.mode);
      });
      this.cameraBtns.set(def.mode, btn);
      camGroup.appendChild(btn);
    }

    // 3D Orbit / Spotter Target Switcher (OWN | INTRUDER | KSFO TOWER)
    const targetSep = document.createElement("span");
    targetSep.className = "xvs-cam-sep";
    targetSep.textContent = "|";
    camGroup.appendChild(targetSep);

    const targetGroup = document.createElement("div");
    targetGroup.className = "xvs-target-group";
    targetGroup.setAttribute("data-testid", "orbit-target-bar");
    const targetDefs: ReadonlyArray<{ target: OrbitTargetMode; label: string; title: string }> = [
      { target: "ownship", label: "OWN", title: "3D Orbit around Ownship twin-jet" },
      { target: "intruder", label: "INTRUDER", title: "3D Orbit around approaching Intruder + WCV & MeshDAA rays (Key 5)" },
      { target: "tower", label: "KSFO TOWER", title: "KSFO 28L Control Tower spotter camera tracking ownship (Key 4)" },
    ];
    for (const tdef of targetDefs) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "xvs-btn xvs-cam-btn xvs-target-btn";
      btn.setAttribute("data-testid", `orbit-target-${tdef.target}`);
      btn.dataset.active = tdef.target === "ownship" ? "true" : "false";
      btn.textContent = tdef.label;
      btn.title = tdef.title;
      btn.addEventListener("click", () => {
        this.setCameraMode("orbit");
        this.setOrbitTarget(tdef.target);
        callbacks.onOrbitTargetChange?.(tdef.target);
      });
      this.orbitTargetBtns.set(tdef.target, btn);
      targetGroup.appendChild(btn);
    }
    camGroup.appendChild(targetGroup);

    topBar.append(
      brand,
      this.renderBadgeEl,
      this.aiBadgeEl,
      this.terrainBadgeEl,
      this.orthoBadgeEl,
      this.simplexBadgeEl,
      this.bridgeBadgeEl,
      camGroup,
    );
    this.root.appendChild(topBar);

    // 2. Unmistakable Degraded / Fallback Banner (hidden ONLY when zero degraded reasons exist)
    this.degradedBannerEl = document.createElement("div");
    this.degradedBannerEl.className = "xvs-degraded-banner";
    this.degradedBannerEl.setAttribute("data-testid", "degraded-banner");
    this.degradedBannerEl.style.display = "none";
    this.root.appendChild(this.degradedBannerEl);

    // 3. Dynamic Top-Center Avionics Annunciator Banner (M1..M4 + Critical Safety Alerts)
    this.alertCalloutEl = document.createElement("div");
    this.alertCalloutEl.className = "xvs-alert-callout";
    this.alertCalloutEl.setAttribute("data-testid", "alert-callout");
    this.alertCalloutEl.style.display = "none";
    this.root.appendChild(this.alertCalloutEl);

    // 4. Left Control Panel: Red-Team Guided Missions + Manual Physical Fault Injection
    const leftPanel = document.createElement("aside");
    leftPanel.className = "xvs-mission-panel";

    const mHeader = document.createElement("div");
    mHeader.className = "xvs-section-label";
    mHeader.textContent = "GUIDED RED-TEAM MISSIONS (45s)";
    leftPanel.appendChild(mHeader);

    const mList = document.createElement("div");
    mList.className = "xvs-mission-list";
    for (const mission of ALL_MISSIONS) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "xvs-btn xvs-mission-btn";
      btn.setAttribute("data-testid", `mission-btn-${mission.id}`);
      btn.textContent = mission.name;
      btn.title = mission.summary;
      btn.addEventListener("click", () => callbacks.onSelectMission(mission.id));
      this.missionBtns.set(mission.id, btn);
      mList.appendChild(btn);
    }
    leftPanel.appendChild(mList);

    const playbackRow = document.createElement("div");
    playbackRow.className = "xvs-btn-row";
    this.playPauseBtn = document.createElement("button");
    this.playPauseBtn.type = "button";
    this.playPauseBtn.className = "xvs-btn xvs-btn-sm";
    this.playPauseBtn.textContent = "PAUSE";
    this.playPauseBtn.addEventListener("click", () => callbacks.onTogglePlayPause());

    const resetBtn = document.createElement("button");
    resetBtn.type = "button";
    resetBtn.className = "xvs-btn xvs-btn-sm";
    resetBtn.textContent = "RESET";
    resetBtn.addEventListener("click", () => callbacks.onResetMission());

    playbackRow.append(this.playPauseBtn, resetBtn);
    leftPanel.appendChild(playbackRow);

    this.missionStatusEl = document.createElement("div");
    this.missionStatusEl.className = "xvs-mission-status";
    this.missionStatusEl.textContent = "Active: M1 (0.0 / 45.0 s)";
    leftPanel.appendChild(this.missionStatusEl);

    // Manual physical fault injection sliders
    const fHeader = document.createElement("div");
    fHeader.className = "xvs-section-label";
    fHeader.style.marginTop = "10px";
    fHeader.textContent = "PHYSICAL FAULT INJECTION";
    leftPanel.appendChild(fHeader);

    const sCam = this._makeSlider(leftPanel, "Cam-R Drift (deg)", -2.0, 2.0, 0.05, 0);
    this.camRSlider = sCam.input;
    this.camRValEl = sCam.valEl;

    const sSpoof = this._makeSlider(leftPanel, "GPS Spoof East (m/s)", 0.0, 5.0, 0.1, 0);
    this.spoofSlider = sSpoof.input;
    this.spoofValEl = sSpoof.valEl;

    const sFog = this._makeSlider(leftPanel, "Fog Contrast", 0.1, 1.0, 0.02, 1.0);
    this.fogSlider = sFog.input;
    this.fogValEl = sFog.valEl;

    const sCorner = this._makeSlider(leftPanel, "Runway Corner Bias (px)", 0, 30, 1, 0);
    this.cornerSlider = sCorner.input;
    this.cornerValEl = sCorner.valEl;

    const toggleRow = document.createElement("div");
    toggleRow.className = "xvs-toggle-col";
    this.efvsCheck = this._makeCheckbox(toggleRow, "8–14 μm LWIR EFVS");
    this.peer3Check = this._makeCheckbox(toggleRow, "3rd MeshDAA Peer");

    const detRow = document.createElement("label");
    detRow.className = "xvs-select-row";
    const detLbl = document.createElement("span");
    detLbl.textContent = "Detector";
    this.detectorSelect = document.createElement("select");
    this.detectorSelect.className = "xvs-select";
    this.detectorSelect.setAttribute("data-testid", "detector-select");
    const detectorOptions: ReadonlyArray<{ value: UiDetectorMode; label: string }> = [
      { value: "d0", label: "D0 (Baseline)" },
      { value: "d1_kf", label: "D1-KF (Gate-G1 ONNX)" },
      { value: "xvs_d2", label: "XVS-D2 (Colab G4 Trained)" },
    ];
    for (const item of detectorOptions) {
      const opt = document.createElement("option");
      opt.value = item.value;
      opt.textContent = item.label;
      this.detectorSelect.appendChild(opt);
    }
    this.detectorSelect.value = "xvs_d2";
    this.detectorSelect.addEventListener("change", () => {
      const nextUiMode = this.detectorSelect.value as UiDetectorMode;
      this.currentUiDetectorMode = nextUiMode;
      this.currentPerturbations = {
        ...this.currentPerturbations,
        detectorMode: toLegacyDetectorMode(nextUiMode),
      };
      this.callbacks.onDetectorModeChange?.(nextUiMode);
    });
    detRow.append(detLbl, this.detectorSelect);
    toggleRow.appendChild(detRow);
    leftPanel.appendChild(toggleRow);

    // 4b. 3DGS SPZ World Overlay Controls (OFF | KSFO PREVIEW | LOAD .SPZ — VISUAL_STRESS_ONLY)
    const spzHeader = document.createElement("div");
    spzHeader.className = "xvs-section-label";
    spzHeader.style.marginTop = "8px";
    spzHeader.textContent = "3DGS WORLD (VISUAL_STRESS_ONLY)";
    leftPanel.appendChild(spzHeader);

    const spzBar = document.createElement("div");
    spzBar.className = "xvs-spz-row";
    spzBar.setAttribute("data-testid", "spz-control-bar");

    this.spzOffBtn = document.createElement("button");
    this.spzOffBtn.type = "button";
    this.spzOffBtn.className = "xvs-btn xvs-btn-sm xvs-spz-btn";
    this.spzOffBtn.setAttribute("data-testid", "spz-btn-off");
    this.spzOffBtn.dataset.active = "true";
    this.spzOffBtn.textContent = "OFF";
    this.spzOffBtn.addEventListener("click", () => callbacks.onSpzModeChange?.("off"));

    this.spzPreviewBtn = document.createElement("button");
    this.spzPreviewBtn.type = "button";
    this.spzPreviewBtn.className = "xvs-btn xvs-btn-sm xvs-spz-btn";
    this.spzPreviewBtn.setAttribute("data-testid", "spz-btn-preview");
    this.spzPreviewBtn.dataset.active = "false";
    this.spzPreviewBtn.textContent = "KSFO PREVIEW";
    this.spzPreviewBtn.addEventListener("click", () => callbacks.onSpzModeChange?.("preview"));

    const spzFileInput = document.createElement("input");
    spzFileInput.type = "file";
    spzFileInput.accept = ".spz";
    spzFileInput.style.display = "none";
    spzFileInput.addEventListener("change", () => {
      const f = spzFileInput.files?.[0];
      if (f) {
        callbacks.onSpzFileSelected?.(f);
        spzFileInput.value = "";
      }
    });

    this.spzLoadBtn = document.createElement("button");
    this.spzLoadBtn.type = "button";
    this.spzLoadBtn.className = "xvs-btn xvs-btn-sm xvs-spz-btn";
    this.spzLoadBtn.setAttribute("data-testid", "spz-btn-load");
    this.spzLoadBtn.dataset.active = "false";
    this.spzLoadBtn.textContent = "LOAD .SPZ";
    this.spzLoadBtn.title = "Load Niantic SPZ v2 (.spz) 3DGS world from Notebook 3 (or drag-and-drop onto canvas)";
    this.spzLoadBtn.addEventListener("click", () => spzFileInput.click());

    spzBar.append(this.spzOffBtn, this.spzPreviewBtn, this.spzLoadBtn, spzFileInput);
    leftPanel.appendChild(spzBar);

    this.spzStatusEl = document.createElement("div");
    this.spzStatusEl.className = "xvs-spz-status";
    this.spzStatusEl.setAttribute("data-testid", "spz-status-badge");
    this.spzStatusEl.dataset.severity = "muted";
    this.spzStatusEl.textContent = "OFF · Drop .spz or click KSFO PREVIEW";
    leftPanel.appendChild(this.spzStatusEl);

    this.root.appendChild(leftPanel);

    // 5. Compact Bottom-Center HSI / Traffic Radar Inset
    this.hsiWrapEl = document.createElement("div");
    this.hsiWrapEl.className = "xvs-hsi-inset";
    this.hsiWrapEl.setAttribute("data-testid", "hsi-radar-inset");
    this.hsiSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    this.hsiSvg.setAttribute("viewBox", "0 0 186 186");
    this.hsiSvg.setAttribute("width", "186");
    this.hsiSvg.setAttribute("height", "186");
    this.hsiWrapEl.appendChild(this.hsiSvg);
    this.root.appendChild(this.hsiWrapEl);

    // 6. Right Telemetry Drawer + DR-13 5-Mode Optical-ODD Strip + DR-09 IDAL-D VLM Semantic Gate
    this.drawer = new TelemetryDrawer(this.root, callbacks.onDownloadProof);

    const oddSec = document.createElement("div");
    oddSec.className = "xvs-drawer-section xvs-odd-section";
    oddSec.setAttribute("data-testid", "optical-odd-strip");
    const oddHdr = document.createElement("div");
    oddHdr.className = "xvs-section-label";
    oddHdr.textContent = "OPTICAL-ODD (DR-13 <1ms NO-ML)";
    const oddGrid = document.createElement("div");
    oddGrid.className = "xvs-odd-grid";
    const makeOddPill = (id: string, titleText: string, tooltip: string): HTMLElement => {
      const pill = document.createElement("div");
      pill.className = "xvs-odd-pill";
      pill.setAttribute("data-testid", `odd-pill-${id}`);
      pill.dataset.severity = "nominal";
      pill.title = tooltip;
      const k = document.createElement("span");
      k.className = "xvs-odd-k";
      k.textContent = titleText;
      const v = document.createElement("span");
      v.className = "xvs-odd-v";
      v.textContent = "OK";
      pill.append(k, v);
      oddGrid.appendChild(pill);
      return pill;
    };
    this.oddPillRain = makeOddPill("rain", "RAIN", "DR-13 Mode 1 Raindrop / Blur ODD Monitor");
    this.oddPillIce = makeOddPill("ice", "ICE", "DR-13 Mode 2 Window Icing — AF447 ZERO-AIRDATA");
    this.oddPillGlare = makeOddPill("glare", "GLARE", "DR-13 Mode 3 Solar Glare SPA-Gated Monitor");
    this.oddPillFog = makeOddPill("fog", "FOG", "DR-13 Mode 4 Fog MOR < 1609 m -> THERMAL_PRIMARY");
    this.oddPillLwir = makeOddPill("lwir", "LWIR", "DR-13 Mode 5 8-14 um LWIR FFC / Bad-Pixel Health");
    const oddMeta = document.createElement("div");
    oddMeta.className = "xvs-vlm-meta";
    oddMeta.textContent = "AF447 ZERO-AIRDATA · MOR<1609m→THERMAL";
    oddSec.append(oddHdr, oddGrid, oddMeta);

    const vlmSec = document.createElement("div");
    vlmSec.className = "xvs-drawer-section";
    vlmSec.setAttribute("data-testid", "vlm-gate-section");
    const vlmHdr = document.createElement("div");
    vlmHdr.className = "xvs-section-label";
    vlmHdr.textContent = "VLM SEMANTIC GATE (DR-09 IDAL-D)";
    const vlmSub = document.createElement("div");
    vlmSub.className = "xvs-vlm-meta";
    vlmSub.textContent = "ZERO ACTUATOR AUTH · PESSIMISTIC UNION";
    this.vlmGateStatusEl = document.createElement("div");
    this.vlmGateStatusEl.className = "xvs-vlm-status";
    this.vlmGateStatusEl.setAttribute("data-testid", "vlm-gate-status");
    this.vlmGateStatusEl.dataset.severity = "info";
    this.vlmGateStatusEl.textContent = "NOMINAL STANDBY (<=200ms)";
    vlmSec.append(vlmHdr, this.vlmGateStatusEl, vlmSub);

    const drawerBody = this.root.querySelector('[data-testid="telemetry-drawer"] .xvs-drawer-body');
    const proofBtn = this.root.querySelector('[data-testid="download-proof-btn"]');
    // DR-12 480 Hz deterministic lockstep clock sync sub-line inside PROFILER section
    const profilerSec = drawerBody?.querySelector(".xvs-drawer-section");
    if (profilerSec) {
      const lockstepLine = document.createElement("div");
      lockstepLine.className = "xvs-lockstep-line";
      lockstepLine.setAttribute("data-testid", "lockstep-sync-badge");
      lockstepLine.textContent = "480Hz LOCKSTEP | 60Hz LWIR | 30Hz EO";
      profilerSec.appendChild(lockstepLine);
    }
    if (drawerBody && proofBtn) {
      drawerBody.insertBefore(oddSec, proofBtn);
      drawerBody.insertBefore(vlmSec, proofBtn);
    } else if (drawerBody) {
      drawerBody.append(oddSec, vlmSec);
    }

    mountParent.appendChild(this.root);
  }

  private _makeBadge(testId: string, text: string, severity: string): HTMLElement {
    const el = document.createElement("span");
    el.className = "xvs-badge";
    el.setAttribute("data-testid", testId);
    el.dataset.severity = severity;
    el.textContent = text;
    return el;
  }

  private _makeSlider(
    parent: HTMLElement,
    label: string,
    min: number,
    max: number,
    step: number,
    init: number,
  ): { input: HTMLInputElement; valEl: HTMLElement } {
    const wrap = document.createElement("div");
    wrap.className = "xvs-slider-row";
    const top = document.createElement("div");
    top.className = "xvs-bar-top";
    const l = document.createElement("span");
    l.textContent = label;
    const valEl = document.createElement("span");
    valEl.textContent = String(init);
    top.append(l, valEl);

    const input = document.createElement("input");
    input.type = "range";
    input.className = "xvs-slider";
    input.min = String(min);
    input.max = String(max);
    input.step = String(step);
    input.value = String(init);
    input.addEventListener("input", () => this._emitManualControls());

    wrap.append(top, input);
    parent.appendChild(wrap);
    return { input, valEl };
  }

  private _makeCheckbox(parent: HTMLElement, label: string): HTMLInputElement {
    const lbl = document.createElement("label");
    lbl.className = "xvs-check-row";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.addEventListener("change", () => this._emitManualControls());
    const txt = document.createElement("span");
    txt.textContent = label;
    lbl.append(cb, txt);
    parent.appendChild(lbl);
    return cb;
  }

  private _emitManualControls(): void {
    const deg = Number(this.camRSlider.value);
    const spoofEast = Number(this.spoofSlider.value);
    const fog = Number(this.fogSlider.value);
    const cornerPx = Number(this.cornerSlider.value);
    this.camRValEl.textContent = `${deg.toFixed(2)}°`;
    this.spoofValEl.textContent = `${spoofEast.toFixed(1)}`;
    this.fogValEl.textContent = `${fog.toFixed(2)}×`;
    this.cornerValEl.textContent = `${cornerPx.toFixed(0)} px`;

    const uiMode = this.detectorSelect.value as UiDetectorMode;
    this.currentUiDetectorMode = uiMode;

    const next: PhysicalPerturbationState = {
      camRBiasRad: (deg * Math.PI) / 180,
      gnssSpoofVelNedMps: [0, spoofEast, 0],
      thirdPeerActive: this.peer3Check.checked,
      detectorMode: toLegacyDetectorMode(uiMode),
      fogContrast: fog,
      lwirEfvsEnabled: this.efvsCheck.checked,
      runwayCornerBiasPx: cornerPx,
    };
    this.currentPerturbations = next;
    this.callbacks.onManualPerturbationsChange(next);
  }

  setDetectorUiMode(mode: UiDetectorMode): void {
    this.currentUiDetectorMode = mode;
    this.detectorSelect.value = mode;
  }

  getDetectorUiMode(): UiDetectorMode {
    return this.currentUiDetectorMode;
  }

  syncPerturbationControls(p: PhysicalPerturbationState, explicitUiMode?: UiDetectorMode): void {
    this.currentPerturbations = p;
    const deg = (p.camRBiasRad * 180) / Math.PI;
    this.camRSlider.value = deg.toFixed(2);
    this.camRValEl.textContent = `${deg.toFixed(2)}°`;
    this.spoofSlider.value = p.gnssSpoofVelNedMps[1].toFixed(1);
    this.spoofValEl.textContent = `${p.gnssSpoofVelNedMps[1].toFixed(1)}`;
    this.fogSlider.value = p.fogContrast.toFixed(2);
    this.fogValEl.textContent = `${p.fogContrast.toFixed(2)}×`;
    this.cornerSlider.value = p.runwayCornerBiasPx.toFixed(0);
    this.cornerValEl.textContent = `${p.runwayCornerBiasPx.toFixed(0)} px`;
    this.efvsCheck.checked = p.lwirEfvsEnabled;
    this.peer3Check.checked = p.thirdPeerActive;
    const resolvedUiMode: UiDetectorMode =
      explicitUiMode ?? (p.detectorMode === "D0" ? "d0" : "xvs_d2");
    this.currentUiDetectorMode = resolvedUiMode;
    this.detectorSelect.value = resolvedUiMode;
  }

  setCameraMode(mode: CameraViewMode): void {
    for (const [m, btn] of this.cameraBtns.entries()) {
      btn.dataset.active = m === mode ? "true" : "false";
    }
  }

  setOrbitTarget(target: OrbitTargetMode): void {
    for (const [t, btn] of this.orbitTargetBtns.entries()) {
      btn.dataset.active = t === target ? "true" : "false";
    }
  }

  updateSpzStatus(diag: SpzDiagnostics): void {
    this.spzOffBtn.dataset.active = diag.mode === "off" ? "true" : "false";
    this.spzPreviewBtn.dataset.active = diag.mode === "preview" ? "true" : "false";
    this.spzLoadBtn.dataset.active = diag.mode === "custom" ? "true" : "false";
    if (diag.lastError) {
      this.spzStatusEl.dataset.severity = "caution";
      this.spzStatusEl.textContent = `SPZ ERR: ${diag.lastError}`;
    } else if (diag.mode === "off") {
      this.spzStatusEl.dataset.severity = "muted";
      this.spzStatusEl.textContent = "OFF · Drop .spz or click KSFO PREVIEW";
    } else if (diag.mode === "preview") {
      this.spzStatusEl.dataset.severity = "info";
      this.spzStatusEl.textContent = `PREVIEW · ${diag.renderSplats.toLocaleString()} splats · ${diag.gtTier}`;
    } else {
      this.spzStatusEl.dataset.severity = "nominal";
      this.spzStatusEl.textContent = `${diag.sourceLabel} · ${diag.renderSplats.toLocaleString()}/${diag.totalFileSplats.toLocaleString()} splats · ${diag.gtTier}`;
    }
  }

  setMissionStatus(activeId: MissionId, simTimeS: number, durationS: number, paused: boolean): void {
    for (const [id, btn] of this.missionBtns.entries()) {
      btn.dataset.active = id === activeId ? "true" : "false";
    }
    this.playPauseBtn.textContent = paused ? "PLAY" : "PAUSE";
    this.missionStatusEl.textContent = `Active: ${activeId} (${simTimeS.toFixed(1)} / ${durationS.toFixed(1)} s)${paused ? " [PAUSED]" : ""}`;
  }

  updateBadges(state: CockpitBadgeState): void {
    this.renderBadgeEl.textContent = state.renderBadge;
    this.renderBadgeEl.dataset.severity =
      state.renderBadge === "RENDER: WebGPU" ? "nominal" : "caution";

    this.aiBadgeEl.textContent = state.aiBadge;
    this.aiBadgeEl.dataset.severity =
      state.aiBadge === "AI: ORT-WebGPU-FP16"
        ? "nominal"
        : state.aiBadge === "AI: LOADING STAGE-1"
          ? "info"
          : "caution";

    this.terrainBadgeEl.textContent = state.terrainBadge;
    this.terrainBadgeEl.dataset.severity =
      state.terrainBadge === "TERRAIN: USGS-3DEP-GEOID18" ? "nominal" : "caution";

    const showOrtho =
      Boolean(state.orthoLoaded) && state.terrainBadge === "TERRAIN: USGS-3DEP-GEOID18";
    this.orthoBadgeEl.style.display = showOrtho ? "inline-flex" : "none";

    const sLabel = SIMPLEX_LABELS[state.simplexMode];
    this.simplexBadgeEl.textContent = `SIMPLEX: ${sLabel}`;
    this.simplexBadgeEl.dataset.severity =
      state.simplexMode === 4
        ? "nominal"
        : state.simplexMode === 0
          ? "alert"
          : "caution";

    const br = state.bridge;
    if (br.state === "connected") {
      this.bridgeBadgeEl.textContent = `BRIDGE: LIVE (${br.gdl90Sent + br.mavlinkSent} tx)`;
      this.bridgeBadgeEl.dataset.severity = "nominal";
    } else if (br.state === "connecting") {
      this.bridgeBadgeEl.textContent = "BRIDGE: CONNECTING";
      this.bridgeBadgeEl.dataset.severity = "info";
    } else if (br.state === "error") {
      this.bridgeBadgeEl.textContent = "BRIDGE: UNREACHABLE";
      this.bridgeBadgeEl.dataset.severity = "caution";
    } else {
      this.bridgeBadgeEl.textContent = "BRIDGE: OFFLINE";
      this.bridgeBadgeEl.dataset.severity = "muted";
    }

    if (state.degradedReasons.length > 0) {
      this.degradedBannerEl.style.display = "block";
      this.degradedBannerEl.textContent = `EXPLICIT FALLBACK ACTIVE: ${state.degradedReasons.join(" | ")}`;
    } else {
      this.degradedBannerEl.style.display = "none";
    }
  }

  setOpticalOddGlare(glareStrength: number): void {
    this.lastGlareStrength = Number.isFinite(glareStrength) ? glareStrength : 0.0;
  }

  private _setOddPill(pill: HTMLElement, text: string, severity: "nominal" | "info" | "caution" | "alert"): void {
    pill.dataset.severity = severity;
    const valEl = pill.querySelector(".xvs-odd-v");
    if (valEl) valEl.textContent = text;
  }

  updateTelemetry(snap: TelemetryDrawerSnapshot): void {
    this.drawer.update(snap);

    // DR-13 Live 5-Mode Deterministic Optical-ODD Health Strip (<1 ms no-ML)
    const efvsActive = Boolean(snap.efvsEnabled ?? this.currentPerturbations.lwirEfvsEnabled);
    const fogContrast = this.currentPerturbations.fogContrast;
    const inhibited = snap.simplexMode === 0 || snap.mandatoryGoAround;

    this._setOddPill(this.oddPillRain, "OK", "nominal");
    this._setOddPill(this.oddPillIce, "OK", "nominal");
    if (this.lastGlareStrength > 0.35) {
      this._setOddPill(this.oddPillGlare, "WARN", "caution");
    } else {
      this._setOddPill(this.oddPillGlare, "OK", "nominal");
    }
    if (fogContrast < 0.35) {
      if (efvsActive) {
        this._setOddPill(this.oddPillFog, "THERM", "info");
      } else {
        this._setOddPill(this.oddPillFog, "EXCEED", "alert");
      }
    } else {
      this._setOddPill(this.oddPillFog, "OK", "nominal");
    }
    if (inhibited) {
      this._setOddPill(this.oddPillLwir, "INHIB", "caution");
    } else if (efvsActive) {
      this._setOddPill(this.oddPillLwir, "ACTIVE", "info");
    } else {
      this._setOddPill(this.oddPillLwir, "OK", "nominal");
    }

    // DR-09 IDAL-D VLM Semantic Gate deterministic 6-stage isolation status
    if (inhibited) {
      this.vlmGateStatusEl.dataset.severity = "caution";
      this.vlmGateStatusEl.textContent = "VLM INHIBITED (P0 / GO-AROUND)";
    } else if (fogContrast < 0.35 && !efvsActive) {
      this.vlmGateStatusEl.dataset.severity = "caution";
      this.vlmGateStatusEl.textContent = "OPTICAL ODD MASKED (FOG)";
    } else if (efvsActive) {
      this.vlmGateStatusEl.dataset.severity = "nominal";
      this.vlmGateStatusEl.textContent = "LWIR ROI ADVISORY (2-OF-3 OK)";
    } else {
      this.vlmGateStatusEl.dataset.severity = "info";
      this.vlmGateStatusEl.textContent = "NOMINAL STANDBY (<=200ms)";
    }

    const camRExcluded = (snap.camHealthMask & 0x02) === 0;
    const camRDriftDeg = snap.camRBiasDeg ?? 0;

    if (snap.mandatoryGoAround || snap.simplexMode === 0) {
      this.alertCalloutEl.style.display = "block";
      this.alertCalloutEl.dataset.severity = "alert";
      this.alertCalloutEl.textContent =
        "FOG / V-VPL > 10.0 m — MANDATORY GO-AROUND CLIMB-OUT (CONFORMAL CUES INHIBITED)";
    } else if (snap.spoofAlarm) {
      this.alertCalloutEl.style.display = "block";
      this.alertCalloutEl.dataset.severity = "caution";
      this.alertCalloutEl.textContent =
        `GNSS SPOOFING DETECTED (CUSUM ${snap.cusumStat.toFixed(1)} > 10.53) — COASTING ON XVS-NAV`;
    } else if (snap.wc1Alert || snap.ecbfInfeasible) {
      this.alertCalloutEl.style.display = "block";
      this.alertCalloutEl.dataset.severity = "alert";
      this.alertCalloutEl.textContent =
        "TRAFFIC ALERT — 3-SHIP MESHDAA + ECBF-QP AVOIDANCE MANEUVER";
    } else if (snap.ghostRejected) {
      this.alertCalloutEl.style.display = "block";
      this.alertCalloutEl.dataset.severity = "nominal";
      this.alertCalloutEl.textContent =
        "3-SHIP TRIPLE-RAY CONSENSUS — GHOST REJECTED & TRAFFIC TRACKED";
    } else if (camRExcluded || Math.abs(camRDriftDeg) >= 0.35) {
      this.alertCalloutEl.style.display = "block";
      this.alertCalloutEl.dataset.severity = "caution";
      const sign = camRDriftDeg >= 0 ? "+" : "";
      this.alertCalloutEl.textContent =
        `CAM-R ${sign}${camRDriftDeg.toFixed(1)}° EXTRINSIC DRIFT DETECTED — 5-CAM DEGRADED ISOLATION`;
    } else if (snap.missionId) {
      this.alertCalloutEl.style.display = "block";
      this.alertCalloutEl.dataset.severity = "info";
      if (snap.missionId === "M1") {
        this.alertCalloutEl.textContent =
          "M1: 6-CAM EXTRINSIC MONITORING — NOMINAL 6-CAM ATTITUDE LOCK";
      } else if (snap.missionId === "M2") {
        this.alertCalloutEl.textContent =
          "M2: GNSS INTEGRITY MONITOR — 8-ARM CUSUM WATCHING BANK 0";
      } else if (snap.missionId === "M3") {
        this.alertCalloutEl.textContent =
          "M3: XVS-MESHDAA COOPERATIVE DAA — TRACKING APPROACHING TRAFFIC";
      } else {
        this.alertCalloutEl.textContent = snap.efvsEnabled
          ? "M4: KSFO 28L 3° APPROACH — 8–14 μm LWIR EFVS + VISION-RAIM ACTIVE"
          : "M4: KSFO 28L 3° APPROACH — HITS TUNNEL + VISION-RAIM MONITORING";
      }
    } else {
      this.alertCalloutEl.style.display = "none";
    }
  }

  /**
   * Updates both the conformal full-viewport SVG HUD overlay (TD box, AI detection boxes,
   * KSFO 28L threshold bracket) and the compact bottom-center heading-up HSI traffic radar.
   */
  updateConformalAndHsi(data: ConformalAndHsiFrameData): void {
    this._renderConformalSvg(data);
    this._renderHsiRadar(data);
  }

  private _renderConformalSvg(data: ConformalAndHsiFrameData): void {
    const w = Math.max(1, Math.round(data.viewportWidth));
    const h = Math.max(1, Math.round(data.viewportHeight));
    this.conformalSvg.setAttribute("viewBox", `0 0 ${w} ${h}`);

    // Hard Safety Lock: Conformal SVG cues MUST blank when simplexMode === 0 (INVALID_INHIBIT)
    if (data.conformalBlanked) {
      if (this.conformalSvg.childNodes.length > 0) {
        this.conformalSvg.innerHTML = "";
      }
      return;
    }

    const parts: string[] = [];
    const scale = Math.max(1.0, w / 1280);
    const fs = (11.5 * scale).toFixed(1);
    const fsSm = (10.0 * scale).toFixed(1);
    const camMode: CameraViewMode = data.cameraMode ?? "cockpit";
    const isCockpitCam = camMode === "cockpit";

    // 1. KSFO 28L Runway Conformal Bracket & PAPI Readout (pilot-eye cockpit HUD only)
    const rwy = data.runway28L;
    if (
      isCockpitCam &&
      rwy &&
      rwy.visible &&
      allFinite(
        rwy.screenX,
        rwy.screenY,
        rwy.halfWidthPx,
        rwy.halfHeightPx,
        rwy.vVplM,
        rwy.glideslopeDeg,
        rwy.papiWhiteCount,
        rwy.distM,
      )
    ) {
      const rx = rwy.screenX;
      const ry = rwy.screenY;
      const hw = Math.max(24 * scale, Math.min(180 * scale, rwy.halfWidthPx));
      const hh = Math.max(16 * scale, Math.min(110 * scale, rwy.halfHeightPx));
      const cLen = Math.max(10 * scale, Math.min(24 * scale, hw * 0.35));
      const col = rwy.vVplM > 10.0 ? "#ff4b55" : data.efvsEnabled ? "#d6ffea" : "#2ce68a";
      const sw = (2.0 * scale).toFixed(1);

      // 4 crisp corner brackets around KSFO 28L threshold
      const x0 = rx - hw;
      const x1 = rx + hw;
      const y0 = ry - hh;
      const y1 = ry + hh;
      parts.push(
        `<g stroke="${col}" stroke-width="${sw}" fill="none" opacity="0.95">` +
          `<path d="M${x0.toFixed(1)},${(y0 + cLen).toFixed(1)} V${y0.toFixed(1)} H${(x0 + cLen).toFixed(1)}"/>` +
          `<path d="M${(x1 - cLen).toFixed(1)},${y0.toFixed(1)} H${x1.toFixed(1)} V${(y0 + cLen).toFixed(1)}"/>` +
          `<path d="M${x0.toFixed(1)},${(y1 - cLen).toFixed(1)} V${y1.toFixed(1)} H${(x0 + cLen).toFixed(1)}"/>` +
          `<path d="M${(x1 - cLen).toFixed(1)},${y1.toFixed(1)} H${x1.toFixed(1)} V${(y1 - cLen).toFixed(1)}"/>` +
          `</g>`,
      );
      const papiTag = `KSFO 28L · ${(rwy.distM / 1000).toFixed(1)}km · GS ${rwy.glideslopeDeg.toFixed(1)}° [${formatPapiPattern(rwy.papiWhiteCount)}]`;
      const tagW = (papiTag.length * 6.8 + 14) * scale;
      const tagH = 19 * scale;
      const tx = Math.max(12 * scale, Math.min(w - tagW - 12 * scale, rx - tagW * 0.5));
      const ty = Math.max(88 * scale, y0 - tagH - 6 * scale);
      parts.push(
        `<rect x="${tx.toFixed(1)}" y="${ty.toFixed(1)}" width="${tagW.toFixed(1)}" height="${tagH.toFixed(1)}" rx="${(3 * scale).toFixed(1)}" fill="rgba(8,14,20,0.82)" stroke="${col}" stroke-width="${(1.2 * scale).toFixed(1)}"/>` +
          `<text x="${(tx + 7 * scale).toFixed(1)}" y="${(ty + 13.5 * scale).toFixed(1)}" fill="${col}" font-size="${fsSm}" font-weight="700" font-family="B612 Mono, Consolas, monospace">${escapeXml(papiTag)}</text>`,
      );
    }

    // 2. Live Stage-1 AI Detection ROI Frame & Cyan Dashed Detection Boxes (pilot-eye cockpit HUD only)
    if (isCockpitCam && data.detectorMode !== "d0") {
      // Forward camera narrow-FOV 256x256 Pass-4 AI scan window centered on viewport
      const roiSize = Math.min(w, h) * 0.44;
      const roiX = (w - roiSize) * 0.5;
      const roiY = (h - roiSize) * 0.5;
      const swDash = (1.1 * scale).toFixed(1);
      parts.push(
        `<rect x="${roiX.toFixed(1)}" y="${roiY.toFixed(1)}" width="${roiSize.toFixed(1)}" height="${roiSize.toFixed(1)}" fill="none" stroke="rgba(76,201,255,0.28)" stroke-width="${swDash}" stroke-dasharray="${(6 * scale).toFixed(1)} ${(5 * scale).toFixed(1)}"/>` +
          `<text x="${(roiX + 5 * scale).toFixed(1)}" y="${(roiY + 12 * scale).toFixed(1)}" fill="rgba(76,201,255,0.72)" font-size="${fsSm}" font-family="B612 Mono, Consolas, monospace">${data.detectorMode === "xvs_d2" ? "AI ROI · XVS-D2 6-CH CONVGRU" : "AI ROI · D1-KF ONNX"}</text>`,
      );

      for (const det of data.aiDetections.slice(0, 6)) {
        if (!allFinite(det.u, det.v, det.widthPx, det.heightPx, det.score, det.sigmaPx)) {
          continue;
        }
        const cx = roiX + (det.u / 256.0) * roiSize;
        const cy = roiY + (det.v / 256.0) * roiSize;
        const bw = Math.max(18 * scale, (det.widthPx / 256.0) * roiSize);
        const bh = Math.max(18 * scale, (det.heightPx / 256.0) * roiSize);
        const bx = cx - bw * 0.5;
        const by = cy - bh * 0.5;
        const lbl = `${det.modelId === "xvs_d2_web_fp16" ? "D2" : "D1"} ${(det.score * 100).toFixed(0)}% ±${det.sigmaPx.toFixed(1)}px`;
        parts.push(
          `<rect x="${bx.toFixed(1)}" y="${by.toFixed(1)}" width="${bw.toFixed(1)}" height="${bh.toFixed(1)}" fill="rgba(76,201,255,0.08)" stroke="#4cc9ff" stroke-width="${(1.6 * scale).toFixed(1)}" stroke-dasharray="${(5 * scale).toFixed(1)} ${(3 * scale).toFixed(1)}"/>` +
            `<text x="${bx.toFixed(1)}" y="${(by - 4 * scale).toFixed(1)}" fill="#4cc9ff" font-size="${fsSm}" font-weight="700" font-family="B612 Mono, Consolas, monospace">${escapeXml(lbl)}</text>`,
        );
      }
    }

    // 3. Conformal Traffic TD Box, DO-365B Diamond & Off-Screen Directional Cue Arrows
    for (const tr of data.targets) {
      const col = tr.ghostRejected
        ? "#ff4b55"
        : tr.alertTier >= 3
          ? "#ff4b55"
          : tr.alertTier === 2
            ? "#ffbe3b"
            : tr.meshVerified
              ? "#2ce68a"
              : "#4cc9ff";
      const sw = (2.2 * scale).toFixed(1);

      if (
        tr.visible &&
        allFinite(tr.screenX, tr.screenY, tr.boxHalfPx, tr.rangeM, tr.ttcS) &&
        tr.screenX >= 0 &&
        tr.screenX <= w &&
        tr.screenY >= 0 &&
        tr.screenY <= h
      ) {
        const sx = tr.screenX;
        const sy = tr.screenY;
        const hb = Math.max(22 * scale, Math.min(72 * scale, tr.boxHalfPx));
        const cl = Math.max(8 * scale, hb * 0.34);

        if (tr.ghostRejected) {
          // Dashed crimson box + bold X for rejected epipolar ghost
          parts.push(
            `<rect x="${(sx - hb).toFixed(1)}" y="${(sy - hb).toFixed(1)}" width="${(hb * 2).toFixed(1)}" height="${(hb * 2).toFixed(1)}" fill="rgba(255,75,85,0.10)" stroke="#ff4b55" stroke-width="${sw}" stroke-dasharray="${(6 * scale).toFixed(1)} ${(4 * scale).toFixed(1)}"/>` +
              `<line x1="${(sx - hb * 0.55).toFixed(1)}" y1="${(sy - hb * 0.55).toFixed(1)}" x2="${(sx + hb * 0.55).toFixed(1)}" y2="${(sy + hb * 0.55).toFixed(1)}" stroke="#ff4b55" stroke-width="${sw}"/>` +
              `<line x1="${(sx + hb * 0.55).toFixed(1)}" y1="${(sy - hb * 0.55).toFixed(1)}" x2="${(sx - hb * 0.55).toFixed(1)}" y2="${(sy + hb * 0.55).toFixed(1)}" stroke="#ff4b55" stroke-width="${sw}"/>`,
          );
        } else {
          // Fighter/Avionics Corner-Bracket TD Box + Inner DO-365B Diamond
          const x0 = sx - hb;
          const x1 = sx + hb;
          const y0 = sy - hb;
          const y1 = sy + hb;
          const dm = Math.max(8 * scale, hb * 0.32);
          parts.push(
            `<g stroke="${col}" stroke-width="${sw}" fill="none">` +
              `<path d="M${x0.toFixed(1)},${(y0 + cl).toFixed(1)} V${y0.toFixed(1)} H${(x0 + cl).toFixed(1)}"/>` +
              `<path d="M${(x1 - cl).toFixed(1)},${y0.toFixed(1)} H${x1.toFixed(1)} V${(y0 + cl).toFixed(1)}"/>` +
              `<path d="M${x0.toFixed(1)},${(y1 - cl).toFixed(1)} V${y1.toFixed(1)} H${(x0 + cl).toFixed(1)}"/>` +
              `<path d="M${(x1 - cl).toFixed(1)},${y1.toFixed(1)} H${x1.toFixed(1)} V${(y1 - cl).toFixed(1)}"/>` +
              `<polygon points="${sx.toFixed(1)},${(sy - dm).toFixed(1)} ${(sx + dm).toFixed(1)},${sy.toFixed(1)} ${sx.toFixed(1)},${(sy + dm).toFixed(1)} ${(sx - dm).toFixed(1)},${sy.toFixed(1)}" fill="${col}" fill-opacity="0.18"/>` +
              `</g>`,
          );
        }

        // Crisp high-contrast pilot callout tag anchored above TD box
        const statusStr = tr.ghostRejected
          ? "GHOST REJECTED"
          : tr.alertTier >= 3
            ? "WCV WARNING"
            : tr.alertTier === 2
              ? "CAUTION"
              : tr.meshVerified
                ? "MESH 3-SHIP"
                : "TRACK";
        const cpaStr = tr.ttcS < 900 ? `CPA ${Math.round(tr.ttcS)}s` : "CPA --";
        const label = tr.ghostRejected
          ? `${tr.id} · GHOST REJECTED`
          : `INTRUDER ${tr.id} · RNG ${Math.round(tr.rangeM)}m · ${cpaStr} · ${statusStr}`;
        const tagW = (label.length * 7.0 + 16) * scale;
        const tagH = 20 * scale;
        const tx = Math.max(14 * scale, Math.min(w - tagW - 14 * scale, sx - tagW * 0.5));
        const ty = Math.max(92 * scale, sy - hb - tagH - 7 * scale);
        parts.push(
          `<rect x="${tx.toFixed(1)}" y="${ty.toFixed(1)}" width="${tagW.toFixed(1)}" height="${tagH.toFixed(1)}" rx="${(3 * scale).toFixed(1)}" fill="rgba(8,14,20,0.88)" stroke="${col}" stroke-width="${(1.4 * scale).toFixed(1)}"/>` +
            `<text x="${(tx + 8 * scale).toFixed(1)}" y="${(ty + 14 * scale).toFixed(1)}" fill="${col}" font-size="${fs}" font-weight="700" font-family="B612 Mono, Consolas, monospace">${escapeXml(label)}</text>`,
        );

        // DR-10 FRIES EVS rotor-harmonic discrimination sub-tag anchored cleanly below TD box
        const friesText = tr.ghostRejected
          ? "GHOST · NO PROP"
          : tr.isGhost
            ? "UNVERIFIED · NO PROP"
            : "EVS PROP 180Hz · ACFT";
        const friesCol = tr.ghostRejected || tr.isGhost ? "#ff4b55" : "#4cc9ff";
        const friesW = (friesText.length * 6.2 + 12) * scale;
        const friesH = 15 * scale;
        const fx = Math.max(14 * scale, Math.min(w - friesW - 14 * scale, sx - friesW * 0.5));
        const fy = Math.min(h - friesH - 44 * scale, sy + hb + 5 * scale);
        if (allFinite(fx, fy, friesW, friesH) && fy > ty + tagH + 4 * scale) {
          parts.push(
            `<rect x="${fx.toFixed(1)}" y="${fy.toFixed(1)}" width="${friesW.toFixed(1)}" height="${friesH.toFixed(1)}" rx="${(2.5 * scale).toFixed(1)}" fill="rgba(6,12,18,0.84)" stroke="${friesCol}" stroke-width="${(1.0 * scale).toFixed(1)}" opacity="0.92"/>` +
              `<text x="${(fx + 6 * scale).toFixed(1)}" y="${(fy + 11 * scale).toFixed(1)}" fill="${friesCol}" font-size="${fsSm}" font-weight="700" font-family="B612 Mono, Consolas, monospace">${escapeXml(friesText)}</text>`,
          );
        }
      } else if (!tr.ghostRejected && allFinite(tr.relBearingRad, tr.rangeM)) {
        // Off-screen traffic directional chevron clamped to safe HUD inner perimeter
        const isRight = tr.relBearingRad >= 0;
        const ex = isRight ? w - 335 * scale : 318 * scale;
        const ey = Math.max(130 * scale, Math.min(h - 140 * scale, h * 0.44));
        const dir = isRight ? 1 : -1;
        const label = `${tr.id} ${Math.round(tr.rangeM)}m ${isRight ? "►" : "◄"}`;
        parts.push(
          `<g>` +
            `<polygon points="${(ex + dir * 10 * scale).toFixed(1)},${ey.toFixed(1)} ${(ex - dir * 6 * scale).toFixed(1)},${(ey - 8 * scale).toFixed(1)} ${(ex - dir * 6 * scale).toFixed(1)},${(ey + 8 * scale).toFixed(1)}" fill="${col}"/>` +
            `<text x="${(isRight ? ex - 96 * scale : ex + 14 * scale).toFixed(1)}" y="${(ey + 4 * scale).toFixed(1)}" fill="${col}" font-size="${fsSm}" font-weight="700" font-family="B612 Mono, Consolas, monospace">${escapeXml(label)}</text>` +
            `</g>`,
        );
      }
    }

    this.conformalSvg.innerHTML = parts.join("");
  }

  private _renderHsiRadar(data: ConformalAndHsiFrameData): void {
    const cx = 93;
    const cy = 96;
    const outerR = 72;
    const maxRangeM = 3000.0;
    const pxPerM = outerR / maxRangeM;

    const safeYawRad = Number.isFinite(data.yawRad) ? data.yawRad : 0.0;
    const safeAltM = Number.isFinite(data.altWgs84M) ? data.altWgs84M : 0.0;
    const cosY = Math.cos(safeYawRad);
    const sinY = Math.sin(safeYawRad);
    // Rotate NED (relN, relE) into heading-up screen coords: +X right, -Y forward (up on radar)
    const toRadar = (relN: number, relE: number): [number, number] => {
      const fwd = relN * cosY + relE * sinY;
      const right = -relN * sinY + relE * cosY;
      return [cx + right * pxPerM, cy - fwd * pxPerM];
    };

    const hdgDeg = (((safeYawRad * 180) / Math.PI) % 360 + 360) % 360;
    const parts: string[] = [];

    // Circular radar background + range rings (1 km, 2 km, 3 km)
    parts.push(
      `<circle cx="${cx}" cy="${cy}" r="${outerR}" fill="rgba(6,13,20,0.92)" stroke="rgba(76,201,255,0.55)" stroke-width="1.3"/>` +
        `<circle cx="${cx}" cy="${cy}" r="${(outerR * 0.667).toFixed(1)}" fill="none" stroke="rgba(110,145,185,0.28)" stroke-width="0.9" stroke-dasharray="3 3"/>` +
        `<circle cx="${cx}" cy="${cy}" r="${(outerR * 0.333).toFixed(1)}" fill="none" stroke="rgba(110,145,185,0.28)" stroke-width="0.9" stroke-dasharray="3 3"/>`,
    );

    // 88-deg forward camera FOV cone wedge
    const fovHalf = (44 * Math.PI) / 180;
    const fxL = cx - Math.sin(fovHalf) * outerR;
    const fyL = cy - Math.cos(fovHalf) * outerR;
    const fxR = cx + Math.sin(fovHalf) * outerR;
    const fyR = cy - Math.cos(fovHalf) * outerR;
    parts.push(
      `<path d="M${cx},${cy} L${fxL.toFixed(1)},${fyL.toFixed(1)} A${outerR},${outerR} 0 0,1 ${fxR.toFixed(1)},${fyR.toFixed(1)} Z" fill="rgba(44,230,138,0.08)" stroke="rgba(44,230,138,0.28)" stroke-width="0.8"/>`,
    );

    // Cardinal compass markers rotated heading-up
    const cardinals: ReadonlyArray<{ lbl: string; angDeg: number }> = [
      { lbl: "N", angDeg: 0 },
      { lbl: "E", angDeg: 90 },
      { lbl: "S", angDeg: 180 },
      { lbl: "W", angDeg: 270 },
    ];
    for (const c of cardinals) {
      const a = (c.angDeg * Math.PI) / 180 - safeYawRad;
      const tx = cx + Math.sin(a) * (outerR - 8);
      const ty = cy - Math.cos(a) * (outerR - 8) + 3.5;
      parts.push(
        `<text x="${tx.toFixed(1)}" y="${ty.toFixed(1)}" text-anchor="middle" fill="${c.lbl === "N" ? "#ffbe3b" : "#93a7bd"}" font-size="9" font-weight="700" font-family="B612 Mono, Consolas, monospace">${c.lbl}</text>`,
      );
    }

    // KSFO 28L Runway centerline segment on HSI (clamped neatly inside radar ring)
    if (data.hsiRunway28LRelNe) {
      const [rN, rE] = data.hsiRunway28LRelNe;
      if (allFinite(rN, rE, data.hsiRunway28LHdgRad)) {
        const dR = Math.hypot(rN, rE);
        if (dR < 12000) {
          const endN = rN + Math.cos(data.hsiRunway28LHdgRad) * 3200;
          const endE = rE + Math.sin(data.hsiRunway28LHdgRad) * 3200;
          const [x0, y0] = toRadar(rN, rE);
          const [x1, y1] = toRadar(endN, endE);
          // Clip line cleanly inside circular bezel via SVG clipPath
          parts.push(
            `<defs><clipPath id="xvs-hsi-clip"><circle cx="${cx}" cy="${cy}" r="${outerR - 2}"/></clipPath></defs>` +
              `<g clip-path="url(#xvs-hsi-clip)">` +
              `<line x1="${x0.toFixed(1)}" y1="${y0.toFixed(1)}" x2="${x1.toFixed(1)}" y2="${y1.toFixed(1)}" stroke="#2ce68a" stroke-width="3.2" stroke-linecap="round" opacity="0.88"/>` +
              `<circle cx="${x0.toFixed(1)}" cy="${y0.toFixed(1)}" r="2.6" fill="#d6ffea"/>` +
              `</g>`,
          );
        }
      }
    }

    // Optional 3-ship MeshDAA cooperative peers on radar
    if (data.hsiPeersRelNe && data.hsiPeersRelNe.length > 0) {
      for (const peer of data.hsiPeersRelNe) {
        if (!allFinite(peer[0], peer[1])) continue;
        const [px, py] = toRadar(peer[0], peer[1]);
        const distPx = Math.hypot(px - cx, py - cy);
        const k = distPx > outerR - 6 ? (outerR - 6) / distPx : 1.0;
        const bx = cx + (px - cx) * k;
        const by = cy + (py - cy) * k;
        parts.push(
          `<circle cx="${bx.toFixed(1)}" cy="${by.toFixed(1)}" r="2.8" fill="none" stroke="#4cc9ff" stroke-width="1.2"/>`,
        );
      }
    }

    // Live intruder blips + velocity vectors
    for (const tr of data.hsiTracks) {
      if (!allFinite(tr.relNorthM, tr.relEastM)) continue;
      const [rx, ry] = toRadar(tr.relNorthM, tr.relEastM);
      const distPx = Math.hypot(rx - cx, ry - cy);
      const clamped = distPx > outerR - 7;
      const k = clamped ? (outerR - 7) / distPx : 1.0;
      const bx = cx + (rx - cx) * k;
      const by = cy + (ry - cy) * k;

      const col = tr.ghostRejected
        ? "#ff4b55"
        : tr.alertTier >= 3
          ? "#ff4b55"
          : tr.alertTier === 2
            ? "#ffbe3b"
            : tr.meshVerified
              ? "#2ce68a"
              : "#4cc9ff";

      if (tr.ghostRejected) {
        parts.push(
          `<line x1="${(bx - 4).toFixed(1)}" y1="${(by - 4).toFixed(1)}" x2="${(bx + 4).toFixed(1)}" y2="${(by + 4).toFixed(1)}" stroke="#ff4b55" stroke-width="1.6"/>` +
            `<line x1="${(bx + 4).toFixed(1)}" y1="${(by - 4).toFixed(1)}" x2="${(bx - 4).toFixed(1)}" y2="${(by + 4).toFixed(1)}" stroke="#ff4b55" stroke-width="1.6"/>`,
        );
      } else {
        // Velocity leader line (8s lookahead)
        if (allFinite(tr.velNorthMps, tr.velEastMps)) {
          const [vxEnd, vyEnd] = toRadar(
            tr.relNorthM + tr.velNorthMps * 8.0,
            tr.relEastM + tr.velEastMps * 8.0,
          );
          const vdx = vxEnd - rx;
          const vdy = vyEnd - ry;
          const vLen = Math.hypot(vdx, vdy);
          if (vLen > 1.0) {
            const cap = Math.min(18, vLen);
            parts.push(
              `<line x1="${bx.toFixed(1)}" y1="${by.toFixed(1)}" x2="${(bx + (vdx / vLen) * cap).toFixed(1)}" y2="${(by + (vdy / vLen) * cap).toFixed(1)}" stroke="${col}" stroke-width="1.3"/>`,
            );
          }
        }
        parts.push(
          `<polygon points="${bx.toFixed(1)},${(by - 4.5).toFixed(1)} ${(bx + 4.5).toFixed(1)},${by.toFixed(1)} ${bx.toFixed(1)},${(by + 4.5).toFixed(1)} ${(bx - 4.5).toFixed(1)},${by.toFixed(1)}" fill="${col}" stroke="#081018" stroke-width="0.7"/>` +
            `<text x="${(bx + 6).toFixed(1)}" y="${(by + 3).toFixed(1)}" fill="${col}" font-size="8.5" font-weight="700" font-family="B612 Mono, Consolas, monospace">${escapeXml(tr.id)}</text>`,
        );
      }
    }

    // Ownship aircraft icon at center
    parts.push(
      `<polygon points="${cx},${cy - 7} ${cx + 5.5},${cy + 5} ${cx},${cy + 2.2} ${cx - 5.5},${cy + 5}" fill="#ffffff" stroke="#081018" stroke-width="0.8"/>`,
    );

    // Top digital heading pill + sensor mode tag + bottom range legend
    const sensorTag = data.efvsEnabled ? "LWIR EFVS" : "EO DAY";
    const sensorCol = data.efvsEnabled ? "#2ce68a" : "#4cc9ff";
    parts.push(
      `<rect x="64" y="3" width="58" height="16" rx="3" fill="rgba(8,14,22,0.95)" stroke="#4cc9ff" stroke-width="1"/>` +
        `<text x="93" y="14.5" text-anchor="middle" fill="#e6f0fa" font-size="10.5" font-weight="700" font-family="B612 Mono, Consolas, monospace">${hdgDeg.toFixed(0).padStart(3, "0")}°</text>` +
        `<text x="8" y="14" fill="${sensorCol}" font-size="8.5" font-weight="700" font-family="B612 Mono, Consolas, monospace">${sensorTag}</text>` +
        `<text x="178" y="14" text-anchor="end" fill="#93a7bd" font-size="8.5" font-family="B612 Mono, Consolas, monospace">3km</text>` +
        `<text x="93" y="181" text-anchor="middle" fill="#93a7bd" font-size="8.5" font-family="B612 Mono, Consolas, monospace">HSI TRAFFIC · ALT ${Math.round(safeAltM)}m</text>`,
    );

    this.hsiSvg.innerHTML = parts.join("");
  }

  getCurrentPerturbations(): PhysicalPerturbationState {
    return this.currentPerturbations;
  }
}
