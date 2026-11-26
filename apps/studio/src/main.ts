/**
 * OpenXVS Studio v1.1 Entry Point — Photorealistic & Flyable 3D Simulator + 3D Cockpit
 * + 6/6 Pure-Rust ORT + XFeat + LighterGlue + MultiShipCoord IPC + Avionics Telemetry.
 */

import * as THREE from "three";
import "./ui/styles.css";
import type { VisualWorldMode } from "./scene/atmosphere_world.ts";
import { StudioEngine } from "./scene/studio_engine.ts";
import { parseKsfoTerrainPack } from "./sim/ksfo_terrain_pack.ts";
import {
  PilotInputCapture,
  type FlightControlMode,
} from "./sim/manual_flight_controller.ts";
import {
  StudioScenarioSim,
  type StudioScenarioId,
  type WeatherState,
} from "./sim/studio_sim.ts";
import {
  NON_FLIGHT_SIM_ONLY,
  NON_FLIGHT_WATERMARK,
  StudioBridgeClient,
  type BridgeTransportMode,
  type NativeSensorHealthPayload,
} from "./ui/bridge_client.ts";
import type { StudioCameraPreset, StudioLocale } from "./ui/i18n.ts";
import type { RealWorldBenchModal } from "./ui/realworld_bench.ts";
import { StudioHud } from "./ui/studio_hud.ts";
import { ensureImmutableNonFlightWatermark } from "./ui/watermark.ts";

export interface OpenXvsStudioApi {
  setScenario(id: StudioScenarioId): void;
  setCamera(preset: StudioCameraPreset): void;
  setVisualMode(mode: VisualWorldMode): void;
  setTerrainStyle(mode: "photoreal" | "obsidian" | "hybrid"): void;
  setControlMode(mode: FlightControlMode): void;
  setShieldEnabled(enabled: boolean): void;
  setWeather(partial: Partial<WeatherState>): void;
  setLocale(locale: StudioLocale): void;
  setDrawerOpen(open: boolean): void;
  setBenchOpen(open: boolean): Promise<void>;
  setDomeVisible(visible: boolean): void;
  stepSeconds(sec: number): void;
  setManualClock(enabled: boolean): void;
  stepDeterministic(dtSec?: number): void;
  stepFrame(dtSec?: number): void;
  getStatus(): {
    readonly ready: boolean;
    readonly scenario: StudioScenarioId;
    readonly cameraPreset: StudioCameraPreset;
    readonly visualMode: VisualWorldMode;
    readonly terrainStyle: VisualWorldMode;
    readonly controlMode: FlightControlMode;
    readonly shieldEnabled: boolean;
    readonly shieldOverrideActive: boolean;
    readonly ownshipTriangles: number;
    readonly intruderTriangles: number;
    readonly drawCalls: number;
    readonly fps: number;
    readonly terrainLoaded: boolean;
    readonly orthoLoaded: boolean;
    readonly bridgeOnline: boolean;
    readonly bridgeTransport: BridgeTransportMode;
    readonly sensorHealth: NativeSensorHealthPayload | null;
    readonly nonFlightSimOnly: boolean;
    readonly watermark: string;
    readonly version: string;
    readonly rangeHatM: number;
    readonly rangeSigmaM: number;
    readonly sc3bfSafe: boolean;
    readonly timeOfDayH: number;
  };
}

declare global {
  interface Window {
    __OPENXVS_STUDIO__?: OpenXvsStudioApi;
    __openxvs_studio?: OpenXvsStudioApi;
  }
}

const DRAWER_OCCUPIED_WIDTH_PX = 320;
const MAX_OFFLINE_AUTO_PROBES = 2;

async function bootStudio(): Promise<void> {
  const rootEl = document.getElementById("studio-root");
  const canvasEl = document.getElementById("studio-canvas") as HTMLCanvasElement | null;
  if (!rootEl || !canvasEl) {
    throw new Error("Missing #studio-root or #studio-canvas mount elements");
  }

  // REL4: Mount and lock the immutable non-flight simulation watermark before any HUD/scene boot
  const watermarkHandle = ensureImmutableNonFlightWatermark(rootEl);

  const engine = new StudioEngine(canvasEl);
  const sim = new StudioScenarioSim("cbdr");
  const bridge = new StudioBridgeClient("http://127.0.0.1:8765");
  const appAbort = new AbortController();

  let terrainLoaded = false;
  let orthoLoaded = false;
  let rafId = 0;

  const pilotInput = new PilotInputCapture({
    onToggleEngineOut: () => {
      sim.toggleEngineOut();
    },
    onResetApproach: () => {
      sim.resetApproach();
      engine.resetTrails();
    },
  });

  // Lazily loaded Workbench chunk (keeps initial JS bundle lean)
  let benchInstance: RealWorldBenchModal | null = null;
  let benchLoadPromise: Promise<RealWorldBenchModal> | null = null;

  const ensureBenchModal = async (): Promise<RealWorldBenchModal> => {
    if (benchInstance) return benchInstance;
    if (!benchLoadPromise) {
      benchLoadPromise = import("./ui/realworld_bench.ts").then(({ RealWorldBenchModal }) => {
        benchInstance = new RealWorldBenchModal(rootEl, bridge, (open) => {
          pilotInput.setEnabled(!open);
          hud.setBenchOpen(open);
        });
        benchInstance.setLocale(hud.getLocale());
        return benchInstance;
      });
    }
    return benchLoadPromise;
  };

  const openBenchModal = async (open: boolean): Promise<void> => {
    if (!open) {
      benchInstance?.setOpen(false);
      hud.setBenchOpen(false);
      return;
    }
    const bench = await ensureBenchModal();
    bench.setLocale(hud.getLocale());
    bench.setOpen(true);
  };

  // Bounded exponential backoff on `:8765` health poll (stops auto-polling after 2 failed HTTP probes)
  let bridgePollTimer: ReturnType<typeof setTimeout> | null = null;
  let bridgeLoopInFlight = false;
  let offlineAutoProbes = 0;

  const clearBridgeTimer = () => {
    if (bridgePollTimer !== null) {
      clearTimeout(bridgePollTimer);
      bridgePollTimer = null;
    }
  };

  const scheduleNextBridgePoll = (delayMs: number) => {
    clearBridgeTimer();
    if (appAbort.signal.aborted) return;
    bridgePollTimer = setTimeout(() => {
      void runBridgePollCycle(false);
    }, delayMs);
  };

  const runBridgePollCycle = async (manualRetry: boolean): Promise<boolean> => {
    if (bridgeLoopInFlight) return bridge.isOnline();
    if (manualRetry) {
      offlineAutoProbes = 0;
      clearBridgeTimer();
    }
    bridgeLoopInFlight = true;
    try {
      const isNative = bridge.isNativeTauriIpc();
      if (!bridge.isOnline()) {
        if (!isNative && !manualRetry && offlineAutoProbes >= MAX_OFFLINE_AUTO_PROBES) {
          return false;
        }
        if (!isNative) {
          offlineAutoProbes++;
        }
        const health = await bridge.probeHealth();
        hud.syncBridgeBadge(true);
        if (health) {
          offlineAutoProbes = 0;
          const sensorHealth = await bridge.fetchSensorHealth();
          if (sensorHealth) hud.applySensorHealth(sensorHealth);
          if (hud.isDrawerOpen()) {
            const first = await bridge.fetchDemoTick(sim.getScenario());
            if (first) hud.applyBridgeChannels(first);
          }
          return true;
        }
        return false;
      } else if (!benchInstance?.isOpen()) {
        if (hud.isDrawerOpen()) {
          const chs = await bridge.fetchDemoTick(sim.getScenario());
          if (chs) hud.applyBridgeChannels(chs);
        }
        const sensorHealth = await bridge.fetchSensorHealth();
        if (sensorHealth) hud.applySensorHealth(sensorHealth);
        return bridge.isOnline();
      }
      return true;
    } finally {
      bridgeLoopInFlight = false;
      const isNative = bridge.isNativeTauriIpc();
      if (bridge.isOnline()) {
        scheduleNextBridgePoll(950);
      } else if (isNative) {
        scheduleNextBridgePoll(5000);
      } else if (offlineAutoProbes < MAX_OFFLINE_AUTO_PROBES) {
        // Exponential backoff before pausing auto-probes (15s after 1st miss -> pause after 2nd)
        scheduleNextBridgePoll(15000 * offlineAutoProbes);
      }
    }
  };

  const hud = new StudioHud(rootEl, bridge, {
    onScenarioChange: (scen) => {
      sim.setScenario(scen);
      hud.setWeather(sim.getWeather());
      engine.resetTrails();
    },
    onCameraChange: (preset) => {
      engine.setCameraPreset(preset, false);
    },
    onVisualModeChange: (mode) => {
      engine.setVisualMode(mode);
    },
    onControlModeChange: (mode) => {
      sim.setControlMode(mode);
    },
    onShieldToggle: (enabled) => {
      sim.setShieldEnabled(enabled);
    },
    onWeatherChange: (partial) => {
      sim.setWeather(partial);
    },
    onToggleEngineOut: () => {
      sim.toggleEngineOut();
    },
    onResetApproach: () => {
      sim.resetApproach();
      engine.resetTrails();
    },
    onDomeToggle: (visible) => {
      engine.manifolds.setSensorDomeVisible(visible);
    },
    onOpenVideoBench: () => {
      void openBenchModal(true);
    },
    onDrawerToggle: (open) => {
      engine.setDrawerOffsetPx(open ? DRAWER_OCCUPIED_WIDTH_PX : 0);
    },
    onLocaleChange: (loc) => {
      benchInstance?.setLocale(loc);
    },
    onRetryBridge: async () => runBridgePollCycle(true),
  });

  // Sync initial camera framing offset with default drawer state
  engine.setDrawerOffsetPx(hud.isDrawerOpen() ? DRAWER_OCCUPIED_WIDTH_PX : 0, true);

  // Camera & UI hotkeys (1-4 cameras, M manual/autopilot, V video lab, B visual mode, Escape closes popovers/dialog)
  window.addEventListener(
    "keydown",
    (ev) => {
      if (ev.key === "Escape") {
        if (benchInstance?.isOpen()) {
          benchInstance.setOpen(false);
          return;
        }
        if (hud.isWeatherOpen()) {
          ev.preventDefault();
          hud.setWeatherOpen(false, true);
          return;
        }
        return;
      }

      if (
        ev.ctrlKey ||
        ev.metaKey ||
        ev.altKey ||
        ev.target instanceof HTMLInputElement ||
        ev.target instanceof HTMLSelectElement ||
        ev.target instanceof HTMLTextAreaElement ||
        (ev.target instanceof HTMLElement && ev.target.isContentEditable)
      ) {
        return;
      }

      if (benchInstance?.isOpen()) return;
      if (ev.key === "1") {
        engine.setCameraPreset("chase");
        hud.setCameraPreset("chase");
      } else if (ev.key === "2") {
        engine.setCameraPreset("wingman");
        hud.setCameraPreset("wingman");
      } else if (ev.key === "3") {
        engine.setCameraPreset("overview");
        hud.setCameraPreset("overview");
      } else if (ev.key === "4") {
        engine.setCameraPreset("cockpit");
        hud.setCameraPreset("cockpit");
      } else if (ev.key === "m" || ev.key === "M") {
        const next: FlightControlMode =
          sim.getControlMode() === "autopilot" ? "manual" : "autopilot";
        sim.setControlMode(next);
        hud.setControlMode(next);
      } else if (ev.key === "b" || ev.key === "B") {
        const next: VisualWorldMode =
          engine.getVisualMode() === "photoreal" ? "obsidian" : "photoreal";
        engine.setVisualMode(next);
        hud.setVisualMode(next);
      } else if (ev.key === "v" || ev.key === "V") {
        void openBenchModal(true);
      }
    },
    { signal: appAbort.signal },
  );

  // Deterministic teardown on unload
  window.addEventListener(
    "beforeunload",
    () => {
      cancelAnimationFrame(rafId);
      clearBridgeTimer();
      appAbort.abort();
      benchInstance?.dispose();
      hud.dispose();
      pilotInput.dispose();
      engine.dispose();
      watermarkHandle.dispose();
    },
    { signal: appAbort.signal },
  );

  // Load real USGS 3DEP 256x256 terrain pack + verified local USGS NAIP ortho in parallel (non-blocking)
  void (async () => {
    try {
      const resp = await fetch("./assets/ksfo_3dep_geoid18_v1.bin");
      if (resp.ok) {
        const buf = await resp.arrayBuffer();
        const pack = parseKsfoTerrainPack(buf);
        engine.setTerrainPack(pack);
        sim.setTerrainPack(pack);
        terrainLoaded = true;
      }
    } catch {
      // Graceful procedural San Bruno + Bay fallback stays active
    }
  })();

  const texLoader = new THREE.TextureLoader();
  let coreTex: THREE.Texture | null = null;
  let innerTex: THREE.Texture | null = null;
  const applyOrtho = () => {
    engine.setOrthoTextures(coreTex, innerTex);
    orthoLoaded = Boolean(coreTex || innerTex);
  };
  texLoader.load(
    "./assets/ksfo_ortho_core2k_v2.jpg",
    (tex) => {
      coreTex = tex;
      applyOrtho();
    },
    undefined,
    () => undefined,
  );
  texLoader.load(
    "./assets/ksfo_ortho_inner_v1.jpg",
    (tex) => {
      innerTex = tex;
      applyOrtho();
    },
    undefined,
    () => undefined,
  );

  // Start native IPC immediately; in browser preview mode defer first probe lightly so initial paint finishes cleanly
  if (bridge.isNativeTauriIpc()) {
    void runBridgePollCycle(false);
  } else {
    scheduleNextBridgePoll(300);
  }

  // Seed 1.6s initially so the ActiveRange S-swerve & contrails look alive on Frame 1
  for (let i = 0; i < 48; i++) {
    const snap = sim.step(1 / 30);
    if (i >= 36) engine.renderFrame(snap, 1 / 30);
  }

  let lastMs = performance.now();
  let emaFrameMs = 16.67;
  let frameSampleCount = 0;
  let manualClock = false;
  let latestSnap = sim.getSnapshot();
  hud.updateTelemetry(latestSnap, 60, true);

  const animate = (nowMs: number) => {
    if (manualClock) {
      rafId = 0;
      return;
    }
    const rawFrameMs = Math.max(1, nowMs - lastMs);
    const simDtMs = Math.min(50, rawFrameMs);
    lastMs = nowMs;

    // Skip the very first rAF tick's shader-compile boot delta; use un-clamped rawFrameMs for honest FPS
    frameSampleCount++;
    if (frameSampleCount === 2) {
      emaFrameMs = rawFrameMs;
    } else if (frameSampleCount > 2) {
      emaFrameMs = emaFrameMs * 0.85 + rawFrameMs * 0.15;
    }
    const fps = Math.min(144, 1000 / Math.max(1.0, emaFrameMs));

    const pInput = sim.getControlMode() === "manual" ? pilotInput.sample() : undefined;
    latestSnap = sim.step(simDtMs * 0.001, pInput);
    engine.renderFrame(latestSnap, simDtMs * 0.001);
    hud.updateTelemetry(latestSnap, fps);
    rafId = requestAnimationFrame(animate);
  };
  rafId = requestAnimationFrame(animate);

  const stepDeterministicFrame = (dtSec = 1 / 30): void => {
    const safeDt = Math.max(1e-4, Math.min(0.25, Number.isFinite(dtSec) ? dtSec : 1 / 30));
    const subSteps = Math.max(1, Math.ceil(safeDt / 0.05));
    const subDt = safeDt / subSteps;
    const pInput = sim.getControlMode() === "manual" ? pilotInput.sample() : undefined;
    for (let i = 0; i < subSteps; i++) {
      latestSnap = sim.step(subDt, pInput);
    }
    engine.renderFrame(latestSnap, safeDt);
    const nominalFps = Math.min(144, Math.max(1, Math.round(1 / safeDt)));
    hud.updateTelemetry(latestSnap, nominalFps, true);
  };

  // Deterministic inspection API for Playwright Visual Quality Gate & automation
  const studioApi: OpenXvsStudioApi = {
    setScenario: (id) => {
      sim.setScenario(id);
      hud.setScenario(id);
      hud.setWeather(sim.getWeather());
      engine.resetTrails();
      for (let i = 0; i < 20; i++) {
        latestSnap = sim.step(1 / 30);
      }
      engine.renderFrame(latestSnap, 1 / 30);
      hud.updateTelemetry(latestSnap, Math.min(144, 1000 / Math.max(1.0, emaFrameMs)), true);
    },
    setCamera: (preset) => {
      engine.setCameraPreset(preset, true);
      hud.setCameraPreset(preset);
      engine.renderFrame(latestSnap, 1 / 60);
    },
    setVisualMode: (mode) => {
      engine.setVisualMode(mode);
      hud.setVisualMode(mode);
      engine.renderFrame(latestSnap, 1 / 60);
    },
    setTerrainStyle: (mode) => {
      const mapped: VisualWorldMode = mode === "obsidian" ? "obsidian" : "photoreal";
      engine.setVisualMode(mapped);
      hud.setVisualMode(mapped);
      engine.renderFrame(latestSnap, 1 / 60);
    },
    setControlMode: (mode) => {
      sim.setControlMode(mode);
      hud.setControlMode(mode);
    },
    setShieldEnabled: (enabled) => {
      sim.setShieldEnabled(enabled);
      hud.setShieldEnabled(enabled);
    },
    setWeather: (partial) => {
      const w = sim.setWeather(partial);
      hud.setWeather(w);
      latestSnap = sim.getSnapshot();
      engine.renderFrame(latestSnap, 1 / 60);
    },
    setLocale: (loc) => {
      hud.setLocale(loc);
      benchInstance?.setLocale(loc);
      hud.updateTelemetry(latestSnap, Math.min(144, 1000 / Math.max(1.0, emaFrameMs)), true);
    },
    setDrawerOpen: (open) => {
      hud.setDrawerOpen(open);
      engine.setDrawerOffsetPx(open ? DRAWER_OCCUPIED_WIDTH_PX : 0, true);
      engine.renderFrame(latestSnap, 1 / 60);
    },
    setBenchOpen: async (open) => {
      await openBenchModal(open);
    },
    setDomeVisible: (visible) => {
      engine.manifolds.setSensorDomeVisible(visible);
      hud.setDomeVisible(visible);
      engine.renderFrame(latestSnap, 1 / 60);
    },
    stepSeconds: (sec) => {
      const steps = Math.max(1, Math.round(sec * 30));
      for (let i = 0; i < steps; i++) {
        latestSnap = sim.step(1 / 30);
      }
      engine.renderFrame(latestSnap, 1 / 30);
      hud.updateTelemetry(latestSnap, Math.min(144, 1000 / Math.max(1.0, emaFrameMs)), true);
    },
    setManualClock: (enabled) => {
      manualClock = Boolean(enabled);
      if (manualClock) {
        if (rafId !== 0) {
          cancelAnimationFrame(rafId);
          rafId = 0;
        }
      } else {
        lastMs = performance.now();
        if (rafId === 0) {
          rafId = requestAnimationFrame(animate);
        }
      }
    },
    stepDeterministic: (dtSec = 1 / 30) => {
      stepDeterministicFrame(dtSec);
    },
    stepFrame: (dtSec = 1 / 30) => {
      stepDeterministicFrame(dtSec);
    },
    getStatus: () => {
      const st = engine.getStats();
      return {
        ready: true,
        scenario: sim.getScenario(),
        cameraPreset: st.cameraPreset,
        visualMode: st.visualMode,
        terrainStyle: st.visualMode,
        controlMode: sim.getControlMode(),
        shieldEnabled: sim.isShieldEnabled(),
        shieldOverrideActive: latestSnap.shieldOverrideActive,
        ownshipTriangles: st.ownshipTriangles,
        intruderTriangles: st.intruderTriangles,
        drawCalls: st.drawCalls,
        fps: Math.round(Math.min(144, 1000 / Math.max(1.0, emaFrameMs))),
        terrainLoaded,
        orthoLoaded,
        bridgeOnline: bridge.isOnline(),
        bridgeTransport: bridge.getTransport(),
        sensorHealth: bridge.getSensorHealth(),
        nonFlightSimOnly: NON_FLIGHT_SIM_ONLY,
        watermark: NON_FLIGHT_WATERMARK,
        version: "1.1",
        rangeHatM: latestSnap.activeRange.rHatM,
        rangeSigmaM: latestSnap.activeRange.sigmaRM,
        sc3bfSafe: latestSnap.sc3bfSafe,
        timeOfDayH: latestSnap.weather.timeOfDayH,
      };
    },
  };
  window.__OPENXVS_STUDIO__ = studioApi;
  window.__openxvs_studio = studioApi;
}

void bootStudio().catch((err) => {
  console.error("[OpenXVS Studio] Fatal init error:", err);
});
