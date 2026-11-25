/**
 * OpenXVS Studio v5.2 Entry Point — Photorealistic & Flyable 3D Simulator + 3D Cockpit
 * + 6/6 Pure-Rust ORT + XFeat + LighterGlue + MultiShipCoord IPC + v5.2 Avionics Telemetry.
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
import { RealWorldBenchModal } from "./ui/realworld_bench.ts";
import { StudioHud } from "./ui/studio_hud.ts";

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
  setBenchOpen(open: boolean): void;
  setDomeVisible(visible: boolean): void;
  stepSeconds(sec: number): void;
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
  }
}

function ensureImmutableNonFlightWatermark(rootEl: HTMLElement): void {
  let observer: MutationObserver | null = null;
  let applying = false;

  const applyWatermark = () => {
    if (applying) return;
    applying = true;
    observer?.disconnect();
    try {
      let el = document.getElementById("studio-non-flight-watermark");
      if (!el || el.parentElement !== rootEl) {
        el = document.createElement("div");
        el.id = "studio-non-flight-watermark";
        rootEl.appendChild(el);
      }
      if (el.className !== "studio-non-flight-watermark") {
        el.className = "studio-non-flight-watermark";
      }
      if (el.textContent !== NON_FLIGHT_WATERMARK) {
        el.textContent = NON_FLIGHT_WATERMARK;
      }
      if (el.getAttribute("role") !== "status") {
        el.setAttribute("role", "status");
      }
      if (el.getAttribute("aria-live") !== "polite") {
        el.setAttribute("aria-live", "polite");
      }
      if (el.getAttribute("data-non-flight-sim-only") !== "true") {
        el.setAttribute("data-non-flight-sim-only", "true");
      }
      if (
        el.style.getPropertyValue("display") !== "flex" ||
        el.style.getPropertyPriority("display") !== "important"
      ) {
        el.style.setProperty("display", "flex", "important");
      }
      if (
        el.style.getPropertyValue("visibility") !== "visible" ||
        el.style.getPropertyPriority("visibility") !== "important"
      ) {
        el.style.setProperty("visibility", "visible", "important");
      }
      if (
        el.style.getPropertyValue("opacity") !== "1" ||
        el.style.getPropertyPriority("opacity") !== "important"
      ) {
        el.style.setProperty("opacity", "1", "important");
      }
    } finally {
      applying = false;
      if (observer) {
        observer.takeRecords();
        observer.observe(rootEl, {
          childList: true,
          subtree: true,
          attributes: true,
          characterData: true,
        });
      }
    }
  };

  if (typeof MutationObserver !== "undefined") {
    observer = new MutationObserver(() => {
      applyWatermark();
    });
  }
  applyWatermark();
}

async function bootStudio(): Promise<void> {
  const rootEl = document.getElementById("studio-root");
  const canvasEl = document.getElementById("studio-canvas") as HTMLCanvasElement | null;
  if (!rootEl || !canvasEl) {
    throw new Error("Missing #studio-root or #studio-canvas mount elements");
  }

  // REL4: Mount and lock the immutable non-flight simulation watermark before any HUD/scene boot
  ensureImmutableNonFlightWatermark(rootEl);

  const engine = new StudioEngine(canvasEl);
  const sim = new StudioScenarioSim("cbdr");
  const bridge = new StudioBridgeClient("http://127.0.0.1:8765");

  let terrainLoaded = false;
  let orthoLoaded = false;

  const pilotInput = new PilotInputCapture({
    onToggleEngineOut: () => {
      sim.toggleEngineOut();
    },
    onResetApproach: () => {
      sim.resetApproach();
      engine.resetTrails();
    },
  });

  const bench = new RealWorldBenchModal(rootEl, bridge, (open) => {
    // Pause flight keyboard capture whenever user interacts inside Video & Camera modal
    pilotInput.setEnabled(!open);
  });

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
      bench.setLocale(hud.getLocale());
      bench.setOpen(true);
    },
  });

  // Camera & UI hotkeys (1-4 cameras, M manual/autopilot, V video lab, B visual mode, Escape closes modal)
  window.addEventListener("keydown", (ev) => {
    if (
      ev.target instanceof HTMLInputElement ||
      ev.target instanceof HTMLSelectElement ||
      ev.target instanceof HTMLTextAreaElement
    ) {
      return;
    }
    if (ev.key === "Escape" && bench.isOpen()) {
      bench.setOpen(false);
      return;
    }
    if (bench.isOpen()) return;
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
      const next: FlightControlMode = sim.getControlMode() === "autopilot" ? "manual" : "autopilot";
      sim.setControlMode(next);
      hud.setControlMode(next);
    } else if (ev.key === "b" || ev.key === "B") {
      const next: VisualWorldMode = engine.getVisualMode() === "photoreal" ? "obsidian" : "photoreal";
      engine.setVisualMode(next);
      hud.setVisualMode(next);
    } else if (ev.key === "v" || ev.key === "V") {
      bench.setLocale(hud.getLocale());
      bench.setOpen(true);
    }
  });

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

  // Non-blocking sequential 5/5 bridge poller (never cancels in-flight requests!)
  let bridgeLoopScheduled = false;
  const runBridgePollCycle = async (): Promise<void> => {
    if (bridgeLoopScheduled) return;
    bridgeLoopScheduled = true;
    try {
      if (!bridge.isOnline()) {
        const health = await bridge.probeHealth();
        hud.syncBridgeBadge();
        if (health) {
          const sensorHealth = await bridge.fetchSensorHealth();
          if (sensorHealth) hud.applySensorHealth(sensorHealth);
          if (hud.isDrawerOpen()) {
            const first = await bridge.fetchDemoTick(sim.getScenario());
            if (first) hud.applyBridgeChannels(first);
          }
        }
      } else if (!bench.isOpen()) {
        if (hud.isDrawerOpen()) {
          const chs = await bridge.fetchDemoTick(sim.getScenario());
          if (chs) hud.applyBridgeChannels(chs);
        }
        const sensorHealth = await bridge.fetchSensorHealth();
        if (sensorHealth) hud.applySensorHealth(sensorHealth);
      }
    } finally {
      bridgeLoopScheduled = false;
      const nextDelayMs = bridge.isOnline() ? 950 : 12000;
      setTimeout(() => {
        void runBridgePollCycle();
      }, nextDelayMs);
    }
  };
  void runBridgePollCycle();

  // Seed 1.6s initially so the ActiveRange S-swerve & contrails look alive on Frame 1
  for (let i = 0; i < 48; i++) {
    const snap = sim.step(1 / 30);
    if (i >= 36) engine.renderFrame(snap, 1 / 30);
  }

  let lastMs = performance.now();
  let emaFrameMs = 16.67;
  let latestSnap = sim.getSnapshot();

  const animate = (nowMs: number) => {
    const dtMs = Math.min(50, Math.max(1, nowMs - lastMs));
    lastMs = nowMs;
    emaFrameMs = emaFrameMs * 0.9 + dtMs * 0.1;
    const fps = Math.min(144, 1000 / Math.max(1.0, emaFrameMs));

    const pInput = sim.getControlMode() === "manual" ? pilotInput.sample() : undefined;
    latestSnap = sim.step(dtMs * 0.001, pInput);
    engine.renderFrame(latestSnap, dtMs * 0.001);
    hud.updateTelemetry(latestSnap, fps);
    requestAnimationFrame(animate);
  };
  requestAnimationFrame(animate);

  // Deterministic inspection API for Playwright Visual Quality Gate & automation
  window.__OPENXVS_STUDIO__ = {
    setScenario: (id) => {
      sim.setScenario(id);
      hud.setScenario(id);
      hud.setWeather(sim.getWeather());
      engine.resetTrails();
      for (let i = 0; i < 20; i++) {
        latestSnap = sim.step(1 / 30);
      }
      engine.renderFrame(latestSnap, 1 / 30);
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
      bench.setLocale(loc);
    },
    setDrawerOpen: (open) => {
      hud.setDrawerOpen(open);
    },
    setBenchOpen: (open) => {
      bench.setLocale(hud.getLocale());
      bench.setOpen(open);
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
      hud.updateTelemetry(latestSnap, 60);
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
        version: "5.2",
        rangeHatM: latestSnap.activeRange.rHatM,
        rangeSigmaM: latestSnap.activeRange.sigmaRM,
        sc3bfSafe: latestSnap.sc3bfSafe,
        timeOfDayH: latestSnap.weather.timeOfDayH,
      };
    },
  };
}

void bootStudio().catch((err) => {
  console.error("[OpenXVS Studio v5.2] Fatal init error:", err);
});
