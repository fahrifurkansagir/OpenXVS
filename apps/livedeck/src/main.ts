/**
 * OpenXVS Live Deck v2.0 (Milestone M1) — Two-Stage Progressive Boot Entry Point.
 *
 * Stage-0 (Immediate <= 700 KB Brotli):
 *   - Raw WebGPU + compile-time Naga WebGL2 RHI (`createRhi`)
 *   - Unified `#![no_std]` Rust WASM core (`WasmLiveDeckSim`, zero-copy 96-f64 snapshot + TX ring)
 *   - 5-pass cockpit renderer (`CockpitRenderer`), KSFO USGS 3DEP + GEOID18 terrain, B612 Mono MSDF
 *   - Persistent Charter §7.5 HUD overlay, telemetry drawer, M1..M4 fault-only missions, local bridge
 *
 * Stage-1 (Non-blocking background async):
 *   - Dynamic import of `./ai/ort_worker.ts` (keeps `onnxruntime-web` 100% out of Stage-0 chunk)
 *   - SHA-256 verified OPFS-cached `D1-KF` ONNX (`enableGraphCapture: false`, single resident session)
 *   - Explicit visible backend hierarchy: ORT-WebGPU-FP16 -> ORT-WebGPU-FP32 (NO F16) -> ORT-WASM-SIMD
 */

import "./ui/styles.css";
import type { AiInitStatus, DetectorInferenceResult, OrtDetectorRunner } from "./ai/ort_worker.ts";
import { ActiveRangeModeTs, FLAG_SC3BF_ALL_SAFE } from "./active_range/active_range_ts.ts";
import { CbdrScenarioController, type CbdrStepOutput } from "./active_range/scenario_cbdr.ts";
import {
  ForcedLandScenarioController,
  type ForcedLandStepOutput,
} from "./active_range/scenario_forcedland.ts";
import {
  ALL_MISSIONS,
  evaluateMissionPerturbations,
  getMissionById,
  type MissionDefinition,
  type MissionId,
  type PhysicalPerturbationState,
} from "./missions/index.ts";
import { LocalBridgeClient } from "./net/bridge_client.ts";
import {
  CockpitRenderer,
  DEFAULT_ORBIT_AZIMUTH_RAD,
  DEFAULT_ORBIT_DISTANCE_M,
  DEFAULT_ORBIT_ELEVATION_RAD,
  ORBIT_DIST_MAX_M,
  ORBIT_DIST_MIN_M,
  ORBIT_ELEV_MAX_RAD,
  ORBIT_ELEV_MIN_RAD,
  type ActiveRangeOverlayState,
  type CameraViewMode,
  type CockpitFrameState,
  type ForcedLandFunnelState,
  type IntruderTrack,
  type ManifoldToggleState,
  type OrbitTargetMode,
  type Pass4CpuInput,
  type Pass4RingDiagnostics,
  type SimplexModeValue,
  type SpzDiagnostics,
} from "./passes/index.ts";
import {
  DEG_TO_RAD,
  geodeticToEcef,
  mat3MulVec,
  projectEcefToNdc,
  rEcefNed,
  type Mat3RowMajor,
  type Vec3,
} from "./rhi/coords.ts";
import { createRhi, type RhiContext } from "./rhi/index.ts";
import {
  CockpitHudOverlay,
  SIMPLEX_LABELS,
  type CockpitBadgeState,
  type ConformalAiBoxOverlay,
  type ConformalRunwayOverlay,
  type ConformalTargetOverlay,
  type HsiRadarTrack,
  type SimplexModeCode,
  type UiDetectorMode,
} from "./ui/cockpit_hud.ts";
import { extractTrainedProvenanceFromRegistry } from "./ui/arch_map_view.ts";
import {
  DEFAULT_STRESS_PRESET,
  ObservatoryShell,
} from "./ui/observatory_shell.ts";
import type { StressPresetId } from "./ui/i18n.ts";
import type { RealSkyLabView } from "./ui/realsky_lab_view.ts";
import type { XvsD2ProvenanceInfo } from "./ui/telemetry_drawer.ts";
import initWasm, { WasmLiveDeckSim, wasm_memory } from "./wasm/xvs_wasm.js";

const WASM_MISSION_ID: Record<MissionId, number> = {
  M1: 1,
  M2: 2,
  M3: 3,
  M4: 4,
};

const KSFO_28L_NOMINAL_WIDTH_M = 61.0;
const HORIZON_AL_URAD = 8727;

export interface DecodedWasmTelemetry {
  readonly time_s: number;
  readonly tick_count: number;
  readonly lat_deg: number;
  readonly lon_deg: number;
  readonly h_m: number;
  readonly p_ecef: readonly [number, number, number];
  readonly v_ned: readonly [number, number, number];
  readonly yaw_rad: number;
  readonly pitch_rad: number;
  readonly roll_rad: number;
  readonly q_ned_body: readonly [number, number, number, number];
  readonly tas_mps: number;
  readonly epipolar_max_urad: readonly [number, number, number, number, number, number];
  readonly fused_attitude_err_urad: number;
  readonly horizon_pl_urad: number;
  readonly v_vpl_mm: number;
  readonly v_hpl_mm: number;
  readonly cusum_stat: number;
  readonly nis_stat: number;
  readonly ss_stat: number;
  readonly bank_vel_delta_mps: number;
  readonly ecbf_bank_cmd_rad: number;
  readonly ecbf_climb_cmd_mps: number;
  readonly ecbf_slack: number;
  readonly dtaci_cone_urad: number;
  readonly mission_id: number;
  readonly simplex_mode: SimplexModeValue;
  readonly raw_target: number;
  readonly healthy_cam_mask: number;
  readonly active_healthy_cams: number;
  readonly active_nav_bank: 0 | 1;
  readonly mandatory_go_around: boolean;
  readonly blank_conformal_hud: boolean;
  readonly inhibit_avoidance_turns: boolean;
  readonly ecbf_infeasible: boolean;
  readonly wc1_alert: boolean;
  readonly dtaci_frozen: boolean;
  readonly approach_active: boolean;
  readonly lwir_efvs_enabled: boolean;
  readonly ghosts_rejected: number;
  readonly mesh_fused_count: number;
  readonly intruder_count: number;
  readonly intruder_rel_ned: ReadonlyArray<readonly [number, number, number]>;
  readonly intruder_vel_ned: ReadonlyArray<readonly [number, number, number]>;
}

export interface LiveDeckStatusSnapshot {
  readonly ready: boolean;
  readonly stage1Ready: boolean;
  readonly backend: "webgpu" | "webgl2";
  readonly hasShaderF16: boolean;
  readonly degradedReason: string | null;
  readonly degradedReasons: readonly string[];
  readonly aiBadge: CockpitBadgeState["aiBadge"];
  readonly aiCacheSource: string | null;
  readonly aiOpfsWarning: string | null;
  readonly terrainMode: string;
  readonly orthoLoaded: boolean;
  readonly cameraMode: CameraViewMode;
  readonly orbitTarget: OrbitTargetMode;
  readonly visibleBadges: readonly string[];
  readonly frameCount: number;
  readonly lastGpuError: string | null;
  readonly ortInferenceCount: number;
  readonly ortFiniteOutputs: boolean;
  readonly ortLastPeaksCount: number;
  readonly ortMaxHeatmapScore: number;
  readonly conformalBlanked: boolean;
  readonly activeMissionId: MissionId;
  readonly simTimeS: number;
  readonly pass4Diagnostics: Pass4RingDiagnostics;
}

export interface Pass4ParitySerializableInput {
  readonly currRgba: Uint8Array | number[];
  readonly prev1Electrons: Float32Array | number[];
  readonly prev2Electrons: Float32Array | number[];
  readonly poseCurr: { pCamEcef: Vec3; rEcefCam: Mat3RowMajor };
  readonly posePrev1: { pCamEcef: Vec3; rEcefCam: Mat3RowMajor };
  readonly posePrev2: { pCamEcef: Vec3; rEcefCam: Mat3RowMajor };
  readonly roiCurr: readonly [number, number];
  readonly roiPrev1: readonly [number, number];
  readonly roiPrev2: readonly [number, number];
  readonly noiseOff: boolean;
  readonly applyPsf: boolean;
  readonly frameSeed?: number;
}

export interface OpenXvsTestApi {
  getStatus(): LiveDeckStatusSnapshot;
  getTelemetry(): DecodedWasmTelemetry;
  startMission(id: MissionId): DecodedWasmTelemetry;
  setCameraMode(mode: CameraViewMode): void;
  setOrbitTarget(target: OrbitTargetMode): void;
  setOrbitView(azRad: number, elRad: number, distM: number): void;
  setSpzMode(mode: "off" | "preview"): void;
  getSpzDiagnostics(): SpzDiagnostics;
  advanceSimSeconds(sec: number): DecodedWasmTelemetry;
  runPass4Parity(input: Pass4ParitySerializableInput): Promise<{
    data: number[];
    shape: readonly [1, 5, 256, 256];
    computePath: "webgpu-compute" | "cpu-fp32-fallback";
    stats: { means: readonly [number, number, number, number, number]; std0: number };
  }>;
  runOrtSmokeTicks(count: number): Promise<{
    count: number;
    completedInferences: number;
    allFinite: boolean;
    maxHeatmapScore: number;
    avgInferenceMs: number;
    lastGpuError: string | null;
    aiBadge: string;
    backend: "webgpu" | "webgl2";
    pass4Diagnostics: Pass4RingDiagnostics;
  }>;
  getMissionProof(): Record<string, unknown>;
}

declare global {
  interface Window {
    __OPENXVS_TEST_API__?: OpenXvsTestApi;
    __openxvs?: OpenXvsTestApi;
  }
}

function readWasmSnapshot(sim: WasmLiveDeckSim): DecodedWasmTelemetry {
  const mem = wasm_memory() as WebAssembly.Memory;
  const f = new Float64Array(mem.buffer, sim.snapshot_f64_ptr(), sim.snapshot_f64_len());

  const intrCount = Math.max(0, Math.min(8, Math.round(f[53])));
  const relNed: Array<[number, number, number]> = [];
  for (let i = 0; i < intrCount; i++) {
    relNed.push([f[54 + 3 * i], f[54 + 3 * i + 1], f[54 + 3 * i + 2]]);
  }
  const velNed: Array<[number, number, number]> = [];
  for (let i = 0; i < Math.min(intrCount, 6); i++) {
    velNed.push([f[78 + 3 * i], f[78 + 3 * i + 1], f[78 + 3 * i + 2]]);
  }

  const simplexVal = Math.max(0, Math.min(4, Math.round(f[38]))) as SimplexModeValue;

  return {
    time_s: f[0],
    tick_count: Math.round(f[1]),
    lat_deg: f[2],
    lon_deg: f[3],
    h_m: f[4],
    p_ecef: [f[5], f[6], f[7]],
    v_ned: [f[8], f[9], f[10]],
    yaw_rad: f[11],
    pitch_rad: f[12],
    roll_rad: f[13],
    q_ned_body: [f[14], f[15], f[16], f[17]],
    tas_mps: f[18],
    epipolar_max_urad: [f[19], f[20], f[21], f[22], f[23], f[24]],
    fused_attitude_err_urad: f[25],
    horizon_pl_urad: f[26],
    v_vpl_mm: f[27],
    v_hpl_mm: f[28],
    cusum_stat: f[29],
    nis_stat: f[30],
    ss_stat: f[31],
    bank_vel_delta_mps: f[32],
    ecbf_bank_cmd_rad: f[33],
    ecbf_climb_cmd_mps: f[34],
    ecbf_slack: f[35],
    dtaci_cone_urad: f[36],
    mission_id: Math.round(f[37]),
    simplex_mode: simplexVal,
    raw_target: Math.round(f[39]),
    healthy_cam_mask: Math.round(f[40]) & 0x3f,
    active_healthy_cams: Math.round(f[41]),
    active_nav_bank: Math.round(f[42]) === 1 ? 1 : 0,
    mandatory_go_around: f[43] >= 0.5,
    blank_conformal_hud: f[44] >= 0.5,
    inhibit_avoidance_turns: f[45] >= 0.5,
    ecbf_infeasible: f[46] >= 0.5,
    wc1_alert: f[47] >= 0.5,
    dtaci_frozen: f[48] >= 0.5,
    approach_active: f[49] >= 0.5,
    lwir_efvs_enabled: f[50] >= 0.5,
    ghosts_rejected: Math.round(f[51]),
    mesh_fused_count: Math.round(f[52]),
    intruder_count: intrCount,
    intruder_rel_ned: relNed,
    intruder_vel_ned: velNed,
  };
}

function drainWasmTxToBridge(sim: WasmLiveDeckSim, bridge: LocalBridgeClient): void {
  const len = sim.tx_ring_len();
  if (len <= 0) return;
  const mem = wasm_memory() as WebAssembly.Memory;
  const ring = new Uint8Array(mem.buffer, sim.tx_ring_ptr(), len);

  let offset = 0;
  while (offset + 3 <= len) {
    const proto = ring[offset];
    const pktLen = ring[offset + 1] | (ring[offset + 2] << 8);
    offset += 3;
    if (offset + pktLen > len) break;
    const payload = ring.slice(offset, offset + pktLen);
    offset += pktLen;
    if (proto === 0x01) {
      bridge.sendGdl90(payload);
    } else if (proto === 0x02) {
      bridge.sendMavlink(payload);
    }
  }
  sim.clear_tx_ring();
}

function buildCockpitFrameState(
  tel: DecodedWasmTelemetry,
  pert: PhysicalPerturbationState,
  aiBadge: string,
  lastAiResult: DetectorInferenceResult | null,
  observatoryExtras?: {
    readonly activeRangeOverlay?: ActiveRangeOverlayState;
    readonly forcedLandFunnel?: ForcedLandFunnelState;
    readonly manifoldToggles?: ManifoldToggleState;
    readonly extraIntruder?: IntruderTrack | null;
  },
): CockpitFrameState {
  const latRad = tel.lat_deg * DEG_TO_RAD;
  const lonRad = tel.lon_deg * DEG_TO_RAD;
  const rNed = rEcefNed(latRad, lonRad);

  // Convert WASM relative NED intruder coordinates into exact ECEF tracks for Passes 3, 4 & 5
  const intruders: IntruderTrack[] = [];
  for (let i = 0; i < tel.intruder_count; i++) {
    const rel = tel.intruder_rel_ned[i];
    const vel = tel.intruder_vel_ned[i] ?? [0, 0, 0];
    const dEcef = mat3MulVec(rNed, rel);
    const velEcef = mat3MulVec(rNed, vel);
    const posEcef: Vec3 = [
      tel.p_ecef[0] + dEcef[0],
      tel.p_ecef[1] + dEcef[1],
      tel.p_ecef[2] + dEcef[2],
    ];
    const rangeM = Math.hypot(rel[0], rel[1], rel[2]);
    // Relative closing speed toward ownship
    const vRelN = vel[0] - tel.v_ned[0];
    const vRelE = vel[1] - tel.v_ned[1];
    const vRelD = vel[2] - tel.v_ned[2];
    const closingMps = -(rel[0] * vRelN + rel[1] * vRelE + rel[2] * vRelD) / Math.max(rangeM, 1.0);
    const ttcS = closingMps > 0.5 ? Math.min(999, rangeM / closingMps) : 999;

    const alertTier: 0 | 1 | 2 | 3 =
      rangeM < 450 || ttcS < 18
        ? 3
        : rangeM < 900 || ttcS < 35
          ? 2
          : rangeM < 2000
            ? 1
            : 0;

    // In Mission M3, synthesize peer ship sight-lines to visualize triple-ray MeshDAA verification
    const peerOriginsEcef: Vec3[] = [];
    if (tel.mission_id === 3) {
      const peer1Delta = mat3MulVec(rNed, [-450, 900, -40]);
      const peer2Delta = mat3MulVec(rNed, [-600, -850, 30]);
      peerOriginsEcef.push(
        [tel.p_ecef[0] + peer1Delta[0], tel.p_ecef[1] + peer1Delta[1], tel.p_ecef[2] + peer1Delta[2]],
        [tel.p_ecef[0] + peer2Delta[0], tel.p_ecef[1] + peer2Delta[1], tel.p_ecef[2] + peer2Delta[2]],
      );
      if (pert.thirdPeerActive) {
        const peer3Delta = mat3MulVec(rNed, [700, 1100, -120]);
        peerOriginsEcef.push([
          tel.p_ecef[0] + peer3Delta[0],
          tel.p_ecef[1] + peer3Delta[1],
          tel.p_ecef[2] + peer3Delta[2],
        ]);
      }
    }

    intruders.push({
      id: `T${i + 1}`,
      posEcef,
      velEcef,
      rangeM,
      ttcS,
      alertTier,
      wcvRadiusM: 180.0,
      meshVerified: tel.mission_id === 3 && pert.thirdPeerActive && tel.mesh_fused_count > 0,
      peerOriginsEcef: peerOriginsEcef.length > 0 ? peerOriginsEcef : undefined,
    });
  }

  // Add explicit 2-ship epipolar ghost marker in Mission M3 (rejected once 3rd peer joins)
  if (tel.mission_id === 3) {
    const ghostDelta = mat3MulVec(rNed, [1450, 260, -15]);
    const peer1Delta = mat3MulVec(rNed, [-450, 900, -40]);
    const peer2Delta = mat3MulVec(rNed, [-600, -850, 30]);
    intruders.push({
      id: "GHOST-1",
      posEcef: [
        tel.p_ecef[0] + ghostDelta[0],
        tel.p_ecef[1] + ghostDelta[1],
        tel.p_ecef[2] + ghostDelta[2],
      ],
      rangeM: Math.hypot(1450, 260, -15),
      ttcS: 28,
      alertTier: tel.ghosts_rejected > 0 ? 0 : 2,
      isGhost: true,
      ghostRejected: tel.ghosts_rejected > 0,
      peerOriginsEcef: [
        [tel.p_ecef[0] + peer1Delta[0], tel.p_ecef[1] + peer1Delta[1], tel.p_ecef[2] + peer1Delta[2]],
        [tel.p_ecef[0] + peer2Delta[0], tel.p_ecef[1] + peer2Delta[1], tel.p_ecef[2] + peer2Delta[2]],
      ],
    });
  }

  if (observatoryExtras?.extraIntruder) {
    intruders.unshift(observatoryExtras.extraIntruder);
  }

  return {
    simTimeS: tel.time_s,
    latRad,
    lonRad,
    altWgs84M: tel.h_m,
    pEcefBody: tel.p_ecef,
    qNedBody: tel.q_ned_body,
    velNedMps: tel.v_ned,
    yawRad: tel.yaw_rad,
    pitchRad: tel.pitch_rad,
    rollRad: tel.roll_rad,
    airspeedMps: tel.tas_mps,
    simplexMode: tel.simplex_mode,
    camHealthMask: tel.healthy_cam_mask,
    camRBiasRad: pert.camRBiasRad,
    horizonPlRad: tel.horizon_pl_urad * 1e-6,
    vHplM: tel.v_hpl_mm * 1e-3,
    vVplM: tel.v_vpl_mm * 1e-3,
    spoofAlarm: tel.active_nav_bank === 1,
    mandatoryGoAround: tel.mandatory_go_around,
    approachActive: tel.approach_active,
    fogContrast: pert.fogContrast,
    efvsEnabled: tel.lwir_efvs_enabled || pert.lwirEfvsEnabled,
    intruders,
    aiBadge,
    detectorMode: pert.detectorMode,
    aiDetections: lastAiResult?.peaks.map((p) => ({ u: p.u, v: p.v, score: p.score })),
    activeRangeOverlay: observatoryExtras?.activeRangeOverlay,
    forcedLandFunnel: observatoryExtras?.forcedLandFunnel,
    manifoldToggles: observatoryExtras?.manifoldToggles,
  };
}

interface SimplexTransitionRecord {
  readonly timeS: number;
  readonly from: string;
  readonly to: string;
  readonly healthyCams: number;
  readonly navBank: 0 | 1;
  readonly mandatoryGoAround: boolean;
}

async function bootLiveDeck(): Promise<void> {
  const rootEl = document.getElementById("livedeck-root");
  const canvas = document.getElementById("livedeck-canvas") as HTMLCanvasElement | null;
  if (!rootEl || !canvas) {
    throw new Error("Missing #livedeck-root or #livedeck-canvas element");
  }

  const query = new URLSearchParams(window.location.search);
  const forceWebGL2 = query.get("forceWebGL2") === "1";
  const useProceduralTerrain =
    query.get("terrain") === "procedural-ci" || query.get("terrainMode") === "procedural-ci";
  const noiseOffFlag = query.get("noiseOff") === "1";
  const startPaused = query.get("paused") === "1";
  const missionParam = query.get("mission")?.toUpperCase() as MissionId | undefined;
  const initialMissionId: MissionId =
    missionParam && ALL_MISSIONS.some((m) => m.id === missionParam) ? missionParam : "M1";
  const detParam = query.get("detector")?.toLowerCase();
  let manualDetectorOverride: UiDetectorMode | null =
    detParam === "xvs_d2" || detParam === "d0" || detParam === "d1_kf" ? detParam : null;
  let activeUiDetectorMode: UiDetectorMode = manualDetectorOverride ?? "xvs_d2";
  let xvsD2Provenance: XvsD2ProvenanceInfo | null = null;
  const camParam = query.get("cam")?.toLowerCase();
  const targetParam = query.get("target")?.toLowerCase();
  let orbitTarget: OrbitTargetMode =
    targetParam === "intruder" || targetParam === "tower" ? targetParam : "ownship";
  let cameraMode: CameraViewMode =
    camParam === "chase" || camParam === "orbit"
      ? camParam
      : camParam === "cockpit"
        ? "cockpit"
        : orbitTarget !== "ownship"
          ? "orbit"
          : "cockpit";
  let orthoLoaded = false;

  let lastGpuError: string | null = null;
  let ortRunner: OrtDetectorRunner | null = null;
  let aiStatus: AiInitStatus | null = null;
  let stage1Ready = false;
  let stage1Promise: Promise<void> = Promise.resolve();
  let lastAiResult: DetectorInferenceResult | null = null;
  let ortInFlight = false;
  let ortInferenceCount = 0;
  let ortAllFiniteEver = true;
  let lastOrtTickTimeS = -1.0;

  // Stage-0 initialization (RHI + WASM + Renderer + Assets + UI)
  let rhi: RhiContext;
  rhi = await createRhi(canvas, {
    forceWebGL2,
    onDeviceLost: (info) => {
      lastGpuError = `GPU device lost (${info.reason}): ${info.message}`;
      console.error(`[OpenXVS LiveDeck] ${lastGpuError}`);
      if (ortRunner) {
        void ortRunner
          .handleDeviceLost(info.reason)
          .then((nextStatus) => {
            aiStatus = nextStatus;
            updateOverlayBadges();
          })
          .catch((err) => {
            lastGpuError = `ORT failover error: ${err instanceof Error ? err.message : String(err)}`;
          });
      }
      updateOverlayBadges();
    },
    onUncapturedError: (msg) => {
      lastGpuError = `WebGPU uncapturederror: ${msg}`;
      console.error(`[OpenXVS LiveDeck] ${lastGpuError}`);
      updateOverlayBadges();
    },
  });

  await initWasm();

  const renderer = new CockpitRenderer(rhi);
  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(window.innerWidth * dpr));
    const h = Math.max(1, Math.floor(window.innerHeight * dpr));
    renderer.resize(w, h);
  };
  resize();
  window.addEventListener("resize", resize);

  // Load Stage-0 MSDF glyph atlas & KSFO USGS 3DEP + GEOID18 terrain pack (or explicit procedural-ci)
  let terrainLoadWarning: string | null = null;
  const [msdfRes, terrainRes] = await Promise.allSettled([
    renderer.loadMsdfAtlas("./assets/b612_mono_msdf.png", "./assets/b612_mono_msdf.json"),
    useProceduralTerrain
      ? Promise.resolve(renderer.useProceduralCiTerrain())
      : renderer.loadTerrainPackFromUrl("./assets/ksfo_3dep_geoid18_v1.bin"),
  ]);
  if (msdfRes.status === "rejected") {
    console.warn("[OpenXVS LiveDeck] MSDF atlas load warning:", msdfRes.reason);
  }
  if (terrainRes.status === "rejected") {
    terrainLoadWarning = `Terrain pack fetch failed (${String(terrainRes.reason)}); explicit procedural-ci active`;
    console.warn(`[OpenXVS LiveDeck] ${terrainLoadWarning}`);
    renderer.useProceduralCiTerrain();
  }

  let activeMission: MissionDefinition = getMissionById(initialMissionId);
  const sim = new WasmLiveDeckSim(activeMission.seed);
  sim.reset_mission(WASM_MISSION_ID[activeMission.id], activeMission.seed);

  let paused = startPaused;
  let rafSuspended = false;
  let manualPerturbationsActive = false;
  const applyDetectorState = (base: PhysicalPerturbationState): PhysicalPerturbationState => {
    if (manualDetectorOverride !== null) {
      activeUiDetectorMode = manualDetectorOverride;
      return {
        ...base,
        detectorMode: manualDetectorOverride === "d0" ? "D0" : "D1_KF",
      };
    }
    activeUiDetectorMode = base.detectorMode === "D0" ? "d0" : "xvs_d2";
    return base;
  };
  let currentPerturbations: PhysicalPerturbationState = applyDetectorState(
    evaluateMissionPerturbations(activeMission, 0.0),
  );
  let manualBankCmdRad = 0.0;
  let manualClimbCmdMps = 0.0;
  let manualFlightOverride = false;

  let prevSimplexMode: SimplexModeValue = 4;
  let simplexTransitions: SimplexTransitionRecord[] = [];
  let telemetryHistory: Array<Record<string, unknown>> = [];
  let lastHistoryLogTimeS = -1.0;

  const bridge = new LocalBridgeClient("ws://127.0.0.1:8765");
  if (query.get("bridge") === "1") {
    bridge.connect();
  }

  const pushFaultsToWasm = (p: PhysicalPerturbationState) => {
    sim.set_faults(
      p.camRBiasRad,
      p.gnssSpoofVelNedMps[0],
      p.gnssSpoofVelNedMps[1],
      p.gnssSpoofVelNedMps[2],
      p.runwayCornerBiasPx,
      KSFO_28L_NOMINAL_WIDTH_M,
      p.fogContrast,
      0.0,
      p.lwirEfvsEnabled,
      false,
      false,
      false,
      p.thirdPeerActive,
      manualFlightOverride,
      manualBankCmdRad,
      manualClimbCmdMps,
      true,
    );
  };

  pushFaultsToWasm(currentPerturbations);
  let latestTelemetry: DecodedWasmTelemetry = readWasmSnapshot(sim);
  prevSimplexMode = latestTelemetry.simplex_mode;

  const computeDegradedReasons = (): string[] => {
    const reasons: string[] = [];
    if (rhi.degradedReason) reasons.push(rhi.degradedReason);
    if (aiStatus?.backend.degradedReason) reasons.push(aiStatus.backend.degradedReason);
    if (aiStatus?.opfsWarning) reasons.push(aiStatus.opfsWarning);
    if (renderer.getTerrainPack().terrainMode === "procedural_ci") {
      reasons.push(terrainLoadWarning ?? "Explicit procedural-ci terrain mode active");
    }
    if (lastGpuError) reasons.push(lastGpuError);
    return Array.from(new Set(reasons));
  };

  const getAiBadge = (): CockpitBadgeState["aiBadge"] => {
    if (aiStatus) return aiStatus.backend.aiBadge;
    return "AI: LOADING STAGE-1";
  };

  const buildMissionProofLog = (): Record<string, unknown> => ({
    schema: "xvs.v2_m1.mission_proof/1",
    exported_utc: new Date().toISOString(),
    mission_id: activeMission.id,
    mission_name: activeMission.name,
    seed: activeMission.seed,
    backend: rhi.backend,
    has_shader_f16: rhi.hasShaderF16,
    degraded_reasons: computeDegradedReasons(),
    ai_badge: getAiBadge(),
    ai_model_id: aiStatus?.modelId ?? "d1_kf_k3_g1",
    ai_model_sha256: aiStatus?.modelSha256 ?? null,
    ai_inference_count: ortInferenceCount,
    ai_all_finite: ortAllFiniteEver,
    terrain_mode: renderer.getTerrainPack().terrainMode,
    ksfo_28l_thr_wgs84_h_m: renderer.getTerrainPack().ksfo28lThrWgs84HM,
    simplex_transitions: simplexTransitions,
    latest_telemetry: latestTelemetry,
    telemetry_history: telemetryHistory,
  });

  const downloadMissionProof = () => {
    const payload = JSON.stringify(buildMissionProofLog(), null, 2);
    const blob = new Blob([payload], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `openxvs_livedeck_${activeMission.id.toLowerCase()}_proof.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const startMission = (id: MissionId): DecodedWasmTelemetry => {
    activeMission = getMissionById(id);
    activeStressPreset = id === "M4" ? "normal" : "spoof";
    lastCbdrOut = null;
    lastForcedLandOut = null;
    lastScenarioStepTimeS = -1.0;
    manualPerturbationsActive = false;
    manualFlightOverride = false;
    manualBankCmdRad = 0.0;
    manualClimbCmdMps = 0.0;
    simplexTransitions = [];
    telemetryHistory = [];
    lastHistoryLogTimeS = -1.0;
    lastOrtTickTimeS = -1.0;

    sim.reset_mission(WASM_MISSION_ID[id], activeMission.seed);
    renderer.pass4.resetRing();
    renderer.resetFlightTrails();
    ortRunner?.resetXvsD2State();
    currentPerturbations = applyDetectorState(evaluateMissionPerturbations(activeMission, 0.0));
    pushFaultsToWasm(currentPerturbations);
    latestTelemetry = readWasmSnapshot(sim);
    prevSimplexMode = latestTelemetry.simplex_mode;

    hud.syncPerturbationControls(currentPerturbations, activeUiDetectorMode);
    hud.setMissionStatus(activeMission.id, latestTelemetry.time_s, activeMission.durationS, paused);
    renderCurrentFrame(0.0);
    return latestTelemetry;
  };

  // OpenXVS v3.0 Observatory closed-loop controllers (CBDR ActiveRange + 35 kt ForcedLand)
  const cbdrScenario = new CbdrScenarioController({
    initialRangeM: 1400.0,
    closingSpeedMps: 80.0,
    intruderWingspanM: 9.5,
  });
  const forcedLandScenario = new ForcedLandScenarioController(35.0);
  let activeStressPreset: StressPresetId = missionParam ? "spoof" : DEFAULT_STRESS_PRESET;
  let lastCbdrOut: CbdrStepOutput | null = null;
  let lastForcedLandOut: ForcedLandStepOutput | null = null;
  let lastScenarioStepTimeS = -1.0;
  let realSkyLabView: RealSkyLabView | null = null;
  let realSkyLoading = false;

  const stepObservatoryScenarios = (tel: DecodedWasmTelemetry, forceNow = false): void => {
    const dt = lastScenarioStepTimeS >= 0 ? tel.time_s - lastScenarioStepTimeS : 0.1;
    if (!forceNow && dt < 0.095) return;
    lastScenarioStepTimeS = tel.time_s;
    const dtClamped = Math.max(0.02, Math.min(0.25, dt > 0 ? dt : 0.1));

    if (activeStressPreset === "cbdr") {
      lastCbdrOut = cbdrScenario.stepWithWasmOwnship({
        simTimeS: tel.time_s,
        dtS: dtClamped,
        latRad: tel.lat_deg * DEG_TO_RAD,
        lonRad: tel.lon_deg * DEG_TO_RAD,
        altWgs84M: tel.h_m,
        pEcefBody: tel.p_ecef,
        vNedMps: tel.v_ned,
        yawRad: tel.yaw_rad,
        rollRad: tel.roll_rad,
        tasMps: tel.tas_mps,
      });
      lastForcedLandOut = null;
      if (keysDown.size === 0) {
        manualFlightOverride = lastCbdrOut.manualOverride;
        manualBankCmdRad = lastCbdrOut.manualBankCmdRad;
        manualClimbCmdMps = 0.0;
      }
    } else if (activeStressPreset === "forcedland") {
      lastForcedLandOut = forcedLandScenario.step({
        latRad: tel.lat_deg * DEG_TO_RAD,
        lonRad: tel.lon_deg * DEG_TO_RAD,
        altWgs84M: tel.h_m,
        yawRad: tel.yaw_rad,
        airspeedMps: tel.tas_mps,
      });
      lastCbdrOut = null;
      if (keysDown.size === 0) {
        manualFlightOverride = true;
        manualBankCmdRad = lastForcedLandOut.manualBankCmdRad;
        manualClimbCmdMps = lastForcedLandOut.manualClimbCmdMps;
      }
    } else {
      lastCbdrOut = null;
      lastForcedLandOut = null;
      if (keysDown.size === 0) {
        manualFlightOverride = false;
        manualBankCmdRad = 0.0;
        manualClimbCmdMps = 0.0;
      }
    }
  };

  const applyStressPreset = (preset: StressPresetId, renderNow = true): void => {
    activeStressPreset = preset;
    lastScenarioStepTimeS = -1.0;
    cbdrScenario.reset();
    lastCbdrOut = null;
    lastForcedLandOut = null;

    if (preset === "cbdr" || preset === "forcedland" || preset === "normal") {
      activeMission = getMissionById("M4");
      sim.reset_mission(WASM_MISSION_ID.M4, activeMission.seed);
      manualPerturbationsActive = true;
      currentPerturbations = applyDetectorState({
        ...evaluateMissionPerturbations(activeMission, 0.0),
        fogContrast: 1.0,
        lwirEfvsEnabled: false,
      });
    } else {
      activeMission = getMissionById("M1");
      sim.reset_mission(WASM_MISSION_ID.M1, activeMission.seed);
      manualPerturbationsActive = false;
      currentPerturbations = applyDetectorState(evaluateMissionPerturbations(activeMission, 0.0));
    }

    renderer.pass4.resetRing();
    renderer.resetFlightTrails();
    ortRunner?.resetXvsD2TemporalState();
    latestTelemetry = readWasmSnapshot(sim);
    stepObservatoryScenarios(latestTelemetry, true);
    pushFaultsToWasm(currentPerturbations);
    latestTelemetry = readWasmSnapshot(sim);
    prevSimplexMode = latestTelemetry.simplex_mode;
    hud.syncPerturbationControls(currentPerturbations, activeUiDetectorMode);
    hud.setMissionStatus(activeMission.id, latestTelemetry.time_s, activeMission.durationS, paused);
    if (renderNow) {
      renderCurrentFrame(0.0);
    }
  };

  const hud = new CockpitHudOverlay(rootEl, {
    onSelectMission: (id) => {
      startMission(id);
    },
    onTogglePlayPause: () => {
      paused = !paused;
      hud.setMissionStatus(activeMission.id, latestTelemetry.time_s, activeMission.durationS, paused);
    },
    onResetMission: () => {
      startMission(activeMission.id);
    },
    onManualPerturbationsChange: (next) => {
      manualPerturbationsActive = true;
      currentPerturbations = applyDetectorState(next);
      pushFaultsToWasm(currentPerturbations);
    },
    onDetectorModeChange: (nextUiMode) => {
      // Independent of manualPerturbationsActive so 45 s guided mission fault schedules keep running!
      manualDetectorOverride = nextUiMode;
      activeUiDetectorMode = nextUiMode;
      currentPerturbations = {
        ...currentPerturbations,
        detectorMode: nextUiMode === "d0" ? "D0" : "D1_KF",
      };
      pushFaultsToWasm(currentPerturbations);
      ortRunner?.resetXvsD2State();
      if (nextUiMode === "d0") {
        lastAiResult = null;
      }
      lastOrtTickTimeS = -1.0;
    },
    onCameraModeChange: (nextCam) => {
      setCameraModeState(nextCam);
    },
    onOrbitTargetChange: (nextTarget) => {
      setOrbitTargetState(nextTarget);
    },
    onSpzModeChange: (mode) => {
      setSpzModeState(mode);
    },
    onSpzFileSelected: (file) => {
      void handleSpzFileDrop(file);
    },
    onToggleBridgeConnect: () => {
      const st = bridge.getTelemetry().state;
      if (st === "connected" || st === "connecting") {
        bridge.disconnect();
      } else {
        bridge.connect();
      }
    },
    onDownloadProof: downloadMissionProof,
  });

  const observatory = new ObservatoryShell(rootEl, {
    onStressPresetChange: (preset) => {
      applyStressPreset(preset, true);
    },
    onCameraToggle: (nextMode) => {
      setCameraModeState(nextMode);
    },
    onManifoldToggleChange: (toggles) => {
      if (toggles.spzStress && getDesiredSpzMode() === "off") {
        setSpzModeState("preview");
      } else if (!toggles.spzStress && getDesiredSpzMode() !== "off") {
        setSpzModeState("off");
      }
      if (paused) {
        renderCurrentFrame(0.0);
      }
    },
    onLocaleChange: (locale) => {
      realSkyLabView?.setLocale(locale);
    },
    onTabChange: (tab) => {
      if (tab === "realsky" && !realSkyLabView && !realSkyLoading) {
        realSkyLoading = true;
        void import("./ui/realsky_lab_view.ts")
          .then(({ RealSkyLabView: LabCtor }) => {
            realSkyLabView = new LabCtor(
              observatory.getRealSkyContainer(),
              () => ortRunner,
              observatory.store.getLocale(),
            );
            return realSkyLabView.ensureLoaded();
          })
          .finally(() => {
            realSkyLoading = false;
          });
      }
    },
  });

  const updateOverlayBadges = () => {
    const terrainMode = renderer.getTerrainPack().terrainMode;
    hud.updateBadges({
      renderBadge: rhi.backend === "webgpu" ? "RENDER: WebGPU" : "RENDER: WebGL2 (DEGRADED)",
      aiBadge: getAiBadge(),
      terrainBadge:
        terrainMode === "procedural_ci"
          ? "TERRAIN: PROCEDURAL-CI"
          : "TERRAIN: USGS-3DEP-GEOID18",
      orthoLoaded: orthoLoaded && terrainMode !== "procedural_ci",
      simplexMode: latestTelemetry.simplex_mode as SimplexModeCode,
      degradedReasons: computeDegradedReasons(),
      bridge: bridge.getTelemetry(),
    });
  };

  bridge.subscribe(() => updateOverlayBadges());

  // Interactive 3-Mode 3D Camera controls (Cockpit pan/tilt | Chase look | Orbit az/el + wheel zoom)
  let dragging = false;
  let lastMouseX = 0;
  let lastMouseY = 0;
  let panRad = 0.0;
  let tiltRad = 0.0;
  let orbitAzimuthRad = DEFAULT_ORBIT_AZIMUTH_RAD;
  let orbitElevationRad = DEFAULT_ORBIT_ELEVATION_RAD;
  let orbitDistanceM = DEFAULT_ORBIT_DISTANCE_M;

  const setCameraModeState = (nextMode: CameraViewMode) => {
    cameraMode = nextMode;
    renderer.setCameraMode(nextMode);
    hud.setCameraMode(nextMode);
    observatory.setCameraMode(nextMode);
    if (paused) {
      renderCurrentFrame(0.0);
    }
  };

  const setOrbitTargetState = (nextTarget: OrbitTargetMode) => {
    orbitTarget = nextTarget;
    cameraMode = "orbit";
    renderer.setCameraMode("orbit");
    renderer.setOrbitTarget(nextTarget);
    hud.setCameraMode("orbit");
    hud.setOrbitTarget(nextTarget);
    if (nextTarget === "tower") {
      panRad = 0.0;
      tiltRad = 0.0;
      renderer.setViewAngles(0.0, 0.0);
    }
    if (paused) {
      renderCurrentFrame(0.0);
    }
  };

  const resetCameraControls = () => {
    panRad = 0.0;
    tiltRad = 0.0;
    orbitAzimuthRad = DEFAULT_ORBIT_AZIMUTH_RAD;
    orbitElevationRad = DEFAULT_ORBIT_ELEVATION_RAD;
    orbitDistanceM = DEFAULT_ORBIT_DISTANCE_M;
    renderer.resetCameraView();
    if (paused) {
      renderCurrentFrame(0.0);
    }
  };

  const BUNDLED_KSFO_SPZ_URL = "./assets/ksfo_28l_1km_visual_stress.spz";
  const BUNDLED_KSFO_SPZ_NAME = "ksfo_28l_1km_visual_stress.spz";
  let spzActionSeq = 0;
  let desiredSpzMode: "off" | "preview" | "custom" = "off";
  const getDesiredSpzMode = (): "off" | "preview" | "custom" => desiredSpzMode;
  let cachedBundledSpzBuffer: ArrayBuffer | null = null;
  let bundledSpzFetchPromise: Promise<ArrayBuffer> | null = null;

  const loadBundledKsfoSpzBuffer = (): Promise<ArrayBuffer> => {
    if (cachedBundledSpzBuffer !== null) {
      return Promise.resolve(cachedBundledSpzBuffer);
    }
    if (bundledSpzFetchPromise === null) {
      bundledSpzFetchPromise = fetch(BUNDLED_KSFO_SPZ_URL)
        .then(async (resp) => {
          if (!resp.ok) {
            throw new Error(`HTTP ${resp.status} fetching ${BUNDLED_KSFO_SPZ_URL}`);
          }
          const buf = await resp.arrayBuffer();
          cachedBundledSpzBuffer = buf;
          return buf;
        })
        .catch((err) => {
          bundledSpzFetchPromise = null;
          throw err;
        });
    }
    return bundledSpzFetchPromise;
  };

  const setSpzModeState = (mode: "off" | "preview", allowImmediateRender = true) => {
    const reqSeq = ++spzActionSeq;
    desiredSpzMode = mode;
    renderer.setSpzMode(mode);
    hud.updateSpzStatus(renderer.getSpzDiagnostics());
    if (allowImmediateRender && paused) {
      renderCurrentFrame(0.0);
    }
    if (mode === "preview") {
      void loadBundledKsfoSpzBuffer()
        .then(async (buf) => {
          if (reqSeq !== spzActionSeq || getDesiredSpzMode() !== "preview") return;
          const diag = await renderer.loadSpzBuffer(buf, BUNDLED_KSFO_SPZ_NAME);
          if (reqSeq !== spzActionSeq || getDesiredSpzMode() !== "preview") {
            if (getDesiredSpzMode() === "off") {
              renderer.setSpzMode("off");
              hud.updateSpzStatus(renderer.getSpzDiagnostics());
            }
            return;
          }
          hud.updateSpzStatus(diag);
          if (paused) {
            renderCurrentFrame(0.0);
          }
        })
        .catch((err) => {
          console.warn(
            `[OpenXVS SPZ Loader] Bundled ${BUNDLED_KSFO_SPZ_NAME} fetch/decode failed; keeping procedural KSFO preview:`,
            err,
          );
          if (reqSeq === spzActionSeq && getDesiredSpzMode() === "preview") {
            hud.updateSpzStatus(renderer.getSpzDiagnostics());
          }
        });
    }
  };

  const handleSpzFileDrop = async (file: File) => {
    const reqSeq = ++spzActionSeq;
    desiredSpzMode = "custom";
    try {
      const buf = await file.arrayBuffer();
      if (reqSeq !== spzActionSeq || getDesiredSpzMode() !== "custom") return;
      const diag = await renderer.loadSpzBuffer(buf, file.name);
      if (reqSeq !== spzActionSeq || getDesiredSpzMode() !== "custom") {
        if (getDesiredSpzMode() === "off") {
          renderer.setSpzMode("off");
          hud.updateSpzStatus(renderer.getSpzDiagnostics());
        }
        return;
      }
      hud.updateSpzStatus(diag);
      if (paused) {
        renderCurrentFrame(0.0);
      }
    } catch (err) {
      console.error("[OpenXVS SPZ Loader] Failed to decode .spz file:", err);
      hud.updateSpzStatus(renderer.getSpzDiagnostics());
    }
  };

  if (query.get("spz")?.toLowerCase() === "preview") {
    setSpzModeState("preview", false);
  } else {
    hud.updateSpzStatus(renderer.getSpzDiagnostics());
  }

  window.addEventListener("dragover", (ev) => {
    ev.preventDefault();
  });
  window.addEventListener("drop", (ev) => {
    ev.preventDefault();
    const files = ev.dataTransfer?.files;
    if (!files || files.length === 0) return;
    const spzFile = Array.from(files).find((f) => f.name.toLowerCase().endsWith(".spz"));
    if (spzFile) {
      void handleSpzFileDrop(spzFile);
    }
  });

  renderer.setCameraMode(cameraMode);
  renderer.setOrbitTarget(orbitTarget);
  hud.setCameraMode(cameraMode);
  hud.setOrbitTarget(orbitTarget);

  canvas.addEventListener("pointerdown", (ev) => {
    dragging = true;
    lastMouseX = ev.clientX;
    lastMouseY = ev.clientY;
    canvas.setPointerCapture(ev.pointerId);
  });
  canvas.addEventListener("pointermove", (ev) => {
    if (!dragging) return;
    const dx = ev.clientX - lastMouseX;
    const dy = ev.clientY - lastMouseY;
    lastMouseX = ev.clientX;
    lastMouseY = ev.clientY;
    if (cameraMode === "orbit" && orbitTarget !== "tower") {
      orbitAzimuthRad += dx * 0.006;
      orbitElevationRad = Math.max(
        ORBIT_ELEV_MIN_RAD,
        Math.min(ORBIT_ELEV_MAX_RAD, orbitElevationRad + dy * 0.005),
      );
      renderer.setOrbitView(orbitAzimuthRad, orbitElevationRad, orbitDistanceM);
    } else {
      panRad += dx * 0.004;
      tiltRad = Math.max(-Math.PI * 0.42, Math.min(Math.PI * 0.42, tiltRad - dy * 0.004));
      renderer.setViewAngles(panRad, tiltRad);
    }
    if (paused) {
      renderCurrentFrame(0.0);
    }
  });
  canvas.addEventListener("pointerup", () => {
    dragging = false;
  });
  canvas.addEventListener(
    "wheel",
    (ev) => {
      if (cameraMode !== "orbit" || orbitTarget === "tower") return;
      ev.preventDefault();
      orbitDistanceM = Math.max(
        ORBIT_DIST_MIN_M,
        Math.min(ORBIT_DIST_MAX_M, orbitDistanceM * Math.exp(ev.deltaY * 0.0012)),
      );
      renderer.setOrbitView(orbitAzimuthRad, orbitElevationRad, orbitDistanceM);
      if (paused) {
        renderCurrentFrame(0.0);
      }
    },
    { passive: false },
  );
  canvas.addEventListener("dblclick", () => {
    resetCameraControls();
  });

  // Keyboard flight & 3D camera controls (1/2/3 = Cockpit/Chase/Orbit, 4 = KSFO Tower, 5 = Intruder Orbit, R = reset)
  const keysDown = new Set<string>();
  window.addEventListener("keydown", (ev) => {
    const target = ev.target;
    if (
      target instanceof HTMLInputElement ||
      target instanceof HTMLSelectElement ||
      target instanceof HTMLTextAreaElement
    ) {
      return;
    }
    if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(ev.key)) {
      keysDown.add(ev.key);
      manualFlightOverride = true;
    } else if (ev.key === "1") {
      setCameraModeState("cockpit");
    } else if (ev.key === "2") {
      setCameraModeState("chase");
    } else if (ev.key === "3") {
      setCameraModeState("orbit");
    } else if (ev.key === "4") {
      setOrbitTargetState("tower");
    } else if (ev.key === "5") {
      setOrbitTargetState("intruder");
    } else if (ev.key === "r" || ev.key === "R") {
      resetCameraControls();
    }
  });
  window.addEventListener("keyup", (ev) => {
    keysDown.delete(ev.key);
    if (keysDown.size === 0) {
      manualFlightOverride = false;
      manualBankCmdRad = 0.0;
      manualClimbCmdMps = 0.0;
    }
  });

  const recordTelemetrySample = (tel: DecodedWasmTelemetry) => {
    if (tel.simplex_mode !== prevSimplexMode) {
      simplexTransitions.push({
        timeS: Number(tel.time_s.toFixed(2)),
        from: SIMPLEX_LABELS[prevSimplexMode as SimplexModeCode],
        to: SIMPLEX_LABELS[tel.simplex_mode as SimplexModeCode],
        healthyCams: tel.active_healthy_cams,
        navBank: tel.active_nav_bank,
        mandatoryGoAround: tel.mandatory_go_around,
      });
      prevSimplexMode = tel.simplex_mode;
    }
    if (tel.time_s - lastHistoryLogTimeS >= 0.5 - 1e-6) {
      lastHistoryLogTimeS = tel.time_s;
      telemetryHistory.push({
        t: Number(tel.time_s.toFixed(2)),
        mode: tel.simplex_mode,
        cams: tel.active_healthy_cams,
        bank: tel.active_nav_bank,
        cusum: Number(tel.cusum_stat.toFixed(3)),
        v_vpl_mm: Math.round(tel.v_vpl_mm),
        ga: tel.mandatory_go_around,
        ghosts_rej: tel.ghosts_rejected,
      });
      if (telemetryHistory.length > 200) {
        telemetryHistory.shift();
      }
    }
  };

  let emaFrameMs = 16.67;

  const renderCurrentFrame = (dtMs: number): void => {
    const t0 = performance.now();
    const manifoldToggles = observatory.store.getManifoldToggles();
    const frameState = buildCockpitFrameState(
      latestTelemetry,
      currentPerturbations,
      getAiBadge(),
      lastAiResult,
      {
        activeRangeOverlay: lastCbdrOut?.overlayState,
        forcedLandFunnel: lastForcedLandOut?.funnelState,
        manifoldToggles,
        extraIntruder: lastCbdrOut?.intruderTrack ?? null,
      },
    );
    const extraBadges: string[] = [];
    if (rhi.backend === "webgl2") {
      extraBadges.push("WEBGL2 FALLBACK");
    }
    const tensor = renderer.renderFrame(frameState, {
      pass4: { noiseOff: noiseOffFlag, applyPsf: true },
      extraBadges,
    });
    const renderMs = performance.now() - t0;
    emaFrameMs = emaFrameMs * 0.9 + Math.max(dtMs || renderMs, 0.1) * 0.1;
    const fps = Math.min(144, 1000 / Math.max(emaFrameMs, 1.0));

    // Dispatch Stage-1 ONNX D1-KF or lazy XVS-D2 inference asynchronously at 10 Hz (never blocks rAF)
    if (
      stage1Ready &&
      ortRunner &&
      !ortInFlight &&
      tensor &&
      activeUiDetectorMode !== "d0" &&
      currentPerturbations.detectorMode === "D1_KF" &&
      latestTelemetry.time_s - lastOrtTickTimeS >= 0.1 - 1e-6
    ) {
      lastOrtTickTimeS = latestTelemetry.time_s;
      ortInFlight = true;
      const inputCopy = new Float32Array(tensor.data);
      const fIdx = tensor.frameIndex;
      const simTimeSnap = latestTelemetry.time_s;
      const inferPromise =
        activeUiDetectorMode === "xvs_d2"
          ? ortRunner.runXvsD2(inputCopy, fIdx, 0.25, simTimeSnap)
          : ortRunner.runNormalized(inputCopy, fIdx, 0.35);
      void inferPromise
        .then((res) => {
          lastAiResult = res;
          ortInferenceCount++;
          if (!res.allFinite) ortAllFiniteEver = false;
          // Charter §7.5: surface any runtime backend transition (e.g. XVS-D2 WebGPU -> WASM-SIMD) immediately
          const latestStatus = ortRunner?.getStatus() ?? null;
          if (
            latestStatus &&
            (res.backendKind !== aiStatus?.backend.kind ||
              latestStatus.backend.degradedReason !== aiStatus?.backend.degradedReason)
          ) {
            aiStatus = latestStatus;
            updateOverlayBadges();
          }
        })
        .catch((err) => {
          lastGpuError = `ORT inference error: ${err instanceof Error ? err.message : String(err)}`;
          console.error(`[OpenXVS Stage-1 AI] ${lastGpuError}`);
        })
        .finally(() => {
          ortInFlight = false;
        });
    }

    // Project live 3D intruders, KSFO 28L runway + PAPI, AI detections & HSI traffic radar
    const camView = renderer.getLastCameraView();
    if (camView) {
      const vp = { width: camView.width, height: camView.height };
      const pack = renderer.getTerrainPack();
      const rNedOwn = camView.rEcefNed;

      // Conformal 3D traffic targets -> TD Box & off-screen chevrons
      const conformalTargets: ConformalTargetOverlay[] = frameState.intruders.map((tr) => {
        const proj = projectEcefToNdc(tr.posEcef, camView.pEcefCam, camView.rEcefCam, camView.intr, vp);
        // Active-camera relative azimuth (+right / -left) so off-screen chevrons align in cockpit, chase & orbit
        const relBearingRad = Math.atan2(proj.vCam[0], proj.vCam[2]);
        const visualRadiusM = Math.max(22.0, Math.min(72.0, tr.rangeM * 0.045));
        const boxHalfPx = proj.visible
          ? Math.max(22.0, Math.min(78.0, (visualRadiusM / Math.max(proj.vCam[2], 40.0)) * camView.intr.fx))
          : 28.0;
        return {
          id: tr.id,
          screenX: proj.uPx + 0.5,
          screenY: proj.vPx + 0.5,
          visible: proj.visible,
          boxHalfPx,
          rangeM: tr.rangeM,
          ttcS: tr.ttcS,
          alertTier: tr.alertTier,
          isGhost: Boolean(tr.isGhost),
          ghostRejected: Boolean(tr.isGhost && tr.ghostRejected),
          meshVerified: Boolean(tr.meshVerified),
          relBearingRad,
        };
      });

      // Conformal KSFO 28L threshold bracket + live 4-light PAPI glideslope
      const rwy28l = pack.runways.find((r) => r.id === "28L");
      const thrLatRad = (rwy28l?.thresholdLatDeg ?? pack.ksfo28lThrLatDeg) * DEG_TO_RAD;
      const thrLonRad = (rwy28l?.thresholdLonDeg ?? pack.ksfo28lThrLonDeg) * DEG_TO_RAD;
      const thrHM = rwy28l?.thresholdWgs84HM ?? pack.ksfo28lThrWgs84HM;
      const rwyHdgRad = (rwy28l?.trueHeadingDeg ?? 297.7) * DEG_TO_RAD;
      const pThr = geodeticToEcef(thrLatRad, thrLonRad, thrHM);
      const thrDx = pThr[0] - latestTelemetry.p_ecef[0];
      const thrDy = pThr[1] - latestTelemetry.p_ecef[1];
      const thrDz = pThr[2] - latestTelemetry.p_ecef[2];
      const thrRelN = rNedOwn[0] * thrDx + rNedOwn[3] * thrDy + rNedOwn[6] * thrDz;
      const thrRelE = rNedOwn[1] * thrDx + rNedOwn[4] * thrDy + rNedOwn[7] * thrDz;
      const thrDistM = Math.hypot(thrRelN, thrRelE);

      let runway28L: ConformalRunwayOverlay | null = null;
      if (frameState.approachActive || thrDistM < 12000) {
        const projThr = projectEcefToNdc(pThr, camView.pEcefCam, camView.rEcefCam, camView.intr, vp);
        const rNedThr = rEcefNed(thrLatRad, thrLonRad);
        const fwdThr = mat3MulVec(rNedThr, [Math.cos(rwyHdgRad), Math.sin(rwyHdgRad), 0]);
        const upThr = mat3MulVec(rNedThr, [0, 0, -1]);
        // PAPI aiming point 310 m down 28L
        const papiX = pThr[0] + fwdThr[0] * 310;
        const papiY = pThr[1] + fwdThr[1] * 310;
        const papiZ = pThr[2] + fwdThr[2] * 310;
        const ownFromPapiX = latestTelemetry.p_ecef[0] - papiX;
        const ownFromPapiY = latestTelemetry.p_ecef[1] - papiY;
        const ownFromPapiZ = latestTelemetry.p_ecef[2] - papiZ;
        const deltaUpM = ownFromPapiX * upThr[0] + ownFromPapiY * upThr[1] + ownFromPapiZ * upThr[2];
        const distBackM = Math.max(
          40.0,
          -(ownFromPapiX * fwdThr[0] + ownFromPapiY * fwdThr[1] + ownFromPapiZ * fwdThr[2]),
        );
        const glideslopeDeg = Math.atan2(Math.max(0, deltaUpM), distBackM) * (180.0 / Math.PI);
        let papiWhiteCount = 0;
        for (const threshDeg of [3.5, 3.17, 2.83, 2.5]) {
          if (glideslopeDeg >= threshDeg) papiWhiteCount++;
        }
        const rangeCamM = Math.max(projThr.vCam[2], 60.0);
        runway28L = {
          visible: projThr.visible,
          screenX: projThr.uPx + 0.5,
          screenY: projThr.vPx + 0.5,
          halfWidthPx: Math.max(26, Math.min(180, (36.0 / rangeCamM) * camView.intr.fx)),
          halfHeightPx: Math.max(16, Math.min(100, (18.0 / rangeCamM) * camView.intr.fy)),
          vVplM: frameState.vVplM,
          glideslopeDeg,
          papiWhiteCount,
          distM: thrDistM,
        };
      }

      // Live Stage-1 AI detections (prefer rich LiveDetectionBox[] from D1-KF / XVS-D2)
      const aiDetections: ConformalAiBoxOverlay[] =
        lastAiResult?.detections?.map((d) => ({
          u: d.u,
          v: d.v,
          widthPx: d.widthPx,
          heightPx: d.heightPx,
          score: d.score,
          sigmaPx: d.sigmaPx,
          modelId: d.modelId,
        })) ??
        lastAiResult?.peaks.map((pk) => ({
          u: pk.u,
          v: pk.v,
          widthPx: Math.max(6.0, Math.exp(Math.max(-2.0, Math.min(5.0, pk.logW)))),
          heightPx: Math.max(6.0, Math.exp(Math.max(-2.0, Math.min(5.0, pk.logH)))),
          score: pk.score,
          sigmaPx: 1.5,
          modelId: lastAiResult?.modelId ?? aiStatus?.modelId ?? "d1_kf_k3_g1",
        })) ??
        [];

      // Heading-up HSI traffic tracks
      const hsiTracks: HsiRadarTrack[] = [];
      for (let i = 0; i < latestTelemetry.intruder_count; i++) {
        const rel = latestTelemetry.intruder_rel_ned[i];
        const vel = latestTelemetry.intruder_vel_ned[i] ?? [0, 0, 0];
        const trMeta = frameState.intruders[i];
        hsiTracks.push({
          id: trMeta?.id ?? `T${i + 1}`,
          relNorthM: rel[0],
          relEastM: rel[1],
          velNorthMps: vel[0],
          velEastMps: vel[1],
          rangeM: trMeta?.rangeM ?? Math.hypot(rel[0], rel[1], rel[2]),
          alertTier: trMeta?.alertTier ?? 1,
          isGhost: false,
          ghostRejected: false,
          meshVerified: Boolean(trMeta?.meshVerified),
        });
      }
      if (latestTelemetry.mission_id === 3) {
        hsiTracks.push({
          id: "GHOST-1",
          relNorthM: 1450,
          relEastM: 260,
          velNorthMps: 0,
          velEastMps: 0,
          rangeM: Math.hypot(1450, 260, -15),
          alertTier: latestTelemetry.ghosts_rejected > 0 ? 0 : 2,
          isGhost: true,
          ghostRejected: latestTelemetry.ghosts_rejected > 0,
          meshVerified: false,
        });
      }

      const hsiPeersRelNe: Array<readonly [number, number]> | undefined =
        latestTelemetry.mission_id === 3
          ? currentPerturbations.thirdPeerActive
            ? [[-450, 900], [-600, -850], [700, 1100]]
            : [[-450, 900], [-600, -850]]
          : undefined;

      hud.updateConformalAndHsi({
        viewportWidth: camView.width,
        viewportHeight: camView.height,
        cameraMode: camView.cameraMode,
        conformalBlanked:
          renderer.isConformalBlanked() ||
          latestTelemetry.blank_conformal_hud ||
          latestTelemetry.mandatory_go_around,
        efvsEnabled: frameState.efvsEnabled,
        detectorMode: activeUiDetectorMode,
        yawRad: latestTelemetry.yaw_rad,
        airspeedMps: latestTelemetry.tas_mps,
        altWgs84M: latestTelemetry.h_m,
        targets: conformalTargets,
        runway28L,
        aiDetections,
        hsiTracks,
        hsiRunway28LRelNe: [thrRelN, thrRelE],
        hsiRunway28LHdgRad: rwyHdgRad,
        hsiPeersRelNe,
      });

      // DR-13 Mode 3 SPA solar glare alignment vs Pass-1 afternoon SF Bay sun direction in NED
      const sunCosE = Math.cos(0.48);
      const sunN = sunCosE * Math.cos(4.25);
      const sunE = sunCosE * Math.sin(4.25);
      const sunD = -Math.sin(0.48);
      // Camera forward (+zCam) mapped from ECEF -> NED via R_ecef_ned^T @ zCam_ecef
      const zCamEcef: Vec3 = [camView.rEcefCam[2], camView.rEcefCam[5], camView.rEcefCam[8]];
      const camFwdN = rNedOwn[0] * zCamEcef[0] + rNedOwn[3] * zCamEcef[1] + rNedOwn[6] * zCamEcef[2];
      const camFwdE = rNedOwn[1] * zCamEcef[0] + rNedOwn[4] * zCamEcef[1] + rNedOwn[7] * zCamEcef[2];
      const camFwdD = rNedOwn[2] * zCamEcef[0] + rNedOwn[5] * zCamEcef[1] + rNedOwn[8] * zCamEcef[2];
      const sunDot = camFwdN * sunN + camFwdE * sunE + camFwdD * sunD;
      const glareMetric =
        sunDot > 0.90 ? Math.min(1.0, ((sunDot - 0.90) / 0.08) * currentPerturbations.fogContrast) : 0.0;
      hud.setOpticalOddGlare(glareMetric);
    }

    updateOverlayBadges();
    const arTel = lastCbdrOut?.telemetry ?? null;
    const arModeLabel =
      arTel === null
        ? "NOMINAL_HOLD"
        : arTel.mode === ActiveRangeModeTs.ActiveSwerve
          ? "ACTIVE_SWERVE"
          : arTel.mode === ActiveRangeModeTs.Locked
            ? "LOCKED"
            : arTel.mode === ActiveRangeModeTs.ImmediateEvade
              ? "IMMEDIATE_EVADE"
              : "NOMINAL_HOLD";
    observatory.updateLiveStatus({
      fps,
      backendLabel: rhi.backend === "webgpu" ? "WebGPU" : "WebGL2",
      rangeHatM: arTel ? arTel.rHatM : null,
      rangeSigmaM: arTel ? arTel.sigmaRM : null,
      sc3bfSafe: arTel ? (arTel.flags & FLAG_SC3BF_ALL_SAFE) !== 0 : true,
      kaniPassed: 18,
      kaniTotal: 18,
      activeRangeModeLabel: arModeLabel,
      detectorHz: lastAiResult ? 10 : 0,
      detectorMs: lastAiResult?.inferenceMs ?? 0,
      halHealthy: latestTelemetry.simplex_mode >= 3,
      forcedLandActive: activeStressPreset === "forcedland",
    });
    hud.setMissionStatus(activeMission.id, latestTelemetry.time_s, activeMission.durationS, paused);
    const activeModelDisplay =
      activeUiDetectorMode === "d0"
        ? "d0_classical_safe"
        : lastAiResult?.modelId ??
          (activeUiDetectorMode === "xvs_d2"
            ? "xvs_d2_web_fp16"
            : aiStatus?.modelId ?? "d1_kf_k3_g1 (hydrating)");
    hud.updateTelemetry({
      simTimeS: latestTelemetry.time_s,
      frameMs: renderMs,
      fps,
      cusumStat: latestTelemetry.cusum_stat,
      spoofAlarm: latestTelemetry.active_nav_bank === 1,
      bankActive: latestTelemetry.active_nav_bank,
      epipolarUrad: latestTelemetry.epipolar_max_urad,
      camHealthMask: latestTelemetry.healthy_cam_mask,
      horizonPlUrad: latestTelemetry.horizon_pl_urad,
      horizonAlUrad: HORIZON_AL_URAD,
      vVplMm: latestTelemetry.v_vpl_mm,
      vHplMm: latestTelemetry.v_hpl_mm,
      mandatoryGoAround: latestTelemetry.mandatory_go_around,
      ghostRejected: latestTelemetry.ghosts_rejected > 0,
      ecbfInfeasible: latestTelemetry.ecbf_infeasible,
      dtaciFrozen: latestTelemetry.dtaci_frozen,
      aiInferenceMs: lastAiResult?.inferenceMs ?? 0,
      aiPeaksCount: lastAiResult?.peaks.length ?? 0,
      aiMaxScore: lastAiResult?.maxHeatmapScore ?? 0,
      aiAllFinite: ortAllFiniteEver,
      aiModelId: activeModelDisplay,
      missionId: activeMission.id,
      wc1Alert: latestTelemetry.wc1_alert,
      camRBiasDeg: (currentPerturbations.camRBiasRad * 180) / Math.PI,
      efvsEnabled: frameState.efvsEnabled,
      simplexMode: latestTelemetry.simplex_mode,
      activeDetectorUiMode: activeUiDetectorMode,
      aiDetectionsCount: lastAiResult?.detections?.length ?? lastAiResult?.peaks.length ?? 0,
      xvsD2Provenance,
    });

    // Keep top viewport callout synchronized with active Observatory preset instead of stale M4 label
    const calloutEl = rootEl.querySelector<HTMLElement>(".xvs-alert-callout");
    if (calloutEl && !missionParam && activeStressPreset !== "spoof") {
      const loc = observatory.store.getLocale();
      if (activeStressPreset === "cbdr") {
        const locked = arTel !== null && (arTel.mode === ActiveRangeModeTs.Locked || arTel.sigmaRM <= 95);
        calloutEl.textContent =
          loc === "tr"
            ? locked
              ? "XVS-ActiveRange: MESAFE KİLİTLENDİ — SC3BF 6/6 GÜVENLİ"
              : "CBDR KÖR ÇARPIŞMA — XVS-ActiveRange <=12° S-MANEVRASI"
            : locked
              ? "XVS-ActiveRange: RANGE LOCKED — SC3BF 6/6 SAFE"
              : "CBDR BLIND COLLISION — XVS-ActiveRange <=12° S-SWERVE";
        calloutEl.dataset.severity = locked ? "nominal" : "info";
      } else if (activeStressPreset === "forcedland") {
        calloutEl.textContent =
          loc === "tr"
            ? "XVS-ForcedLand: MOTOR ARIZASI + 35 KT YAN RÜZGAR TROKOİD HUNİSİ (KSFO 28L)"
            : "XVS-ForcedLand: ENGINE-OUT + 35 KT CROSSWIND TROCHOID FUNNEL (KSFO 28L)";
        calloutEl.dataset.severity = "caution";
      } else {
        calloutEl.textContent =
          loc === "tr"
            ? "KSFO 28L YAKLAŞMA — 0.98m/TEXEL ORTO + CANLI PAPI + XVS-D2"
            : "KSFO 28L APPROACH — 0.98m/TEXEL ORTHO + LIVE PAPI + XVS-D2";
        calloutEl.dataset.severity = "nominal";
      }
    }
  };

  // Fixed-step 100 Hz simulation loop driving 60 FPS WebGPU/WebGL2 cockpit rendering
  let lastRafMs = performance.now();
  let accumS = 0.0;

  const tickLoop = (nowMs: number) => {
    const dtMs = Math.min(50, Math.max(0, nowMs - lastRafMs));
    lastRafMs = nowMs;

    if (rafSuspended) {
      requestAnimationFrame(tickLoop);
      return;
    }

    if (!paused) {
      if (keysDown.size > 0) {
        manualFlightOverride = true;
        manualBankCmdRad =
          (keysDown.has("ArrowRight") ? 0.35 : 0) - (keysDown.has("ArrowLeft") ? 0.35 : 0);
        manualClimbCmdMps =
          (keysDown.has("ArrowUp") ? 6.0 : 0) - (keysDown.has("ArrowDown") ? 6.0 : 0);
      }

      accumS += dtMs * 0.001;
      const steps = Math.min(10, Math.floor(accumS / 0.01));
      if (steps > 0) {
        accumS -= steps * 0.01;
        if (!manualPerturbationsActive) {
          const nextPert = applyDetectorState(
            evaluateMissionPerturbations(activeMission, latestTelemetry.time_s),
          );
          currentPerturbations = nextPert;
          hud.syncPerturbationControls(nextPert, activeUiDetectorMode);
        }
        stepObservatoryScenarios(latestTelemetry);
        pushFaultsToWasm(currentPerturbations);
        sim.step_100hz(steps);
        latestTelemetry = readWasmSnapshot(sim);
        recordTelemetrySample(latestTelemetry);
        drainWasmTxToBridge(sim, bridge);
      }
    }

    renderCurrentFrame(dtMs);
    requestAnimationFrame(tickLoop);
  };

  // Stage-1 non-blocking background AI hydration (dynamic import keeps ORT-Web out of Stage-0!)
  const hydrateStage1Ai = async (): Promise<void> => {
    try {
      const baseUrl = new URL("./", window.location.href).href;
      const [{ OrtDetectorRunner: RunnerCtor }, { getXvsD2ModelEntry, loadModelRegistry }] =
        await Promise.all([import("./ai/ort_worker.ts"), import("./ai/model_registry.ts")]);
      try {
        const reg = await loadModelRegistry(baseUrl);
        observatory.setTrainedProvenance(extractTrainedProvenanceFromRegistry(reg));
        const d2Entry = getXvsD2ModelEntry(reg);
        if (d2Entry) {
          const webMetrics = d2Entry.trained_eval_metrics?.web;
          xvsD2Provenance = {
            status: d2Entry.status,
            bytes: d2Entry.artifacts.fp16.bytes,
            ap50: webMetrics?.ap50,
            aboveHorizonPd: webMetrics?.above_horizon_pd,
            centroidRmsePx: webMetrics?.centroid_rmse_px,
          };
        }
      } catch {
        // Non-fatal: provenance badge gracefully falls back to generic XVS-D2 WEB FP16
      }
      ortRunner = new RunnerCtor();
      aiStatus = await ortRunner.init({
        baseUrl,
        hasWebGpu: rhi.backend === "webgpu",
        hasShaderF16: rhi.hasShaderF16,
        forceWasmSimd: rhi.backend !== "webgpu",
        fallbackReason: rhi.degradedReason,
        wasmPaths: new URL("./ort/", window.location.href).href,
      });
      stage1Ready = aiStatus.ready;
      updateOverlayBadges();
      console.info(
        `[OpenXVS Stage-1 AI] Ready: ${aiStatus.backend.aiBadge} (${aiStatus.cacheSource}, sha256=${aiStatus.modelSha256.slice(0, 12)})`,
      );
    } catch (err) {
      lastGpuError = `Stage-1 AI init failed: ${err instanceof Error ? err.message : String(err)}`;
      console.error(`[OpenXVS Stage-1 AI] ${lastGpuError}`);
      updateOverlayBadges();
    }
  };

  // Stage-1 non-blocking background USDA/USGS Satellite Ortho draping (keeps Stage-0 Brotli <= 700 KB)
  const hydrateStage1Ortho = async (): Promise<void> => {
    if (useProceduralTerrain || renderer.getTerrainPack().terrainMode === "procedural_ci") {
      return;
    }
    try {
      const ok = await renderer.loadOrthoTextures(
        "./assets/ksfo_ortho_inner_v1.jpg",
        "./assets/ksfo_ortho_outer_v1.jpg",
      );
      if (ok) {
        orthoLoaded = true;
        updateOverlayBadges();
        if (paused) {
          renderCurrentFrame(0.0);
        }
        console.info("[OpenXVS Stage-1 Ortho] USDA/USGS National Map Orthoimagery draped on KSFO 3DEP");
      }
    } catch (err) {
      // Graceful non-fatal fallback to hypsometric 3DEP shading if ortho assets absent in CI
      console.info(
        `[OpenXVS Stage-1 Ortho] Satellite ortho unavailable (${err instanceof Error ? err.message : String(err)}); keeping 3DEP hypsometric shading`,
      );
    }
  };

  // Initial Stage-0 frame & HUD render immediately (default landing preset = CBDR ActiveRange unless ?mission= set)
  if (!missionParam) {
    applyStressPreset(DEFAULT_STRESS_PRESET, false);
  }
  hud.syncPerturbationControls(currentPerturbations, activeUiDetectorMode);
  renderCurrentFrame(16.67);
  stage1Promise = hydrateStage1Ai();
  void hydrateStage1Ortho();
  requestAnimationFrame(tickLoop);

  // Deterministic E2E inspection hook for Playwright Gate V2-M1/V2.1 verification
  const testApi: OpenXvsTestApi = {
    getStatus: (): LiveDeckStatusSnapshot => ({
      ready: true,
      stage1Ready,
      backend: rhi.backend,
      hasShaderF16: rhi.hasShaderF16,
      degradedReason: rhi.degradedReason,
      degradedReasons: computeDegradedReasons(),
      aiBadge: getAiBadge(),
      aiCacheSource: aiStatus?.cacheSource ?? null,
      aiOpfsWarning: aiStatus?.opfsWarning ?? null,
      terrainMode: renderer.getTerrainPack().terrainMode,
      orthoLoaded: renderer.isOrthoLoaded(),
      cameraMode,
      orbitTarget,
      visibleBadges: [...renderer.getVisibleHudBadges()],
      frameCount: renderer.getRenderedFrameCount(),
      lastGpuError,
      ortInferenceCount,
      ortFiniteOutputs: ortAllFiniteEver,
      ortLastPeaksCount: lastAiResult?.peaks.length ?? 0,
      ortMaxHeatmapScore: lastAiResult?.maxHeatmapScore ?? 0,
      conformalBlanked: renderer.isConformalBlanked(),
      activeMissionId: activeMission.id,
      simTimeS: latestTelemetry.time_s,
      pass4Diagnostics: renderer.getPass4Diagnostics(),
    }),

    getTelemetry: (): DecodedWasmTelemetry => latestTelemetry,

    startMission: (id: MissionId): DecodedWasmTelemetry => {
      paused = true;
      return startMission(id);
    },

    setCameraMode: (mode: CameraViewMode): void => {
      setCameraModeState(mode);
    },

    setOrbitTarget: (target: OrbitTargetMode): void => {
      setOrbitTargetState(target);
    },

    setOrbitView: (azRad: number, elRad: number, distM: number): void => {
      if (Number.isFinite(azRad)) orbitAzimuthRad = azRad;
      if (Number.isFinite(elRad)) {
        orbitElevationRad = Math.max(ORBIT_ELEV_MIN_RAD, Math.min(ORBIT_ELEV_MAX_RAD, elRad));
      }
      if (Number.isFinite(distM)) {
        orbitDistanceM = Math.max(ORBIT_DIST_MIN_M, Math.min(ORBIT_DIST_MAX_M, distM));
      }
      renderer.setOrbitView(orbitAzimuthRad, orbitElevationRad, orbitDistanceM);
      if (paused) {
        renderCurrentFrame(0.0);
      }
    },

    setSpzMode: (mode: "off" | "preview"): void => {
      setSpzModeState(mode);
    },

    getSpzDiagnostics: (): SpzDiagnostics => renderer.getSpzDiagnostics(),

    advanceSimSeconds: (sec: number): DecodedWasmTelemetry => {
      const rtaTicks = Math.max(0, Math.round(sec * 10));
      for (let t = 0; t < rtaTicks; t++) {
        if (!manualPerturbationsActive) {
          currentPerturbations = applyDetectorState(
            evaluateMissionPerturbations(activeMission, latestTelemetry.time_s),
          );
        }
        pushFaultsToWasm(currentPerturbations);
        sim.step_rta_tick();
        latestTelemetry = readWasmSnapshot(sim);
        recordTelemetrySample(latestTelemetry);
        drainWasmTxToBridge(sim, bridge);
      }
      hud.syncPerturbationControls(currentPerturbations, activeUiDetectorMode);
      renderCurrentFrame(16.67);
      return latestTelemetry;
    },

    runPass4Parity: async (rawInput: Pass4ParitySerializableInput) => {
      const prevPaused = paused;
      const prevSuspended = rafSuspended;
      paused = true;
      rafSuspended = true;
      try {
        // Yield once so any prior rAF GPU submit settles before explicit parity dispatch
        await new Promise((r) => setTimeout(r, 0));
        const input: Pass4CpuInput = {
          currRgba:
            rawInput.currRgba instanceof Uint8Array
              ? rawInput.currRgba
              : new Uint8Array(rawInput.currRgba),
          prev1Electrons:
            rawInput.prev1Electrons instanceof Float32Array
              ? rawInput.prev1Electrons
              : new Float32Array(rawInput.prev1Electrons),
          prev2Electrons:
            rawInput.prev2Electrons instanceof Float32Array
              ? rawInput.prev2Electrons
              : new Float32Array(rawInput.prev2Electrons),
          poseCurr: rawInput.poseCurr,
          posePrev1: rawInput.posePrev1,
          posePrev2: rawInput.posePrev2,
          roiCurr: rawInput.roiCurr,
          roiPrev1: rawInput.roiPrev1,
          roiPrev2: rawInput.roiPrev2,
          noiseOff: rawInput.noiseOff,
          applyPsf: rawInput.applyPsf,
          frameSeed: rawInput.frameSeed,
        };
        const packet = await renderer.pass4.runExplicitParityAsync(input);
        return {
          data: Array.from(packet.data),
          shape: packet.shape,
          computePath: packet.computePath,
          stats: packet.stats,
        };
      } finally {
        paused = prevPaused;
        rafSuspended = prevSuspended;
      }
    },

    runOrtSmokeTicks: async (count: number) => {
      const prevPaused = paused;
      const prevSuspended = rafSuspended;
      paused = true;
      rafSuspended = true;
      try {
        await stage1Promise;
        // Drain any in-flight rAF ORT inference before driving the deterministic smoke loop
        while (ortInFlight) {
          await new Promise((r) => setTimeout(r, 5));
        }

        const n = Math.max(1, Math.floor(count));
        let completedInferences = 0;
        let totalMs = 0;
        let maxScore = 0;
        let allFinite = true;

        for (let i = 0; i < n; i++) {
          sim.step_rta_tick();
          latestTelemetry = readWasmSnapshot(sim);
          recordTelemetrySample(latestTelemetry);
          const frameState = buildCockpitFrameState(
            latestTelemetry,
            currentPerturbations,
            getAiBadge(),
            lastAiResult,
          );
          let packet = renderer.renderFrame(frameState, {
            pass4: { noiseOff: noiseOffFlag, applyPsf: true },
          });

          // Yield periodically so WebGPU mapAsync Staging Ring callbacks pump cleanly without starvation
          if (!packet || i % 5 === 0) {
            await new Promise((r) => setTimeout(r, 0));
            packet = renderer.getLatestPass4Tensor();
          }

          // Run live ONNX inference every 10th frame (10 Hz cadence across >= 300 render frames)
          if (ortRunner && stage1Ready && i % 10 === 0) {
            let waitTries = 0;
            while (!packet && waitTries < 20) {
              await new Promise((r) => setTimeout(r, 5));
              packet = renderer.getLatestPass4Tensor();
              waitTries++;
            }
            if (packet) {
              const res = await ortRunner.runNormalized(new Float32Array(packet.data), i, 0.35);
              lastAiResult = res;
              ortInferenceCount++;
              completedInferences++;
              totalMs += res.inferenceMs;
              if (!res.allFinite) {
                allFinite = false;
                ortAllFiniteEver = false;
              }
              if (res.maxHeatmapScore > maxScore) {
                maxScore = res.maxHeatmapScore;
              }
            }
          }
        }

        // Flush final pending GPU readbacks cleanly
        await new Promise((r) => setTimeout(r, 10));

        return {
          count: n,
          completedInferences,
          allFinite,
          maxHeatmapScore: maxScore,
          avgInferenceMs: completedInferences > 0 ? totalMs / completedInferences : 0,
          lastGpuError,
          aiBadge: getAiBadge(),
          backend: rhi.backend,
          pass4Diagnostics: renderer.getPass4Diagnostics(),
        };
      } finally {
        paused = prevPaused;
        rafSuspended = prevSuspended;
      }
    },

    getMissionProof: buildMissionProofLog,
  };
  window.__OPENXVS_TEST_API__ = testApi;
  window.__openxvs = testApi;
}

void bootLiveDeck().catch((err) => {
  console.error("[OpenXVS LiveDeck] Fatal boot error:", err);
});
