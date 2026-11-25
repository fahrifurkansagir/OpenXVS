/**
 * Unified 5-Pass OpenXVS Live Deck v2.1 Cockpit & 3D External Camera Renderer.
 * Orchestrates Passes 1–5 deterministically over either WebGPU or explicit Naga WebGL2 RHI,
 * with Stage-1 USGS/USDA satellite orthophoto draping and 3-Mode 3D Camera Suite
 * (`cockpit` | `chase` | `orbit`).
 */

import {
  DEG_TO_RAD,
  buildCamViewProjMatrix,
  buildFreeLookRBodyCam,
  buildPinholeIntrinsics,
  geodeticToEcef,
  mat3Mul,
  mat3MulVec,
  mat3Transpose,
  quatToRotMat,
  rEcefNed,
  type Mat3RowMajor,
  type Vec3,
} from "../rhi/coords.ts";
import type { RhiContext } from "../rhi/types.ts";
import { Pass1DasEfvs } from "./pass1_das_efvs.ts";
import { Pass2TerrainCdlod, createProceduralCiTerrainPack, parseKsfoTerrainPack } from "./pass2_terrain_cdlod.ts";
import {
  Pass2bSpzWorld,
  type SpzDiagnostics,
  type SpzMode,
} from "./pass2b_spz_world.ts";
import { Pass3AirframeTraffic } from "./pass3_airframe_traffic.ts";
import { Pass4AiRoi, type Pass4RingDiagnostics } from "./pass4_ai_roi.ts";
import { Pass5SdfMsdfHud } from "./pass5_sdf_msdf_hud.ts";
import type {
  ActiveCameraView,
  CameraViewMode,
  CockpitFrameState,
  CockpitRenderOptions,
  MsdfAtlasMetadata,
  OrbitTargetMode,
  Pass4TensorPacket,
  TerrainPackData,
} from "./types.ts";

export * from "./types.ts";
export { Pass1DasEfvs } from "./pass1_das_efvs.ts";
export {
  Pass2TerrainCdlod,
  parseKsfoTerrainPack,
  createProceduralCiTerrainPack,
} from "./pass2_terrain_cdlod.ts";
export {
  Pass2bSpzWorld,
  SPZ_MAGIC_U32,
  SPZ_VERSION,
  SPZ_HEADER_BYTES,
  SPZ_SH_C0,
  SPZ_MAX_CHROMA_SPREAD,
  MAX_RENDER_SPLATS,
  clampSpzChromaRgb,
  parseSpzV2Raw,
  parseSpzV2Gzip,
  buildKsfo28lPreviewSpzCloud,
  type SpzMode,
  type SpzHeader,
  type SpzSplatCloud,
  type SpzDiagnostics,
} from "./pass2b_spz_world.ts";
export {
  Pass3AirframeTraffic,
  MAX_MANIFOLD_VERTICES,
  computePapiLightStates,
  buildActiveRangeConeGeometry,
  buildSc3bfBarrierTubeGeometry,
  buildForcedLandFunnelGeometry,
  type ActiveRangeOverlayState,
  type ForcedLandFunnelWaypoint,
  type ForcedLandFunnelState,
  type ManifoldToggleState,
  type PapiLightState,
  type ManifoldGeometryOutput,
  type ActiveRangeConeGeometry,
  type ActiveRangeConeParams,
  type Sc3bfBarrierTubeParams,
  type ForcedLandFunnelParams,
} from "./pass3_airframe_traffic.ts";
export {
  Pass4AiRoi,
  PASS4_TILE_W,
  PASS4_TILE_H,
  PASS4_TILE_N,
  PASS4_CHANNELS,
  PASS4_TENSOR_ELEMENTS,
  PASS4_SENSOR_DEFAULTS,
  srgbChannelToLinear,
  srgbRgbaToElectronsCpu,
  manualBilinearSampleF32,
  warpElectrons6DofWgs84Cpu,
  normalizeD1StackCpu,
  computePass4NormalizedCpu,
  type Pass4CameraPose,
  type Pass4CpuInput,
  type Pass4RingDiagnostics,
} from "./pass4_ai_roi.ts";
export { Pass5SdfMsdfHud } from "./pass5_sdf_msdf_hud.ts";

export const ORBIT_ELEV_MIN_RAD = -0.15;
export const ORBIT_ELEV_MAX_RAD = 1.35;
export const ORBIT_DIST_MIN_M = 45.0;
export const ORBIT_DIST_MAX_M = 2200.0;
export const DEFAULT_ORBIT_AZIMUTH_RAD = 0.52;
export const DEFAULT_ORBIT_ELEVATION_RAD = 0.30;
export const DEFAULT_ORBIT_DISTANCE_M = 160.0;
export const KSFO_28L_TOWER_LAT_DEG = 37.6148;
export const KSFO_28L_TOWER_LON_DEG = -122.3612;
export const KSFO_28L_TOWER_HEIGHT_AGL_M = 46.0;
export const KSFO_28L_DEFAULT_THR_WGS84_H_M = -28.67;

function normalize3(v: Vec3, fallback: Vec3): Vec3 {
  const len = Math.hypot(v[0], v[1], v[2]);
  if (!Number.isFinite(len) || len < 1e-8) return fallback;
  return [v[0] / len, v[1] / len, v[2] / len];
}

function cross3(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

/**
 * Construct exact OpenCV CAM (RDF: +x right, +y down, +z forward) rotation `R_ecef_cam`
 * with `R^T R = I` and `det(R) = +1` pointing from `pCamEcef` directly at `pTargetEcef`.
 */
export function buildLookAtREcefCam(
  pCamEcef: Vec3,
  pTargetEcef: Vec3,
  upRefEcef: Vec3,
  fallbackRefEcef: Vec3 = [1, 0, 0],
): Mat3RowMajor {
  const zCam = normalize3(
    [
      pTargetEcef[0] - pCamEcef[0],
      pTargetEcef[1] - pCamEcef[1],
      pTargetEcef[2] - pCamEcef[2],
    ],
    [0, 0, 1],
  );
  let xRaw = cross3(zCam, upRefEcef);
  if (Math.hypot(xRaw[0], xRaw[1], xRaw[2]) < 1e-5) {
    xRaw = cross3(zCam, fallbackRefEcef);
  }
  const xCam = normalize3(xRaw, [1, 0, 0]);
  // Right-handed OpenCV CAM: x=Right, y=Down=Forward x Right, z=Forward => det([x,y,z]) = +1
  const yCam = normalize3(cross3(zCam, xCam), [0, 1, 0]);
  return [
    xCam[0], yCam[0], zCam[0],
    xCam[1], yCam[1], zCam[1],
    xCam[2], yCam[2], zCam[2],
  ];
}

/** Local pan (right + around +yCam) & tilt (up + around +xCam) look offset in OpenCV CAM frame. */
function buildCamPanTiltOffset(panRad: number, tiltRad: number): Mat3RowMajor {
  const cp = Math.cos(panRad);
  const sp = Math.sin(panRad);
  const ct = Math.cos(tiltRad);
  const st = Math.sin(tiltRad);
  // Ry(pan) @ Rx(tilt) in OpenCV RDF: pan right swings +z toward +x; tilt up swings +z toward -y
  const ry: Mat3RowMajor = [
    cp, 0, sp,
    0, 1, 0,
    -sp, 0, cp,
  ];
  const rx: Mat3RowMajor = [
    1, 0, 0,
    0, ct, -st,
    0, st, ct,
  ];
  return mat3Mul(ry, rx);
}

export interface CameraViewOverrides {
  readonly cameraMode?: CameraViewMode;
  readonly orbitTarget?: OrbitTargetMode;
  readonly orbitAzimuthRad?: number;
  readonly orbitElevationRad?: number;
  readonly orbitDistanceM?: number;
  readonly ksfo28lThrWgs84HM?: number;
}

export function buildActiveCameraView(
  state: CockpitFrameState,
  width: number,
  height: number,
  overridePanRad?: number,
  overrideTiltRad?: number,
  cameraOverrides?: CameraViewOverrides,
): ActiveCameraView {
  const pEcefBody: Vec3 =
    state.pEcefBody ?? geodeticToEcef(state.latRad, state.lonRad, state.altWgs84M);
  const rNed = rEcefNed(state.latRad, state.lonRad);
  const rNedBody = quatToRotMat(state.qNedBody);
  const rEcefBody = mat3Mul(rNed, rNedBody);

  const cameraMode: CameraViewMode = cameraOverrides?.cameraMode ?? state.cameraMode ?? "cockpit";
  const orbitTarget: OrbitTargetMode = cameraOverrides?.orbitTarget ?? state.orbitTarget ?? "ownship";
  const panRad = overridePanRad ?? state.panRad ?? 0.0;
  const tiltRad = overrideTiltRad ?? state.tiltRad ?? 0.0;

  const localNorthEcef: Vec3 = [rNed[0], rNed[3], rNed[6]];
  const localUpEcef: Vec3 = [-rNed[2], -rNed[5], -rNed[8]];

  let pEcefCam: Vec3;
  let rEcefCam: Mat3RowMajor;
  let rBodyCam: Mat3RowMajor;
  let defaultFovHDeg = 88.0;

  if (cameraMode === "chase") {
    // Trail smoothly behind & above ownship in BODY FRD [-68, 0, -20] m, looking +40 m ahead of nose
    // with level local-up horizon so pilot sees the 3D ownship jet banking visibly over KSFO
    const chaseOffsetEcef = mat3MulVec(rEcefBody, [-68.0, 0.0, -20.0]);
    pEcefCam = [
      pEcefBody[0] + chaseOffsetEcef[0],
      pEcefBody[1] + chaseOffsetEcef[1],
      pEcefBody[2] + chaseOffsetEcef[2],
    ];
    const lookAheadEcef = mat3MulVec(rEcefBody, [40.0, 0.0, 0.0]);
    const pTarget: Vec3 = [
      pEcefBody[0] + lookAheadEcef[0],
      pEcefBody[1] + lookAheadEcef[1],
      pEcefBody[2] + lookAheadEcef[2],
    ];
    const rBase = buildLookAtREcefCam(pEcefCam, pTarget, localUpEcef, localNorthEcef);
    rEcefCam =
      Math.abs(panRad) > 1e-6 || Math.abs(tiltRad) > 1e-6
        ? mat3Mul(rBase, buildCamPanTiltOffset(panRad, tiltRad))
        : rBase;
    rBodyCam = mat3Mul(mat3Transpose(rEcefBody), rEcefCam);
    defaultFovHDeg = 74.0;
  } else if (cameraMode === "orbit" && orbitTarget === "tower") {
    // Fixed KSFO 28L Control Tower / Runway-Side Spotter 3D Camera tracking ownship automatically
    const towerLatRad = KSFO_28L_TOWER_LAT_DEG * DEG_TO_RAD;
    const towerLonRad = KSFO_28L_TOWER_LON_DEG * DEG_TO_RAD;
    const thrH =
      typeof cameraOverrides?.ksfo28lThrWgs84HM === "number" &&
      Number.isFinite(cameraOverrides.ksfo28lThrWgs84HM)
        ? cameraOverrides.ksfo28lThrWgs84HM
        : KSFO_28L_DEFAULT_THR_WGS84_H_M;
    pEcefCam = geodeticToEcef(towerLatRad, towerLonRad, thrH + KSFO_28L_TOWER_HEIGHT_AGL_M);
    const rNedTower = rEcefNed(towerLatRad, towerLonRad);
    const towerNorthEcef: Vec3 = [rNedTower[0], rNedTower[3], rNedTower[6]];
    const towerUpEcef: Vec3 = [-rNedTower[2], -rNedTower[5], -rNedTower[8]];
    const rBase = buildLookAtREcefCam(pEcefCam, pEcefBody, towerUpEcef, towerNorthEcef);
    rEcefCam =
      Math.abs(panRad) > 1e-6 || Math.abs(tiltRad) > 1e-6
        ? mat3Mul(rBase, buildCamPanTiltOffset(panRad, tiltRad))
        : rBase;
    rBodyCam = mat3Mul(mat3Transpose(rEcefBody), rEcefCam);
    defaultFovHDeg = 54.0;
  } else if (cameraMode === "orbit") {
    // Full 360-deg spherical 3D orbit around ownship or active non-ghost intruder in local NED frame
    let pFocus: Vec3 = pEcefBody;
    if (orbitTarget === "intruder") {
      const targetTrack =
        state.intruders.find((t) => !t.isGhost && !t.ghostRejected) ??
        state.intruders.find((t) => !t.ghostRejected) ??
        state.intruders[0];
      if (targetTrack && targetTrack.posEcef.every(Number.isFinite)) {
        pFocus = targetTrack.posEcef;
      }
    }
    const azRel = cameraOverrides?.orbitAzimuthRad ?? state.orbitAzimuthRad ?? DEFAULT_ORBIT_AZIMUTH_RAD;
    const rawEl = cameraOverrides?.orbitElevationRad ?? state.orbitElevationRad ?? DEFAULT_ORBIT_ELEVATION_RAD;
    const rawDist = cameraOverrides?.orbitDistanceM ?? state.orbitDistanceM ?? DEFAULT_ORBIT_DISTANCE_M;
    const el = Math.max(ORBIT_ELEV_MIN_RAD, Math.min(ORBIT_ELEV_MAX_RAD, Number.isFinite(rawEl) ? rawEl : DEFAULT_ORBIT_ELEVATION_RAD));
    const dist = Math.max(ORBIT_DIST_MIN_M, Math.min(ORBIT_DIST_MAX_M, Number.isFinite(rawDist) ? rawDist : DEFAULT_ORBIT_DISTANCE_M));
    const worldAz = state.yawRad + Math.PI + (Number.isFinite(azRel) ? azRel : DEFAULT_ORBIT_AZIMUTH_RAD);
    const cosEl = Math.cos(el);
    const offsetNed: Vec3 = [
      dist * cosEl * Math.cos(worldAz),
      dist * cosEl * Math.sin(worldAz),
      -dist * Math.sin(el),
    ];
    const offsetEcef = mat3MulVec(rNed, offsetNed);
    pEcefCam = [
      pFocus[0] + offsetEcef[0],
      pFocus[1] + offsetEcef[1],
      pFocus[2] + offsetEcef[2],
    ];
    rEcefCam = buildLookAtREcefCam(pEcefCam, pFocus, localUpEcef, localNorthEcef);
    rBodyCam = mat3Mul(mat3Transpose(rEcefBody), rEcefCam);
    defaultFovHDeg = 64.0;
  } else {
    // Pilot cockpit eye lever arm in BODY FRD (metres forward of ARP) — identical to v2.0 conformal lock
    const eyeLeverEcef = mat3MulVec(rEcefBody, [11.5, 0.0, -0.8]);
    pEcefCam = [
      pEcefBody[0] + eyeLeverEcef[0],
      pEcefBody[1] + eyeLeverEcef[1],
      pEcefBody[2] + eyeLeverEcef[2],
    ];
    rBodyCam = buildFreeLookRBodyCam(panRad, tiltRad);
    rEcefCam = mat3Mul(rEcefBody, rBodyCam);
    defaultFovHDeg = 88.0;
  }

  const intr = buildPinholeIntrinsics(width, height, state.fovHDeg ?? defaultFovHDeg);
  const viewProj = buildCamViewProjMatrix(rEcefCam, intr, {
    width,
    height,
    zNear: 1.0,
    zFar: 200000.0,
  });

  return {
    cameraMode,
    orbitTarget,
    pEcefBody,
    pEcefCam,
    rEcefNed: rNed,
    rNedBody,
    rEcefBody,
    rBodyCam,
    rEcefCam,
    intr,
    width,
    height,
    viewProj,
  };
}

export class CockpitRenderer {
  readonly rhi: RhiContext;
  readonly pass1: Pass1DasEfvs;
  readonly pass2: Pass2TerrainCdlod;
  readonly pass2b: Pass2bSpzWorld;
  readonly pass3: Pass3AirframeTraffic;
  readonly pass4: Pass4AiRoi;
  readonly pass5: Pass5SdfMsdfHud;

  private cameraMode: CameraViewMode = "cockpit";
  private orbitTarget: OrbitTargetMode = "ownship";
  private panRad = 0.0;
  private tiltRad = 0.0;
  private orbitAzimuthRad = DEFAULT_ORBIT_AZIMUTH_RAD;
  private orbitElevationRad = DEFAULT_ORBIT_ELEVATION_RAD;
  private orbitDistanceM = DEFAULT_ORBIT_DISTANCE_M;
  private lastView: ActiveCameraView | null = null;
  private renderedFrames = 0;

  constructor(rhi: RhiContext, initialTerrainPack?: TerrainPackData) {
    this.rhi = rhi;
    this.pass1 = new Pass1DasEfvs(rhi);
    this.pass2 = new Pass2TerrainCdlod(rhi, initialTerrainPack);
    this.pass2b = new Pass2bSpzWorld(rhi);
    this.pass3 = new Pass3AirframeTraffic(rhi);
    this.pass4 = new Pass4AiRoi(rhi);
    this.pass5 = new Pass5SdfMsdfHud(rhi);
  }

  /** Load binary `ksfo_3dep_geoid18_v1.bin` from URL or switch to explicit `procedural-ci`. */
  async loadTerrainPackFromUrl(url = "./assets/ksfo_3dep_geoid18_v1.bin"): Promise<TerrainPackData> {
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`Failed to load KSFO terrain pack ${url} (HTTP ${resp.status})`);
    }
    const buf = await resp.arrayBuffer();
    const pack = parseKsfoTerrainPack(buf);
    this.pass2.setTerrainPack(pack);
    return pack;
  }

  setTerrainPack(pack: TerrainPackData): void {
    this.pass2.setTerrainPack(pack);
  }

  useProceduralCiTerrain(): TerrainPackData {
    const pack = createProceduralCiTerrainPack();
    this.pass2.setTerrainPack(pack);
    return pack;
  }

  getTerrainPack(): TerrainPackData {
    return this.pass2.getTerrainPack();
  }

  async loadOrthoTextures(
    innerUrl = "./assets/ksfo_ortho_inner_v1.jpg",
    outerUrl = "./assets/ksfo_ortho_outer_v1.jpg",
    coreUrl = "./assets/ksfo_ortho_core2k_v2.jpg",
  ): Promise<boolean> {
    return this.pass2.loadOrthoTexturesFromUrls(innerUrl, outerUrl, coreUrl);
  }

  async loadOrthoCoreTexture(
    coreUrl = "./assets/ksfo_ortho_core2k_v2.jpg",
  ): Promise<boolean> {
    return this.pass2.loadOrthoCoreTextureFromUrl(coreUrl);
  }

  setOrthoTextures(
    inner: { rgba: Uint8Array; width: number; height: number },
    outer: { rgba: Uint8Array; width: number; height: number },
  ): void {
    this.pass2.setOrthoTextures(inner, outer);
  }

  setOrthoCoreTexture(core: { rgba: Uint8Array; width: number; height: number }): void {
    this.pass2.setOrthoCoreTexture(core);
  }

  isOrthoLoaded(): boolean {
    return this.pass2.isOrthoLoaded();
  }

  isOrthoCoreLoaded(): boolean {
    return this.pass2.isOrthoCoreLoaded();
  }

  async loadMsdfAtlas(
    pngUrl = "./assets/b612_mono_msdf.png",
    jsonUrl = "./assets/b612_mono_msdf.json",
  ): Promise<void> {
    await this.pass5.loadMsdfAtlasFromUrls(pngUrl, jsonUrl);
  }

  setMsdfAtlas(rgba: Uint8Array, width: number, height: number, meta: MsdfAtlasMetadata): void {
    this.pass5.setMsdfAtlas(rgba, width, height, meta);
  }

  setCameraMode(mode: CameraViewMode): void {
    this.cameraMode = mode;
  }

  getCameraMode(): CameraViewMode {
    return this.cameraMode;
  }

  setOrbitTarget(target: OrbitTargetMode): void {
    this.orbitTarget = target;
  }

  getOrbitTarget(): OrbitTargetMode {
    return this.orbitTarget;
  }

  setViewAngles(panRad: number, tiltRad: number): void {
    this.panRad = panRad;
    this.tiltRad = Math.max(-Math.PI * 0.45, Math.min(Math.PI * 0.45, tiltRad));
  }

  setOrbitView(azimuthRad: number, elevationRad: number, distanceM: number): void {
    if (Number.isFinite(azimuthRad)) this.orbitAzimuthRad = azimuthRad;
    if (Number.isFinite(elevationRad)) {
      this.orbitElevationRad = Math.max(ORBIT_ELEV_MIN_RAD, Math.min(ORBIT_ELEV_MAX_RAD, elevationRad));
    }
    if (Number.isFinite(distanceM)) {
      this.orbitDistanceM = Math.max(ORBIT_DIST_MIN_M, Math.min(ORBIT_DIST_MAX_M, distanceM));
    }
  }

  getOrbitView(): { azimuthRad: number; elevationRad: number; distanceM: number } {
    return {
      azimuthRad: this.orbitAzimuthRad,
      elevationRad: this.orbitElevationRad,
      distanceM: this.orbitDistanceM,
    };
  }

  resetCameraView(): void {
    this.panRad = 0.0;
    this.tiltRad = 0.0;
    this.orbitAzimuthRad = DEFAULT_ORBIT_AZIMUTH_RAD;
    this.orbitElevationRad = DEFAULT_ORBIT_ELEVATION_RAD;
    this.orbitDistanceM = DEFAULT_ORBIT_DISTANCE_M;
  }

  resetFlightTrails(): void {
    this.pass3.resetTrails();
  }

  setSpzMode(mode: "off" | "preview"): void {
    this.pass2b.setMode(mode);
  }

  async loadSpzBuffer(buffer: ArrayBuffer | Uint8Array, sourceLabel?: string): Promise<SpzDiagnostics> {
    return this.pass2b.loadSpzBuffer(buffer, sourceLabel);
  }

  getSpzMode(): SpzMode {
    return this.pass2b.getMode();
  }

  getSpzDiagnostics(): SpzDiagnostics {
    return this.pass2b.getDiagnostics();
  }

  resize(width: number, height: number): void {
    this.rhi.resize(width, height);
  }

  renderFrame(
    state: CockpitFrameState,
    options: CockpitRenderOptions = {},
  ): Pass4TensorPacket | null {
    const width = this.rhi.getWidth();
    const height = this.rhi.getHeight();
    const activeMode = state.cameraMode ?? this.cameraMode;
    const activeTarget = state.orbitTarget ?? this.orbitTarget;
    const pack = this.pass2.getTerrainPack();
    const view = buildActiveCameraView(
      state,
      width,
      height,
      state.panRad ?? this.panRad,
      state.tiltRad ?? this.tiltRad,
      {
        cameraMode: activeMode,
        orbitTarget: activeTarget,
        orbitAzimuthRad: state.orbitAzimuthRad ?? this.orbitAzimuthRad,
        orbitElevationRad: state.orbitElevationRad ?? this.orbitElevationRad,
        orbitDistanceM: state.orbitDistanceM ?? this.orbitDistanceM,
        ksfo28lThrWgs84HM: pack.ksfo28lThrWgs84HM,
      },
    );
    this.lastView = view;

    // Render Passes 1 -> 2 -> 2b (3DGS SPZ overlay) -> 3 -> 5 on the primary cockpit framebuffer
    this.rhi.beginFrame([0.02, 0.04, 0.07, 1.0]);
    this.pass1.encode(state, view);
    this.pass2.encode(state, view);
    this.pass2b.encode(state, view, pack);
    this.pass3.encode(state, view, pack);
    this.pass5.encode(state, view, pack, options.extraBadges);
    this.rhi.endFrame();

    // Execute Pass 4 (narrow-FOV traffic_ring_v1 D1-KF pipeline + 3-slot async Staging Ring)
    const tensor = this.pass4.step(state, view, options.pass4);
    this.renderedFrames++;
    return tensor;
  }

  getLatestPass4Tensor(): Pass4TensorPacket | null {
    return this.pass4.getLatestNormalizedTensor();
  }

  getPass4Diagnostics(): Pass4RingDiagnostics {
    return this.pass4.getDiagnostics();
  }

  getLastCameraView(): ActiveCameraView | null {
    return this.lastView;
  }

  getRenderedFrameCount(): number {
    return this.renderedFrames;
  }

  isConformalBlanked(): boolean {
    return this.pass5.isConformalBlanked();
  }

  getVisibleHudBadges(): readonly string[] {
    return this.pass5.getVisibleBadges();
  }

  dispose(): void {
    this.pass1.dispose();
    this.pass2.dispose();
    this.pass2b.dispose();
    this.pass3.dispose();
    this.pass4.dispose();
    this.pass5.dispose();
    this.rhi.dispose();
  }
}
