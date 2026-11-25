/**
 * Shared Render Pass & Frame State Interfaces for OpenXVS Live Deck (Task 6 -> Task 7 boundary).
 * Strictly obeys CONVENTIONS.md v1.0 (SI units: metres, seconds, radians; ECEF/NED/BODY/CAM).
 */

import type { Mat3RowMajor, PinholeIntrinsics, QuatHamilton, Vec3 } from "../rhi/coords.ts";

/** Matches xvs-rta-core SimplexMode #[repr(u8)] discriminant values. */
export type SimplexModeValue = 0 | 1 | 2 | 3 | 4;

/** Live Deck v2.1 3D Camera Suite modes. */
export type CameraViewMode = "cockpit" | "chase" | "orbit";

/** Live Deck v2.2 3D Orbit / Spotter camera target focus. */
export type OrbitTargetMode = "ownship" | "intruder" | "tower";

export const SIMPLEX_MODE_LABELS: Record<SimplexModeValue, string> = {
  0: "P0 INVALID-INHIBIT",
  1: "P1 FALLBACK-D0-SAFE",
  2: "P2 DEGRADED-COAST",
  3: "P3 ISOLATED-5CAM",
  4: "P4 NOMINAL-ML",
};

export interface IntruderTrack {
  readonly id: string;
  readonly posEcef: Vec3;
  readonly velEcef?: Vec3;
  readonly rangeM: number;
  readonly ttcS: number;
  /** DO-365B 4-tier alert level: 0=Traffic, 1=Preventive, 2=Corrective Caution, 3=Warning. */
  readonly alertTier: 0 | 1 | 2 | 3;
  readonly wcvRadiusM?: number;
  readonly isGhost?: boolean;
  readonly ghostRejected?: boolean;
  readonly meshVerified?: boolean;
  /** Peer aircraft ECEF positions forming XVS-MeshDAA sight-lines to this track. */
  readonly peerOriginsEcef?: readonly Vec3[];
}

export interface AiDetectionPeak {
  readonly u: number;
  readonly v: number;
  readonly score: number;
}

export interface CockpitFrameState {
  readonly simTimeS: number;
  readonly latRad: number;
  readonly lonRad: number;
  readonly altWgs84M: number;
  /** Optional precomputed ECEF position (m); derived from geodetic if omitted. */
  readonly pEcefBody?: Vec3;
  /** Hamilton quaternion (w, x, y, z), w >= 0, mapping BODY -> NED. */
  readonly qNedBody: QuatHamilton;
  /** Ownship velocity in local NED [vn, ve, vd] (m/s). */
  readonly velNedMps: Vec3;
  readonly yawRad: number;
  readonly pitchRad: number;
  readonly rollRad: number;
  readonly airspeedMps: number;
  /** Simplex RTA mode (0 = INVALID_INHIBIT blanks conformal cues <= 1 tick + next vsync). */
  readonly simplexMode: SimplexModeValue;
  /** 6-bit camera health bitmask ordered F(0), R(1), B(2), L(3), U(4), D(5). */
  readonly camHealthMask: number;
  /** Physical Cam-R extrinsic drift perturbation in radians (M1). */
  readonly camRBiasRad: number;
  /** Attitude/horizon protection level PL_0.01 in radians for conformal whiskers. */
  readonly horizonPlRad: number;
  /** Vision-RAIM horizontal & vertical protection levels (metres). */
  readonly vHplM: number;
  readonly vVplM: number;
  readonly spoofAlarm: boolean;
  readonly mandatoryGoAround: boolean;
  readonly approachActive: boolean;
  /** Fog/haze visual contrast multiplier in [0.08, 1.0] (e.g., 0.22 in M4). */
  readonly fogContrast: number;
  /** Physics-inspired 8-14 um LWIR EFVS toggle. */
  readonly efvsEnabled: boolean;
  readonly glareStrength?: number;
  /** Interactive free-look pan (right +) and tilt (up +) in radians. */
  readonly panRad?: number;
  readonly tiltRad?: number;
  /** Active 3D camera mode (default "cockpit"). */
  readonly cameraMode?: CameraViewMode;
  /** Active 3D orbit/spotter target focus (default "ownship"). */
  readonly orbitTarget?: OrbitTargetMode;
  /** Spherical 3D orbit azimuth (rad), elevation (rad), and distance (m) around target. */
  readonly orbitAzimuthRad?: number;
  readonly orbitElevationRad?: number;
  readonly orbitDistanceM?: number;
  /** Active viewport horizontal FOV in degrees (default 88 deg). */
  readonly fovHDeg?: number;
  readonly intruders: readonly IntruderTrack[];
  /** Explicit AI backend badge string supplied by Stage-1 ORT worker (Charter §7.5). */
  readonly aiBadge?: string;
  readonly detectorMode?: "D0" | "D1_KF";
  readonly aiDetections?: readonly AiDetectionPeak[];
}

export interface TerrainRingData {
  readonly name: string;
  readonly extentKm: number;
  readonly width: number;
  readonly height: number;
  readonly latMin: number;
  readonly latMax: number;
  readonly lonMin: number;
  readonly lonMax: number;
  readonly minWgs84HM: number;
  readonly maxWgs84HM: number;
  readonly heightsU16: Uint16Array;
}

export interface RunwayGeometry {
  readonly id: string;
  readonly kind: string;
  readonly widthM: number;
  readonly lengthM: number;
  readonly trueHeadingDeg: number;
  readonly thresholdLatDeg: number;
  readonly thresholdLonDeg: number;
  readonly thresholdWgs84HM: number;
  readonly oppositeEndLatDeg: number;
  readonly oppositeEndLonDeg: number;
  readonly oppositeEndWgs84HM: number;
  /** 4 WGS-84 corner points [latDeg, lonDeg, hM] if supplied by pack. */
  readonly polygon?: ReadonlyArray<readonly [number, number, number]>;
}

export interface TerrainPackData {
  readonly terrainMode: "usgs_3dep_geoid18_ellipsoidal" | "procedural_ci" | string;
  readonly geoidModel: string;
  readonly datum: string;
  readonly originLatDeg: number;
  readonly originLonDeg: number;
  readonly ksfo28lThrLatDeg: number;
  readonly ksfo28lThrLonDeg: number;
  readonly ksfo28lThrWgs84HM: number;
  readonly hMinM: number;
  readonly hStepM: number;
  readonly outerRing: TerrainRingData;
  readonly innerRing: TerrainRingData;
  readonly runways: readonly RunwayGeometry[];
}

export interface ActiveCameraView {
  readonly cameraMode: CameraViewMode;
  readonly orbitTarget: OrbitTargetMode;
  readonly pEcefBody: Vec3;
  readonly pEcefCam: Vec3;
  readonly rEcefNed: Mat3RowMajor;
  readonly rNedBody: Mat3RowMajor;
  readonly rEcefBody: Mat3RowMajor;
  readonly rBodyCam: Mat3RowMajor;
  readonly rEcefCam: Mat3RowMajor;
  readonly intr: PinholeIntrinsics;
  readonly width: number;
  readonly height: number;
  readonly viewProj: Float32Array;
}

export interface Pass4RenderOptions {
  /** Deterministic parity flag: true disables EMVA-1288 shot/read noise. */
  readonly noiseOff?: boolean;
  /** Apply 5x5 matched ring_mono_v0_nodist optical PSF (default true). */
  readonly applyPsf?: boolean;
  /** Optional override for ROI top-left (u0, v0) inside 2448x2048 traffic_ring_v1 frame. */
  readonly roiOrigin?: readonly [number, number];
}

export interface Pass4TensorPacket {
  readonly data: Float32Array;
  readonly shape: readonly [1, 5, 256, 256];
  readonly simTimeS: number;
  readonly frameIndex: number;
  readonly roiU0: number;
  readonly roiV0: number;
  readonly noiseOff: boolean;
  readonly computePath: "webgpu-compute" | "cpu-fp32-fallback";
  readonly stats: {
    readonly means: readonly [number, number, number, number, number];
    readonly std0: number;
  };
}

export interface CockpitRenderOptions {
  readonly pass4?: Pass4RenderOptions;
  /** Optional explicit external badge strings (e.g., from AI worker). */
  readonly extraBadges?: readonly string[];
}

export interface MsdfGlyphEntry {
  readonly char: string;
  readonly u0: number;
  readonly v0: number;
  readonly u1: number;
  readonly v1: number;
  readonly advance: number;
  readonly planeW: number;
  readonly planeH: number;
}

export interface MsdfAtlasMetadata {
  readonly schema: string;
  readonly font_id: string;
  readonly atlas_width: number;
  readonly atlas_height: number;
  readonly cell_width: number;
  readonly cell_height: number;
  readonly cols: number;
  readonly rows: number;
  readonly first_code: number;
  readonly last_code: number;
  readonly px_range: number;
  readonly glyphs: Record<string, MsdfGlyphEntry>;
}
