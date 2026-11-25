import { describe, expect, it } from "vitest";
import {
  DEG_TO_RAD,
  eulerZyxToQuat,
  geodeticToEcef,
  mat3Mul,
  mat3Transpose,
  projectEcefToNdc,
  type Mat3RowMajor,
} from "../rhi/coords.ts";
import {
  KSFO_28L_TOWER_HEIGHT_AGL_M,
  KSFO_28L_TOWER_LAT_DEG,
  KSFO_28L_TOWER_LON_DEG,
  ORBIT_DIST_MAX_M,
  ORBIT_DIST_MIN_M,
  buildActiveCameraView,
  buildLookAtREcefCam,
  type CockpitFrameState,
} from "./index.ts";

function mat3Det(m: Mat3RowMajor): number {
  return (
    m[0] * (m[4] * m[8] - m[5] * m[7]) -
    m[1] * (m[3] * m[8] - m[5] * m[6]) +
    m[2] * (m[3] * m[7] - m[4] * m[6])
  );
}

function mat3OrthoMaxErr(m: Mat3RowMajor): number {
  let maxErr = 0;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) {
        s += m[k * 3 + i] * m[k * 3 + j];
      }
      const target = i === j ? 1.0 : 0.0;
      maxErr = Math.max(maxErr, Math.abs(s - target));
    }
  }
  return maxErr;
}

function makeSampleFrameState(overrides: Partial<CockpitFrameState> = {}): CockpitFrameState {
  const latRad = 37.61172 * DEG_TO_RAD;
  const lonRad = -122.358367 * DEG_TO_RAD;
  const altWgs84M = 420.0;
  const yawRad = 297.7 * DEG_TO_RAD;
  const pitchRad = -3.0 * DEG_TO_RAD;
  const rollRad = 12.0 * DEG_TO_RAD;
  return {
    simTimeS: 10.0,
    latRad,
    lonRad,
    altWgs84M,
    pEcefBody: geodeticToEcef(latRad, lonRad, altWgs84M),
    qNedBody: eulerZyxToQuat(yawRad, pitchRad, rollRad),
    velNedMps: [30.0, -56.0, 3.2],
    yawRad,
    pitchRad,
    rollRad,
    airspeedMps: 64.0,
    simplexMode: 4,
    camHealthMask: 0x3f,
    camRBiasRad: 0.0,
    horizonPlRad: 0.002,
    vHplM: 12.0,
    vVplM: 4.5,
    spoofAlarm: false,
    mandatoryGoAround: false,
    approachActive: true,
    fogContrast: 1.0,
    efvsEnabled: false,
    intruders: [],
    ...overrides,
  };
}

describe("Live Deck v2.1 3-Mode 3D Camera Suite (cockpit / chase / orbit)", () => {
  it("produces strictly orthonormal det=+1 rEcefCam and finite viewProj across all 3 modes", () => {
    const state = makeSampleFrameState();
    for (const mode of ["cockpit", "chase", "orbit"] as const) {
      const view = buildActiveCameraView(state, 1280, 720, 0.0, 0.0, { cameraMode: mode });
      expect(view.cameraMode).toBe(mode);
      expect(mat3OrthoMaxErr(view.rEcefCam)).toBeLessThan(1e-12);
      expect(Math.abs(mat3Det(view.rEcefCam) - 1.0)).toBeLessThan(1e-12);
      expect(Array.from(view.viewProj).every(Number.isFinite)).toBe(true);
      expect(view.pEcefCam.every(Number.isFinite)).toBe(true);
    }
  });

  it("places chase camera ~70.9 m behind & above ownship with ownship visible in frame", () => {
    const state = makeSampleFrameState();
    const view = buildActiveCameraView(state, 1280, 720, 0.0, 0.0, { cameraMode: "chase" });
    const distM = Math.hypot(
      view.pEcefCam[0] - view.pEcefBody[0],
      view.pEcefCam[1] - view.pEcefBody[1],
      view.pEcefCam[2] - view.pEcefBody[2],
    );
    expect(distM).toBeCloseTo(Math.hypot(68.0, 20.0), 4);

    const projBody = projectEcefToNdc(view.pEcefBody, view.pEcefCam, view.rEcefCam, view.intr, {
      width: view.width,
      height: view.height,
    });
    expect(projBody.inFront).toBe(true);
    expect(projBody.visible).toBe(true);
    expect(projBody.vCam[2]).toBeGreaterThan(50.0);
  });

  it("centers ownship ARP at exact viewport center across 360-deg orbit azimuths & clamped distances", () => {
    const state = makeSampleFrameState();
    for (const azDeg of [0, 75, 180, 270]) {
      const view = buildActiveCameraView(state, 1280, 720, 0.0, 0.0, {
        cameraMode: "orbit",
        orbitAzimuthRad: azDeg * DEG_TO_RAD,
        orbitElevationRad: 0.45,
        orbitDistanceM: 220.0,
      });
      const distM = Math.hypot(
        view.pEcefCam[0] - view.pEcefBody[0],
        view.pEcefCam[1] - view.pEcefBody[1],
        view.pEcefCam[2] - view.pEcefBody[2],
      );
      expect(distM).toBeCloseTo(220.0, 4);

      const proj = projectEcefToNdc(view.pEcefBody, view.pEcefCam, view.rEcefCam, view.intr, {
        width: view.width,
        height: view.height,
      });
      expect(proj.inFront).toBe(true);
      expect(proj.visible).toBe(true);
      expect(Math.abs(proj.uPx - view.intr.cx)).toBeLessThan(1e-5);
      expect(Math.abs(proj.vPx - view.intr.cy)).toBeLessThan(1e-5);
    }
  });

  it("clamps extreme orbit elevation/distance and resists NaN without emitting non-finite matrices", () => {
    const state = makeSampleFrameState();
    const nearView = buildActiveCameraView(state, 1280, 720, 0, 0, {
      cameraMode: "orbit",
      orbitAzimuthRad: 0.2,
      orbitElevationRad: 4.5, // exceeds +1.35 clamp
      orbitDistanceM: 5.0,    // below 45 m min clamp
    });
    const nearDist = Math.hypot(
      nearView.pEcefCam[0] - nearView.pEcefBody[0],
      nearView.pEcefCam[1] - nearView.pEcefBody[1],
      nearView.pEcefCam[2] - nearView.pEcefBody[2],
    );
    expect(nearDist).toBeCloseTo(ORBIT_DIST_MIN_M, 4);
    expect(mat3OrthoMaxErr(nearView.rEcefCam)).toBeLessThan(1e-12);
    expect(Math.abs(mat3Det(nearView.rEcefCam) - 1.0)).toBeLessThan(1e-12);

    const farView = buildActiveCameraView(state, 1280, 720, 0, 0, {
      cameraMode: "orbit",
      orbitAzimuthRad: Number.NaN,
      orbitElevationRad: Number.NaN,
      orbitDistanceM: 99999.0,
    });
    const farDist = Math.hypot(
      farView.pEcefCam[0] - farView.pEcefBody[0],
      farView.pEcefCam[1] - farView.pEcefBody[1],
      farView.pEcefCam[2] - farView.pEcefBody[2],
    );
    expect(farDist).toBeCloseTo(ORBIT_DIST_MAX_M, 4);
    expect(Array.from(farView.viewProj).every(Number.isFinite)).toBe(true);
  });

  it("handles degenerate look-at collinear with up reference gracefully via fallback axis", () => {
    const r = buildLookAtREcefCam([0, 0, 100], [0, 0, 0], [0, 0, 1], [1, 0, 0]);
    expect(mat3OrthoMaxErr(r)).toBeLessThan(1e-12);
    expect(Math.abs(mat3Det(r) - 1.0)).toBeLessThan(1e-12);
  });

  it("maintains exact rEcefBody * rBodyCam === rEcefCam and rNedBody * rBodyCam === R_ned_cam across all 3 modes (Pass 1 sky/horizon sync)", () => {
    const state = makeSampleFrameState();
    for (const mode of ["cockpit", "chase", "orbit"] as const) {
      const view = buildActiveCameraView(state, 1280, 720, 0.25, -0.1, {
        cameraMode: mode,
        orbitAzimuthRad: 1.4,
        orbitElevationRad: 0.42,
        orbitDistanceM: 185.0,
      });
      const composedEcefCam = mat3Mul(view.rEcefBody, view.rBodyCam);
      let maxEcefErr = 0;
      for (let i = 0; i < 9; i++) {
        maxEcefErr = Math.max(maxEcefErr, Math.abs(composedEcefCam[i] - view.rEcefCam[i]));
      }
      expect(maxEcefErr).toBeLessThan(1e-12);

      const pass1RNedCam = mat3Mul(view.rNedBody, view.rBodyCam);
      const refRNedCam = mat3Mul(mat3Transpose(view.rEcefNed), view.rEcefCam);
      let maxNedErr = 0;
      for (let i = 0; i < 9; i++) {
        maxNedErr = Math.max(maxNedErr, Math.abs(pass1RNedCam[i] - refRNedCam[i]));
      }
      expect(maxNedErr).toBeLessThan(1e-12);
    }
  });

  it("centers orbit/intruder on active non-ghost intruder (with safe ownship fallback when empty) and preserves Pass-1 sky lock", () => {
    const base = makeSampleFrameState();
    const intruderPos = geodeticToEcef(37.624 * DEG_TO_RAD, -122.371 * DEG_TO_RAD, 460.0);
    const ghostPos = geodeticToEcef(37.619 * DEG_TO_RAD, -122.350 * DEG_TO_RAD, 410.0);
    const stateWithTraffic = makeSampleFrameState({
      intruders: [
        {
          id: "GHOST-1",
          posEcef: ghostPos,
          rangeM: 1200,
          ttcS: 30,
          alertTier: 0,
          isGhost: true,
          ghostRejected: true,
        },
        {
          id: "T1",
          posEcef: intruderPos,
          rangeM: 1650,
          ttcS: 22,
          alertTier: 2,
          meshVerified: true,
        },
      ],
    });

    const view = buildActiveCameraView(stateWithTraffic, 1280, 720, 0, 0, {
      cameraMode: "orbit",
      orbitTarget: "intruder",
      orbitAzimuthRad: 0.7,
      orbitElevationRad: 0.35,
      orbitDistanceM: 190.0,
    });
    expect(view.orbitTarget).toBe("intruder");
    expect(mat3OrthoMaxErr(view.rEcefCam)).toBeLessThan(1e-12);
    expect(Math.abs(mat3Det(view.rEcefCam) - 1.0)).toBeLessThan(1e-12);

    const distToIntruder = Math.hypot(
      view.pEcefCam[0] - intruderPos[0],
      view.pEcefCam[1] - intruderPos[1],
      view.pEcefCam[2] - intruderPos[2],
    );
    expect(distToIntruder).toBeCloseTo(190.0, 4);

    const projIntr = projectEcefToNdc(intruderPos, view.pEcefCam, view.rEcefCam, view.intr, {
      width: view.width,
      height: view.height,
    });
    expect(projIntr.inFront).toBe(true);
    expect(projIntr.visible).toBe(true);
    expect(Math.abs(projIntr.uPx - view.intr.cx)).toBeLessThan(1e-5);
    expect(Math.abs(projIntr.vPx - view.intr.cy)).toBeLessThan(1e-5);

    const composed = mat3Mul(view.rEcefBody, view.rBodyCam);
    for (let i = 0; i < 9; i++) {
      expect(Math.abs(composed[i] - view.rEcefCam[i])).toBeLessThan(1e-12);
    }

    // Empty intruder array falls back safely to ownship ARP
    const fallbackView = buildActiveCameraView(base, 1280, 720, 0, 0, {
      cameraMode: "orbit",
      orbitTarget: "intruder",
      orbitDistanceM: 175.0,
    });
    const projOwn = projectEcefToNdc(fallbackView.pEcefBody, fallbackView.pEcefCam, fallbackView.rEcefCam, fallbackView.intr, {
      width: fallbackView.width,
      height: fallbackView.height,
    });
    expect(projOwn.inFront).toBe(true);
    expect(Math.abs(projOwn.uPx - fallbackView.intr.cx)).toBeLessThan(1e-5);
    expect(Math.abs(projOwn.vPx - fallbackView.intr.cy)).toBeLessThan(1e-5);
  });

  it("anchors orbit/tower at exact KSFO 28L Control Tower ECEF tracking pEcefBody with <1e-12 Pass-1 sky sync", () => {
    const state = makeSampleFrameState();
    const thrH = -28.29;
    const expectedTowerEcef = geodeticToEcef(
      KSFO_28L_TOWER_LAT_DEG * DEG_TO_RAD,
      KSFO_28L_TOWER_LON_DEG * DEG_TO_RAD,
      thrH + KSFO_28L_TOWER_HEIGHT_AGL_M,
    );

    const view = buildActiveCameraView(state, 1280, 720, 0, 0, {
      cameraMode: "orbit",
      orbitTarget: "tower",
      ksfo28lThrWgs84HM: thrH,
    });
    expect(view.cameraMode).toBe("orbit");
    expect(view.orbitTarget).toBe("tower");
    expect(mat3OrthoMaxErr(view.rEcefCam)).toBeLessThan(1e-12);
    expect(Math.abs(mat3Det(view.rEcefCam) - 1.0)).toBeLessThan(1e-12);

    const posErr = Math.hypot(
      view.pEcefCam[0] - expectedTowerEcef[0],
      view.pEcefCam[1] - expectedTowerEcef[1],
      view.pEcefCam[2] - expectedTowerEcef[2],
    );
    expect(posErr).toBeLessThan(1e-6);

    const projOwn = projectEcefToNdc(view.pEcefBody, view.pEcefCam, view.rEcefCam, view.intr, {
      width: view.width,
      height: view.height,
    });
    expect(projOwn.inFront).toBe(true);
    expect(projOwn.visible).toBe(true);
    expect(Math.abs(projOwn.uPx - view.intr.cx)).toBeLessThan(1e-5);
    expect(Math.abs(projOwn.vPx - view.intr.cy)).toBeLessThan(1e-5);

    const composed = mat3Mul(view.rEcefBody, view.rBodyCam);
    for (let i = 0; i < 9; i++) {
      expect(Math.abs(composed[i] - view.rEcefCam[i])).toBeLessThan(1e-12);
    }
    const pass1RNedCam = mat3Mul(view.rNedBody, view.rBodyCam);
    const refRNedCam = mat3Mul(mat3Transpose(view.rEcefNed), view.rEcefCam);
    for (let i = 0; i < 9; i++) {
      expect(Math.abs(pass1RNedCam[i] - refRNedCam[i])).toBeLessThan(1e-12);
    }
  });
});
