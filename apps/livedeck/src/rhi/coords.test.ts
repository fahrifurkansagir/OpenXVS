import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { parseKsfoTerrainPack } from "../passes/pass2_terrain_cdlod.ts";
import {
  DEG_TO_RAD,
  RIG_CAMERA_ORDER,
  TRAFFIC_RING_V1_IFOV_DEG_PER_PX,
  TRAFFIC_RING_V1_INTRINSICS,
  XVS6_CUBE_V1_CAMERAS,
  XVS6_CUBE_V1_INTRINSICS,
  buildCamViewProjMatrix,
  buildFreeLookRBodyCam,
  buildPinholeIntrinsics,
  buildRoiIntrinsics,
  computeRigCameraPose,
  ecefToGeodetic,
  eulerZyxToQuat,
  eulerZyxToRotMat,
  geodeticToEcef,
  mat3Mul,
  mat3MulVec,
  mat3TransposeMulVec,
  mulMat4ColMajorVec3,
  ndcToPixelCenter,
  projectEcefToNdc,
  quatToRotMat,
  rEcefNed,
  type Vec3,
} from "./coords.ts";

describe("CONVENTIONS.md §1-§5 exact WGS-84, frames & OpenCV CAM -> WebGPU NDC projection", () => {
  it("satisfies WGS-84 geodetic <-> ECEF round-trip < 1e-6 m at KSFO 28L (h=-28.67 m) and 600 m AGL", () => {
    const cases = [
      { latDeg: 37.61172, lonDeg: -122.358367, hM: -28.6714 },
      { latDeg: 37.6188056, lonDeg: -122.3754167, hM: 600.0 },
      { latDeg: 0.0, lonDeg: 0.0, hM: 12000.0 },
      { latDeg: -33.8688, lonDeg: 151.2093, hM: 350.0 },
    ];
    for (const c of cases) {
      const ecef = geodeticToEcef(c.latDeg * DEG_TO_RAD, c.lonDeg * DEG_TO_RAD, c.hM);
      const back = ecefToGeodetic(ecef[0], ecef[1], ecef[2]);
      const ecef2 = geodeticToEcef(back.latRad, back.lonRad, back.hM);
      const errM = Math.hypot(ecef[0] - ecef2[0], ecef[1] - ecef2[1], ecef[2] - ecef2[2]);
      expect(errM).toBeLessThan(1e-6);
      expect(Math.abs(back.hM - c.hM)).toBeLessThan(1e-6);
    }
  });

  it("obeys mandatory Euler & Hamilton quaternion sign tests (CONVENTIONS.md §4)", () => {
    // psi = 90 deg => body x points East [0, 1, 0] in NED
    const rYaw90 = eulerZyxToRotMat(90 * DEG_TO_RAD, 0, 0);
    const bxEast = mat3MulVec(rYaw90, [1, 0, 0]);
    expect(bxEast[0]).toBeCloseTo(0, 12);
    expect(bxEast[1]).toBeCloseTo(1, 12);
    expect(bxEast[2]).toBeCloseTo(0, 12);

    // theta > 0 => nose up (body x has negative Down component)
    const rPitchUp = eulerZyxToRotMat(0, 15 * DEG_TO_RAD, 0);
    const bxUp = mat3MulVec(rPitchUp, [1, 0, 0]);
    expect(bxUp[2]).toBeLessThan(-0.2);

    // phi > 0 => right wing down (body y has positive Down component)
    const rRollRight = eulerZyxToRotMat(0, 0, 20 * DEG_TO_RAD);
    const byDown = mat3MulVec(rRollRight, [0, 1, 0]);
    expect(byDown[2]).toBeGreaterThan(0.3);

    // Quaternion canonical w >= 0 and matches Euler rotation matrix
    const q = eulerZyxToQuat(285 * DEG_TO_RAD, -3 * DEG_TO_RAD, 12 * DEG_TO_RAD);
    expect(q[0]).toBeGreaterThanOrEqual(0.0);
    const rFromQ = quatToRotMat(q);
    const rFromE = eulerZyxToRotMat(285 * DEG_TO_RAD, -3 * DEG_TO_RAD, 12 * DEG_TO_RAD);
    for (let i = 0; i < 9; i++) {
      expect(rFromQ[i]).toBeCloseTo(rFromE[i], 12);
    }
  });

  it("matches rig_nominal.json camera order F,R,B,L,U,D and traffic_ring_v1 IFOV", () => {
    expect(XVS6_CUBE_V1_CAMERAS.map((c) => c.id)).toEqual(RIG_CAMERA_ORDER);
    expect(TRAFFIC_RING_V1_INTRINSICS.fx).toBe(2319.0);
    expect(TRAFFIC_RING_V1_IFOV_DEG_PER_PX).toBeCloseTo(0.024707, 5);
  });

  it("clips behind-camera points (v_cam.z <= 0) cleanly without NaN propagation", () => {
    const pCam: Vec3 = geodeticToEcef(37.6188 * DEG_TO_RAD, -122.3754 * DEG_TO_RAD, 600.0);
    const rNed = rEcefNed(37.6188 * DEG_TO_RAD, -122.3754 * DEG_TO_RAD);
    const rBody = mat3Mul(rNed, eulerZyxToRotMat(298 * DEG_TO_RAD, -3 * DEG_TO_RAD, 0));
    const { pEcefCam, rEcefCam } = computeRigCameraPose(pCam, rBody, XVS6_CUBE_V1_CAMERAS[0]);

    // Point strictly behind the forward camera (v_cam = [0, 0, -500])
    const behindDelta = mat3MulVec(rEcefCam, [0, 0, -500]);
    const pBehind: Vec3 = [
      pEcefCam[0] + behindDelta[0],
      pEcefCam[1] + behindDelta[1],
      pEcefCam[2] + behindDelta[2],
    ];
    const resBehind = projectEcefToNdc(
      pBehind,
      pEcefCam,
      rEcefCam,
      XVS6_CUBE_V1_INTRINSICS,
      { width: 1024, height: 1024 },
    );
    expect(resBehind.inFront).toBe(false);
    expect(resBehind.visible).toBe(false);

    // Point on optical plane (v_cam.z == 0)
    const onPlaneDelta = mat3MulVec(rEcefCam, [100, 0, 0]);
    const pOnPlane: Vec3 = [
      pEcefCam[0] + onPlaneDelta[0],
      pEcefCam[1] + onPlaneDelta[1],
      pEcefCam[2] + onPlaneDelta[2],
    ];
    const resPlane = projectEcefToNdc(
      pOnPlane,
      pEcefCam,
      rEcefCam,
      XVS6_CUBE_V1_INTRINSICS,
      { width: 1024, height: 1024 },
    );
    expect(resPlane.inFront).toBe(false);
    expect(resPlane.visible).toBe(false);
  });

  it("achieves <= 0.05 px (target < 1e-3 px) OpenCV CAM -> WebGPU NDC parity across all 6 XVS6-CUBE-v1 cameras and free-look viewports", () => {
    // Test at both KSFO 28L threshold (h = -28.67 m) and SF Bay cruise (h = 600 m AGL)
    const altitudes = [-28.6714, 600.0];
    let maxErrPx = 0.0;

    for (const hM of altitudes) {
      const latRad = 37.61172 * DEG_TO_RAD;
      const lonRad = -122.358367 * DEG_TO_RAD;
      const pBody = geodeticToEcef(latRad, lonRad, hM);
      const rNed = rEcefNed(latRad, lonRad);
      const rBody = mat3Mul(rNed, eulerZyxToRotMat(298.0 * DEG_TO_RAD, -3.0 * DEG_TO_RAD, 5.5 * DEG_TO_RAD));

      // All 6 XVS6-CUBE-v1 cameras (F, R, B, L, U, D)
      for (const camSpec of XVS6_CUBE_V1_CAMERAS) {
        const { pEcefCam, rEcefCam } = computeRigCameraPose(pBody, rBody, camSpec);
        const vp = { width: 1024, height: 1024, zNear: 1.0, zFar: 200000.0 };
        const viewProjF32 = buildCamViewProjMatrix(rEcefCam, XVS6_CUBE_V1_INTRINSICS, vp);

        const sampleRaysCam: Vec3[] = [
          [0, 0, 1200],
          [-400, -350, 900],
          [420, -380, 1500],
          [-300, 410, 2200],
          [500, 480, 5000],
          [12.34, -8.76, 15000],
        ];

        for (const vCamTrue of sampleRaysCam) {
          const dEcef = mat3MulVec(rEcefCam, vCamTrue);
          const pObj: Vec3 = [pEcefCam[0] + dEcef[0], pEcefCam[1] + dEcef[1], pEcefCam[2] + dEcef[2]];

          // Ground truth per CONVENTIONS.md §5
          const pc = mat3TransposeMulVec(rEcefCam, dEcef);
          const uRef = XVS6_CUBE_V1_INTRINSICS.fx * (pc[0] / pc[2]) + XVS6_CUBE_V1_INTRINSICS.cx;
          const vRef = XVS6_CUBE_V1_INTRINSICS.fy * (pc[1] / pc[2]) + XVS6_CUBE_V1_INTRINSICS.cy;

          // 1. projectEcefToNdc -> pixel center recovery
          const proj = projectEcefToNdc(pObj, pEcefCam, rEcefCam, XVS6_CUBE_V1_INTRINSICS, vp);
          expect(proj.inFront).toBe(true);
          expect(proj.zNdc).toBeGreaterThanOrEqual(0.0);
          expect(proj.zNdc).toBeLessThanOrEqual(1.0);
          const rec1 = ndcToPixelCenter(proj.xNdc, proj.yNdc, vp);
          const err1 = Math.hypot(rec1.uPx - uRef, rec1.vPx - vRef);
          maxErrPx = Math.max(maxErrPx, err1);

          // 2. GPU Float32 ViewProj matrix (simulating WGSL vertex shader + WebGL2 Naga Z-remap)
          const clip = mulMat4ColMajorVec3(viewProjF32, [dEcef[0], dEcef[1], dEcef[2]]);
          const xNdcGpu = clip[0] / clip[3];
          const yNdcGpu = clip[1] / clip[3];
          const zNdcGpu = clip[2] / clip[3];
          expect(zNdcGpu).toBeGreaterThanOrEqual(0.0);
          expect(zNdcGpu).toBeLessThanOrEqual(1.0);

          // Simulate Naga ADJUST_COORDINATE_SPACE + WebGL2 FBO vertical blit
          const zNdcGl = (clip[2] * 2.0 - clip[3]) / clip[3];
          expect(zNdcGl).toBeGreaterThanOrEqual(-1.0);
          expect(zNdcGl).toBeLessThanOrEqual(1.0);

          const recGpu = ndcToPixelCenter(xNdcGpu, yNdcGpu, vp);
          const errGpu = Math.hypot(recGpu.uPx - uRef, recGpu.vPx - vRef);
          maxErrPx = Math.max(maxErrPx, errGpu);
          expect(errGpu).toBeLessThan(1e-3);
        }
      }

      // Free-look wide 16:9 viewport (1280x720) with head pan/tilt
      const rBodyCamFree = buildFreeLookRBodyCam(25 * DEG_TO_RAD, -8 * DEG_TO_RAD);
      const rEcefFree = mat3Mul(rBody, rBodyCamFree);
      const vpFree = { width: 1280, height: 720, zNear: 1.0, zFar: 200000.0 };
      const intrFree = buildPinholeIntrinsics(1280, 720, 90.0);
      const vpMatFree = buildCamViewProjMatrix(rEcefFree, intrFree, vpFree);

      for (const vCamTrue of [[0, 0, 800], [-300, 120, 1400], [450, -180, 3200]] as Vec3[]) {
        const dEcef = mat3MulVec(rEcefFree, vCamTrue);
        const pObj: Vec3 = [pBody[0] + dEcef[0], pBody[1] + dEcef[1], pBody[2] + dEcef[2]];
        const uRef = intrFree.fx * (vCamTrue[0] / vCamTrue[2]) + intrFree.cx;
        const vRef = intrFree.fy * (vCamTrue[1] / vCamTrue[2]) + intrFree.cy;

        const clip = mulMat4ColMajorVec3(vpMatFree, dEcef);
        const rec = ndcToPixelCenter(clip[0] / clip[3], clip[1] / clip[3], vpFree);
        const err = Math.hypot(rec.uPx - uRef, rec.vPx - vRef);
        maxErrPx = Math.max(maxErrPx, err);
        expect(err).toBeLessThan(1e-3);

        const proj = projectEcefToNdc(pObj, pBody, rEcefFree, intrFree, vpFree);
        expect(proj.visible).toBe(true);
        expect(Math.hypot(proj.uPx - uRef, proj.vPx - vRef)).toBeLessThan(1e-9);
      }

      // Narrow-FOV traffic_ring_v1 256x256 off-center ROI sub-crop
      const roiIntr = buildRoiIntrinsics(TRAFFIC_RING_V1_INTRINSICS, 1096, 896, 256, 256);
      const roiVp = { width: 256, height: 256, zNear: 1.0, zFar: 200000.0 };
      const roiMat = buildCamViewProjMatrix(mat3Mul(rBody, XVS6_CUBE_V1_CAMERAS[0].rBodyCam), roiIntr, roiVp);
      const vCamRoi: Vec3 = [45.0, -22.0, 3000.0];
      const dRoi = mat3MulVec(mat3Mul(rBody, XVS6_CUBE_V1_CAMERAS[0].rBodyCam), vCamRoi);
      const uRoiRef = roiIntr.fx * (vCamRoi[0] / vCamRoi[2]) + roiIntr.cx;
      const vRoiRef = roiIntr.fy * (vCamRoi[1] / vCamRoi[2]) + roiIntr.cy;
      const clipRoi = mulMat4ColMajorVec3(roiMat, dRoi);
      const recRoi = ndcToPixelCenter(clipRoi[0] / clipRoi[3], clipRoi[1] / clipRoi[3], roiVp);
      const errRoi = Math.hypot(recRoi.uPx - uRoiRef, recRoi.vPx - vRoiRef);
      maxErrPx = Math.max(maxErrPx, errRoi);
      expect(errRoi).toBeLessThan(1e-3);
    }

    expect(maxErrPx).toBeLessThan(0.05);
  });

  it("parses the real USGS 3DEP + GEOID18 ksfo_3dep_geoid18_v1.bin pack cleanly", () => {
    const binPath = resolve(process.cwd(), "public/assets/ksfo_3dep_geoid18_v1.bin");
    const raw = readFileSync(binPath);
    const ab = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
    const pack = parseKsfoTerrainPack(ab);

    expect(pack.terrainMode).toBe("usgs_3dep_geoid18_ellipsoidal");
    expect(pack.geoidModel).toBe("GEOID18");
    expect(pack.ksfo28lThrWgs84HM).toBeGreaterThan(-35.0);
    expect(pack.ksfo28lThrWgs84HM).toBeLessThan(-22.0);
    expect(pack.innerRing.width).toBe(256);
    expect(pack.innerRing.height).toBe(256);
    expect(pack.outerRing.width).toBe(256);
    expect(pack.outerRing.height).toBe(256);
    expect(pack.runways.map((r) => r.id)).toEqual(["28L", "28R", "Taxiway C"]);
  });
});
