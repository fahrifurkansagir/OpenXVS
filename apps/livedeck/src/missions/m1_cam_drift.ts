import type { MissionDefinition } from "./types.ts";

/**
 * Mission M1: Camera R +1.2 deg (+0.020944 rad) Extrinsic Drift over SF Bay.
 * Injects ONLY physical misalignment on Camera R at t = 8.0 s.
 */
export const MISSION_M1_CAM_DRIFT: MissionDefinition = {
  id: "M1",
  scenarioIndex: 0,
  name: "M1: Cam-R +1.2° Extrinsic Drift",
  summary:
    "Cruise over SF Bay (600 m AGL). At t = 8.0 s, inject +0.020944 rad (+1.2°) physical extrinsic step misalignment on Camera R.",
  seed: 20261101,
  durationS: 45,
  initialAltAglM: 600,
  initialPerturbations: {
    camRBiasRad: 0.0,
    gnssSpoofVelNedMps: [0.0, 0.0, 0.0],
    thirdPeerActive: false,
    detectorMode: "D1_KF",
    fogContrast: 1.0,
    lwirEfvsEnabled: false,
    runwayCornerBiasPx: 0.0,
  },
  events: [
    {
      timeS: 8.0,
      label: "Inject Cam-R +0.020944 rad (+1.2°) extrinsic step",
      patch: {
        camRBiasRad: 0.020944,
      },
    },
  ],
};
