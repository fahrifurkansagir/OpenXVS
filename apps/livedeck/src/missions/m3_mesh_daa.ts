import type { MissionDefinition } from "./types.ts";

/**
 * Mission M3: 3-Ship MeshDAA Ghost Rejection & Live D0 <-> D1-KF Toggle.
 * Injects ONLY 3rd cooperative peer availability at t = 12.0 s and detector toggle at t = 20.0 s.
 */
export const MISSION_M3_MESH_DAA: MissionDefinition = {
  id: "M3",
  scenarioIndex: 2,
  name: "M3: 3-Ship MeshDAA & D0 ⇄ D1-KF",
  summary:
    "3 converging 3D intruders + 2-ship epipolar ghost; 3rd peer joins at t = 12.0 s; live D0 → D1-KF toggle at t = 20.0 s.",
  seed: 20261103,
  durationS: 45,
  initialAltAglM: 800,
  domainShiftNotice:
    "Live D1-KF browser inference demo (domain-shifted from Gate-G1 Cesium G3a training data; zero Pd claims from Live Deck renderer — numerical parity verified on Gate-G1 fixtures in V2-M1-4).",
  initialPerturbations: {
    camRBiasRad: 0.0,
    gnssSpoofVelNedMps: [0.0, 0.0, 0.0],
    thirdPeerActive: false,
    detectorMode: "D0",
    fogContrast: 1.0,
    lwirEfvsEnabled: false,
    runwayCornerBiasPx: 0.0,
  },
  events: [
    {
      timeS: 12.0,
      label: "3rd cooperative MeshDAA peer joins",
      patch: {
        thirdPeerActive: true,
      },
    },
    {
      timeS: 20.0,
      label: "Toggle detector D0 → D1-KF (live browser ONNX)",
      patch: {
        detectorMode: "D1_KF",
      },
    },
  ],
};
