import type { MissionDefinition } from "./types.ts";

/**
 * Mission M4: KSFO 28L 0.22x Fog + LWIR EFVS + Vision-RAIM Corner Bias Perturbation.
 * Injects ONLY fog contrast (0.22), LWIR EFVS toggle at t = 10.0 s, and +18 px threshold corner bias at t = 24.0 s.
 */
export const MISSION_M4_KSFO_AUTOLAND: MissionDefinition = {
  id: "M4",
  scenarioIndex: 3,
  name: "M4: KSFO 28L Fog + EFVS + V-RAIM",
  summary:
    "3° approach to KSFO 28L (h = -28.67 m WGS-84, 28R & Taxiway C adjacent) in 0.22× fog; LWIR EFVS ON at t = 10.0 s; +18 px corner bias at t = 24.0 s.",
  seed: 20261104,
  durationS: 45,
  initialAltAglM: 350,
  referenceRunwayWgs84HM: -28.6714,
  initialPerturbations: {
    camRBiasRad: 0.0,
    gnssSpoofVelNedMps: [0.0, 0.0, 0.0],
    thirdPeerActive: false,
    detectorMode: "D1_KF",
    fogContrast: 0.22,
    lwirEfvsEnabled: false,
    runwayCornerBiasPx: 0.0,
  },
  events: [
    {
      timeS: 10.0,
      label: "Toggle 8–14 μm LWIR EFVS ON",
      patch: {
        lwirEfvsEnabled: true,
      },
    },
    {
      timeS: 24.0,
      label: "Inject +18 px runway threshold corner bias",
      patch: {
        runwayCornerBiasPx: 18.0,
      },
    },
  ],
};
