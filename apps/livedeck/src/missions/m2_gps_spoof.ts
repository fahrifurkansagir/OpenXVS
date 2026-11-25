import type { MissionDefinition } from "./types.ts";

/**
 * Mission M2: +2.5 m/s East Stealth GNSS Velocity Spoofing Ramp.
 * Injects ONLY East GNSS velocity bias into GNSS measurement stream at t = 8.0 s.
 */
export const MISSION_M2_GPS_SPOOF: MissionDefinition = {
  id: "M2",
  scenarioIndex: 1,
  name: "M2: +2.5 m/s Stealth GPS Spoofing",
  summary:
    "Coastal IFR leg (1,200 m AGL). At t = 8.0 s, inject +2.5 m/s East velocity spoofing ramp into GNSS Bank 0.",
  seed: 20261102,
  durationS: 45,
  initialAltAglM: 1200,
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
      label: "Inject +2.5 m/s East GNSS velocity ramp",
      patch: {
        gnssSpoofVelNedMps: [0.0, 2.5, 0.0],
      },
    },
  ],
};
