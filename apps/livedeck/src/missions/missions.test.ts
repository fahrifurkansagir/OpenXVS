/**
 * TDD purity & schedule verification for OpenXVS Live Deck Red-Team Missions M1..M4.
 *
 * Enforces SP1_LIVEDECK_DESIGN.md §4.5 Anti-Scripting Rule (M6):
 *   Mission definitions may ONLY emit seeded initial scenario IDs and time-stamped
 *   physical/sensor fault perturbation structs — NEVER Simplex modes, alarms, or verdicts.
 */

import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import {
  ALLOWED_PERTURBATION_KEYS,
  ALL_MISSIONS,
  evaluateMissionPerturbations,
  getMissionById,
  type MissionId,
} from "./index.ts";
import {
  StagingRingBuffer,
  extractDetectionPeaks,
  selectAiBackend,
} from "../ai/ort_worker.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));

const FORBIDDEN_VERDICT_PATTERNS = [
  /\bsimplexMode\b/,
  /\bspoofAlarm\b/,
  /\bgoAroundLatched\b/,
  /\bmandatoryGoAround\b/,
  /\bisolatedCam\b/,
  /\bforceMode\b/,
  /\boverrideAlarm\b/,
  /\bverdictOverride\b/,
];

describe("Red-Team Missions M1..M4 strict anti-scripting purity (M6)", () => {
  it("statically contains ZERO forbidden verdict/mode override identifiers in mission files", () => {
    const files = [
      "m1_cam_drift.ts",
      "m2_gps_spoof.ts",
      "m3_mesh_daa.ts",
      "m4_ksfo_autoland.ts",
    ];
    for (const file of files) {
      const src = readFileSync(resolve(__dirname, file), "utf-8");
      for (const pat of FORBIDDEN_VERDICT_PATTERNS) {
        expect(src, `Forbidden override pattern ${pat} found in ${file}`).not.toMatch(pat);
      }
    }
  });

  it("dynamically emits ONLY whitelisted physical perturbation keys at every timestamp", () => {
    const allowedSet = new Set<string>(ALLOWED_PERTURBATION_KEYS);
    expect(ALL_MISSIONS.map((m) => m.id)).toEqual(["M1", "M2", "M3", "M4"]);

    const seenSeeds = new Set<number>();
    for (const mission of ALL_MISSIONS) {
      expect(mission.durationS).toBe(45);
      expect(seenSeeds.has(mission.seed)).toBe(false);
      seenSeeds.add(mission.seed);

      // Initial state keys strictly within allowlist
      for (const key of Object.keys(mission.initialPerturbations)) {
        expect(allowedSet.has(key), `Disallowed initial key ${key} in ${mission.id}`).toBe(true);
      }

      // Monotonic event timestamps & whitelisted patch keys only
      let prevT = -1;
      for (const ev of mission.events) {
        expect(ev.timeS).toBeGreaterThan(prevT);
        expect(ev.timeS).toBeLessThanOrEqual(mission.durationS);
        prevT = ev.timeS;
        for (const key of Object.keys(ev.patch)) {
          expect(allowedSet.has(key), `Disallowed patch key ${key} in ${mission.id}`).toBe(true);
        }
      }

      // Evaluate across full 45s timeline
      for (const t of [0, 7.9, 8.0, 10.0, 12.0, 20.0, 24.0, 45.0]) {
        const st = evaluateMissionPerturbations(mission, t);
        expect(Object.keys(st).sort()).toEqual([...ALLOWED_PERTURBATION_KEYS].sort());
      }
    }
  });

  it("matches exact SP1_LIVEDECK_DESIGN.md §4.5 physical perturbation timings", () => {
    // M1: Cam-R +0.020944 rad (+1.2 deg) at t = 8.0 s
    const m1 = getMissionById("M1");
    expect(m1.initialAltAglM).toBe(600);
    expect(evaluateMissionPerturbations(m1, 7.9).camRBiasRad).toBe(0);
    expect(evaluateMissionPerturbations(m1, 8.0).camRBiasRad).toBeCloseTo(0.020944, 6);

    // M2: +2.5 m/s East GNSS velocity spoof ramp at t = 8.0 s
    const m2 = getMissionById("M2");
    expect(m2.initialAltAglM).toBe(1200);
    expect(evaluateMissionPerturbations(m2, 7.9).gnssSpoofVelNedMps).toEqual([0, 0, 0]);
    expect(evaluateMissionPerturbations(m2, 8.0).gnssSpoofVelNedMps).toEqual([0, 2.5, 0]);

    // M3: 3rd peer joins at t = 12.0 s; D0 -> D1_KF toggle at t = 20.0 s
    const m3 = getMissionById("M3");
    expect(evaluateMissionPerturbations(m3, 0.0).thirdPeerActive).toBe(false);
    expect(evaluateMissionPerturbations(m3, 0.0).detectorMode).toBe("D0");
    expect(evaluateMissionPerturbations(m3, 12.0).thirdPeerActive).toBe(true);
    expect(evaluateMissionPerturbations(m3, 19.9).detectorMode).toBe("D0");
    expect(evaluateMissionPerturbations(m3, 20.0).detectorMode).toBe("D1_KF");

    // M4: KSFO 28L 0.22x fog; LWIR EFVS ON at t = 10.0 s; +18 px corner bias at t = 24.0 s
    const m4 = getMissionById("M4");
    expect(evaluateMissionPerturbations(m4, 0.0).fogContrast).toBeCloseTo(0.22, 6);
    expect(evaluateMissionPerturbations(m4, 9.9).lwirEfvsEnabled).toBe(false);
    expect(evaluateMissionPerturbations(m4, 10.0).lwirEfvsEnabled).toBe(true);
    expect(evaluateMissionPerturbations(m4, 23.9).runwayCornerBiasPx).toBe(0);
    expect(evaluateMissionPerturbations(m4, 24.0).runwayCornerBiasPx).toBe(18.0);
    expect(m4.referenceRunwayWgs84HM).toBeLessThan(-22.0);
    expect(m4.referenceRunwayWgs84HM).toBeGreaterThan(-35.0);

    const ids: MissionId[] = ["M1", "M2", "M3", "M4"];
    expect(ids.every((id) => getMissionById(id).id === id)).toBe(true);
  });
});

describe("Stage-1 AI backend hierarchy, 3-slot StagingRingBuffer & peak extractor", () => {
  it("enforces explicit WebGPU-FP16 -> WebGPU-FP32 (NO F16) -> WASM-SIMD hierarchy (never WebGL EP)", () => {
    const b1 = selectAiBackend({ hasWebGpu: true, hasShaderF16: true });
    expect(b1.kind).toBe("webgpu-fp16");
    expect(b1.executionProvider).toBe("webgpu");
    expect(b1.aiBadge).toBe("AI: ORT-WebGPU-FP16");
    expect(b1.degradedReason).toBeNull();

    const b2 = selectAiBackend({ hasWebGpu: true, hasShaderF16: false });
    expect(b2.kind).toBe("webgpu-fp32");
    expect(b2.executionProvider).toBe("webgpu");
    expect(b2.aiBadge).toBe("AI: ORT-WebGPU-FP32 (NO F16)");
    expect(b2.degradedReason).not.toBeNull();

    const b3 = selectAiBackend({ hasWebGpu: false, hasShaderF16: false });
    expect(b3.kind).toBe("wasm-simd");
    expect(b3.executionProvider).toBe("wasm");
    expect(b3.aiBadge).toBe("AI: ORT-WASM-SIMD");
    expect(b3.degradedReason).not.toBeNull();
  });

  it("cycles all 3 StagingRingBuffer slots cleanly without blocking", () => {
    const ring = new StagingRingBuffer(3, 16);
    expect(ring.slotCount).toBe(3);
    const s0 = ring.acquireWriteSlot();
    const s1 = ring.acquireWriteSlot();
    const s2 = ring.acquireWriteSlot();
    expect(s0).not.toBeNull();
    expect(s1).not.toBeNull();
    expect(s2).not.toBeNull();
    ring.commitWriteSlot(s0!.slotIndex);
    ring.commitWriteSlot(s1!.slotIndex);
    ring.commitWriteSlot(s2!.slotIndex);
    expect(ring.acquireWriteSlot()).toBeNull();

    const r0 = ring.acquireReadySlot();
    expect(r0?.slotIndex).toBe(s0!.slotIndex);
    ring.releaseSlot(r0!.slotIndex);
    expect(ring.acquireWriteSlot()?.slotIndex).toBe(s0!.slotIndex);
  });

  it("extracts sub-pixel 3x3 NMS peaks and flags non-finite tensors", () => {
    const w = 16;
    const h = 16;
    const hw = w * h;
    const hm = new Float32Array(hw);
    const off = new Float32Array(2 * hw);
    const lsc = new Float32Array(2 * hw);
    const idx = 7 * w + 9;
    hm[idx] = 0.92;
    hm[idx - 1] = 0.41; // suppressed by 3x3 NMS
    off[idx] = 0.25;
    off[hw + idx] = -0.3;
    lsc[idx] = 1.1;
    lsc[hw + idx] = 1.2;

    const res = extractDetectionPeaks(hm, off, lsc, w, h, 0.35, 8);
    expect(res.allFinite).toBe(true);
    expect(res.peaks.length).toBe(1);
    expect(res.peaks[0].u).toBeCloseTo(9.25, 5);
    expect(res.peaks[0].v).toBeCloseTo(6.7, 5);

    hm[3] = Number.NaN;
    const bad = extractDetectionPeaks(hm, off, lsc, w, h, 0.35, 8);
    expect(bad.allFinite).toBe(false);
  });
});

