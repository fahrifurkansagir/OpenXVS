import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  ActiveRangeControllerTs,
  ActiveRangeModeTs,
  R_INIT_M,
  WIRE_SIZE_BYTES,
  decodeWire,
  encodeWire,
  bytesToHex,
  selectActiveSwerveTs,
  type ActiveRangeConfigTs,
} from "../src/active_range/active_range_ts.ts";
import { CbdrScenarioController } from "../src/active_range/scenario_cbdr.ts";
import { DEG_TO_RAD } from "../src/rhi/coords.ts";

interface GoldenStep {
  readonly own: {
    readonly v_ned: [number, number, number];
    readonly a_ned: [number, number, number];
    readonly heading_rad: number;
    readonly tas_mps: number;
    readonly v_stall_mps: number;
  };
  readonly meas: {
    readonly alpha_rad: number;
    readonly beta_rad: number;
    readonly size_px: number;
    readonly size_rate_px_s: number;
    readonly contrast_ratio: number;
    readonly sky_beta_ext: number;
    readonly daytime_clear: boolean;
    readonly track_age: number;
    readonly dt_s: number;
  };
  readonly out: {
    readonly seq: number;
    readonly mode: number;
    readonly active_bank_idx: number;
    readonly flags: number;
    readonly r_hat_m: number;
    readonly sigma_r_m: number;
    readonly r_safe_lower_m: number;
    readonly rho_dot_inv_s: number;
    readonly cmd_bank_rad: number;
    readonly min_sc3bf_margin: number;
    readonly fim_gain: number;
    readonly weights: [number, number, number, number, number, number];
    readonly crc32: number;
  };
  readonly wire_hex: string;
}

interface GoldenDocument {
  readonly schema: string;
  readonly config: {
    readonly r_init_m: [number, number, number, number, number, number];
    readonly sigma_bearing_rad: number;
    readonly q_hat_conformal: number;
  };
  readonly trajectories: ReadonlyArray<{
    readonly name: string;
    readonly steps: readonly GoldenStep[];
  }>;
}

describe("XVS-ActiveRange TypeScript Rust-Golden Parity & CBDR Closed-Loop Scenario", () => {
  it("test_ts_matches_rust_golden_vectors", () => {
    const goldenPath = resolve(
      dirname(fileURLToPath(import.meta.url)),
      "../../../tests/golden/active_range_vectors.json",
    );
    const doc = JSON.parse(readFileSync(goldenPath, "utf-8")) as GoldenDocument;
    expect(doc.trajectories.length).toBeGreaterThanOrEqual(8);

    const cfg: ActiveRangeConfigTs = {
      rInitM: doc.config.r_init_m,
      sigmaBearingRad: doc.config.sigma_bearing_rad,
      qHatConformal: doc.config.q_hat_conformal,
    };

    let maxAbsErr = 0.0;
    let totalSteps = 0;

    for (const traj of doc.trajectories) {
      const ctrl = new ActiveRangeControllerTs(cfg);
      for (const st of traj.steps) {
        const out = ctrl.step(
          {
            vNed: st.own.v_ned,
            aNed: st.own.a_ned,
            headingRad: st.own.heading_rad,
            tasMps: st.own.tas_mps,
            vStallMps: st.own.v_stall_mps,
          },
          {
            alphaRad: st.meas.alpha_rad,
            betaRad: st.meas.beta_rad,
            sizePx: st.meas.size_px,
            sizeRatePxS: st.meas.size_rate_px_s,
            contrastRatio: st.meas.contrast_ratio,
            skyBetaExt: st.meas.sky_beta_ext,
            daytimeClear: st.meas.daytime_clear,
            trackAge: st.meas.track_age,
            dtS: st.meas.dt_s,
          },
        );

        expect(out.seq).toBe(st.out.seq);
        expect(out.mode).toBe(st.out.mode);
        expect(out.activeBankIdx).toBe(st.out.active_bank_idx);
        expect(out.flags).toBe(st.out.flags);

        const floatPairs: ReadonlyArray<readonly [number, number]> = [
          [out.rHatM, st.out.r_hat_m],
          [out.sigmaRM, st.out.sigma_r_m],
          [out.rSafeLowerM, st.out.r_safe_lower_m],
          [out.rhoDotInvS, st.out.rho_dot_inv_s],
          [out.cmdBankRad, st.out.cmd_bank_rad],
          [out.minSc3bfMargin, st.out.min_sc3bf_margin],
          [out.fimGain, st.out.fim_gain],
          ...out.weights.map((w, idx) => [w, st.out.weights[idx]] as const),
        ];
        for (const [actual, expected] of floatPairs) {
          const err = Math.abs(actual - expected);
          if (err > maxAbsErr) maxAbsErr = err;
          expect(err).toBeLessThan(1e-5);
        }

        const wire = encodeWire(out);
        expect(wire.byteLength).toBe(WIRE_SIZE_BYTES);
        expect(out.crc32).toBe(st.out.crc32);
        expect(bytesToHex(wire)).toBe(st.wire_hex);

        const decoded = decodeWire(wire);
        expect(decoded).not.toBeNull();
        expect(decoded?.crc32).toBe(st.out.crc32);
        expect(decoded?.mode).toBe(st.out.mode);

        // Single-byte corruption must fail CRC32 check fail-closed
        const tampered = new Uint8Array(wire);
        tampered[10] ^= 0x01;
        expect(decodeWire(tampered)).toBeNull();

        totalSteps++;
      }
    }

    expect(totalSteps).toBeGreaterThan(100);
    expect(maxAbsErr).toBeLessThan(1e-5);
  });

  it("test_cbdr_scenario_collapses_range_cone_via_active_swerve", () => {
    const scenario = new CbdrScenarioController({
      initialRangeM: 1200.0,
      closingSpeedMps: 75.0,
      intruderWingspanM: 9.5,
      ownshipTasMps: 45.0,
    });

    let swerveFired = false;
    let maxAbsBankRad = 0.0;
    let finalRelErr = 1.0;
    let initialSigma = 0.0;
    let finalSigma = 0.0;

    // Step 4.0 s at 10 Hz (40 ticks)
    for (let i = 0; i < 40; i++) {
      const res = scenario.stepAutonomous(0.1);
      if (i === 0) initialSigma = res.telemetry.sigmaRM;
      if (res.telemetry.mode === ActiveRangeModeTs.ActiveSwerve) {
        swerveFired = true;
      }
      const absBank = Math.abs(res.manualBankCmdRad);
      if (absBank > maxAbsBankRad) maxAbsBankRad = absBank;
      finalSigma = res.telemetry.sigmaRM;
      finalRelErr = Math.abs(res.telemetry.rHatM - res.trueRangeM) / Math.max(res.trueRangeM, 1.0);
    }

    expect(swerveFired).toBe(true);
    expect(maxAbsBankRad).toBeLessThanOrEqual(12.0 * DEG_TO_RAD + 1e-5);
    expect(finalRelErr).toBeLessThan(0.08);
    expect(finalSigma).toBeLessThan(initialSigma * 0.25);
  });

  it("test_counterexample_in_cone_near_fov_edge_escalates_to_immediate_evade", () => {
    // D16-P1-M1 regression test: exact counterexample from @reviewer (#392)
    // Intruder at alpha_body = +46.8 deg, r_hat = 600 m, sigma_r = 300 m, r_safe_lower = 155 m,
    // rho_dot = -0.19 s^-1 (tau = 5.26 s > 4.5 s). Straight flight (phi=0) must NOT be false-safe;
    // empty feasible set must fail-closed to ImmediateEvade (±25 deg).
    const alphaBodyRad = 46.8 * DEG_TO_RAD;
    const dec = selectActiveSwerveTs({
      rBanks: [R_INIT_M[0], R_INIT_M[1], R_INIT_M[2], R_INIT_M[3], R_INIT_M[4], R_INIT_M[5]],
      weights: [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
      rHatM: 600.0,
      sigmaRM: 300.0,
      rSafeLowerM: 155.0,
      rhoDotInvS: -0.19,
      alphaDot: 0.0,
      betaDot: 0.0,
      alphaBodyRad,
      alphaRad: alphaBodyRad,
      betaRad: 0.0,
      headingRad: 0.0,
      tasMps: 35.0,
      vStallMps: 14.5,
      sigmaBearingRad: 0.00055,
      cbdrProbeActive: false,
      wasLocked: false,
      preferredSwerveSign: 1.0,
    });
    expect(dec.mode).toBe(ActiveRangeModeTs.ImmediateEvade);
    expect(Math.abs(dec.cmdBankRad)).toBeGreaterThan(0.4);
    expect(dec.sc3bfAllSafe).toBe(false);
  });
});
