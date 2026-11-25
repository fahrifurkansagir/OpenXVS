import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  createIntruderHelicopter,
  createOwnshipAircraft,
} from "../src/scene/aircraft_factory.ts";
import { computeSolarAtmosphere } from "../src/scene/atmosphere_world.ts";
import { createCockpitInteriorRig } from "../src/scene/cockpit_rig.ts";
import {
  ActiveRangeControllerTs,
  bytesToHex,
  decodeWire,
  encodeWire,
} from "../src/sim/active_range_ts.ts";
import {
  parseKsfoTerrainPack,
  sampleKsfoElevationM,
} from "../src/sim/ksfo_terrain_pack.ts";
import {
  evaluateRunwayTouchdown,
  stepManualFlight6DoF,
  type ManualFlightState,
} from "../src/sim/manual_flight_controller.ts";
import {
  StudioScenarioSim,
  computeTrochoidWaypointsEnu,
} from "../src/sim/studio_sim.ts";
import {
  STUDIO_APP_TITLE,
  STUDIO_ASSURANCE_V52_SPEC,
  STUDIO_NAV_V5_SPEC,
  STUDIO_STACK_V5_SPEC,
  STUDIO_VERSION_LABEL,
  getStudioStrings,
} from "../src/ui/i18n.ts";
import {
  NON_FLIGHT_SIM_ONLY,
  NON_FLIGHT_WATERMARK,
  StudioBridgeClient,
  stampNonFlightExport,
  type NativeSensorHealthPayload,
} from "../src/ui/bridge_client.ts";
import {
  executeNativeVideoSequence,
  getBuiltinVideoClipConfig,
  renderHilLoopbackCardsHtml,
  renderVideoSummaryCardsHtml,
  summarizeNativeSequenceFrames,
} from "../src/ui/realworld_bench.ts";
import {
  formatBearingTbcBadge,
  formatFsyncPrecisionBadge,
  formatLwirFfcContrastBadge,
  formatMultiShipCoordBadge,
  formatSectorOpticalHealth,
  formatSensorHealthBadge,
  getFormalAssuranceBadges,
} from "../src/ui/studio_hud.ts";
import { popcount12, renderFormalAssuranceStripHtml } from "../src/ui/v52_telemetry.ts";

describe("OpenXVS Studio v5.2 Core Geometry, Atmosphere, 3D Cockpit, 6-DoF Shield, 186/186 ActiveRange & Native IPC", () => {
  it("builds high-poly multi-part PBR aircraft (> 4,000 tris), helicopter (> 2,000 tris), and 3D cockpit rig", () => {
    const ownship = createOwnshipAircraft();
    const heli = createIntruderHelicopter();
    const cockpit = createCockpitInteriorRig();

    expect(ownship.triangleCount).toBeGreaterThan(4000);
    expect(heli.triangleCount).toBeGreaterThan(2000);
    expect(cockpit.group.children.length).toBeGreaterThanOrEqual(8);

    ownship.update(0.016, 1.0, false);
    ownship.update(0.016, 1.5, true);
    heli.update(0.016, 1.0);
    expect(Number.isFinite(ownship.leftProp.rotation.z)).toBe(true);
    expect(Number.isFinite(heli.mainRotor.rotation.y)).toBe(true);

    // Physical atmosphere shifts smoothly from golden hour daylight -> midnight obsidian
    const golden = computeSolarAtmosphere(16.5, "photoreal", 0.4);
    const noon = computeSolarAtmosphere(12.5, "photoreal", 0.2);
    const obs = computeSolarAtmosphere(16.5, "obsidian", 0.4);
    expect(golden.daylightFactor).toBeGreaterThan(0.8);
    expect(golden.goldenFactor).toBeGreaterThan(0.2);
    expect(noon.sunElevationDeg).toBeGreaterThan(golden.sunElevationDeg);
    expect(obs.daylightFactor).toBe(0.0);
  });

  it("parses real USGS 3DEP 256x256 ksfo_3dep_geoid18_v1.bin and samples accurate KSFO elevations", () => {
    const binPath = fileURLToPath(
      new URL("../public/assets/ksfo_3dep_geoid18_v1.bin", import.meta.url),
    );
    const raw = readFileSync(binPath);
    const ab = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
    const pack = parseKsfoTerrainPack(ab);

    expect(pack.inner.width).toBe(256);
    expect(pack.inner.height).toBe(256);
    expect(pack.outer.width).toBe(256);
    expect(pack.outer.height).toBe(256);

    const thrElev = sampleKsfoElevationM(pack, 0, 0);
    expect(Math.abs(thrElev)).toBeLessThan(8.0);
    const sanBrunoElev = sampleKsfoElevationM(pack, -5200, 7200);
    expect(sanBrunoElev).toBeGreaterThan(180.0);
    const bayElev = sampleKsfoElevationM(pack, 3500, 500);
    expect(bayElev).toBeLessThan(-2.0);
  });

  it("proves Buyer A/B Manual Flight Safety Shield ON vs OFF, crosswind drift, and KSFO 28L touchdown grading", () => {
    const baseState: ManualFlightState = {
      posX: 1200,
      posY: 220,
      posZ: 600,
      headingRad: (297.7 * Math.PI) / 180,
      pitchRad: 0.0,
      rollRad: 0.0,
      tasMps: 52.0,
      climbMps: 0.0,
      throttle: 0.65,
      engineOutRight: false,
    };

    // 1. Direct head-on CBDR intruder at 260 m ahead: Shield ON overrides; Shield OFF permits breach!
    const fwdX = Math.sin(baseState.headingRad);
    const fwdZ = -Math.cos(baseState.headingRad);
    const closeIntr = {
      x: baseState.posX + fwdX * 140.0,
      y: baseState.posY,
      z: baseState.posZ + fwdZ * 140.0,
    };
    const intrVel = { x: -fwdX * 35.0, y: 0, z: -fwdZ * 35.0 };

    const withShield = stepManualFlight6DoF({
      prev: baseState,
      input: { pitch: 0, roll: 0, yaw: 0, throttleDelta: 0 },
      dtS: 0.05,
      simTimeS: 1.0,
      crosswindKt: 12.0,
      terrainElevM: 0.0,
      intruderPos: closeIntr,
      intruderVel: intrVel,
      shieldEnabled: true,
    });
    expect(withShield.shieldOverrideActive).toBe(true);
    expect(withShield.shieldReason).toBe("SC3BF CBDR");
    expect(withShield.envelopeBreach).toBe(false);
    expect(Math.abs(withShield.state.rollRad)).toBeGreaterThan(0.05);

    const withoutShield = stepManualFlight6DoF({
      prev: baseState,
      input: { pitch: 0, roll: 0, yaw: 0, throttleDelta: 0 },
      dtS: 0.05,
      simTimeS: 1.0,
      crosswindKt: 12.0,
      terrainElevM: 0.0,
      intruderPos: closeIntr,
      intruderVel: intrVel,
      shieldEnabled: false,
    });
    expect(withoutShield.shieldOverrideActive).toBe(false);
    expect(withoutShield.envelopeBreach).toBe(true);
    expect(withoutShield.breachReason).toContain("COLLISION");

    // 2. Terrain floor protection when diving low outside final approach box
    const lowDive: ManualFlightState = { ...baseState, posX: -4000, posZ: -4000, posY: 32, climbMps: -8.0 };
    const terrainGuard = stepManualFlight6DoF({
      prev: lowDive,
      input: { pitch: -1, roll: 0, yaw: 0, throttleDelta: 0 },
      dtS: 0.05,
      simTimeS: 2.0,
      crosswindKt: 15.0,
      terrainElevM: 0.0,
      intruderPos: null,
      intruderVel: null,
      shieldEnabled: true,
    });
    expect(terrainGuard.shieldOverrideActive).toBe(true);
    expect(terrainGuard.shieldReason).toBe("TERRAIN FLOOR");

    // 3. KSFO 28L touchdown grading (KUSURSUZ vs GÜVENLİ vs SERT İNİŞ)
    const perfect = evaluateRunwayTouchdown(fwdX * 300, 1.8, fwdZ * 300, -0.8); // ~157 fpm on centerline
    expect(perfect.landed).toBe(true);
    expect(perfect.onRunway).toBe(true);
    expect(perfect.grade).toBe("KUSURSUZ");

    const hard = evaluateRunwayTouchdown(fwdX * 300, 1.8, fwdZ * 300, -4.5); // ~885 fpm hard
    expect(hard.landed).toBe(true);
    expect(hard.grade).toBe("SERT İNİŞ");
  });

  it("matches 186/186 bit-exact Rust xvs-active-range golden vectors & wire CRC32", () => {
    const goldenPath = fileURLToPath(
      new URL("../../../tests/golden/active_range_vectors.json", import.meta.url),
    );
    const doc = JSON.parse(readFileSync(goldenPath, "utf-8")) as {
      config: {
        r_init_m: [number, number, number, number, number, number];
        sigma_bearing_rad: number;
        q_hat_conformal: number;
      };
      trajectories: Array<{
        name: string;
        steps: Array<{
          own: {
            v_ned: [number, number, number];
            a_ned: [number, number, number];
            heading_rad: number;
            tas_mps: number;
            v_stall_mps: number;
          };
          meas: {
            alpha_rad: number;
            beta_rad: number;
            size_px: number;
            size_rate_px_s: number;
            contrast_ratio: number;
            sky_beta_ext: number;
            daytime_clear: boolean;
            track_age: number;
            dt_s: number;
          };
          out: {
            mode: number;
            active_bank_idx: number;
            flags: number;
            r_hat_m: number;
            sigma_r_m: number;
            r_safe_lower_m: number;
          };
          wire_hex: string;
        }>;
      }>;
    };

    let totalSteps = 0;
    for (const sc of doc.trajectories) {
      const ctrl = new ActiveRangeControllerTs({
        rInitM: doc.config.r_init_m,
        sigmaBearingRad: doc.config.sigma_bearing_rad,
        qHatConformal: doc.config.q_hat_conformal,
      });
      for (const st of sc.steps) {
        totalSteps++;
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
        expect(out.mode).toBe(st.out.mode);
        expect(out.activeBankIdx).toBe(st.out.active_bank_idx);
        expect(out.flags).toBe(st.out.flags);
        expect(Math.abs(out.rHatM - st.out.r_hat_m)).toBeLessThan(1e-4);
        expect(Math.abs(out.sigmaRM - st.out.sigma_r_m)).toBeLessThan(1e-4);
        const wire = encodeWire(out);
        expect(bytesToHex(wire)).toBe(st.wire_hex);
        expect(decodeWire(wire)).not.toBeNull();
      }
    }
    expect(totalSteps).toBe(186);
  });

  it("drives closed-loop ActiveRange CBDR swerve, 35 kt ForcedLand trochoid, TR/EN i18n, and Video Bench summary", () => {
    const sim = new StudioScenarioSim("cbdr");
    const initialSigma = sim.getSnapshot().activeRange.sigmaRM;
    let minSigma = initialSigma;
    for (let i = 0; i < 360; i++) {
      const snap = sim.step(1 / 30);
      if (snap.activeRange.sigmaRM < minSigma) minSigma = snap.activeRange.sigmaRM;
    }
    expect(minSigma).toBeLessThan(initialSigma * 0.65);

    const troch = computeTrochoidWaypointsEnu({ x: 2400, y: 360, z: 1200 }, 35.0, 51.4, 20);
    expect(troch.waypoints).toHaveLength(21);
    expect(troch.maxLateralWarpM).toBeGreaterThan(40.0);

    const tr = getStudioStrings("tr");
    const en = getStudioStrings("en");
    expect(tr.modePhotoreal).toBe("Gerçek Dünya");
    expect(en.modePhotoreal).toBe("Photorealistic");
    expect(tr.bench.runVideoBtn.length).toBeGreaterThan(4);
    expect(en.bench.runVideoBtn.length).toBeGreaterThan(4);

    const htmlTr = renderVideoSummaryCardsHtml("tr", {
      frames_processed: 60,
      detection_rate: 0.9167,
      mean_latency_ms: 44.6,
      max_confidence: 0.5485,
      active_range_locked_frames: 42,
      checkpoint_sha256: "2c9578f362f6bbbcc0f9ce1de925bf5dbce5459d75dfb387c79c810023750d0d",
      output_mp4_bytes: 135447,
      download_url: "/api/video_job/abcdef012345/download",
    });
    expect(htmlTr).toContain("91.7%");
    expect(htmlTr).toContain("44.6 ms");
    expect(htmlTr).toContain("42/60");
    expect(htmlTr).toContain("SHA 2c9578f3");

    // D16-W21-m2: channel-aware built-in clip & ActiveRange sub-note gating
    expect(getBuiltinVideoClipConfig("tr", "land")).toEqual({
      clip: "runway_approach",
      label: "1-Tık KSFO Yaklaşma Klibi",
    });
    expect(getBuiltinVideoClipConfig("en", "land")).toEqual({
      clip: "runway_approach",
      label: "1-Click KSFO Approach Clip",
    });
    expect(getBuiltinVideoClipConfig("tr", "d2").clip).toBe("aot_realsky");
    expect(getBuiltinVideoClipConfig("en", "thermal").clip).toBe("aot_realsky");
    expect(getBuiltinVideoClipConfig("en", "nav").clip).toBe("aot_realsky");

    const zeroD2Tr = renderVideoSummaryCardsHtml(
      "tr",
      {
        channel: "d2",
        frames_processed: 30,
        detection_rate: 0.9,
        mean_latency_ms: 40.0,
        max_confidence: 0.7,
        active_range_locked_frames: 0,
        checkpoint_sha256: "2c9578f362f6bbbcc0f9ce1de925bf5dbce5459d75dfb387c79c810023750d0d",
        output_mp4_bytes: 100000,
        download_url: "/api/video_job/d2zero/download",
      },
      "d2",
    );
    expect(zeroD2Tr).toContain("Pasif düz uçuş");
    expect(zeroD2Tr).not.toContain("Yalnızca D2 kanalı");

    const zeroLandEn = renderVideoSummaryCardsHtml(
      "en",
      {
        frames_processed: 30,
        detection_rate: 1.0,
        mean_latency_ms: 21.0,
        max_confidence: 0.88,
        active_range_locked_frames: 0,
        checkpoint_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        output_mp4_bytes: 100000,
        download_url: "/api/video_job/landzero/download",
      },
      "land",
    );
    expect(zeroLandEn).toContain("D2 channel only");
    expect(zeroLandEn).not.toContain("Passive straight flight");

    // Wave 20D HIL Loopback 4-card certification renderer
    const hilHtmlTr = renderHilLoopbackCardsHtml("tr", {
      pass: true,
      timestamp_utc: "2026-11-22T09:20:00+00:00",
      elapsed_ms: 14.2,
      packets_verified: 121,
      tamper_rejected: true,
      gpl_free: true,
      checks: [
        { name: "mavlink2_52001_signed_udp_loopback", pass: true, details: { crc_sig_verified: 100, tamper_cases_rejected: 3 } },
        { name: "dr06_fixed_wing_safety_and_pod_plugin_health", pass: true },
        { name: "active_range_64b_wire_and_crc32", pass: true, details: { tamper_cases_rejected: 4 } },
        { name: "flightgear_generic_udp_loopback", pass: true, details: { telemetry_packets_verified: 10 } },
      ],
    });
    expect(hilHtmlTr).toContain("4/4 PASS");
    expect(hilHtmlTr).toContain("GPL-FREE");
    expect(hilHtmlTr).toContain("100/100");
    expect(hilHtmlTr).toContain("DR-06");
    expect(hilHtmlTr).toContain("64B");
    expect(hilHtmlTr).toContain("FlightGear");
  });

  it("uses native Tauri IPC primary transport (zero mixed-content HTTP) and stamps immutable non-flight watermark", async () => {
    expect(NON_FLIGHT_SIM_ONLY).toBe(true);
    expect(NON_FLIGHT_WATERMARK).toBe(
      "SIMULATION & BENCHMARK EVALUATION ONLY — ZERO ACTUATION — NOT FOR FLIGHT",
    );

    const calls: string[] = [];
    const fakeInvoke = async <T = unknown>(
      cmd: string,
      _args?: Record<string, unknown>,
    ): Promise<T> => {
      calls.push(cmd);
      if (cmd === "native_health") {
        return {
          status: "ok",
          host_gpu_name: "Pure-Rust ORT (Zero-Python Native IPC)",
          vram_mb: 0,
          non_flight_sim_only: true,
          watermark: NON_FLIGHT_WATERMARK,
          models: { xvs_d2: {}, xvs_land: {}, xvs_thermal: {}, xvs_nav: {} },
        } as T;
      }
      if (cmd === "native_demo_tick") {
        return {
          channels: {
            eo_sky: { summary: "RUST-ORT D2", latency_ms: 4.2, checkpoint_sha256: "2c9578f3".repeat(8) },
          },
          non_flight_sim_only: true,
          watermark: NON_FLIGHT_WATERMARK,
        } as T;
      }
      if (cmd === "native_hil_connect") {
        throw new Error("EU PLD Art. 4(7) non-flight lockout");
      }
      return {} as T;
    };

    const client = new StudioBridgeClient("http://127.0.0.1:8765", fakeInvoke);
    expect(client.isNativeTauriIpc()).toBe(true);
    const health = await client.probeHealth();
    expect(health?.ok).toBe(true);
    expect(health?.transport).toBe("tauri-ipc");
    expect(health?.nonFlightSimOnly).toBe(true);
    expect(health?.watermark).toBe(NON_FLIGHT_WATERMARK);

    const tick = await client.fetchDemoTick("cbdr");
    expect(tick?.d2?.summary).toBe("RUST-ORT D2");
    expect(tick?.d2?.checkpointShaShort).toBe("2c9578f3");

    const blocked = await client.connectHilBridge({ mode: "mavlink2_udp" });
    expect(blocked).toBeNull();
    expect(client.getLastError()).toContain("EU PLD Art. 4(7)");
    expect(calls).toEqual(["native_health", "native_demo_tick", "native_hil_connect"]);

    const stamped = stampNonFlightExport({ job_id: "abc123" });
    expect(stamped.non_flight_sim_only).toBe(true);
    expect(stamped.actuator_authority).toBe(false);
    expect(stamped.watermark).toBe(NON_FLIGHT_WATERMARK);
  });

  it("enforces verified OpenXVS Studio v5.2 copy and eliminates all stale v4.5 / Faz-A strings", () => {
    expect(STUDIO_VERSION_LABEL).toBe("v5.2");
    expect(STUDIO_APP_TITLE).toBe("OpenXVS Studio v5.2");
    expect(STUDIO_STACK_V5_SPEC).toBe("6/6 Pure-Rust ORT + XFeat + LighterGlue + MultiShipCoord");
    expect(STUDIO_NAV_V5_SPEC).toBe(
      "XVS-Nav v5.0: MNV4-Hybrid-M ONNX · 13.1ms P50 · 25.3MB S1+XFeat+LighterGlue FP16 · 0-shot KSFO R@1 91.4% / R@5 96.7% · HPL <20m · 8-Arm CUSUM",
    );
    expect(STUDIO_ASSURANCE_V52_SPEC).toContain("26/26 Kani CBMC");
    expect(STUDIO_ASSURANCE_V52_SPEC).toContain("5,228");
    expect(STUDIO_ASSURANCE_V52_SPEC).toContain("Apalache N=2/N=3");
    expect(STUDIO_ASSURANCE_V52_SPEC).toContain("Cortex-M7 thumbv7em");
    expect(STUDIO_ASSURANCE_V52_SPEC).toContain("188 KB Flash / 0 B RAM");
    expect(STUDIO_ASSURANCE_V52_SPEC).toContain("Sequence-VPR (0 HMI)");
    expect(STUDIO_ASSURANCE_V52_SPEC).toContain("M-of-N TBC");

    const tr = getStudioStrings("tr");
    const en = getStudioStrings("en");
    expect(tr.brandVersion).toBe("v5.2");
    expect(en.brandVersion).toBe("v5.2");
    expect(tr.bench.title).toContain("OpenXVS Studio v5.2");
    expect(en.bench.title).toContain("OpenXVS Studio v5.2");
    expect(tr.navV5Spec).toContain("91.4%");
    expect(tr.navV5Spec).toContain("13.1ms P50");
    expect(tr.navV5Spec).toContain("25.3MB S1+XFeat+LighterGlue FP16");
    expect(en.navV5Spec).toContain("96.7%");
    expect(tr.bench.subtitle).toContain(STUDIO_STACK_V5_SPEC);
    expect(en.bench.subtitle).toContain(STUDIO_NAV_V5_SPEC);

    const auditedPaths = [
      "../index.html",
      "../src/main.ts",
      "../src/ui/i18n.ts",
      "../src/ui/bridge_client.ts",
      "../src/ui/v52_telemetry.ts",
      "../src/ui/studio_hud.ts",
      "../src/ui/realworld_bench.ts",
      "../src/ui/styles.css",
    ];
    for (const rel of auditedPaths) {
      const text = readFileSync(fileURLToPath(new URL(rel, import.meta.url)), "utf-8");
      expect(text, `${rel} must not contain stale v4.5`).not.toMatch(/v4\.5|v4_5/i);
      expect(text, `${rel} must not contain stale 81.8%`).not.toContain("81.8%");
      expect(text, `${rel} must not contain stale 94.5%`).not.toContain("94.5%");
      expect(text, `${rel} must not contain stale 30.6m`).not.toContain("30.6m");
      expect(text, `${rel} must not contain stale Faz-A 86.5%`).not.toContain("86.5%");
      expect(text, `${rel} must not contain stale Faz-A 94.9%`).not.toContain("94.9%");
      expect(text, `${rel} must not contain stale Faz-A 13.9ms`).not.toContain("13.9ms");
    }

    const indexHtml = readFileSync(fileURLToPath(new URL("../index.html", import.meta.url)), "utf-8");
    expect(indexHtml).toContain("OpenXVS Studio v5.2");
    expect(indexHtml.replace(/&amp;/g, "&")).toContain(NON_FLIGHT_WATERMARK);

    const pkg = JSON.parse(
      readFileSync(fileURLToPath(new URL("../package.json", import.meta.url)), "utf-8"),
    ) as { version: string };
    expect(pkg.version).toBe("5.2.0");
  });

  it("executes standalone Video Lab sequential Pure-Rust ORT ticks via native_video_sequence_tick", async () => {
    const seenArgs: Array<{ cmd: string; args?: Record<string, unknown> }> = [];
    const fakeInvoke = async <T = unknown>(
      cmd: string,
      args?: Record<string, unknown>,
    ): Promise<T> => {
      seenArgs.push({ cmd, args });
      if (cmd === "native_video_sequence_tick") {
        const idx = Number(args?.frameIdx ?? 0);
        return {
          channel: String(args?.channel ?? "d2"),
          model: "xvs_d2_web_fp16.onnx",
          backend: "ort-rust",
          real_weights: true,
          checkpoint_sha256: "2c9578f362f6bbbcc0f9ce1de925bf5dbce5459d75dfb387c79c810023750d0d",
          latency_ms: 9.5 + idx,
          confidence: idx === 1 ? 0.12 : 0.82,
          summary: idx >= 2 ? "D2 TRACK LOCKED" : "D2 SWERVE",
          annotated_jpeg_b64: "A".repeat(48),
          metrics: {
            total_frames: 4,
            active_range: { mode: idx >= 2 ? "Locked" : "Swerve", sigma_r_m: idx >= 2 ? 42.0 : 180.0 },
          },
          non_flight_sim_only: true,
          actuator_authority: false,
          watermark: NON_FLIGHT_WATERMARK,
        } as T;
      }
      return {} as T;
    };

    const client = new StudioBridgeClient("http://127.0.0.1:8765", fakeInvoke);
    const ticked: number[] = [];
    const res = await executeNativeVideoSequence(
      client,
      "d2",
      60,
      (rec) => ticked.push(rec.frameIdx),
      () => false,
      0,
    );

    expect(ticked).toEqual([0, 1, 2, 3]);
    expect(seenArgs.map((s) => s.cmd)).toEqual([
      "native_video_sequence_tick",
      "native_video_sequence_tick",
      "native_video_sequence_tick",
      "native_video_sequence_tick",
    ]);
    expect(seenArgs[2]?.args).toEqual({ channel: "d2", frameIdx: 2 });
    expect(res.frames).toHaveLength(4);
    expect(res.summary.frames_processed).toBe(4);
    expect(res.summary.detection_rate).toBeCloseTo(0.75, 4);
    expect(res.summary.mean_latency_ms).toBeCloseTo(11.0, 4);
    expect(res.summary.max_confidence).toBeCloseTo(0.82, 4);
    expect(res.summary.active_range_locked_frames).toBe(2);
    expect(res.summary.checkpoint_sha256).toBe(
      "2c9578f362f6bbbcc0f9ce1de925bf5dbce5459d75dfb387c79c810023750d0d",
    );

    // Channel-aware native labels and empty-sequence aggregator safety
    expect(getBuiltinVideoClipConfig("en", "d2", true).label).toContain("Pure-Rust ORT");
    expect(getBuiltinVideoClipConfig("tr", "land", true).label).toContain("KSFO");
    expect(summarizeNativeSequenceFrames("nav", []).frames_processed).toBe(0);
  });

  it("wires native_sensor_health & native_export_eval_report IPC and formats HUD badges cleanly", async () => {
    const nominalHealth: NativeSensorHealthPayload = {
      eo_blur_score: 142.5,
      eo_glare_ratio: 0.02,
      lwir_nuc_frozen: false,
      lwir_freeze_ticks: 0,
      timestamp_skew_ms: 4.2,
      q_eo_effective: 0.94,
      q_lwir_effective: 0.96,
      thermal_mode: "NOMINAL_FUSION",
      multiship_coord_enabled: true,
      f1_far91113_active: true,
      f4_ghost_hysteresis_active: true,
    };

    const fakeInvoke = async <T = unknown>(cmd: string): Promise<T> => {
      if (cmd === "native_sensor_health") {
        return nominalHealth as T;
      }
      if (cmd === "native_export_eval_report") {
        return {
          filename: "openxvs_v5_standalone_eval_report.json",
          json_content: JSON.stringify({
            version: "5.0",
            non_flight_sim_only: true,
            actuator_authority: false,
            watermark: NON_FLIGHT_WATERMARK,
          }),
          sha256: "abcdef0123456789".repeat(4),
          non_flight_sim_only: true,
          actuator_authority: false,
          watermark: NON_FLIGHT_WATERMARK,
        } as T;
      }
      return {} as T;
    };

    const client = new StudioBridgeClient("http://127.0.0.1:8765", fakeInvoke);
    const sh = await client.fetchSensorHealth();
    expect(sh).toEqual(nominalHealth);
    expect(client.getSensorHealth()).toEqual(nominalHealth);

    const nominalShBadge = formatSensorHealthBadge(sh, "en");
    expect(nominalShBadge.state).toBe("nominal");
    expect(nominalShBadge.text).toContain("EO/LWIR OK");

    const nucFrozenBadge = formatSensorHealthBadge(
      { ...nominalHealth, lwir_nuc_frozen: true, lwir_freeze_ticks: 5, q_lwir_effective: 0.0 },
      "en",
    );
    expect(nucFrozenBadge.state).toBe("alert");
    expect(nucFrozenBadge.text).toContain("NUC FREEZE");
    expect(nucFrozenBadge.text).toContain("5t");

    const skewBadge = formatSensorHealthBadge(
      { ...nominalHealth, timestamp_skew_ms: 265.0 },
      "tr",
    );
    expect(skewBadge.state).toBe("alert");
    expect(skewBadge.text).toContain("265ms");

    const degradedBadge = formatSensorHealthBadge(
      { ...nominalHealth, q_eo_effective: 0.34, thermal_mode: "EO_DEGRADED_LWIR_PRIMARY" },
      "en",
    );
    expect(degradedBadge.state).toBe("caution");
    expect(degradedBadge.text).toContain("34%");

    const coordBadge = formatMultiShipCoordBadge(sh, "en");
    expect(coordBadge.state).toBe("nominal");
    expect(coordBadge.text).toContain("F1 91.113");
    expect(coordBadge.text).toContain("F4 GHOST");

    const coordOffBadge = formatMultiShipCoordBadge(
      { ...nominalHealth, multiship_coord_enabled: false },
      "en",
    );
    expect(coordOffBadge.state).toBe("caution");
    expect(coordOffBadge.text).toContain("OFF");

    const report = await client.exportNativeEvalReport();
    expect(report).not.toBeNull();
    expect(report?.filename).toBe("openxvs_v5_standalone_eval_report.json");
    expect(report?.sha256).toHaveLength(64);
    expect(report?.non_flight_sim_only).toBe(true);
    expect(report?.actuator_authority).toBe(false);
    expect(report?.watermark).toBe(NON_FLIGHT_WATERMARK);
  });

  it("formats v5.2 12-Sector EO optical health and fails closed cleanly on missing/NaN fields", () => {
    expect(popcount12(0x000)).toBe(0);
    expect(popcount12(0x005)).toBe(2);
    expect(popcount12(0x0fff)).toBe(12);
    expect(popcount12(Number.NaN)).toBe(12);
    expect(popcount12(-1)).toBe(12);

    const standby = formatSectorOpticalHealth(null, "en");
    expect(standby.state).toBe("standby");
    expect(standby.sectors).toHaveLength(12);
    expect(standby.sectorBands.every((b) => b === "standby")).toBe(true);

    const baseHealth: NativeSensorHealthPayload = {
      eo_blur_score: 165.0,
      eo_glare_ratio: 0.01,
      lwir_nuc_frozen: false,
      lwir_freeze_ticks: 0,
      timestamp_skew_ms: 0.45,
      q_eo_effective: 0.96,
      q_lwir_effective: 0.95,
      thermal_mode: "NOMINAL_FUSION",
      multiship_coord_enabled: true,
      f1_far91113_active: true,
      f4_ghost_hysteresis_active: true,
    };

    // Legacy v5.0 payload (no additive v5.2 sector fields) falls back cleanly to q_eo_effective
    const legacyOk = formatSectorOpticalHealth(baseHealth, "tr");
    expect(legacyOk.legacyFallback).toBe(true);
    expect(legacyOk.failedClosed).toBe(false);
    expect(legacyOk.state).toBe("nominal");
    expect(legacyOk.minSectorQ).toBeCloseTo(0.96, 4);
    expect(legacyOk.degradedCount).toBe(0);
    expect(legacyOk.text).toContain("min %96");

    // Legacy payload with NaN q_eo_effective fails closed to alert
    const legacyNan = formatSectorOpticalHealth({ ...baseHealth, q_eo_effective: Number.NaN }, "en");
    expect(legacyNan.failedClosed).toBe(true);
    expect(legacyNan.state).toBe("alert");
    expect(legacyNan.degradedCount).toBe(12);

    // Full v5.2 nominal 12-sector map
    const nominalV52 = formatSectorOpticalHealth(
      {
        ...baseHealth,
        sector_q_eo: [0.95, 0.92, 0.94, 0.91, 0.96, 0.93, 0.9, 0.88, 0.94, 0.95, 0.91, 0.93],
        min_sector_q_eo: 0.88,
        degraded_sector_mask: 0,
      },
      "en",
    );
    expect(nominalV52.state).toBe("nominal");
    expect(nominalV52.minSectorQ).toBeCloseTo(0.88, 4);
    expect(nominalV52.degradedCount).toBe(0);
    expect(nominalV52.text).toContain("min 88%");
    expect(nominalV52.text).toContain("0/12");

    // Localized sun-glint/raindrop blur in 2 sectors even with high global q_eo_effective = 0.96
    const sunGlintPayload: NativeSensorHealthPayload = {
      ...baseHealth,
      sector_q_eo: [0.96, 0.34, 0.95, 0.95, 0.42, 0.94, 0.96, 0.95, 0.94, 0.96, 0.95, 0.96],
      min_sector_q_eo: 0.34,
      degraded_sector_mask: (1 << 1) | (1 << 4),
    };
    const glintSector = formatSectorOpticalHealth(sunGlintPayload, "en");
    expect(glintSector.state).toBe("caution");
    expect(glintSector.degradedCount).toBe(2);
    expect(glintSector.minSectorQ).toBeCloseTo(0.34, 4);
    expect(glintSector.sectorBands[1]).toBe("caution");
    expect(glintSector.sectorBands[4]).toBe("caution");
    expect(glintSector.sectorBands[0]).toBe("nominal");
    expect(glintSector.text).toContain("2/12 DEG");

    // Top-level sensor health HUD badge also catches localized sector degradation
    const topGlintBadge = formatSensorHealthBadge(sunGlintPayload, "tr");
    expect(topGlintBadge.state).toBe("caution");
    expect(topGlintBadge.text).toContain("SEKTÖR");
    expect(topGlintBadge.text).toContain("2/12");

    // Fail-closed on NaN sector entry, NaN min_sector_q_eo, or invalid degraded_sector_mask
    const nanSector = formatSectorOpticalHealth(
      {
        ...baseHealth,
        sector_q_eo: [0.95, Number.NaN, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
        min_sector_q_eo: 0.95,
        degraded_sector_mask: 0,
      },
      "en",
    );
    expect(nanSector.failedClosed).toBe(true);
    expect(nanSector.state).toBe("alert");
    expect(nanSector.minSectorQ).toBe(0);
    expect(nanSector.degradedCount).toBeGreaterThanOrEqual(1);
    expect(nanSector.sectorBands[1]).toBe("alert");

    const nanMask = formatSectorOpticalHealth(
      { ...baseHealth, min_sector_q_eo: Number.NaN, degraded_sector_mask: Number.POSITIVE_INFINITY },
      "tr",
    );
    expect(nanMask.failedClosed).toBe(true);
    expect(nanMask.state).toBe("alert");
    expect(nanMask.degradedCount).toBe(12);
    expect(nanMask.text).toContain("HATALI");
  });

  it("formats v5.2 LWIR FFC / Contrast-Collapse & Dual-Tier FSYNC <=1.0ms guards with fail-closed safety", () => {
    const baseHealth: NativeSensorHealthPayload = {
      eo_blur_score: 150.0,
      eo_glare_ratio: 0.02,
      lwir_nuc_frozen: false,
      lwir_freeze_ticks: 0,
      timestamp_skew_ms: 0.35,
      q_eo_effective: 0.94,
      q_lwir_effective: 0.95,
      thermal_mode: "NOMINAL_FUSION",
      multiship_coord_enabled: true,
      f1_far91113_active: true,
      f4_ghost_hysteresis_active: true,
    };

    // LWIR standby & legacy nominal
    expect(formatLwirFfcContrastBadge(null, "en").state).toBe("standby");
    const lwirLegacy = formatLwirFfcContrastBadge(baseHealth, "en");
    expect(lwirLegacy.state).toBe("nominal");
    expect(lwirLegacy.text).toBe("LWIR FFC OK");

    // LWIR nominal with dynamic range
    const lwirNom = formatLwirFfcContrastBadge(
      {
        ...baseHealth,
        lwir_ffc_imminent: false,
        lwir_ffc_in_progress: false,
        lwir_dynamic_range_p95_p05: 138.4,
        lwir_contrast_collapsed: false,
      },
      "en",
    );
    expect(lwirNom.state).toBe("nominal");
    expect(lwirNom.text).toContain("Δ138 DN");

    // LWIR FFC imminent -> caution
    const lwirImm = formatLwirFfcContrastBadge({ ...baseHealth, lwir_ffc_imminent: true }, "tr");
    expect(lwirImm.state).toBe("caution");
    expect(lwirImm.text).toContain("FFC YAKIN");

    // LWIR FFC in-progress -> alert (+ top badge alert)
    const ffcActivePayload: NativeSensorHealthPayload = { ...baseHealth, lwir_ffc_in_progress: true };
    const lwirAct = formatLwirFfcContrastBadge(ffcActivePayload, "en");
    expect(lwirAct.state).toBe("alert");
    expect(lwirAct.text).toContain("FFC ACTIVE");
    expect(formatSensorHealthBadge(ffcActivePayload, "en").state).toBe("alert");
    expect(formatSensorHealthBadge(ffcActivePayload, "en").text).toContain("FFC MASK");

    // LWIR alias `lwir_ffc_flag` (bitmask & NaN fail-closed)
    expect(formatLwirFfcContrastBadge({ ...baseHealth, lwir_ffc_flag: 1 }, "en").state).toBe("caution");
    expect(formatLwirFfcContrastBadge({ ...baseHealth, lwir_ffc_flag: 2 }, "en").state).toBe("alert");
    const nanFfcFlag = formatLwirFfcContrastBadge({ ...baseHealth, lwir_ffc_flag: Number.NaN }, "en");
    expect(nanFfcFlag.failedClosed).toBe(true);
    expect(nanFfcFlag.state).toBe("alert");

    // LWIR p95-p05 contrast collapse (< 10 DN) even when NUC is not frozen
    const contrastPayload: NativeSensorHealthPayload = {
      ...baseHealth,
      lwir_dynamic_range_p95_p05: 4.8,
      lwir_contrast_collapsed: true,
    };
    const lwirDrop = formatLwirFfcContrastBadge(contrastPayload, "tr");
    expect(lwirDrop.state).toBe("alert");
    expect(lwirDrop.contrastCollapsed).toBe(true);
    expect(lwirDrop.text).toContain("KONTRAST");
    expect(formatSensorHealthBadge(contrastPayload, "en").state).toBe("alert");

    // NaN dynamic range fails closed to alert
    const lwirNanDr = formatLwirFfcContrastBadge(
      { ...baseHealth, lwir_dynamic_range_p95_p05: Number.NaN },
      "en",
    );
    expect(lwirNanDr.failedClosed).toBe(true);
    expect(lwirNanDr.state).toBe("alert");
    expect(lwirNanDr.contrastCollapsed).toBe(true);

    // Dual-Tier FSYNC <= 1.0 ms precision vs > 1.0 ms warn vs > 200 ms hard breaker
    expect(formatFsyncPrecisionBadge(null, "tr").state).toBe("standby");

    const fsyncOk = formatFsyncPrecisionBadge(
      { ...baseHealth, timestamp_skew_ms: 0.42, fsync_precision_ok: true, timestamp_monotonic_ok: true },
      "en",
    );
    expect(fsyncOk.state).toBe("nominal");
    expect(fsyncOk.fsyncPrecisionOk).toBe(true);
    expect(fsyncOk.text).toBe("FSYNC <=1ms · 0.42ms");

    const fsyncWarn = formatFsyncPrecisionBadge(
      { ...baseHealth, timestamp_skew_ms: 4.5, fsync_precision_ok: false, timestamp_monotonic_ok: true },
      "en",
    );
    expect(fsyncWarn.state).toBe("caution");
    expect(fsyncWarn.fsyncPrecisionOk).toBe(false);
    expect(fsyncWarn.text).toContain(">1ms");

    // Contradictory fsync_precision_ok=true when skew=3.2ms (> 1.0ms) fails closed to caution
    const fsyncMismatch = formatFsyncPrecisionBadge(
      { ...baseHealth, timestamp_skew_ms: 3.2, fsync_precision_ok: true },
      "en",
    );
    expect(fsyncMismatch.state).toBe("caution");
    expect(fsyncMismatch.fsyncPrecisionOk).toBe(false);

    // Hard skew breaker (> 200ms) or non-monotonic clock -> alert
    const fsyncHard = formatFsyncPrecisionBadge(
      { ...baseHealth, timestamp_skew_ms: 245.0, timestamp_skew_fault: true, fsync_precision_ok: false },
      "tr",
    );
    expect(fsyncHard.state).toBe("alert");
    expect(fsyncHard.hardSkewFault).toBe(true);
    expect(fsyncHard.text).toContain("KESİCİ");

    // NaN / negative skew fails closed cleanly
    const fsyncNan = formatFsyncPrecisionBadge({ ...baseHealth, timestamp_skew_ms: Number.NaN }, "en");
    expect(fsyncNan.failedClosed).toBe(true);
    expect(fsyncNan.hardSkewFault).toBe(true);
    expect(fsyncNan.state).toBe("alert");
    expect(formatSensorHealthBadge({ ...baseHealth, timestamp_skew_ms: Number.NaN }, "en").state).toBe("alert");
  });

  it("formats M-of-N Bearing TBC status, verified v5.2 Formal/Bare-Metal Assurance badges, and full v5.2 IPC payload", async () => {
    const fullV52Health: NativeSensorHealthPayload = {
      eo_blur_score: 188.4,
      eo_glare_ratio: 0.01,
      lwir_nuc_frozen: false,
      lwir_freeze_ticks: 0,
      timestamp_skew_ms: 0.38,
      q_eo_effective: 0.95,
      q_lwir_effective: 0.96,
      thermal_mode: "NOMINAL_FUSION",
      multiship_coord_enabled: true,
      f1_far91113_active: true,
      f4_ghost_hysteresis_active: true,
      eo_occlusion_ratio: 0.0,
      eo_focus_quality: 0.97,
      grid_4x4_q_eo: Array<number>(16).fill(0.95),
      sector_q_eo: Array<number>(12).fill(0.94),
      min_sector_q_eo: 0.94,
      degraded_sector_mask: 0,
      lwir_ffc_imminent: false,
      lwir_ffc_in_progress: false,
      lwir_dynamic_range_p95_p05: 145.0,
      lwir_contrast_collapsed: false,
      timestamp_monotonic_ok: true,
      timestamp_skew_fault: false,
      fsync_precision_ok: true,
      bearing_track_confirmed: true,
      bearing_track_hits_in_window: 4,
      ring0_authority_inhibited: false,
      last_swerve_role: "OPTICAL_FAR91113_RIGHT",
      non_flight_sim_only: true,
      actuator_authority: false,
      watermark: NON_FLIGHT_WATERMARK,
    };

    const fakeInvoke = async <T = unknown>(cmd: string): Promise<T> => {
      if (cmd === "native_health") {
        return {
          version: "5.2.0",
          status: "ok",
          host_gpu_name: "Pure-Rust ORT v5.2",
          vram_mb: 0,
          non_flight_sim_only: true,
          watermark: NON_FLIGHT_WATERMARK,
          models: { xvs_d2: {}, xvs_land: {}, xvs_thermal: {}, xvs_nav: {} },
        } as T;
      }
      if (cmd === "native_sensor_health") {
        return fullV52Health as T;
      }
      return {} as T;
    };

    const client = new StudioBridgeClient("http://127.0.0.1:8765", fakeInvoke);
    const h = await client.probeHealth();
    expect(h?.version).toBe("5.2.0");
    const sh = await client.fetchSensorHealth();
    expect(sh?.min_sector_q_eo).toBe(0.94);
    expect(sh?.fsync_precision_ok).toBe(true);
    expect(sh?.bearing_track_confirmed).toBe(true);

    // M-of-N Bearing TBC states
    expect(formatBearingTbcBadge(null, "en").state).toBe("standby");
    const tbcConfirmed = formatBearingTbcBadge(fullV52Health, "en");
    expect(tbcConfirmed.state).toBe("nominal");
    expect(tbcConfirmed.confirmed).toBe(true);
    expect(tbcConfirmed.text).toContain("CONF 4/5");

    const tbcTent = formatBearingTbcBadge(
      { ...fullV52Health, bearing_track_confirmed: false, bearing_track_hits_in_window: 2 },
      "tr",
    );
    expect(tbcTent.state).toBe("caution");
    expect(tbcTent.text).toContain("ADAY 2/5");

    const tbcIdle = formatBearingTbcBadge(
      { ...fullV52Health, bearing_track_confirmed: false, bearing_track_hits_in_window: 0 },
      "en",
    );
    expect(tbcIdle.state).toBe("standby");
    expect(tbcIdle.text).toContain("IDLE 0/5");

    const tbcMulti = formatBearingTbcBadge(
      { ...fullV52Health, bearing_tbc_confirmed_tracks: 2 },
      "en",
    );
    expect(tbcMulti.state).toBe("nominal");
    expect(tbcMulti.text).toContain("2 CONF");

    const tbcNan = formatBearingTbcBadge(
      { ...fullV52Health, bearing_track_hits_in_window: Number.NaN },
      "en",
    );
    expect(tbcNan.failedClosed).toBe(true);
    expect(tbcNan.state).toBe("alert");

    // Verified Formal & Bare-Metal Assurance badges (TR & EN)
    const badgesEn = getFormalAssuranceBadges("en");
    const badgesTr = getFormalAssuranceBadges("tr");
    expect(badgesEn).toHaveLength(4);
    expect(badgesTr).toHaveLength(4);
    const stripEn = renderFormalAssuranceStripHtml("en");
    const stripTr = renderFormalAssuranceStripHtml("tr");
    expect(stripEn).toContain("26/26 Kani CBMC");
    expect(stripEn).toContain("5,228");
    expect(stripEn).toContain("Apalache N=2/N=3");
    expect(stripEn).toContain("Cortex-M7 thumbv7em");
    expect(stripEn).toContain("188KB/0B");
    expect(stripEn).toContain("Sequence-VPR · 0 HMI");
    expect(stripTr).toContain("26/26 Kani CBMC");
    expect(stripTr).toContain("Apalache N=2/N=3");
    expect(stripTr).toContain("Cortex-M7 thumbv7em");
    expect(stripTr).toContain("Sequence-VPR · 0 HMI");
  });
});
