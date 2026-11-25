import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  ForcedLandScenarioController,
  KSFO_28L_THRESHOLD,
  computeWindWarpedTrochoidPath,
} from "../src/active_range/scenario_forcedland.ts";
import {
  ARCH_NODE_IDS,
  DEFAULT_LOCALE,
  getObservatoryStrings,
} from "../src/ui/i18n.ts";
import {
  ARCH_FLOW_EDGES,
  DEFAULT_TRAINED_PROVENANCE,
  extractTrainedProvenanceFromRegistry,
  renderArchMapSvgMarkup,
  renderModelProvenanceStripHtml,
} from "../src/ui/arch_map_view.ts";
import {
  D2_FRAME_FLOATS,
  OrtDetectorRunner,
  downsample256To64,
  estimateSadTranslation64,
  rgbaToGray256,
  shiftPlane256,
} from "../src/ai/ort_worker.ts";
import { DEG_TO_RAD, geodeticToEcef } from "../src/rhi/coords.ts";

describe("Wave 14A Observatory Part 1: ForcedLand Trochoid, TR/EN i18n, Arch Map & ORT GMC", () => {
  it("test_forcedland_trochoid_warps_with_35kt_crosswind", () => {
    const startLatRad = 37.592 * DEG_TO_RAD;
    const startLonRad = -122.312 * DEG_TO_RAD;
    const startAltM = 540.0;

    const calmPath = computeWindWarpedTrochoidPath({
      latRad: startLatRad,
      lonRad: startLonRad,
      altWgs84M: startAltM,
      crosswindKt: 0.0,
      numSegments: 16,
    });

    const windyPath = computeWindWarpedTrochoidPath({
      latRad: startLatRad,
      lonRad: startLonRad,
      altWgs84M: startAltM,
      crosswindKt: 35.0,
      numSegments: 16,
    });

    expect(calmPath.waypoints).toHaveLength(17);
    expect(windyPath.waypoints).toHaveLength(17);

    // Mid-glide waypoint must show clear non-zero lateral wind warp under 35 kt crosswind
    const midIdx = 8;
    const cMid = calmPath.waypoints[midIdx].posEcef;
    const wMid = windyPath.waypoints[midIdx].posEcef;
    const midShiftM = Math.hypot(
      wMid[0] - cMid[0],
      wMid[1] - cMid[1],
      wMid[2] - cMid[2],
    );
    expect(midShiftM).toBeGreaterThan(45.0);
    expect(windyPath.maxLateralWindWarpM).toBeGreaterThan(45.0);
    expect(calmPath.maxLateralWindWarpM).toBeLessThan(1e-3);

    // Final waypoint terminates at KSFO 28L threshold (< 1.5 m)
    const thrEcef = geodeticToEcef(
      KSFO_28L_THRESHOLD.latDeg * DEG_TO_RAD,
      KSFO_28L_THRESHOLD.lonDeg * DEG_TO_RAD,
      KSFO_28L_THRESHOLD.altWgs84M,
    );
    const lastWindy = windyPath.waypoints[windyPath.waypoints.length - 1].posEcef;
    const errEndM = Math.hypot(
      lastWindy[0] - thrEcef[0],
      lastWindy[1] - thrEcef[1],
      lastWindy[2] - thrEcef[2],
    );
    expect(errEndM).toBeLessThan(1.5);

    // Controller produces finite WASM manual_override bank/climb commands + Pass 3 funnel state
    const ctrl = new ForcedLandScenarioController(35.0);
    const out = ctrl.step({
      latRad: startLatRad,
      lonRad: startLonRad,
      altWgs84M: startAltM,
      yawRad: 298.0 * DEG_TO_RAD,
      airspeedMps: 52.0,
    });
    expect(out.manualOverride).toBe(true);
    expect(Number.isFinite(out.manualBankCmdRad)).toBe(true);
    expect(Math.abs(out.manualBankCmdRad)).toBeLessThanOrEqual(25.0 * DEG_TO_RAD + 1e-6);
    expect(Number.isFinite(out.manualClimbCmdMps)).toBe(true);
    expect(out.manualClimbCmdMps).toBeLessThan(0.0);
    expect(out.funnelState.active).toBe(true);
    expect(out.funnelState.waypoints?.length).toBe(17);
    expect(out.diagnostics.crosswindKt).toBe(35.0);
  });

  it("test_i18n_tr_en_toggle_covers_all_tabs_and_nodes", () => {
    expect(DEFAULT_LOCALE).toBe("tr");
    expect(ARCH_NODE_IDS).toHaveLength(8);

    const tr = getObservatoryStrings("tr");
    const en = getObservatoryStrings("en");

    // All 3 top-level tabs present & non-empty in both locales
    for (const tabKey of ["scene", "arch", "realsky"] as const) {
      expect(tr.tabs[tabKey].trim().length).toBeGreaterThan(4);
      expect(en.tabs[tabKey].trim().length).toBeGreaterThan(4);
      expect(tr.tabs[tabKey]).not.toBe(en.tabs[tabKey]);
    }

    // All 4 stress presets and 4 manifold toggles covered
    for (const pKey of ["cbdr", "forcedland", "spoof", "normal"] as const) {
      expect(tr.stressPresets[pKey].label.trim().length).toBeGreaterThan(3);
      expect(en.stressPresets[pKey].label.trim().length).toBeGreaterThan(3);
    }
    for (const mKey of ["activeRangeCone", "sc3bfTube", "forcedLandFunnel", "spzStress"] as const) {
      expect(tr.manifolds[mKey].trim().length).toBeGreaterThan(3);
      expect(en.manifolds[mKey].trim().length).toBeGreaterThan(3);
    }

    // All 8 Tab 2 architecture nodes have complete title, badge, whatItDoes, ifMissing & repoPaths
    for (const id of ARCH_NODE_IDS) {
      for (const dict of [tr, en]) {
        const node = dict.archNodes[id];
        expect(node.title.trim().length).toBeGreaterThan(2);
        expect(node.badge.trim().length).toBeGreaterThan(2);
        expect(node.whatItDoes.trim().length).toBeGreaterThan(15);
        expect(node.ifMissing.trim().length).toBeGreaterThan(15);
        expect(node.repoPaths.length).toBeGreaterThan(0);
      }
    }

    // Arch map SVG renders all 8 nodes & flow edges in both TR and EN
    const svgTr = renderArchMapSvgMarkup("tr", "xvs_activerange", {
      fps: 60,
      detectorHz: 15,
      detectorMs: 6.2,
      kaniPassed: 18,
      kaniTotal: 18,
      activeRangeMode: "ACTIVE_SWERVE",
      forcedLandActive: false,
      halHealthy: true,
    });
    const svgEn = renderArchMapSvgMarkup("en", "xvs_d2", {
      fps: 60,
      detectorHz: 15,
      detectorMs: 6.2,
      kaniPassed: 18,
      kaniTotal: 18,
      activeRangeMode: "NOMINAL_HOLD",
      forcedLandActive: true,
      halHealthy: true,
    });
    expect(ARCH_FLOW_EDGES.length).toBeGreaterThanOrEqual(7);
    for (const id of ARCH_NODE_IDS) {
      expect(svgTr).toContain(`data-arch-node="${id}"`);
      expect(svgEn).toContain(`data-arch-node="${id}"`);
    }
    expect(svgTr).toContain("18/18");
  });

  it("test_ort_worker_sad_translation_and_temporal_reset", () => {
    // Build a synthetic 256x256 textured sky + bright cloud pattern and shift it by (+12, -8) px
    const prev256 = new Float32Array(D2_FRAME_FLOATS);
    for (let y = 0; y < 256; y++) {
      for (let x = 0; x < 256; x++) {
        const blob1 = Math.exp(-(((x - 96) ** 2 + (y - 112) ** 2) / (2 * 22 * 22))) * 160;
        const blob2 = Math.exp(-(((x - 168) ** 2 + (y - 148) ** 2) / (2 * 18 * 18))) * 140;
        const grad = 40 + 0.15 * x + 0.1 * y;
        prev256[y * 256 + x] = grad + blob1 + blob2;
      }
    }
    const trueDx = 12;
    const trueDy = -8;
    const curr256 = shiftPlane256(prev256, trueDx, trueDy);

    const prev64 = downsample256To64(prev256);
    const curr64 = downsample256To64(curr256);
    const est = estimateSadTranslation64(prev64, curr64, 6);

    // 64x64 downsampled SAD recovers the 12, -8 px shift cleanly within 4 px (1 coarse cell)
    expect(Math.abs(est.dxPx - trueDx)).toBeLessThanOrEqual(4);
    expect(Math.abs(est.dyPx - trueDy)).toBeLessThanOrEqual(4);

    // Verify RGBA -> 256x256 grayscale preserves Rec.709 luminance accurately
    const rgba = new Uint8ClampedArray(256 * 256 * 4);
    for (let i = 0; i < 256 * 256; i++) {
      rgba[i * 4 + 0] = 100;
      rgba[i * 4 + 1] = 150;
      rgba[i * 4 + 2] = 200;
      rgba[i * 4 + 3] = 255;
    }
    const gray = rgbaToGray256(rgba, 256, 256);
    expect(gray.length).toBe(D2_FRAME_FLOATS);
    expect(Math.abs(gray[0] - (0.2126 * 100 + 0.7152 * 150 + 0.0722 * 200))).toBeLessThan(0.5);

    // Verify OrtDetectorRunner.resetXvsD2TemporalState zeroes hidden state & clears ring cleanly
    const runner = new OrtDetectorRunner();
    const before = runner.getXvsD2TemporalStateSnapshot();
    expect(before.resetCount).toBe(0);
    runner.resetXvsD2TemporalState();
    const after = runner.getXvsD2TemporalStateSnapshot();
    expect(after.resetCount).toBe(1);
    expect(after.ringLength).toBe(0);
    expect(after.hiddenNorm).toBe(0);
    expect(after.hasPrev64).toBe(false);
  });

  it("test_arch_map_renders_verified_colab_l4_provenance_badges", () => {
    const registryPath = resolve(__dirname, "../public/models/registry.json");
    const rawRegistry = JSON.parse(readFileSync(registryPath, "utf-8")) as unknown;
    const extracted = extractTrainedProvenanceFromRegistry(rawRegistry);
    expect(extracted.hydratedFromRegistry).toBe(true);
    expect(extracted.l4TensorsChecked).toBe(336);
    expect(extracted.l4NonFiniteTensors).toBe(0);

    const hydrated = { ...DEFAULT_TRAINED_PROVENANCE, ...extracted };
    const htmlTr = renderModelProvenanceStripHtml("tr", hydrated);
    const htmlEn = renderModelProvenanceStripHtml("en", hydrated);

    expect(htmlTr).toContain("Eğitilmiş Model Provenansı (Gerçek GPU)");
    expect(htmlEn).toContain("Trained Model Provenance (Real GPU)");

    for (const html of [htmlTr, htmlEn]) {
      expect(html).toContain('data-hydrated="true"');
      expect(html).toContain("XVS-D2");
      expect(html).toContain("Colab G4 · AP50 0.493 · Pd 78.0%");
      expect(html).toContain("XVS-Nav");
      expect(html).toContain("Colab L4 · R@1 81.8% · R@5 94.5%");
      expect(html).toContain("XVS-Land");
      expect(html).toContain("Colab L4 · SimCC 3.41px · HMI 0");
      expect(html).toContain("XVS-Thermal");
      expect(html).toContain("Colab L4 · EASA 5/5 · Vacuity 1.00");
      expect(html).toContain("XVS-World");
      expect(html).toContain("Colab A100 · PSNR 56.6 dB");
      expect(html).toContain("336/336");
      expect(html).toContain("46b9364b");
    }
  });
});
