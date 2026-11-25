import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it, vi } from "vitest";
import {
  DEFAULT_MANIFOLD_TOGGLES,
  DEFAULT_STRESS_PRESET,
  ObservatoryStateStore,
} from "../src/ui/observatory_shell.ts";
import {
  RealSkyLabController,
  buildRangeComparisonChartSvg,
} from "../src/ui/realsky_lab_view.ts";

describe("OpenXVS v3.0 Observatory UI Shell, Real-Sky Lab Robustness & Stage-0 Isolation", () => {
  it("test_realsky_lab_handles_corrupt_drop_and_resets_temporal_state", async () => {
    const resetSpy = vi.fn();
    const ctrl = new RealSkyLabController({
      resetXvsD2TemporalState: resetSpy,
    });

    // Switching AOT clip resets ConvGRU temporal state exactly once
    ctrl.selectEncounter(0);
    expect(resetSpy).toHaveBeenCalledTimes(1);
    ctrl.selectEncounter(1);
    expect(resetSpy).toHaveBeenCalledTimes(2);

    // Dropping a 0-byte corrupt .txt file: resets temporal state, sets graceful error, never throws
    const emptyTxt = new File([], "corrupt.txt", { type: "text/plain" });
    const res1 = await ctrl.handleDroppedFile(emptyTxt);
    expect(res1.ok).toBe(false);
    expect(res1.errorMessage.length).toBeGreaterThan(5);
    expect(ctrl.getLastError()).toBe(res1.errorMessage);
    expect(resetSpy).toHaveBeenCalledTimes(3);

    // Dropping a non-empty invalid video blob with garbage bytes: degrades cleanly without throwing
    const badVideo = new File([new Uint8Array([0x00, 0x01, 0x02, 0x03])], "broken.mp4", {
      type: "video/mp4",
    });
    const res2 = await ctrl.handleDroppedFile(badVideo);
    expect(res2.ok).toBe(false);
    expect(res2.errorMessage.length).toBeGreaterThan(5);
    expect(resetSpy).toHaveBeenCalledTimes(4);

    // SVG comparison chart renders GPS Truth, CP-MSC Safe Bound, and Naive Bbox polylines
    const svg = buildRangeComparisonChartSvg("tr", [
      { frameIdx: 0, gpsTruthM: 420, cpMscSafeM: 280, naiveBboxM: 760 },
      { frameIdx: 1, gpsTruthM: 405, cpMscSafeM: 295, naiveBboxM: 740 },
      { frameIdx: 2, gpsTruthM: 390, cpMscSafeM: 305, naiveBboxM: 710 },
    ]);
    expect(svg).toContain("<svg");
    expect(svg).toContain("polyline");
  });

  it("test_stage0_manifest_excludes_realsky_and_core2k_from_initial_bundle", () => {
    const baseDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
    const mainSrc = readFileSync(resolve(baseDir, "src/main.ts"), "utf-8");
    const pass2Src = readFileSync(resolve(baseDir, "src/passes/pass2_terrain_cdlod.ts"), "utf-8");
    const realskySrc = readFileSync(resolve(baseDir, "src/ui/realsky_lab_view.ts"), "utf-8");

    // Stage-1 assets MUST NEVER be statically imported into JS chunks (must stay dynamic fetch URLs)
    expect(mainSrc).not.toMatch(/import\s+[^;]*realsky_aot_pack/);
    expect(mainSrc).not.toMatch(/import\s+[^;]*ksfo_ortho_core2k_v2/);
    expect(pass2Src).not.toMatch(/import\s+[^;]*ksfo_ortho_core2k_v2/);
    expect(realskySrc).not.toMatch(/import\s+[^;]*realsky_aot_pack/);

    // Verify lazy URL references exist cleanly in Stage-1 loaders
    expect(pass2Src).toContain("./assets/ksfo_ortho_core2k_v2.jpg");
    expect(realskySrc).toContain("./assets/realsky_aot_pack/manifest.json");
  });

  it("test_observatory_shell_defaults_tr_cbdr_and_spz_off", () => {
    const store = new ObservatoryStateStore();
    expect(store.getLocale()).toBe("tr");
    expect(store.getActiveTab()).toBe("scene");
    expect(store.getStressPreset()).toBe(DEFAULT_STRESS_PRESET);
    expect(DEFAULT_STRESS_PRESET).toBe("cbdr");
    expect(DEFAULT_MANIFOLD_TOGGLES.activeRangeCone).toBe(true);
    expect(DEFAULT_MANIFOLD_TOGGLES.sc3bfTube).toBe(true);
    expect(DEFAULT_MANIFOLD_TOGGLES.spzStress).toBe(false);

    // Switching to ForcedLand preset auto-enables ForcedLand funnel while keeping SPZ OFF
    store.setStressPreset("forcedland");
    expect(store.getManifoldToggles().forcedLandFunnel).toBe(true);
    expect(store.getManifoldToggles().spzStress).toBe(false);

    // Instant locale toggle TR <-> EN
    store.toggleLocale();
    expect(store.getLocale()).toBe("en");
    store.toggleLocale();
    expect(store.getLocale()).toBe("tr");
  });
});
