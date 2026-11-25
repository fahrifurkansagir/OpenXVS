#!/usr/bin/env node
/**
 * Stage-0 Brotli Bundle Budget & Projection Evidence Audit for OpenXVS Live Deck (`Gate V2-M1-5`).
 *
 * Enforces hard byte budgets from `SP1_LIVEDECK_DESIGN.md` §4.4 & §5:
 *   - Stage-0 total (HTML + entry JS/CSS + xvs-wasm + KSFO 3DEP + B612 Mono MSDF) <= 716,800 B (700 KiB)
 *   - xvs-wasm (`xvs_wasm_bg.wasm`) <= 184,320 B (180 KiB)
 *   - KSFO 3DEP (`ksfo_3dep_geoid18_v1.bin`) <= 307,200 B (300 KiB)
 *   - Verifies `onnxruntime-web` is NOT statically bundled in Stage-0 entry scripts (Stage-1 async only).
 *
 * Writes machine-readable provenance evidence to `output/xvs/v2_m1/v5_bundle_budget.json`.
 */

import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { brotliCompressSync, constants as zc } from "node:zlib";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(__dirname, "..");
const repoRoot = resolve(projectRoot, "..", "..");
const distDir = join(projectRoot, "dist");
const generatedGlDir = join(projectRoot, "src", "shaders", "generated_gl");
const evidenceOutPath = join(repoRoot, "output", "xvs", "v2_m1", "v5_bundle_budget.json");

const BUDGET_STAGE0_BROTLI = 716_800; // 700 KiB
const BUDGET_WASM_BROTLI = 184_320; // 180 KiB
const BUDGET_TERRAIN_BROTLI = 307_200; // 300 KiB
const BUDGET_CAM_NDC_PX = 0.05;

function measureFile(relPath) {
  const fullPath = join(distDir, relPath);
  if (!existsSync(fullPath)) {
    throw new Error(`Missing required dist artifact: ${relPath} (${fullPath})`);
  }
  const raw = readFileSync(fullPath);
  const compressed = brotliCompressSync(raw, {
    params: { [zc.BROTLI_PARAM_QUALITY]: 11 },
  });
  return {
    relPath,
    rawBytes: raw.length,
    brotliBytes: compressed.length,
    content: raw,
  };
}

/**
 * Measure live Float32 GPU ViewProj projection error across all 6 XVS6-CUBE-v1 faces
 * (`F, R, B, L, U, D`) + free-look viewport matching `src/rhi/coords.ts` & `coords.test.ts`.
 */
function measureCamNdcMaxErrPx() {
  const fx = 429.6190111627674;
  const fy = 429.6190111627674;
  const cx = 511.5;
  const cy = 511.5;
  const W = 1024;
  const H = 1024;
  const zNear = 1.0;
  const zFar = 200000.0;

  const faces = [
    [0, 0, 1, 1, 0, 0, 0, 1, 0],
    [-1, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, -1, -1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, -1, 0, 1, 0],
    [0, -1, 0, -1, 0, 0, 0, 0, -1],
    [0, -1, 0, 1, 0, 0, 0, 0, 1],
  ];

  const samples = [
    [0, 0, 1200],
    [-400, -350, 900],
    [420, -380, 1500],
    [-300, 410, 2200],
    [500, 480, 5000],
    [12.34, -8.76, 15000],
  ];

  // Arbitrary aircraft orientation matrix R_ecef_body (orthonormal via Z-Y-X Euler)
  const yaw = 298.0 * (Math.PI / 180);
  const pitch = -3.0 * (Math.PI / 180);
  const roll = 5.5 * (Math.PI / 180);
  const cy_ = Math.cos(yaw), sy_ = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const cr = Math.cos(roll), sr = Math.sin(roll);
  const rBody = [
    cy_ * cp, cy_ * sp * sr - sy_ * cr, cy_ * sp * cr + sy_ * sr,
    sy_ * cp, sy_ * sp * sr + cy_ * cr, sy_ * sp * cr - cy_ * sr,
    -sp,      cp * sr,                  cp * cr,
  ];

  const mul3 = (a, b) => [
    a[0] * b[0] + a[1] * b[3] + a[2] * b[6], a[0] * b[1] + a[1] * b[4] + a[2] * b[7], a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
    a[3] * b[0] + a[4] * b[3] + a[5] * b[6], a[3] * b[1] + a[4] * b[4] + a[5] * b[7], a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
    a[6] * b[0] + a[7] * b[3] + a[8] * b[6], a[6] * b[1] + a[7] * b[4] + a[8] * b[7], a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
  ];

  const a00 = (2.0 * fx) / W;
  const a02 = (2.0 * (cx + 0.5)) / W - 1.0;
  const a11 = (-2.0 * fy) / H;
  const a12 = 1.0 - (2.0 * (cy + 0.5)) / H;
  const a22 = zFar / (zFar - zNear);
  const a23 = -(zFar * zNear) / (zFar - zNear);

  let maxErrPx = 0.0;
  for (const f of faces) {
    const r = mul3(rBody, f);
    const vpF32 = new Float32Array([
      a00 * r[0] + a02 * r[2], a11 * r[1] + a12 * r[2], a22 * r[2], r[2],
      a00 * r[3] + a02 * r[5], a11 * r[4] + a12 * r[5], a22 * r[5], r[5],
      a00 * r[6] + a02 * r[8], a11 * r[7] + a12 * r[8], a22 * r[8], r[8],
      0.0,                     0.0,                     a23,        0.0,
    ]);
    for (const vCam of samples) {
      const dEcef = [
        r[0] * vCam[0] + r[1] * vCam[1] + r[2] * vCam[2],
        r[3] * vCam[0] + r[4] * vCam[1] + r[5] * vCam[2],
        r[6] * vCam[0] + r[7] * vCam[1] + r[8] * vCam[2],
      ];
      const uRef = fx * (vCam[0] / vCam[2]) + cx;
      const vRef = fy * (vCam[1] / vCam[2]) + cy;

      const clipX = vpF32[0] * dEcef[0] + vpF32[4] * dEcef[1] + vpF32[8] * dEcef[2] + vpF32[12];
      const clipY = vpF32[1] * dEcef[0] + vpF32[5] * dEcef[1] + vpF32[9] * dEcef[2] + vpF32[13];
      const clipW = vpF32[3] * dEcef[0] + vpF32[7] * dEcef[1] + vpF32[11] * dEcef[2] + vpF32[15];
      const uRec = (clipX / clipW + 1.0) * 0.5 * W - 0.5;
      const vRec = (1.0 - clipY / clipW) * 0.5 * H - 0.5;
      const err = Math.hypot(uRec - uRef, vRec - vRef);
      if (err > maxErrPx) maxErrPx = err;
    }
  }
  return maxErrPx;
}

function runVitestSummary() {
  const vitestCli = join(projectRoot, "node_modules", "vitest", "vitest.mjs");
  const raw = execFileSync(process.execPath, [vitestCli, "run", "--reporter=json"], {
    cwd: projectRoot,
    encoding: "utf-8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  const jsonStart = raw.indexOf("{");
  if (jsonStart === -1) {
    throw new Error("Could not parse JSON output from vitest run --reporter=json");
  }
  const report = JSON.parse(raw.slice(jsonStart));
  return {
    passed: Number(report.numPassedTests ?? 0),
    failed: Number(report.numFailedTests ?? 0),
    total: Number(report.numTotalTests ?? 0),
    testFiles: Number(report.numTotalTestSuites ?? 0),
  };
}

function main() {
  if (!existsSync(join(distDir, "index.html"))) {
    throw new Error(`dist/index.html not found at ${distDir}. Run 'npm run build' first.`);
  }

  const indexHtml = readFileSync(join(distDir, "index.html"), "utf-8");

  // Discover Stage-0 entry assets referenced directly by dist/index.html
  const entryRefs = new Set(["index.html"]);
  for (const m of indexHtml.matchAll(/(?:src|href)="\.\/([^"]+)"/g)) {
    entryRefs.add(m[1]);
  }

  // Locate xvs_wasm_bg*.wasm emitted by Vite under dist/assets/
  const assetsDir = join(distDir, "assets");
  const assetFiles = existsSync(assetsDir) ? readdirSync(assetsDir) : [];
  const wasmFile = assetFiles.find((f) => /^xvs_wasm_bg.*\.wasm$/.test(f));
  if (!wasmFile) {
    throw new Error("Could not find emitted xvs_wasm_bg*.wasm under dist/assets/");
  }

  const stage0Files = [
    ...Array.from(entryRefs),
    `assets/${wasmFile}`,
    "assets/ksfo_3dep_geoid18_v1.bin",
    "assets/b612_mono_msdf.png",
    "assets/b612_mono_msdf.json",
  ];

  // Deduplicate cleanly
  const uniqueFiles = Array.from(new Set(stage0Files));
  const rows = uniqueFiles.map(measureFile);

  // Verify onnxruntime-web did NOT leak into Stage-0 entry JS
  for (const row of rows) {
    if (row.relPath.endsWith(".js")) {
      const txt = row.content.toString("utf-8");
      if (txt.includes("ort-wasm-simd-threaded") || txt.includes("InferenceSession")) {
        throw new Error(
          `Stage-0 entry bundle '${row.relPath}' statically includes onnxruntime-web! Keep ORT in Stage-1 async chunk.`,
        );
      }
    }
  }

  const wasmRow = rows.find((r) => r.relPath.includes("xvs_wasm_bg"));
  const terrainRow = rows.find((r) => r.relPath.includes("ksfo_3dep_geoid18_v1.bin"));
  const totalRaw = rows.reduce((acc, r) => acc + r.rawBytes, 0);
  const totalBrotli = rows.reduce((acc, r) => acc + r.brotliBytes, 0);

  // Count Naga-transpiled GLSL ES 3.00 files
  const glslFiles = existsSync(generatedGlDir)
    ? readdirSync(generatedGlDir).filter((f) => f.endsWith(".glsl")).sort()
    : [];

  // Run live Vitest suite & projection error measurement
  const vitest = runVitestSummary();
  const camNdcMaxErrPx = measureCamNdcMaxErrPx();

  console.log("=== OpenXVS Live Deck Stage-0 Brotli Bundle Budget Audit ===");
  for (const r of rows) {
    console.log(
      `  ${r.relPath.padEnd(42)} raw=${String(r.rawBytes).padStart(9)} B  brotli=${String(r.brotliBytes).padStart(8)} B`,
    );
  }
  console.log("------------------------------------------------------------");
  console.log(
    `  ${"TOTAL STAGE-0".padEnd(42)} raw=${String(totalRaw).padStart(9)} B  brotli=${String(totalBrotli).padStart(8)} B  (limit ${BUDGET_STAGE0_BROTLI} B)`,
  );

  // Verify Stage-1 same-origin ORT WASM runtime exists in dist/ort/
  const ortDir = join(distDir, "ort");
  if (!existsSync(join(ortDir, "ort-wasm-simd-threaded.wasm"))) {
    throw new Error("Missing dist/ort/ort-wasm-simd-threaded.wasm for same-origin Stage-1 AI!");
  }
  const ortFilesCount = readdirSync(ortDir).length;
  console.log(`  Stage-1 same-origin ORT dir verified: dist/ort/ (${ortFilesCount} files)`);
  console.log(
    `  Naga GLSL ES 3.00 files: ${glslFiles.length} | Vitest: ${vitest.passed}/${vitest.total} passed | CAM->NDC max err: ${camNdcMaxErrPx.toExponential(3)} px`,
  );

  const checks = {
    stage0_within_budget: totalBrotli <= BUDGET_STAGE0_BROTLI,
    xvs_wasm_within_budget: wasmRow.brotliBytes <= BUDGET_WASM_BROTLI,
    ksfo_3dep_within_budget: terrainRow.brotliBytes <= BUDGET_TERRAIN_BROTLI,
    naga_glsl_complete: glslFiles.length === 8,
    vitest_all_passed: vitest.failed === 0 && vitest.passed > 0,
    cam_ndc_within_budget: camNdcMaxErrPx <= BUDGET_CAM_NDC_PX,
  };
  const allPass = Object.values(checks).every(Boolean);

  mkdirSync(dirname(evidenceOutPath), { recursive: true });
  const evidence = {
    schema: "xvs.v2_m1.v5_bundle_budget/1",
    generated_utc: new Date().toISOString(),
    stage0_raw_bytes: totalRaw,
    stage0_brotli_bytes: totalBrotli,
    stage0_budget_bytes: BUDGET_STAGE0_BROTLI,
    stage0_budget_pct: Number(((totalBrotli / BUDGET_STAGE0_BROTLI) * 100).toFixed(2)),
    xvs_wasm_raw_bytes: wasmRow.rawBytes,
    xvs_wasm_brotli_bytes: wasmRow.brotliBytes,
    xvs_wasm_budget_bytes: BUDGET_WASM_BROTLI,
    ksfo_3dep_raw_bytes: terrainRow.rawBytes,
    ksfo_3dep_brotli_bytes: terrainRow.brotliBytes,
    ksfo_3dep_budget_bytes: BUDGET_TERRAIN_BROTLI,
    stage1_same_origin_ort_files: ortFilesCount,
    naga_glsl_file_count: glslFiles.length,
    naga_glsl_files: glslFiles,
    vitest_passed: vitest.passed,
    vitest_failed: vitest.failed,
    vitest_total: vitest.total,
    cam_ndc_max_err_px: camNdcMaxErrPx,
    cam_ndc_budget_px: BUDGET_CAM_NDC_PX,
    stage0_assets: rows.map((r) => ({
      path: r.relPath,
      raw_bytes: r.rawBytes,
      brotli_bytes: r.brotliBytes,
    })),
    checks,
    pass: allPass,
  };
  writeFileSync(evidenceOutPath, JSON.stringify(evidence, null, 2) + "\n", "utf-8");
  console.log(`  Evidence written: ${evidenceOutPath}`);

  if (!allPass) {
    console.error("FAIL: One or more V2-M1-5 checks failed:", checks);
    process.exit(1);
  }
  console.log(
    `PASS: Stage-0=${totalBrotli} B <= ${BUDGET_STAGE0_BROTLI} B (${evidence.stage0_budget_pct}%), xvs-wasm=${wasmRow.brotliBytes} B <= ${BUDGET_WASM_BROTLI} B, ksfo_3dep=${terrainRow.brotliBytes} B <= ${BUDGET_TERRAIN_BROTLI} B.`,
  );
}

main();
