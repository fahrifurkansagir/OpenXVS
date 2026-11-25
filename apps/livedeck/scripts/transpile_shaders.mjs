#!/usr/bin/env node
/**
 * Compile-time dual shader pipeline using host `naga-cli` (v30.0.2).
 *
 * - Transpiles vertex & fragment passes (pass1, pass2, pass3, pass5) from WGSL into
 *   WebGL2 GLSL ES 3.00 (`src/shaders/generated_gl/*.{vert,frag}.glsl`) with Naga's
 *   default coordinate-space adjustment (`ADJUST_COORDINATE_SPACE`, mapping WebGPU
 *   z in [0, 1] -> WebGL2 z in [-1, 1]).
 * - Validates the WebGPU compute shader (`pass4_ai_roi.wgsl`) with `naga` (0 errors).
 */

import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(__dirname, "..");
const shadersDir = join(projectRoot, "src", "shaders");
const outDir = join(shadersDir, "generated_gl");

function resolveNagaBinary() {
  const candidates = [
    process.env.NAGA_CLI,
    "naga",
    join(homedir(), ".cargo", "bin", process.platform === "win32" ? "naga.exe" : "naga"),
  ].filter(Boolean);

  for (const bin of candidates) {
    try {
      const version = execFileSync(bin, ["--version"], {
        encoding: "utf-8",
        stdio: ["ignore", "pipe", "pipe"],
      }).trim();
      return { bin, version };
    } catch {
      // Try next candidate
    }
  }
  throw new Error(
    "naga-cli binary not found on PATH or in ~/.cargo/bin. Install naga-cli (v30.0.2) first.",
  );
}

const RENDER_PASSES = [
  "pass1_das_efvs",
  "pass2_terrain_cdlod",
  "pass3_airframe_traffic",
  "pass5_sdf_msdf_hud",
];

const COMPUTE_PASSES = ["pass4_ai_roi"];

function main() {
  const { bin, version } = resolveNagaBinary();
  mkdirSync(outDir, { recursive: true });

  console.log(`[shaders:build] Using ${bin} (version ${version})`);

  const summary = [];

  for (const name of RENDER_PASSES) {
    const wgslPath = join(shadersDir, `${name}.wgsl`);
    if (!existsSync(wgslPath)) {
      throw new Error(`Missing WGSL source: ${wgslPath}`);
    }

    // naga requires .vert / .frag extensions to infer GLSL output stage
    const tmpVert = join(outDir, `${name}.vert`);
    const tmpFrag = join(outDir, `${name}.frag`);
    const finalVert = join(outDir, `${name}.vert.glsl`);
    const finalFrag = join(outDir, `${name}.frag.glsl`);

    try {
      execFileSync(bin, ["--profile", "es300", "--entry-point", "vs_main", wgslPath, tmpVert], {
        stdio: "inherit",
      });
      execFileSync(bin, ["--profile", "es300", "--entry-point", "fs_main", wgslPath, tmpFrag], {
        stdio: "inherit",
      });

      const vertSrc = readFileSync(tmpVert, "utf-8");
      const fragSrc = readFileSync(tmpFrag, "utf-8");

      if (!vertSrc.includes("gl_Position.z * 2.0 - gl_Position.w")) {
        throw new Error(
          `Expected Naga ADJUST_COORDINATE_SPACE z remap in ${name}.vert.glsl`,
        );
      }

      writeFileSync(finalVert, vertSrc, "utf-8");
      writeFileSync(finalFrag, fragSrc, "utf-8");
      summary.push(`${name} -> .vert.glsl (${vertSrc.length} B), .frag.glsl (${fragSrc.length} B)`);
    } finally {
      if (existsSync(tmpVert)) unlinkSync(tmpVert);
      if (existsSync(tmpFrag)) unlinkSync(tmpFrag);
    }
  }

  for (const name of COMPUTE_PASSES) {
    const wgslPath = join(shadersDir, `${name}.wgsl`);
    if (!existsSync(wgslPath)) {
      throw new Error(`Missing compute WGSL source: ${wgslPath}`);
    }
    execFileSync(bin, [wgslPath], { stdio: "inherit" });
    summary.push(`${name}.wgsl -> validated cleanly (WebGPU compute)`);
  }

  for (const line of summary) {
    console.log(`[shaders:build] OK: ${line}`);
  }
}

main();
