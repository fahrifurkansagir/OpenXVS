/**
 * RHI Factory & Barrel Export.
 * Strictly enforces Charter §7.5: ZERO silent fallbacks — every fallback sets an explicit
 * `degradedReason`, logs telemetry, and surfaces visibly on the cockpit HUD.
 */

import type { RhiContext, RhiInitOptions } from "./types.ts";
import { WebGl2RhiContext } from "./webgl2_backend.ts";
import { WebGpuRhiContext } from "./webgpu_backend.ts";

export * from "./types.ts";
export * from "./coords.ts";
export { WebGpuRhiContext } from "./webgpu_backend.ts";
export { WebGl2RhiContext } from "./webgl2_backend.ts";

export async function createRhi(
  canvas: HTMLCanvasElement,
  options: RhiInitOptions = {},
): Promise<RhiContext> {
  if (options.forceWebGL2) {
    const reason = "Explicitly forced WebGL2 mode (forceWebGL2=true)";
    console.warn(`[OpenXVS RHI] ${reason}`);
    const glCtx = WebGl2RhiContext.create(canvas, reason);
    if (options.onDeviceLost) {
      glCtx.onDeviceLost(options.onDeviceLost);
    }
    return glCtx;
  }

  try {
    const attempt = await WebGpuRhiContext.tryCreate(
      canvas,
      options.onDeviceLost,
      options.onUncapturedError,
    );
    if (attempt.ctx) {
      if (attempt.ctx.degradedReason) {
        console.warn(`[OpenXVS RHI] WebGPU degraded: ${attempt.ctx.degradedReason}`);
      } else {
        console.info(
          `[OpenXVS RHI] WebGPU initialized (${attempt.ctx.capabilities.adapterDescription}, shader-f16=true)`,
        );
      }
      return attempt.ctx;
    }
    const reason = `WebGPU unavailable (${attempt.reason ?? "unknown"}) — explicit WebGL2 fallback`;
    console.warn(`[OpenXVS RHI] ${reason}`);
    const glCtx = WebGl2RhiContext.create(canvas, reason);
    if (options.onDeviceLost) {
      glCtx.onDeviceLost(options.onDeviceLost);
    }
    return glCtx;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    const reason = `WebGPU init error (${msg}) — explicit WebGL2 fallback`;
    console.warn(`[OpenXVS RHI] ${reason}`);
    const glCtx = WebGl2RhiContext.create(canvas, reason);
    if (options.onDeviceLost) {
      glCtx.onDeviceLost(options.onDeviceLost);
    }
    return glCtx;
  }
}
