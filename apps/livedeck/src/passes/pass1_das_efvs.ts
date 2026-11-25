/**
 * Pass 1: 6-Camera Spherical Distributed-Aperture System (XVS6-CUBE-v1) & 8-14 um LWIR EFVS.
 * Consumes WGSL on WebGPU and compile-time Naga GLSL ES 3.00 on WebGL2.
 */

import { WGS84_A } from "../rhi/coords.ts";
import type { RhiBufferHandle, RhiContext, RhiPipelineHandle } from "../rhi/types.ts";
import pass1Wgsl from "../shaders/pass1_das_efvs.wgsl?raw";
import pass1GlVert from "../shaders/generated_gl/pass1_das_efvs.vert.glsl?raw";
import pass1GlFrag from "../shaders/generated_gl/pass1_das_efvs.frag.glsl?raw";
import type { ActiveCameraView, CockpitFrameState } from "./types.ts";

// Afternoon SF Bay sun direction in NED (West-Southwest, ~27.5 deg elevation -> negative Down)
// Shared by Pass 1 (Rayleigh+Mie sky) and Pass 2 (terrain Lambert + water GGX specular)
export const SF_BAY_SUN_ELEV_RAD = 0.48;
export const SF_BAY_SUN_AZIM_RAD = 4.25; // ~243.5 deg true (WSW)
export const SF_BAY_SUN_DIR_NED: readonly [number, number, number] = [
  Math.cos(SF_BAY_SUN_ELEV_RAD) * Math.cos(SF_BAY_SUN_AZIM_RAD),
  Math.cos(SF_BAY_SUN_ELEV_RAD) * Math.sin(SF_BAY_SUN_AZIM_RAD),
  -Math.sin(SF_BAY_SUN_ELEV_RAD),
];

// Fullscreen triangle covering NDC [-1, 1]^2
const FULLSCREEN_TRI_VERTS = new Float32Array([
  -1.0, -1.0,
   3.0, -1.0,
  -1.0,  3.0,
]);

export class Pass1DasEfvs {
  private readonly rhi: RhiContext;
  private readonly pipeline: RhiPipelineHandle;
  private readonly vb: RhiBufferHandle;
  private readonly ubo: RhiBufferHandle;
  private readonly uniformData = new Float32Array(36); // 9 x vec4<f32> = 144 bytes

  constructor(rhi: RhiContext) {
    this.rhi = rhi;
    this.vb = rhi.createBuffer("vertex", FULLSCREEN_TRI_VERTS, "pass1-fullscreen-vb");
    this.ubo = rhi.createBuffer("uniform", this.uniformData.byteLength, "pass1-ubo");
    this.pipeline = rhi.createRenderPipeline({
      label: "pass1-das-efvs",
      wgsl: pass1Wgsl,
      glslVert: pass1GlVert,
      glslFrag: pass1GlFrag,
      topology: "triangle-list",
      depthTest: false,
      depthWrite: false,
      blendAlpha: false,
      vertexLayouts: [
        {
          arrayStride: 8,
          attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
        },
      ],
      uniformBindings: [0],
    });
  }

  encode(state: CockpitFrameState, view: ActiveCameraView): void {
    const u = this.uniformData;
    const rbc = view.rBodyCam; // row-major 3x3 -> pack columns into vec4s for WGSL mat3x3
    const rnb = view.rNedBody;

    // rBodyCam col 0 + fx
    u[0] = rbc[0]; u[1] = rbc[3]; u[2] = rbc[6]; u[3] = view.intr.fx;
    // rBodyCam col 1 + fy
    u[4] = rbc[1]; u[5] = rbc[4]; u[6] = rbc[7]; u[7] = view.intr.fy;
    // rBodyCam col 2 + cx
    u[8] = rbc[2]; u[9] = rbc[5]; u[10] = rbc[8]; u[11] = view.intr.cx;

    // rNedBody col 0 + cy
    u[12] = rnb[0]; u[13] = rnb[3]; u[14] = rnb[6]; u[15] = view.intr.cy;
    // rNedBody col 1 + width
    u[16] = rnb[1]; u[17] = rnb[4]; u[18] = rnb[7]; u[19] = view.width;
    // rNedBody col 2 + height
    u[20] = rnb[2]; u[21] = rnb[5]; u[22] = rnb[8]; u[23] = view.height;

    // Afternoon SF Bay sun direction in NED (shared constant with Pass 2 terrain/water)
    const altAboveEllipsoid = Math.max(state.altWgs84M, 0.0);
    // CONVENTIONS.md §7 spherical limb dip approximation: arccos(R / (R + h))
    const horizonDipRad = Math.acos(WGS84_A / (WGS84_A + altAboveEllipsoid));

    u[24] = SF_BAY_SUN_DIR_NED[0];
    u[25] = SF_BAY_SUN_DIR_NED[1];
    u[26] = SF_BAY_SUN_DIR_NED[2];
    u[27] = horizonDipRad;

    // envParams: fogContrast, efvsEnabled, camRBiasRad, simTimeS
    // External 3D Chase/Orbit spectator sky never applies cockpit DAS Cam-R fault cross-hatching
    const isExternalSpectator = view.cameraMode !== "cockpit";
    u[28] = state.fogContrast;
    u[29] = state.efvsEnabled ? 1.0 : 0.0;
    u[30] = isExternalSpectator ? 0.0 : state.camRBiasRad;
    u[31] = state.simTimeS;

    // healthParams: camHealthMask, glareStrength, altitudeM, simplexMode
    u[32] = isExternalSpectator ? 0x3f : state.camHealthMask & 0x3f;
    u[33] = state.glareStrength ?? 0.75;
    u[34] = state.altWgs84M;
    u[35] = state.simplexMode;

    this.rhi.writeBuffer(this.ubo, this.uniformData);
    this.rhi.draw({
      pipeline: this.pipeline,
      uniformBuffers: [{ binding: 0, buffer: this.ubo }],
      vertexBuffers: [this.vb],
      vertexCount: 3,
    });
  }

  dispose(): void {
    this.rhi.destroyBuffer(this.vb);
    this.rhi.destroyBuffer(this.ubo);
  }
}
