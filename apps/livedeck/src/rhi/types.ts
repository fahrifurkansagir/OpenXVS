/**
 * Thin Dual-Backend RHI Types (WebGPU primary + compile-time Naga WebGL2 explicit fallback).
 * Zero silent fallbacks (Charter §7.5): every context exposes explicit backend, shader-f16
 * capability, degradedReason, and WebGPU device.lost hook.
 */

export type RhiBackendKind = "webgpu" | "webgl2";

export interface RhiInitOptions {
  /** Force explicit WebGL2 backend (e.g., `?forceWebGL2=1` E2E parity check). */
  forceWebGL2?: boolean;
  /** Optional hook invoked immediately when a WebGPU device is lost. */
  onDeviceLost?: (info: { reason: string; message: string }) => void;
  /** Optional hook invoked immediately on any WebGPU uncaptured validation/OOM error. */
  onUncapturedError?: (message: string) => void;
}

export interface RhiCapabilities {
  backend: RhiBackendKind;
  hasShaderF16: boolean;
  hasCompute: boolean;
  maxTextureDimension2D: number;
  adapterDescription: string;
  /** Non-null whenever running in a degraded or fallback mode. */
  degradedReason: string | null;
}

export interface RhiPipelineDescriptor {
  label: string;
  wgsl: string;
  glslVert: string;
  glslFrag: string;
  vertexEntry?: string;
  fragmentEntry?: string;
  topology: "triangle-list" | "triangle-strip" | "line-list";
  depthTest: boolean;
  depthWrite: boolean;
  blendAlpha: boolean;
  vertexLayouts: RhiVertexBufferLayout[];
  /** Uniform block binding slots used by this pipeline (matching @group(0) @binding(N)). */
  uniformBindings: number[];
  /** Texture binding slots used by this pipeline (matching @group(0) @binding(N) for texture_2d). */
  textureBindings?: Array<{ textureBinding: number; samplerBinding?: number }>;
}

export interface RhiVertexAttribute {
  shaderLocation: number;
  offset: number;
  format: "float32x2" | "float32x3" | "float32x4";
}

export interface RhiVertexBufferLayout {
  arrayStride: number;
  attributes: RhiVertexAttribute[];
}

export interface RhiBufferHandle {
  readonly id: number;
  readonly byteLength: number;
  readonly usage: "uniform" | "vertex" | "index";
}

export interface RhiTextureHandle {
  readonly id: number;
  readonly width: number;
  readonly height: number;
  readonly format: "rgba8unorm";
}

export interface RhiPipelineHandle {
  readonly id: number;
  readonly label: string;
}

export interface RhiDrawCall {
  pipeline: RhiPipelineHandle;
  uniformBuffers: Array<{ binding: number; buffer: RhiBufferHandle }>;
  textures?: Array<{ textureBinding: number; samplerBinding?: number; texture: RhiTextureHandle }>;
  vertexBuffers: RhiBufferHandle[];
  indexBuffer?: RhiBufferHandle;
  vertexCount?: number;
  indexCount?: number;
  instanceCount?: number;
}

export interface RhiContext {
  readonly backend: RhiBackendKind;
  readonly hasShaderF16: boolean;
  readonly degradedReason: string | null;
  readonly capabilities: RhiCapabilities;
  readonly canvas: HTMLCanvasElement;

  /** Raw WebGPU device when backend === "webgpu", otherwise null. */
  readonly gpuDevice: GPUDevice | null;
  /** Raw WebGL2 context when backend === "webgl2", otherwise null. */
  readonly gl: WebGL2RenderingContext | null;

  /** Register an additional listener for WebGPU device.lost events. */
  onDeviceLost(cb: (info: { reason: string; message: string }) => void): void;
  /** Register an additional listener for WebGPU uncaptured validation/OOM errors. */
  onUncapturedError(cb: (message: string) => void): void;

  /** Ensure backing textures / swapchain match CSS/device pixel dimensions. */
  resize(width: number, height: number): void;
  getWidth(): number;
  getHeight(): number;

  createBuffer(usage: "uniform" | "vertex" | "index", dataOrSize: ArrayBufferView | number, label?: string): RhiBufferHandle;
  writeBuffer(handle: RhiBufferHandle, data: ArrayBufferView, byteOffset?: number): void;
  destroyBuffer(handle: RhiBufferHandle): void;

  createTextureRGBA8(width: number, height: number, data: Uint8Array | null, label?: string): RhiTextureHandle;
  updateTextureRGBA8(handle: RhiTextureHandle, data: Uint8Array): void;
  destroyTexture(handle: RhiTextureHandle): void;

  createRenderPipeline(desc: RhiPipelineDescriptor): RhiPipelineHandle;

  beginFrame(clearColor: readonly [number, number, number, number]): void;
  draw(call: RhiDrawCall): void;
  endFrame(): void;

  dispose(): void;
}
