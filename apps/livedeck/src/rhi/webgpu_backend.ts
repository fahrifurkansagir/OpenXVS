/**
 * Raw WebGPU RHI Backend (Primary).
 * - Zero heavy frameworks; explicit capability reporting (`hasShaderF16`, `degradedReason`).
 * - Exposes `device.lost` callback hook (Charter §7.5 — zero silent failures).
 * - Caches GPUBindGroups keyed by pipeline + resource handles to eliminate per-frame allocations.
 */

import type {
  RhiBufferHandle,
  RhiCapabilities,
  RhiContext,
  RhiDrawCall,
  RhiPipelineDescriptor,
  RhiPipelineHandle,
  RhiTextureHandle,
} from "./types.ts";

interface GpuBufferRecord {
  handle: RhiBufferHandle;
  buffer: GPUBuffer;
}

interface GpuTextureRecord {
  handle: RhiTextureHandle;
  texture: GPUTexture;
  view: GPUTextureView;
}

interface GpuPipelineRecord {
  handle: RhiPipelineHandle;
  pipeline: GPURenderPipeline;
  layout: GPUBindGroupLayout;
}

export class WebGpuRhiContext implements RhiContext {
  readonly backend = "webgpu" as const;
  readonly hasShaderF16: boolean;
  readonly degradedReason: string | null;
  readonly capabilities: RhiCapabilities;
  readonly canvas: HTMLCanvasElement;
  readonly gpuDevice: GPUDevice;
  readonly gl: null = null;

  private readonly ctx: GPUCanvasContext;
  private readonly colorFormat: GPUTextureFormat;
  private readonly linearSampler: GPUSampler;
  private readonly lostListeners: Array<(info: { reason: string; message: string }) => void> = [];
  private readonly errorListeners: Array<(message: string) => void> = [];

  private width: number;
  private height: number;
  private depthTexture: GPUTexture | null = null;
  private depthView: GPUTextureView | null = null;

  private nextId = 1;
  private readonly buffers = new Map<number, GpuBufferRecord>();
  private readonly textures = new Map<number, GpuTextureRecord>();
  private readonly pipelines = new Map<number, GpuPipelineRecord>();
  private readonly bindGroupCache = new Map<string, GPUBindGroup>();

  private activeEncoder: GPUCommandEncoder | null = null;
  private activePass: GPURenderPassEncoder | null = null;
  private disposed = false;

  constructor(
    canvas: HTMLCanvasElement,
    device: GPUDevice,
    ctx: GPUCanvasContext,
    colorFormat: GPUTextureFormat,
    capabilities: RhiCapabilities,
    initialLostCb?: (info: { reason: string; message: string }) => void,
    initialErrorCb?: (message: string) => void,
  ) {
    this.canvas = canvas;
    this.gpuDevice = device;
    this.ctx = ctx;
    this.colorFormat = colorFormat;
    this.capabilities = capabilities;
    this.hasShaderF16 = capabilities.hasShaderF16;
    this.degradedReason = capabilities.degradedReason;

    if (initialLostCb) {
      this.lostListeners.push(initialLostCb);
    }
    if (initialErrorCb) {
      this.errorListeners.push(initialErrorCb);
    }

    this.gpuDevice.addEventListener("uncapturederror", (ev) => {
      const msg = ev.error?.message || "WebGPU uncaptured error";
      for (const cb of this.errorListeners) {
        try {
          cb(msg);
        } catch {
          // Ignore listener throws
        }
      }
    });

    this.width = Math.max(1, canvas.width || canvas.clientWidth || 1280);
    this.height = Math.max(1, canvas.height || canvas.clientHeight || 720);
    this.canvas.width = this.width;
    this.canvas.height = this.height;

    this.ctx.configure({
      device: this.gpuDevice,
      format: this.colorFormat,
      alphaMode: "opaque",
    });

    this.linearSampler = this.gpuDevice.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });

    this.recreateDepthTexture();

    void this.gpuDevice.lost.then((info) => {
      const payload = {
        reason: String(info.reason ?? "unknown"),
        message: info.message || "WebGPU device lost",
      };
      for (const cb of this.lostListeners) {
        try {
          cb(payload);
        } catch {
          // Never allow listener throw to mask device-lost handling
        }
      }
    });
  }

  static async tryCreate(
    canvas: HTMLCanvasElement,
    onDeviceLost?: (info: { reason: string; message: string }) => void,
    onUncapturedError?: (message: string) => void,
  ): Promise<{ ctx: WebGpuRhiContext | null; reason: string | null }> {
    if (typeof navigator === "undefined" || !("gpu" in navigator) || !navigator.gpu) {
      return { ctx: null, reason: "navigator.gpu unavailable" };
    }
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) {
      return { ctx: null, reason: "WebGPU adapter request returned null" };
    }

    const hasShaderF16 = adapter.features.has("shader-f16");
    const requiredFeatures: GPUFeatureName[] = hasShaderF16 ? ["shader-f16"] : [];
    const device = await adapter.requestDevice({ requiredFeatures });

    const webgpuCtx = canvas.getContext("webgpu");
    if (!webgpuCtx) {
      device.destroy();
      return { ctx: null, reason: "canvas.getContext('webgpu') returned null" };
    }

    const colorFormat = navigator.gpu.getPreferredCanvasFormat();
    const info = adapter.info;
    const adapterDescription = info
      ? `${info.vendor || "GPU"} ${info.architecture || ""} ${info.description || ""}`.trim()
      : "WebGPU Adapter";

    const capabilities: RhiCapabilities = {
      backend: "webgpu",
      hasShaderF16,
      hasCompute: true,
      maxTextureDimension2D: device.limits.maxTextureDimension2D,
      adapterDescription,
      degradedReason: hasShaderF16 ? null : "WebGPU adapter lacks shader-f16 (AI uses FP32)",
    };

    return {
      ctx: new WebGpuRhiContext(
        canvas,
        device,
        webgpuCtx,
        colorFormat,
        capabilities,
        onDeviceLost,
        onUncapturedError,
      ),
      reason: null,
    };
  }

  onDeviceLost(cb: (info: { reason: string; message: string }) => void): void {
    this.lostListeners.push(cb);
  }

  onUncapturedError(cb: (message: string) => void): void {
    this.errorListeners.push(cb);
  }

  resize(width: number, height: number): void {
    const w = Math.max(1, Math.floor(width));
    const h = Math.max(1, Math.floor(height));
    if (w === this.width && h === this.height) {
      return;
    }
    this.width = w;
    this.height = h;
    this.canvas.width = w;
    this.canvas.height = h;
    this.recreateDepthTexture();
  }

  getWidth(): number {
    return this.width;
  }

  getHeight(): number {
    return this.height;
  }

  private recreateDepthTexture(): void {
    if (this.depthTexture) {
      this.depthTexture.destroy();
    }
    this.depthTexture = this.gpuDevice.createTexture({
      label: "livedeck-depth24plus",
      size: { width: this.width, height: this.height, depthOrArrayLayers: 1 },
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.depthView = this.depthTexture.createView();
  }

  createBuffer(
    usage: "uniform" | "vertex" | "index",
    dataOrSize: ArrayBufferView | number,
    label?: string,
  ): RhiBufferHandle {
    const rawSize = typeof dataOrSize === "number" ? dataOrSize : dataOrSize.byteLength;
    const alignedSize = Math.max(16, Math.ceil(rawSize / 16) * 16);
    let gpuUsage = GPUBufferUsage.COPY_DST;
    if (usage === "uniform") gpuUsage |= GPUBufferUsage.UNIFORM;
    if (usage === "vertex") gpuUsage |= GPUBufferUsage.VERTEX;
    if (usage === "index") gpuUsage |= GPUBufferUsage.INDEX;

    const buffer = this.gpuDevice.createBuffer({
      label,
      size: alignedSize,
      usage: gpuUsage,
    });

    if (typeof dataOrSize !== "number" && dataOrSize.byteLength > 0) {
      const bytes = new Uint8Array(
        dataOrSize.buffer,
        dataOrSize.byteOffset,
        dataOrSize.byteLength,
      ) as Uint8Array<ArrayBuffer>;
      this.gpuDevice.queue.writeBuffer(buffer, 0, bytes);
    }

    const handle: RhiBufferHandle = {
      id: this.nextId++,
      byteLength: alignedSize,
      usage,
    };
    this.buffers.set(handle.id, { handle, buffer });
    return handle;
  }

  writeBuffer(handle: RhiBufferHandle, data: ArrayBufferView, byteOffset = 0): void {
    const rec = this.buffers.get(handle.id);
    if (!rec) {
      throw new Error(`Unknown RHI buffer handle ${handle.id}`);
    }
    const bytes = new Uint8Array(
      data.buffer,
      data.byteOffset,
      data.byteLength,
    ) as Uint8Array<ArrayBuffer>;
    this.gpuDevice.queue.writeBuffer(rec.buffer, byteOffset, bytes);
  }

  destroyBuffer(handle: RhiBufferHandle): void {
    const rec = this.buffers.get(handle.id);
    if (rec) {
      rec.buffer.destroy();
      this.buffers.delete(handle.id);
      this.bindGroupCache.clear();
    }
  }

  createTextureRGBA8(
    width: number,
    height: number,
    data: Uint8Array | null,
    label?: string,
  ): RhiTextureHandle {
    const texture = this.gpuDevice.createTexture({
      label,
      size: { width, height, depthOrArrayLayers: 1 },
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    if (data) {
      this.gpuDevice.queue.writeTexture(
        { texture },
        data as Uint8Array<ArrayBuffer>,
        { bytesPerRow: width * 4, rowsPerImage: height },
        { width, height, depthOrArrayLayers: 1 },
      );
    }
    const handle: RhiTextureHandle = {
      id: this.nextId++,
      width,
      height,
      format: "rgba8unorm",
    };
    this.textures.set(handle.id, { handle, texture, view: texture.createView() });
    return handle;
  }

  updateTextureRGBA8(handle: RhiTextureHandle, data: Uint8Array): void {
    const rec = this.textures.get(handle.id);
    if (!rec) {
      throw new Error(`Unknown RHI texture handle ${handle.id}`);
    }
    this.gpuDevice.queue.writeTexture(
      { texture: rec.texture },
      data as Uint8Array<ArrayBuffer>,
      { bytesPerRow: handle.width * 4, rowsPerImage: handle.height },
      { width: handle.width, height: handle.height, depthOrArrayLayers: 1 },
    );
  }

  destroyTexture(handle: RhiTextureHandle): void {
    const rec = this.textures.get(handle.id);
    if (rec) {
      rec.texture.destroy();
      this.textures.delete(handle.id);
      this.bindGroupCache.clear();
    }
  }

  createRenderPipeline(desc: RhiPipelineDescriptor): RhiPipelineHandle {
    const module = this.gpuDevice.createShaderModule({
      label: `${desc.label}-wgsl`,
      code: desc.wgsl,
    });

    const layoutEntries: GPUBindGroupLayoutEntry[] = [];
    for (const binding of desc.uniformBindings) {
      layoutEntries.push({
        binding,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform" },
      });
    }
    for (const tb of desc.textureBindings ?? []) {
      layoutEntries.push({
        binding: tb.textureBinding,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        texture: { sampleType: "float" },
      });
      if (tb.samplerBinding !== undefined) {
        layoutEntries.push({
          binding: tb.samplerBinding,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          sampler: { type: "filtering" },
        });
      }
    }

    const bindGroupLayout = this.gpuDevice.createBindGroupLayout({
      label: `${desc.label}-bgl`,
      entries: layoutEntries,
    });

    const pipelineLayout = this.gpuDevice.createPipelineLayout({
      label: `${desc.label}-pl`,
      bindGroupLayouts: [bindGroupLayout],
    });

    const vertexBuffers: GPUVertexBufferLayout[] = desc.vertexLayouts.map((vl) => ({
      arrayStride: vl.arrayStride,
      stepMode: "vertex",
      attributes: vl.attributes.map((a) => ({
        shaderLocation: a.shaderLocation,
        offset: a.offset,
        format: a.format,
      })),
    }));

    const pipeline = this.gpuDevice.createRenderPipeline({
      label: desc.label,
      layout: pipelineLayout,
      vertex: {
        module,
        entryPoint: desc.vertexEntry ?? "vs_main",
        buffers: vertexBuffers,
      },
      fragment: {
        module,
        entryPoint: desc.fragmentEntry ?? "fs_main",
        targets: [
          {
            format: this.colorFormat,
            blend: desc.blendAlpha
              ? {
                  color: {
                    srcFactor: "src-alpha",
                    dstFactor: "one-minus-src-alpha",
                    operation: "add",
                  },
                  alpha: {
                    srcFactor: "one",
                    dstFactor: "one-minus-src-alpha",
                    operation: "add",
                  },
                }
              : undefined,
          },
        ],
      },
      primitive: {
        topology: desc.topology,
        cullMode: "none",
      },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: desc.depthWrite,
        depthCompare: desc.depthTest ? "less-equal" : "always",
      },
    });

    const handle: RhiPipelineHandle = {
      id: this.nextId++,
      label: desc.label,
    };
    this.pipelines.set(handle.id, { handle, pipeline, layout: bindGroupLayout });
    return handle;
  }

  beginFrame(clearColor: readonly [number, number, number, number]): void {
    if (this.disposed) return;
    if (!this.depthView) {
      this.recreateDepthTexture();
    }
    const colorView = this.ctx.getCurrentTexture().createView();
    const encoder = this.gpuDevice.createCommandEncoder({ label: "livedeck-frame" });
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: colorView,
          clearValue: { r: clearColor[0], g: clearColor[1], b: clearColor[2], a: clearColor[3] },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: this.depthView!,
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });
    this.activeEncoder = encoder;
    this.activePass = pass;
  }

  draw(call: RhiDrawCall): void {
    const pass = this.activePass;
    if (!pass) return;
    const pipeRec = this.pipelines.get(call.pipeline.id);
    if (!pipeRec) {
      throw new Error(`Unknown RHI pipeline handle ${call.pipeline.id}`);
    }

    const cacheKey = [
      call.pipeline.id,
      ...call.uniformBuffers.map((u) => `u${u.binding}:${u.buffer.id}`),
      ...(call.textures ?? []).map((t) => `t${t.textureBinding}:${t.samplerBinding ?? "-"}:${t.texture.id}`),
    ].join("|");

    let bindGroup = this.bindGroupCache.get(cacheKey);
    if (!bindGroup) {
      const entries: GPUBindGroupEntry[] = [];
      for (const ub of call.uniformBuffers) {
        const bRec = this.buffers.get(ub.buffer.id);
        if (!bRec) throw new Error(`Unknown uniform buffer ${ub.buffer.id}`);
        entries.push({ binding: ub.binding, resource: { buffer: bRec.buffer } });
      }
      for (const tb of call.textures ?? []) {
        const tRec = this.textures.get(tb.texture.id);
        if (!tRec) throw new Error(`Unknown texture ${tb.texture.id}`);
        entries.push({ binding: tb.textureBinding, resource: tRec.view });
        if (tb.samplerBinding !== undefined) {
          entries.push({ binding: tb.samplerBinding, resource: this.linearSampler });
        }
      }
      bindGroup = this.gpuDevice.createBindGroup({
        layout: pipeRec.layout,
        entries,
      });
      this.bindGroupCache.set(cacheKey, bindGroup);
    }

    pass.setPipeline(pipeRec.pipeline);
    pass.setBindGroup(0, bindGroup);

    for (let i = 0; i < call.vertexBuffers.length; i++) {
      const vbRec = this.buffers.get(call.vertexBuffers[i].id);
      if (!vbRec) throw new Error(`Unknown vertex buffer ${call.vertexBuffers[i].id}`);
      pass.setVertexBuffer(i, vbRec.buffer);
    }

    if (call.indexBuffer && call.indexCount !== undefined) {
      const ibRec = this.buffers.get(call.indexBuffer.id);
      if (!ibRec) throw new Error(`Unknown index buffer ${call.indexBuffer.id}`);
      pass.setIndexBuffer(ibRec.buffer, "uint32");
      pass.drawIndexed(call.indexCount, call.instanceCount ?? 1, 0, 0, 0);
    } else if (call.vertexCount !== undefined) {
      pass.draw(call.vertexCount, call.instanceCount ?? 1, 0, 0);
    }
  }

  endFrame(): void {
    if (!this.activePass || !this.activeEncoder) return;
    this.activePass.end();
    this.gpuDevice.queue.submit([this.activeEncoder.finish()]);
    this.activePass = null;
    this.activeEncoder = null;
  }

  dispose(): void {
    this.disposed = true;
    for (const rec of this.buffers.values()) rec.buffer.destroy();
    for (const rec of this.textures.values()) rec.texture.destroy();
    this.buffers.clear();
    this.textures.clear();
    this.pipelines.clear();
    this.bindGroupCache.clear();
    if (this.depthTexture) {
      this.depthTexture.destroy();
      this.depthTexture = null;
      this.depthView = null;
    }
    this.gpuDevice.destroy();
  }
}
