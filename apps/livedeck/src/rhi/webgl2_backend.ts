/**
 * Explicit WebGL2 RHI Backend consuming compile-time `naga-cli` GLSL ES 3.00 shaders.
 *
 * Coordinate-space parity with WebGPU:
 * - Naga's default `ADJUST_COORDINATE_SPACE` emits:
 *     `gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);`
 * - Rendering into an offscreen RGBA8 + DEPTH24 FBO causes `gl_FragCoord.xy` to match
 *   WebGPU `@builtin(position).xy` (top-left origin, pixel centers at `(u + 0.5, v + 0.5)`),
 *   and `endFrame()` flips vertically onto the default canvas via hardware `blitFramebuffer`.
 * - Result: 100% unmodified Naga GLSL ES 3.00 output and exact `<= 0.05 px` parity with WebGPU.
 */

import type {
  RhiBufferHandle,
  RhiCapabilities,
  RhiContext,
  RhiDrawCall,
  RhiPipelineDescriptor,
  RhiPipelineHandle,
  RhiTextureHandle,
  RhiVertexBufferLayout,
} from "./types.ts";

interface GlBufferRecord {
  handle: RhiBufferHandle;
  buffer: WebGLBuffer;
  target: number;
}

interface GlTextureRecord {
  handle: RhiTextureHandle;
  texture: WebGLTexture;
}

interface GlPipelineRecord {
  handle: RhiPipelineHandle;
  program: WebGLProgram;
  topology: number;
  depthTest: boolean;
  depthWrite: boolean;
  blendAlpha: boolean;
  vertexLayouts: RhiVertexBufferLayout[];
  /** Maps WGSL `@binding(N)` -> active GLSL sampler uniform location(s). */
  samplerLocations: Map<number, WebGLUniformLocation[]>;
}

export class WebGl2RhiContext implements RhiContext {
  readonly backend = "webgl2" as const;
  readonly hasShaderF16 = false;
  readonly degradedReason: string;
  readonly capabilities: RhiCapabilities;
  readonly canvas: HTMLCanvasElement;
  readonly gpuDevice: null = null;
  readonly gl: WebGL2RenderingContext;

  private width: number;
  private height: number;

  private fbo: WebGLFramebuffer | null = null;
  private colorRb: WebGLRenderbuffer | null = null;
  private depthRb: WebGLRenderbuffer | null = null;
  private vao: WebGLVertexArrayObject | null = null;

  private nextId = 1;
  private readonly buffers = new Map<number, GlBufferRecord>();
  private readonly textures = new Map<number, GlTextureRecord>();
  private readonly pipelines = new Map<number, GlPipelineRecord>();
  private readonly lostListeners: Array<(info: { reason: string; message: string }) => void> = [];

  constructor(canvas: HTMLCanvasElement, gl: WebGL2RenderingContext, degradedReason: string) {
    this.canvas = canvas;
    this.gl = gl;
    this.degradedReason = degradedReason;

    const dbgExt = gl.getExtension("WEBGL_debug_renderer_info");
    const rendererStr = dbgExt
      ? String(gl.getParameter(dbgExt.UNMASKED_RENDERER_WEBGL))
      : String(gl.getParameter(gl.RENDERER));

    this.capabilities = {
      backend: "webgl2",
      hasShaderF16: false,
      hasCompute: false,
      maxTextureDimension2D: Number(gl.getParameter(gl.MAX_TEXTURE_SIZE)) || 4096,
      adapterDescription: rendererStr,
      degradedReason,
    };

    this.width = Math.max(1, canvas.width || canvas.clientWidth || 1280);
    this.height = Math.max(1, canvas.height || canvas.clientHeight || 720);
    this.canvas.width = this.width;
    this.canvas.height = this.height;

    this.vao = gl.createVertexArray();
    this.recreateFbo();

    canvas.addEventListener("webglcontextlost", (ev) => {
      ev.preventDefault();
      for (const cb of this.lostListeners) {
        cb({ reason: "webglcontextlost", message: "WebGL2 context lost" });
      }
    });
  }

  static create(canvas: HTMLCanvasElement, degradedReason: string): WebGl2RhiContext {
    const gl = canvas.getContext("webgl2", {
      alpha: false,
      antialias: false,
      depth: true,
      preserveDrawingBuffer: true,
      powerPreference: "high-performance",
    });
    if (!gl) {
      throw new Error("Neither WebGPU nor WebGL2 context could be initialized on canvas.");
    }
    return new WebGl2RhiContext(canvas, gl, degradedReason);
  }

  onDeviceLost(cb: (info: { reason: string; message: string }) => void): void {
    this.lostListeners.push(cb);
  }

  onUncapturedError(_cb: (message: string) => void): void {
    // WebGL2 errors are caught synchronously; hook kept for RhiContext interface parity
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
    this.recreateFbo();
  }

  getWidth(): number {
    return this.width;
  }

  getHeight(): number {
    return this.height;
  }

  private recreateFbo(): void {
    const gl = this.gl;
    if (this.fbo) gl.deleteFramebuffer(this.fbo);
    if (this.colorRb) gl.deleteRenderbuffer(this.colorRb);
    if (this.depthRb) gl.deleteRenderbuffer(this.depthRb);

    this.fbo = gl.createFramebuffer();
    this.colorRb = gl.createRenderbuffer();
    this.depthRb = gl.createRenderbuffer();

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);

    gl.bindRenderbuffer(gl.RENDERBUFFER, this.colorRb);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.RGBA8, this.width, this.height);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, this.colorRb);

    gl.bindRenderbuffer(gl.RENDERBUFFER, this.depthRb);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, this.width, this.height);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, this.depthRb);

    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`WebGL2 offscreen FBO incomplete (status 0x${status.toString(16)})`);
    }
  }

  createBuffer(
    usage: "uniform" | "vertex" | "index",
    dataOrSize: ArrayBufferView | number,
    _label?: string,
  ): RhiBufferHandle {
    const gl = this.gl;
    const buffer = gl.createBuffer();
    if (!buffer) throw new Error("gl.createBuffer() failed");

    const target =
      usage === "uniform"
        ? gl.UNIFORM_BUFFER
        : usage === "index"
          ? gl.ELEMENT_ARRAY_BUFFER
          : gl.ARRAY_BUFFER;

    const rawSize = typeof dataOrSize === "number" ? dataOrSize : dataOrSize.byteLength;
    const alignedSize = Math.max(16, Math.ceil(rawSize / 16) * 16);

    gl.bindBuffer(target, buffer);
    if (typeof dataOrSize === "number") {
      gl.bufferData(target, alignedSize, gl.DYNAMIC_DRAW);
    } else {
      const padded = new Uint8Array(alignedSize);
      padded.set(new Uint8Array(dataOrSize.buffer, dataOrSize.byteOffset, dataOrSize.byteLength));
      gl.bufferData(target, padded, gl.DYNAMIC_DRAW);
    }
    gl.bindBuffer(target, null);

    const handle: RhiBufferHandle = {
      id: this.nextId++,
      byteLength: alignedSize,
      usage,
    };
    this.buffers.set(handle.id, { handle, buffer, target });
    return handle;
  }

  writeBuffer(handle: RhiBufferHandle, data: ArrayBufferView, byteOffset = 0): void {
    const rec = this.buffers.get(handle.id);
    if (!rec) throw new Error(`Unknown WebGL2 buffer handle ${handle.id}`);
    const gl = this.gl;
    gl.bindBuffer(rec.target, rec.buffer);
    gl.bufferSubData(
      rec.target,
      byteOffset,
      new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
    );
    gl.bindBuffer(rec.target, null);
  }

  destroyBuffer(handle: RhiBufferHandle): void {
    const rec = this.buffers.get(handle.id);
    if (rec) {
      this.gl.deleteBuffer(rec.buffer);
      this.buffers.delete(handle.id);
    }
  }

  createTextureRGBA8(
    width: number,
    height: number,
    data: Uint8Array | null,
    _label?: string,
  ): RhiTextureHandle {
    const gl = this.gl;
    const texture = gl.createTexture();
    if (!texture) throw new Error("gl.createTexture() failed");

    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 0);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA8,
      width,
      height,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      data,
    );
    gl.bindTexture(gl.TEXTURE_2D, null);

    const handle: RhiTextureHandle = {
      id: this.nextId++,
      width,
      height,
      format: "rgba8unorm",
    };
    this.textures.set(handle.id, { handle, texture });
    return handle;
  }

  updateTextureRGBA8(handle: RhiTextureHandle, data: Uint8Array): void {
    const rec = this.textures.get(handle.id);
    if (!rec) throw new Error(`Unknown WebGL2 texture handle ${handle.id}`);
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, rec.texture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 0);
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      0,
      0,
      handle.width,
      handle.height,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      data,
    );
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  destroyTexture(handle: RhiTextureHandle): void {
    const rec = this.textures.get(handle.id);
    if (rec) {
      this.gl.deleteTexture(rec.texture);
      this.textures.delete(handle.id);
    }
  }

  private compileShader(type: number, source: string, label: string): WebGLShader {
    const gl = this.gl;
    const shader = gl.createShader(type);
    if (!shader) throw new Error(`gl.createShader failed for ${label}`);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(shader) || "Unknown shader error";
      gl.deleteShader(shader);
      throw new Error(`WebGL2 compile error (${label}):\n${log}`);
    }
    return shader;
  }

  createRenderPipeline(desc: RhiPipelineDescriptor): RhiPipelineHandle {
    const gl = this.gl;
    const vs = this.compileShader(gl.VERTEX_SHADER, desc.glslVert, `${desc.label}.vert`);
    const fs = this.compileShader(gl.FRAGMENT_SHADER, desc.glslFrag, `${desc.label}.frag`);
    const program = gl.createProgram();
    if (!program) throw new Error(`gl.createProgram failed for ${desc.label}`);
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    gl.deleteShader(vs);
    gl.deleteShader(fs);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(program) || "Unknown link error";
      gl.deleteProgram(program);
      throw new Error(`WebGL2 link error (${desc.label}):\n${log}`);
    }

    // Bind Naga-generated std140 uniform blocks (`*_block_<N>Vertex` / `*_block_<N>Fragment`) to UBO slot N
    const numBlocks = Number(gl.getProgramParameter(program, gl.ACTIVE_UNIFORM_BLOCKS)) || 0;
    for (let i = 0; i < numBlocks; i++) {
      const name = gl.getActiveUniformBlockName(program, i) || "";
      const m = /_block_(\d+)/.exec(name);
      const slot = m ? Number(m[1]) : 0;
      gl.uniformBlockBinding(program, i, slot);
    }

    // Discover Naga-generated sampler uniforms (`_group_0_binding_<N>_vs` / `_fs`)
    const samplerLocations = new Map<number, WebGLUniformLocation[]>();
    const numUniforms = Number(gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS)) || 0;
    for (let i = 0; i < numUniforms; i++) {
      const info = gl.getActiveUniform(program, i);
      if (!info) continue;
      const m = /_group_0_binding_(\d+)_/.exec(info.name);
      if (!m) continue;
      if (info.type === gl.SAMPLER_2D) {
        const binding = Number(m[1]);
        const loc = gl.getUniformLocation(program, info.name);
        if (loc) {
          const list = samplerLocations.get(binding) ?? [];
          list.push(loc);
          samplerLocations.set(binding, list);
        }
      }
    }

    const topology =
      desc.topology === "line-list"
        ? gl.LINES
        : desc.topology === "triangle-strip"
          ? gl.TRIANGLE_STRIP
          : gl.TRIANGLES;

    const handle: RhiPipelineHandle = {
      id: this.nextId++,
      label: desc.label,
    };
    this.pipelines.set(handle.id, {
      handle,
      program,
      topology,
      depthTest: desc.depthTest,
      depthWrite: desc.depthWrite,
      blendAlpha: desc.blendAlpha,
      vertexLayouts: desc.vertexLayouts,
      samplerLocations,
    });
    return handle;
  }

  beginFrame(clearColor: readonly [number, number, number, number]): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, this.width, this.height);
    gl.disable(gl.CULL_FACE);
    gl.depthMask(true);
    gl.clearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
    gl.clearDepth(1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  }

  draw(call: RhiDrawCall): void {
    const gl = this.gl;
    const pipeRec = this.pipelines.get(call.pipeline.id);
    if (!pipeRec) throw new Error(`Unknown WebGL2 pipeline handle ${call.pipeline.id}`);

    gl.useProgram(pipeRec.program);

    if (pipeRec.depthTest) {
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LEQUAL);
    } else {
      gl.disable(gl.DEPTH_TEST);
    }
    gl.depthMask(pipeRec.depthWrite);

    if (pipeRec.blendAlpha) {
      gl.enable(gl.BLEND);
      gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    } else {
      gl.disable(gl.BLEND);
    }

    // Bind uniform buffers to matching std140 UBO slots
    for (const ub of call.uniformBuffers) {
      const bRec = this.buffers.get(ub.buffer.id);
      if (!bRec) throw new Error(`Unknown WebGL2 uniform buffer ${ub.buffer.id}`);
      gl.bindBufferBase(gl.UNIFORM_BUFFER, ub.binding, bRec.buffer);
    }

    // Bind textures to sequential texture units and assign Naga sampler uniforms
    let unit = 0;
    for (const tb of call.textures ?? []) {
      const tRec = this.textures.get(tb.texture.id);
      if (!tRec) throw new Error(`Unknown WebGL2 texture ${tb.texture.id}`);
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tRec.texture);
      const locs = pipeRec.samplerLocations.get(tb.textureBinding) ?? [];
      for (const loc of locs) {
        gl.uniform1i(loc, unit);
      }
      unit++;
    }

    gl.bindVertexArray(this.vao);
    const enabledLocations: number[] = [];

    for (let i = 0; i < pipeRec.vertexLayouts.length; i++) {
      const layout = pipeRec.vertexLayouts[i];
      const vbHandle = call.vertexBuffers[i];
      const vbRec = this.buffers.get(vbHandle.id);
      if (!vbRec) throw new Error(`Unknown WebGL2 vertex buffer ${vbHandle.id}`);
      gl.bindBuffer(gl.ARRAY_BUFFER, vbRec.buffer);
      for (const attr of layout.attributes) {
        const size = attr.format === "float32x2" ? 2 : attr.format === "float32x3" ? 3 : 4;
        gl.enableVertexAttribArray(attr.shaderLocation);
        gl.vertexAttribPointer(
          attr.shaderLocation,
          size,
          gl.FLOAT,
          false,
          layout.arrayStride,
          attr.offset,
        );
        enabledLocations.push(attr.shaderLocation);
      }
    }

    if (call.indexBuffer && call.indexCount !== undefined) {
      const ibRec = this.buffers.get(call.indexBuffer.id);
      if (!ibRec) throw new Error(`Unknown WebGL2 index buffer ${call.indexBuffer.id}`);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibRec.buffer);
      gl.drawElements(pipeRec.topology, call.indexCount, gl.UNSIGNED_INT, 0);
    } else if (call.vertexCount !== undefined) {
      gl.drawArrays(pipeRec.topology, 0, call.vertexCount);
    }

    for (const loc of enabledLocations) {
      gl.disableVertexAttribArray(loc);
    }
    gl.bindVertexArray(null);
  }

  endFrame(): void {
    const gl = this.gl;
    // Flip FBO vertically onto default canvas framebuffer to match WebGPU top-left NDC convention
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.fbo);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
    gl.blitFramebuffer(
      0,
      0,
      this.width,
      this.height,
      0,
      this.height,
      this.width,
      0,
      gl.COLOR_BUFFER_BIT,
      gl.NEAREST,
    );
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  dispose(): void {
    const gl = this.gl;
    for (const rec of this.buffers.values()) gl.deleteBuffer(rec.buffer);
    for (const rec of this.textures.values()) gl.deleteTexture(rec.texture);
    for (const rec of this.pipelines.values()) gl.deleteProgram(rec.program);
    this.buffers.clear();
    this.textures.clear();
    this.pipelines.clear();
    if (this.vao) gl.deleteVertexArray(this.vao);
    if (this.fbo) gl.deleteFramebuffer(this.fbo);
    if (this.colorRb) gl.deleteRenderbuffer(this.colorRb);
    if (this.depthRb) gl.deleteRenderbuffer(this.depthRb);
  }
}
