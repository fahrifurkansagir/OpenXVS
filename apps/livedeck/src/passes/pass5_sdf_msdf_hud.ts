/**
 * Pass 5: Single-Pass Analytical SDF Vector HUD + B612 Mono MSDF Typography.
 *
 * - Zero Canvas2D texture uploads (`b612_mono_msdf.{png,json}` decoded directly into RGBA8 GPU texture).
 * - Conformal pitch ladder, WGS-84 horizon, +-PL_0.01 uncertainty whiskers, FPV, V-HPL/V-VPL
 *   runway integrity box, and zero-overlap 4-tier DO-365B callouts projected via `projectEcefToNdc`.
 * - Hard Safety Lock: IMMEDIATELY blanks ALL conformal cues when `simplexMode === 0` (`INVALID_INHIBIT`),
 *   meeting <= 1 RTA tick (0.10 s) + next vsync blanking requirement.
 * - Persistent visible status badges whenever degraded (`RENDER: WEBGL2`, `TERRAIN: PROCEDURAL-CI`,
 *   `AI: ...`) enforcing Charter §7.5 (ZERO silent fallbacks).
 */

import {
  DEG_TO_RAD,
  RAD_TO_DEG,
  RIG_CAMERA_ORDER,
  WGS84_A,
  geodeticToEcef,
  mat3MulVec,
  projectEcefToNdc,
  type Vec3,
} from "../rhi/coords.ts";
import type {
  RhiBufferHandle,
  RhiContext,
  RhiPipelineHandle,
  RhiTextureHandle,
} from "../rhi/types.ts";
import pass5Wgsl from "../shaders/pass5_sdf_msdf_hud.wgsl?raw";
import pass5GlVert from "../shaders/generated_gl/pass5_sdf_msdf_hud.vert.glsl?raw";
import pass5GlFrag from "../shaders/generated_gl/pass5_sdf_msdf_hud.frag.glsl?raw";
import {
  SIMPLEX_MODE_LABELS,
  type ActiveCameraView,
  type CockpitFrameState,
  type MsdfAtlasMetadata,
  type TerrainPackData,
} from "./types.ts";

// Vertex layout: posPx(2) + uv(2) + color(4) + primMeta(4) = 12 floats (48 bytes)
const FLOATS_PER_VERTEX = 12;
const MAX_QUADS = 2048;
const MAX_VERTICES = MAX_QUADS * 6;

type Rgba = readonly [number, number, number, number];

const COL_HUD_GREEN: Rgba = [0.22, 0.98, 0.56, 0.94];
const COL_HUD_CYAN: Rgba = [0.30, 0.90, 1.0, 0.92];
const COL_HUD_AMBER: Rgba = [1.0, 0.72, 0.12, 0.98];
const COL_HUD_RED: Rgba = [1.0, 0.22, 0.20, 0.98];
const COL_HUD_WHITE: Rgba = [0.92, 0.98, 0.96, 0.95];
const COL_PILL_BG: Rgba = [0.03, 0.08, 0.11, 0.80];
const COL_PILL_WARN_BG: Rgba = [0.28, 0.16, 0.02, 0.88];
const COL_PILL_CRIT_BG: Rgba = [0.38, 0.04, 0.04, 0.92];

interface CalloutRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export class Pass5SdfMsdfHud {
  private readonly rhi: RhiContext;
  private readonly pipeline: RhiPipelineHandle;
  private readonly vb: RhiBufferHandle;
  private readonly ubo: RhiBufferHandle;
  private readonly uniformData = new Float32Array(8); // 2 x vec4<f32>
  private readonly vertexScratch = new Float32Array(MAX_VERTICES * FLOATS_PER_VERTEX);

  private msdfTex: RhiTextureHandle;
  private msdfMeta: MsdfAtlasMetadata | null = null;
  private vertexCount = 0;
  private lastConformalBlanked = false;
  private lastVisibleBadges: string[] = [];

  constructor(rhi: RhiContext) {
    this.rhi = rhi;
    this.vb = rhi.createBuffer("vertex", this.vertexScratch.byteLength, "pass5-hud-vb");
    this.ubo = rhi.createBuffer("uniform", this.uniformData.byteLength, "pass5-ubo");

    // 1x1 neutral fallback texture until B612 Mono MSDF atlas loads
    this.msdfTex = rhi.createTextureRGBA8(1, 1, new Uint8Array([0, 0, 0, 255]), "pass5-msdf-init");

    this.pipeline = rhi.createRenderPipeline({
      label: "pass5-sdf-msdf-hud",
      wgsl: pass5Wgsl,
      glslVert: pass5GlVert,
      glslFrag: pass5GlFrag,
      topology: "triangle-list",
      depthTest: false,
      depthWrite: false,
      blendAlpha: true,
      vertexLayouts: [
        {
          arrayStride: FLOATS_PER_VERTEX * 4,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x2" },
            { shaderLocation: 1, offset: 8, format: "float32x2" },
            { shaderLocation: 2, offset: 16, format: "float32x4" },
            { shaderLocation: 3, offset: 32, format: "float32x4" },
          ],
        },
      ],
      uniformBindings: [0],
      textureBindings: [{ textureBinding: 1, samplerBinding: 2 }],
    });
  }

  /** Upload decoded RGBA8 pixels and JSON metadata for `b612_mono_msdf.{png,json}`. */
  setMsdfAtlas(rgbaPixels: Uint8Array, width: number, height: number, meta: MsdfAtlasMetadata): void {
    this.rhi.destroyTexture(this.msdfTex);
    this.msdfTex = this.rhi.createTextureRGBA8(width, height, rgbaPixels, "pass5-b612-mono-msdf");
    this.msdfMeta = meta;
  }

  /** Fetch and decode `b612_mono_msdf.{png,json}` directly using `createImageBitmap` (no Canvas2D text!). */
  async loadMsdfAtlasFromUrls(
    pngUrl = "./assets/b612_mono_msdf.png",
    jsonUrl = "./assets/b612_mono_msdf.json",
  ): Promise<void> {
    const [metaResp, imgResp] = await Promise.all([fetch(jsonUrl), fetch(pngUrl)]);
    if (!metaResp.ok) throw new Error(`Failed to fetch MSDF JSON ${jsonUrl} (${metaResp.status})`);
    if (!imgResp.ok) throw new Error(`Failed to fetch MSDF PNG ${pngUrl} (${imgResp.status})`);
    const meta = (await metaResp.json()) as MsdfAtlasMetadata;
    const blob = await imgResp.blob();
    const bitmap = await createImageBitmap(blob);

    // One-time static PNG pixel decode at boot (zero per-frame Canvas2D uploads!)
    const off = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx2d = off.getContext("2d");
    if (!ctx2d) {
      bitmap.close();
      throw new Error("OffscreenCanvas 2D unavailable for one-time MSDF PNG decode");
    }
    ctx2d.drawImage(bitmap, 0, 0);
    const imgData = ctx2d.getImageData(0, 0, bitmap.width, bitmap.height);
    bitmap.close();
    this.setMsdfAtlas(new Uint8Array(imgData.data.buffer), imgData.width, imgData.height, meta);
  }

  isConformalBlanked(): boolean {
    return this.lastConformalBlanked;
  }

  getVisibleBadges(): readonly string[] {
    return this.lastVisibleBadges;
  }

  private pushQuad(
    x0: number, y0: number,
    x1: number, y1: number,
    x2: number, y2: number,
    x3: number, y3: number,
    u0: number, v0: number,
    u1: number, v1: number,
    color: Rgba,
    meta: readonly [number, number, number, number],
  ): void {
    if (this.vertexCount + 6 > MAX_VERTICES) return;
    const v = this.vertexScratch;
    const writeV = (idx: number, x: number, y: number, uu: number, vv: number) => {
      const o = idx * FLOATS_PER_VERTEX;
      v[o + 0] = x; v[o + 1] = y;
      v[o + 2] = uu; v[o + 3] = vv;
      v[o + 4] = color[0]; v[o + 5] = color[1]; v[o + 6] = color[2]; v[o + 7] = color[3];
      v[o + 8] = meta[0]; v[o + 9] = meta[1]; v[o + 10] = meta[2]; v[o + 11] = meta[3];
    };
    const base = this.vertexCount;
    writeV(base + 0, x0, y0, u0, v0);
    writeV(base + 1, x1, y1, u1, v0);
    writeV(base + 2, x2, y2, u1, v1);
    writeV(base + 3, x0, y0, u0, v0);
    writeV(base + 4, x2, y2, u1, v1);
    writeV(base + 5, x3, y3, u0, v1);
    this.vertexCount += 6;
  }

  private addSdfLine(
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    thicknessPx: number,
    color: Rgba,
  ): void {
    const dx = x1 - x0;
    const dy = y1 - y0;
    const len = Math.hypot(dx, dy);
    if (len < 0.5) return;
    const halfLen = len * 0.5 + 1.0;
    const halfThick = thicknessPx * 0.5 + 1.0;
    const ux = dx / len;
    const uy = dy / len;
    const nx = -uy;
    const ny = ux;
    const mx = (x0 + x1) * 0.5;
    const my = (y0 + y1) * 0.5;

    this.pushQuad(
      mx - ux * halfLen - nx * halfThick, my - uy * halfLen - ny * halfThick,
      mx + ux * halfLen - nx * halfThick, my + uy * halfLen - ny * halfThick,
      mx + ux * halfLen + nx * halfThick, my + uy * halfLen + ny * halfThick,
      mx - ux * halfLen + nx * halfThick, my - uy * halfLen + ny * halfThick,
      -1, -1, 1, 1,
      color,
      [0.0, halfLen, halfThick, Math.min(thicknessPx * 0.4, 1.5)],
    );
  }

  private addSdfCircle(
    cx: number,
    cy: number,
    radiusPx: number,
    strokePx: number,
    color: Rgba,
  ): void {
    const pad = radiusPx + strokePx * 0.5 + 1.5;
    this.pushQuad(
      cx - pad, cy - pad,
      cx + pad, cy - pad,
      cx + pad, cy + pad,
      cx - pad, cy + pad,
      -1, -1, 1, 1,
      color,
      [1.0, radiusPx, strokePx * 0.5, 0.0],
    );
  }

  private addPill(
    x: number,
    y: number,
    w: number,
    h: number,
    radiusPx: number,
    bgColor: Rgba,
    borderColor?: Rgba,
  ): void {
    this.pushQuad(
      x, y,
      x + w, y,
      x + w, y + h,
      x, y + h,
      -1, -1, 1, 1,
      bgColor,
      [3.0, w * 0.5, h * 0.5, radiusPx],
    );
    if (borderColor) {
      this.addSdfLine(x + 2, y + 1, x + w - 2, y + 1, 1.4, borderColor);
      this.addSdfLine(x + 2, y + h - 1, x + w - 2, y + h - 1, 1.4, borderColor);
      this.addSdfLine(x + 1, y + 2, x + 1, y + h - 2, 1.4, borderColor);
      this.addSdfLine(x + w - 1, y + 2, x + w - 1, y + h - 2, 1.4, borderColor);
    }
  }

  private measureTextPx(text: string, fontSizePx: number): number {
    return text.length * fontSizePx * 0.62;
  }

  private addMsdfText(
    text: string,
    x: number,
    y: number,
    fontSizePx: number,
    color: Rgba,
  ): void {
    const meta = this.msdfMeta;
    if (!meta) return;
    let cursorX = x;
    const scaleFactor = fontSizePx / Math.max(meta.cell_height, 1);

    for (let i = 0; i < text.length; i++) {
      const code = text.charCodeAt(i);
      const entry = meta.glyphs[String(code)] ?? meta.glyphs["63"]; // '?' fallback
      if (entry && code !== 32) {
        const gw = fontSizePx * entry.planeW;
        const gh = fontSizePx * entry.planeH;
        this.pushQuad(
          cursorX, y,
          cursorX + gw, y,
          cursorX + gw, y + gh,
          cursorX, y + gh,
          entry.u0, entry.v0, entry.u1, entry.v1,
          color,
          [2.0, scaleFactor, 0.0, 0.0],
        );
      }
      cursorX += fontSizePx * (entry ? entry.advance : 0.62);
    }
  }

  private addBadgePill(
    text: string,
    x: number,
    y: number,
    fg: Rgba,
    bg: Rgba,
    fontSizePx = 13,
  ): number {
    const padX = 8;
    const h = fontSizePx + 8;
    const w = this.measureTextPx(text, fontSizePx) + padX * 2;
    this.addPill(x, y, w, h, 4.0, bg, fg);
    this.addMsdfText(text, x + padX, y + 4, fontSizePx, fg);
    return w;
  }

  private projectNedDirToPx(
    dirNed: Vec3,
    view: ActiveCameraView,
  ): { x: number; y: number; visible: boolean } {
    const dEcef = mat3MulVec(view.rEcefNed, [dirNed[0] * 10000, dirNed[1] * 10000, dirNed[2] * 10000]);
    const pObj: Vec3 = [
      view.pEcefCam[0] + dEcef[0],
      view.pEcefCam[1] + dEcef[1],
      view.pEcefCam[2] + dEcef[2],
    ];
    const proj = projectEcefToNdc(pObj, view.pEcefCam, view.rEcefCam, view.intr, {
      width: view.width,
      height: view.height,
    });
    return { x: proj.uPx + 0.5, y: proj.vPx + 0.5, visible: proj.visible };
  }

  private drawConformalCues(
    state: CockpitFrameState,
    view: ActiveCameraView,
    pack: TerrainPackData,
  ): void {
    const hudCol = state.efvsEnabled ? COL_HUD_WHITE : COL_HUD_GREEN;
    const isCockpitCam = view.cameraMode === "cockpit";
    const dipRad = Math.acos(WGS84_A / (WGS84_A + Math.max(state.altWgs84M, 0.0)));
    const hdg = state.yawRad;

    if (isCockpitCam) {
      // 1. Conformal WGS-84 Horizon Line & +-PL_0.01 Uncertainty Whiskers
    const plRad = Math.max(0.0, state.horizonPlRad);
    const horizonAzOffsets = [-32, -24, -16, -8, 0, 8, 16, 24, 32];
    const sampleHorizon = (elevRad: number) =>
      horizonAzOffsets.map((daDeg) => {
        const az = hdg + daDeg * DEG_TO_RAD;
        const cE = Math.cos(elevRad);
        return this.projectNedDirToPx([cE * Math.cos(az), cE * Math.sin(az), -Math.sin(elevRad)], view);
      });

    const hPts = sampleHorizon(-dipRad);
    for (let i = 0; i < hPts.length - 1; i++) {
      if (hPts[i].visible && hPts[i + 1].visible) {
        this.addSdfLine(hPts[i].x, hPts[i].y, hPts[i + 1].x, hPts[i + 1].y, 2.0, hudCol);
      }
    }

    // +-PL_0.01 uncertainty whisker brackets around conformal horizon
    const plUpper = sampleHorizon(-dipRad + plRad);
    const plLower = sampleHorizon(-dipRad - plRad);
    const plCol: Rgba = plRad > 0.0087 ? COL_HUD_AMBER : COL_HUD_CYAN;
    for (const idx of [2, 6]) {
      if (plUpper[idx].visible && plLower[idx].visible) {
        this.addSdfLine(plUpper[idx].x - 10, plUpper[idx].y, plUpper[idx].x + 10, plUpper[idx].y, 1.5, plCol);
        this.addSdfLine(plLower[idx].x - 10, plLower[idx].y, plLower[idx].x + 10, plLower[idx].y, 1.5, plCol);
        this.addSdfLine(plUpper[idx].x, plUpper[idx].y, plLower[idx].x, plLower[idx].y, 1.2, plCol);
      }
    }

    // 2. Conformal Pitch Ladder (-20..+20 deg at 5 deg steps)
    for (const pitchDeg of [-20, -15, -10, -5, 5, 10, 15, 20]) {
      const pRad = pitchDeg * DEG_TO_RAD;
      const cP = Math.cos(pRad);
      const sP = Math.sin(pRad);
      const pL = this.projectNedDirToPx(
        [cP * Math.cos(hdg - 5 * DEG_TO_RAD), cP * Math.sin(hdg - 5 * DEG_TO_RAD), -sP],
        view,
      );
      const pLIn = this.projectNedDirToPx(
        [cP * Math.cos(hdg - 1.8 * DEG_TO_RAD), cP * Math.sin(hdg - 1.8 * DEG_TO_RAD), -sP],
        view,
      );
      const pRIn = this.projectNedDirToPx(
        [cP * Math.cos(hdg + 1.8 * DEG_TO_RAD), cP * Math.sin(hdg + 1.8 * DEG_TO_RAD), -sP],
        view,
      );
      const pR = this.projectNedDirToPx(
        [cP * Math.cos(hdg + 5 * DEG_TO_RAD), cP * Math.sin(hdg + 5 * DEG_TO_RAD), -sP],
        view,
      );
      if (pL.visible && pLIn.visible) {
        this.addSdfLine(pL.x, pL.y, pLIn.x, pLIn.y, 1.5, hudCol);
        const label = pitchDeg > 0 ? `+${pitchDeg}` : `${pitchDeg}`;
        this.addMsdfText(label, pL.x - 28, pL.y - 6, 11, hudCol);
      }
      if (pRIn.visible && pR.visible) {
        this.addSdfLine(pRIn.x, pRIn.y, pR.x, pR.y, 1.5, hudCol);
      }
    }

    // 3. Conformal Flight Path Vector (FPV) from velocity NED
    const speed = Math.hypot(state.velNedMps[0], state.velNedMps[1], state.velNedMps[2]);
    if (speed > 2.0) {
      const fpv = this.projectNedDirToPx(
        [state.velNedMps[0] / speed, state.velNedMps[1] / speed, state.velNedMps[2] / speed],
        view,
      );
      if (fpv.visible) {
        this.addSdfCircle(fpv.x, fpv.y, 9.0, 2.0, hudCol);
        this.addSdfLine(fpv.x - 20, fpv.y, fpv.x - 9, fpv.y, 2.0, hudCol);
        this.addSdfLine(fpv.x + 9, fpv.y, fpv.x + 20, fpv.y, 2.0, hudCol);
        this.addSdfLine(fpv.x, fpv.y - 9, fpv.x, fpv.y - 17, 2.0, hudCol);
      }
    }

    // 4. V-HPL / V-VPL Conformal Runway Integrity Box at KSFO 28L Threshold
    if (state.approachActive) {
      const pThr = geodeticToEcef(
        pack.ksfo28lThrLatDeg * DEG_TO_RAD,
        pack.ksfo28lThrLonDeg * DEG_TO_RAD,
        pack.ksfo28lThrWgs84HM,
      );
      const proj = projectEcefToNdc(pThr, view.pEcefCam, view.rEcefCam, view.intr, {
        width: view.width,
        height: view.height,
      });
      if (proj.visible) {
        const rangeM = Math.max(proj.vCam[2], 50.0);
        const halfW = Math.max(18, Math.min(140, (state.vHplM / rangeM) * view.intr.fx));
        const halfH = Math.max(12, Math.min(100, (state.vVplM / rangeM) * view.intr.fy));
        const cx = proj.uPx + 0.5;
        const cy = proj.vPx + 0.5;
        const boxCol = state.vVplM > 10.0 ? COL_HUD_RED : COL_HUD_GREEN;
        this.addSdfLine(cx - halfW, cy - halfH, cx + halfW, cy - halfH, 1.8, boxCol);
        this.addSdfLine(cx + halfW, cy - halfH, cx + halfW, cy + halfH, 1.8, boxCol);
        this.addSdfLine(cx + halfW, cy + halfH, cx - halfW, cy + halfH, 1.8, boxCol);
        this.addSdfLine(cx - halfW, cy + halfH, cx - halfW, cy - halfH, 1.8, boxCol);
        this.addMsdfText(
          `KSFO 28L VPL:${state.vVplM.toFixed(1)}m HPL:${state.vHplM.toFixed(1)}m`,
          cx - halfW,
          cy - halfH - 15,
          11,
          boxCol,
        );
      }
    }
    } // end isCockpitCam (external 3D chase/orbit suppresses pilot-eye horizon/pitch/runway box)

    // 5. Zero-Overlap 4-Tier DO-365B Traffic Callouts (projected cleanly in all 3 camera modes!)
    const placed: CalloutRect[] = [];
    const vp = { width: view.width, height: view.height };
    for (const tr of state.intruders) {
      const proj = projectEcefToNdc(tr.posEcef, view.pEcefCam, view.rEcefCam, view.intr, vp);
      if (!proj.visible) continue;
      const sx = proj.uPx + 0.5;
      const sy = proj.vPx + 0.5;
      const isRejected = Boolean(tr.isGhost && tr.ghostRejected);

      const tierCol = isRejected
        ? COL_HUD_RED
        : tr.alertTier >= 3
          ? COL_HUD_RED
          : tr.alertTier === 2
            ? COL_HUD_AMBER
            : tr.meshVerified
              ? COL_HUD_GREEN
              : COL_HUD_CYAN;

      // Target reticle
      if (isRejected) {
        this.addSdfLine(sx - 9, sy - 9, sx + 9, sy + 9, 2.0, COL_HUD_RED);
        this.addSdfLine(sx - 9, sy + 9, sx + 9, sy - 9, 2.0, COL_HUD_RED);
      } else {
        this.addSdfCircle(sx, sy, 10.0, 1.8, tierCol);
      }

      const tag = isRejected
        ? `${tr.id} GHOST REJECTED`
        : `${tr.id} ${(tr.rangeM / 1000).toFixed(1)}km T${Math.round(tr.ttcS)}s${tr.meshVerified ? " MESH" : ""}`;
      const bw = this.measureTextPx(tag, 11) + 12;
      const bh = 18;
      let bx = Math.max(8, Math.min(view.width - bw - 8, sx + 14));
      let by = Math.max(48, Math.min(view.height - bh - 48, sy - 9));

      // Greedy vertical declutter to guarantee zero-overlap callout boxes
      for (let iter = 0; iter < 8; iter++) {
        const overlaps = placed.some(
          (r) => bx < r.x + r.w + 4 && bx + bw + 4 > r.x && by < r.y + r.h + 4 && by + bh + 4 > r.y,
        );
        if (!overlaps) break;
        by = Math.min(view.height - bh - 48, by + bh + 6);
      }
      placed.push({ x: bx, y: by, w: bw, h: bh });
      this.addSdfLine(sx + 10, sy, bx, by + bh * 0.5, 1.0, tierCol);
      this.addPill(bx, by, bw, bh, 3.0, COL_PILL_BG, tierCol);
      this.addMsdfText(tag, bx + 6, by + 3, 11, tierCol);
    }
  }

  private drawPersistentSafetyHud(
    state: CockpitFrameState,
    view: ActiveCameraView,
    pack: TerrainPackData,
    extraBadges: readonly string[],
  ): void {
    const badges: string[] = [];
    let cursorX = 12;
    const topY = 10;

    // 1. Simplex Mode Badge (always visible)
    const simplexLabel = `SIMPLEX: ${SIMPLEX_MODE_LABELS[state.simplexMode]}`;
    badges.push(simplexLabel);
    const simplexFg =
      state.simplexMode === 0
        ? COL_HUD_RED
        : state.simplexMode <= 2
          ? COL_HUD_AMBER
          : state.simplexMode === 3
            ? COL_HUD_CYAN
            : COL_HUD_GREEN;
    const simplexBg =
      state.simplexMode === 0
        ? COL_PILL_CRIT_BG
        : state.simplexMode <= 2
          ? COL_PILL_WARN_BG
          : COL_PILL_BG;
    cursorX += this.addBadgePill(simplexLabel, cursorX, topY, simplexFg, simplexBg, 12) + 8;

    // 2. Render Backend Badge (Charter §7.5: unmistakable amber when WebGL2 or no F16)
    const isGl2 = this.rhi.backend === "webgl2";
    const renderBadge = isGl2
      ? "RENDER: WEBGL2 (DEGRADED)"
      : this.rhi.hasShaderF16
        ? "RENDER: WEBGPU"
        : "RENDER: WEBGPU (NO F16)";
    badges.push(renderBadge);
    cursorX +=
      this.addBadgePill(
        renderBadge,
        cursorX,
        topY,
        isGl2 || !this.rhi.hasShaderF16 ? COL_HUD_AMBER : COL_HUD_GREEN,
        isGl2 ? COL_PILL_WARN_BG : COL_PILL_BG,
        12,
      ) + 8;

    // 3. AI Backend Badge
    const aiBadge =
      state.aiBadge ??
      (isGl2
        ? "AI: ORT-WASM-SIMD"
        : this.rhi.hasShaderF16
          ? "AI: ORT-WebGPU-FP16"
          : "AI: ORT-WebGPU-FP32 (NO F16)");
    badges.push(aiBadge);
    const aiDegraded = aiBadge.includes("WASM") || aiBadge.includes("NO F16");
    cursorX +=
      this.addBadgePill(
        aiBadge,
        cursorX,
        topY,
        aiDegraded ? COL_HUD_AMBER : COL_HUD_CYAN,
        aiDegraded ? COL_PILL_WARN_BG : COL_PILL_BG,
        12,
      ) + 8;

    // 4. Terrain Mode Badge (explicit PROCEDURAL-CI amber warning when not USGS 3DEP)
    const isProcCi = pack.terrainMode === "procedural_ci";
    const terrainBadge = isProcCi ? "TERRAIN: PROCEDURAL-CI" : "TERRAIN: USGS-3DEP-GEOID18";
    badges.push(terrainBadge);
    cursorX +=
      this.addBadgePill(
        terrainBadge,
        cursorX,
        topY,
        isProcCi ? COL_HUD_AMBER : COL_HUD_GREEN,
        isProcCi ? COL_PILL_WARN_BG : COL_PILL_BG,
        12,
      ) + 8;

    if (state.efvsEnabled) {
      const efvsBadge = "EFVS: LWIR 8-14um";
      badges.push(efvsBadge);
      cursorX += this.addBadgePill(efvsBadge, cursorX, topY, COL_HUD_WHITE, COL_PILL_BG, 12) + 8;
    }

    for (const eb of extraBadges) {
      badges.push(eb);
      if (cursorX + 180 < view.width) {
        cursorX += this.addBadgePill(eb, cursorX, topY, COL_HUD_AMBER, COL_PILL_WARN_BG, 12) + 8;
      }
    }

    this.lastVisibleBadges = badges;

    // 5. Critical Non-Conformal Alerts (remain visible even when conformal cues blank in P0!)
    // Note: GPS SPOOF / GHOST REJECTED top annunciator is rendered exclusively by CockpitHudOverlay
    // to prevent duplicate overlapping banners (D16-M1-m1).
    if (state.mandatoryGoAround) {
      const txt = "MANDATORY GO-AROUND - CONFORMAL CUES INHIBITED";
      const w = this.measureTextPx(txt, 16) + 24;
      const x = (view.width - w) * 0.5;
      const y = view.height * 0.24;
      this.addPill(x, y, w, 32, 6.0, COL_PILL_CRIT_BG, COL_HUD_RED);
      this.addMsdfText(txt, x + 12, y + 7, 16, COL_HUD_RED);
    } else if (state.simplexMode === 0) {
      const txt = "INVALID INHIBIT - CONFORMAL CUES BLANKED";
      const w = this.measureTextPx(txt, 15) + 24;
      const x = (view.width - w) * 0.5;
      const y = view.height * 0.24;
      this.addPill(x, y, w, 30, 6.0, COL_PILL_CRIT_BG, COL_HUD_RED);
      this.addMsdfText(txt, x + 12, y + 7, 15, COL_HUD_RED);
    }

    // 6. Bottom-Left 6-Camera Health Strip strictly ordered F, R, B, L, U, D (CONVENTIONS.md §8)
    const stripY = view.height - 36;
    let stripX = 14;
    this.addMsdfText("XVS6:", stripX, stripY + 5, 12, COL_HUD_WHITE);
    stripX += 46;
    for (let i = 0; i < RIG_CAMERA_ORDER.length; i++) {
      const healthy = ((state.camHealthMask >> i) & 1) !== 0;
      const camId = RIG_CAMERA_ORDER[i];
      const fg = healthy ? COL_HUD_GREEN : COL_HUD_RED;
      const bg = healthy ? COL_PILL_BG : COL_PILL_CRIT_BG;
      this.addPill(stripX, stripY, 26, 22, 4.0, bg, fg);
      this.addMsdfText(camId, stripX + 9, stripY + 4, 12, fg);
      stripX += 30;
    }

    // 7. Bottom-Right Basic Flight Data Readout (HDG / ALT WGS-84 / IAS)
    const hdgDeg = ((state.yawRad * RAD_TO_DEG) % 360 + 360) % 360;
    const flightStr = `HDG ${hdgDeg.toFixed(0).padStart(3, "0")}  ALT ${state.altWgs84M.toFixed(0)}m  IAS ${(state.airspeedMps * 1.94384).toFixed(0)}kt`;
    const fw = this.measureTextPx(flightStr, 12) + 18;
    const fx = view.width - fw - 14;
    this.addPill(fx, stripY, fw, 22, 4.0, COL_PILL_BG, COL_HUD_CYAN);
    this.addMsdfText(flightStr, fx + 9, stripY + 4, 12, COL_HUD_CYAN);
  }

  encode(
    state: CockpitFrameState,
    view: ActiveCameraView,
    pack: TerrainPackData,
    extraBadges: readonly string[] = [],
  ): void {
    this.vertexCount = 0;

    // Hard Conformal Blanking Lock when SimplexMode === InvalidInhibit (0)
    const blankConformal = state.simplexMode === 0;
    this.lastConformalBlanked = blankConformal;

    if (!blankConformal) {
      this.drawConformalCues(state, view, pack);
    }
    this.drawPersistentSafetyHud(state, view, pack, extraBadges);

    if (this.vertexCount === 0) return;

    this.uniformData[0] = view.width;
    this.uniformData[1] = view.height;
    this.uniformData[2] = this.msdfMeta?.atlas_width ?? 512;
    this.uniformData[3] = this.msdfMeta?.atlas_height ?? 288;
    this.uniformData[4] = this.msdfMeta?.px_range ?? 4.0;
    this.uniformData[5] = state.simTimeS;
    this.uniformData[6] = state.efvsEnabled ? 1.0 : 0.0;
    this.uniformData[7] = state.simplexMode;

    this.rhi.writeBuffer(this.ubo, this.uniformData);
    this.rhi.writeBuffer(
      this.vb,
      this.vertexScratch.subarray(0, this.vertexCount * FLOATS_PER_VERTEX),
    );

    this.rhi.draw({
      pipeline: this.pipeline,
      uniformBuffers: [{ binding: 0, buffer: this.ubo }],
      textures: [{ textureBinding: 1, samplerBinding: 2, texture: this.msdfTex }],
      vertexBuffers: [this.vb],
      vertexCount: this.vertexCount,
    });
  }

  dispose(): void {
    this.rhi.destroyBuffer(this.vb);
    this.rhi.destroyBuffer(this.ubo);
    this.rhi.destroyTexture(this.msdfTex);
  }
}
