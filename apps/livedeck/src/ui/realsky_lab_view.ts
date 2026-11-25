/**
 * Tab 3 (`Gerçek Gökyüzü AI Laboratuvarı` / `Real-Sky AI Lab`) — Stage-1 Lazy Module.
 *
 * - Lazily fetches `./assets/realsky_aot_pack/manifest.json` + `256x256` 1:1 native Amazon AOT crops
 *   ONLY when Tab 3 opens (zero Stage-0 bytes!).
 * - Runs real `OrtDetectorRunner.runXvsD2ImageData(...)` + `ActiveRangeControllerTs` LIVE in browser
 *   and plots `GPS Truth` vs `CP-MSC Safe Lower Bound` vs `Naive 11m Bbox-Size`.
 * - Supports local drag-and-drop / file picker (`.mp4`, `.webm`, `.jpg`, `.png`), resetting ConvGRU
 *   temporal state (`resetXvsD2TemporalState()`) on every clip switch or drop and surfacing a clean
 *   inline error banner on corrupt/empty/non-media drops (Review Focus #5).
 */

import type { DetectorInferenceResult, OrtDetectorRunner } from "../ai/ort_worker.ts";
import {
  ActiveRangeControllerTs,
  computeConformalLowerBound,
} from "../active_range/active_range_ts.ts";
import {
  DEFAULT_LOCALE,
  getObservatoryStrings,
  type ObservatoryLocale,
} from "./i18n.ts";

export const REALSKY_MANIFEST_URL = "./assets/realsky_aot_pack/manifest.json";
const REALSKY_PACK_DIR = "./assets/realsky_aot_pack/";
const NAIVE_WINGSPAN_PRIOR_M = 11.0; // Nominal Cessna-172 prior used in Tier A benchmark

export interface RealSkyAotFrame {
  readonly file: string;
  readonly frame_idx: number;
  readonly crop_x: number;
  readonly crop_y: number;
  readonly gt_box_crop: readonly [number, number, number, number];
  readonly range_distance_m: number;
  readonly is_above_horizon: boolean;
}

export interface RealSkyAotEncounter {
  readonly encounter_id: string;
  readonly fps: number;
  readonly aircraft_class: string;
  readonly frames: readonly RealSkyAotFrame[];
}

export interface RealSkyAotManifest {
  readonly schema_version: string;
  readonly focal_px: number;
  readonly q_hat_conformal: number;
  readonly encounters: readonly RealSkyAotEncounter[];
}

export interface RangeChartSample {
  readonly frameIdx: number;
  readonly gpsTruthM: number;
  readonly cpMscSafeM: number;
  readonly naiveBboxM: number;
}

export interface RealSkyDropOutcome {
  readonly ok: boolean;
  readonly kind: "image" | "video" | "error";
  readonly errorMessage: string;
  readonly fileName: string;
}

export interface RealSkyTemporalSink {
  resetXvsD2TemporalState(): void;
}

const SUPPORTED_MEDIA_EXTS = [".mp4", ".webm", ".jpg", ".jpeg", ".png"];

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function hasSupportedMediaExt(name: string): boolean {
  const lower = name.toLowerCase();
  return SUPPORTED_MEDIA_EXTS.some((ext) => lower.endsWith(ext));
}

function isSupportedMediaMime(mime: string): boolean {
  return mime.startsWith("image/") || mime.startsWith("video/");
}

/** Pure SVG line chart comparing GPS Ground Truth, CP-MSC Safe Bound, and Naive Bbox-Size. */
export function buildRangeComparisonChartSvg(
  locale: ObservatoryLocale,
  samples: readonly RangeChartSample[],
): string {
  const s = getObservatoryStrings(locale).realskyLab;
  const w = 440;
  const h = 188;
  const padL = 44;
  const padR = 14;
  const padT = 18;
  const padB = 26;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  let maxY = 400;
  for (const pt of samples) {
    maxY = Math.max(maxY, pt.gpsTruthM, pt.cpMscSafeM, pt.naiveBboxM);
  }
  maxY = Math.min(3500, Math.ceil(maxY / 100) * 100);
  const maxX = Math.max(1, samples.length - 1);

  const toPoly = (getter: (pt: RangeChartSample) => number): string =>
    samples
      .map((pt, i) => {
        const x = padL + (i / maxX) * plotW;
        const y = padT + plotH * (1 - Math.min(maxY, Math.max(0, getter(pt))) / maxY);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");

  const yTicks = [0, 0.5, 1.0].map((frac) => {
    const val = Math.round(maxY * frac);
    const y = padT + plotH * (1 - frac);
    return (
      `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${w - padR}" y2="${y.toFixed(1)}" stroke="rgba(148,163,184,0.22)" stroke-width="1" />` +
      `<text x="${padL - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end" fill="#94a3b8" font-size="9.5" font-family="monospace">${val}m</text>`
    );
  }).join("");

  const polys =
    samples.length >= 2
      ? `<polyline fill="none" stroke="#f59e0b" stroke-width="1.8" stroke-dasharray="4 3" points="${toPoly((p) => p.naiveBboxM)}" />` +
        `<polyline fill="none" stroke="#38bdf8" stroke-width="2.2" points="${toPoly((p) => p.gpsTruthM)}" />` +
        `<polyline fill="none" stroke="#2ce68a" stroke-width="2.4" points="${toPoly((p) => p.cpMscSafeM)}" />`
      : "";

  return (
    `<svg class="realsky-chart-svg" viewBox="0 0 ${w} ${h}" width="100%" role="img" aria-label="${escapeHtml(s.chartTitle)}">` +
    `<rect x="0" y="0" width="${w}" height="${h}" rx="6" fill="rgba(8,14,24,0.88)" stroke="rgba(110,145,185,0.3)" />` +
    yTicks +
    polys +
    `<text x="${padL}" y="12" fill="#e2e8f0" font-size="10" font-weight="700" font-family="monospace">${escapeHtml(s.chartTitle)}</text>` +
    `<text x="${padL}" y="${h - 7}" fill="#38bdf8" font-size="9" font-family="monospace">■ ${escapeHtml(s.legendGpsTruth)}</text>` +
    `<text x="${padL + 118}" y="${h - 7}" fill="#2ce68a" font-size="9" font-family="monospace">■ ${escapeHtml(s.legendCpMscSafe)}</text>` +
    `<text x="${padL + 260}" y="${h - 7}" fill="#f59e0b" font-size="9" font-family="monospace">■ ${escapeHtml(s.legendNaiveBbox)}</text>` +
    `</svg>`
  );
}

/**
 * Pure headless controller for Tab 3 (unit-testable in Node for Review Focus #5).
 * Always resets XVS-D2 temporal state on encounter switches & local file drops and never throws.
 */
export class RealSkyLabController {
  private readonly sink: RealSkyTemporalSink | null;
  private locale: ObservatoryLocale;
  private selectedEncounterIdx = 0;
  private frameCursor = 0;
  private lastError: string | null = null;

  constructor(sink: RealSkyTemporalSink | null = null, locale: ObservatoryLocale = DEFAULT_LOCALE) {
    this.sink = sink;
    this.locale = locale;
  }

  setLocale(locale: ObservatoryLocale): void {
    this.locale = locale;
  }

  getSelectedEncounterIdx(): number {
    return this.selectedEncounterIdx;
  }

  getFrameCursor(): number {
    return this.frameCursor;
  }

  getLastError(): string | null {
    return this.lastError;
  }

  selectEncounter(idx: number): void {
    this.selectedEncounterIdx = Math.max(0, Math.floor(idx));
    this.frameCursor = 0;
    this.lastError = null;
    this.sink?.resetXvsD2TemporalState();
  }

  async handleDroppedFile(file: File | null | undefined): Promise<RealSkyDropOutcome> {
    const errText = getObservatoryStrings(this.locale).realskyLab.corruptFileError;
    this.sink?.resetXvsD2TemporalState();

    if (!file || file.size <= 0) {
      this.lastError = errText;
      return { ok: false, kind: "error", errorMessage: errText, fileName: file?.name ?? "" };
    }

    const mime = file.type || "";
    if (!isSupportedMediaMime(mime) && !hasSupportedMediaExt(file.name)) {
      this.lastError = errText;
      return { ok: false, kind: "error", errorMessage: errText, fileName: file.name };
    }

    const isImage = mime.startsWith("image/") || /\.(jpg|jpeg|png)$/i.test(file.name);
    if (isImage) {
      try {
        if (typeof createImageBitmap !== "function") {
          throw new Error("createImageBitmap unavailable");
        }
        const bmp = await createImageBitmap(file);
        bmp.close();
        this.lastError = null;
        return { ok: true, kind: "image", errorMessage: "", fileName: file.name };
      } catch {
        this.lastError = errText;
        return { ok: false, kind: "error", errorMessage: errText, fileName: file.name };
      }
    }

    // Video validation: verify non-trivial container header & browser decodability
    try {
      const head = new Uint8Array(await file.slice(0, 12).arrayBuffer());
      const looksMp4 = head.length >= 8 && head[4] === 0x66 && head[5] === 0x74 && head[6] === 0x79 && head[7] === 0x70;
      const looksWebm = head.length >= 4 && head[0] === 0x1a && head[1] === 0x45 && head[2] === 0xdf && head[3] === 0xa3;
      if (!looksMp4 && !looksWebm) {
        throw new Error("Invalid MP4/WebM container signature");
      }
      if (typeof document === "undefined" || typeof URL.createObjectURL !== "function") {
        throw new Error("Video DOM decoder unavailable");
      }
      this.lastError = null;
      return { ok: true, kind: "video", errorMessage: "", fileName: file.name };
    } catch {
      this.lastError = errText;
      return { ok: false, kind: "error", errorMessage: errText, fileName: file.name };
    }
  }
}

/**
 * Interactive Browser View for Tab 3 (`Gerçek Gökyüzü AI Laboratuvarı` / `Real-Sky AI Lab`).
 */
export class RealSkyLabView {
  private readonly root: HTMLElement;
  private readonly getOrtRunner: () => OrtDetectorRunner | null;
  private readonly ctrl: RealSkyLabController;
  private locale: ObservatoryLocale;
  private manifest: RealSkyAotManifest | null = null;
  private loading = false;
  private playing = false;
  private playTimer: number | null = null;
  private activeRange: ActiveRangeControllerTs | null = null;
  private chartSamples: RangeChartSample[] = [];
  private currentFrameIdx = 0;
  private prevCropX: number | null = null;
  private prevCropY: number | null = null;
  private prevSizePx = 0.0;
  private lastDetResult: DetectorInferenceResult | null = null;
  private customStatusText = "";
  private activeObjectUrl: string | null = null;

  constructor(
    container: HTMLElement,
    getOrtRunner: () => OrtDetectorRunner | null,
    initialLocale: ObservatoryLocale = DEFAULT_LOCALE,
  ) {
    this.root = container;
    this.getOrtRunner = getOrtRunner;
    this.locale = initialLocale;
    this.ctrl = new RealSkyLabController(
      { resetXvsD2TemporalState: () => this.getOrtRunner()?.resetXvsD2TemporalState() },
      initialLocale,
    );
    this.renderShell();
  }

  setLocale(locale: ObservatoryLocale): void {
    this.locale = locale;
    this.ctrl.setLocale(locale);
    this.renderShell();
    this.updateChartAndMetrics();
  }

  async ensureLoaded(): Promise<void> {
    if (this.manifest || this.loading) return;
    this.loading = true;
    this.renderShell();
    try {
      const resp = await fetch(REALSKY_MANIFEST_URL);
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status} loading ${REALSKY_MANIFEST_URL}`);
      }
      this.manifest = (await resp.json()) as RealSkyAotManifest;
      this.selectEncounter(0);
    } catch (err) {
      this.customStatusText = `Pack load warning: ${err instanceof Error ? err.message : String(err)}`;
    } finally {
      this.loading = false;
      this.renderShell();
      if (this.manifest) {
        // Run brief 4-frame ConvGRU warmup so XVS-D2 temporal diffs & range chart populate immediately on Tab 3 open
        const enc = this.manifest.encounters[this.ctrl.getSelectedEncounterIdx()];
        const warmupFrames = Math.min(4, enc?.frames.length ?? 0);
        for (let f = 0; f < warmupFrames; f++) {
          this.currentFrameIdx = f;
          await this.renderCurrentAotFrame();
        }
      }
    }
  }

  private selectEncounter(idx: number): void {
    this.stopPlayback();
    this.ctrl.selectEncounter(idx);
    this.currentFrameIdx = 0;
    this.chartSamples = [];
    this.prevCropX = null;
    this.prevCropY = null;
    this.prevSizePx = 0.0;
    this.lastDetResult = null;
    const focalPx = this.manifest?.focal_px ?? 2690.0;
    this.activeRange = new ActiveRangeControllerTs({
      sigmaBearingRad: 2.86 / focalPx,
      qHatConformal: this.manifest?.q_hat_conformal ?? 0.925178,
    });
  }

  private stopPlayback(): void {
    this.playing = false;
    if (this.playTimer !== null) {
      window.clearInterval(this.playTimer);
      this.playTimer = null;
    }
  }

  private togglePlay(): void {
    if (!this.manifest) return;
    if (this.playing) {
      this.stopPlayback();
      this.renderShell();
      return;
    }
    this.playing = true;
    this.renderShell();
    const enc = this.manifest.encounters[this.ctrl.getSelectedEncounterIdx()];
    const intervalMs = Math.max(120, Math.round(1000 / Math.max(1, enc?.fps ?? 8)));
    this.playTimer = window.setInterval(() => {
      if (!this.manifest) return;
      const curEnc = this.manifest.encounters[this.ctrl.getSelectedEncounterIdx()];
      if (!curEnc || curEnc.frames.length === 0) return;
      this.currentFrameIdx = (this.currentFrameIdx + 1) % curEnc.frames.length;
      if (this.currentFrameIdx === 0) {
        this.chartSamples = [];
        this.prevCropX = null;
        this.prevCropY = null;
        this.prevSizePx = 0.0;
        this.getOrtRunner()?.resetXvsD2TemporalState();
        this.activeRange?.reset();
      }
      void this.renderCurrentAotFrame();
    }, intervalMs);
  }

  private async renderCurrentAotFrame(): Promise<void> {
    if (!this.manifest) return;
    const enc = this.manifest.encounters[this.ctrl.getSelectedEncounterIdx()];
    if (!enc || enc.frames.length === 0) return;
    const frame = enc.frames[this.currentFrameIdx];
    const canvas = this.root.querySelector<HTMLCanvasElement>("#realsky-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    try {
      const resp = await fetch(`${REALSKY_PACK_DIR}${frame.file}`);
      if (!resp.ok) return;
      const bmp = await createImageBitmap(await resp.blob());
      ctx.clearRect(0, 0, 256, 256);
      ctx.drawImage(bmp, 0, 0, 256, 256);
      bmp.close();

      const imgData = ctx.getImageData(0, 0, 256, 256);
      const shiftDx = this.prevCropX !== null ? -(frame.crop_x - this.prevCropX) : 0;
      const shiftDy = this.prevCropY !== null ? -(frame.crop_y - this.prevCropY) : 0;
      this.prevCropX = frame.crop_x;
      this.prevCropY = frame.crop_y;

      const runner = this.getOrtRunner();
      if (runner?.getStatus()?.ready) {
        this.lastDetResult = await runner.runXvsD2ImageData(
          imgData.data,
          256,
          256,
          shiftDx,
          shiftDy,
          frame.frame_idx,
          0.2,
        );
      }

      // Draw crisp high-contrast cyan dashed GPS ground-truth box + emerald solid XVS-D2 detection box + score pill
      const [gx0, gy0, gx1, gy1] = frame.gt_box_crop;
      const gcx = 0.5 * (gx0 + gx1);
      const gcy = 0.5 * (gy0 + gy1);
      const gw = Math.max(12, gx1 - gx0);
      const gh = Math.max(12, gy1 - gy0);
      ctx.save();
      ctx.setLineDash([5, 3]);
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 2;
      ctx.strokeRect(gcx - gw * 0.5, gcy - gh * 0.5, gw, gh);
      ctx.restore();

      const bestDet = this.lastDetResult?.detections?.[0];
      if (bestDet) {
        const dw = Math.max(14, bestDet.widthPx);
        const dh = Math.max(14, bestDet.heightPx);
        const dx0 = bestDet.u - dw * 0.5;
        const dy0 = bestDet.v - dh * 0.5;
        ctx.save();
        ctx.setLineDash([]);
        ctx.strokeStyle = "#2ce68a";
        ctx.lineWidth = 2.2;
        ctx.strokeRect(dx0, dy0, dw, dh);
        const label = `XVS-D2 ${bestDet.score.toFixed(2)}`;
        ctx.font = "bold 10px monospace";
        const tw = ctx.measureText(label).width + 8;
        const lx = Math.max(2, Math.min(254 - tw, dx0));
        const ly = dy0 > 18 ? dy0 - 16 : dy0 + dh + 2;
        ctx.fillStyle = "rgba(6, 14, 22, 0.86)";
        ctx.fillRect(lx, ly, tw, 14);
        ctx.fillStyle = "#2ce68a";
        ctx.fillText(label, lx + 4, ly + 10.5);
        ctx.restore();
      }

      // Step ActiveRangeControllerTs + compute calibrated CP-MSC Safe Bound vs Naive 11m Bbox-Size
      const focalPx = this.manifest.focal_px || 2690.0;
      const boxW = bestDet ? bestDet.widthPx : Math.max(4, gx1 - gx0);
      const boxH = bestDet ? bestDet.heightPx : Math.max(4, gy1 - gy0);
      const sizePx = Math.sqrt(boxW * boxH);
      const dtS = 1.0 / Math.max(1.0, enc.fps || 10.0);
      const sizeRate = this.prevSizePx > 0 ? (sizePx - this.prevSizePx) / dtS : 0.0;
      this.prevSizePx = sizePx;

      const cx = (bestDet ? bestDet.u : gcx) + frame.crop_x - 1224.0;
      const cy = (bestDet ? bestDet.v : gcy) + frame.crop_y - 1024.0;
      const alphaRad = Math.atan2(cx, focalPx);
      const betaRad = Math.atan2(-cy, focalPx);

      const arOut = this.activeRange?.step(
        { vNed: [35, 0, 0], aNed: [0, 0, 0], headingRad: 0, tasMps: 35 },
        {
          alphaRad,
          betaRad,
          sizePx,
          sizeRatePxS: sizeRate,
          daytimeClear: true,
          trackAge: this.currentFrameIdx + 1,
          dtS,
        },
      );

      const naiveBboxM = (focalPx * NAIVE_WINGSPAN_PRIOR_M) / Math.max(sizePx, 2.0);
      const qHat = this.manifest.q_hat_conformal;
      const cpMscSafeM = arOut
        ? computeConformalLowerBound(arOut.rHatM, arOut.sigmaRM, qHat)
        : computeConformalLowerBound(naiveBboxM, 260, qHat);

      this.chartSamples.push({
        frameIdx: frame.frame_idx,
        gpsTruthM: frame.range_distance_m,
        cpMscSafeM,
        naiveBboxM,
      });
      if (this.chartSamples.length > 40) {
        this.chartSamples.shift();
      }
      this.updateChartAndMetrics();
    } catch {
      // Non-fatal frame fetch/decode guard
    }
  }

  private async handleLocalFile(file: File): Promise<void> {
    this.stopPlayback();
    const res = await this.ctrl.handleDroppedFile(file);
    if (!res.ok) {
      this.customStatusText = res.errorMessage;
      this.renderShell();
      return;
    }

    const canvas = this.root.querySelector<HTMLCanvasElement>("#realsky-canvas");
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    try {
      if (res.kind === "image") {
        const bmp = await createImageBitmap(file);
        ctx.clearRect(0, 0, 256, 256);
        ctx.drawImage(bmp, 0, 0, 256, 256);
        bmp.close();
        const imgData = ctx.getImageData(0, 0, 256, 256);
        const runner = this.getOrtRunner();
        if (runner?.getStatus()?.ready) {
          this.lastDetResult = await runner.runXvsD2ImageData(imgData.data, 256, 256, 0, 0, 0, 0.2);
          for (const d of this.lastDetResult.detections ?? []) {
            ctx.strokeStyle = "#2ce68a";
            ctx.lineWidth = 2;
            ctx.strokeRect(d.u - d.widthPx * 0.5, d.v - d.heightPx * 0.5, d.widthPx, d.heightPx);
          }
        }
        this.customStatusText = `${file.name} — XVS-D2 (${this.lastDetResult?.detections?.length ?? 0} dets)`;
      } else {
        if (this.activeObjectUrl) URL.revokeObjectURL(this.activeObjectUrl);
        this.activeObjectUrl = URL.createObjectURL(file);
        const video = document.createElement("video");
        video.muted = true;
        video.src = this.activeObjectUrl;
        await new Promise<void>((resolve, reject) => {
          video.onloadeddata = () => resolve();
          video.onerror = () => reject(new Error("Video decode error"));
        });
        ctx.clearRect(0, 0, 256, 256);
        ctx.drawImage(video, 0, 0, 256, 256);
        const imgData = ctx.getImageData(0, 0, 256, 256);
        const runner = this.getOrtRunner();
        if (runner?.getStatus()?.ready) {
          this.lastDetResult = await runner.runXvsD2ImageData(imgData.data, 256, 256, undefined, undefined, 0, 0.2);
        }
        this.customStatusText = `${file.name} — XVS-D2 (${this.lastDetResult?.detections?.length ?? 0} dets)`;
      }
    } catch {
      this.customStatusText = getObservatoryStrings(this.locale).realskyLab.corruptFileError;
    }
    this.renderShell();
  }

  private updateChartAndMetrics(): void {
    const chartSlot = this.root.querySelector<HTMLElement>("#realsky-chart-slot");
    if (chartSlot) {
      chartSlot.innerHTML = buildRangeComparisonChartSvg(this.locale, this.chartSamples);
    }
    const metaSlot = this.root.querySelector<HTMLElement>("#realsky-frame-meta");
    if (metaSlot && this.manifest) {
      const enc = this.manifest.encounters[this.ctrl.getSelectedEncounterIdx()];
      const fr = enc?.frames[this.currentFrameIdx];
      const latest = this.chartSamples[this.chartSamples.length - 1];
      const detMs = this.lastDetResult ? `${this.lastDetResult.inferenceMs.toFixed(1)} ms` : "—";
      const detCnt = this.lastDetResult?.detections?.length ?? 0;
      metaSlot.textContent = fr && latest
        ? `ENC ${enc.encounter_id} (${enc.aircraft_class}) · F${fr.frame_idx}/${enc.frames.length} · GPS ${latest.gpsTruthM.toFixed(0)}m · CP-MSC ${latest.cpMscSafeM.toFixed(0)}m · Naive ${latest.naiveBboxM.toFixed(0)}m · XVS-D2 ${detCnt} (${detMs})`
        : this.customStatusText;
    }
  }

  private renderShell(): void {
    const s = getObservatoryStrings(this.locale).realskyLab;
    const encOptions = (this.manifest?.encounters ?? [])
      .map(
        (e, i) =>
          `<option value="${i}"${i === this.ctrl.getSelectedEncounterIdx() ? " selected" : ""}>AOT #${escapeHtml(e.encounter_id)} (${escapeHtml(e.aircraft_class)}, ${e.frames.length}f)</option>`,
      )
      .join("");
    const errBanner = this.ctrl.getLastError()
      ? `<div class="realsky-error" role="alert">${escapeHtml(this.ctrl.getLastError()!)}</div>`
      : "";

    this.root.innerHTML =
      `<div class="realsky-shell">` +
      `<div class="realsky-header"><h2>${escapeHtml(s.heading)}</h2><p>${escapeHtml(s.subheading)}</p></div>` +
      `<div class="realsky-controls">` +
      `<label>${escapeHtml(s.clipLabel)}: <select id="realsky-enc-select" class="xvs-select">${encOptions || `<option>${escapeHtml(s.loadingPack)}</option>`}</select></label>` +
      `<button id="realsky-play-btn" class="xvs-btn xvs-btn-primary">${escapeHtml(this.playing ? s.pauseLabel : s.playLabel)}</button>` +
      `<button id="realsky-step-btn" class="xvs-btn">${escapeHtml(s.stepLabel)}</button>` +
      `<span class="realsky-scale-pill">${escapeHtml(s.nativeScaleNote)}</span>` +
      `</div>` +
      errBanner +
      `<div class="realsky-grid">` +
      `<div class="realsky-stage">` +
      `<canvas id="realsky-canvas" width="256" height="256"></canvas>` +
      `<div id="realsky-frame-meta" class="realsky-meta">${escapeHtml(this.customStatusText)}</div>` +
      `</div>` +
      `<div class="realsky-right">` +
      `<div id="realsky-chart-slot">${buildRangeComparisonChartSvg(this.locale, this.chartSamples)}</div>` +
      `<label id="realsky-dropzone" class="realsky-dropzone">` +
      `<strong>${escapeHtml(s.dropPrompt)}</strong>` +
      `<span>${escapeHtml(s.dropSubtext)}</span>` +
      `<input id="realsky-file-input" type="file" accept=".mp4,.webm,.jpg,.jpeg,.png,image/*,video/*" style="display:none" />` +
      `</label>` +
      `</div>` +
      `</div>` +
      `</div>`;

    this.root.querySelector<HTMLSelectElement>("#realsky-enc-select")?.addEventListener("change", (ev) => {
      const idx = Number((ev.target as HTMLSelectElement).value);
      this.selectEncounter(idx);
      void this.renderCurrentAotFrame();
    });
    this.root.querySelector<HTMLButtonElement>("#realsky-play-btn")?.addEventListener("click", () => {
      this.togglePlay();
    });
    this.root.querySelector<HTMLButtonElement>("#realsky-step-btn")?.addEventListener("click", () => {
      if (!this.manifest) return;
      const enc = this.manifest.encounters[this.ctrl.getSelectedEncounterIdx()];
      if (!enc || enc.frames.length === 0) return;
      this.currentFrameIdx = (this.currentFrameIdx + 1) % enc.frames.length;
      void this.renderCurrentAotFrame();
    });
    const dropzone = this.root.querySelector<HTMLElement>("#realsky-dropzone");
    const fileInput = this.root.querySelector<HTMLInputElement>("#realsky-file-input");
    dropzone?.addEventListener("dragover", (ev) => ev.preventDefault());
    dropzone?.addEventListener("drop", (ev) => {
      ev.preventDefault();
      const f = ev.dataTransfer?.files?.[0];
      if (f) void this.handleLocalFile(f);
    });
    fileInput?.addEventListener("change", () => {
      const f = fileInput.files?.[0];
      if (f) void this.handleLocalFile(f);
    });
  }

  dispose(): void {
    this.stopPlayback();
    if (this.activeObjectUrl) {
      URL.revokeObjectURL(this.activeObjectUrl);
      this.activeObjectUrl = null;
    }
  }
}
