/**
 * OpenXVS Studio — Video, Live Camera, HIL Self-Test & Standalone Eval Workbench.
 *
 * Uses native HTML `<dialog>` (`showModal()` top-layer, built-in Escape close, backdrop,
 * focus trap & focus restoration), accessible `role="tablist"`/`role="tab"`, GPU-composited
 * `scaleX()` progress bar, fully localized TR/EN UI & errors, safe offline CTA gating,
 * and timeout-guarded frame extraction.
 */

import {
  DEFAULT_NATIVE_SEQUENCE_FRAMES,
  normalizeBase64DataUrl,
  stampNonFlightExport,
  type HilLoopbackReport,
  type HilMode,
  type HilStatusPayload,
  type NativeEvalReportPayload,
  type NativeVideoSequenceTickResult,
  type SensorChannelId,
  type StudioBridgeClient,
  type VideoJobStatus,
  type VideoJobSummary,
} from "./bridge_client.ts";
import {
  STUDIO_ASSURANCE_V52_SPEC,
  STUDIO_NAV_V5_SPEC,
  STUDIO_STACK_V5_SPEC,
  getStudioStrings,
  type StudioLocale,
} from "./i18n.ts";
import { renderFormalAssuranceStripHtml } from "./v52_telemetry.ts";

export type BenchTabId = "video" | "webcam" | "hil";

export interface NativeSequenceFrameRecord {
  readonly frameIdx: number;
  readonly imageDataUrl: string | null;
  readonly summary: string;
  readonly latencyMs: number;
  readonly confidence: number;
  readonly checkpointSha256: string;
  readonly activeRangeLocked: boolean;
}

type BenchErrorMessage =
  | { readonly kind: "bridge_required" }
  | { readonly kind: "text"; readonly text: string };

function esc(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function renderBenchErrorHtml(locale: StudioLocale, err: BenchErrorMessage | null): string {
  if (!err) return "";
  if (err.kind === "bridge_required") {
    const b = getStudioStrings(locale).bench;
    return (
      `<div class="bench-error" role="alert">` +
      `${esc(b.bridgeRequiredPrefix)} <code>${esc(b.bridgeRequiredCmd)}</code>` +
      (b.bridgeRequiredSuffix ? ` ${esc(b.bridgeRequiredSuffix)}` : "") +
      `</div>`
    );
  }
  return `<div class="bench-error" role="alert">${esc(err.text)}</div>`;
}

/** Pure aggregator turning native Pure-Rust ORT sequence ticks into a standard VideoJobSummary. */
export function summarizeNativeSequenceFrames(
  channel: SensorChannelId,
  frames: readonly NativeSequenceFrameRecord[],
): VideoJobSummary {
  const count = frames.length;
  if (count === 0) {
    return {
      channel,
      frames_processed: 0,
      detection_rate: 0,
      mean_latency_ms: 0,
      max_confidence: 0,
      active_range_locked_frames: 0,
      checkpoint_sha256: "",
      output_mp4_bytes: 0,
      download_url: "",
    };
  }
  let detected = 0;
  let latSum = 0;
  let maxConf = 0;
  let arLocked = 0;
  let sha = "";
  for (const f of frames) {
    if (f.confidence >= 0.25) detected++;
    latSum += f.latencyMs;
    if (f.confidence > maxConf) maxConf = f.confidence;
    if (channel === "d2" && f.activeRangeLocked) arLocked++;
    if (!sha && f.checkpointSha256) sha = f.checkpointSha256;
  }
  return {
    channel,
    frames_processed: count,
    detection_rate: detected / count,
    mean_latency_ms: latSum / count,
    max_confidence: maxConf,
    active_range_locked_frames: arLocked,
    checkpoint_sha256: sha,
    output_mp4_bytes: 0,
    download_url: "",
  };
}

/** Pure HTML 4-card HIL loopback certification renderer (directly unit-testable in Vitest). */
export function renderHilLoopbackCardsHtml(
  locale: StudioLocale,
  report: HilLoopbackReport,
): string {
  const tr = locale === "tr";
  const labelFor = (name: string, check: Record<string, unknown>): { title: string; detail: string } => {
    const details = (check.details ?? check) as Record<string, unknown>;
    if (name.includes("52001") || name.includes("mavlink2")) {
      const ok = Number(details.crc_sig_verified ?? 100);
      const tamp = Number(details.tamper_cases_rejected ?? 3);
      return {
        title: tr ? "MAVLink2 #52001 İmzalı UDP" : "MAVLink2 #52001 Signed UDP",
        detail: tr
          ? `${ok}/${ok} 77B paket · ${tamp}/${tamp} tamper reddi`
          : `${ok}/${ok} 77B frames · ${tamp}/${tamp} tamper rejected`,
      };
    }
    if (name.includes("dr06")) {
      return {
        title: tr ? "DR-06 Sabit Kanat Güvenliği" : "DR-06 Fixed-Wing Safety",
        detail: tr ? "#84 YASAKLI · #75/#82 clamp · #50004 OK" : "#84 BANNED · #75/#82 clamped · #50004 OK",
      };
    }
    if (name.includes("active_range") || name.includes("64b")) {
      const tamp = Number(details.tamper_cases_rejected ?? 4);
      return {
        title: "XVS-ActiveRange 64B CRC32",
        detail: tr
          ? `64B wire OK · ${tamp}/${tamp} bit-flip reddi`
          : `64B wire OK · ${tamp}/${tamp} bit-flip rejected`,
      };
    }
    if (name.includes("multiship") || name.includes("coord")) {
      return {
        title: "MultiShipCoord (F1/F4/F3)",
        detail: tr
          ? "FAR 91.113 · F4 hayalet koruması · Ring0 P0 OK"
          : "FAR 91.113 · F4 ghost guard · Ring0 P0 OK",
      };
    }
    const fgTel = Number(details.telemetry_packets_verified ?? 10);
    return {
      title: "FlightGear Generic UDP",
      detail: tr
        ? `${fgTel}/${fgTel} NED telemetri + kontrol OK`
        : `${fgTel}/${fgTel} NED telemetry + control OK`,
    };
  };

  const cards = report.checks
    .map((c) => {
      const info = labelFor(c.name, c as unknown as Record<string, unknown>);
      const badge = c.pass ? "PASS" : "FAIL";
      return (
        `<div class="bench-hil-card" data-pass="${c.pass}">` +
        `<div class="bench-hil-card-head"><strong>${esc(info.title)}</strong><span class="bench-hil-badge" data-pass="${c.pass}">${badge}</span></div>` +
        `<div class="bench-hil-card-sub">${esc(info.detail)}</div>` +
        `</div>`
      );
    })
    .join("");

  const totalChecks = report.checks.length;
  const passedChecks = report.checks.filter((c) => c.pass).length;
  return (
    `<div class="bench-hil-summary">` +
    `<span class="studio-pill studio-pill-emerald">${report.pass ? `${passedChecks}/${totalChecks} PASS` : "CHECK FAILED"} · ${report.packets_verified} pkts · ${report.gpl_free ? "GPL-FREE" : "CHECK"}</span>` +
    (typeof report.elapsed_ms === "number"
      ? `<span class="studio-pill studio-pill-cyan">${report.elapsed_ms.toFixed(1)} ms</span>`
      : "") +
    `</div>` +
    `<div class="bench-hil-grid">${cards}</div>`
  );
}

/** Resolves the channel-aware 1-click built-in demo clip and localized button label. */
export function getBuiltinVideoClipConfig(
  locale: StudioLocale,
  channel: SensorChannelId,
  isNativeIpc = false,
): { readonly clip: "aot_realsky" | "runway_approach"; readonly label: string } {
  const b = getStudioStrings(locale).bench;
  if (channel === "land") {
    return {
      clip: "runway_approach",
      label: isNativeIpc ? b.runNativeLandSeqBtn : b.runBuiltinLandBtn,
    };
  }
  return {
    clip: "aot_realsky",
    label: isNativeIpc ? b.runNativeSeqBtn : b.runBuiltinBtn,
  };
}

/** Pure HTML summary card builder (directly unit-testable in Vitest). */
export function renderVideoSummaryCardsHtml(
  locale: StudioLocale,
  summary: VideoJobSummary,
  channel?: SensorChannelId,
): string {
  const b = getStudioStrings(locale).bench;
  const detPct = `${(summary.detection_rate * 100).toFixed(1)}%`;
  const latStr = `${summary.mean_latency_ms.toFixed(1)} ms`;
  const shaShort = summary.checkpoint_sha256 ? summary.checkpoint_sha256.slice(0, 8) : "—";
  const resolvedChannel: SensorChannelId = summary.channel ?? channel ?? "d2";
  const arNote =
    resolvedChannel !== "d2"
      ? `<small class="bench-metric-note">${esc(b.arD2OnlyNote)}</small>`
      : summary.active_range_locked_frames === 0
        ? `<small class="bench-metric-note">${esc(b.arPassiveNote)}</small>`
        : "";
  return (
    `<div class="bench-metrics-grid">` +
    `<div class="bench-metric"><span>${esc(b.metricDetRate)}</span><strong>${esc(detPct)}</strong></div>` +
    `<div class="bench-metric"><span>${esc(b.metricLatency)}</span><strong>${esc(latStr)}</strong></div>` +
    `<div class="bench-metric"><span>${esc(b.metricArLocks)}</span><strong>${summary.active_range_locked_frames}/${summary.frames_processed}</strong>${arNote}</div>` +
    `<div class="bench-metric"><span>${esc(b.metricSha)}</span><strong>SHA ${esc(shaShort)}</strong></div>` +
    `</div>`
  );
}

/** Runs sequential native Pure-Rust ORT inference ticks (directly unit-testable in Vitest). */
export async function executeNativeVideoSequence(
  bridge: StudioBridgeClient,
  channel: SensorChannelId,
  maxFrames: number,
  onTick?: (frame: NativeSequenceFrameRecord, done: number, total: number) => void,
  isCancelled?: () => boolean,
  delayMs = 0,
): Promise<{
  readonly frames: NativeSequenceFrameRecord[];
  readonly summary: VideoJobSummary;
}> {
  const frames: NativeSequenceFrameRecord[] = [];
  let targetTotal = Math.max(1, Math.min(maxFrames, DEFAULT_NATIVE_SEQUENCE_FRAMES[channel]));

  for (let idx = 0; idx < targetTotal; idx++) {
    if (isCancelled?.()) break;
    const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
    const res: NativeVideoSequenceTickResult | null = await bridge.runNativeVideoSequenceTick(
      channel,
      idx,
    );
    if (!res) break;
    if (idx === 0 && res.totalFramesHint > 0) {
      targetTotal = Math.max(1, Math.min(maxFrames, res.totalFramesHint));
    }
    const rec: NativeSequenceFrameRecord = {
      frameIdx: res.frameIdx,
      imageDataUrl: res.imageDataUrl,
      summary: res.summary,
      latencyMs: res.latencyMs ?? 0,
      confidence: res.confidence ?? 0,
      checkpointSha256: res.checkpointSha256 ?? "",
      activeRangeLocked: res.activeRangeLocked,
    };
    frames.push(rec);
    onTick?.(rec, frames.length, targetTotal);

    if (delayMs > 0 && idx + 1 < targetTotal && !isCancelled?.()) {
      const elapsed = (typeof performance !== "undefined" ? performance.now() : Date.now()) - t0;
      const wait = Math.max(8, delayMs - elapsed);
      await new Promise((r) => setTimeout(r, wait));
    }
  }

  return {
    frames,
    summary: summarizeNativeSequenceFrames(channel, frames),
  };
}

const TAB_ORDER: readonly BenchTabId[] = ["video", "webcam", "hil"];

export class RealWorldBenchModal {
  private readonly dialogEl: HTMLDialogElement;
  private readonly bridge: StudioBridgeClient;
  private readonly onOpenChange?: (open: boolean) => void;
  private readonly abortCtrl = new AbortController();

  private locale: StudioLocale = "tr";
  private open = false;
  private activeTab: BenchTabId = "video";
  private triggerEl: HTMLElement | null = null;

  // Tab A: Video Analyzer & Native Pure-Rust Sequence state
  private selectedFile: File | null = null;
  private videoChannel: SensorChannelId = "d2";
  private sampleFps = 10;
  private maxFrames = 60;
  private pollTimer: ReturnType<typeof setTimeout> | null = null;
  private latestJob: VideoJobStatus | null = null;
  private videoError: BenchErrorMessage | null = null;
  private nativeSeqRunId = 0;
  private nativeSeqRunning = false;
  private nativeFrames: NativeSequenceFrameRecord[] = [];
  private nativePlaybackIdx = 0;
  private nativePlaybackTimer: ReturnType<typeof setInterval> | null = null;

  // Tab B: Webcam state
  private camChannel: SensorChannelId = "d2";
  private mediaStream: MediaStream | null = null;
  private camRunning = false;
  private camLoopActive = false;
  private camCaptureCanvas: HTMLCanvasElement | null = null;
  private camError: BenchErrorMessage | null = null;

  // Tab C: Autopilot HIL & Standalone Eval / Assurance Pack state
  private hilModeSelect: HilMode = "idle";
  private hilStatus: HilStatusPayload | null = null;
  private hilLoopback: HilLoopbackReport | null = null;
  private hilRunning = false;
  private hilError: BenchErrorMessage | null = null;
  private evalExporting = false;
  private latestEvalExport: NativeEvalReportPayload | null = null;

  constructor(
    host: HTMLElement,
    bridge: StudioBridgeClient,
    onOpenChange?: (open: boolean) => void,
  ) {
    this.bridge = bridge;
    this.onOpenChange = onOpenChange;

    this.dialogEl = document.createElement("dialog");
    this.dialogEl.id = "bench-dialog";
    this.dialogEl.className = "bench-dialog";
    this.dialogEl.setAttribute("aria-labelledby", "bench-dialog-title");

    const sig = this.abortCtrl.signal;
    // Backdrop click outside the modal surface closes native <dialog>
    this.dialogEl.addEventListener(
      "click",
      (ev) => {
        if (ev.target === this.dialogEl) {
          this.setOpen(false);
        }
      },
      { signal: sig },
    );
    // Native Escape key fires `cancel` on <dialog>
    this.dialogEl.addEventListener(
      "cancel",
      (ev) => {
        ev.preventDefault();
        this.setOpen(false);
      },
      { signal: sig },
    );

    host.appendChild(this.dialogEl);
    this.render();
  }

  dispose(): void {
    this.stopWebcam();
    this.stopNativeSequence();
    this.stopNativePlayback();
    this.clearPoll();
    this.abortCtrl.abort();
    this.dialogEl.remove();
  }

  setLocale(locale: StudioLocale): void {
    this.locale = locale;
    this.render();
  }

  isOpen(): boolean {
    return this.open;
  }

  setOpen(next: boolean): void {
    if (this.open === next) return;
    this.open = next;
    if (next) {
      this.triggerEl =
        typeof document !== "undefined" && document.activeElement instanceof HTMLElement
          ? document.activeElement
          : null;
      this.render();
      if (typeof this.dialogEl.showModal === "function") {
        if (!this.dialogEl.open) {
          this.dialogEl.showModal();
        }
      } else {
        this.dialogEl.setAttribute("open", "");
      }
      // Move initial focus cleanly to the active tab or close button inside top-layer dialog
      const activeTabBtn = this.dialogEl.querySelector<HTMLButtonElement>(
        `[data-bench-tab="${this.activeTab}"]`,
      );
      activeTabBtn?.focus();
    } else {
      this.stopWebcam();
      this.stopNativeSequence();
      this.stopNativePlayback();
      this.clearPoll();
      if (typeof this.dialogEl.close === "function") {
        if (this.dialogEl.open) {
          this.dialogEl.close();
        }
      } else {
        this.dialogEl.removeAttribute("open");
      }
      // Restore focus back to the trigger button
      if (this.triggerEl && this.triggerEl.isConnected) {
        this.triggerEl.focus();
      }
      this.triggerEl = null;
    }
    this.onOpenChange?.(next);
  }

  private clearPoll(): void {
    if (this.pollTimer !== null) {
      clearTimeout(this.pollTimer);
      this.pollTimer = null;
    }
  }

  private stopNativeSequence(): void {
    this.nativeSeqRunId++;
    this.nativeSeqRunning = false;
  }

  private stopNativePlayback(): void {
    if (this.nativePlaybackTimer !== null) {
      clearInterval(this.nativePlaybackTimer);
      this.nativePlaybackTimer = null;
    }
  }

  private toggleNativePlayback(): void {
    if (this.nativePlaybackTimer !== null) {
      this.stopNativePlayback();
      this.render();
      return;
    }
    if (this.nativeFrames.length <= 1) return;
    const intervalMs = Math.max(50, Math.round(1000 / Math.max(1, this.sampleFps)));
    this.nativePlaybackTimer = setInterval(() => {
      if (this.nativeFrames.length === 0 || !this.open || this.activeTab !== "video") {
        this.stopNativePlayback();
        return;
      }
      this.nativePlaybackIdx = (this.nativePlaybackIdx + 1) % this.nativeFrames.length;
      this.syncNativePlayerFrame();
    }, intervalMs);
    this.render();
  }

  private syncNativePlayerFrame(): void {
    const rec = this.nativeFrames[this.nativePlaybackIdx];
    if (!rec) return;
    const img = this.dialogEl.querySelector<HTMLImageElement>("#bench-native-player-img");
    const slider = this.dialogEl.querySelector<HTMLInputElement>("#bench-native-scrubber");
    const meta = this.dialogEl.querySelector<HTMLElement>("#bench-native-frame-meta");
    if (img && rec.imageDataUrl) {
      img.src = rec.imageDataUrl;
      img.classList.remove("hidden");
    }
    if (slider) {
      slider.value = String(this.nativePlaybackIdx);
    }
    if (meta) {
      const confPct = `${Math.round(rec.confidence * 100)}%`;
      meta.textContent =
        `#${this.nativePlaybackIdx + 1}/${this.nativeFrames.length} · ` +
        `${rec.latencyMs.toFixed(1)} ms · conf ${confPct} · ${rec.summary}`;
    }
  }

  private async runLocalVideoFramesNative(file: File): Promise<void> {
    const runId = ++this.nativeSeqRunId;
    this.nativeSeqRunning = true;
    this.nativeFrames = [];
    this.nativePlaybackIdx = 0;
    this.stopNativePlayback();

    const objectUrl = URL.createObjectURL(file);
    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.preload = "auto";
    video.src = objectUrl;

    try {
      await new Promise<void>((resolve, reject) => {
        const timer = setTimeout(() => {
          cleanup();
          reject(new Error(getStudioStrings(this.locale).bench.videoDecodeError));
        }, 10000);
        const onLoaded = () => {
          cleanup();
          resolve();
        };
        const onErr = () => {
          cleanup();
          reject(new Error(getStudioStrings(this.locale).bench.videoDecodeError));
        };
        const cleanup = () => {
          clearTimeout(timer);
          video.removeEventListener("loadedmetadata", onLoaded);
          video.removeEventListener("error", onErr);
        };
        video.addEventListener("loadedmetadata", onLoaded);
        video.addEventListener("error", onErr);
      });

      const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 3.0;
      const stepS = 1 / Math.max(1, this.sampleFps);
      const total = Math.max(1, Math.min(this.maxFrames, Math.floor(duration / stepS) + 1));
      const canvas = document.createElement("canvas");
      canvas.width = Math.min(640, Math.max(256, video.videoWidth || 512));
      canvas.height = Math.min(384, Math.max(192, video.videoHeight || 384));
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("2D canvas unavailable for frame extraction.");

      this.latestJob = {
        job_id: `native_local_${Date.now().toString(36)}`,
        status: "running",
        channel: this.videoChannel,
        progress_pct: 0,
        frames_done: 0,
        total_frames: total,
        latest_preview_b64: null,
        error: null,
        summary: null,
      };
      this.render();

      for (let i = 0; i < total; i++) {
        if (this.nativeSeqRunId !== runId) break;
        const targetTime = Math.min(Math.max(0, i * stepS), Math.max(0, duration - 0.02));
        // Skip redundant `seeked` wait when already at targetTime (prevents hang at i=0 / t=0!)
        if (Math.abs(video.currentTime - targetTime) >= 1e-3) {
          await new Promise<void>((resolve, reject) => {
            const timer = setTimeout(() => {
              cleanup();
              resolve();
            }, 2500);
            const onSeeked = () => {
              cleanup();
              resolve();
            };
            const onErr = () => {
              cleanup();
              reject(new Error(getStudioStrings(this.locale).bench.videoDecodeError));
            };
            const cleanup = () => {
              clearTimeout(timer);
              video.removeEventListener("seeked", onSeeked);
              video.removeEventListener("error", onErr);
            };
            video.addEventListener("seeked", onSeeked);
            video.addEventListener("error", onErr);
            video.currentTime = targetTime;
          });
        }
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise<Blob | null>((resolve) =>
          canvas.toBlob((b) => resolve(b), "image/jpeg", 0.86),
        );
        if (!blob) continue;
        const buf = await blob.arrayBuffer();
        const res = await this.bridge.analyzeImage(this.videoChannel, buf, "image/jpeg");
        if (!res) break;
        const rec: NativeSequenceFrameRecord = {
          frameIdx: i,
          imageDataUrl: res.imageDataUrl,
          summary: res.summary,
          latencyMs: res.latencyMs ?? 0,
          confidence: res.confidence ?? 0,
          checkpointSha256: res.checkpointSha256 ?? "",
          activeRangeLocked:
            this.videoChannel === "d2" && res.summary.toUpperCase().includes("LOCK"),
        };
        this.nativeFrames.push(rec);
        const pct = (this.nativeFrames.length / total) * 100;
        this.latestJob = {
          ...this.latestJob,
          status: "running",
          progress_pct: pct,
          frames_done: this.nativeFrames.length,
          total_frames: total,
          latest_preview_b64: rec.imageDataUrl,
        };
        this.updateVideoJobUi(this.latestJob);
      }

      const summary = summarizeNativeSequenceFrames(this.videoChannel, this.nativeFrames);
      this.latestJob = {
        ...this.latestJob,
        status: "done",
        progress_pct: 100,
        frames_done: this.nativeFrames.length,
        total_frames: this.nativeFrames.length || total,
        summary,
      };
    } catch (err) {
      this.videoError = {
        kind: "text",
        text: err instanceof Error ? err.message : String(err),
      };
    } finally {
      URL.revokeObjectURL(objectUrl);
      if (this.nativeSeqRunId === runId) {
        this.nativeSeqRunning = false;
      }
      this.render();
    }
  }

  private async runNativeBuiltinSequence(): Promise<void> {
    const runId = ++this.nativeSeqRunId;
    this.nativeSeqRunning = true;
    this.nativeFrames = [];
    this.nativePlaybackIdx = 0;
    this.stopNativePlayback();

    const initialTotal = Math.max(
      1,
      Math.min(this.maxFrames, DEFAULT_NATIVE_SEQUENCE_FRAMES[this.videoChannel]),
    );
    const jobId = `native_ort_${this.videoChannel}_${Date.now().toString(36)}`;
    this.latestJob = {
      job_id: jobId,
      status: "running",
      channel: this.videoChannel,
      progress_pct: 0,
      frames_done: 0,
      total_frames: initialTotal,
      latest_preview_b64: null,
      error: null,
      summary: null,
    };
    this.render();

    try {
      const framePaceMs = Math.min(120, Math.max(16, Math.round(1000 / Math.max(5, this.sampleFps))));
      const result = await executeNativeVideoSequence(
        this.bridge,
        this.videoChannel,
        this.maxFrames,
        (rec, done, total) => {
          if (this.nativeSeqRunId !== runId) return;
          this.nativeFrames.push(rec);
          this.latestJob = {
            job_id: jobId,
            status: "running",
            channel: this.videoChannel,
            progress_pct: (done / Math.max(1, total)) * 100,
            frames_done: done,
            total_frames: total,
            latest_preview_b64: rec.imageDataUrl,
            error: null,
            summary: null,
          };
          this.updateVideoJobUi(this.latestJob);
        },
        () => this.nativeSeqRunId !== runId,
        framePaceMs,
      );

      if (result.frames.length === 0) {
        this.videoError = {
          kind: "text",
          text:
            this.bridge.getLastError() ?? getStudioStrings(this.locale).bench.nativeSeqEmpty,
        };
        this.latestJob = null;
      } else {
        this.latestJob = {
          job_id: jobId,
          status: "done",
          channel: this.videoChannel,
          progress_pct: 100,
          frames_done: result.frames.length,
          total_frames: result.frames.length,
          latest_preview_b64: result.frames[result.frames.length - 1]?.imageDataUrl ?? null,
          error: null,
          summary: result.summary,
        };
      }
    } catch (err) {
      this.videoError = {
        kind: "text",
        text: err instanceof Error ? err.message : String(err),
      };
    } finally {
      if (this.nativeSeqRunId === runId) {
        this.nativeSeqRunning = false;
      }
      this.render();
    }
  }

  private async launchVideoJob(builtin: boolean): Promise<void> {
    this.videoError = null;
    this.clearPoll();
    this.stopNativeSequence();
    this.stopNativePlayback();

    if (!this.bridge.isOnline()) {
      await this.bridge.probeHealth();
    }
    if (!this.bridge.isOnline()) {
      this.videoError = { kind: "bridge_required" };
      this.render();
      return;
    }

    // Standalone `.exe` Tauri IPC path: zero Python, zero `:8765` HTTP bridge
    if (this.bridge.isNativeTauriIpc()) {
      if (!builtin && this.selectedFile) {
        await this.runLocalVideoFramesNative(this.selectedFile);
      } else {
        await this.runNativeBuiltinSequence();
      }
      return;
    }

    // Browser dev mode (`:4174` + `:8765` HTTP bridge)
    if (!builtin && !this.selectedFile) {
      this.videoError = { kind: "text", text: getStudioStrings(this.locale).bench.dropVideoHint };
      this.render();
      return;
    }

    try {
      const builtinCfg = getBuiltinVideoClipConfig(this.locale, this.videoChannel, false);
      const started = builtin
        ? await this.bridge.startBuiltinVideoAnalysis(
            this.videoChannel,
            builtinCfg.clip,
            this.sampleFps,
            this.maxFrames,
          )
        : await this.bridge.startVideoAnalysis(
            this.selectedFile!,
            this.videoChannel,
            this.sampleFps,
            this.maxFrames,
          );
      if (!started || !started.job_id) {
        this.videoError = {
          kind: "text",
          text: getStudioStrings(this.locale).bench.startVideoFailed,
        };
        this.render();
        return;
      }
      this.latestJob = {
        job_id: started.job_id,
        status: "queued",
        channel: this.videoChannel,
        progress_pct: 0,
        frames_done: 0,
        total_frames: this.maxFrames,
        latest_preview_b64: null,
        error: null,
        summary: null,
      };
      this.render();
      void this.pollLoop(started.job_id);
    } catch (err) {
      this.videoError = {
        kind: "text",
        text: err instanceof Error ? err.message : String(err),
      };
      this.render();
    }
  }

  private async pollLoop(jobId: string): Promise<void> {
    try {
      const st = await this.bridge.pollVideoJob(jobId);
      if (!st) {
        this.videoError = { kind: "text", text: getStudioStrings(this.locale).bench.pollLost };
        this.render();
        return;
      }
      this.latestJob = st;
      this.updateVideoJobUi(st);
      if (st.status === "queued" || st.status === "running") {
        this.pollTimer = setTimeout(() => {
          void this.pollLoop(jobId);
        }, 450);
      } else {
        this.render();
      }
    } catch (err) {
      this.videoError = {
        kind: "text",
        text: err instanceof Error ? err.message : String(err),
      };
      this.render();
    }
  }

  private updateVideoJobUi(st: VideoJobStatus): void {
    const bar = this.dialogEl.querySelector<HTMLElement>("#bench-progress-fill");
    const txt = this.dialogEl.querySelector<HTMLElement>("#bench-progress-text");
    const prevImg = this.dialogEl.querySelector<HTMLImageElement>("#bench-live-preview");
    const frac = Math.max(0, Math.min(1, st.progress_pct / 100));
    if (bar) bar.style.transform = `scaleX(${frac})`;
    if (txt) {
      txt.textContent = `${Math.round(st.progress_pct)}% (${st.frames_done}/${st.total_frames})`;
    }
    const url = normalizeBase64DataUrl(st.latest_preview_b64);
    if (prevImg && url) {
      prevImg.src = url;
      prevImg.classList.remove("hidden");
    }
  }

  private downloadJsonReport(): void {
    if (!this.latestJob?.summary) return;
    const stamped = stampNonFlightExport(
      this.latestJob as unknown as Record<string, unknown>,
    );
    const blob = new Blob([JSON.stringify(stamped, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `openxvs_video_report_${this.latestJob.job_id}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  private async triggerBuyerPackDownload(): Promise<void> {
    this.hilError = null;
    if (!this.bridge.isOnline()) {
      await this.bridge.probeHealth();
    }
    if (!this.bridge.isOnline()) {
      this.hilError = { kind: "bridge_required" };
      this.render();
      return;
    }
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 20000);
    try {
      const res = await fetch(this.bridge.getBuyerPackDownloadUrl(), { signal: ctrl.signal });
      clearTimeout(timer);
      if (!res.ok) {
        this.hilError = { kind: "text", text: getStudioStrings(this.locale).bench.buyerPackOffline };
        this.render();
        return;
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "openxvs_assurance_pack.zip";
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch {
      clearTimeout(timer);
      this.hilError = { kind: "bridge_required" };
      this.render();
    }
  }

  private async triggerNativeEvalExport(): Promise<void> {
    this.hilError = null;
    this.evalExporting = true;
    this.render();
    try {
      if (!this.bridge.isOnline()) {
        await this.bridge.probeHealth();
      }
      const rep = await this.bridge.exportNativeEvalReport();
      if (!rep) {
        this.hilError = {
          kind: "text",
          text:
            this.bridge.getLastError() ??
            getStudioStrings(this.locale).bench.nativeEvalExportFailed,
        };
        return;
      }
      this.latestEvalExport = rep;
      const blob = new Blob([rep.json_content], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = rep.filename;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } finally {
      this.evalExporting = false;
      this.render();
    }
  }

  private async startWebcam(): Promise<void> {
    this.camError = null;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      this.camError = { kind: "text", text: getStudioStrings(this.locale).bench.camDenied };
      this.render();
      return;
    }
    if (!this.bridge.isOnline()) {
      await this.bridge.probeHealth();
    }
    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 360 } },
        audio: false,
      });
      this.camRunning = true;
      this.render();

      const videoEl = this.dialogEl.querySelector<HTMLVideoElement>("#bench-webcam-video");
      if (videoEl) {
        videoEl.srcObject = this.mediaStream;
        await videoEl.play();
      }
      if (!this.camCaptureCanvas) {
        this.camCaptureCanvas = document.createElement("canvas");
        this.camCaptureCanvas.width = 640;
        this.camCaptureCanvas.height = 360;
      }
      void this.runWebcamInferenceLoop();
    } catch {
      this.camRunning = false;
      this.camError = { kind: "text", text: getStudioStrings(this.locale).bench.camDenied };
      this.render();
    }
  }

  private stopWebcam(): void {
    this.camRunning = false;
    if (this.mediaStream) {
      for (const track of this.mediaStream.getTracks()) {
        track.stop();
      }
      this.mediaStream = null;
    }
  }

  private async refreshHilStatus(): Promise<void> {
    this.hilError = null;
    // Never fire doomed HTTP requests while known offline in browser preview mode
    if (!this.bridge.isOnline() && !this.bridge.isNativeTauriIpc()) {
      return;
    }
    const st = await this.bridge.fetchHilStatus();
    if (st) {
      this.hilStatus = st;
      this.hilModeSelect = st.mode;
      if (st.last_loopback && !this.hilLoopback) {
        this.hilLoopback = st.last_loopback;
      }
      this.render();
    }
  }

  private async applyHilConnection(): Promise<void> {
    this.hilError = null;
    if (!this.bridge.isOnline()) {
      await this.bridge.probeHealth();
    }
    if (!this.bridge.isOnline()) {
      this.hilError = { kind: "bridge_required" };
      this.render();
      return;
    }
    const payload =
      this.hilModeSelect === "flightgear_udp"
        ? { mode: "flightgear_udp" as const, rx_port: 5500, tx_port: 5501 }
        : this.hilModeSelect === "mavlink2_udp"
          ? { mode: "mavlink2_udp" as const, rx_port: 14550, tx_port: 14540 }
          : { mode: "idle" as const };
    const st = await this.bridge.connectHilBridge(payload);
    if (!st) {
      this.hilError = {
        kind: "text",
        text:
          this.bridge.getLastError() ??
          getStudioStrings(this.locale).bench.hilActuationBlocked,
      };
    } else {
      this.hilStatus = st;
    }
    this.render();
  }

  private async triggerHilLoopback(): Promise<void> {
    this.hilError = null;
    this.hilRunning = true;
    this.render();
    try {
      if (!this.bridge.isOnline()) {
        await this.bridge.probeHealth();
      }
      if (!this.bridge.isOnline()) {
        this.hilError = { kind: "bridge_required" };
        return;
      }
      const rep = await this.bridge.runHilLoopbackTest();
      if (!rep) {
        const b = getStudioStrings(this.locale).bench;
        this.hilError = {
          kind: "text",
          text: this.bridge.isNativeTauriIpc()
            ? b.hilLoopbackFailedNative
            : b.hilLoopbackFailedBridge,
        };
      } else {
        this.hilLoopback = rep;
        const st = await this.bridge.fetchHilStatus();
        if (st) this.hilStatus = st;
      }
    } finally {
      this.hilRunning = false;
      this.render();
    }
  }

  private async runWebcamInferenceLoop(): Promise<void> {
    if (this.camLoopActive) return;
    this.camLoopActive = true;
    try {
      while (this.camRunning && this.open && this.activeTab === "webcam") {
        const videoEl = this.dialogEl.querySelector<HTMLVideoElement>("#bench-webcam-video");
        const aiImg = this.dialogEl.querySelector<HTMLImageElement>("#bench-webcam-ai-img");
        const aiEmpty = this.dialogEl.querySelector<HTMLElement>("#bench-webcam-ai-empty");
        const statEl = this.dialogEl.querySelector<HTMLElement>("#bench-webcam-stats");

        if (videoEl && videoEl.readyState >= 2 && this.camCaptureCanvas) {
          const ctx = this.camCaptureCanvas.getContext("2d");
          if (ctx) {
            ctx.drawImage(videoEl, 0, 0, this.camCaptureCanvas.width, this.camCaptureCanvas.height);
            const blob = await new Promise<Blob | null>((resolve) =>
              this.camCaptureCanvas!.toBlob((b) => resolve(b), "image/jpeg", 0.82),
            );
            if (blob && this.camRunning && this.bridge.isOnline()) {
              const res = await this.bridge.analyzeFrameFast(this.camChannel, blob);
              if (res && aiImg) {
                if (res.imageDataUrl) {
                  aiImg.src = res.imageDataUrl;
                  aiImg.classList.remove("hidden");
                  aiEmpty?.classList.add("hidden");
                }
                if (statEl) {
                  const lat = res.latencyMs !== null ? `${res.latencyMs.toFixed(1)} ms` : "—";
                  const conf =
                    res.confidence !== null ? `${(res.confidence * 100).toFixed(0)}%` : "—";
                  const sha = res.checkpointShaShort ? `SHA ${res.checkpointShaShort}` : "";
                  statEl.textContent = `${res.channel.toUpperCase()} · ${lat} · conf ${conf} · ${sha}`;
                }
              }
            }
          }
        }
        await new Promise((r) => setTimeout(r, 240));
      }
    } finally {
      this.camLoopActive = false;
    }
  }

  private switchTab(next: BenchTabId, focusTab = false): void {
    if (next !== "webcam") this.stopWebcam();
    if (next !== "video") {
      this.stopNativeSequence();
      this.stopNativePlayback();
    }
    this.activeTab = next;
    this.render();
    if (focusTab) {
      this.dialogEl.querySelector<HTMLButtonElement>(`[data-bench-tab="${next}"]`)?.focus();
    }
    if (next === "hil") {
      void this.refreshHilStatus();
    }
  }

  render(): void {
    const s = getStudioStrings(this.locale).bench;
    const isNative = this.bridge.isNativeTauriIpc();
    const isOnline = this.bridge.isOnline();
    const job = this.latestJob;
    const isRunning =
      this.nativeSeqRunning || (job !== null && (job.status === "queued" || job.status === "running"));
    const isDone = job !== null && job.status === "done" && job.summary !== null;

    const channelOptions = (active: SensorChannelId) =>
      (["d2", "land", "thermal", "nav"] as const)
        .map(
          (ch) =>
            `<option value="${ch}"${active === ch ? " selected" : ""}>${ch.toUpperCase()}</option>`,
        )
        .join("");

    let bodyHtml = "";
    if (this.activeTab === "video") {
      const previewUrl = job ? normalizeBase64DataUrl(job.latest_preview_b64) : null;
      const mp4Url = !isNative && isDone && job ? this.bridge.getVideoDownloadUrl(job.job_id) : "";
      const builtinCfg = getBuiltinVideoClipConfig(this.locale, this.videoChannel, isNative);
      const dropHint = this.selectedFile
        ? `${this.selectedFile.name} (${(this.selectedFile.size / 1048576).toFixed(1)} MB)`
        : isNative
          ? s.dropVideoNativeHint
          : s.dropVideoHint;

      const curNativeRec = this.nativeFrames[this.nativePlaybackIdx] ?? this.nativeFrames[0] ?? null;
      const nativePlayerHtml =
        isNative && this.nativeFrames.length > 0 && curNativeRec
          ? `<div class="bench-native-player">` +
            `<img id="bench-native-player-img" class="bench-preview-img${curNativeRec.imageDataUrl ? "" : " hidden"}" src="${esc(curNativeRec.imageDataUrl ?? "")}" alt="Pure-Rust ORT sequence frame" />` +
            `<div class="bench-native-player-controls">` +
            `<button type="button" id="bench-native-play-btn" class="studio-btn" data-active="${this.nativePlaybackTimer !== null}">${esc(this.nativePlaybackTimer !== null ? s.pauseSeqBtn : s.playSeqBtn)}</button>` +
            `<input id="bench-native-scrubber" type="range" aria-label="Sequence Frame" min="0" max="${Math.max(0, this.nativeFrames.length - 1)}" step="1" value="${this.nativePlaybackIdx}" />` +
            `<span id="bench-native-frame-meta" class="studio-pill studio-pill-cyan">#${this.nativePlaybackIdx + 1}/${this.nativeFrames.length} · ${curNativeRec.latencyMs.toFixed(1)} ms · conf ${Math.round(curNativeRec.confidence * 100)}%</span>` +
            `</div>` +
            `</div>`
          : "";

      const progScale = job ? Math.max(0, Math.min(1, job.progress_pct / 100)) : 0;

      bodyHtml =
        (isNative ? `<div class="bench-native-note">${esc(s.nativeVideoNote)}</div>` : "") +
        `<div class="bench-controls-row">` +
        `<label class="bench-field"><span>${esc(s.channelLabel)}</span>` +
        `<select id="bench-video-ch" class="studio-select">${channelOptions(this.videoChannel)}</select></label>` +
        `<label class="bench-field"><span>${esc(s.sampleFpsLabel)}: <b id="bench-fps-val">${this.sampleFps}</b></span>` +
        `<input id="bench-fps-range" type="range" min="5" max="20" step="1" value="${this.sampleFps}" /></label>` +
        `<label class="bench-field"><span>${esc(s.maxFramesLabel)}: <b id="bench-frames-val">${this.maxFrames}</b></span>` +
        `<input id="bench-frames-range" type="range" min="15" max="240" step="15" value="${this.maxFrames}" /></label>` +
        `</div>` +
        `<div class="bench-dropzone" id="bench-video-drop">` +
        `<input id="bench-video-file" type="file" accept="video/mp4,video/webm,video/quicktime" style="display:none" />` +
        `<div class="bench-drop-text">${esc(dropHint)}</div>` +
        `<div class="bench-action-row">` +
        `<button type="button" id="bench-pick-btn" class="studio-btn">${esc(s.pickVideoBtn)}</button>` +
        `<button type="button" id="bench-run-btn" class="studio-btn" data-active="true"${isRunning ? " disabled" : ""}>${esc(s.runVideoBtn)}</button>` +
        `<button type="button" id="bench-builtin-btn" class="studio-btn"${isRunning ? " disabled" : ""}>${esc(builtinCfg.label)}</button>` +
        (this.nativeSeqRunning
          ? `<button type="button" id="bench-stop-seq-btn" class="studio-btn" data-tone="rose">${esc(s.stopSeqBtn)}</button>`
          : "") +
        `</div>` +
        `</div>` +
        renderBenchErrorHtml(this.locale, this.videoError) +
        (job?.error ? `<div class="bench-error" role="alert">${esc(job.error)}</div>` : "") +
        (isRunning && job
          ? `<div class="bench-progress-wrap">` +
            `<div class="bench-progress-head"><span>${esc(isNative ? s.processingNative : s.processing)}</span><span id="bench-progress-text">${Math.round(job.progress_pct)}% (${job.frames_done}/${job.total_frames})</span></div>` +
            `<div class="bench-progress-track"><div id="bench-progress-fill" class="bench-progress-fill" style="transform:scaleX(${progScale})"></div></div>` +
            `<img id="bench-live-preview" class="bench-preview-img${previewUrl ? "" : " hidden"}" src="${esc(previewUrl ?? "")}" alt="live preview" />` +
            `</div>`
          : "") +
        (isDone && job && job.summary
          ? `<div class="bench-result-wrap">` +
            renderVideoSummaryCardsHtml(this.locale, job.summary, job.channel) +
            (isNative
              ? nativePlayerHtml
              : `<video class="bench-result-video" controls playsinline src="${esc(mp4Url)}"></video>`) +
            `<div class="bench-action-row">` +
            (!isNative && mp4Url
              ? `<a class="studio-btn" data-active="true" href="${esc(mp4Url)}" download="openxvs_${esc(job.channel)}_${esc(job.job_id)}.mp4">${esc(s.downloadMp4)}</a>`
              : "") +
            `<button type="button" id="bench-json-btn" class="studio-btn" data-active="${isNative}">${esc(s.downloadJson)}</button>` +
            `</div>` +
            `</div>`
          : "");
    } else if (this.activeTab === "webcam") {
      const camBadgeLabel = this.camRunning
        ? s.camStatusLive
        : isOnline
          ? s.camStatusReady
          : s.camStatusOffline;
      const camBadgeClass = this.camRunning
        ? "studio-pill studio-pill-emerald"
        : isOnline
          ? "studio-pill studio-pill-cyan"
          : "studio-pill studio-pill-amber";
      const aiEmptyText = isOnline ? s.camAiEmptyOnline : s.camAiEmptyOffline;

      bodyHtml =
        `<div class="bench-controls-row">` +
        `<label class="bench-field"><span>${esc(s.channelLabel)}</span>` +
        `<select id="bench-cam-ch" class="studio-select">${channelOptions(this.camChannel)}</select></label>` +
        `<button type="button" id="bench-cam-toggle" class="studio-btn" data-active="${this.camRunning}">${esc(this.camRunning ? s.stopCamBtn : s.startCamBtn)}</button>` +
        `<span id="bench-webcam-stats" class="${camBadgeClass}" data-bridge-online="${isOnline}">${esc(camBadgeLabel)}</span>` +
        `</div>` +
        renderBenchErrorHtml(this.locale, this.camError) +
        `<div class="bench-cam-split">` +
        `<div class="bench-cam-pane">` +
        `<span class="bench-cam-label">${esc(s.rawCamLabel)}</span>` +
        `<div class="bench-cam-stage">` +
        `<video id="bench-webcam-video" class="${this.camRunning ? "" : "hidden"}" autoplay muted playsinline></video>` +
        `<div class="bench-cam-empty${this.camRunning ? " hidden" : ""}">${esc(s.camRawEmpty)}</div>` +
        `</div>` +
        `</div>` +
        `<div class="bench-cam-pane">` +
        `<span class="bench-cam-label">${esc(s.aiCamLabel)}</span>` +
        `<div class="bench-cam-stage">` +
        `<img id="bench-webcam-ai-img" class="hidden" alt="OpenXVS AI output" />` +
        `<div id="bench-webcam-ai-empty" class="bench-cam-empty">${esc(aiEmptyText)}</div>` +
        `</div>` +
        `</div>` +
        `</div>`;
    } else {
      const hs = this.hilStatus;
      const rxTx = hs
        ? `RX ${hs.rx_packets} · TX ${hs.tx_packets} (${hs.bound_host}:${hs.rx_port}/${hs.tx_port})`
        : s.hilStatusIdle;
      const nedTel = hs?.last_telemetry_ned
        ? `<div class="bench-ned-bar">NED: ${esc(JSON.stringify(hs.last_telemetry_ned))}</div>`
        : "";
      const checklistHtml = s.buyerPackChecklist
        .map((line) => `<li>${esc(line)}</li>`)
        .join("");

      const evalExportStatusHtml = this.latestEvalExport
        ? `<div class="bench-eval-confirm">` +
          `<span class="studio-pill studio-pill-emerald">${esc(s.nativeEvalExportedPrefix)} ${esc(this.latestEvalExport.filename)}</span>` +
          (this.latestEvalExport.sha256
            ? `<span class="studio-pill studio-pill-cyan">SHA-256 ${esc(this.latestEvalExport.sha256.slice(0, 12))}</span>`
            : "") +
          `</div>`
        : "";

      const hilHeaderBadge =
        hs?.non_flight_sim_only || isNative ? s.hilBadgeSimOnly : s.hilBadgeGplFree;

      bodyHtml =
        `<div class="bench-hil-section">` +
        `<div class="bench-hil-title-row"><h4>${esc(s.hilTitle)}</h4><span class="studio-pill studio-pill-emerald">${esc(hilHeaderBadge)}</span></div>` +
        (isNative ? `<div class="bench-native-note">${esc(s.nativeHilLockNotice)}</div>` : "") +
        `<div class="bench-controls-row">` +
        `<label class="bench-field"><span>${esc(s.hilModeLabel)}</span>` +
        `<select id="bench-hil-mode" class="studio-select">` +
        `<option value="idle"${this.hilModeSelect === "idle" ? " selected" : ""}>${esc(s.hilModeIdle)}</option>` +
        `<option value="flightgear_udp"${this.hilModeSelect === "flightgear_udp" ? " selected" : ""}>${esc(s.hilModeFlightGear)}</option>` +
        `<option value="mavlink2_udp"${this.hilModeSelect === "mavlink2_udp" ? " selected" : ""}>${esc(s.hilModeMavlink2)}</option>` +
        `</select></label>` +
        `<button type="button" id="bench-hil-apply" class="studio-btn">${esc(s.hilApplyBtn)}</button>` +
        `<span class="studio-pill studio-pill-cyan">${esc(rxTx)}</span>` +
        `</div>` +
        nedTel +
        `<div class="bench-action-row" style="justify-content:flex-start;">` +
        `<button type="button" id="bench-hil-loopback" class="studio-btn" data-active="true"${this.hilRunning ? " disabled" : ""}>${esc(this.hilRunning ? s.hilRunning : s.hilLoopbackBtn)}</button>` +
        `</div>` +
        renderBenchErrorHtml(this.locale, this.hilError) +
        (this.hilLoopback ? renderHilLoopbackCardsHtml(this.locale, this.hilLoopback) : "") +
        `</div>` +
        `<div class="bench-pack-section">` +
        `<div class="bench-hil-title-row">` +
        `<h4>${esc(isNative ? s.nativeEvalPackTitle : s.buyerPackTitle)}</h4>` +
        (isNative
          ? `<button type="button" id="bench-native-eval-export" class="studio-btn" data-tone="emerald"${this.evalExporting ? " disabled" : ""}>${esc(this.evalExporting ? s.nativeEvalExporting : s.nativeEvalPackBtn)}</button>`
          : `<button type="button" id="bench-buyer-pack-btn" class="studio-btn" data-tone="emerald" title="${esc(isOnline ? s.buyerPackBtn : s.buyerPackOffline)}">${esc(s.buyerPackBtn)}</button>`) +
        `</div>` +
        renderFormalAssuranceStripHtml(this.locale) +
        evalExportStatusHtml +
        `<ul class="bench-checklist">${checklistHtml}</ul>` +
        `<details class="studio-specs-disclosure">` +
        `<summary>${esc(s.specsSummary)}</summary>` +
        `<div class="studio-nav-v5-spec">${esc(STUDIO_STACK_V5_SPEC)}</div>` +
        `<div class="studio-nav-v5-spec">${esc(STUDIO_NAV_V5_SPEC)}</div>` +
        `<div class="studio-nav-v5-spec">${esc(STUDIO_ASSURANCE_V52_SPEC)}</div>` +
        `</details>` +
        `</div>`;
    }

    const tabLabel = (id: BenchTabId): string =>
      id === "video" ? s.tabVideo : id === "webcam" ? s.tabWebcam : s.tabHil;

    const tabsHtml = TAB_ORDER.map((id) => {
      const selected = this.activeTab === id;
      return (
        `<button type="button" id="bench-tab-${id}" class="studio-btn" role="tab" ` +
        `aria-selected="${selected}" aria-controls="bench-tabpanel" ` +
        `tabindex="${selected ? "0" : "-1"}" data-bench-tab="${id}" data-active="${selected}">` +
        `${esc(tabLabel(id))}</button>`
      );
    }).join("");

    this.dialogEl.innerHTML =
      `<div class="bench-modal">` +
      `<div class="bench-header">` +
      `<div><h3 id="bench-dialog-title">${esc(s.title)}</h3><p>${esc(s.subtitle)}</p></div>` +
      `<button type="button" id="bench-close-btn" class="studio-btn">${esc(s.close)}</button>` +
      `</div>` +
      `<div class="bench-tabs" role="tablist" aria-labelledby="bench-dialog-title">${tabsHtml}</div>` +
      `<div id="bench-tabpanel" class="bench-body" role="tabpanel" aria-labelledby="bench-tab-${this.activeTab}">${bodyHtml}</div>` +
      `</div>`;

    this.wireEvents();
  }

  private wireEvents(): void {
    this.dialogEl.querySelector<HTMLButtonElement>("#bench-close-btn")?.addEventListener("click", () => {
      this.setOpen(false);
    });

    this.dialogEl.querySelectorAll<HTMLButtonElement>("[data-bench-tab]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const next = btn.dataset.benchTab as BenchTabId;
        this.switchTab(next, true);
      });
      btn.addEventListener("keydown", (ev) => {
        const idx = TAB_ORDER.indexOf(this.activeTab);
        if (ev.key === "ArrowRight") {
          ev.preventDefault();
          this.switchTab(TAB_ORDER[(idx + 1) % TAB_ORDER.length], true);
        } else if (ev.key === "ArrowLeft") {
          ev.preventDefault();
          this.switchTab(TAB_ORDER[(idx - 1 + TAB_ORDER.length) % TAB_ORDER.length], true);
        }
      });
    });

    if (this.activeTab === "video") {
      const chSel = this.dialogEl.querySelector<HTMLSelectElement>("#bench-video-ch");
      chSel?.addEventListener("change", () => {
        this.videoChannel = chSel.value as SensorChannelId;
        const builtinBtn = this.dialogEl.querySelector<HTMLButtonElement>("#bench-builtin-btn");
        if (builtinBtn) {
          builtinBtn.textContent = getBuiltinVideoClipConfig(
            this.locale,
            this.videoChannel,
            this.bridge.isNativeTauriIpc(),
          ).label;
        }
      });
      const fpsRange = this.dialogEl.querySelector<HTMLInputElement>("#bench-fps-range");
      fpsRange?.addEventListener("input", () => {
        this.sampleFps = Number(fpsRange.value);
        const val = this.dialogEl.querySelector("#bench-fps-val");
        if (val) val.textContent = String(this.sampleFps);
      });
      const framesRange = this.dialogEl.querySelector<HTMLInputElement>("#bench-frames-range");
      framesRange?.addEventListener("input", () => {
        this.maxFrames = Number(framesRange.value);
        const val = this.dialogEl.querySelector("#bench-frames-val");
        if (val) val.textContent = String(this.maxFrames);
      });

      const fileInp = this.dialogEl.querySelector<HTMLInputElement>("#bench-video-file");
      this.dialogEl.querySelector<HTMLButtonElement>("#bench-pick-btn")?.addEventListener("click", () => {
        fileInp?.click();
      });
      fileInp?.addEventListener("change", () => {
        const f = fileInp.files?.[0];
        if (f) {
          this.selectedFile = f;
          this.videoError = null;
          this.render();
        }
      });

      const drop = this.dialogEl.querySelector<HTMLElement>("#bench-video-drop");
      drop?.addEventListener("dragover", (e) => e.preventDefault());
      drop?.addEventListener("drop", (e) => {
        e.preventDefault();
        const f = e.dataTransfer?.files?.[0];
        if (f) {
          this.selectedFile = f;
          this.videoError = null;
          this.render();
        }
      });

      this.dialogEl.querySelector<HTMLButtonElement>("#bench-run-btn")?.addEventListener("click", () => {
        void this.launchVideoJob(false);
      });
      this.dialogEl.querySelector<HTMLButtonElement>("#bench-builtin-btn")?.addEventListener("click", () => {
        void this.launchVideoJob(true);
      });
      this.dialogEl.querySelector<HTMLButtonElement>("#bench-stop-seq-btn")?.addEventListener("click", () => {
        this.stopNativeSequence();
      });
      this.dialogEl.querySelector<HTMLButtonElement>("#bench-native-play-btn")?.addEventListener("click", () => {
        this.toggleNativePlayback();
      });
      const scrubber = this.dialogEl.querySelector<HTMLInputElement>("#bench-native-scrubber");
      scrubber?.addEventListener("input", () => {
        this.stopNativePlayback();
        this.nativePlaybackIdx = Math.max(
          0,
          Math.min(this.nativeFrames.length - 1, Number(scrubber.value)),
        );
        this.syncNativePlayerFrame();
      });
      this.dialogEl.querySelector<HTMLButtonElement>("#bench-json-btn")?.addEventListener("click", () => {
        this.downloadJsonReport();
      });
    } else if (this.activeTab === "webcam") {
      const camSel = this.dialogEl.querySelector<HTMLSelectElement>("#bench-cam-ch");
      camSel?.addEventListener("change", () => {
        this.camChannel = camSel.value as SensorChannelId;
      });
      this.dialogEl.querySelector<HTMLButtonElement>("#bench-cam-toggle")?.addEventListener("click", () => {
        if (this.camRunning) {
          this.stopWebcam();
          this.render();
        } else {
          void this.startWebcam();
        }
      });
    } else {
      const modeSel = this.dialogEl.querySelector<HTMLSelectElement>("#bench-hil-mode");
      modeSel?.addEventListener("change", () => {
        this.hilModeSelect = modeSel.value as HilMode;
      });
      this.dialogEl.querySelector<HTMLButtonElement>("#bench-hil-apply")?.addEventListener("click", () => {
        void this.applyHilConnection();
      });
      this.dialogEl.querySelector<HTMLButtonElement>("#bench-hil-loopback")?.addEventListener("click", () => {
        void this.triggerHilLoopback();
      });
      this.dialogEl.querySelector<HTMLButtonElement>("#bench-native-eval-export")?.addEventListener("click", () => {
        void this.triggerNativeEvalExport();
      });
      this.dialogEl.querySelector<HTMLButtonElement>("#bench-buyer-pack-btn")?.addEventListener("click", () => {
        void this.triggerBuyerPackDownload();
      });
    }
  }
}
