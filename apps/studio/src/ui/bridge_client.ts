/**
 * Dual-Transport AI Inference Client for OpenXVS Studio v5.2:
 *   1. PRIMARY (Tauri v2 standalone `.exe`): Native in-process Pure-Rust `ort` (`load-dynamic`)
 *      IPC via `window.__TAURI__.core.invoke` / `window.__TAURI_INTERNALS__.invoke` — ZERO Python
 *      and ZERO `https://tauri.localhost -> http://127.0.0.1:8765` mixed-content block!
 *      Supports 6/6 Pure-Rust ORT + XFeat + LighterGlue FP16 + MultiShipCoord, v5.2
 *      `native_sensor_health` (12-sector EO, LWIR FFC/contrast-collapse, dual-tier FSYNC <=1ms,
 *      M-of-N Bearing TBC), `native_video_sequence_tick`, and `native_export_eval_report`.
 *   2. FALLBACK (Browser dev preview on Vite `:4174`): Local HTTP bridge (`http://127.0.0.1:8765`).
 *
 * Enforces REL4 immutable `SIMULATION & BENCHMARK EVALUATION ONLY — ZERO ACTUATION — NOT FOR FLIGHT`
 * watermark across UI status and exported reports.
 */

import type { StudioSimSnapshot } from "../sim/studio_sim.ts";

export const NON_FLIGHT_SIM_ONLY = true as const;
export const ACTUATOR_AUTHORITY = false as const;
export const NON_FLIGHT_WATERMARK =
  "SIMULATION & BENCHMARK EVALUATION ONLY — ZERO ACTUATION — NOT FOR FLIGHT" as const;

export type SensorChannelId = "d2" | "land" | "thermal" | "nav";
export type BridgeTransportMode = "tauri-ipc" | "http-dev" | "offline";

export const DEFAULT_NATIVE_SEQUENCE_FRAMES: Readonly<Record<SensorChannelId, number>> = {
  d2: 60,
  land: 36,
  thermal: 24,
  nav: 24,
};

export type TauriInvokeFn = <T = unknown>(
  cmd: string,
  args?: Record<string, unknown>,
) => Promise<T>;

export interface BridgeHealthPayload {
  readonly ok: boolean;
  readonly version?: string;
  readonly device: string;
  readonly vramMb: number | null;
  readonly modelsLoaded: readonly string[];
  readonly transport: BridgeTransportMode;
  readonly nonFlightSimOnly: boolean;
  readonly watermark: string;
}

export interface BridgeChannelResult {
  readonly channel: SensorChannelId;
  readonly imageDataUrl: string | null;
  readonly summary: string;
  readonly latencyMs: number | null;
  readonly confidence: number | null;
  readonly checkpointSha256: string | null;
  readonly checkpointShaShort: string | null;
  readonly watermark?: string;
  readonly metrics?: Record<string, unknown> | null;
}

export interface NativeSensorHealthPayload {
  readonly eo_blur_score: number;
  readonly eo_glare_ratio: number;
  readonly lwir_nuc_frozen: boolean;
  readonly lwir_freeze_ticks: number;
  readonly timestamp_skew_ms: number;
  readonly q_eo_effective: number;
  readonly q_lwir_effective: number;
  readonly thermal_mode: string;
  readonly multiship_coord_enabled: boolean;
  readonly f1_far91113_active: boolean;
  readonly f4_ghost_hysteresis_active: boolean;
  readonly eo_occlusion_ratio?: number;
  readonly eo_focus_quality?: number;
  /** v5.2 additive 4x4 spatial EO quality grid (`[f64; 16]`). */
  readonly grid_4x4_q_eo?: readonly number[];
  /** v5.2 additive 12-azimuth-sector EO worst-band quality map (`[f64; 12]`). */
  readonly sector_q_eo?: readonly number[];
  /** v5.2 additive minimum quality across all 12 EO azimuth sectors (`0.0..=1.0`). */
  readonly min_sector_q_eo?: number;
  /** v5.2 additive 12-bit degraded sector bitmask (`u16`, bit `s` set when `sector_q_eo[s] < 0.50`). */
  readonly degraded_sector_mask?: number;
  /** v5.2 additive LWIR hardware FFC imminent bit (`bit 0`). */
  readonly lwir_ffc_imminent?: boolean;
  /** v5.2 additive LWIR hardware FFC in-progress bit (`bit 1`, zero-tick frame mask). */
  readonly lwir_ffc_in_progress?: boolean;
  /** Optional legacy/dev-bridge FFC flag alias (boolean or bitmask bit0=imminent, bit1=in_progress). */
  readonly lwir_ffc_flag?: boolean | number;
  /** v5.2 additive LWIR O(N) histogram `p95 - p05` dynamic range in DN (collapse `< 10.0`). */
  readonly lwir_dynamic_range_p95_p05?: number;
  /** v5.2 additive LWIR contrast-collapse backstop flag (`p95 - p05 < 10 DN`). */
  readonly lwir_contrast_collapsed?: boolean;
  readonly timestamp_monotonic_ok?: boolean;
  readonly timestamp_skew_fault?: boolean;
  /** v5.2 additive dual-tier FSYNC hardware precision check (`<= 1.0 ms` & monotonic). */
  readonly fsync_precision_ok?: boolean;
  /** v5.2 additive zero-alloc M-of-N (`3-of-5`) bearing track confirmation status. */
  readonly bearing_track_confirmed?: boolean;
  /** v5.2 additive sliding-window hit count (`0..=5`) for M-of-N bearing TBC tracker. */
  readonly bearing_track_hits_in_window?: number;
  /** Optional multi-track TBC confirmed count alias for HTTP dev bridges. */
  readonly bearing_tbc_confirmed_tracks?: number;
  readonly ring0_authority_inhibited?: boolean;
  readonly last_swerve_role?: string;
  readonly non_flight_sim_only?: boolean;
  readonly actuator_authority?: boolean;
  readonly watermark?: string;
}

export interface NativeVideoSequenceTickResult extends BridgeChannelResult {
  readonly frameIdx: number;
  readonly totalFramesHint: number;
  readonly activeRangeLocked: boolean;
}

export interface NativeEvalReportPayload {
  readonly filename: string;
  readonly json_content: string;
  readonly sha256: string;
  readonly non_flight_sim_only: boolean;
  readonly actuator_authority: boolean;
  readonly watermark: string;
}

export interface VideoJobSummary {
  readonly channel?: SensorChannelId;
  readonly frames_processed: number;
  readonly detection_rate: number;
  readonly mean_latency_ms: number;
  readonly max_confidence: number;
  readonly active_range_locked_frames: number;
  readonly checkpoint_sha256: string;
  readonly output_mp4_bytes: number;
  readonly download_url: string;
}

export interface VideoJobStatus {
  readonly job_id: string;
  readonly status: "queued" | "running" | "done" | "error";
  readonly channel: SensorChannelId;
  readonly progress_pct: number;
  readonly frames_done: number;
  readonly total_frames: number;
  readonly latest_preview_b64: string | null;
  readonly error: string | null;
  readonly summary: VideoJobSummary | null;
}

export type HilMode = "idle" | "flightgear_udp" | "mavlink2_udp" | "locked_non_flight_sim_only";

export interface HilLoopbackCheck {
  readonly name: string;
  readonly pass: boolean;
  readonly details?: Record<string, unknown>;
  readonly [key: string]: unknown;
}

export interface HilLoopbackReport {
  readonly pass: boolean;
  readonly timestamp_utc: string;
  readonly elapsed_ms?: number;
  readonly packets_verified: number;
  readonly mavlink_52001_packets_verified?: number;
  readonly tamper_rejected: boolean;
  readonly gpl_free: boolean;
  readonly non_flight_sim_only?: boolean;
  readonly watermark?: string;
  readonly checks: readonly HilLoopbackCheck[];
}

export interface HilStatusPayload {
  readonly mode: HilMode;
  readonly bound_host: string;
  readonly rx_port: number;
  readonly tx_port: number;
  readonly rx_packets: number;
  readonly tx_packets: number;
  readonly tx_enabled?: boolean;
  readonly actuator_authority?: boolean;
  readonly non_flight_sim_only?: boolean;
  readonly eu_pld_notice?: string;
  readonly watermark?: string;
  readonly last_telemetry_ned: Record<string, unknown> | null;
  readonly last_loopback: HilLoopbackReport | null;
  readonly gpl_free: boolean;
}

export interface NativeRtaTickResult {
  readonly mode: string;
  readonly mode_id: number;
  readonly raw_target: string;
  readonly active_healthy_cams: number;
  readonly mandatory_go_around: boolean;
  readonly inhibit_avoidance_turns: boolean;
  readonly tick_index: number;
  readonly odd_yaml_sha256: string;
  readonly non_flight_sim_only: boolean;
  readonly actuator_authority: boolean;
  readonly watermark: string;
}

export function detectTauriInvoke(): TauriInvokeFn | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as {
    __TAURI__?: { core?: { invoke?: TauriInvokeFn } };
    __TAURI_INTERNALS__?: { invoke?: TauriInvokeFn };
  };
  if (typeof w.__TAURI__?.core?.invoke === "function") {
    return w.__TAURI__.core.invoke.bind(w.__TAURI__.core);
  }
  if (typeof w.__TAURI_INTERNALS__?.invoke === "function") {
    return w.__TAURI_INTERNALS__.invoke.bind(w.__TAURI_INTERNALS__);
  }
  return null;
}

export function arrayBufferToBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    const slice = bytes.subarray(i, Math.min(bytes.length, i + chunk));
    binary += String.fromCharCode(...slice);
  }
  return btoa(binary);
}

const STRICT_IMAGE_DATA_URL_RE = /^data:image\/(jpeg|png|webp);base64,[A-Za-z0-9+/=\s]+$/i;
const RAW_BASE64_RE = /^[A-Za-z0-9+/=\s]+$/;

export function normalizeBase64DataUrl(b64: unknown): string | null {
  if (typeof b64 !== "string") return null;
  const trimmed = b64.trim();
  if (trimmed.length < 32) return null;
  if (trimmed.startsWith("data:")) {
    return STRICT_IMAGE_DATA_URL_RE.test(trimmed) ? trimmed : null;
  }
  return RAW_BASE64_RE.test(trimmed) ? `data:image/jpeg;base64,${trimmed}` : null;
}

export function stampNonFlightExport<T extends Record<string, unknown>>(
  payload: T,
): T & {
  readonly non_flight_sim_only: true;
  readonly actuator_authority: false;
  readonly watermark: typeof NON_FLIGHT_WATERMARK;
} {
  return {
    ...payload,
    non_flight_sim_only: NON_FLIGHT_SIM_ONLY,
    actuator_authority: ACTUATOR_AUTHORITY,
    watermark: NON_FLIGHT_WATERMARK,
  };
}

function detectActiveRangeLocked(
  summary: string,
  metrics: Record<string, unknown> | null,
): boolean {
  if (metrics) {
    if (metrics.active_range_locked === true) return true;
    const ar = metrics.active_range as Record<string, unknown> | undefined;
    if (ar && typeof ar === "object") {
      if (typeof ar.mode === "string" && ar.mode.toLowerCase().includes("lock")) return true;
      if (ar.mode_id === 2) return true;
      if (typeof ar.sigma_r_m === "number" && ar.sigma_r_m <= 95) return true;
    }
  }
  return summary.toUpperCase().includes("LOCK");
}

export class StudioBridgeClient {
  readonly baseUrl: string;
  private readonly injectedInvoke: TauriInvokeFn | null;
  private online = false;
  private transport: BridgeTransportMode = "offline";
  private health: BridgeHealthPayload | null = null;
  private sensorHealth: NativeSensorHealthPayload | null = null;
  private tickFrame = 0;
  private inFlight = false;
  private consecutiveFailures = 0;
  private lastError: string | null = null;

  constructor(baseUrl = "http://127.0.0.1:8765", invokeOverride?: TauriInvokeFn | null) {
    this.baseUrl = baseUrl;
    this.injectedInvoke = invokeOverride === undefined ? null : invokeOverride;
  }

  private getTauriInvoke(): TauriInvokeFn | null {
    if (this.injectedInvoke) return this.injectedInvoke;
    return detectTauriInvoke();
  }

  isNativeTauriIpc(): boolean {
    return this.getTauriInvoke() !== null;
  }

  getTransport(): BridgeTransportMode {
    return this.transport;
  }

  isOnline(): boolean {
    return this.online;
  }

  getHealth(): BridgeHealthPayload | null {
    return this.health;
  }

  getSensorHealth(): NativeSensorHealthPayload | null {
    return this.sensorHealth;
  }

  getLastError(): string | null {
    return this.lastError;
  }

  getVideoDownloadUrl(jobId: string): string {
    return `${this.baseUrl}/api/video_job/${encodeURIComponent(jobId)}/download`;
  }

  getBuyerPackDownloadUrl(): string {
    return `${this.baseUrl}/api/export_buyer_pack`;
  }

  async stepNativeRta(input: Record<string, unknown> = {}): Promise<NativeRtaTickResult | null> {
    const invoke = this.getTauriInvoke();
    if (!invoke) return null;
    try {
      return await invoke<NativeRtaTickResult>("native_rta_step", { input });
    } catch {
      return null;
    }
  }

  async fetchSensorHealth(): Promise<NativeSensorHealthPayload | null> {
    const invoke = this.getTauriInvoke();
    if (invoke) {
      try {
        const res = await invoke<NativeSensorHealthPayload>("native_sensor_health");
        if (res && typeof res === "object") {
          this.sensorHealth = res;
          return res;
        }
        return null;
      } catch {
        return null;
      }
    }
    if (!this.online) return null;
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 1500);
    try {
      const res = await fetch(`${this.baseUrl}/api/sensor_health`, { signal: ctrl.signal });
      clearTimeout(timer);
      if (!res.ok) return null;
      const data = (await res.json()) as NativeSensorHealthPayload;
      this.sensorHealth = data;
      return data;
    } catch {
      clearTimeout(timer);
      return null;
    }
  }

  async runNativeVideoSequenceTick(
    channel: SensorChannelId,
    frameIdx: number,
  ): Promise<NativeVideoSequenceTickResult | null> {
    this.lastError = null;
    const invoke = this.getTauriInvoke();
    if (!invoke) return null;
    try {
      const entry = await invoke<Record<string, unknown>>("native_video_sequence_tick", {
        channel,
        frameIdx,
      });
      const base = this.parseChannelEntry(channel, entry);
      const metrics = base.metrics ?? null;
      const rawTotal =
        metrics && typeof metrics.total_frames === "number"
          ? metrics.total_frames
          : metrics && typeof metrics.sequence_frames === "number"
            ? metrics.sequence_frames
            : DEFAULT_NATIVE_SEQUENCE_FRAMES[channel];
      return {
        ...base,
        frameIdx,
        totalFramesHint: Math.max(1, Math.round(rawTotal)),
        activeRangeLocked:
          channel === "d2" && detectActiveRangeLocked(base.summary, metrics),
      };
    } catch (err) {
      this.lastError = err instanceof Error ? err.message : String(err);
      return null;
    }
  }

  async exportNativeEvalReport(): Promise<NativeEvalReportPayload | null> {
    this.lastError = null;
    const invoke = this.getTauriInvoke();
    if (!invoke) return null;
    try {
      const rep = await invoke<NativeEvalReportPayload>("native_export_eval_report");
      if (!rep || typeof rep.json_content !== "string") return null;
      return {
        filename: rep.filename || "openxvs_v5_eval_report.json",
        json_content: rep.json_content,
        sha256: rep.sha256 || "",
        non_flight_sim_only: rep.non_flight_sim_only !== false,
        actuator_authority: Boolean(rep.actuator_authority),
        watermark: rep.watermark || NON_FLIGHT_WATERMARK,
      };
    } catch (err) {
      this.lastError = err instanceof Error ? err.message : String(err);
      return null;
    }
  }

  async fetchHilStatus(): Promise<HilStatusPayload | null> {
    const invoke = this.getTauriInvoke();
    if (invoke) {
      try {
        return await invoke<HilStatusPayload>("native_hil_status");
      } catch {
        return null;
      }
    }
    if (!this.online) return null;
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 3000);
    try {
      const res = await fetch(`${this.baseUrl}/api/hil/status`, { signal: ctrl.signal });
      clearTimeout(timer);
      if (!res.ok) return null;
      return (await res.json()) as HilStatusPayload;
    } catch {
      clearTimeout(timer);
      return null;
    }
  }

  async runHilLoopbackTest(): Promise<HilLoopbackReport | null> {
    const invoke = this.getTauriInvoke();
    if (invoke) {
      try {
        return await invoke<HilLoopbackReport>("native_hil_loopback_test");
      } catch {
        return null;
      }
    }
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 10000);
    try {
      const res = await fetch(`${this.baseUrl}/api/hil/loopback_test`, {
        method: "POST",
        signal: ctrl.signal,
      });
      clearTimeout(timer);
      if (!res.ok) return null;
      return (await res.json()) as HilLoopbackReport;
    } catch {
      clearTimeout(timer);
      return null;
    }
  }

  async connectHilBridge(payload: {
    readonly mode: HilMode;
    readonly rx_port?: number;
    readonly tx_port?: number;
  }): Promise<HilStatusPayload | null> {
    this.lastError = null;
    const invoke = this.getTauriInvoke();
    if (invoke) {
      try {
        return await invoke<HilStatusPayload>("native_hil_connect", { mode: payload.mode });
      } catch (err) {
        this.lastError = err instanceof Error ? err.message : String(err);
        return null;
      }
    }
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 4000);
    try {
      const res = await fetch(`${this.baseUrl}/api/hil/connect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: ctrl.signal,
      });
      clearTimeout(timer);
      if (!res.ok) {
        try {
          const errDoc = (await res.json()) as { error?: string };
          if (errDoc?.error) this.lastError = errDoc.error;
        } catch {
          this.lastError = `HTTP ${res.status}`;
        }
        return null;
      }
      return (await res.json()) as HilStatusPayload;
    } catch {
      clearTimeout(timer);
      return null;
    }
  }

  async probeHealth(): Promise<BridgeHealthPayload | null> {
    // 1. Primary path: Native Tauri v2 Pure-Rust `ort` IPC (zero mixed-content, zero Python)
    const invoke = this.getTauriInvoke();
    if (invoke) {
      try {
        const data = await invoke<Record<string, unknown>>("native_health");
        const modelsObj = data.models as Record<string, unknown> | undefined;
        this.health = {
          ok: true,
          version: typeof data.version === "string" ? data.version : undefined,
          device: String(
            data.host_gpu_name ??
              data.compute_device ??
              "6/6 Pure-Rust ORT + XFeat + LighterGlue + MultiShipCoord",
          ),
          vramMb: typeof data.vram_mb === "number" ? data.vram_mb : 0,
          modelsLoaded: modelsObj
            ? Object.keys(modelsObj)
            : [
                "xvs_d2",
                "xvs_land",
                "xvs_thermal",
                "xvs_nav",
                "xvs_xfeat",
                "xvs_lighterglue",
                "xvs_active_range",
              ],
          transport: "tauri-ipc",
          nonFlightSimOnly: data.non_flight_sim_only !== false,
          watermark: String(data.watermark ?? NON_FLIGHT_WATERMARK),
        };
        this.online = true;
        this.transport = "tauri-ipc";
        this.consecutiveFailures = 0;
        return this.health;
      } catch (err) {
        this.lastError = err instanceof Error ? err.message : String(err);
        this.online = false;
        this.transport = "offline";
        return null;
      }
    }

    // 2. Fallback path: Browser Vite dev preview (`http://127.0.0.1:8765`)
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 2500);
      const res = await fetch(`${this.baseUrl}/api/health`, { signal: ctrl.signal });
      clearTimeout(timer);
      if (!res.ok) {
        this.online = false;
        this.transport = "offline";
        return null;
      }
      const data = (await res.json()) as Record<string, unknown>;
      const modelsObj = data.models as Record<string, unknown> | undefined;
      this.health = {
        ok: true,
        version: typeof data.version === "string" ? data.version : undefined,
        device: String(
          data.host_gpu_name ?? data.gpu_name ?? data.compute_device ?? data.device ?? "Local GPU",
        ),
        vramMb: typeof data.vram_mb === "number" ? data.vram_mb : null,
        modelsLoaded: modelsObj
          ? Object.keys(modelsObj)
          : ["xvs_d2", "xvs_land", "xvs_thermal", "xvs_nav"],
        transport: "http-dev",
        nonFlightSimOnly: Boolean(data.non_flight_sim_only ?? NON_FLIGHT_SIM_ONLY),
        watermark: String(data.watermark ?? NON_FLIGHT_WATERMARK),
      };
      this.online = true;
      this.transport = "http-dev";
      this.consecutiveFailures = 0;
      return this.health;
    } catch {
      this.online = false;
      this.transport = "offline";
      return null;
    }
  }

  private parseChannelEntry(id: SensorChannelId, entry: Record<string, unknown>): BridgeChannelResult {
    const imageDataUrl = normalizeBase64DataUrl(
      entry.annotated_jpeg_b64 ?? entry.image_b64 ?? entry.preview_b64,
    );
    const rawSha = entry.checkpoint_sha256 ?? entry.sha256;
    const checkpointSha256 = typeof rawSha === "string" && rawSha.length >= 8 ? rawSha : null;
    const checkpointShaShort = checkpointSha256 ? checkpointSha256.slice(0, 8) : null;
    const metrics =
      entry.metrics && typeof entry.metrics === "object"
        ? (entry.metrics as Record<string, unknown>)
        : null;
    return {
      channel: id,
      imageDataUrl,
      summary: String(entry.summary ?? entry.status ?? "LIVE"),
      latencyMs: typeof entry.latency_ms === "number" ? entry.latency_ms : null,
      confidence: typeof entry.confidence === "number" ? entry.confidence : null,
      checkpointSha256,
      checkpointShaShort,
      watermark: typeof entry.watermark === "string" ? entry.watermark : NON_FLIGHT_WATERMARK,
      metrics,
    };
  }

  private mapChannelsRecord(
    json: Record<string, unknown>,
  ): Partial<Record<SensorChannelId, BridgeChannelResult>> {
    const channels = (json.channels ?? json) as Record<string, Record<string, unknown>>;
    const out: Partial<Record<SensorChannelId, BridgeChannelResult>> = {};
    const mapKeys: ReadonlyArray<readonly [SensorChannelId, readonly string[]]> = [
      ["d2", ["d2", "eo_sky", "xvs_d2"]],
      ["land", ["land", "runway_land", "xvs_land"]],
      ["thermal", ["thermal", "thermal_dve", "xvs_thermal"]],
      ["nav", ["nav", "crossview_nav", "xvs_nav"]],
    ];
    for (const [id, aliases] of mapKeys) {
      const entry = aliases.map((k) => channels[k]).find((v) => v && typeof v === "object");
      if (!entry) continue;
      out[id] = this.parseChannelEntry(id, entry);
    }
    return out;
  }

  async fetchDemoTick(
    scenario: string,
  ): Promise<Partial<Record<SensorChannelId, BridgeChannelResult>> | null> {
    if (!this.online || this.inFlight) return null;
    this.inFlight = true;
    const frameNum = this.tickFrame++;

    const invoke = this.getTauriInvoke();
    if (invoke) {
      try {
        const json = await invoke<Record<string, unknown>>("native_demo_tick", {
          scenario,
          frame: frameNum,
          includeImages: true,
        });
        this.consecutiveFailures = 0;
        return this.mapChannelsRecord(json);
      } catch {
        this.consecutiveFailures++;
        if (this.consecutiveFailures >= 3) {
          this.online = false;
          this.transport = "offline";
        }
        return null;
      } finally {
        this.inFlight = false;
      }
    }

    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 8000);
    try {
      const res = await fetch(
        `${this.baseUrl}/api/demo_tick?scenario=${encodeURIComponent(scenario)}&frame=${frameNum}&images=1`,
        { signal: ctrl.signal },
      );
      clearTimeout(timer);
      if (!res.ok) {
        this.consecutiveFailures++;
        if (this.consecutiveFailures >= 3) {
          this.online = false;
          this.transport = "offline";
        }
        return null;
      }
      this.consecutiveFailures = 0;
      const json = (await res.json()) as Record<string, unknown>;
      return this.mapChannelsRecord(json);
    } catch {
      clearTimeout(timer);
      this.consecutiveFailures++;
      if (this.consecutiveFailures >= 3) {
        this.online = false;
        this.transport = "offline";
      }
      return null;
    } finally {
      this.inFlight = false;
    }
  }

  async analyzeImage(
    channel: SensorChannelId,
    bytes: ArrayBuffer,
    mimeType = "image/jpeg",
  ): Promise<BridgeChannelResult | null> {
    const invoke = this.getTauriInvoke();
    if (invoke) {
      try {
        const imageB64 = arrayBufferToBase64(bytes);
        const entry = await invoke<Record<string, unknown>>("native_analyze_image", {
          channel,
          imageB64,
        });
        return this.parseChannelEntry(channel, entry);
      } catch {
        return null;
      }
    }

    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 10000);
    try {
      const res = await fetch(`${this.baseUrl}/api/analyze_image?channel=${channel}`, {
        method: "POST",
        headers: { "Content-Type": mimeType },
        body: bytes,
        signal: ctrl.signal,
      });
      clearTimeout(timer);
      if (!res.ok) return null;
      const entry = (await res.json()) as Record<string, unknown>;
      return this.parseChannelEntry(channel, entry);
    } catch {
      clearTimeout(timer);
      return null;
    }
  }

  async analyzeFrameFast(
    channel: SensorChannelId,
    jpegBlob: Blob,
  ): Promise<BridgeChannelResult | null> {
    const invoke = this.getTauriInvoke();
    if (invoke) {
      try {
        const buf = await jpegBlob.arrayBuffer();
        return await this.analyzeImage(channel, buf, "image/jpeg");
      } catch {
        return null;
      }
    }

    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 6000);
    try {
      const res = await fetch(`${this.baseUrl}/api/analyze_frame_fast?channel=${channel}`, {
        method: "POST",
        headers: { "Content-Type": "image/jpeg" },
        body: jpegBlob,
        signal: ctrl.signal,
      });
      clearTimeout(timer);
      if (!res.ok) return null;
      const entry = (await res.json()) as Record<string, unknown>;
      return this.parseChannelEntry(channel, entry);
    } catch {
      clearTimeout(timer);
      return null;
    }
  }

  async startVideoAnalysis(
    file: Blob,
    channel: SensorChannelId,
    sampleFps: number,
    maxFrames: number,
  ): Promise<{ readonly job_id: string; readonly status: string } | null> {
    this.lastError = null;
    if (this.isNativeTauriIpc()) {
      this.lastError =
        "Standalone .exe uses Pure-Rust ORT sequential frame analysis via native IPC.";
      return null;
    }
    const qs = `channel=${encodeURIComponent(channel)}&sample_fps=${Math.round(sampleFps)}&max_frames=${Math.round(maxFrames)}`;
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 60000);
    try {
      const res = await fetch(`${this.baseUrl}/api/analyze_video?${qs}`, {
        method: "POST",
        headers: { "Content-Type": file.type || "video/mp4" },
        body: file,
        signal: ctrl.signal,
      });
      clearTimeout(timer);
      if (!res.ok) {
        this.lastError = `HTTP ${res.status}`;
        return null;
      }
      return (await res.json()) as { readonly job_id: string; readonly status: string };
    } catch (err) {
      clearTimeout(timer);
      this.lastError = err instanceof Error ? err.message : String(err);
      return null;
    }
  }

  async startBuiltinVideoAnalysis(
    channel: SensorChannelId,
    clip = "aot_realsky",
    sampleFps = 10,
    maxFrames = 60,
  ): Promise<{ readonly job_id: string; readonly status: string } | null> {
    this.lastError = null;
    if (this.isNativeTauriIpc()) {
      this.lastError =
        "Standalone .exe runs built-in flight sequences natively via native_video_sequence_tick.";
      return null;
    }
    const qs = `channel=${encodeURIComponent(channel)}&clip=${encodeURIComponent(clip)}&sample_fps=${Math.round(sampleFps)}&max_frames=${Math.round(maxFrames)}`;
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 15000);
    try {
      const res = await fetch(`${this.baseUrl}/api/analyze_builtin_video?${qs}`, {
        method: "POST",
        signal: ctrl.signal,
      });
      clearTimeout(timer);
      if (!res.ok) {
        this.lastError = `HTTP ${res.status}`;
        return null;
      }
      return (await res.json()) as { readonly job_id: string; readonly status: string };
    } catch (err) {
      clearTimeout(timer);
      this.lastError = err instanceof Error ? err.message : String(err);
      return null;
    }
  }

  async pollVideoJob(jobId: string): Promise<VideoJobStatus | null> {
    if (this.isNativeTauriIpc()) return null;
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 8000);
    try {
      const res = await fetch(`${this.baseUrl}/api/video_job/${encodeURIComponent(jobId)}`, {
        signal: ctrl.signal,
      });
      clearTimeout(timer);
      if (!res.ok) return null;
      return (await res.json()) as VideoJobStatus;
    } catch {
      clearTimeout(timer);
      return null;
    }
  }
}

interface FallbackGradientCache {
  readonly w: number;
  readonly h: number;
  readonly d2Sky: CanvasGradient;
  readonly lwir: CanvasGradient;
}

const fallbackGradientMap = new WeakMap<CanvasRenderingContext2D, FallbackGradientCache>();

function getFallbackGradients(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
): FallbackGradientCache {
  const cached = fallbackGradientMap.get(ctx);
  if (cached && cached.w === w && cached.h === h) {
    return cached;
  }
  const d2Sky = ctx.createLinearGradient(0, 0, 0, h);
  d2Sky.addColorStop(0, "#071326");
  d2Sky.addColorStop(0.62, "#133154");
  d2Sky.addColorStop(0.75, "#3a4f6b");
  d2Sky.addColorStop(1, "#0a111c");

  const splitX = Math.floor(w * 0.44);
  const lwir = ctx.createLinearGradient(splitX, 0, w, h);
  lwir.addColorStop(0, "#120726");
  lwir.addColorStop(0.55, "#4c1d95");
  lwir.addColorStop(1, "#f59e0b");

  const next: FallbackGradientCache = { w, h, d2Sky, lwir };
  fallbackGradientMap.set(ctx, next);
  return next;
}

/** Formats crisp HTML/CSS overlay telemetry captions for the 4 fallback sensor cards (never cropped). */
export function formatFallbackChannelCaption(
  channel: SensorChannelId,
  snap: StudioSimSnapshot,
): string {
  if (channel === "d2") {
    return `CBDR-1 ${Math.round(snap.activeRange.rHatM)}m ±${Math.round(snap.activeRange.sigmaRM)}m`;
  }
  if (channel === "land") {
    return "28L 12-KPT · SimCC 3.41px · HMI 0";
  }
  if (channel === "thermal") {
    return `EO | LWIR · EASA 5/5 · V=${snap.thermalVacuity.toFixed(2)}`;
  }
  return snap.spoofAlarm
    ? "SPOOF ISOLATED · 8-ARM CUSUM · R@5 96.7%"
    : "MNV4 · KSFO R@1 91.4% · 13.1ms P50";
}

/**
 * Renders crisp live avionics synthetic sensor thumbnails when native/dev AI backend is offline.
 * Telemetry text is rendered in HTML `.studio-channel-caption` so zero pixels are ever cropped.
 */
export function renderFallbackSensorCanvas(
  canvas: HTMLCanvasElement,
  channel: SensorChannelId,
  snap: StudioSimSnapshot,
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const w = canvas.width;
  const h = canvas.height;
  const t = snap.simTimeS;
  const grads = getFallbackGradients(ctx, w, h);

  ctx.clearRect(0, 0, w, h);

  if (channel === "d2") {
    ctx.fillStyle = grads.d2Sky;
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = "rgba(56, 189, 248, 0.35)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h * 0.74);
    ctx.lineTo(w, h * 0.74);
    ctx.stroke();

    const tx = w * (0.52 + 0.14 * Math.sin(t * 0.8));
    const ty = h * (0.56 + 0.04 * Math.cos(t * 1.1));
    ctx.fillStyle = "#e2e8f0";
    ctx.fillRect(tx - 5, ty - 2, 10, 4);
    ctx.strokeStyle = snap.activeRange.sigmaRM < 100 ? "#34d399" : "#22d3ee";
    ctx.lineWidth = 1.6;
    ctx.strokeRect(tx - 16, ty - 11, 32, 22);
  } else if (channel === "land") {
    ctx.fillStyle = "#080f1d";
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "rgba(56, 189, 248, 0.2)";
    for (let i = 1; i < 5; i++) {
      ctx.beginPath();
      ctx.moveTo(0, (h * i) / 5);
      ctx.lineTo(w, (h * i) / 5);
      ctx.stroke();
    }
    const cx = w * 0.5 + Math.sin(t * 0.5) * 6;
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(cx - 12, h * 0.34);
    ctx.lineTo(cx + 12, h * 0.34);
    ctx.lineTo(cx + 54, h * 0.94);
    ctx.lineTo(cx - 54, h * 0.94);
    ctx.closePath();
    ctx.stroke();
    ctx.fillStyle = "#34d399";
    for (let k = 0; k < 6; k++) {
      const f = k / 5;
      const y = h * (0.34 + 0.6 * f);
      const hw = 12 + 42 * f;
      ctx.fillRect(cx - hw - 2, y - 2, 4, 4);
      ctx.fillRect(cx + hw - 2, y - 2, 4, 4);
    }
  } else if (channel === "thermal") {
    const splitX = Math.floor(w * 0.44);
    ctx.fillStyle = snap.weather.visibilityKm < 4.0 ? "#94a3b8" : "#1e293b";
    ctx.fillRect(0, 0, splitX, h);
    ctx.fillStyle = grads.lwir;
    ctx.fillRect(splitX, 0, w - splitX, h);
    ctx.strokeStyle = "#fde68a";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(splitX + 26, h * 0.34);
    ctx.lineTo(splitX + 64, h * 0.92);
    ctx.moveTo(splitX + 42, h * 0.34);
    ctx.lineTo(splitX + 94, h * 0.92);
    ctx.stroke();
    ctx.strokeStyle = "#a78bfa";
    ctx.beginPath();
    ctx.moveTo(splitX, 0);
    ctx.lineTo(splitX, h);
    ctx.stroke();
  } else {
    ctx.fillStyle = "#06101c";
    ctx.fillRect(0, 0, w, h);
    const mid = w * 0.5;
    ctx.strokeStyle = "rgba(148, 163, 184, 0.25)";
    ctx.strokeRect(4, 24, mid - 8, h - 29);
    ctx.strokeRect(mid + 4, 24, mid - 8, h - 29);
    ctx.strokeStyle = snap.spoofAlarm ? "#f59e0b" : "#34d399";
    ctx.lineWidth = 1.2;
    for (let i = 0; i < 5; i++) {
      const y1 = 30 + i * 9 + Math.sin(t + i) * 1.8;
      const y2 = 30 + i * 9 + (snap.spoofAlarm ? 3 : 0);
      ctx.beginPath();
      ctx.moveTo(18 + (i % 3) * 18, y1);
      ctx.lineTo(mid + 18 + (i % 3) * 18, y2);
      ctx.stroke();
    }
  }
}
