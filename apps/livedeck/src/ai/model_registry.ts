/**
 * OpenXVS Live Deck Stage-1 Model Registry & SHA-256 Verified OPFS Loader.
 *
 * Reads `public/models/registry.json`, verifies model artifacts via Web Crypto SHA-256
 * before creating any ONNX Runtime session, and caches verified bytes in browser OPFS
 * with explicit (never silent) fallback to in-memory ArrayBuffer when OPFS is unavailable.
 */

export interface ModelArtifactSpec {
  readonly path: string;
  readonly bytes: number;
  readonly sha256: string;
}

export interface ModelTensorSpec {
  readonly name: string;
  readonly dtype: string;
  readonly shape: readonly number[];
}

export interface ActiveModelEntry {
  readonly id: string;
  readonly display_name: string;
  readonly status: "active";
  readonly milestone: string;
  readonly source_checkpoint: string;
  readonly source_checkpoint_sha256: string;
  readonly tau_star: number;
  readonly opset: number;
  readonly input: ModelTensorSpec;
  readonly outputs: readonly ModelTensorSpec[];
  readonly preprocessing: {
    readonly forward_normalized_only: boolean;
    readonly ring_intrinsics: string;
    readonly fx_px: number;
    readonly fy_px: number;
    readonly kept_frame_step_dt_s: number;
    readonly full_well_e: number;
    readonly std0_clamp_min: number;
    readonly norm_clamp_range: readonly [number, number];
  };
  readonly artifacts: {
    readonly fp32: ModelArtifactSpec;
    readonly fp16: ModelArtifactSpec;
  };
}

export interface XvsD2EvalTierMetrics {
  readonly pd_at_fpi_0p1?: number;
  readonly pd_at_fpi_0p5?: number;
  readonly pd_at_fpi_1p0?: number;
  readonly ap50?: number;
  readonly centroid_rmse_px?: number;
  readonly above_horizon_pd?: number;
  readonly below_horizon_pd?: number;
  readonly num_eval_clips?: number;
  readonly num_pos_clips?: number;
  readonly num_neg_clips?: number;
}

export interface XvsD2ModelEntry {
  readonly id: string;
  readonly display_name: string;
  readonly status:
    | "local_v2_m2_architecture_seed_awaiting_colab_g4_weights"
    | "trained_colab_g4_blackwell"
    | "active";
  readonly milestone: string;
  readonly opset: number;
  readonly inputs: readonly ModelTensorSpec[];
  readonly outputs: readonly ModelTensorSpec[];
  readonly artifacts: {
    readonly fp16: ModelArtifactSpec;
    readonly fp32?: ModelArtifactSpec;
  };
  readonly notes?: string;
  readonly source_archive_sha256?: string;
  readonly trained_eval_metrics?: {
    readonly teacher?: XvsD2EvalTierMetrics;
    readonly edge?: XvsD2EvalTierMetrics;
    readonly edge_slim?: XvsD2EvalTierMetrics;
    readonly web?: XvsD2EvalTierMetrics;
  };
}

export interface ReservedModelEntry {
  readonly id: string;
  readonly display_name: string;
  readonly status: "reserved_m2";
  readonly milestone: string;
  readonly notes: string;
}

export type ModelRegistryEntry = ActiveModelEntry | XvsD2ModelEntry | ReservedModelEntry;

export interface ModelRegistryDocument {
  readonly schema: string;
  readonly active_model_id: string;
  readonly models: readonly ModelRegistryEntry[];
}

export type CacheSource = "opfs-hit" | "network-opfs-stored" | "network-memory-no-opfs";

export interface VerifiedModelPayload {
  readonly bytes: Uint8Array;
  readonly sha256: string;
  readonly cacheSource: CacheSource;
  readonly opfsWarning: string | null;
}

export async function computeSha256Hex(data: ArrayBuffer | Uint8Array): Promise<string> {
  const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  const digest = await crypto.subtle.digest("SHA-256", copy);
  const view = new Uint8Array(digest);
  let hex = "";
  for (let i = 0; i < view.length; i++) {
    hex += view[i].toString(16).padStart(2, "0");
  }
  return hex;
}

export async function loadModelRegistry(baseUrl = "./"): Promise<ModelRegistryDocument> {
  const normBase = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  const resp = await fetch(`${normBase}models/registry.json`, { cache: "no-cache" });
  if (!resp.ok) {
    throw new Error(`Failed to fetch models/registry.json: HTTP ${resp.status}`);
  }
  const doc = (await resp.json()) as ModelRegistryDocument;
  if (doc.schema !== "xvs.livedeck.model_registry/1") {
    throw new Error(`Unsupported model registry schema: ${String(doc.schema)}`);
  }
  return doc;
}

export function getActiveModelEntry(registry: ModelRegistryDocument): ActiveModelEntry {
  const entry = registry.models.find(
    (m): m is ActiveModelEntry =>
      m.id === registry.active_model_id && m.status === "active" && "input" in m,
  );
  if (!entry) {
    throw new Error(`Active model '${registry.active_model_id}' not found in registry.json`);
  }
  return entry;
}

export function getXvsD2ModelEntry(
  registry: ModelRegistryDocument,
  modelId = "xvs_d2_web_fp16",
): XvsD2ModelEntry | null {
  const entry = registry.models.find(
    (m): m is XvsD2ModelEntry => m.id === modelId && "inputs" in m && "artifacts" in m,
  );
  return entry ?? null;
}

interface OpfsDirLike {
  getDirectoryHandle(name: string, opts?: { create?: boolean }): Promise<OpfsDirLike>;
  getFileHandle(name: string, opts?: { create?: boolean }): Promise<OpfsFileLike>;
}

interface OpfsFileLike {
  getFile(): Promise<Blob>;
  createWritable(): Promise<{
    write(data: Uint8Array<ArrayBuffer>): Promise<void>;
    close(): Promise<void>;
  }>;
}

async function tryGetOpfsModelDir(): Promise<{ dir: OpfsDirLike | null; reason: string | null }> {
  try {
    const nav = globalThis.navigator as
      | (Navigator & { storage?: { getDirectory?: () => Promise<OpfsDirLike> } })
      | undefined;
    if (!nav?.storage?.getDirectory) {
      return { dir: null, reason: "OPFS navigator.storage.getDirectory unavailable" };
    }
    const root = await nav.storage.getDirectory();
    const dir = await root.getDirectoryHandle("openxvs-models", { create: true });
    return { dir, reason: null };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return { dir: null, reason: `OPFS unavailable (${msg})` };
  }
}

export async function loadVerifiedModelBytes(
  artifact: ModelArtifactSpec,
  baseUrl = "./",
): Promise<VerifiedModelPayload> {
  const expectedSha = artifact.sha256.toLowerCase();
  const cacheName = `${expectedSha}.onnx`;
  const opfsProbe = await tryGetOpfsModelDir();

  // 1. Attempt OPFS cache hit with strict SHA-256 verification
  if (opfsProbe.dir) {
    try {
      const fh = await opfsProbe.dir.getFileHandle(cacheName, { create: false });
      const blob = await fh.getFile();
      const cachedBytes = new Uint8Array(await blob.arrayBuffer());
      const actualSha = await computeSha256Hex(cachedBytes);
      if (actualSha === expectedSha) {
        return {
          bytes: cachedBytes,
          sha256: actualSha,
          cacheSource: "opfs-hit",
          opfsWarning: null,
        };
      }
    } catch {
      // Cache miss — fall through to verified local fetch
    }
  }

  // 2. Fetch model artifact from same-origin public/models/
  const normBase = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  const url = `${normBase}${artifact.path}`;
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Failed to fetch ONNX artifact ${url}: HTTP ${resp.status}`);
  }
  const bytes = new Uint8Array(await resp.arrayBuffer());
  const actualSha = await computeSha256Hex(bytes);
  if (actualSha !== expectedSha) {
    throw new Error(
      `SHA-256 mismatch for ${artifact.path}: expected ${expectedSha}, got ${actualSha}`,
    );
  }

  // 3. Persist into OPFS when available; surface explicit warning otherwise (never silent)
  if (opfsProbe.dir) {
    try {
      const fh = await opfsProbe.dir.getFileHandle(cacheName, { create: true });
      const writable = await fh.createWritable();
      await writable.write(bytes);
      await writable.close();
      return {
        bytes,
        sha256: actualSha,
        cacheSource: "network-opfs-stored",
        opfsWarning: null,
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return {
        bytes,
        sha256: actualSha,
        cacheSource: "network-memory-no-opfs",
        opfsWarning: `OPFS write failed (${msg}); using in-memory ArrayBuffer`,
      };
    }
  }

  return {
    bytes,
    sha256: actualSha,
    cacheSource: "network-memory-no-opfs",
    opfsWarning: opfsProbe.reason ?? "OPFS unavailable; using in-memory ArrayBuffer",
  };
}
