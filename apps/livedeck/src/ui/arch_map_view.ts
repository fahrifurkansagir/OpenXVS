/**
 * Tab 2 (`Kim Ne Yapıyor?` / `Live System Architecture`) Interactive Live-Pulsing 8-Node SVG Flow Map.
 * Pure TypeScript + SVG + DOM (zero dependencies), bilingual TR/EN, click-to-inspect explanation card.
 */

import {
  ARCH_NODE_IDS,
  DEFAULT_LOCALE,
  getObservatoryStrings,
  type ArchNodeId,
  type ObservatoryLocale,
} from "./i18n.ts";

export interface ArchLiveMetrics {
  readonly fps: number;
  readonly detectorHz: number;
  readonly detectorMs: number;
  readonly kaniPassed: number;
  readonly kaniTotal: number;
  readonly activeRangeMode: string;
  readonly forcedLandActive: boolean;
  readonly halHealthy: boolean;
  readonly nb2Status?: string;
}

export interface TrainedSuiteProvenance {
  readonly d2Ap50: number;
  readonly d2AboveHorizonPd: number;
  readonly navRecallAt1: number;
  readonly navRecallAt5: number;
  readonly navShaShort: string;
  readonly landSimccRmsePx: number;
  readonly landHmiCount: number;
  readonly landShaShort: string;
  readonly thermalEasaPass: boolean;
  readonly thermalDualBlindVacuity: number;
  readonly thermalShaShort: string;
  readonly worldPsnrDb: number;
  readonly l4GpuLabel: string;
  readonly l4TensorsChecked: number;
  readonly l4NonFiniteTensors: number;
  readonly l4ArchiveShaShort: string;
  readonly hydratedFromRegistry: boolean;
}

export const DEFAULT_TRAINED_PROVENANCE: TrainedSuiteProvenance = {
  d2Ap50: 0.4925876665703411,
  d2AboveHorizonPd: 0.7804878048780488,
  navRecallAt1: 0.818359375,
  navRecallAt5: 0.9453125,
  navShaShort: "6a21fffa",
  landSimccRmsePx: 3.4098807545148366,
  landHmiCount: 0,
  landShaShort: "5d930932",
  thermalEasaPass: true,
  thermalDualBlindVacuity: 1.0,
  thermalShaShort: "009c7cca",
  worldPsnrDb: 56.6,
  l4GpuLabel: "Colab L4",
  l4TensorsChecked: 336,
  l4NonFiniteTensors: 0,
  l4ArchiveShaShort: "46b9364b",
  hydratedFromRegistry: false,
};

interface RawTrainedSuiteL4 {
  readonly status?: string;
  readonly gpu?: string;
  readonly archive_sha256?: string;
  readonly tensors_checked?: number;
  readonly non_finite_tensor_count?: number;
  readonly models?: {
    readonly xvs_nav?: {
      readonly checkpoint_sha256?: string;
      readonly metrics?: { readonly recall_at_1?: number; readonly recall_at_5?: number };
    };
    readonly xvs_land?: {
      readonly checkpoint_sha256?: string;
      readonly metrics?: { readonly simcc_rmse_px?: number; readonly hmi_count?: number };
    };
    readonly xvs_thermal?: {
      readonly checkpoint_sha256?: string;
      readonly metrics?: {
        readonly vraim_easa_rlm_07_11_pass?: boolean;
        readonly dual_blind_vacuity_mean?: number;
      };
    };
  };
}

interface RawRegistryShape {
  readonly models?: ReadonlyArray<{
    readonly id?: string;
    readonly trained_eval_metrics?: {
      readonly web?: { readonly ap50?: number; readonly above_horizon_pd?: number };
    };
  }>;
  readonly trained_suite_l4?: RawTrainedSuiteL4;
}

/**
 * Safely extracts real-GPU model provenance from an already-loaded `registry.json` document.
 */
export function extractTrainedProvenanceFromRegistry(rawRegistry: unknown): Partial<TrainedSuiteProvenance> {
  if (!rawRegistry || typeof rawRegistry !== "object") return {};
  const reg = rawRegistry as RawRegistryShape;
  const out: {
    d2Ap50?: number;
    d2AboveHorizonPd?: number;
    navRecallAt1?: number;
    navRecallAt5?: number;
    navShaShort?: string;
    landSimccRmsePx?: number;
    landHmiCount?: number;
    landShaShort?: string;
    thermalEasaPass?: boolean;
    thermalDualBlindVacuity?: number;
    thermalShaShort?: string;
    l4GpuLabel?: string;
    l4TensorsChecked?: number;
    l4NonFiniteTensors?: number;
    l4ArchiveShaShort?: string;
    hydratedFromRegistry?: boolean;
  } = {};

  const d2Web = reg.models?.find((m) => m.id === "xvs_d2_web_fp16")?.trained_eval_metrics?.web;
  if (typeof d2Web?.ap50 === "number" && Number.isFinite(d2Web.ap50)) {
    out.d2Ap50 = d2Web.ap50;
  }
  if (typeof d2Web?.above_horizon_pd === "number" && Number.isFinite(d2Web.above_horizon_pd)) {
    out.d2AboveHorizonPd = d2Web.above_horizon_pd;
  }

  const l4 = reg.trained_suite_l4;
  if (l4 && typeof l4 === "object") {
    out.hydratedFromRegistry = true;
    if (typeof l4.gpu === "string" && l4.gpu.trim().length > 0) {
      out.l4GpuLabel = l4.gpu.includes("L4") ? "Colab L4" : l4.gpu.trim();
    }
    if (typeof l4.tensors_checked === "number" && Number.isFinite(l4.tensors_checked)) {
      out.l4TensorsChecked = l4.tensors_checked;
    }
    if (typeof l4.non_finite_tensor_count === "number" && Number.isFinite(l4.non_finite_tensor_count)) {
      out.l4NonFiniteTensors = l4.non_finite_tensor_count;
    }
    if (typeof l4.archive_sha256 === "string" && l4.archive_sha256.length >= 8) {
      out.l4ArchiveShaShort = l4.archive_sha256.slice(0, 8);
    }

    const nav = l4.models?.xvs_nav;
    if (typeof nav?.metrics?.recall_at_1 === "number") out.navRecallAt1 = nav.metrics.recall_at_1;
    if (typeof nav?.metrics?.recall_at_5 === "number") out.navRecallAt5 = nav.metrics.recall_at_5;
    if (typeof nav?.checkpoint_sha256 === "string" && nav.checkpoint_sha256.length >= 8) {
      out.navShaShort = nav.checkpoint_sha256.slice(0, 8);
    }

    const land = l4.models?.xvs_land;
    if (typeof land?.metrics?.simcc_rmse_px === "number") out.landSimccRmsePx = land.metrics.simcc_rmse_px;
    if (typeof land?.metrics?.hmi_count === "number") out.landHmiCount = land.metrics.hmi_count;
    if (typeof land?.checkpoint_sha256 === "string" && land.checkpoint_sha256.length >= 8) {
      out.landShaShort = land.checkpoint_sha256.slice(0, 8);
    }

    const thermal = l4.models?.xvs_thermal;
    if (typeof thermal?.metrics?.vraim_easa_rlm_07_11_pass === "boolean") {
      out.thermalEasaPass = thermal.metrics.vraim_easa_rlm_07_11_pass;
    }
    if (typeof thermal?.metrics?.dual_blind_vacuity_mean === "number") {
      out.thermalDualBlindVacuity = thermal.metrics.dual_blind_vacuity_mean;
    }
    if (typeof thermal?.checkpoint_sha256 === "string" && thermal.checkpoint_sha256.length >= 8) {
      out.thermalShaShort = thermal.checkpoint_sha256.slice(0, 8);
    }
  }

  return out;
}

export interface ArchNodeLayout {
  readonly id: ArchNodeId;
  readonly x: number;
  readonly y: number;
  readonly w: number;
  readonly h: number;
  readonly accent: string;
}

export interface ArchFlowEdge {
  readonly from: ArchNodeId;
  readonly to: ArchNodeId;
  readonly color: string;
  readonly dashed?: boolean;
}

export const ARCH_NODE_LAYOUTS: readonly ArchNodeLayout[] = [
  { id: "sensors", x: 24, y: 34, w: 196, h: 92, accent: "#38bdf8" },
  { id: "xvs_d2", x: 262, y: 34, w: 196, h: 92, accent: "#22d3ee" },
  { id: "xvs_activerange", x: 500, y: 34, w: 196, h: 92, accent: "#34d399" },
  { id: "sc3bf_kani", x: 738, y: 34, w: 198, h: 92, accent: "#10b981" },
  { id: "xvs_forcedland", x: 24, y: 204, w: 196, h: 92, accent: "#f59e0b" },
  { id: "xvs_os_hal", x: 262, y: 204, w: 196, h: 92, accent: "#a78bfa" },
  { id: "xvs_fleet", x: 500, y: 204, w: 196, h: 92, accent: "#60a5fa" },
  { id: "webgpu_observatory", x: 738, y: 204, w: 198, h: 92, accent: "#2dd4bf" },
];

export const ARCH_FLOW_EDGES: readonly ArchFlowEdge[] = [
  { from: "sensors", to: "xvs_d2", color: "#38bdf8" },
  { from: "xvs_d2", to: "xvs_activerange", color: "#22d3ee" },
  { from: "xvs_activerange", to: "sc3bf_kani", color: "#34d399" },
  { from: "sc3bf_kani", to: "webgpu_observatory", color: "#10b981" },
  { from: "sensors", to: "xvs_os_hal", color: "#a78bfa", dashed: true },
  { from: "xvs_os_hal", to: "xvs_fleet", color: "#60a5fa" },
  { from: "xvs_fleet", to: "webgpu_observatory", color: "#2dd4bf" },
  { from: "xvs_forcedland", to: "webgpu_observatory", color: "#f59e0b", dashed: true },
];

const DEFAULT_METRICS: ArchLiveMetrics = {
  fps: 60,
  detectorHz: 15,
  detectorMs: 5.8,
  kaniPassed: 18,
  kaniTotal: 18,
  activeRangeMode: "NOMINAL_HOLD",
  forcedLandActive: false,
  halHealthy: true,
};

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function getLayoutById(id: ArchNodeId): ArchNodeLayout {
  const found = ARCH_NODE_LAYOUTS.find((n) => n.id === id);
  return found ?? ARCH_NODE_LAYOUTS[0];
}

function formatNodeLiveBadge(id: ArchNodeId, metrics: ArchLiveMetrics, fallbackBadge: string): string {
  switch (id) {
    case "sensors":
      return `${Math.round(metrics.fps)} Hz · 6+1 CAM`;
    case "xvs_d2":
      return `${metrics.detectorHz.toFixed(0)} Hz · ${metrics.detectorMs.toFixed(1)} ms`;
    case "xvs_activerange":
      return metrics.activeRangeMode;
    case "sc3bf_kani":
      return `${metrics.kaniPassed}/${metrics.kaniTotal} Kani`;
    case "xvs_forcedland":
      return metrics.forcedLandActive ? "ACTIVE 35 KT" : fallbackBadge;
    case "xvs_os_hal":
      return metrics.halHealthy ? "CRC OK · 48B" : "ISOLATED";
    case "xvs_fleet":
      return fallbackBadge;
    case "webgpu_observatory":
      return `${Math.round(metrics.fps)} FPS · 0.98m`;
  }
}

/**
 * Pure SVG markup renderer for the 8-node live data-flow diagram (directly unit-testable in Node).
 */
export function renderArchMapSvgMarkup(
  locale: ObservatoryLocale,
  selectedNodeId: ArchNodeId,
  metrics: ArchLiveMetrics = DEFAULT_METRICS,
): string {
  const strings = getObservatoryStrings(locale);

  const edgeSvg = ARCH_FLOW_EDGES.map((edge, idx) => {
    const a = getLayoutById(edge.from);
    const b = getLayoutById(edge.to);
    const sameRow = Math.abs(a.y - b.y) < 10;
    let d: string;
    if (sameRow) {
      const x1 = a.x + a.w;
      const y1 = a.y + a.h * 0.5;
      const x2 = b.x;
      const y2 = b.y + b.h * 0.5;
      const mx = (x1 + x2) * 0.5;
      d = `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`;
    } else if (edge.from === "xvs_forcedland" && edge.to === "webgpu_observatory") {
      // Smooth bottom arc connecting ForcedLand -> Observatory without crossing inner nodes
      const x1 = a.x + a.w * 0.5;
      const y1 = a.y + a.h;
      const x2 = b.x + b.w * 0.5;
      const y2 = b.y + b.h;
      d = `M ${x1} ${y1} C ${x1} ${y1 + 48}, ${x2} ${y2 + 48}, ${x2} ${y2}`;
    } else {
      const x1 = a.x + a.w * 0.5;
      const y1 = a.y + a.h;
      const x2 = b.x + b.w * 0.5;
      const y2 = b.y;
      const my = (y1 + y2) * 0.5;
      d = `M ${x1} ${y1} C ${x1} ${my}, ${x2} ${my}, ${x2} ${y2}`;
    }

    const dashAttr = edge.dashed ? ' stroke-dasharray="6 5"' : "";
    const pathId = `arch-edge-${idx}`;
    const durS = (1.6 + (idx % 3) * 0.45).toFixed(2);
    return (
      `<path id="${pathId}" d="${d}" fill="none" stroke="${edge.color}" stroke-opacity="0.55" stroke-width="2.2"${dashAttr} />` +
      `<circle r="3.4" fill="${edge.color}">` +
      `<animateMotion dur="${durS}s" repeatCount="indefinite" path="${d}" />` +
      `</circle>`
    );
  }).join("");

  const nodeSvg = ARCH_NODE_LAYOUTS.map((layout) => {
    const info = strings.archNodes[layout.id];
    const selected = layout.id === selectedNodeId;
    const liveBadge = formatNodeLiveBadge(layout.id, metrics, info.badge);
    const strokeW = selected ? 2.6 : 1.25;
    const fill = selected ? "rgba(15, 32, 52, 0.96)" : "rgba(10, 20, 34, 0.88)";
    const glowAttr = selected ? ' filter="url(#arch-node-glow)"' : "";

    return (
      `<g class="arch-node-card" data-arch-node="${layout.id}" tabindex="0" role="button" style="cursor:pointer"${glowAttr}>` +
      `<rect x="${layout.x}" y="${layout.y}" width="${layout.w}" height="${layout.h}" rx="10" fill="${fill}" stroke="${layout.accent}" stroke-width="${strokeW}" />` +
      `<rect x="${layout.x + 10}" y="${layout.y + 10}" width="${layout.w - 20}" height="20" rx="5" fill="${layout.accent}" fill-opacity="0.16" />` +
      `<text x="${layout.x + 18}" y="${layout.y + 24}" fill="${layout.accent}" font-size="11" font-weight="700" font-family="monospace">${escapeHtml(liveBadge)}</text>` +
      `<text x="${layout.x + 14}" y="${layout.y + 52}" fill="#f1f5f9" font-size="14" font-weight="700" font-family="system-ui,sans-serif">${escapeHtml(info.title)}</text>` +
      `<text x="${layout.x + 14}" y="${layout.y + 72}" fill="#94a3b8" font-size="11.5" font-family="system-ui,sans-serif">${escapeHtml(info.subtitle)}</text>` +
      `</g>`
    );
  }).join("");

  return (
    `<svg class="arch-flow-svg" viewBox="0 0 960 360" width="100%" role="img" aria-label="${escapeHtml(strings.archView.heading)}">` +
    `<defs>` +
    `<filter id="arch-node-glow" x="-15%" y="-15%" width="130%" height="130%">` +
    `<feDropShadow dx="0" dy="0" stdDeviation="5" flood-color="#22d3ee" flood-opacity="0.55" />` +
    `</filter>` +
    `</defs>` +
    `<g class="arch-edges">${edgeSvg}</g>` +
    `<g class="arch-nodes">${nodeSvg}</g>` +
    `</svg>`
  );
}

/**
 * Pure HTML renderer for the verified real-GPU Trained Model Provenance strip in Tab 2
 * (directly unit-testable in Node without jsdom).
 */
export function renderModelProvenanceStripHtml(
  locale: ObservatoryLocale,
  provenance: TrainedSuiteProvenance = DEFAULT_TRAINED_PROVENANCE,
): string {
  const heading =
    locale === "tr"
      ? "Eğitilmiş Model Provenansı (Gerçek GPU)"
      : "Trained Model Provenance (Real GPU)";
  const finiteTag =
    provenance.l4NonFiniteTensors === 0
      ? locale === "tr"
        ? `${provenance.l4TensorsChecked}/${provenance.l4TensorsChecked} sonlu tensör`
        : `${provenance.l4TensorsChecked}/${provenance.l4TensorsChecked} finite tensors`
      : `${provenance.l4NonFiniteTensors} non-finite`;
  const sourceTag = provenance.hydratedFromRegistry
    ? `registry.json · ${finiteTag} · sha ${provenance.l4ArchiveShaShort}`
    : `${finiteTag} · sha ${provenance.l4ArchiveShaShort}`;

  const pills: ReadonlyArray<{
    readonly id: string;
    readonly name: string;
    readonly detail: string;
    readonly accent: string;
    readonly sha?: string;
  }> = [
    {
      id: "xvs_d2",
      name: "XVS-D2",
      detail: `Colab G4 · AP50 ${provenance.d2Ap50.toFixed(3)} · Pd ${(provenance.d2AboveHorizonPd * 100).toFixed(1)}%`,
      accent: "#22d3ee",
    },
    {
      id: "xvs_nav",
      name: "XVS-Nav",
      detail: `${provenance.l4GpuLabel} · R@1 ${(provenance.navRecallAt1 * 100).toFixed(1)}% · R@5 ${(provenance.navRecallAt5 * 100).toFixed(1)}%`,
      accent: "#34d399",
      sha: provenance.navShaShort,
    },
    {
      id: "xvs_land",
      name: "XVS-Land",
      detail: `${provenance.l4GpuLabel} · SimCC ${provenance.landSimccRmsePx.toFixed(2)}px · HMI ${provenance.landHmiCount}`,
      accent: "#38bdf8",
      sha: provenance.landShaShort,
    },
    {
      id: "xvs_thermal",
      name: "XVS-Thermal",
      detail: `${provenance.l4GpuLabel} · ${provenance.thermalEasaPass ? "EASA 5/5" : "EASA CHECK"} · Vacuity ${provenance.thermalDualBlindVacuity.toFixed(2)}`,
      accent: "#a78bfa",
      sha: provenance.thermalShaShort,
    },
    {
      id: "xvs_world",
      name: "XVS-World",
      detail: `Colab A100 · PSNR ${provenance.worldPsnrDb.toFixed(1)} dB`,
      accent: "#f59e0b",
    },
  ];

  const pillHtml = pills
    .map(
      (p) =>
        `<div data-provenance-model="${p.id}" title="${escapeHtml(p.sha ? `${p.name} sha256:${p.sha}` : p.name)}" ` +
        `style="display:flex;align-items:center;gap:6px;padding:5px 9px;border-radius:8px;` +
        `background:rgba(15,23,42,0.82);border:1px solid ${p.accent}55;font-size:11.5px;line-height:1.3;">` +
        `<span style="font-weight:700;color:${p.accent};white-space:nowrap;">${escapeHtml(p.name)}</span>` +
        `<span style="color:#e2e8f0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;white-space:nowrap;">${escapeHtml(p.detail)}</span>` +
        `</div>`,
    )
    .join("");

  return (
    `<div class="arch-inspector-card" data-arch-provenance-strip="true" data-hydrated="${provenance.hydratedFromRegistry}" ` +
    `style="margin-top:10px;padding:10px 14px;">` +
    `<div style="display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px;">` +
    `<span class="arch-inspector-label" style="margin:0;">${escapeHtml(heading)}</span>` +
    `<span class="arch-inspector-badge" style="font-size:11px;">${escapeHtml(sourceTag)}</span>` +
    `</div>` +
    `<div style="display:flex;flex-wrap:wrap;gap:8px;">${pillHtml}</div>` +
    `</div>`
  );
}

/**
 * Interactive DOM controller for Tab 2 (`Kim Ne Yapıyor?` / `Live System Architecture`).
 */
export class ArchMapView {
  private readonly root: HTMLElement;
  private locale: ObservatoryLocale;
  private selectedNodeId: ArchNodeId = "xvs_activerange";
  private metrics: ArchLiveMetrics = DEFAULT_METRICS;
  private provenance: TrainedSuiteProvenance = DEFAULT_TRAINED_PROVENANCE;

  constructor(container: HTMLElement, initialLocale: ObservatoryLocale = DEFAULT_LOCALE) {
    this.root = container;
    this.locale = initialLocale;
    this.render();
  }

  setLocale(locale: ObservatoryLocale): void {
    if (this.locale === locale) return;
    this.locale = locale;
    this.render();
  }

  selectNode(nodeId: ArchNodeId): void {
    if (!ARCH_NODE_IDS.includes(nodeId)) return;
    this.selectedNodeId = nodeId;
    this.render();
  }

  getSelectedNode(): ArchNodeId {
    return this.selectedNodeId;
  }

  updateMetrics(partial: Partial<ArchLiveMetrics>): void {
    this.metrics = { ...this.metrics, ...partial };
    this.render();
  }

  setTrainedProvenance(partial: Partial<TrainedSuiteProvenance>): void {
    this.provenance = { ...this.provenance, ...partial };
    this.render();
  }

  render(): void {
    const strings = getObservatoryStrings(this.locale);
    const selected = strings.archNodes[this.selectedNodeId];
    const liveBadge = formatNodeLiveBadge(this.selectedNodeId, this.metrics, selected.badge);
    const svg = renderArchMapSvgMarkup(this.locale, this.selectedNodeId, this.metrics);
    const provenanceHtml = renderModelProvenanceStripHtml(this.locale, this.provenance);

    const pathItems = selected.repoPaths
      .map((p) => `<code class="arch-repo-path">${escapeHtml(p)}</code>`)
      .join(" ");

    this.root.innerHTML =
      `<div class="arch-map-shell">` +
      `<div class="arch-map-header">` +
      `<h2>${escapeHtml(strings.archView.heading)}</h2>` +
      `<p>${escapeHtml(strings.archView.subheading)}</p>` +
      `</div>` +
      `<div class="arch-svg-wrap">${svg}</div>` +
      `<div class="arch-inspector-card" data-selected-node="${this.selectedNodeId}">` +
      `<div class="arch-inspector-head">` +
      `<strong>${escapeHtml(selected.title)}</strong>` +
      `<span class="arch-inspector-sub">${escapeHtml(selected.subtitle)}</span>` +
      `<span class="arch-inspector-badge">${escapeHtml(liveBadge)}</span>` +
      `</div>` +
      `<div class="arch-inspector-grid">` +
      `<div class="arch-inspector-col">` +
      `<div class="arch-inspector-label">${escapeHtml(strings.archView.whatItDoesLabel)}</div>` +
      `<p>${escapeHtml(selected.whatItDoes)}</p>` +
      `</div>` +
      `<div class="arch-inspector-col">` +
      `<div class="arch-inspector-label">${escapeHtml(strings.archView.ifMissingLabel)}</div>` +
      `<p>${escapeHtml(selected.ifMissing)}</p>` +
      `</div>` +
      `</div>` +
      `<div class="arch-inspector-paths">` +
      `<span class="arch-inspector-label">${escapeHtml(strings.archView.repoPathsLabel)}:</span> ${pathItems}` +
      `</div>` +
      `</div>` +
      provenanceHtml +
      `</div>`;

    const nodeEls = this.root.querySelectorAll<SVGGElement>("[data-arch-node]");
    nodeEls.forEach((el) => {
      const id = el.getAttribute("data-arch-node") as ArchNodeId | null;
      if (!id) return;
      el.addEventListener("click", () => this.selectNode(id));
      el.addEventListener("keydown", (ev: KeyboardEvent) => {
        if (ev.key === "Enter" || ev.key === " ") {
          ev.preventDefault();
          this.selectNode(id);
        }
      });
    });
  }
}
