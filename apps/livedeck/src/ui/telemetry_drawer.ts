/**
 * Collapsible OpenXVS Live Deck Telemetry Drawer.
 *
 * Displays live integrity/navigation/DAA metrics emitted by `xvs-wasm` & `ort_worker.ts`:
 *   - 8-arm Cholesky Page's CUSUM vs `h_c = 10.530`
 *   - 6-camera (`F, R, B, L, U, D`) epipolar residual bars vs `6,109 urad` (`0.35 deg`)
 *   - Vision-RAIM `V-HPL` / `V-VPL` vs `10 m` Cat I VAL & `40 m` HAL
 *   - `DtACI` OOD write-gate & 3D `ECBF-QP` infeasibility / escape status
 *   - Live `D1-KF` browser ONNX inference timing, peak count & domain-shift notice
 *   - Frame-time profiler (FPS / frame ms) & one-click Mission Proof JSON export
 */

export const CUSUM_THRESHOLD_HC = 10.53;
export const EPIPOLAR_CAM_LIMIT_URAD = 6109;
export const CAT1_VAL_MM = 10_000;
export const CAT1_HAL_MM = 40_000;
export const CAMERA_LABELS = ["F", "R", "B", "L", "U", "D"] as const;

export interface XvsD2ProvenanceInfo {
  readonly status: string;
  readonly bytes: number;
  readonly ap50?: number;
  readonly aboveHorizonPd?: number;
  readonly centroidRmsePx?: number;
}

export interface TelemetryDrawerSnapshot {
  readonly simTimeS: number;
  readonly frameMs: number;
  readonly fps: number;
  readonly cusumStat: number;
  readonly spoofAlarm: boolean;
  readonly bankActive: 0 | 1;
  readonly epipolarUrad: readonly [number, number, number, number, number, number];
  readonly camHealthMask: number;
  readonly horizonPlUrad: number;
  readonly horizonAlUrad: number;
  readonly vVplMm: number;
  readonly vHplMm: number;
  readonly mandatoryGoAround: boolean;
  readonly ghostRejected: boolean;
  readonly ecbfInfeasible: boolean;
  readonly dtaciFrozen: boolean;
  readonly aiInferenceMs: number;
  readonly aiPeaksCount: number;
  readonly aiMaxScore: number;
  readonly aiAllFinite: boolean;
  readonly aiModelId: string;
  readonly missionId?: "M1" | "M2" | "M3" | "M4";
  readonly wc1Alert?: boolean;
  readonly camRBiasDeg?: number;
  readonly efvsEnabled?: boolean;
  readonly simplexMode?: 0 | 1 | 2 | 3 | 4;
  readonly activeDetectorUiMode?: "d0" | "d1_kf" | "xvs_d2";
  readonly aiDetectionsCount?: number;
  readonly xvsD2Provenance?: XvsD2ProvenanceInfo | null;
}

export class TelemetryDrawer {
  private readonly root: HTMLElement;
  private readonly bodyEl: HTMLElement;
  private readonly toggleBtn: HTMLButtonElement;
  private readonly cusumFill: HTMLElement;
  private readonly cusumVal: HTMLElement;
  private readonly camFills: HTMLElement[] = [];
  private readonly camVals: HTMLElement[] = [];
  private readonly vplFill: HTMLElement;
  private readonly vplVal: HTMLElement;
  private readonly hplVal: HTMLElement;
  private readonly flagsVal: HTMLElement;
  private readonly aiVal: HTMLElement;
  private readonly aiProvenanceBadge: HTMLElement;
  private readonly perfVal: HTMLElement;
  private collapsed = false;

  constructor(container: HTMLElement, onDownloadProof: () => void) {
    this.root = document.createElement("aside");
    this.root.className = "xvs-drawer";
    this.root.setAttribute("data-testid", "telemetry-drawer");

    const header = document.createElement("div");
    header.className = "xvs-drawer-header";

    const title = document.createElement("span");
    title.className = "xvs-drawer-title";
    title.textContent = "INTEGRITY & DAA TELEMETRY";

    this.toggleBtn = document.createElement("button");
    this.toggleBtn.type = "button";
    this.toggleBtn.className = "xvs-btn xvs-btn-sm";
    this.toggleBtn.textContent = "HIDE";
    this.toggleBtn.addEventListener("click", () => this.setCollapsed(!this.collapsed));

    header.append(title, this.toggleBtn);
    this.root.appendChild(header);

    this.bodyEl = document.createElement("div");
    this.bodyEl.className = "xvs-drawer-body";

    // 1. Frame profiler
    this.perfVal = this._addMetricRow("PROFILER", "0.0 FPS | 0.0 ms | t=0.0s");

    // 2. CUSUM bar vs h_c = 10.530
    const cusumSec = this._createSection("GNSS SPOOFING (8-ARM CUSUM vs h_c=10.530)");
    const { fill: cFill, label: cVal } = this._createBarRow("CUSUM", "0.00 / 10.53");
    this.cusumFill = cFill;
    this.cusumVal = cVal;
    cusumSec.append(cFill.parentElement!.parentElement!);
    this.bodyEl.appendChild(cusumSec);

    // 3. 6-cam epipolar residual bars
    const camSec = this._createSection("6-CAM EPIPOLAR RESIDUALS (LIMIT 6,109 μrad)");
    for (const cid of CAMERA_LABELS) {
      const { fill, label } = this._createBarRow(`CAM ${cid}`, "0 μrad");
      this.camFills.push(fill);
      this.camVals.push(label);
      camSec.appendChild(fill.parentElement!.parentElement!);
    }
    this.bodyEl.appendChild(camSec);

    // 4. Vision-RAIM V-VPL / V-HPL vs Cat I VAL
    const raimSec = this._createSection("VISION-RAIM (CAT I VAL 10.0 m)");
    const { fill: vFill, label: vVal } = this._createBarRow("V-VPL", "0.00 m / 10.0 m");
    this.vplFill = vFill;
    this.vplVal = vVal;
    raimSec.appendChild(vFill.parentElement!.parentElement!);
    this.hplVal = this._addMetricRowTo(raimSec, "V-HPL / HOR-PL", "0.00 m | 0 μrad");
    this.bodyEl.appendChild(raimSec);

    // 5. DAA / DtACI / Safety Flags
    this.flagsVal = this._addMetricRow(
      "SAFETY / DAA",
      "BANK 0 | GHOST: -- | QP: OK | DtACI: LIVE",
    );

    // 6. Live Stage-1 ONNX inference (XVS-D2 / D1-KF) + provenance badge + explicit domain-shift notice
    const aiSec = this._createSection("STAGE-1 AI DETECTOR (XVS-D2 / D1-KF)");
    this.aiProvenanceBadge = document.createElement("span");
    this.aiProvenanceBadge.className = "xvs-badge";
    this.aiProvenanceBadge.setAttribute("data-testid", "xvs-d2-provenance-badge");
    this.aiProvenanceBadge.dataset.severity = "nominal";
    this.aiProvenanceBadge.style.fontSize = "9.5px";
    this.aiProvenanceBadge.style.padding = "2px 6px";
    this.aiProvenanceBadge.style.alignSelf = "flex-start";
    this.aiProvenanceBadge.style.display = "none";
    this.aiProvenanceBadge.textContent = "XVS-D2 WEB FP16";
    aiSec.appendChild(this.aiProvenanceBadge);
    this.aiVal = this._addMetricRowTo(aiSec, "INFERENCE", "pending Stage-1 hydration...");
    const notice = document.createElement("div");
    notice.className = "xvs-domain-notice";
    notice.textContent =
      "Domain-shifted live browser demo — real AOT metrics measured on held-out Amazon AOT flights.";
    aiSec.appendChild(notice);
    this.bodyEl.appendChild(aiSec);

    // 7. Download Mission Proof JSON button
    const dlBtn = document.createElement("button");
    dlBtn.type = "button";
    dlBtn.className = "xvs-btn xvs-btn-primary xvs-btn-full";
    dlBtn.setAttribute("data-testid", "download-proof-btn");
    dlBtn.textContent = "DOWNLOAD MISSION PROOF JSON";
    dlBtn.addEventListener("click", onDownloadProof);
    this.bodyEl.appendChild(dlBtn);

    this.root.appendChild(this.bodyEl);
    container.appendChild(this.root);
  }

  private _createSection(titleText: string): HTMLElement {
    const sec = document.createElement("div");
    sec.className = "xvs-drawer-section";
    const h = document.createElement("div");
    h.className = "xvs-section-label";
    h.textContent = titleText;
    sec.appendChild(h);
    return sec;
  }

  private _addMetricRow(keyText: string, initVal: string): HTMLElement {
    const sec = this._createSection(keyText);
    const val = document.createElement("div");
    val.className = "xvs-metric-value";
    val.textContent = initVal;
    sec.appendChild(val);
    this.bodyEl.appendChild(sec);
    return val;
  }

  private _addMetricRowTo(parent: HTMLElement, keyText: string, initVal: string): HTMLElement {
    const row = document.createElement("div");
    row.className = "xvs-kv-row";
    const k = document.createElement("span");
    k.className = "xvs-kv-key";
    k.textContent = keyText;
    const v = document.createElement("span");
    v.className = "xvs-kv-val";
    v.textContent = initVal;
    row.append(k, v);
    parent.appendChild(row);
    return v;
  }

  private _createBarRow(
    keyText: string,
    initVal: string,
  ): { fill: HTMLElement; label: HTMLElement } {
    const row = document.createElement("div");
    row.className = "xvs-bar-row";

    const top = document.createElement("div");
    top.className = "xvs-bar-top";
    const k = document.createElement("span");
    k.textContent = keyText;
    const label = document.createElement("span");
    label.textContent = initVal;
    top.append(k, label);

    const track = document.createElement("div");
    track.className = "xvs-bar-track";
    const fill = document.createElement("div");
    fill.className = "xvs-bar-fill";
    track.appendChild(fill);

    row.append(top, track);
    return { fill, label };
  }

  setCollapsed(next: boolean): void {
    this.collapsed = next;
    this.bodyEl.style.display = next ? "none" : "flex";
    this.toggleBtn.textContent = next ? "SHOW" : "HIDE";
  }

  update(snap: TelemetryDrawerSnapshot): void {
    this.perfVal.textContent = `${snap.fps.toFixed(1)} FPS | ${snap.frameMs.toFixed(2)} ms | t=${snap.simTimeS.toFixed(1)}s`;

    // CUSUM bar
    const cusumPct = Math.min(100, Math.max(0, (snap.cusumStat / CUSUM_THRESHOLD_HC) * 100));
    this.cusumFill.style.width = `${cusumPct.toFixed(1)}%`;
    this.cusumFill.dataset.severity = snap.spoofAlarm
      ? "alert"
      : cusumPct > 65
        ? "caution"
        : "nominal";
    this.cusumVal.textContent = `${snap.cusumStat.toFixed(2)} / ${CUSUM_THRESHOLD_HC.toFixed(2)}${snap.spoofAlarm ? " [SPOOF]" : ""}`;

    // 6-cam epipolar bars
    for (let i = 0; i < 6; i++) {
      const urad = snap.epipolarUrad[i];
      const healthy = Boolean((snap.camHealthMask >> i) & 1);
      const pct = Math.min(100, Math.max(0, (urad / EPIPOLAR_CAM_LIMIT_URAD) * 100));
      this.camFills[i].style.width = `${pct.toFixed(1)}%`;
      this.camFills[i].dataset.severity = !healthy
        ? "alert"
        : urad > EPIPOLAR_CAM_LIMIT_URAD * 0.7
          ? "caution"
          : "nominal";
      this.camVals[i].textContent = `${Math.round(urad)} μrad${healthy ? "" : " [EXCL]"}`;
    }

    // V-VPL / V-HPL
    const vplPct = Math.min(100, Math.max(0, (snap.vVplMm / CAT1_VAL_MM) * 100));
    this.vplFill.style.width = `${vplPct.toFixed(1)}%`;
    this.vplFill.dataset.severity =
      snap.mandatoryGoAround || snap.vVplMm > CAT1_VAL_MM
        ? "alert"
        : vplPct > 75
          ? "caution"
          : "nominal";
    this.vplVal.textContent = `${(snap.vVplMm / 1000).toFixed(2)} m / 10.0 m${snap.mandatoryGoAround ? " [GA]" : ""}`;
    this.hplVal.textContent = `${(snap.vHplMm / 1000).toFixed(2)} m | PL ${Math.round(snap.horizonPlUrad)}/${Math.round(snap.horizonAlUrad)} μrad`;

    // Safety / DAA flags
    const bankStr = snap.bankActive === 1 ? "BANK 1 (VIS)" : "BANK 0 (GNSS)";
    const ghostStr = snap.ghostRejected ? "GHOST REJECTED" : "NOMINAL";
    const qpStr = snap.ecbfInfeasible ? "ESCAPE" : "OK";
    const dtStr = snap.dtaciFrozen ? "FROZEN" : "LIVE";
    this.flagsVal.textContent = `${bankStr} | ${ghostStr} | QP: ${qpStr} | DtACI: ${dtStr}`;

    // AI inference + dynamic Colab G4 provenance badge from registry.json
    const finStr = snap.aiAllFinite ? "FINITE" : "NON-FINITE!";
    const modeTag =
      snap.activeDetectorUiMode === "xvs_d2"
        ? "XVS-D2"
        : snap.activeDetectorUiMode === "d0"
          ? "D0-SAFE"
          : "D1-KF";
    const boxCount = snap.aiDetectionsCount ?? snap.aiPeaksCount;
    this.aiVal.textContent = `[${modeTag}] ${snap.aiModelId} | ${snap.aiInferenceMs.toFixed(1)} ms | det=${boxCount} | max=${snap.aiMaxScore.toFixed(3)} (${finStr})`;

    if (snap.activeDetectorUiMode === "xvs_d2") {
      this.aiProvenanceBadge.style.display = "inline-flex";
      const prov = snap.xvsD2Provenance;
      if (
        prov &&
        prov.status === "trained_colab_g4_blackwell" &&
        typeof prov.ap50 === "number" &&
        Number.isFinite(prov.ap50) &&
        typeof prov.aboveHorizonPd === "number" &&
        Number.isFinite(prov.aboveHorizonPd) &&
        Number.isFinite(prov.bytes) &&
        prov.bytes > 0
      ) {
        this.aiProvenanceBadge.dataset.severity = "nominal";
        const kb = Math.floor(prov.bytes / 1000);
        this.aiProvenanceBadge.textContent =
          `COLAB G4 AOT · AP50 ${prov.ap50.toFixed(3)} · Pd ${(prov.aboveHorizonPd * 100).toFixed(1)}% · ${kb} KB FP16`;
      } else {
        this.aiProvenanceBadge.dataset.severity = "info";
        this.aiProvenanceBadge.textContent = "XVS-D2 WEB FP16";
      }
    } else {
      this.aiProvenanceBadge.style.display = "none";
    }
  }
}
