/**
 * Bilingual (`TR / EN`) Dictionary for OpenXVS v3.0 Observatory (default `tr`, < 3 KiB Brotli).
 */

export type ObservatoryLocale = "tr" | "en";
export type ObservatoryTabId = "scene" | "arch" | "realsky";
export type StressPresetId = "cbdr" | "forcedland" | "spoof" | "normal";
export type ManifoldId = "activeRangeCone" | "sc3bfTube" | "forcedLandFunnel" | "spzStress";

export const ARCH_NODE_IDS = [
  "sensors",
  "xvs_d2",
  "xvs_activerange",
  "sc3bf_kani",
  "xvs_forcedland",
  "xvs_os_hal",
  "xvs_fleet",
  "webgpu_observatory",
] as const;

export type ArchNodeId = (typeof ARCH_NODE_IDS)[number];

export interface ArchNodeStrings {
  readonly title: string;
  readonly subtitle: string;
  readonly badge: string;
  readonly whatItDoes: string;
  readonly ifMissing: string;
  readonly repoPaths: readonly string[];
}

export interface StressPresetStrings {
  readonly label: string;
  readonly hint: string;
}

export interface ObservatoryStrings {
  readonly locale: ObservatoryLocale;
  readonly appTitle: string;
  readonly localeToggleLabel: string;
  readonly tabs: Readonly<Record<ObservatoryTabId, string>>;
  readonly statusCards: {
    readonly rangeUncertainty: string;
    readonly sc3bfSafety: string;
    readonly kaniProof: string;
    readonly engineGlide: string;
    readonly techDrawerToggle: string;
  };
  readonly stressBarTitle: string;
  readonly stressPresets: Readonly<Record<StressPresetId, StressPresetStrings>>;
  readonly manifoldsTitle: string;
  readonly manifolds: Readonly<Record<ManifoldId, string>>;
  readonly archView: {
    readonly heading: string;
    readonly subheading: string;
    readonly whatItDoesLabel: string;
    readonly ifMissingLabel: string;
    readonly repoPathsLabel: string;
    readonly liveTelemetryLabel: string;
    readonly clickPrompt: string;
  };
  readonly archNodes: Readonly<Record<ArchNodeId, ArchNodeStrings>>;
  readonly realskyLab: {
    readonly heading: string;
    readonly subheading: string;
    readonly clipLabel: string;
    readonly playLabel: string;
    readonly pauseLabel: string;
    readonly stepLabel: string;
    readonly dropPrompt: string;
    readonly dropSubtext: string;
    readonly corruptFileError: string;
    readonly chartTitle: string;
    readonly legendGpsTruth: string;
    readonly legendCpMscSafe: string;
    readonly legendNaiveBbox: string;
    readonly loadingPack: string;
    readonly nativeScaleNote: string;
  };
}

export const DEFAULT_LOCALE: ObservatoryLocale = "tr";

const LD = "apps/livedeck/";
const P = LD + "src/passes/";
const AR = "xvs-active-range/src/";
const NODE_META: Readonly<Record<ArchNodeId, readonly [string, string, readonly string[]]>> = {
  sensors: ["6-Cam DAS + LWIR", "360°", [P + "pass1_das_efvs.ts", P + "pass4_ai_roi.ts"]],
  xvs_d2: ["257K ONNX ConvGRU", "552KB FP16", [LD + "public/models/xvs_d2_web_fp16.onnx", LD + "src/ai/ort_worker.ts"]],
  xvs_activerange: ["6-Bank CP-MSC EKF", "Mono", [AR + "msc_rpekf.rs", "src/xvs/v2/xvs_active_range/filter.py"]],
  sc3bf_kani: ["SC3BF + Kani CBMC", "18/18 Kani", [AR + "sc3bf_fim.rs", AR + "kani_proofs.rs"]],
  xvs_forcedland: ["35kt Trochoid", "Glide", [LD + "src/active_range/scenario_forcedland.ts", P + "pass3_airframe_traffic.ts"]],
  xvs_os_hal: ["Simplex RTA #50004", "MAVLink 48B", ["xvs-rta-core/src/lib.rs", "xvs-os-hal/src/lib.rs"]],
  xvs_fleet: ["3-Ship MeshDAA", "Anti-Spoof", [P + "pass3_airframe_traffic.ts", "scripts/xvs/xvs_certgen.py"]],
  webgpu_observatory: ["5-Pass WebGPU/GL2", "60FPS 0.98m", [P + "pass2_terrain_cdlod.ts", P + "index.ts"]],
};

const n = (id: ArchNodeId, title: string, whatItDoes: string, ifMissing: string): ArchNodeStrings => {
  const [subtitle, badge, repoPaths] = NODE_META[id];
  return { title, subtitle, badge, whatItDoes, ifMissing, repoPaths };
};

const TR_STRINGS: ObservatoryStrings = {
  locale: "tr",
  appTitle: "OpenXVS v3.0 Gözlemevi",
  localeToggleLabel: "EN",
  tabs: { scene: "3D Uçuş & Güvenlik Sahnesi", arch: "Kim Ne Yapıyor?", realsky: "Gerçek Gökyüzü AI Laboratuvarı" },
  statusCards: {
    rangeUncertainty: "Mesafe Belirsizliği",
    sc3bfSafety: "SC3BF 6/6 Güvenlik",
    kaniProof: "Kani Kanıt: 18/18 YEŞİL",
    engineGlide: "Motor Arızası Süzülme",
    techDrawerToggle: "Teknik Telemetri",
  },
  stressBarTitle: "Red-Team Stres Senaryoları",
  stressPresets: {
    cbdr: { label: "CBDR Kör Çarpışma (ActiveRange)", hint: "<=12° S-manevrası ile mesafe kilidi" },
    forcedland: { label: "Motor Arızası + 35 kt Yan Rüzgar", hint: "3B trokoid süzülme hunisi (KSFO 28L)" },
    spoof: { label: "GPS Spoof + Sensör Arızası", hint: "Kamera izolasyonu + Simplex RTA" },
    normal: { label: "Normal KSFO 28L Yaklaşma (XVS-D2)", hint: "0.98m orto + PAPI + XVS-D2" },
  },
  manifoldsTitle: "3B Güvenlik Kalkanları",
  manifolds: {
    activeRangeCone: "ActiveRange Konisi",
    sc3bfTube: "SC3BF Bariyeri",
    forcedLandFunnel: "ForcedLand Hunisi",
    spzStress: "3DGS Stress (SPZ)",
  },
  archView: {
    heading: "Canlı Sistem Mimarisi — Kim Ne Yapıyor?",
    subheading: "Bir katmana tıklayın: rolünü ve kaynak dosyalarını görün.",
    whatItDoesLabel: "Ne yapar?",
    ifMissingLabel: "Olmazsa ne bozulur?",
    repoPathsLabel: "Kaynaklar",
    liveTelemetryLabel: "Canlı Akış",
    clickPrompt: "Katman seçin.",
  },
  archNodes: {
    sensors: n("sensors", "Kamera / Sensör", "6 kamera, arka balık gözü ve IMU verisini eşzamanlı toplar.", "Uçak ufuk ve trafiğe karşı kör kalır."),
    xvs_d2: n("xvs_d2", "XVS-D2", "6-kanal fark ve ConvGRU ile 4-10 px uzak uçakları yakalar.", "Radarsız uzak tehditler fark edilemez."),
    xvs_activerange: n("xvs_activerange", "XVS-ActiveRange", "<=12° S-manevrası ve konformal alt sınırla mesafe kilitler.", "CBDR rotasında tek kamera mesafe ölçemez."),
    sc3bf_kani: n("sc3bf_kani", "SC3BF-FIM & Kani", "6 hipotezde çarpışma bariyerini Rust Kani ile ispatlar.", "Manevra uçağı tehdide yaklaştırabilir."),
    xvs_forcedland: n("xvs_forcedland", "XVS-ForcedLand", "35 kt yan rüzgarda KSFO 28L süzülme hunisini çizer.", "Yan rüzgar uçağı pistten savurur."),
    xvs_os_hal: n("xvs_os_hal", "XVS-OS-HAL #50004", "Arızalı kamerayı izole eder, CRC32 #50004 sağlık paketi yollar.", "Bozuk sensör otopilot ve HUD'u zehirler."),
    xvs_fleet: n("xvs_fleet", "XVS-Fleet / MeshDAA", "3-uçak görüş hatlarıyla sahte ADS-B hayaletlerini reddeder.", "Sahte hedefler yanlış kaçınma tetikler."),
    webgpu_observatory: n("webgpu_observatory", "WebGPU Gözlemevi", "0.98m KSFO arazisi, Rayleigh gök ve 3B kalkanları çizer.", "Güvenlik sınırları canlı doğrulanamaz."),
  },
  realskyLab: {
    heading: "Gerçek Gökyüzü AI Laboratuvarı (1:1)",
    subheading: "Gerçek Amazon AOT kayıtlarında tarayıcı içi canlı XVS-D2 ($0).",
    clipLabel: "AOT Kaydı",
    playLabel: "Oynat",
    pauseLabel: "Duraklat",
    stepLabel: "1 Kare",
    dropPrompt: "Gökyüzü videosu/fotoğrafı sürükleyin (.mp4/.webm/.jpg/.png)",
    dropSubtext: "%100 yerel analiz — buluta yüklenmez.",
    corruptFileError: "Dosya çözülemedi veya boş: geçerli medya seçin.",
    chartTitle: "GPS vs CP-MSC Alt Sınır vs Saf Kutu",
    legendGpsTruth: "GPS Gerçek",
    legendCpMscSafe: "CP-MSC Alt Sınır",
    legendNaiveBbox: "Saf Kutu-Boyutu",
    loadingPack: "AOT paketi yükleniyor...",
    nativeScaleNote: "1:1 doğal 256×256 kırpım",
  },
};

const EN_STRINGS: ObservatoryStrings = {
  locale: "en",
  appTitle: "OpenXVS v3.0 Observatory",
  localeToggleLabel: "TR",
  tabs: { scene: "3D Flight & Safety Scene", arch: "Live System Architecture", realsky: "Real-Sky AI Lab" },
  statusCards: {
    rangeUncertainty: "Range Uncertainty",
    sc3bfSafety: "SC3BF 6/6 Safety",
    kaniProof: "Kani Proof: 18/18 GREEN",
    engineGlide: "Engine-Out Glide",
    techDrawerToggle: "Technical Telemetry",
  },
  stressBarTitle: "Adversarial Red-Team Presets",
  stressPresets: {
    cbdr: { label: "CBDR Blind Collision (ActiveRange)", hint: "Monocular range lock via <=12° S-swerve" },
    forcedland: { label: "Engine-Out + 35 kt Crosswind", hint: "3D trochoid glide funnel (KSFO 28L)" },
    spoof: { label: "GPS Spoof + Sensor Fault", hint: "Camera isolation + Simplex RTA" },
    normal: { label: "Normal KSFO 28L Approach (XVS-D2)", hint: "0.98m ortho + PAPI + XVS-D2" },
  },
  manifoldsTitle: "3D Safety Envelopes",
  manifolds: {
    activeRangeCone: "ActiveRange Cone",
    sc3bfTube: "SC3BF Barrier",
    forcedLandFunnel: "ForcedLand Funnel",
    spzStress: "3DGS Stress (SPZ)",
  },
  archView: {
    heading: "Live System Architecture — Who Does What?",
    subheading: "Click any layer to inspect its safety role and repo files.",
    whatItDoesLabel: "What does it do?",
    ifMissingLabel: "What breaks without it?",
    repoPathsLabel: "Repo Files",
    liveTelemetryLabel: "Live Flow",
    clickPrompt: "Select a layer.",
  },
  archNodes: {
    sensors: n("sensors", "Cameras / Sensors", "Captures 6-camera DAS, rear fisheye, LWIR, and IMU streams.", "Aircraft flies blind to horizon and traffic."),
    xvs_d2: n("xvs_d2", "XVS-D2", "Detects 4-10 px airborne intruders via 6-ch diffs and ConvGRU.", "Non-cooperative distant threats go unseen."),
    xvs_activerange: n("xvs_activerange", "XVS-ActiveRange", "Locks unknown-size range via <=12° S-swerve + conformal bound.", "Single camera cannot observe CBDR depth."),
    sc3bf_kani: n("sc3bf_kani", "SC3BF-FIM & Kani", "Proves 6-bank collision-cone invariance via Rust Kani CBMC.", "Info turn could steer toward the threat."),
    xvs_forcedland: n("xvs_forcedland", "XVS-ForcedLand", "Steers 35 kt crosswind-warped 3D trochoid glide to KSFO 28L.", "Crosswind pushes unpowered glide off runway."),
    xvs_os_hal: n("xvs_os_hal", "XVS-OS-HAL #50004", "Isolates faulted sensors under Simplex RTA + CRC32 #50004.", "Corrupt camera or spoofed GPS poisons HUD."),
    xvs_fleet: n("xvs_fleet", "XVS-Fleet / MeshDAA", "Cross-checks 3-ship rays to reject spoofed ADS-B ghosts.", "Spoofed ghosts trigger false evasive turns."),
    webgpu_observatory: n("webgpu_observatory", "WebGPU Observatory", "Renders 0.98m KSFO ortho, Rayleigh sky & 3D safety shields.", "Operators lose live visual verification."),
  },
  realskyLab: {
    heading: "Real-Sky AI Lab (1:1 Scale)",
    subheading: "Live in-browser XVS-D2 ONNX on real Amazon AOT encounters ($0).",
    clipLabel: "AOT Clip",
    playLabel: "Play",
    pauseLabel: "Pause",
    stepLabel: "Step +1",
    dropPrompt: "Drop sky video or image (.mp4/.webm/.jpg/.png)",
    dropSubtext: "100% local in-browser — zero upload.",
    corruptFileError: "Cannot decode or empty file: pick valid media.",
    chartTitle: "GPS Truth vs CP-MSC Safe Bound vs Naive Bbox",
    legendGpsTruth: "GPS Truth",
    legendCpMscSafe: "CP-MSC Safe Bound",
    legendNaiveBbox: "Naive Bbox-Size",
    loadingPack: "Loading 1:1 AOT pack...",
    nativeScaleNote: "1:1 native 256×256 crops",
  },
};

export function getObservatoryStrings(locale: ObservatoryLocale = DEFAULT_LOCALE): ObservatoryStrings {
  return locale === "en" ? EN_STRINGS : TR_STRINGS;
}
