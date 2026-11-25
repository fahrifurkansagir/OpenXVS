/**
 * Bilingual (`TR / EN`) UI Strings for OpenXVS Studio v5.2.
 * Strictly civil & dual-use aviation terminology.
 */

import type { VisualWorldMode } from "../scene/atmosphere_world.ts";
import type { StudioScenarioId } from "../sim/studio_sim.ts";

export type StudioLocale = "tr" | "en";
export type StudioCameraPreset = "chase" | "wingman" | "overview" | "cockpit";
export type TerrainStyleMode = VisualWorldMode;

export const STUDIO_VERSION_LABEL = "v5.2" as const;
export const STUDIO_APP_TITLE = "OpenXVS Studio v5.2" as const;
export const STUDIO_STACK_V5_SPEC = "6/6 Pure-Rust ORT + XFeat + LighterGlue + MultiShipCoord" as const;
export const STUDIO_NAV_V5_SPEC =
  "XVS-Nav v5.0: MNV4-Hybrid-M ONNX · 13.1ms P50 · 25.3MB S1+XFeat+LighterGlue FP16 · 0-shot KSFO R@1 91.4% / R@5 96.7% · HPL <20m · 8-Arm CUSUM" as const;
export const STUDIO_ASSURANCE_V52_SPEC =
  "26/26 Kani CBMC (5,228 checks) · Apalache N=2/N=3 · Cortex-M7 thumbv7em (188 KB Flash / 0 B RAM) · Sequence-VPR (0 HMI) · M-of-N TBC" as const;

export interface StudioStrings {
  readonly brandTitle: string;
  readonly brandVersion: string;
  readonly brandBadge: string;
  readonly brandBadgeTitle: string;
  readonly navV5Spec: string;
  readonly stackV5Spec: string;
  readonly localeToggle: string;
  readonly modePhotoreal: string;
  readonly modeObsidian: string;
  readonly controlAutopilot: string;
  readonly controlManual: string;
  readonly shieldOn: string;
  readonly shieldOff: string;
  readonly weatherBtn: string;
  readonly videoLabBtn: string;
  readonly sensorsToggle: string;
  readonly domeToggle: string;
  readonly manualHint: string;
  readonly cameras: Record<StudioCameraPreset, string>;
  readonly scenarios: Record<StudioScenarioId, { readonly label: string; readonly banner: string }>;
  readonly weather: {
    readonly title: string;
    readonly timeOfDay: string;
    readonly visibility: string;
    readonly crosswind: string;
    readonly cloudCover: string;
    readonly rain: string;
    readonly engineOutBtn: string;
    readonly resetBtn: string;
  };
  readonly telemetry: {
    readonly range: string;
    readonly sc3bf: string;
    readonly bank: string;
    readonly speed: string;
    readonly locked: string;
    readonly swerve: string;
    readonly safe: string;
    readonly evade: string;
    readonly shieldOverridePrefix: string;
    readonly breachPrefix: string;
    readonly touchdownPrefix: string;
  };
  readonly hudStatus: {
    readonly sensorOk: string;
    readonly sensorStandby: string;
    readonly sensorNucFreeze: string;
    readonly sensorLwirFfc: string;
    readonly sensorLwirContrast: string;
    readonly sensorSectorDegraded: string;
    readonly sensorSkew: string;
    readonly sensorDegraded: string;
    readonly coordActive: string;
    readonly coordStandby: string;
  };
  readonly drawer: {
    readonly title: string;
    readonly bridgeOnline: string;
    readonly bridgeNative: string;
    readonly bridgeFallback: string;
    readonly avionicsTitle: string;
    readonly assuranceTitle: string;
    readonly uploadHint: string;
    readonly uploadBtn: string;
    readonly channels: {
      readonly d2: string;
      readonly land: string;
      readonly thermal: string;
      readonly nav: string;
    };
  };
  readonly bench: {
    readonly title: string;
    readonly subtitle: string;
    readonly tabVideo: string;
    readonly tabWebcam: string;
    readonly tabHil: string;
    readonly close: string;
    readonly dropVideoHint: string;
    readonly dropVideoNativeHint: string;
    readonly pickVideoBtn: string;
    readonly runVideoBtn: string;
    readonly runBuiltinBtn: string;
    readonly runBuiltinLandBtn: string;
    readonly runNativeSeqBtn: string;
    readonly runNativeLandSeqBtn: string;
    readonly stopSeqBtn: string;
    readonly playSeqBtn: string;
    readonly pauseSeqBtn: string;
    readonly nativeVideoNote: string;
    readonly arPassiveNote: string;
    readonly arD2OnlyNote: string;
    readonly sampleFpsLabel: string;
    readonly maxFramesLabel: string;
    readonly channelLabel: string;
    readonly processing: string;
    readonly processingNative: string;
    readonly downloadMp4: string;
    readonly downloadJson: string;
    readonly metricDetRate: string;
    readonly metricLatency: string;
    readonly metricArLocks: string;
    readonly metricSha: string;
    readonly startCamBtn: string;
    readonly stopCamBtn: string;
    readonly rawCamLabel: string;
    readonly aiCamLabel: string;
    readonly camDenied: string;
    readonly bridgeRequired: string;
    readonly hilTitle: string;
    readonly hilModeLabel: string;
    readonly hilApplyBtn: string;
    readonly hilLoopbackBtn: string;
    readonly hilRunning: string;
    readonly nativeHilLockNotice: string;
    readonly buyerPackTitle: string;
    readonly buyerPackBtn: string;
    readonly nativeEvalPackTitle: string;
    readonly nativeEvalPackBtn: string;
    readonly nativeEvalExporting: string;
    readonly nativeEvalExportedPrefix: string;
    readonly buyerPackChecklist: readonly string[];
  };
}

const STRINGS: Record<StudioLocale, StudioStrings> = {
  tr: {
    brandTitle: "OPENXVS STUDIO",
    brandVersion: STUDIO_VERSION_LABEL,
    brandBadge: "6/6 ORT · XFeat · LG · Coord",
    brandBadgeTitle: STUDIO_STACK_V5_SPEC,
    navV5Spec: STUDIO_NAV_V5_SPEC,
    stackV5Spec: STUDIO_STACK_V5_SPEC,
    localeToggle: "TR / EN",
    modePhotoreal: "Gerçek Dünya",
    modeObsidian: "Taktik Opsidyen",
    controlAutopilot: "Otopilot",
    controlManual: "Manuel (WASD)",
    shieldOn: "Kalkan AÇIK",
    shieldOff: "Kalkan KAPALI",
    weatherBtn: "Hava",
    videoLabBtn: "Video • HIL Lab",
    sensorsToggle: "4-Kanal AI",
    domeToggle: "6-Cam Kubbe",
    manualHint: "WASD/Oklar: Uç · Q/E: Dümen · Shift/Ctrl: Gaz · F: Motor Arızası · R: Sıfırla",
    cameras: {
      chase: "Taktik Takip",
      wingman: "Kanat Yakın",
      overview: "Kuşbakışı",
      cockpit: "Kokpit",
    },
    scenarios: {
      cbdr: {
        label: "CBDR Kör Çarpışma",
        banner: "XVS-ActiveRange · <=12° S-Manevra · MultiShipCoord",
      },
      forcedland: {
        label: "Motor Arızası + 35 kt",
        banner: "XVS-ForcedLand · 35 kt Trokoid (KSFO 28L)",
      },
      thermal_fog: {
        label: "Siste Termal İniş",
        banner: "XVS-ThermalNet · 8–14 µm LWIR · NUC Koruma",
      },
      gps_spoof: {
        label: "GPS Spoof Kalkanı",
        banner: "XVS-Nav v5.0 · KSFO R@1 91.4% / R@5 96.7% · 13.1ms · CUSUM",
      },
    },
    weather: {
      title: "Atmosfer & Hava Kontrolü",
      timeOfDay: "Günün Saati",
      visibility: "Görüş (km)",
      crosswind: "Yan Rüzgar (kt)",
      cloudCover: "Bulutluluk",
      rain: "Yağmur",
      engineOutBtn: "Sağ Motor Arızası (F)",
      resetBtn: "Yaklaşmayı Sıfırla (R)",
    },
    telemetry: {
      range: "MESAFE",
      sc3bf: "SC3BF",
      bank: "YATIŞ",
      speed: "HIZ",
      locked: "KİLİTLENDİ",
      swerve: "S-MANEVRA",
      safe: "6/6 GÜVENLİ",
      evade: "KAÇINMA",
      shieldOverridePrefix: "KALKAN MÜDAHALESİ",
      breachPrefix: "KALKAN KAPALI — ZARF İHLALİ",
      touchdownPrefix: "KSFO 28L İNİŞ",
    },
    hudStatus: {
      sensorOk: "SENS: EO/LWIR OK",
      sensorStandby: "SENS: SİMÜLASYON",
      sensorNucFreeze: "SENS: LWIR NUC DONDU",
      sensorLwirFfc: "SENS: LWIR FFC MASKE",
      sensorLwirContrast: "SENS: LWIR KONTRAST ÇÖKTÜ",
      sensorSectorDegraded: "SENS: EO SEKTÖR BOZUK",
      sensorSkew: "SENS: ZAMAN KAYMASI",
      sensorDegraded: "SENS: DÜŞÜK KALİTE",
      coordActive: "COORD: F1 91.113 · F4 GHOST",
      coordStandby: "COORD: F1 91.113 · F4 HAZIR",
    },
    drawer: {
      title: "4-Kanal AI Sensör",
      bridgeOnline: "6/6 AI (:8765)",
      bridgeNative: "RUST-ORT IPC",
      bridgeFallback: "Sentetik Mod",
      avionicsTitle: "v5.2 Aviyonik Sağlık & Doğrulama",
      assuranceTitle: "Formal & Bare-Metal Güvence",
      uploadHint: "Herhangi bir gökyüzü / pist görüntüsü sürükle veya yükle",
      uploadBtn: "Kendi Görüntünü Test Et",
      channels: {
        d2: "XVS-D2 · EO Gökyüzü",
        land: "XVS-Land · Pist 12-KPT",
        thermal: "XVS-Thermal · LWIR Sis",
        nav: "XVS-Nav v5.0 · MNV4",
      },
    },
    bench: {
      title: "OpenXVS Studio v5.2 — Video, Kamera & Doğrulama Lab",
      subtitle: `${STUDIO_STACK_V5_SPEC} · ${STUDIO_NAV_V5_SPEC}`,
      tabVideo: "Video / Sekans Analizörü",
      tabWebcam: "Canlı Webcam / USB Kamera",
      tabHil: "HIL Öz-Test & Eval Paketi",
      close: "Kapat",
      dropVideoHint: ".MP4, .WEBM veya .MOV uçuş / drone videosunu buraya bırakın veya seçin (maks. 150 MB)",
      dropVideoNativeHint: "Standalone .EXE: Dahili sekansı 1 tıkla Pure-Rust ORT ile oynatın veya yerel .MP4/.WEBM karelerini işleyin",
      pickVideoBtn: "Video Dosyası Seç",
      runVideoBtn: "Gerçek AI ile İşle",
      runBuiltinBtn: "1-Tık Gerçek Uçuş Klibi (AOT 1080p)",
      runBuiltinLandBtn: "1-Tık KSFO Yaklaşma Klibi",
      runNativeSeqBtn: "1-Tık Pure-Rust ORT Sekansı",
      runNativeLandSeqBtn: "1-Tık KSFO Yaklaşma Sekansı (Rust ORT)",
      stopSeqBtn: "Sekansı Durdur",
      playSeqBtn: "Oynat",
      pauseSeqBtn: "Duraklat",
      nativeVideoNote: "Standalone Pure-Rust ORT IPC: Python ve :8765 köprüsü olmadan gerçek ardışık uçuş kareleri yerel ONNX Runtime ile işlenir.",
      arPassiveNote: "Pasif düz uçuş · S-manevra yok",
      arD2OnlyNote: "Yalnızca D2 kanalı",
      sampleFpsLabel: "Örnekleme FPS",
      maxFramesLabel: "Maks. Kare",
      channelLabel: "Model Kanalı",
      processing: "Gerçek AI modeli kareleri işliyor...",
      processingNative: "Pure-Rust ORT IPC ardışık kareleri işliyor...",
      downloadMp4: "İşaretli .MP4 İndir",
      downloadJson: "JSON Rapor İndir",
      metricDetRate: "Tespit Oranı",
      metricLatency: "Ort. Gecikme",
      metricArLocks: "ActiveRange Kilidi",
      metricSha: "Checkpoint SHA-256",
      startCamBtn: "Kamerayı Başlat (~4 FPS Canlı AI)",
      stopCamBtn: "Kamerayı Durdur",
      rawCamLabel: "Ham Kamera Girişi",
      aiCamLabel: "OpenXVS Canlı AI Çıktısı",
      camDenied: "Kamera erişimi reddedildi veya bağlı kamera bulunamadı. Lütfen tarayıcı izinlerini kontrol edin.",
      bridgeRequired: "Yerel 5/5 AI sunucusu (http://127.0.0.1:8765) çevrimdışı. Tarayıcı dev modunda `python scripts/xvs/studio_live_server.py` çalıştırın.",
      hilTitle: "Sıfır-GPL MAVLink2 (ArduPlane/PX4) & FlightGear UDP Köprüsü",
      hilModeLabel: "Köprü Modu",
      hilApplyBtn: "UDP Köprüsünü Uygula",
      hilLoopbackBtn: "1-Tık Canlı Protokol Öz-Testi (Loopback Self-Test)",
      hilRunning: "127.0.0.1 UDP loopback sertifikasyon kontrolleri çalışıyor...",
      nativeHilLockNotice: "Standalone .EXE · NON_FLIGHT_SIM_ONLY=1 (EU PLD Art. 4(7)): dış UDP aktivasyonu kapalıdır; 1-Tık yerel öz-test aktiftir.",
      buyerPackTitle: "1-Tık Alıcı Doğrulama & Sertifikasyon Paketi (.ZIP)",
      buyerPackBtn: "Alıcı Doğrulama Paketini İndir (.ZIP)",
      nativeEvalPackTitle: "1-Tık Standalone Değerlendirme Raporu (.JSON + SHA-256)",
      nativeEvalPackBtn: "Standalone Eval Raporu İndir (.JSON)",
      nativeEvalExporting: "Eval raporu üretiliyor...",
      nativeEvalExportedPrefix: "İndirildi:",
      buyerPackChecklist: [
        "01_executive_summary.json — 6/6 Pure-Rust ORT + XFeat + LighterGlue FP16 SHA-256 + Dürüst G5_GAP Kapsamı",
        "02_nav_v5_benchmark.json — XVS-Nav v5.0 MNV4-Hybrid-M · 13.1ms P50 · 25.3MB S1+XFeat+LighterGlue FP16 · KSFO R@1 91.4% / R@5 96.7% · Sequence-VPR 0 HMI",
        "03_formal_assurance.json — 26/26 Kani CBMC (5,228 kontrol) + Apalache N=2/N=3 + Cortex-M7 thumbv7em (188 KB Flash / 0 B RAM)",
        "04_sensor_health_and_hil.json — 12-Sektör EO + LWIR FFC/Kontrast + FSYNC <=1.0ms + M-of-N TBC + 100/100 İmzalı #52001",
        "05_latest_video_analysis.json — Ardışık Gerçek Uçuş Karesi Analiz Raporu (Pure-Rust ORT / AOT)",
        "06_FIELD_TRIAL_PROTOCOL.md — Tier-0A ($0) → Tier-0B ($408) → Faz-4 Uçuş Rehberi",
      ],
    },
  },
  en: {
    brandTitle: "OPENXVS STUDIO",
    brandVersion: STUDIO_VERSION_LABEL,
    brandBadge: "6/6 ORT · XFeat · LG · Coord",
    brandBadgeTitle: STUDIO_STACK_V5_SPEC,
    navV5Spec: STUDIO_NAV_V5_SPEC,
    stackV5Spec: STUDIO_STACK_V5_SPEC,
    localeToggle: "EN / TR",
    modePhotoreal: "Photorealistic",
    modeObsidian: "Tactical Obsidian",
    controlAutopilot: "Autopilot",
    controlManual: "Manual (WASD)",
    shieldOn: "Shield ON",
    shieldOff: "Shield OFF",
    weatherBtn: "Weather",
    videoLabBtn: "Video • HIL Lab",
    sensorsToggle: "4-Ch AI",
    domeToggle: "6-Cam Dome",
    manualHint: "WASD/Arrows: Pitch & Roll · Q/E: Rudder · Shift/Ctrl: Throttle · F: Engine-Out · R: Reset",
    cameras: {
      chase: "Tactical Chase",
      wingman: "Wingman Close",
      overview: "KSFO Overview",
      cockpit: "Cockpit",
    },
    scenarios: {
      cbdr: {
        label: "CBDR Collision",
        banner: "XVS-ActiveRange · <=12° S-Swerve · MultiShipCoord",
      },
      forcedland: {
        label: "Engine-Out + 35 kt",
        banner: "XVS-ForcedLand · 35 kt Trochoid (KSFO 28L)",
      },
      thermal_fog: {
        label: "Thermal Fog Land",
        banner: "XVS-ThermalNet · 8–14 µm LWIR · NUC Guard",
      },
      gps_spoof: {
        label: "GPS Spoof Shield",
        banner: "XVS-Nav v5.0 · KSFO R@1 91.4% / R@5 96.7% · 13.1ms · CUSUM",
      },
    },
    weather: {
      title: "Atmosphere & Weather",
      timeOfDay: "Time of Day",
      visibility: "Visibility (km)",
      crosswind: "Crosswind (kt)",
      cloudCover: "Cloud Cover",
      rain: "Rain",
      engineOutBtn: "Right Engine-Out (F)",
      resetBtn: "Reset Approach (R)",
    },
    telemetry: {
      range: "RANGE",
      sc3bf: "SC3BF",
      bank: "BANK",
      speed: "TAS",
      locked: "LOCKED",
      swerve: "SWERVE",
      safe: "6/6 SAFE",
      evade: "EVADE",
      shieldOverridePrefix: "SHIELD OVERRIDE",
      breachPrefix: "SHIELD OFF — ENVELOPE BREACH",
      touchdownPrefix: "KSFO 28L TOUCHDOWN",
    },
    hudStatus: {
      sensorOk: "SENS: EO/LWIR OK",
      sensorStandby: "SENS: SIM STANDBY",
      sensorNucFreeze: "SENS: LWIR NUC FREEZE",
      sensorLwirFfc: "SENS: LWIR FFC MASK",
      sensorLwirContrast: "SENS: LWIR CONTRAST DROP",
      sensorSectorDegraded: "SENS: EO SECTOR DEG",
      sensorSkew: "SENS: SKEW GUARD",
      sensorDegraded: "SENS: DEGRADED",
      coordActive: "COORD: F1 91.113 · F4 GHOST",
      coordStandby: "COORD: F1 91.113 · F4 READY",
    },
    drawer: {
      title: "4-Ch Live AI",
      bridgeOnline: "6/6 AI (:8765)",
      bridgeNative: "RUST-ORT IPC",
      bridgeFallback: "Synthetic Feed",
      avionicsTitle: "v5.2 Avionics Health & Assurance",
      assuranceTitle: "Formal & Bare-Metal Assurance",
      uploadHint: "Drop or pick any sky / runway image for live AI analysis",
      uploadBtn: "Test Your Image",
      channels: {
        d2: "XVS-D2 · EO Sky",
        land: "XVS-Land · Runway 12-KPT",
        thermal: "XVS-Thermal · LWIR Fog",
        nav: "XVS-Nav v5.0 · MNV4",
      },
    },
    bench: {
      title: "OpenXVS Studio v5.2 — Video, Camera & Eval Lab",
      subtitle: `${STUDIO_STACK_V5_SPEC} · ${STUDIO_NAV_V5_SPEC}`,
      tabVideo: "Video / Sequence Analyzer",
      tabWebcam: "Live Webcam / USB Cam",
      tabHil: "HIL Self-Test & Eval Pack",
      close: "Close",
      dropVideoHint: "Drop or pick an .MP4, .WEBM, or .MOV flight/sky video (max 150 MB)",
      dropVideoNativeHint: "Standalone .EXE: Play built-in real frames via Pure-Rust ORT IPC or inspect local .MP4/.WEBM frames",
      pickVideoBtn: "Choose Video File",
      runVideoBtn: "Process with Real AI",
      runBuiltinBtn: "1-Click Built-In Real Flight (AOT 1080p)",
      runBuiltinLandBtn: "1-Click KSFO Approach Clip",
      runNativeSeqBtn: "1-Click Pure-Rust ORT Sequence",
      runNativeLandSeqBtn: "1-Click KSFO Approach Sequence (Rust ORT)",
      stopSeqBtn: "Stop Sequence",
      playSeqBtn: "Play",
      pauseSeqBtn: "Pause",
      nativeVideoNote: "Standalone Pure-Rust ORT IPC: Executes sequential real flight frames natively in ONNX Runtime with zero Python and zero :8765 bridge.",
      arPassiveNote: "Passive straight flight · no S-turn",
      arD2OnlyNote: "D2 channel only",
      sampleFpsLabel: "Sample FPS",
      maxFramesLabel: "Max Frames",
      channelLabel: "Model Channel",
      processing: "Running real OpenXVS checkpoint across video frames...",
      processingNative: "Running Pure-Rust ORT IPC across sequential frames...",
      downloadMp4: "Download Annotated .MP4",
      downloadJson: "Download JSON Report",
      metricDetRate: "Detection Rate",
      metricLatency: "Mean Latency",
      metricArLocks: "ActiveRange Locks",
      metricSha: "Checkpoint SHA-256",
      startCamBtn: "Start Camera (~4 FPS Live AI)",
      stopCamBtn: "Stop Camera",
      rawCamLabel: "Raw Camera Feed",
      aiCamLabel: "OpenXVS Real AI Output",
      camDenied: "Camera permission denied or no camera device detected. Please check browser permissions.",
      bridgeRequired: "Local 5/5 AI bridge (http://127.0.0.1:8765) is offline. Launch `python scripts/xvs/studio_live_server.py` in browser dev mode.",
      hilTitle: "Zero-GPL MAVLink2 (ArduPlane/PX4) & FlightGear UDP Bridge",
      hilModeLabel: "Bridge Mode",
      hilApplyBtn: "Apply UDP Bridge",
      hilLoopbackBtn: "1-Click Live Protocol Self-Test (Loopback)",
      hilRunning: "Running 127.0.0.1 UDP loopback certification checks...",
      nativeHilLockNotice: "Standalone .EXE · NON_FLIGHT_SIM_ONLY=1 (EU PLD Art. 4(7)): outbound UDP actuation is compiled out; 1-Click native self-test is active.",
      buyerPackTitle: "1-Click Buyer Verification & Certification Pack (.ZIP)",
      buyerPackBtn: "Download Buyer Assurance Pack (.ZIP)",
      nativeEvalPackTitle: "1-Click Standalone Eval Report (.JSON + SHA-256)",
      nativeEvalPackBtn: "Download Standalone Eval Report (.JSON)",
      nativeEvalExporting: "Generating eval report...",
      nativeEvalExportedPrefix: "Downloaded:",
      buyerPackChecklist: [
        "01_executive_summary.json — 6/6 Pure-Rust ORT + XFeat + LighterGlue FP16 SHA-256s + Honest G5_GAP Disclosure",
        "02_nav_v5_benchmark.json — XVS-Nav v5.0 MNV4-Hybrid-M · 13.1ms P50 · 25.3MB S1+XFeat+LighterGlue FP16 · KSFO R@1 91.4% / R@5 96.7% · Sequence-VPR 0 HMI",
        "03_formal_assurance.json — 26/26 Kani CBMC (5,228 checks) + Apalache N=2/N=3 + Cortex-M7 thumbv7em (188 KB Flash / 0 B RAM)",
        "04_sensor_health_and_hil.json — 12-Sector EO + LWIR FFC/Contrast + FSYNC <=1.0ms + M-of-N TBC + 100/100 Signed #52001",
        "05_latest_video_analysis.json — Sequential Real Flight Frame Analysis Report (Pure-Rust ORT / AOT)",
        "06_FIELD_TRIAL_PROTOCOL.md — Tier-0A ($0) → Tier-0B ($408) → Phase-4 Trial Guide",
      ],
    },
  },
};

export function getStudioStrings(locale: StudioLocale): StudioStrings {
  return STRINGS[locale];
}
