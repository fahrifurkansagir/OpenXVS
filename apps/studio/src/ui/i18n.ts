/**
 * Bilingual (`TR / EN`) UI Strings for OpenXVS Studio v1.1.
 * Strictly civil & dual-use aviation terminology.
 */

import type { VisualWorldMode } from "../scene/atmosphere_world.ts";
import type {
  BreachReason,
  LandingGrade,
  ShieldReason,
} from "../sim/manual_flight_controller.ts";
import type { StudioScenarioId } from "../sim/studio_sim.ts";

export type StudioLocale = "tr" | "en";
export type StudioCameraPreset = "chase" | "wingman" | "overview" | "cockpit";
export type TerrainStyleMode = VisualWorldMode;

export const STUDIO_VERSION_LABEL = "v1.1" as const;
export const STUDIO_APP_TITLE = "OpenXVS Studio" as const;
export const STUDIO_STACK_V5_SPEC = "6/6 Pure-Rust ORT + XFeat + LighterGlue + MultiShipCoord" as const;
export const STUDIO_NAV_V5_SPEC =
  "XVS-Nav: MNV4-Hybrid-M ONNX · 13.1ms P50 · 25.3MB S1+XFeat+LighterGlue FP16 · 0-shot KSFO R@1 91.4% / R@5 96.7% · HPL <20m · 8-Arm CUSUM" as const;
export const STUDIO_ASSURANCE_V52_SPEC =
  "26/26 Kani CBMC (5,228 checks) · Apalache N=2/N=3 · Cortex-M7 thumbv7em (88 KB Flash / 0 B RAM) · Sequence-VPR (0 HMI) · M-of-N TBC" as const;

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
    readonly shieldReasons: Record<Exclude<ShieldReason, "NONE">, string>;
    readonly breachReasons: Record<BreachReason, string>;
    readonly touchdownGrades: Record<Exclude<LandingGrade, "NONE">, string>;
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
    readonly tabFeeds: string;
    readonly tabAvionics: string;
    readonly bridgeOnline: string;
    readonly bridgeNative: string;
    readonly bridgeFallback: string;
    readonly retryBridgeBtn: string;
    readonly retryingBridge: string;
    readonly avionicsTitle: string;
    readonly assuranceTitle: string;
    readonly specsSummary: string;
    readonly uploadHint: string;
    readonly uploadBtn: string;
    readonly uploadAnalyzing: string;
    readonly uploadOffline: string;
    readonly uploadFailed: string;
    readonly uploadSelectAria: string;
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
    readonly specsSummary: string;
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
    readonly camStatusOffline: string;
    readonly camStatusReady: string;
    readonly camStatusLive: string;
    readonly camRawEmpty: string;
    readonly camAiEmptyOffline: string;
    readonly camAiEmptyOnline: string;
    readonly camDenied: string;
    readonly bridgeRequired: string;
    readonly bridgeRequiredPrefix: string;
    readonly bridgeRequiredCmd: string;
    readonly bridgeRequiredSuffix: string;
    readonly startVideoFailed: string;
    readonly pollLost: string;
    readonly videoDecodeError: string;
    readonly nativeSeqEmpty: string;
    readonly hilTitle: string;
    readonly hilModeLabel: string;
    readonly hilModeIdle: string;
    readonly hilModeFlightGear: string;
    readonly hilModeMavlink2: string;
    readonly hilBadgeSimOnly: string;
    readonly hilBadgeGplFree: string;
    readonly hilStatusIdle: string;
    readonly hilApplyBtn: string;
    readonly hilLoopbackBtn: string;
    readonly hilRunning: string;
    readonly hilActuationBlocked: string;
    readonly hilLoopbackFailedNative: string;
    readonly hilLoopbackFailedBridge: string;
    readonly nativeHilLockNotice: string;
    readonly buyerPackTitle: string;
    readonly buyerPackBtn: string;
    readonly buyerPackOffline: string;
    readonly nativeEvalPackTitle: string;
    readonly nativeEvalPackBtn: string;
    readonly nativeEvalExporting: string;
    readonly nativeEvalExportedPrefix: string;
    readonly nativeEvalExportFailed: string;
    readonly buyerPackChecklist: readonly string[];
  };
}

const STRINGS: Record<StudioLocale, StudioStrings> = {
  tr: {
    brandTitle: "OPENXVS STUDIO",
    brandVersion: STUDIO_VERSION_LABEL,
    brandBadge: "6/6 Yerel AI",
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
        banner: "XVS-Nav · KSFO R@1 91.4% / R@5 96.7% · 13.1ms · CUSUM",
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
      shieldReasons: {
        "SC3BF CBDR": "SC3BF ÇARPIŞMA ÖNLEME",
        "TERRAIN FLOOR": "ARAZİ TABANI KORUMASI",
        "FORCEDLAND GLIDE": "ZORUNLU İNİŞ SÜZÜLME",
      },
      breachReasons: {
        "SC3BF COLLISION": "SC3BF AYIRMA İHLALİ (<152.4m)",
        "TERRAIN FLOOR": "ARAZİ TABANI İHLALİ",
        "ENGINE-OUT STALL": "TEK MOTOR STALL İHLALİ",
      },
      touchdownGrades: {
        PERFECT: "KUSURSUZ",
        SAFE: "GÜVENLİ",
        HARD: "SERT İNİŞ",
      },
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
      tabFeeds: "4-Kanal Akış",
      tabAvionics: "Aviyonik & Güvence",
      bridgeOnline: "6/6 AI (:8765)",
      bridgeNative: "RUST-ORT IPC",
      bridgeFallback: "Sentetik Mod",
      retryBridgeBtn: "Bağlan",
      retryingBridge: "Deneniyor…",
      avionicsTitle: "Aviyonik Sağlık ve Korumalar",
      assuranceTitle: "Formal & Bare-Metal Güvence",
      specsSummary: "Sistem Özellikleri",
      uploadHint: "Canlı AI analizi için gökyüzü veya pist görüntüsü seçin",
      uploadBtn: "Görüntü Test Et",
      uploadAnalyzing: "Analiz ediliyor…",
      uploadOffline: "Yerel 6/6 AI köprüsü çevrimdışı — görüntü analizi için yerel köprüyü başlatın.",
      uploadFailed: "Görüntü analizi başarısız oldu. Lütfen bağlantıyı veya dosya biçimini kontrol edin.",
      uploadSelectAria: "AI Model Kanalı",
      channels: {
        d2: "XVS-D2 · EO Gökyüzü",
        land: "XVS-Land · Pist 12-KPT",
        thermal: "XVS-Thermal · LWIR Sis",
        nav: "XVS-Nav · Görsel Konum",
      },
    },
    bench: {
      title: "OpenXVS Studio — Video, Kamera & Doğrulama Lab",
      subtitle: "Yerel 6/6 ONNX Runtime Çıkarım · Video, Canlı Kamera ve HIL Doğrulama",
      specsSummary: "Sistem ve Model Özellikleri",
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
      camStatusOffline: "ÇEVRİMDIŞI",
      camStatusReady: "HAZIR",
      camStatusLive: "CANLI",
      camRawEmpty: "Canlı girişi başlatmak için “Kamerayı Başlat” düğmesine tıklayın.",
      camAiEmptyOffline: "Yerel 6/6 AI köprüsü çevrimdışı. Canlı AI çıkarımı için yerel köprüyü başlatın.",
      camAiEmptyOnline: "Kamera başlatıldığında işaretli AI çıkarım akışı burada görünür.",
      camDenied: "Kamera erişimi reddedildi veya bağlı kamera bulunamadı. Lütfen tarayıcı izinlerini kontrol edin.",
      bridgeRequired:
        "Yerel 6/6 AI sunucusu (http://127.0.0.1:8765) çevrimdışı. Tarayıcı önizleme modunda python scripts/xvs/studio_live_server.py çalıştırın.",
      bridgeRequiredPrefix: "Yerel 6/6 AI sunucusu (http://127.0.0.1:8765) çevrimdışı. Başlatmak için çalıştırın:",
      bridgeRequiredCmd: "python scripts/xvs/studio_live_server.py",
      bridgeRequiredSuffix: "",
      startVideoFailed: "Video analiz görevi başlatılamadı (yerel :8765 köprüsünü kontrol edin).",
      pollLost: "Video görevi izlenirken yerel köprü bağlantısı kesildi.",
      videoDecodeError: "Tarayıcı bu video kapsayıcısını veya kodeğini çözemedi.",
      nativeSeqEmpty: "Yerel Pure-Rust ORT sekansı kare döndürmedi.",
      hilTitle: "Sıfır-GPL MAVLink2 (ArduPlane/PX4) & FlightGear UDP Köprüsü",
      hilModeLabel: "Köprü Modu",
      hilModeIdle: "Boşta (Bağlı Değil)",
      hilModeFlightGear: "FlightGear UDP (5500 / 5501)",
      hilModeMavlink2: "MAVLink2 UDP — ArduPlane/PX4 (14550 / 14540)",
      hilBadgeSimOnly: "SİMÜLASYON KİLİDİ · SIFIR AKTÜASYON",
      hilBadgeGplFree: "GPL-SİZ · SIFIR PYMAVLINK",
      hilStatusIdle: "127.0.0.1 · BOŞTA",
      hilApplyBtn: "UDP Köprüsünü Uygula",
      hilLoopbackBtn: "1-Tık Canlı Protokol Öz-Testi (Loopback)",
      hilRunning: "127.0.0.1 UDP loopback doğrulama kontrolleri çalışıyor...",
      hilActuationBlocked: "Dış UDP aktüasyonu kapalı (yalnızca simülasyon) veya :8765 köprüsü çevrimdışı.",
      hilLoopbackFailedNative: "Yerel loopback öz-testi başarısız oldu.",
      hilLoopbackFailedBridge: ":8765 üzerinde HIL loopback öz-testi başarısız oldu.",
      nativeHilLockNotice: "Standalone .EXE · Yalnızca Simülasyon (EU PLD Art. 4(7)): dış UDP aktüasyonu kapalıdır; 1-Tık yerel öz-test aktiftir.",
      buyerPackTitle: "1-Tık Doğrulama & Sertifikasyon Paketi (.ZIP)",
      buyerPackBtn: "Doğrulama Paketini İndir (.ZIP)",
      buyerPackOffline: "Paket indirilemedi: yerel 6/6 AI sunucusu (http://127.0.0.1:8765) çevrimdışı.",
      nativeEvalPackTitle: "1-Tık Standalone Değerlendirme Raporu (.JSON + SHA-256)",
      nativeEvalPackBtn: "Standalone Eval Raporu İndir (.JSON)",
      nativeEvalExporting: "Eval raporu üretiliyor...",
      nativeEvalExportedPrefix: "İndirildi:",
      nativeEvalExportFailed: "Standalone değerlendirme raporu dışa aktarılamadı.",
      buyerPackChecklist: [
        "01_executive_summary.json — 6/6 ONNX Runtime Modeli, SHA-256 Doğrulama ve Değerlendirme Kapsamı",
        "02_nav_benchmark.json — XVS-Nav MNV4-Hybrid-M · 13.1ms P50 · 25.3MB FP16 · KSFO R@1 91.4% / R@5 96.7% · Sequence-VPR 0 HMI",
        "03_formal_assurance.json — 26/26 Kani CBMC (5,228 kontrol) + Apalache N=2/N=3 + Cortex-M7 thumbv7em (88 KB Flash / 0 B RAM)",
        "04_sensor_health_and_hil.json — 12-Sektör EO + LWIR FFC/Kontrast + FSYNC <=1.0ms + M-of-N TBC + 100/100 İmzalı #52001",
        "05_latest_video_analysis.json — Ardışık Gerçek Uçuş Karesi Analiz Raporu (Pure-Rust ORT / AOT)",
        "06_FIELD_TRIAL_PROTOCOL.md — Yerden Uçuşa Kademeli Doğrulama ve Güvenlik Protokolü",
      ],
    },
  },
  en: {
    brandTitle: "OPENXVS STUDIO",
    brandVersion: STUDIO_VERSION_LABEL,
    brandBadge: "6/6 Local AI",
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
        banner: "XVS-Nav · KSFO R@1 91.4% / R@5 96.7% · 13.1ms · CUSUM",
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
      shieldReasons: {
        "SC3BF CBDR": "SC3BF CBDR AVOIDANCE",
        "TERRAIN FLOOR": "TERRAIN FLOOR GUARD",
        "FORCEDLAND GLIDE": "FORCEDLAND GLIDE GUARD",
      },
      breachReasons: {
        "SC3BF COLLISION": "SC3BF COLLISION BREACH (<152.4m)",
        "TERRAIN FLOOR": "TERRAIN FLOOR BREACH",
        "ENGINE-OUT STALL": "ENGINE-OUT STALL BREACH",
      },
      touchdownGrades: {
        PERFECT: "PERFECT",
        SAFE: "SAFE",
        HARD: "HARD LANDING",
      },
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
      tabFeeds: "4-Ch Feeds",
      tabAvionics: "Avionics & Assurance",
      bridgeOnline: "6/6 AI (:8765)",
      bridgeNative: "RUST-ORT IPC",
      bridgeFallback: "Synthetic Feed",
      retryBridgeBtn: "Retry",
      retryingBridge: "Connecting…",
      avionicsTitle: "Avionics Health & Guards",
      assuranceTitle: "Formal & Bare-Metal Assurance",
      specsSummary: "System Specifications",
      uploadHint: "Drop or pick any sky / runway image for live AI analysis",
      uploadBtn: "Test Image",
      uploadAnalyzing: "Analyzing…",
      uploadOffline: "Local 6/6 AI bridge is offline — launch the local server to analyze images.",
      uploadFailed: "Image analysis failed. Please check bridge connectivity or image format.",
      uploadSelectAria: "AI Model Channel",
      channels: {
        d2: "XVS-D2 · EO Sky",
        land: "XVS-Land · Runway 12-KPT",
        thermal: "XVS-Thermal · LWIR Fog",
        nav: "XVS-Nav · Visual Geo-Lock",
      },
    },
    bench: {
      title: "OpenXVS Studio — Video, Camera & Eval Lab",
      subtitle: "Local 6/6 ONNX Runtime Inference · Video, Live Camera & HIL Verification",
      specsSummary: "System & Model Specifications",
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
      camStatusOffline: "OFFLINE",
      camStatusReady: "READY",
      camStatusLive: "LIVE",
      camRawEmpty: "Click “Start Camera” to stream local webcam or USB camera video.",
      camAiEmptyOffline: "Local 6/6 AI bridge is offline. Start the local bridge to view live AI detections.",
      camAiEmptyOnline: "Annotated live AI output will render here once the camera starts.",
      camDenied: "Camera permission denied or no camera device detected. Please check browser permissions.",
      bridgeRequired:
        "Local 6/6 AI bridge (http://127.0.0.1:8765) is offline. Launch python scripts/xvs/studio_live_server.py in browser preview mode.",
      bridgeRequiredPrefix: "Local 6/6 AI bridge (http://127.0.0.1:8765) is offline. Launch with:",
      bridgeRequiredCmd: "python scripts/xvs/studio_live_server.py",
      bridgeRequiredSuffix: "",
      startVideoFailed: "Failed to start video analysis job (check :8765 bridge).",
      pollLost: "Lost connection while polling video analysis job.",
      videoDecodeError: "Browser could not decode this video container or codec.",
      nativeSeqEmpty: "Native Pure-Rust ORT sequence tick returned no frames.",
      hilTitle: "Zero-GPL MAVLink2 (ArduPlane/PX4) & FlightGear UDP Bridge",
      hilModeLabel: "Bridge Mode",
      hilModeIdle: "Idle (Disconnected)",
      hilModeFlightGear: "FlightGear UDP (5500 / 5501)",
      hilModeMavlink2: "MAVLink2 UDP — ArduPlane/PX4 (14550 / 14540)",
      hilBadgeSimOnly: "SIMULATION ONLY · ZERO ACTUATION",
      hilBadgeGplFree: "GPL-FREE · ZERO PYMAVLINK",
      hilStatusIdle: "127.0.0.1 · IDLE",
      hilApplyBtn: "Apply UDP Bridge",
      hilLoopbackBtn: "1-Click Live Protocol Self-Test (Loopback)",
      hilRunning: "Running 127.0.0.1 UDP loopback verification checks...",
      hilActuationBlocked: "Outbound HIL actuation is locked (simulation only) or dev bridge (:8765) is offline.",
      hilLoopbackFailedNative: "Native loopback self-test failed.",
      hilLoopbackFailedBridge: "HIL loopback self-test failed on :8765.",
      nativeHilLockNotice: "Standalone .EXE · Simulation Only (EU PLD Art. 4(7)): outbound UDP actuation is compiled out; 1-Click native self-test is active.",
      buyerPackTitle: "1-Click Verification & Assurance Pack (.ZIP)",
      buyerPackBtn: "Download Assurance Pack (.ZIP)",
      buyerPackOffline: "Download unavailable: local 6/6 AI bridge (http://127.0.0.1:8765) is offline.",
      nativeEvalPackTitle: "1-Click Standalone Eval Report (.JSON + SHA-256)",
      nativeEvalPackBtn: "Download Standalone Eval Report (.JSON)",
      nativeEvalExporting: "Generating eval report...",
      nativeEvalExportedPrefix: "Downloaded:",
      nativeEvalExportFailed: "Failed to export standalone eval report via native IPC.",
      buyerPackChecklist: [
        "01_executive_summary.json — 6/6 ONNX Runtime Models, SHA-256 Provenance & Evaluation Scope",
        "02_nav_benchmark.json — XVS-Nav MNV4-Hybrid-M · 13.1ms P50 · 25.3MB FP16 · KSFO R@1 91.4% / R@5 96.7% · Sequence-VPR 0 HMI",
        "03_formal_assurance.json — 26/26 Kani CBMC (5,228 checks) + Apalache N=2/N=3 + Cortex-M7 thumbv7em (88 KB Flash / 0 B RAM)",
        "04_sensor_health_and_hil.json — 12-Sector EO + LWIR FFC/Contrast + FSYNC <=1.0ms + M-of-N TBC + 100/100 Signed #52001",
        "05_latest_video_analysis.json — Sequential Real Flight Frame Analysis Report (Pure-Rust ORT / AOT)",
        "06_FIELD_TRIAL_PROTOCOL.md — Ground-to-Flight Progressive Validation & Safety Protocol",
      ],
    },
  },
};

export function getStudioStrings(locale: StudioLocale): StudioStrings {
  return STRINGS[locale];
}

export function formatTouchdownGrade(grade: LandingGrade, locale: StudioLocale): string {
  if (grade === "NONE") return "";
  return STRINGS[locale].telemetry.touchdownGrades[grade];
}

export function formatShieldReason(reason: ShieldReason, locale: StudioLocale): string {
  if (reason === "NONE") return "";
  return STRINGS[locale].telemetry.shieldReasons[reason];
}

export function formatBreachReason(reason: BreachReason | null, locale: StudioLocale): string {
  if (!reason) return locale === "tr" ? "KALKAN KAPALI" : "SHIELD OFF";
  return STRINGS[locale].telemetry.breachReasons[reason];
}
