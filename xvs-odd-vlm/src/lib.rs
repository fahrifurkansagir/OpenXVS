//! OpenXVS `#![no_std]` `#![forbid(unsafe_code)]` Safety Crate:
//! - **DR-13 §1.3**: Zero-ML 5-Mode Deterministic Optical-ODD Monitor + `O(1)` MLTL Ring Counters
//! - **DR-09 §4**: Asymmetric Simplex 6-Stage DO-178C DAL B/C VLM Verification Gate (IDAL D/E isolation)
//!
//! Strictly civil/dual-use unmanned aviation safety. Zero heap allocations, zero external crates,
//! zero `unsafe`, zero floating-point non-determinism, and 1:1 wire/enum parity with
//! `configs/mavlink/pod_daa.xml` and `src/xvs/v2/xvs_vlm_gate/gate.py`.

#![no_std]
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub const PACKET_MAGIC: [u8; 2] = *b"XV";
pub const PACKET_VERSION: u8 = 1;
pub const PACKET_SIZE: usize = 32;
pub const BODY_SIZE: usize = 28;
pub const NUM_AZIMUTH_SECTORS: u8 = 12;
pub const NUM_RANKED_ZONES: usize = 4;
pub const UNUSED_ZONE_ID: u8 = 0xFF;
pub const MAX_FLZ_ID: u8 = 16;

/// Canonical 1:1 enum matching `configs/mavlink/pod_daa.xml` (`POD_VLM_HAZARD_TYPE`)
/// and `src/xvs/v2/xvs_vlm_gate/gate.py` (`VlmHazard`).
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum VlmHazard {
    None = 0,
    RunwayIncursion = 1,
    BirdFlock = 2,
    ConvectiveVirga = 3,
    CraneProxy = 4,
    FlzHazard = 5,
}

impl VlmHazard {
    #[inline]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::None),
            1 => Some(Self::RunwayIncursion),
            2 => Some(Self::BirdFlock),
            3 => Some(Self::ConvectiveVirga),
            4 => Some(Self::CraneProxy),
            5 => Some(Self::FlzHazard),
            _ => None,
        }
    }

    /// Bitmask bit `1 << (h - 1)` for hazards `1..=5`, or `0` for `None`.
    #[inline]
    pub const fn bit(self) -> u8 {
        match self {
            Self::None => 0,
            Self::RunwayIncursion => 1 << 0,
            Self::BirdFlock => 1 << 1,
            Self::ConvectiveVirga => 1 << 2,
            Self::CraneProxy => 1 << 3,
            Self::FlzHazard => 1 << 4,
        }
    }
}

/// Canonical 1:1 enum matching `configs/mavlink/pod_daa.xml` (`POD_VLM_GATE_VERDICT`)
/// and `src/xvs/v2/xvs_vlm_gate/gate.py` (`GateStatus`).
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum GateStatus {
    NominalAdvisory = 0,
    VlmOffline = 1,
    SchemaOrCrcFail = 2,
    OpticalOddMasked = 3,
    ConformalAbstain = 4,
    DebouncePending = 5,
}

/// EASA Objective RA-08 aleatoric vs epistemic uncertainty separation.
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum UncertaintySource {
    None = 0,
    Aleatoric = 1,
    Epistemic = 2,
    Both = 3,
}

/// DR-13 Stage-3 Optical-ODD deterministic routing action.
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum OddAction {
    Nominal = 0,
    EvsLwirElevated = 1,
    ThermalPrimary = 2,
    DegradedFallback = 3,
}

/// Pure-`core` IEEE 802.3 CRC32 matching Python `zlib.crc32` bit-for-bit.
pub const fn ieee_crc32(bytes: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    let mut i: usize = 0;
    while i < bytes.len() {
        crc ^= bytes[i] as u32;
        let mut bit: u8 = 0;
        while bit < 8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
            bit += 1;
        }
        i += 1;
    }
    !crc
}

/// Fixed 32-byte little-endian VLM advisory packet matching Python `VlmAdvisoryPacket`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct VlmWirePacket {
    pub seq: u32,
    pub capture_ts_ms: u64,
    pub hazard_enum: u8,
    pub azimuth_sector: u8,
    pub ranked_zone_ids: [u8; NUM_RANKED_ZONES],
    pub conformal_valid: u8,
    pub aleatoric_flag: u8,
    pub epistemic_flag: u8,
    pub crc32: u32,
}

impl VlmWirePacket {
    /// Encode into the exact 32-byte wire buffer with computed IEEE 802.3 CRC32.
    pub fn encode(&self) -> [u8; PACKET_SIZE] {
        let mut buf = [0u8; PACKET_SIZE];
        buf[0] = PACKET_MAGIC[0];
        buf[1] = PACKET_MAGIC[1];
        buf[2] = PACKET_VERSION;
        buf[3] = 0;

        let seq_b = self.seq.to_le_bytes();
        buf[4] = seq_b[0];
        buf[5] = seq_b[1];
        buf[6] = seq_b[2];
        buf[7] = seq_b[3];

        let ts_b = self.capture_ts_ms.to_le_bytes();
        buf[8] = ts_b[0];
        buf[9] = ts_b[1];
        buf[10] = ts_b[2];
        buf[11] = ts_b[3];
        buf[12] = ts_b[4];
        buf[13] = ts_b[5];
        buf[14] = ts_b[6];
        buf[15] = ts_b[7];

        buf[16] = self.hazard_enum;
        buf[17] = self.azimuth_sector;
        buf[18] = self.ranked_zone_ids[0];
        buf[19] = self.ranked_zone_ids[1];
        buf[20] = self.ranked_zone_ids[2];
        buf[21] = self.ranked_zone_ids[3];
        buf[22] = self.conformal_valid;
        buf[23] = self.aleatoric_flag;
        buf[24] = self.epistemic_flag;
        buf[25] = 0;
        buf[26] = 0;
        buf[27] = 0;

        let crc = ieee_crc32(&buf[..BODY_SIZE]);
        let crc_b = crc.to_le_bytes();
        buf[28] = crc_b[0];
        buf[29] = crc_b[1];
        buf[30] = crc_b[2];
        buf[31] = crc_b[3];
        buf
    }

    /// Deterministically decode and verify magic, version, zero padding, IEEE CRC32, and schema.
    pub fn decode_and_verify(raw: &[u8; PACKET_SIZE]) -> Result<Self, GateStatus> {
        if raw[0] != PACKET_MAGIC[0] || raw[1] != PACKET_MAGIC[1] || raw[2] != PACKET_VERSION {
            return Err(GateStatus::SchemaOrCrcFail);
        }
        if raw[3] != 0 || raw[25] != 0 || raw[26] != 0 || raw[27] != 0 {
            return Err(GateStatus::SchemaOrCrcFail);
        }
        let wire_crc = u32::from_le_bytes([raw[28], raw[29], raw[30], raw[31]]);
        let expected_crc = ieee_crc32(&raw[..BODY_SIZE]);
        if wire_crc != expected_crc {
            return Err(GateStatus::SchemaOrCrcFail);
        }

        let seq = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
        let capture_ts_ms = u64::from_le_bytes([
            raw[8], raw[9], raw[10], raw[11], raw[12], raw[13], raw[14], raw[15],
        ]);
        let pkt = Self {
            seq,
            capture_ts_ms,
            hazard_enum: raw[16],
            azimuth_sector: raw[17],
            ranked_zone_ids: [raw[18], raw[19], raw[20], raw[21]],
            conformal_valid: raw[22],
            aleatoric_flag: raw[23],
            epistemic_flag: raw[24],
            crc32: wire_crc,
        };
        if !pkt.schema_valid() {
            return Err(GateStatus::SchemaOrCrcFail);
        }
        Ok(pkt)
    }

    /// Stage-2 static enum/range/flag check.
    #[inline]
    pub const fn schema_valid(&self) -> bool {
        VlmHazard::from_u8(self.hazard_enum).is_some()
            && self.azimuth_sector < NUM_AZIMUTH_SECTORS
            && self.conformal_valid <= 1
            && self.aleatoric_flag <= 1
            && self.epistemic_flag <= 1
    }
}

// ===========================================================================
// PART 1: DR-13 Zero-ML 5-Mode Optical-ODD Monitor + O(1) MLTL Ring Counters
// ===========================================================================

/// Fixed-size `O(1)` correlated-block `M-of-N` + `G[0, debounce_ticks-1]` MLTL ring counter.
#[derive(Copy, Clone, Debug)]
pub struct MltlRing<const N: usize> {
    ring: [bool; N],
    head: usize,
    filled: usize,
    running_sum: u16,
    confirm_m: u16,
    debounce_ticks: u8,
    debounce_streak: u8,
}

impl<const N: usize> MltlRing<N> {
    pub const fn new(confirm_m: u16, debounce_ticks: u8) -> Self {
        let m = if confirm_m == 0 {
            1
        } else if (confirm_m as usize) > N {
            N as u16
        } else {
            confirm_m
        };
        let d = if debounce_ticks == 0 { 1 } else { debounce_ticks };
        Self {
            ring: [false; N],
            head: 0,
            filled: 0,
            running_sum: 0,
            confirm_m: m,
            debounce_ticks: d,
            debounce_streak: 0,
        }
    }

    pub fn reset(&mut self) {
        let mut i = 0;
        while i < N {
            self.ring[i] = false;
            i += 1;
        }
        self.head = 0;
        self.filled = 0;
        self.running_sum = 0;
        self.debounce_streak = 0;
    }

    /// `O(1)` ring-sum update (zero loops, zero overflow, zero panic).
    pub fn update(&mut self, raw_trip: bool) -> (u16, bool) {
        if N == 0 {
            return (0, false);
        }
        if self.filled == N {
            if self.ring[self.head] {
                self.running_sum = self.running_sum.saturating_sub(1);
            }
        } else {
            self.filled += 1;
        }
        self.ring[self.head] = raw_trip;
        if raw_trip {
            self.running_sum = self.running_sum.saturating_add(1);
        }
        self.head += 1;
        if self.head == N {
            self.head = 0;
        }

        if self.running_sum >= self.confirm_m {
            self.debounce_streak = self.debounce_streak.saturating_add(1);
        } else {
            self.debounce_streak = 0;
        }
        (self.running_sum, self.debounce_streak >= self.debounce_ticks)
    }
}

/// Deterministic fixed-point physical inputs for all 5 DR-13 Optical-ODD modes.
/// Notice: ZERO airspeed, altitude, AoA, or pitot fields (AF447 structural isolation).
#[derive(Copy, Clone, Debug)]
pub struct OpticalOddSensors {
    /// Mode 1: persisted occluded tile fraction in permille (0..=1000; trip > 120).
    pub droplet_occ_permille: u16,
    /// Mode 1: mean Modified Laplacian sharpness in centi-units (trip low sharpness < 5000).
    pub blur_mlap_centi: u32,
    /// Mode 2: 16-line 512-pt RFFT HF MTF ratio in permille (trip < 140 = 0.35 * 0.40).
    pub mtf_hf_ratio_permille: u16,
    /// Mode 2 & 4 & 5: Boson+ 14-bit Pre-AGC scene std in centi-DN (NETD=134; opaque < 469, healthy > 804).
    pub lwir_std_centidn: u32,
    /// Mode 2: Buck SVP window dew/frost margin `T_win - T_dew_frost` in centi-degC (trip <= 200).
    pub dew_margin_centic: i32,
    /// Mode 3: SPA sun off-boresight angle in centi-deg (cone < hfov/2 + 1200).
    pub sun_boresight_cdeg: u16,
    /// Mode 3: EO horizontal FOV in centi-deg (default 6000 = 60.00 deg).
    pub eo_hfov_cdeg: u16,
    /// Mode 3: saturated pixel fraction (I >= 1008 DN) in permille (trip > 30).
    pub glare_sat_permille: u16,
    /// Mode 3: radial ring contrast ratio C_ring/C_ref in permille (trip < 200).
    pub glare_ring_ratio_permille: u16,
    /// Mode 4: WMO 5% Meteorological Optical Range in meters (trip < 1609 m).
    pub mor_m: u32,
    /// Mode 4: 3DEP-projected horizon vertical Sobel SNR in centi-dB (trip < 300 = 3.00 dB).
    pub horizon_snr_centidb: i32,
    /// Mode 5: 14-bit Pre-AGC median drift |med - ref| in DN (trip > 650).
    pub lwir_median_drift_dn: u16,
    /// Mode 5: column/row FPN HF-PSD ratio in permille (trip > 350).
    pub lwir_fpn_ratio_permille: u16,
    /// Mode 5: 3x3 bad-pixel fraction in ppm (trip > 10_000 = 1.0% Industrial).
    pub lwir_bad_px_ppm: u32,
    /// Mode 5: seconds since last Boson+ FFC (trip > 300 s).
    pub boson_dt_ffc_s: u32,
    /// Mode 5: |T_cam - T_last_ffc| in centi-degC (trip > 100 = 1.00 C).
    pub boson_temp_delta_centic: u16,
    /// Mode 5: instantaneous 1-tick Boson+ shutter FFC freeze flag.
    pub boson_ffc_freeze: bool,
    /// Classical AHRS attitude within optical ODD envelope.
    pub attitude_in_odd: bool,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct OpticalOddOutput {
    pub action: OddAction,
    pub raw_mask: u8,
    pub persisted_mask: u8,
    pub mask_lwir_frame: bool,
    pub vlm_gate_reject: bool,
    pub ito_heater_cmd: bool,
    pub airdata_export_blocked: bool,
}

/// Zero-ML Stage-3 Optical-ODD Monitor enforcing the hard AF447 compile-time invariant.
#[derive(Copy, Clone, Debug)]
pub struct OpticalOddMonitor<
    const N1: usize = 30,
    const N2: usize = 90,
    const N3: usize = 30,
    const N4: usize = 60,
    const N5: usize = 60,
> {
    m1_rain: MltlRing<N1>,
    m2_ice: MltlRing<N2>,
    m3_glare: MltlRing<N3>,
    m4_fog: MltlRing<N4>,
    m5_lwir: MltlRing<N5>,
}

impl OpticalOddMonitor<30, 90, 30, 60, 60> {
    /// Canonical DR-13 §1.3 / §1.4 60 Hz correlated-block persistence configuration.
    pub const fn new_canonical_60hz() -> Self {
        Self {
            m1_rain: MltlRing::new(21, 3),
            m2_ice: MltlRing::new(65, 3),
            m3_glare: MltlRing::new(22, 3),
            m4_fog: MltlRing::new(42, 3),
            m5_lwir: MltlRing::new(45, 3),
        }
    }
}

impl<const N1: usize, const N2: usize, const N3: usize, const N4: usize, const N5: usize>
    OpticalOddMonitor<N1, N2, N3, N4, N5>
{
    /// HARD AF447 INVARIANT (DR-08 / DR-13): optical monitoring NEVER exports airdata.
    pub const AIRDATA_EXPORT_BLOCKED: bool = true;

    pub const fn new_with_specs(
        m1: (u16, u8),
        m2: (u16, u8),
        m3: (u16, u8),
        m4: (u16, u8),
        m5: (u16, u8),
    ) -> Self {
        Self {
            m1_rain: MltlRing::new(m1.0, m1.1),
            m2_ice: MltlRing::new(m2.0, m2.1),
            m3_glare: MltlRing::new(m3.0, m3.1),
            m4_fog: MltlRing::new(m4.0, m4.1),
            m5_lwir: MltlRing::new(m5.0, m5.1),
        }
    }

    pub fn reset(&mut self) {
        self.m1_rain.reset();
        self.m2_ice.reset();
        self.m3_glare.reset();
        self.m4_fog.reset();
        self.m5_lwir.reset();
    }

    pub fn evaluate(&mut self, s: &OpticalOddSensors) -> OpticalOddOutput {
        // Mode 1: Raindrop / water-film / bug occlusion
        let raw_m1 = s.droplet_occ_permille > 120 || s.blur_mlap_centi < 5000;

        // Mode 2: Window icing / frost / condensation (AND-gated: MTF + IR opaque + Buck margin)
        let raw_m2 =
            s.mtf_hf_ratio_permille < 140 && s.lwir_std_centidn < 469 && s.dew_margin_centic <= 200;

        // Mode 3: SPA-gated direct solar veiling glare
        let spa_cone_cdeg: u32 = (s.eo_hfov_cdeg as u32 / 2).saturating_add(1200);
        let spa_in_cone = (s.sun_boresight_cdeg as u32) < spa_cone_cdeg;
        let raw_m3 =
            spa_in_cone && (s.glare_sat_permille > 30 || s.glare_ring_ratio_permille < 200);

        // Mode 4: Low-contrast fog / visibility loss (WMO 5% MOR < 1609 m OR horizon SNR < 3.0 dB)
        let raw_m4 = s.mor_m < 1609 || s.horizon_snr_centidb < 300;

        // Mode 5: LWIR bolometer drift / FPN / bad-pixel surge / stale FFC (masked during FFC freeze)
        let raw_m5 = if s.boson_ffc_freeze {
            false
        } else {
            s.lwir_median_drift_dn > 650
                || s.lwir_fpn_ratio_permille > 350
                || s.lwir_bad_px_ppm > 10_000
                || s.boson_dt_ffc_s > 300
                || s.boson_temp_delta_centic > 100
        };

        let (_, p1) = self.m1_rain.update(raw_m1);
        let (_, p2) = self.m2_ice.update(raw_m2);
        let (_, p3) = self.m3_glare.update(raw_m3);
        let (_, p4) = self.m4_fog.update(raw_m4);
        let (_, p5) = self.m5_lwir.update(raw_m5);

        let raw_mask = (raw_m1 as u8)
            | ((raw_m2 as u8) << 1)
            | ((raw_m3 as u8) << 2)
            | ((raw_m4 as u8) << 3)
            | ((raw_m5 as u8) << 4);
        let persisted_mask =
            (p1 as u8) | ((p2 as u8) << 1) | ((p3 as u8) << 2) | ((p4 as u8) << 3) | ((p5 as u8) << 4);

        let ir_healthy = s.lwir_std_centidn > 804 && !p5 && !s.boson_ffc_freeze;
        let action = if p2 || !s.attitude_in_odd || ((p4 || p1) && !ir_healthy) {
            OddAction::DegradedFallback
        } else if (p4 || p1) && ir_healthy {
            OddAction::ThermalPrimary
        } else if p3 {
            OddAction::EvsLwirElevated
        } else {
            OddAction::Nominal
        };

        let vlm_gate_reject = p2 || p3 || p1 || (p4 && !ir_healthy) || !s.attitude_in_odd;

        OpticalOddOutput {
            action,
            raw_mask,
            persisted_mask,
            mask_lwir_frame: s.boson_ffc_freeze,
            vlm_gate_reject,
            ito_heater_cmd: p2,
            airdata_export_blocked: Self::AIRDATA_EXPORT_BLOCKED,
        }
    }
}

// ===========================================================================
// PART 2: DR-09 6-Stage Deterministic Asymmetric Simplex VLM Verification Gate
// ===========================================================================

#[derive(Copy, Clone, Debug)]
struct HazardDebounce<const N: usize> {
    window: [bool; N],
    head: usize,
    filled: usize,
    hits: u8,
    confirm_m: u8,
    release_clear_frames: u8,
    latched: bool,
    clear_streak: u8,
}

impl<const N: usize> HazardDebounce<N> {
    const fn new(confirm_m: u8, release_clear_frames: u8) -> Self {
        Self {
            window: [false; N],
            head: 0,
            filled: 0,
            hits: 0,
            confirm_m,
            release_clear_frames,
            latched: false,
            clear_streak: 0,
        }
    }

    fn reset(&mut self) {
        let mut i = 0;
        while i < N {
            self.window[i] = false;
            i += 1;
        }
        self.head = 0;
        self.filled = 0;
        self.hits = 0;
        self.latched = false;
        self.clear_streak = 0;
    }

    fn update(&mut self, hit: bool) -> bool {
        if self.filled == N {
            if self.window[self.head] {
                self.hits = self.hits.saturating_sub(1);
            }
        } else {
            self.filled += 1;
        }
        self.window[self.head] = hit;
        if hit {
            self.hits = self.hits.saturating_add(1);
        }
        self.head += 1;
        if self.head == N {
            self.head = 0;
        }

        if !self.latched {
            if self.hits >= self.confirm_m {
                self.latched = true;
                self.clear_streak = 0;
            }
        } else if hit {
            self.clear_streak = 0;
        } else {
            self.clear_streak = self.clear_streak.saturating_add(1);
            if self.clear_streak >= self.release_clear_frames {
                self.reset();
            }
        }
        self.latched
    }
}

/// Stage-6 relative-only FLZ candidate ranking output (strictly inside `reachable_flz_mask`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct RankedFlzList {
    pub ids: [u8; NUM_RANKED_ZONES],
    pub count: u8,
}

/// Stage-6 relative-only FLZ ranker: returns ONLY unique zone IDs `< 16` whose bit is set
/// in `reachable_flz_mask` (zero absolute scalar safety scores!).
pub fn rank_flz_relative_only(
    reachable_flz_mask: u16,
    vlm_ranked_ids: [u8; NUM_RANKED_ZONES],
    allow_vlm_order: bool,
) -> RankedFlzList {
    let mut ids = [UNUSED_ZONE_ID; NUM_RANKED_ZONES];
    let mut count: usize = 0;
    let mut used_mask: u16 = 0;

    if allow_vlm_order {
        let mut i = 0;
        while i < NUM_RANKED_ZONES {
            let zid = vlm_ranked_ids[i];
            if zid < MAX_FLZ_ID {
                let bit = 1u16 << (zid as u32);
                if (reachable_flz_mask & bit) != 0 && (used_mask & bit) == 0 && count < NUM_RANKED_ZONES {
                    ids[count] = zid;
                    count += 1;
                    used_mask |= bit;
                }
            }
            i += 1;
        }
    }

    // Deterministic geometric fallback fill for remaining reachable zones
    let mut z: u8 = 0;
    while z < MAX_FLZ_ID && count < NUM_RANKED_ZONES {
        let bit = 1u16 << (z as u32);
        if (reachable_flz_mask & bit) != 0 && (used_mask & bit) == 0 {
            ids[count] = z;
            count += 1;
            used_mask |= bit;
        }
        z += 1;
    }

    RankedFlzList {
        ids,
        count: count as u8,
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct VlmGateDecision {
    pub status: GateStatus,
    pub stage_reached: u8,
    pub vlm_online: bool,
    pub vlm_used: bool,
    pub actuator_authority: bool,
    pub certified_hazards_mask: u8,
    pub accepted_vlm_mask: u8,
    pub final_hazards_mask: u8,
    pub uncertainty_source: UncertaintySource,
    pub ranked_flz: RankedFlzList,
}

/// DR-09 6-Stage DO-178C DAL B/C Verification Gate isolating an IDAL-D/E VLM peripheral.
#[derive(Copy, Clone, Debug)]
pub struct AsymmetricSimplexVlmGate {
    max_age_ms: u64,
    has_last_seq: bool,
    last_seq: u32,
    db_runway: HazardDebounce<3>,
    db_bird: HazardDebounce<5>,
    db_virga: HazardDebounce<5>,
    db_crane: HazardDebounce<5>,
    db_flz: HazardDebounce<5>,
}

impl AsymmetricSimplexVlmGate {
    /// ASTM F3269-21 §1.2 hard invariant: IDAL-D/E VLM peripheral NEVER holds actuator authority.
    pub const ACTUATOR_AUTHORITY: bool = false;

    pub const fn new(max_age_ms: u64) -> Self {
        Self {
            max_age_ms,
            has_last_seq: false,
            last_seq: 0,
            db_runway: HazardDebounce::new(2, 4),
            db_bird: HazardDebounce::new(3, 5),
            db_virga: HazardDebounce::new(3, 5),
            db_crane: HazardDebounce::new(3, 5),
            db_flz: HazardDebounce::new(3, 5),
        }
    }

    pub fn reset(&mut self) {
        self.has_last_seq = false;
        self.last_seq = 0;
        self.clear_debounce();
    }

    fn clear_debounce(&mut self) {
        self.db_runway.reset();
        self.db_bird.reset();
        self.db_virga.reset();
        self.db_crane.reset();
        self.db_flz.reset();
    }

    #[inline]
    fn active_vlm_mask(&self) -> u8 {
        let mut mask = 0u8;
        if self.db_runway.latched {
            mask |= VlmHazard::RunwayIncursion.bit();
        }
        if self.db_bird.latched {
            mask |= VlmHazard::BirdFlock.bit();
        }
        if self.db_virga.latched {
            mask |= VlmHazard::ConvectiveVirga.bit();
        }
        if self.db_crane.latched {
            mask |= VlmHazard::CraneProxy.bit();
        }
        if self.db_flz.latched {
            mask |= VlmHazard::FlzHazard.bit();
        }
        mask
    }

    #[inline]
    const fn classify_uncertainty(aleatoric: u8, epistemic: u8) -> UncertaintySource {
        match (aleatoric != 0, epistemic != 0) {
            (false, false) => UncertaintySource::None,
            (true, false) => UncertaintySource::Aleatoric,
            (false, true) => UncertaintySource::Epistemic,
            (true, true) => UncertaintySource::Both,
        }
    }

    /// Evaluate a raw 32-byte wire frame through all 6 sequential deterministic stages.
    pub fn evaluate_wire(
        &mut self,
        raw: Option<&[u8; PACKET_SIZE]>,
        now_ms: u64,
        optical_odd_ok: bool,
        cert_hazards_mask: u8,
        reachable_flz_mask: u16,
    ) -> VlmGateDecision {
        let fallback_flz = rank_flz_relative_only(reachable_flz_mask, [UNUSED_ZONE_ID; 4], false);
        let raw_bytes = match raw {
            Some(b) => b,
            None => {
                self.clear_debounce();
                return VlmGateDecision {
                    status: GateStatus::VlmOffline,
                    stage_reached: 1,
                    vlm_online: false,
                    vlm_used: false,
                    actuator_authority: Self::ACTUATOR_AUTHORITY,
                    certified_hazards_mask: cert_hazards_mask,
                    accepted_vlm_mask: 0,
                    final_hazards_mask: cert_hazards_mask,
                    uncertainty_source: UncertaintySource::None,
                    ranked_flz: fallback_flz,
                };
            }
        };

        let pkt = match VlmWirePacket::decode_and_verify(raw_bytes) {
            Ok(p) => p,
            Err(_) => {
                let active = self.active_vlm_mask();
                return VlmGateDecision {
                    status: GateStatus::SchemaOrCrcFail,
                    stage_reached: 2,
                    vlm_online: self.has_last_seq,
                    vlm_used: false,
                    actuator_authority: Self::ACTUATOR_AUTHORITY,
                    certified_hazards_mask: cert_hazards_mask,
                    accepted_vlm_mask: active,
                    final_hazards_mask: cert_hazards_mask | active,
                    uncertainty_source: UncertaintySource::None,
                    ranked_flz: fallback_flz,
                };
            }
        };

        self.evaluate_packet(&pkt, now_ms, optical_odd_ok, cert_hazards_mask, reachable_flz_mask)
    }

    /// Evaluate a decoded `VlmWirePacket` through Stages 1–6 (used by both runtime & Kani proofs).
    pub fn evaluate_packet(
        &mut self,
        pkt: &VlmWirePacket,
        now_ms: u64,
        optical_odd_ok: bool,
        cert_hazards_mask: u8,
        reachable_flz_mask: u16,
    ) -> VlmGateDecision {
        let fallback_flz = rank_flz_relative_only(reachable_flz_mask, [UNUSED_ZONE_ID; 4], false);

        // Stage 1: Watchdog freshness (0 <= now - capture <= max_age_ms) & strict monotonic seq
        let fresh = now_ms >= pkt.capture_ts_ms && (now_ms - pkt.capture_ts_ms) <= self.max_age_ms;
        let monotonic = !self.has_last_seq || pkt.seq > self.last_seq;
        if !fresh || !monotonic {
            self.clear_debounce();
            return VlmGateDecision {
                status: GateStatus::VlmOffline,
                stage_reached: 1,
                vlm_online: false,
                vlm_used: false,
                actuator_authority: Self::ACTUATOR_AUTHORITY,
                certified_hazards_mask: cert_hazards_mask,
                accepted_vlm_mask: 0,
                final_hazards_mask: cert_hazards_mask,
                uncertainty_source: UncertaintySource::None,
                ranked_flz: fallback_flz,
            };
        }

        // Stage 2: Static schema & range validation
        if !pkt.schema_valid() {
            let active = self.active_vlm_mask();
            return VlmGateDecision {
                status: GateStatus::SchemaOrCrcFail,
                stage_reached: 2,
                vlm_online: self.has_last_seq,
                vlm_used: false,
                actuator_authority: Self::ACTUATOR_AUTHORITY,
                certified_hazards_mask: cert_hazards_mask,
                accepted_vlm_mask: active,
                final_hazards_mask: cert_hazards_mask | active,
                uncertainty_source: UncertaintySource::None,
                ranked_flz: fallback_flz,
            };
        }

        // Commit monotonic seq only after Stages 1 & 2 succeed
        self.has_last_seq = true;
        self.last_seq = pkt.seq;
        let uncertainty = Self::classify_uncertainty(pkt.aleatoric_flag, pkt.epistemic_flag);

        // Stage 3: Deterministic classical Optical-ODD mask
        if !optical_odd_ok {
            self.clear_debounce();
            return VlmGateDecision {
                status: GateStatus::OpticalOddMasked,
                stage_reached: 3,
                vlm_online: true,
                vlm_used: false,
                actuator_authority: Self::ACTUATOR_AUTHORITY,
                certified_hazards_mask: cert_hazards_mask,
                accepted_vlm_mask: 0,
                final_hazards_mask: cert_hazards_mask,
                uncertainty_source: uncertainty,
                ranked_flz: fallback_flz,
            };
        }

        // Stage 4: Conformal abstention (NEVER treated as hazard clear) + RA-08 split
        if pkt.conformal_valid == 0 {
            let active = self.active_vlm_mask();
            return VlmGateDecision {
                status: GateStatus::ConformalAbstain,
                stage_reached: 4,
                vlm_online: true,
                vlm_used: false,
                actuator_authority: Self::ACTUATOR_AUTHORITY,
                certified_hazards_mask: cert_hazards_mask,
                accepted_vlm_mask: active,
                final_hazards_mask: cert_hazards_mask | active,
                uncertainty_source: uncertainty,
                ranked_flz: fallback_flz,
            };
        }

        // Stage 5: Bounded M-of-N debounce & asymmetric consecutive-clear hysteresis
        let reported = match VlmHazard::from_u8(pkt.hazard_enum) {
            Some(h) => h,
            None => VlmHazard::None,
        };
        let r_latched = self.db_runway.update(reported == VlmHazard::RunwayIncursion);
        let b_latched = self.db_bird.update(reported == VlmHazard::BirdFlock);
        let v_latched = self.db_virga.update(reported == VlmHazard::ConvectiveVirga);
        let c_latched = self.db_crane.update(reported == VlmHazard::CraneProxy);
        let f_latched = self.db_flz.update(reported == VlmHazard::FlzHazard);

        let active_vlm = self.active_vlm_mask();
        // Strict Monotonic Pessimistic Union (bitwise OR: mathematically impossible to clear cert_hazards)
        let final_mask = cert_hazards_mask | active_vlm;

        let reported_latched = match reported {
            VlmHazard::None => true,
            VlmHazard::RunwayIncursion => r_latched,
            VlmHazard::BirdFlock => b_latched,
            VlmHazard::ConvectiveVirga => v_latched,
            VlmHazard::CraneProxy => c_latched,
            VlmHazard::FlzHazard => f_latched,
        };

        if !reported_latched {
            return VlmGateDecision {
                status: GateStatus::DebouncePending,
                stage_reached: 5,
                vlm_online: true,
                vlm_used: false,
                actuator_authority: Self::ACTUATOR_AUTHORITY,
                certified_hazards_mask: cert_hazards_mask,
                accepted_vlm_mask: active_vlm,
                final_hazards_mask: final_mask,
                uncertainty_source: uncertainty,
                ranked_flz: fallback_flz,
            };
        }

        // Stage 6: Relative-only FLZ candidate ranking strictly inside reachable_flz_mask
        let ranked = rank_flz_relative_only(reachable_flz_mask, pkt.ranked_zone_ids, true);
        VlmGateDecision {
            status: GateStatus::NominalAdvisory,
            stage_reached: 6,
            vlm_online: true,
            vlm_used: true,
            actuator_authority: Self::ACTUATOR_AUTHORITY,
            certified_hazards_mask: cert_hazards_mask,
            accepted_vlm_mask: active_vlm,
            final_hazards_mask: final_mask,
            uncertainty_source: uncertainty,
            ranked_flz: ranked,
        }
    }
}

// ===========================================================================
// Unit Tests (`cargo test`) — Including Bit-Exact Python Wire & CRC32 Parity
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wire_packet_bit_exact_python_parity_and_crc32() {
        // Golden vector produced by Python `xvs.v2.xvs_vlm_gate.VlmAdvisoryPacket.pack()`
        let expected_bytes: [u8; PACKET_SIZE] = [
            88, 86, 1, 0, 42, 0, 0, 0, 123, 104, 229, 207, 139, 1, 0, 0, 1, 11, 7, 2, 5, 255, 1, 0,
            1, 0, 0, 0, 173, 124, 70, 216,
        ];
        let pkt = VlmWirePacket {
            seq: 42,
            capture_ts_ms: 1_700_000_000_123,
            hazard_enum: VlmHazard::RunwayIncursion as u8,
            azimuth_sector: 11,
            ranked_zone_ids: [7, 2, 5, UNUSED_ZONE_ID],
            conformal_valid: 1,
            aleatoric_flag: 0,
            epistemic_flag: 1,
            crc32: 0xd846_7cad,
        };
        assert_eq!(pkt.encode(), expected_bytes);
        let decoded = VlmWirePacket::decode_and_verify(&expected_bytes);
        assert_eq!(decoded, Ok(pkt));

        // Flip 1 bit -> IEEE CRC32 rejects cleanly
        let mut corrupt = expected_bytes;
        corrupt[10] ^= 0x01;
        assert_eq!(
            VlmWirePacket::decode_and_verify(&corrupt),
            Err(GateStatus::SchemaOrCrcFail)
        );
    }

    #[test]
    fn test_vlm_gate_6_stages_debounce_and_pessimistic_union() {
        let mut gate = AsymmetricSimplexVlmGate::new(200);
        let cert_mask = 0b0010_0100u8;

        // 1st single-frame RunwayIncursion -> DebouncePending
        let p1 = VlmWirePacket {
            seq: 1,
            capture_ts_ms: 1000,
            hazard_enum: VlmHazard::RunwayIncursion as u8,
            azimuth_sector: 3,
            ranked_zone_ids: [9, 2, 1, UNUSED_ZONE_ID],
            conformal_valid: 1,
            aleatoric_flag: 0,
            epistemic_flag: 0,
            crc32: 0,
        };
        let d1 = gate.evaluate_wire(Some(&p1.encode()), 1050, true, cert_mask, 0b0000_0000_0000_0111);
        assert_eq!(d1.status, GateStatus::DebouncePending);
        assert_eq!(d1.accepted_vlm_mask, 0);
        assert_eq!(d1.final_hazards_mask, cert_mask);
        assert!(!d1.actuator_authority);

        // 2nd RunwayIncursion within 3-frame window -> confirmed (2-of-3) + relative FLZ ranking
        let p2 = VlmWirePacket { seq: 2, capture_ts_ms: 1200, ..p1 };
        let d2 = gate.evaluate_wire(Some(&p2.encode()), 1250, true, cert_mask, 0b0000_0000_0000_0111);
        assert_eq!(d2.status, GateStatus::NominalAdvisory);
        assert_eq!(d2.accepted_vlm_mask, VlmHazard::RunwayIncursion.bit());
        assert_eq!(d2.final_hazards_mask, cert_mask | VlmHazard::RunwayIncursion.bit());
        // Zone 9 dropped (unreachable), VLM order (2, 1) preserved, zone 0 appended as fallback
        assert_eq!(d2.ranked_flz.ids, [2, 1, 0, UNUSED_ZONE_ID]);
        assert_eq!(d2.ranked_flz.count, 3);

        // 500 ms GPU hang -> VlmOffline, zero perturbation to certified geometric hazards
        let p3 = VlmWirePacket { seq: 3, capture_ts_ms: 1200, ..p1 };
        let d3 = gate.evaluate_wire(Some(&p3.encode()), 1700, true, cert_mask, 0b111);
        assert_eq!(d3.status, GateStatus::VlmOffline);
        assert_eq!(d3.accepted_vlm_mask, 0);
        assert_eq!(d3.final_hazards_mask, cert_mask);
    }

    #[test]
    fn test_optical_odd_fog_thermal_primary_and_af447_invariant() {
        let mut mon = OpticalOddMonitor::<3, 3, 3, 3, 3>::new_with_specs(
            (2, 1),
            (2, 1),
            (2, 1),
            (2, 1),
            (2, 1),
        );
        let fog_healthy_ir = OpticalOddSensors {
            droplet_occ_permille: 0,
            blur_mlap_centi: 15_000,
            mtf_hf_ratio_permille: 400,
            lwir_std_centidn: 2000,
            dew_margin_centic: 600,
            sun_boresight_cdeg: 9000,
            eo_hfov_cdeg: 6000,
            glare_sat_permille: 0,
            glare_ring_ratio_permille: 800,
            mor_m: 1200,
            horizon_snr_centidb: 150,
            lwir_median_drift_dn: 20,
            lwir_fpn_ratio_permille: 50,
            lwir_bad_px_ppm: 100,
            boson_dt_ffc_s: 40,
            boson_temp_delta_centic: 20,
            boson_ffc_freeze: false,
            attitude_in_odd: true,
        };
        let _ = mon.evaluate(&fog_healthy_ir);
        let out = mon.evaluate(&fog_healthy_ir);
        assert_eq!(out.action, OddAction::ThermalPrimary);
        assert!(out.airdata_export_blocked);
        assert!(OpticalOddMonitor::<30, 90, 30, 60, 60>::AIRDATA_EXPORT_BLOCKED);
    }
}

// ===========================================================================
// 5 REAL WSL2 `cargo kani` Formal Bounded Model-Checking Proofs
// ===========================================================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    fn any_packet() -> VlmWirePacket {
        VlmWirePacket {
            seq: kani::any(),
            capture_ts_ms: kani::any(),
            hazard_enum: kani::any(),
            azimuth_sector: kani::any(),
            ranked_zone_ids: kani::any(),
            conformal_valid: kani::any(),
            aleatoric_flag: kani::any(),
            epistemic_flag: kani::any(),
            crc32: kani::any(),
        }
    }

    /// Proof 1: Monotonic Pessimistic Union + Zero Actuator Authority hold for ALL symbolic inputs.
    #[kani::proof]
    fn proof_vlm_monotonic_pessimistic_union_and_zero_actuator_auth() {
        let mut gate = AsymmetricSimplexVlmGate::new(200);
        let cert_hazards_mask: u8 = kani::any();
        let reachable_flz_mask: u16 = kani::any();
        let now_ms: u64 = kani::any();
        let optical_odd_ok: bool = kani::any();
        let pkt = any_packet();

        let out = gate.evaluate_packet(
            &pkt,
            now_ms,
            optical_odd_ok,
            cert_hazards_mask,
            reachable_flz_mask,
        );

        assert!((out.final_hazards_mask & cert_hazards_mask) == cert_hazards_mask);
        assert!(!out.actuator_authority);
    }

    /// Proof 2: Watchdog strictly forces `VlmOffline` & `accepted_vlm_mask == 0` on stale/future/replay.
    #[kani::proof]
    fn proof_vlm_watchdog_enforces_offline_on_stale_or_non_monotonic() {
        let mut gate = AsymmetricSimplexVlmGate::new(200);
        let init_seq: u32 = kani::any();
        let init_ts: u64 = kani::any();
        kani::assume(init_ts <= u64::MAX - 200);

        // Prime gate with one valid packet so `last_seq = init_seq`
        let seed_pkt = VlmWirePacket {
            seq: init_seq,
            capture_ts_ms: init_ts,
            hazard_enum: VlmHazard::None as u8,
            azimuth_sector: 0,
            ranked_zone_ids: [UNUSED_ZONE_ID; 4],
            conformal_valid: 1,
            aleatoric_flag: 0,
            epistemic_flag: 0,
            crc32: 0,
        };
        let _ = gate.evaluate_packet(&seed_pkt, init_ts, true, 0, 0);

        // Symbolic next packet violating freshness OR monotonicity
        let pkt = any_packet();
        let now_ms: u64 = kani::any();
        let cert_mask: u8 = kani::any();
        let stale_or_future = now_ms < pkt.capture_ts_ms || (now_ms - pkt.capture_ts_ms) > 200;
        let non_monotonic = pkt.seq <= init_seq;
        kani::assume(stale_or_future || non_monotonic);

        let out = gate.evaluate_packet(&pkt, now_ms, true, cert_mask, 0);
        assert!(out.status == GateStatus::VlmOffline);
        assert!(out.accepted_vlm_mask == 0);
        assert!(out.final_hazards_mask == cert_mask);
        assert!(!out.actuator_authority);
    }

    /// Proof 3: Optical-ODD fault or Conformal Abstention (`conformal_valid == 0`) blocks unverified VLM veto
    /// while preserving all certified geometric hazards.
    #[kani::proof]
    fn proof_vlm_optical_odd_or_conformal_abstain_preserves_cert_hazards() {
        let mut gate = AsymmetricSimplexVlmGate::new(200);
        let cert_mask: u8 = kani::any();
        let now_ms: u64 = kani::any();
        let optical_odd_ok: bool = kani::any();
        let pkt = any_packet();

        kani::assume(pkt.schema_valid());
        kani::assume(now_ms >= pkt.capture_ts_ms && (now_ms - pkt.capture_ts_ms) <= 200);
        kani::assume(!optical_odd_ok || pkt.conformal_valid == 0);

        let out = gate.evaluate_packet(&pkt, now_ms, optical_odd_ok, cert_mask, 0);

        if !optical_odd_ok {
            assert!(out.status == GateStatus::OpticalOddMasked);
            assert!(out.accepted_vlm_mask == 0);
        } else {
            assert!(out.status == GateStatus::ConformalAbstain);
            assert!(out.accepted_vlm_mask == 0);
        }
        assert!((out.final_hazards_mask & cert_mask) == cert_mask);
        assert!(!out.vlm_used);
        assert!(!out.actuator_authority);
    }

    /// Proof 4: AF447 airdata lock holds for ALL symbolic 5-mode readings (zero overflow/panic),
    /// and persistent severe fog (`MOR < 1609 m`, healthy LWIR, clear window/attitude) yields `ThermalPrimary`.
    #[kani::proof]
    fn proof_optical_odd_af447_and_fog_thermal_handoff() {
        // Part A: Completely unconstrained symbolic reading -> ALWAYS airdata_export_blocked, zero panic
        let mut mon_any = OpticalOddMonitor::<30, 90, 30, 60, 60>::new_canonical_60hz();
        let sym = OpticalOddSensors {
            droplet_occ_permille: kani::any(),
            blur_mlap_centi: kani::any(),
            mtf_hf_ratio_permille: kani::any(),
            lwir_std_centidn: kani::any(),
            dew_margin_centic: kani::any(),
            sun_boresight_cdeg: kani::any(),
            eo_hfov_cdeg: kani::any(),
            glare_sat_permille: kani::any(),
            glare_ring_ratio_permille: kani::any(),
            mor_m: kani::any(),
            horizon_snr_centidb: kani::any(),
            lwir_median_drift_dn: kani::any(),
            lwir_fpn_ratio_permille: kani::any(),
            lwir_bad_px_ppm: kani::any(),
            boson_dt_ffc_s: kani::any(),
            boson_temp_delta_centic: kani::any(),
            boson_ffc_freeze: kani::any(),
            attitude_in_odd: kani::any(),
        };
        let out_any = mon_any.evaluate(&sym);
        assert!(out_any.airdata_export_blocked);

        // Part B: Persistent severe fog (MOR < 1609 m) + healthy LWIR -> ThermalPrimary
        let mut mon_fog = OpticalOddMonitor::<2, 2, 2, 2, 2>::new_with_specs(
            (2, 1),
            (2, 1),
            (2, 1),
            (2, 1),
            (2, 1),
        );
        let mor_fog: u32 = kani::any();
        let lwir_healthy_std: u32 = kani::any();
        kani::assume(mor_fog < 1609);
        kani::assume(lwir_healthy_std > 804);

        let fog_s = OpticalOddSensors {
            droplet_occ_permille: 0,
            blur_mlap_centi: 12_000,
            mtf_hf_ratio_permille: 400,
            lwir_std_centidn: lwir_healthy_std,
            dew_margin_centic: 600,
            sun_boresight_cdeg: 9000,
            eo_hfov_cdeg: 6000,
            glare_sat_permille: 0,
            glare_ring_ratio_permille: 800,
            mor_m: mor_fog,
            horizon_snr_centidb: 200,
            lwir_median_drift_dn: 0,
            lwir_fpn_ratio_permille: 0,
            lwir_bad_px_ppm: 0,
            boson_dt_ffc_s: 30,
            boson_temp_delta_centic: 10,
            boson_ffc_freeze: false,
            attitude_in_odd: true,
        };
        let _ = mon_fog.evaluate(&fog_s);
        let out_fog = mon_fog.evaluate(&fog_s);
        assert!(out_fog.airdata_export_blocked);
        assert!(out_fog.action == OddAction::ThermalPrimary);
    }

    /// Proof 5: Relative-only FLZ ranking returns ONLY zones strictly inside `reachable_mask`
    /// with ZERO duplicates across all symbolic `reachable_mask: u16` and `ranked_ids: [u8; 4]`.
    #[kani::proof]
    fn proof_flz_relative_ranking_subset_and_unique() {
        let reachable_mask: u16 = kani::any();
        let ranked_ids: [u8; 4] = kani::any();
        let allow_vlm: bool = kani::any();

        let res = rank_flz_relative_only(reachable_mask, ranked_ids, allow_vlm);
        assert!(res.count <= 4);

        let mut seen_mask: u16 = 0;
        let mut i: usize = 0;
        while i < 4 {
            if i < (res.count as usize) {
                let zid = res.ids[i];
                assert!(zid < MAX_FLZ_ID);
                let bit = 1u16 << (zid as u32);
                assert!((reachable_mask & bit) != 0);
                assert!((seen_mask & bit) == 0);
                seen_mask |= bit;
            } else {
                assert!(res.ids[i] == UNUSED_ZONE_ID);
            }
            i += 1;
        }
    }
}
