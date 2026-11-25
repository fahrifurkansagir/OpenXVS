//! OpenXVS-OS V3.0 (`DR-16`) `#![no_std]` `#![forbid(unsafe_code)]` Zero-Dependency Crate:
//! - Canonical 64-byte `#[repr(C)]` `Ring1AdvisoryPacket` + pure-`core` IEEE 802.3 CRC32
//! - Deterministic `PluginSupervisorState` with permanent mid-flight `PluginOffline` latching
//!   and Monotonic Pessimistic Union (`out_mode = max(base_mode, tightened_mode)`)
//! - Strict 5-class civil `CivilMissionClass` & fail-closed `PluginManifestPolicy` validator
//! - Universal civil `AutopilotHal` trait with unforgeable `FirmwareGateToken`, Fixed-Wing
//!   safety locks, AF447 airdata ban, and `<= 18,432 B/s` (`< 20%` of 921.6 kbps) UART guard.
//!
//! Strictly civil/dual-use unmanned aviation safety (`Apache-2.0 OR MIT`).
//! ZERO heap allocations, ZERO `unsafe`, ZERO external crates, ZERO actuator authority in Ring-1.

#![no_std]
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub const RING1_ADVISORY_MAGIC_VER: u32 = 0x5856_5303; // "XVS\x03" (OpenXVS ABI v3)
pub const RING1_ADVISORY_SIZE: usize = 64;
pub const RING1_ADVISORY_BODY_SIZE: usize = 60;
pub const DEFAULT_MAX_HEARTBEAT_MISS_MS: u64 = 200;

/// 921,600 baud 8N1 = 92,160 B/s; OpenXVS < 20% serial mandate caps link at 18,432 B/s.
pub const UART_921K_CEILING_BPS: u32 = 92_160;
pub const MAX_UART_921K_20PCT_BPS: u32 = 18_432;
pub const MAVLINK2_OVERHEAD_UNSIGNED_B: u32 = 12;
pub const MAVLINK2_OVERHEAD_SIGNED_B: u32 = 25;

// ---------------------------------------------------------------------------
// Pure-`core` IEEE 802.3 CRC32 (Const 256-entry LUT — exact match to zlib.crc32)
// ---------------------------------------------------------------------------

const fn make_crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i: usize = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut bit = 0;
        while bit < 8 {
            if (c & 1) != 0 {
                c = 0xEDB8_8320 ^ (c >> 1);
            } else {
                c >>= 1;
            }
            bit += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}

const CRC32_TABLE: [u32; 256] = make_crc32_table();

/// Pure-`core`, zero-`unsafe` IEEE 802.3 CRC32 (matches Python `zlib.crc32` bit-for-bit).
pub const fn ieee_crc32(bytes: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    let mut i: usize = 0;
    while i < bytes.len() {
        let idx = ((crc ^ (bytes[i] as u32)) & 0xFF) as usize;
        crc = CRC32_TABLE[idx] ^ (crc >> 8);
        i += 1;
    }
    !crc
}

// ---------------------------------------------------------------------------
// 1. Canonical 64-Byte `#[repr(C)]` Ring-1 Advisory Contract (Zero Actuator Fields)
// ---------------------------------------------------------------------------

/// Strictly civil mission classes permitted in OpenXVS-OS V3.0 Ring-1 plugins (DR-16).
#[repr(u16)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CivilMissionClass {
    WildfireThermal = 1,
    SearchAndRescue = 2,
    MaritimeEnvironmental = 3,
    InfrastructureInspection = 4,
    MedicalCargoLogistics = 5,
}

impl CivilMissionClass {
    #[inline]
    pub const fn from_u16(raw: u16) -> Option<Self> {
        match raw {
            1 => Some(Self::WildfireThermal),
            2 => Some(Self::SearchAndRescue),
            3 => Some(Self::MaritimeEnvironmental),
            4 => Some(Self::InfrastructureInspection),
            5 => Some(Self::MedicalCargoLogistics),
            _ => None,
        }
    }
}

/// Canonical 64-byte `#[repr(C)]` Ring-1 -> Ring-0 advisory wire contract (little-endian).
///
/// Field layout is naturally 8-byte aligned with ZERO hidden compiler padding:
/// - `0..4`:   `magic_ver` (`u32`, `0x5856_5303`)
/// - `4..6`:   `civil_mission_class` (`u16`, `CivilMissionClass` `1..=5`)
/// - `6..8`:   `advisory_flags` (`u16`)
/// - `8..16`:  `plugin_id_hash` (`u64`)
/// - `16..24`: `seq_no` (`u64`)
/// - `24..32`: `timestamp_mono_ns` (`u64`)
/// - `32..36`: `odd_tighten_mask` (`u32`, monotonic OR-only safety tightening mask)
/// - `36..40`: `roi_lat_e7` (`i32`)
/// - `40..44`: `roi_lon_e7` (`i32`)
/// - `44..48`: `roi_alt_mm` (`i32`)
/// - `48..50`: `confidence_q15` (`u16`)
/// - `50..60`: `reserved_zero` (`[u8; 10]`, structurally zero — ZERO actuator fields!)
/// - `60..64`: `ieee_crc32` (`u32`, IEEE 802.3 CRC32 over bytes `0..60`)
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Ring1AdvisoryPacket {
    pub magic_ver: u32,
    pub civil_mission_class: u16,
    pub advisory_flags: u16,
    pub plugin_id_hash: u64,
    pub seq_no: u64,
    pub timestamp_mono_ns: u64,
    pub odd_tighten_mask: u32,
    pub roi_lat_e7: i32,
    pub roi_lon_e7: i32,
    pub roi_alt_mm: i32,
    pub confidence_q15: u16,
    pub reserved_zero: [u8; 10],
    pub ieee_crc32: u32,
}

const _: () = assert!(core::mem::size_of::<Ring1AdvisoryPacket>() == RING1_ADVISORY_SIZE);
const _: () = assert!(core::mem::align_of::<Ring1AdvisoryPacket>() == 8);

impl Ring1AdvisoryPacket {
    /// Serialize the 60-byte CRC-covered body in explicit little-endian wire order.
    pub fn body_to_le_bytes(&self) -> [u8; RING1_ADVISORY_BODY_SIZE] {
        let mut out = [0u8; RING1_ADVISORY_BODY_SIZE];
        let b0 = self.magic_ver.to_le_bytes();
        out[0] = b0[0];
        out[1] = b0[1];
        out[2] = b0[2];
        out[3] = b0[3];

        let b1 = self.civil_mission_class.to_le_bytes();
        out[4] = b1[0];
        out[5] = b1[1];

        let b2 = self.advisory_flags.to_le_bytes();
        out[6] = b2[0];
        out[7] = b2[1];

        let b3 = self.plugin_id_hash.to_le_bytes();
        out[8] = b3[0];
        out[9] = b3[1];
        out[10] = b3[2];
        out[11] = b3[3];
        out[12] = b3[4];
        out[13] = b3[5];
        out[14] = b3[6];
        out[15] = b3[7];

        let b4 = self.seq_no.to_le_bytes();
        out[16] = b4[0];
        out[17] = b4[1];
        out[18] = b4[2];
        out[19] = b4[3];
        out[20] = b4[4];
        out[21] = b4[5];
        out[22] = b4[6];
        out[23] = b4[7];

        let b5 = self.timestamp_mono_ns.to_le_bytes();
        out[24] = b5[0];
        out[25] = b5[1];
        out[26] = b5[2];
        out[27] = b5[3];
        out[28] = b5[4];
        out[29] = b5[5];
        out[30] = b5[6];
        out[31] = b5[7];

        let b6 = self.odd_tighten_mask.to_le_bytes();
        out[32] = b6[0];
        out[33] = b6[1];
        out[34] = b6[2];
        out[35] = b6[3];

        let b7 = self.roi_lat_e7.to_le_bytes();
        out[36] = b7[0];
        out[37] = b7[1];
        out[38] = b7[2];
        out[39] = b7[3];

        let b8 = self.roi_lon_e7.to_le_bytes();
        out[40] = b8[0];
        out[41] = b8[1];
        out[42] = b8[2];
        out[43] = b8[3];

        let b9 = self.roi_alt_mm.to_le_bytes();
        out[44] = b9[0];
        out[45] = b9[1];
        out[46] = b9[2];
        out[47] = b9[3];

        let b10 = self.confidence_q15.to_le_bytes();
        out[48] = b10[0];
        out[49] = b10[1];

        let mut i = 0;
        while i < 10 {
            out[50 + i] = self.reserved_zero[i];
            i += 1;
        }
        out
    }

    /// Compute IEEE 802.3 CRC32 over the first 60 little-endian wire bytes.
    pub fn compute_crc32(&self) -> u32 {
        let body = self.body_to_le_bytes();
        ieee_crc32(&body)
    }

    /// Return a copy with `ieee_crc32` populated from the first 60 bytes.
    pub fn with_valid_crc(mut self) -> Self {
        self.ieee_crc32 = self.compute_crc32();
        self
    }

    /// Serialize full 64-byte little-endian wire packet (including `ieee_crc32`).
    pub fn to_le_bytes(&self) -> [u8; RING1_ADVISORY_SIZE] {
        let body = self.body_to_le_bytes();
        let mut out = [0u8; RING1_ADVISORY_SIZE];
        let mut i = 0;
        while i < RING1_ADVISORY_BODY_SIZE {
            out[i] = body[i];
            i += 1;
        }
        let crc_b = self.ieee_crc32.to_le_bytes();
        out[60] = crc_b[0];
        out[61] = crc_b[1];
        out[62] = crc_b[2];
        out[63] = crc_b[3];
        out
    }

    /// Deserialize 64 little-endian wire bytes without panicking.
    pub fn from_le_bytes(raw: &[u8; RING1_ADVISORY_SIZE]) -> Self {
        Self {
            magic_ver: u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]),
            civil_mission_class: u16::from_le_bytes([raw[4], raw[5]]),
            advisory_flags: u16::from_le_bytes([raw[6], raw[7]]),
            plugin_id_hash: u64::from_le_bytes([
                raw[8], raw[9], raw[10], raw[11], raw[12], raw[13], raw[14], raw[15],
            ]),
            seq_no: u64::from_le_bytes([
                raw[16], raw[17], raw[18], raw[19], raw[20], raw[21], raw[22], raw[23],
            ]),
            timestamp_mono_ns: u64::from_le_bytes([
                raw[24], raw[25], raw[26], raw[27], raw[28], raw[29], raw[30], raw[31],
            ]),
            odd_tighten_mask: u32::from_le_bytes([raw[32], raw[33], raw[34], raw[35]]),
            roi_lat_e7: i32::from_le_bytes([raw[36], raw[37], raw[38], raw[39]]),
            roi_lon_e7: i32::from_le_bytes([raw[40], raw[41], raw[42], raw[43]]),
            roi_alt_mm: i32::from_le_bytes([raw[44], raw[45], raw[46], raw[47]]),
            confidence_q15: u16::from_le_bytes([raw[48], raw[49]]),
            reserved_zero: [
                raw[50], raw[51], raw[52], raw[53], raw[54], raw[55], raw[56], raw[57], raw[58],
                raw[59],
            ],
            ieee_crc32: u32::from_le_bytes([raw[60], raw[61], raw[62], raw[63]]),
        }
    }

    #[inline]
    pub fn is_reserved_all_zero(&self) -> bool {
        let mut i = 0;
        while i < 10 {
            if self.reserved_zero[i] != 0 {
                return false;
            }
            i += 1;
        }
        true
    }
}

// ---------------------------------------------------------------------------
// 2. PluginSupervisorState & Deterministic Permanent PLUGIN_OFFLINE Latching
// ---------------------------------------------------------------------------

/// Lifecycle state matching DR-16 `POD_PLUGIN_HEALTH (#50004)` wire enum.
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PluginLifecycle {
    Init = 0,
    Healthy = 1,
    Degraded = 2,
    PluginOffline = 3,
}

/// Simplex restriction severity ordered monotonically from least to most conservative:
/// `NominalMl (0) < Isolated5Cam (1) < DegradedCoast (2) < FallbackD0Safe (3) < InvalidInhibit (4)`.
///
/// Monotonic Pessimistic Union computes `max(base_mode, tightened_mode)` so Ring-1 advisories
/// can ONLY tighten safety posture and NEVER lower `SimplexMode`!
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum SimplexMode {
    NominalMl = 0,
    Isolated5Cam = 1,
    DegradedCoast = 2,
    FallbackD0Safe = 3,
    InvalidInhibit = 4,
}

impl SimplexMode {
    /// Monotonic Pessimistic Union: returns `max(self, tightened)` (NEVER relaxes!).
    #[inline]
    pub const fn tighten(self, tightened: Self) -> Self {
        if (tightened as u8) > (self as u8) {
            tightened
        } else {
            self
        }
    }

    /// Map Ring-1 `odd_tighten_mask` severity bits to a minimum restriction floor.
    #[inline]
    pub const fn from_odd_tighten_mask(mask: u32) -> Self {
        if (mask & 0xFFFF_FF00) != 0 {
            Self::InvalidInhibit
        } else if (mask & 0x0000_00F0) != 0 {
            Self::FallbackD0Safe
        } else if (mask & 0x0000_000C) != 0 {
            Self::DegradedCoast
        } else if (mask & 0x0000_0003) != 0 {
            Self::Isolated5Cam
        } else {
            Self::NominalMl
        }
    }

    /// Map severity rank to canonical `xvs-rta-core` wire `#[repr(u8)]` (`4 - rank`).
    #[inline]
    pub const fn to_rta_wire_u8(self) -> u8 {
        4u8.saturating_sub(self as u8)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SupervisorFault {
    None,
    AlreadyLatchedOffline,
    InvalidMagic,
    InvalidMissionClass,
    NonZeroReservedBytes,
    Crc32Mismatch,
    NonMonotonicSeq,
    HeartbeatStaleOrFuture,
    PluginCrashed,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SupervisorDecision {
    pub lifecycle: PluginLifecycle,
    pub fault: SupervisorFault,
    pub actuator_authority: bool,
    pub out_mode: SimplexMode,
    pub out_odd_mask: u32,
}

/// Ring-0 supervisor gating a single Ring-1 plugin advisory channel.
/// Any integrity, freshness, replay, or crash fault permanently latches `PluginOffline`.
#[derive(Copy, Clone, Debug)]
pub struct PluginSupervisorState {
    max_heartbeat_miss_ns: u64,
    lifecycle: PluginLifecycle,
    has_seq: bool,
    last_seq_no: u64,
    has_ts: bool,
    last_mono_ns: u64,
    missed_heartbeats: u16,
}

impl PluginSupervisorState {
    /// Compile-time Ring-1 safety lock: Ring-1 plugins NEVER hold actuator authority.
    pub const ACTUATOR_AUTHORITY: bool = false;

    pub const fn new(max_heartbeat_miss_ms: u64) -> Self {
        let clamped_ms = if max_heartbeat_miss_ms == 0 {
            DEFAULT_MAX_HEARTBEAT_MISS_MS
        } else {
            max_heartbeat_miss_ms
        };
        Self {
            max_heartbeat_miss_ns: clamped_ms.saturating_mul(1_000_000),
            lifecycle: PluginLifecycle::Init,
            has_seq: false,
            last_seq_no: 0,
            has_ts: false,
            last_mono_ns: 0,
            missed_heartbeats: 0,
        }
    }

    #[inline]
    pub const fn lifecycle(&self) -> PluginLifecycle {
        self.lifecycle
    }

    #[inline]
    pub const fn missed_heartbeats(&self) -> u16 {
        self.missed_heartbeats
    }

    /// Permanently latch `PluginOffline` upon OS crash / OOM kill / seccomp signal.
    pub fn signal_crash(&mut self) {
        self.lifecycle = PluginLifecycle::PluginOffline;
        self.missed_heartbeats = self.missed_heartbeats.saturating_add(1);
    }

    fn latch_offline(
        &mut self,
        fault: SupervisorFault,
        base_mode: SimplexMode,
        base_odd_mask: u32,
    ) -> SupervisorDecision {
        self.lifecycle = PluginLifecycle::PluginOffline;
        self.missed_heartbeats = self.missed_heartbeats.saturating_add(1);
        SupervisorDecision {
            lifecycle: PluginLifecycle::PluginOffline,
            fault,
            actuator_authority: Self::ACTUATOR_AUTHORITY,
            out_mode: base_mode,
            out_odd_mask: base_odd_mask,
        }
    }

    /// Verify a 64-byte raw wire buffer and apply Monotonic Pessimistic Union.
    pub fn ingest_wire(
        &mut self,
        raw: &[u8; RING1_ADVISORY_SIZE],
        now_mono_ns: u64,
        base_mode: SimplexMode,
        base_odd_mask: u32,
    ) -> SupervisorDecision {
        let pkt = Ring1AdvisoryPacket::from_le_bytes(raw);
        self.ingest_advisory(&pkt, now_mono_ns, base_mode, base_odd_mask)
    }

    /// Verify a `Ring1AdvisoryPacket` (magic, civil class, reserved zero, IEEE-CRC32,
    /// monotonic `seq_no`, heartbeat freshness) and apply Monotonic Pessimistic Union.
    pub fn ingest_advisory(
        &mut self,
        pkt: &Ring1AdvisoryPacket,
        now_mono_ns: u64,
        base_mode: SimplexMode,
        base_odd_mask: u32,
    ) -> SupervisorDecision {
        // Permanent mid-flight latch: once PluginOffline, NEVER auto-clears!
        if self.lifecycle == PluginLifecycle::PluginOffline {
            return SupervisorDecision {
                lifecycle: PluginLifecycle::PluginOffline,
                fault: SupervisorFault::AlreadyLatchedOffline,
                actuator_authority: Self::ACTUATOR_AUTHORITY,
                out_mode: base_mode,
                out_odd_mask: base_odd_mask,
            };
        }

        if pkt.magic_ver != RING1_ADVISORY_MAGIC_VER {
            return self.latch_offline(SupervisorFault::InvalidMagic, base_mode, base_odd_mask);
        }
        if CivilMissionClass::from_u16(pkt.civil_mission_class).is_none() {
            return self.latch_offline(
                SupervisorFault::InvalidMissionClass,
                base_mode,
                base_odd_mask,
            );
        }
        if !pkt.is_reserved_all_zero() {
            return self.latch_offline(
                SupervisorFault::NonZeroReservedBytes,
                base_mode,
                base_odd_mask,
            );
        }
        if pkt.ieee_crc32 != pkt.compute_crc32() {
            return self.latch_offline(SupervisorFault::Crc32Mismatch, base_mode, base_odd_mask);
        }

        // Strict monotonic sequence check (blocks replay and out-of-order frames)
        if self.has_seq && pkt.seq_no <= self.last_seq_no {
            return self.latch_offline(SupervisorFault::NonMonotonicSeq, base_mode, base_odd_mask);
        }

        // Watchdog freshness check: packet age and inter-packet delta <= max_heartbeat_miss_ns
        let pkt_fresh = now_mono_ns >= pkt.timestamp_mono_ns
            && (now_mono_ns - pkt.timestamp_mono_ns) <= self.max_heartbeat_miss_ns;
        let hb_fresh = !self.has_ts
            || (now_mono_ns >= self.last_mono_ns
                && (now_mono_ns - self.last_mono_ns) <= self.max_heartbeat_miss_ns);
        if !pkt_fresh || !hb_fresh {
            return self.latch_offline(
                SupervisorFault::HeartbeatStaleOrFuture,
                base_mode,
                base_odd_mask,
            );
        }

        self.has_seq = true;
        self.last_seq_no = pkt.seq_no;
        self.has_ts = true;
        self.last_mono_ns = now_mono_ns;
        self.lifecycle = if pkt.odd_tighten_mask == 0 {
            PluginLifecycle::Healthy
        } else {
            PluginLifecycle::Degraded
        };

        let tightened_mode = SimplexMode::from_odd_tighten_mask(pkt.odd_tighten_mask);
        let out_mode = base_mode.tighten(tightened_mode);
        let out_odd_mask = base_odd_mask | pkt.odd_tighten_mask;

        SupervisorDecision {
            lifecycle: self.lifecycle,
            fault: SupervisorFault::None,
            actuator_authority: Self::ACTUATOR_AUTHORITY,
            out_mode,
            out_odd_mask,
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Signed `plugin.toml` Capability Manifest Policy Validator
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ManifestRejectReason {
    InvalidCivilMissionClass,
    MissingCivilExportAttestation,
    UnverifiedSignature,
    ActuatorAuthorityForbidden,
    AdvisorySizeExceeded,
    HeartbeatMissBudgetInvalid,
    SwapForbidden,
    OrinGpuAccessForbidden,
    Ring0WriteTopicForbidden,
    NetworkAccessForbidden,
    InvalidResourceQuota,
}

/// Parsed Ring-1 `plugin.toml` security & resource manifest evaluated prior to `fork()/execve()`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PluginManifestPolicy {
    pub mission_class_raw: u16,
    pub non_itar_civil_only: bool,
    pub ed25519_sig_verified: bool,
    pub actuator_authority: bool,
    pub max_advisory_bytes: u16,
    pub max_heartbeat_miss_ms: u32,
    pub swap_max_bytes: u64,
    pub orin_requests_gpu: bool,
    pub requests_rw_ring0_topic: bool,
    pub network_enabled: bool,
    pub memory_max_mb: u32,
    pub pids_max: u16,
}

impl PluginManifestPolicy {
    /// Fail-closed policy gate admitting ONLY strictly civil, sandboxed, zero-actuator plugins.
    pub const fn validate(&self) -> Result<CivilMissionClass, ManifestRejectReason> {
        let mission = match CivilMissionClass::from_u16(self.mission_class_raw) {
            Some(c) => c,
            None => return Err(ManifestRejectReason::InvalidCivilMissionClass),
        };
        if !self.non_itar_civil_only {
            return Err(ManifestRejectReason::MissingCivilExportAttestation);
        }
        if !self.ed25519_sig_verified {
            return Err(ManifestRejectReason::UnverifiedSignature);
        }
        if self.actuator_authority {
            return Err(ManifestRejectReason::ActuatorAuthorityForbidden);
        }
        if self.max_advisory_bytes == 0 || (self.max_advisory_bytes as usize) > RING1_ADVISORY_SIZE {
            return Err(ManifestRejectReason::AdvisorySizeExceeded);
        }
        if self.max_heartbeat_miss_ms == 0 || (self.max_heartbeat_miss_ms as u64) > DEFAULT_MAX_HEARTBEAT_MISS_MS {
            return Err(ManifestRejectReason::HeartbeatMissBudgetInvalid);
        }
        if self.swap_max_bytes > 0 {
            return Err(ManifestRejectReason::SwapForbidden);
        }
        if self.orin_requests_gpu {
            return Err(ManifestRejectReason::OrinGpuAccessForbidden);
        }
        if self.requests_rw_ring0_topic {
            return Err(ManifestRejectReason::Ring0WriteTopicForbidden);
        }
        if self.network_enabled {
            return Err(ManifestRejectReason::NetworkAccessForbidden);
        }
        if self.memory_max_mb == 0 || self.pids_max == 0 || self.pids_max > 64 {
            return Err(ManifestRejectReason::InvalidResourceQuota);
        }
        Ok(mission)
    }
}

// ---------------------------------------------------------------------------
// 4. Universal Civil `AutopilotHal` Trait, Unforgeable `FirmwareGateToken` & UART Budget
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AirframeClass {
    FixedWing,
    QuadPlaneVtol,
    Multicopter,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AutopilotStack {
    ArduPlane { major: u8, minor: u8, patch: u8 },
    Px4 { major: u8, minor: u8, patch: u8 },
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum HalError {
    FirmwareGateFailed,
    InvalidAirframeCmd,
    LinkSaturated,
    AirdataExportBanned,
    TransportFault,
}

/// Unforgeable proof token minted ONLY after verifying FC firmware version & EKF alignment.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct FirmwareGateToken {
    _private: (),
}

/// Verify firmware minimum version (`ArduPlane >= 4.7.2` for Issue #32506; `PX4 >= 1.15.0`)
/// and EKF source alignment before minting an unforgeable `FirmwareGateToken`.
pub const fn verify_firmware_and_ekf_gate(
    stack: AutopilotStack,
    ekf_source_aligned: bool,
) -> Result<FirmwareGateToken, HalError> {
    if !ekf_source_aligned {
        return Err(HalError::FirmwareGateFailed);
    }
    let version_ok = match stack {
        AutopilotStack::ArduPlane {
            major,
            minor,
            patch,
        } => {
            major > 4
                || (major == 4 && minor > 7)
                || (major == 4 && minor == 7 && patch >= 2)
        }
        AutopilotStack::Px4 {
            major,
            minor,
            patch: _,
        } => major > 1 || (major == 1 && minor >= 15),
    };
    if version_ok {
        Ok(FirmwareGateToken { _private: () })
    } else {
        Err(HalError::FirmwareGateFailed)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct CanonicalOdom {
    pub timestamp_us: u64,
    pub pos_ned_mm: [i32; 3],
    pub vel_frd_mm_s: [i32; 3],
    pub quat_q15: [i16; 4],
    pub reset_counter: u8,
}

/// Airframe-typed avoidance escape command from Ring-0 `xvs-rta-core`.
/// On `FixedWing`, `pitch_direct` and `throttle_direct` MUST be `None` (PX4 `NaN`/unset),
/// and velocity-only or multicopter-only setpoints are strictly forbidden!
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AvoidanceVector {
    pub target_course_cdeg: i32,
    pub target_alt_mm: i32,
    pub target_eas_cm_s: u16,
    pub is_velocity_only: bool,
    pub is_multicopter_only: bool,
    pub pitch_direct_cdeg: Option<i16>,
    pub throttle_direct_permille: Option<u16>,
}

impl AvoidanceVector {
    /// Validate airframe kinematic & safety constraints (DR-16 ArduPlane + PX4 traps).
    pub const fn validate_for_airframe(&self, airframe: AirframeClass) -> Result<(), HalError> {
        match airframe {
            AirframeClass::FixedWing => {
                if self.is_velocity_only
                    || self.is_multicopter_only
                    || self.pitch_direct_cdeg.is_some()
                    || self.throttle_direct_permille.is_some()
                    || self.target_eas_cm_s == 0
                {
                    return Err(HalError::InvalidAirframeCmd);
                }
                Ok(())
            }
            AirframeClass::QuadPlaneVtol | AirframeClass::Multicopter => {
                if self.pitch_direct_cdeg.is_some() || self.throttle_direct_permille.is_some() {
                    return Err(HalError::InvalidAirframeCmd);
                }
                Ok(())
            }
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct CanonicalIntruder {
    pub track_id: u16,
    pub rel_north_m: i32,
    pub rel_east_m: i32,
    pub rel_down_m: i32,
    pub closure_cm_s: i32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct OddState32B {
    pub health_bitmask: u8,
    pub airdata_export_blocked: bool,
    pub ieee_crc32: u32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct VlmVeto32B {
    pub hazard_mask: u8,
    pub actuator_authority: bool,
    pub ieee_crc32: u32,
}

/// Canonical 48-byte `#[repr(C)]` MAVLink2 `POD_PLUGIN_HEALTH (#50004)` wire struct.
///
/// Exact MAVLink2 descending-size wire order (`<QQQIHHHBB12B`, zero hidden padding, 8-byte aligned):
/// - `0..8`:   `plugin_id_hash` (`u64`)
/// - `8..16`:  `timestamp_mono_us` (`u64`)
/// - `16..24`: `last_seq_no` (`u64`)
/// - `24..28`: `ieee_crc32` (`u32`, IEEE 802.3 CRC32 over the 44 non-CRC bytes `[0..24] || [28..48]`)
/// - `28..30`: `missed_heartbeats` (`u16`)
/// - `30..32`: `rss_mb` (`u16`)
/// - `32..34`: `cpu_permille` (`u16`)
/// - `34`:     `lifecycle_state` (`u8`, `0=INIT, 1=HEALTHY, 2=DEGRADED, 3=PLUGIN_OFFLINE`)
/// - `35`:     `civil_mission_class` (`u8`, `1..=5`, `0` or `> 5` fail-closed rejected)
/// - `36..48`: `reserved_zero` (`[u8; 12]`, all `0x00`, zero actuator fields)
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PluginHealth48B {
    pub plugin_id_hash: u64,
    pub timestamp_mono_us: u64,
    pub last_seq_no: u64,
    pub ieee_crc32: u32,
    pub missed_heartbeats: u16,
    pub rss_mb: u16,
    pub cpu_permille: u16,
    pub lifecycle_state: u8,
    pub civil_mission_class: u8,
    pub reserved_zero: [u8; 12],
}

const _: () = assert!(core::mem::size_of::<PluginHealth48B>() == 48);
const _: () = assert!(core::mem::align_of::<PluginHealth48B>() == 8);

impl PluginHealth48B {
    /// Serialize the 44 non-CRC wire bytes `[0..24] || [28..48]` in little-endian order.
    pub fn non_crc_le_bytes(&self) -> [u8; 44] {
        let mut out = [0u8; 44];
        let b0 = self.plugin_id_hash.to_le_bytes();
        out[0] = b0[0];
        out[1] = b0[1];
        out[2] = b0[2];
        out[3] = b0[3];
        out[4] = b0[4];
        out[5] = b0[5];
        out[6] = b0[6];
        out[7] = b0[7];

        let b1 = self.timestamp_mono_us.to_le_bytes();
        out[8] = b1[0];
        out[9] = b1[1];
        out[10] = b1[2];
        out[11] = b1[3];
        out[12] = b1[4];
        out[13] = b1[5];
        out[14] = b1[6];
        out[15] = b1[7];

        let b2 = self.last_seq_no.to_le_bytes();
        out[16] = b2[0];
        out[17] = b2[1];
        out[18] = b2[2];
        out[19] = b2[3];
        out[20] = b2[4];
        out[21] = b2[5];
        out[22] = b2[6];
        out[23] = b2[7];

        let b3 = self.missed_heartbeats.to_le_bytes();
        out[24] = b3[0];
        out[25] = b3[1];

        let b4 = self.rss_mb.to_le_bytes();
        out[26] = b4[0];
        out[27] = b4[1];

        let b5 = self.cpu_permille.to_le_bytes();
        out[28] = b5[0];
        out[29] = b5[1];

        out[30] = self.lifecycle_state;
        out[31] = self.civil_mission_class;

        let mut i = 0;
        while i < 12 {
            out[32 + i] = self.reserved_zero[i];
            i += 1;
        }
        out
    }

    /// Compute IEEE 802.3 CRC32 over the 44 non-CRC wire bytes (`[0..24] || [28..48]`).
    pub fn compute_crc32(&self) -> u32 {
        let body = self.non_crc_le_bytes();
        ieee_crc32(&body)
    }

    /// Return a copy with `ieee_crc32` populated from the 44 non-CRC wire bytes.
    pub fn with_valid_crc(mut self) -> Self {
        self.ieee_crc32 = self.compute_crc32();
        self
    }

    /// Serialize full 48-byte MAVLink2 `<QQQIHHHBB12B` wire payload.
    pub fn to_le_bytes(&self) -> [u8; 48] {
        let non_crc = self.non_crc_le_bytes();
        let mut out = [0u8; 48];
        let mut i = 0;
        while i < 24 {
            out[i] = non_crc[i];
            i += 1;
        }
        let crc_b = self.ieee_crc32.to_le_bytes();
        out[24] = crc_b[0];
        out[25] = crc_b[1];
        out[26] = crc_b[2];
        out[27] = crc_b[3];
        let mut j = 0;
        while j < 20 {
            out[28 + j] = non_crc[24 + j];
            j += 1;
        }
        out
    }

    /// Decode and fail-closed validate a 48-byte `POD_PLUGIN_HEALTH (#50004)` payload.
    /// Rejects `lifecycle_state > 3`, `civil_mission_class` outside `1..=5`, non-zero
    /// `reserved_zero`, or IEEE-802.3 CRC32 mismatch.
    pub fn from_le_bytes_verified(bytes: &[u8; 48]) -> Result<Self, HalError> {
        let pkt = Self {
            plugin_id_hash: u64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ]),
            timestamp_mono_us: u64::from_le_bytes([
                bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14],
                bytes[15],
            ]),
            last_seq_no: u64::from_le_bytes([
                bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22],
                bytes[23],
            ]),
            ieee_crc32: u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]),
            missed_heartbeats: u16::from_le_bytes([bytes[28], bytes[29]]),
            rss_mb: u16::from_le_bytes([bytes[30], bytes[31]]),
            cpu_permille: u16::from_le_bytes([bytes[32], bytes[33]]),
            lifecycle_state: bytes[34],
            civil_mission_class: bytes[35],
            reserved_zero: [
                bytes[36], bytes[37], bytes[38], bytes[39], bytes[40], bytes[41], bytes[42],
                bytes[43], bytes[44], bytes[45], bytes[46], bytes[47],
            ],
        };

        if pkt.lifecycle_state > 3 {
            return Err(HalError::InvalidAirframeCmd);
        }
        if CivilMissionClass::from_u16(pkt.civil_mission_class as u16).is_none() {
            return Err(HalError::InvalidAirframeCmd);
        }
        let mut i = 0;
        while i < 12 {
            if pkt.reserved_zero[i] != 0 {
                return Err(HalError::InvalidAirframeCmd);
            }
            i += 1;
        }
        if pkt.ieee_crc32 != pkt.compute_crc32() {
            return Err(HalError::TransportFault);
        }
        Ok(pkt)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct CanonicalVehicleState {
    pub stack: AutopilotStack,
    pub ekf_source_aligned: bool,
    pub armed: bool,
}

/// MAVLink2 serial link budget schedule on 921.6 kbps 8N1 UART (`<= 18,432 B/s` = `< 20%`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Mavlink2UartSchedule {
    pub odom_hz: u16,
    pub daa_summary_hz: u16,
    pub odd_state_hz: u16,
    pub vlm_veto_hz: u16,
    pub plugin_health_hz: u16,
    pub command_int_hz: u16,
    pub adsb_tracks_hz: u16,
    pub signed_mavlink2: bool,
}

impl Mavlink2UartSchedule {
    /// Canonical DR-16 V3.0 full stream (30 Hz ODOM, 20 Hz DAA+ODD, 10 Hz VLM, 5 Hz plugin+CMD, 20 Hz ADSB).
    pub const fn canonical_v3(signed_mavlink2: bool) -> Self {
        Self {
            odom_hz: 30,
            daa_summary_hz: 20,
            odd_state_hz: 20,
            vlm_veto_hz: 10,
            plugin_health_hz: 5,
            command_int_hz: 5,
            adsb_tracks_hz: 20, // 10 tracks @ 2 Hz
            signed_mavlink2,
        }
    }

    /// Compute exact wire bytes/second and enforce `<= 18,432 B/s` (`< 20%` of 921.6 kbps).
    pub const fn checked_bytes_per_sec(&self) -> Result<u32, HalError> {
        let ovh = if self.signed_mavlink2 {
            MAVLINK2_OVERHEAD_SIGNED_B
        } else {
            MAVLINK2_OVERHEAD_UNSIGNED_B
        };
        // Payload sizes per MAVLink2 / pod_daa.xml spec:
        // ODOMETRY(#331)=232B, POD_DAA(#50001)=22B, POD_ODD(#50002)=32B,
        // POD_VLM(#50003)=32B, POD_PLUGIN_HEALTH(#50004)=48B,
        // COMMAND_INT(#75)=35B, ADSB_VEHICLE(#246)=38B
        let b_odom = (self.odom_hz as u32).saturating_mul(232 + ovh);
        let b_daa = (self.daa_summary_hz as u32).saturating_mul(22 + ovh);
        let b_odd = (self.odd_state_hz as u32).saturating_mul(32 + ovh);
        let b_vlm = (self.vlm_veto_hz as u32).saturating_mul(32 + ovh);
        let b_plg = (self.plugin_health_hz as u32).saturating_mul(48 + ovh);
        let b_cmd = (self.command_int_hz as u32).saturating_mul(35 + ovh);
        let b_adsb = (self.adsb_tracks_hz as u32).saturating_mul(38 + ovh);

        let total = b_odom
            .saturating_add(b_daa)
            .saturating_add(b_odd)
            .saturating_add(b_vlm)
            .saturating_add(b_plg)
            .saturating_add(b_cmd)
            .saturating_add(b_adsb);

        if total > MAX_UART_921K_20PCT_BPS {
            Err(HalError::LinkSaturated)
        } else {
            Ok(total)
        }
    }
}

/// Strictly civil, actuator-free autopilot HAL (Apache-2.0 OR MIT).
/// ZERO methods exist for direct PWM, servo, throttle, or non-civil payload release.
pub trait AutopilotHal {
    fn verify_firmware_and_ekf_gate(&mut self) -> Result<FirmwareGateToken, HalError>;

    /// ArduPlane: `#331 ODOMETRY` (gated `>= 4.7.2`). PX4: `/fmu/in/vehicle_visual_odometry`.
    fn send_visual_odometry(
        &mut self,
        gate: &FirmwareGateToken,
        odom: &CanonicalOdom,
    ) -> Result<(), HalError>;

    /// ArduPlane: `COMMAND_INT (#75)` ONLY (`#84` unrepresentable!).
    /// PX4 FW: `FixedWingLateralSetpoint` + `FixedWingLongitudinalSetpoint` (`pitch/throttle = None`).
    fn send_daa_avoidance_maneuver(
        &mut self,
        airframe: AirframeClass,
        cmd: &AvoidanceVector,
    ) -> Result<(), HalError>;

    fn send_traffic_intruder(&mut self, intruder: &CanonicalIntruder) -> Result<(), HalError>;

    /// Emits `32B` / `48B` CRC32 safety contracts with `AIRDATA_EXPORT_BLOCKED = true`.
    fn send_safety_telemetry(
        &mut self,
        odd: &OddState32B,
        veto: &VlmVeto32B,
        health: &PluginHealth48B,
    ) -> Result<(), HalError>;

    fn poll_vehicle_state(&mut self) -> Option<CanonicalVehicleState>;
}

/// Reference `#![no_std]` mock HAL enforcing every DR-16 safety invariant for unit & Kani tests.
#[derive(Copy, Clone, Debug)]
pub struct ReferenceAutopilotHal {
    pub state: CanonicalVehicleState,
    pub uart_schedule: Mavlink2UartSchedule,
    pub odom_sent: u32,
    pub avoidance_sent: u32,
    pub intruders_sent: u32,
    pub telemetry_sent: u32,
}

impl ReferenceAutopilotHal {
    pub const fn new(state: CanonicalVehicleState, uart_schedule: Mavlink2UartSchedule) -> Self {
        Self {
            state,
            uart_schedule,
            odom_sent: 0,
            avoidance_sent: 0,
            intruders_sent: 0,
            telemetry_sent: 0,
        }
    }
}

impl AutopilotHal for ReferenceAutopilotHal {
    fn verify_firmware_and_ekf_gate(&mut self) -> Result<FirmwareGateToken, HalError> {
        verify_firmware_and_ekf_gate(self.state.stack, self.state.ekf_source_aligned)
    }

    fn send_visual_odometry(
        &mut self,
        _gate: &FirmwareGateToken,
        _odom: &CanonicalOdom,
    ) -> Result<(), HalError> {
        let _ = self.uart_schedule.checked_bytes_per_sec()?;
        self.odom_sent = self.odom_sent.saturating_add(1);
        Ok(())
    }

    fn send_daa_avoidance_maneuver(
        &mut self,
        airframe: AirframeClass,
        cmd: &AvoidanceVector,
    ) -> Result<(), HalError> {
        cmd.validate_for_airframe(airframe)?;
        let _ = self.uart_schedule.checked_bytes_per_sec()?;
        self.avoidance_sent = self.avoidance_sent.saturating_add(1);
        Ok(())
    }

    fn send_traffic_intruder(&mut self, _intruder: &CanonicalIntruder) -> Result<(), HalError> {
        let _ = self.uart_schedule.checked_bytes_per_sec()?;
        self.intruders_sent = self.intruders_sent.saturating_add(1);
        Ok(())
    }

    fn send_safety_telemetry(
        &mut self,
        odd: &OddState32B,
        veto: &VlmVeto32B,
        _health: &PluginHealth48B,
    ) -> Result<(), HalError> {
        if !odd.airdata_export_blocked {
            return Err(HalError::AirdataExportBanned);
        }
        if veto.actuator_authority {
            return Err(HalError::InvalidAirframeCmd);
        }
        let _ = self.uart_schedule.checked_bytes_per_sec()?;
        self.telemetry_sent = self.telemetry_sent.saturating_add(1);
        Ok(())
    }

    fn poll_vehicle_state(&mut self) -> Option<CanonicalVehicleState> {
        Some(self.state)
    }
}

// ===========================================================================
// Unit Tests (`cargo test`)
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_advisory(seq_no: u64, ts_ns: u64, odd_mask: u32) -> Ring1AdvisoryPacket {
        Ring1AdvisoryPacket {
            magic_ver: RING1_ADVISORY_MAGIC_VER,
            civil_mission_class: CivilMissionClass::SearchAndRescue as u16,
            advisory_flags: 0x0001,
            plugin_id_hash: 0xDEAD_BEEF_CAFE_0001,
            seq_no,
            timestamp_mono_ns: ts_ns,
            odd_tighten_mask: odd_mask,
            roi_lat_e7: 376_117_120,
            roi_lon_e7: -1_223_583_492,
            roi_alt_mm: 150_000,
            confidence_q15: 28_000,
            reserved_zero: [0u8; 10],
            ieee_crc32: 0,
        }
        .with_valid_crc()
    }

    #[test]
    fn test_ieee_crc32_standard_check_value_and_64b_roundtrip() {
        // Standard IEEE 802.3 CRC-32 check value for ASCII "123456789" is 0xCBF43926
        assert_eq!(ieee_crc32(b"123456789"), 0xCBF4_3926);

        let pkt = sample_advisory(1, 1_000_000_000, 0x0000_0004);
        let wire = pkt.to_le_bytes();
        assert_eq!(wire.len(), 64);
        let decoded = Ring1AdvisoryPacket::from_le_bytes(&wire);
        assert_eq!(decoded, pkt);
        assert_eq!(decoded.ieee_crc32, decoded.compute_crc32());
    }

    #[test]
    fn test_supervisor_monotonic_union_and_permanent_offline_latch() {
        let mut sup = PluginSupervisorState::new(200);
        let p1 = sample_advisory(1, 1_000_000_000, 0x0000_0004);
        let d1 = sup.ingest_advisory(&p1, 1_050_000_000, SimplexMode::NominalMl, 0x0000_0001);
        assert_eq!(d1.lifecycle, PluginLifecycle::Degraded);
        assert_eq!(d1.fault, SupervisorFault::None);
        assert_eq!(d1.out_mode, SimplexMode::DegradedCoast);
        assert_eq!(d1.out_odd_mask, 0x0000_0005);
        assert!(!d1.actuator_authority);

        // Tamper 1 bit in CRC -> permanent PluginOffline latch
        let mut bad = sample_advisory(2, 1_150_000_000, 0);
        bad.ieee_crc32 ^= 0x01;
        let d2 = sup.ingest_advisory(&bad, 1_180_000_000, SimplexMode::Isolated5Cam, 0x0000_0002);
        assert_eq!(d2.lifecycle, PluginLifecycle::PluginOffline);
        assert_eq!(d2.fault, SupervisorFault::Crc32Mismatch);
        assert_eq!(d2.out_mode, SimplexMode::Isolated5Cam);
        assert_eq!(d2.out_odd_mask, 0x0000_0002);

        // Subsequent perfectly valid packet MUST NOT unlatch PluginOffline mid-flight!
        let p3 = sample_advisory(3, 1_200_000_000, 0x0000_00FF);
        let d3 = sup.ingest_advisory(&p3, 1_210_000_000, SimplexMode::NominalMl, 0);
        assert_eq!(d3.lifecycle, PluginLifecycle::PluginOffline);
        assert_eq!(d3.fault, SupervisorFault::AlreadyLatchedOffline);
    }

    #[test]
    fn test_manifest_policy_strict_civil_sandbox_rules() {
        let valid = PluginManifestPolicy {
            mission_class_raw: CivilMissionClass::WildfireThermal as u16,
            non_itar_civil_only: true,
            ed25519_sig_verified: true,
            actuator_authority: false,
            max_advisory_bytes: 64,
            max_heartbeat_miss_ms: 200,
            swap_max_bytes: 0,
            orin_requests_gpu: false,
            requests_rw_ring0_topic: false,
            network_enabled: false,
            memory_max_mb: 1536,
            pids_max: 32,
        };
        assert_eq!(valid.validate(), Ok(CivilMissionClass::WildfireThermal));

        let mut bad_act = valid;
        bad_act.actuator_authority = true;
        assert_eq!(
            bad_act.validate(),
            Err(ManifestRejectReason::ActuatorAuthorityForbidden)
        );

        let mut bad_gpu = valid;
        bad_gpu.orin_requests_gpu = true;
        assert_eq!(
            bad_gpu.validate(),
            Err(ManifestRejectReason::OrinGpuAccessForbidden)
        );

        let mut bad_swap = valid;
        bad_swap.swap_max_bytes = 4096;
        assert_eq!(bad_swap.validate(), Err(ManifestRejectReason::SwapForbidden));
    }

    #[test]
    fn test_autopilot_hal_firmware_gate_fixedwing_af447_and_uart_budget() {
        // ArduPlane 4.7.1 rejected (Issue #32506); 4.7.3 accepted
        assert_eq!(
            verify_firmware_and_ekf_gate(
                AutopilotStack::ArduPlane {
                    major: 4,
                    minor: 7,
                    patch: 1
                },
                true
            ),
            Err(HalError::FirmwareGateFailed)
        );
        let token = verify_firmware_and_ekf_gate(
            AutopilotStack::ArduPlane {
                major: 4,
                minor: 7,
                patch: 3,
            },
            true,
        );
        assert!(token.is_ok());

        // Canonical V3.0 MAVLink2 budget: unsigned & signed both strictly <= 18,432 B/s (< 20%)
        let unsigned_bps = Mavlink2UartSchedule::canonical_v3(false).checked_bytes_per_sec();
        let signed_bps = Mavlink2UartSchedule::canonical_v3(true).checked_bytes_per_sec();
        assert_eq!(unsigned_bps, Ok(10_855));
        assert_eq!(signed_bps, Ok(12_285));

        // FixedWing rejects velocity-only & direct pitch/throttle setpoints
        let mut hal = ReferenceAutopilotHal::new(
            CanonicalVehicleState {
                stack: AutopilotStack::ArduPlane {
                    major: 4,
                    minor: 7,
                    patch: 3,
                },
                ekf_source_aligned: true,
                armed: true,
            },
            Mavlink2UartSchedule::canonical_v3(true),
        );
        let bad_fw = AvoidanceVector {
            target_course_cdeg: 9000,
            target_alt_mm: 200_000,
            target_eas_cm_s: 3500,
            is_velocity_only: true,
            is_multicopter_only: false,
            pitch_direct_cdeg: None,
            throttle_direct_permille: None,
        };
        assert_eq!(
            hal.send_daa_avoidance_maneuver(AirframeClass::FixedWing, &bad_fw),
            Err(HalError::InvalidAirframeCmd)
        );

        // AF447 airdata export ban enforced in send_safety_telemetry
        let bad_odd = OddState32B {
            health_bitmask: 0,
            airdata_export_blocked: false,
            ieee_crc32: 0,
        };
        let veto = VlmVeto32B {
            hazard_mask: 0,
            actuator_authority: false,
            ieee_crc32: 0,
        };
        let health = PluginHealth48B {
            plugin_id_hash: 1,
            timestamp_mono_us: 1000,
            last_seq_no: 1,
            lifecycle_state: PluginLifecycle::Healthy as u8,
            civil_mission_class: CivilMissionClass::SearchAndRescue as u8,
            missed_heartbeats: 0,
            rss_mb: 128,
            cpu_permille: 250,
            reserved_zero: [0u8; 12],
            ieee_crc32: 0,
        };
        assert_eq!(
            hal.send_safety_telemetry(&bad_odd, &veto, &health),
            Err(HalError::AirdataExportBanned)
        );
    }

    #[test]
    fn test_plugin_health_48b_golden_vector_and_validation() {
        let pkt = PluginHealth48B {
            plugin_id_hash: 0x1122_3344_5566_7788,
            timestamp_mono_us: 1_700_000_000,
            last_seq_no: 42,
            ieee_crc32: 0,
            missed_heartbeats: 0,
            rss_mb: 512,
            cpu_permille: 250,
            lifecycle_state: PluginLifecycle::Healthy as u8,
            civil_mission_class: CivilMissionClass::SearchAndRescue as u8,
            reserved_zero: [0u8; 12],
        }
        .with_valid_crc();

        // Exact cross-language Rust <-> Python MAVLink2 POD_PLUGIN_HEALTH (#50004) golden vector
        assert_eq!(pkt.ieee_crc32, 0x8225_C4FB);
        let expected_wire: [u8; 48] = [
            136, 119, 102, 85, 68, 51, 34, 17, 0, 241, 83, 101, 0, 0, 0, 0, 42, 0, 0, 0, 0, 0, 0,
            0, 251, 196, 37, 130, 0, 0, 0, 2, 250, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let wire = pkt.to_le_bytes();
        assert_eq!(wire, expected_wire);
        assert_eq!(PluginHealth48B::from_le_bytes_verified(&wire), Ok(pkt));

        // Fail-closed rejections: civil_mission_class == 0, == 6, lifecycle > 3, non-zero reserved, bad CRC
        let bad_class_zero = PluginHealth48B {
            civil_mission_class: 0,
            ..pkt
        }
        .with_valid_crc()
        .to_le_bytes();
        assert_eq!(
            PluginHealth48B::from_le_bytes_verified(&bad_class_zero),
            Err(HalError::InvalidAirframeCmd)
        );

        let bad_class_six = PluginHealth48B {
            civil_mission_class: 6,
            ..pkt
        }
        .with_valid_crc()
        .to_le_bytes();
        assert_eq!(
            PluginHealth48B::from_le_bytes_verified(&bad_class_six),
            Err(HalError::InvalidAirframeCmd)
        );

        let bad_lifecycle = PluginHealth48B {
            lifecycle_state: 4,
            ..pkt
        }
        .with_valid_crc()
        .to_le_bytes();
        assert_eq!(
            PluginHealth48B::from_le_bytes_verified(&bad_lifecycle),
            Err(HalError::InvalidAirframeCmd)
        );

        let mut bad_reserved = pkt;
        bad_reserved.reserved_zero[11] = 0x01;
        let bad_res_wire = bad_reserved.with_valid_crc().to_le_bytes();
        assert_eq!(
            PluginHealth48B::from_le_bytes_verified(&bad_res_wire),
            Err(HalError::InvalidAirframeCmd)
        );

        let mut bad_crc = wire;
        bad_crc[24] ^= 0x01;
        assert_eq!(
            PluginHealth48B::from_le_bytes_verified(&bad_crc),
            Err(HalError::TransportFault)
        );
    }
}

// ===========================================================================
// 4 REAL WSL2 `cargo kani` Formal Verification Proofs (`10/10 -> 14/14`)
// ===========================================================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    fn any_advisory() -> Ring1AdvisoryPacket {
        Ring1AdvisoryPacket {
            magic_ver: kani::any(),
            civil_mission_class: kani::any(),
            advisory_flags: kani::any(),
            plugin_id_hash: kani::any(),
            seq_no: kani::any(),
            timestamp_mono_ns: kani::any(),
            odd_tighten_mask: kani::any(),
            roi_lat_e7: kani::any(),
            roi_lon_e7: kani::any(),
            roi_alt_mm: kani::any(),
            confidence_q15: kani::any(),
            reserved_zero: kani::any(),
            ieee_crc32: kani::any(),
        }
    }

    fn any_simplex_mode() -> SimplexMode {
        let v: u8 = kani::any();
        kani::assume(v <= 4);
        match v {
            0 => SimplexMode::NominalMl,
            1 => SimplexMode::Isolated5Cam,
            2 => SimplexMode::DegradedCoast,
            3 => SimplexMode::FallbackD0Safe,
            _ => SimplexMode::InvalidInhibit,
        }
    }

    /// Proof 1: Corrupt CRC / stale heartbeat / replay / non-zero reserved ALWAYS permanently latches
    /// `PluginLifecycle::PluginOffline` (and subsequent packets can NEVER unlatch it), zero panic.
    #[kani::proof]
    fn kani_ring1_advisory_crc_and_offline_latch() {
        let mut sup = PluginSupervisorState::new(DEFAULT_MAX_HEARTBEAT_MISS_MS);

        // Prime supervisor with one known-good packet at t = 1_000_000_000 ns, seq = 10
        let seed = Ring1AdvisoryPacket {
            magic_ver: RING1_ADVISORY_MAGIC_VER,
            civil_mission_class: CivilMissionClass::WildfireThermal as u16,
            advisory_flags: 0,
            plugin_id_hash: 0x1234,
            seq_no: 10,
            timestamp_mono_ns: 1_000_000_000,
            odd_tighten_mask: 0,
            roi_lat_e7: 0,
            roi_lon_e7: 0,
            roi_alt_mm: 0,
            confidence_q15: 0,
            reserved_zero: [0u8; 10],
            ieee_crc32: 0,
        }
        .with_valid_crc();
        let _ = sup.ingest_advisory(&seed, 1_010_000_000, SimplexMode::NominalMl, 0);

        // Symbolic fault injection across all 4 Ring-1 failure modes:
        //   0 = arbitrary non-zero 32-bit CRC corruption
        //   1 = replay / non-monotonic seq_no <= 10
        //   2 = arbitrary stale (> 200 ms) or future monotonic timestamp
        //   3 = non-zero reserved_zero byte (rogue actuator payload injection)
        let fault_kind: u8 = kani::any();
        kani::assume(fault_kind < 4);
        let base_mode = any_simplex_mode();
        let base_odd: u32 = kani::any();

        let mut pkt = Ring1AdvisoryPacket {
            seq_no: 11,
            timestamp_mono_ns: 1_020_000_000,
            ..seed
        }
        .with_valid_crc();
        let mut now_ns: u64 = 1_030_000_000;

        if fault_kind == 0 {
            let corrupt_mask: u32 = kani::any();
            kani::assume(corrupt_mask != 0);
            pkt.ieee_crc32 ^= corrupt_mask;
        } else if fault_kind == 1 {
            let replay_seq: u64 = kani::any();
            kani::assume(replay_seq <= 10);
            pkt.seq_no = replay_seq;
            pkt.ieee_crc32 = pkt.compute_crc32();
        } else if fault_kind == 2 {
            now_ns = kani::any();
            kani::assume(
                now_ns < pkt.timestamp_mono_ns
                    || (now_ns - pkt.timestamp_mono_ns) > 200_000_000
                    || now_ns < 1_010_000_000
                    || (now_ns - 1_010_000_000) > 200_000_000,
            );
        } else {
            let idx: usize = kani::any();
            let val: u8 = kani::any();
            kani::assume(idx < 10 && val != 0);
            pkt.reserved_zero[idx] = val;
        }

        let d1 = sup.ingest_advisory(&pkt, now_ns, base_mode, base_odd);
        assert!(d1.lifecycle == PluginLifecycle::PluginOffline);
        assert!(d1.out_mode == base_mode);
        assert!(d1.out_odd_mask == base_odd);
        assert!(!d1.actuator_authority);

        // Even a subsequent valid packet NEVER unlatches PluginOffline mid-flight
        let recovery_attempt = Ring1AdvisoryPacket {
            seq_no: 12,
            timestamp_mono_ns: 1_040_000_000,
            ..seed
        }
        .with_valid_crc();
        let d2 = sup.ingest_advisory(&recovery_attempt, 1_050_000_000, base_mode, base_odd);
        assert!(d2.lifecycle == PluginLifecycle::PluginOffline);
        assert!(!d2.actuator_authority);
    }

    /// Proof 2: Manifest policy validator NEVER admits `actuator_authority=true`, Orin GPU,
    /// `swap_max_bytes > 0`, RW Ring-0 topic, network access, oversized advisory, or non-civil class.
    #[kani::proof]
    fn kani_manifest_policy_forbids_actuator_and_orin_gpu() {
        let policy = PluginManifestPolicy {
            mission_class_raw: kani::any(),
            non_itar_civil_only: kani::any(),
            ed25519_sig_verified: kani::any(),
            actuator_authority: kani::any(),
            max_advisory_bytes: kani::any(),
            max_heartbeat_miss_ms: kani::any(),
            swap_max_bytes: kani::any(),
            orin_requests_gpu: kani::any(),
            requests_rw_ring0_topic: kani::any(),
            network_enabled: kani::any(),
            memory_max_mb: kani::any(),
            pids_max: kani::any(),
        };

        if let Ok(mission) = policy.validate() {
            assert!(CivilMissionClass::from_u16(policy.mission_class_raw) == Some(mission));
            assert!(policy.non_itar_civil_only);
            assert!(policy.ed25519_sig_verified);
            assert!(!policy.actuator_authority);
            assert!(policy.max_advisory_bytes >= 1 && policy.max_advisory_bytes <= 64);
            assert!(policy.max_heartbeat_miss_ms >= 1 && policy.max_heartbeat_miss_ms <= 200);
            assert!(policy.swap_max_bytes == 0);
            assert!(!policy.orin_requests_gpu);
            assert!(!policy.requests_rw_ring0_topic);
            assert!(!policy.network_enabled);
            assert!(policy.memory_max_mb > 0);
            assert!(policy.pids_max >= 1 && policy.pids_max <= 64);
        }
    }

    /// Proof 3: ArduPlane < 4.7.2 (or PX4 < 1.15) NEVER mints `FirmwareGateToken`;
    /// AF447 airdata violation ALWAYS fails; FixedWing forbids velocity-only/direct-pitch;
    /// and MAVLink2 UART budget > 18,432 B/s ALWAYS returns `Err(HalError::LinkSaturated)`.
    #[kani::proof]
    fn kani_firmware_gate_and_uart_budget() {
        let major: u8 = kani::any();
        let minor: u8 = kani::any();
        let patch: u8 = kani::any();
        let ekf_aligned: bool = kani::any();

        // Part A: ArduPlane < 4.7.2 or unaligned EKF NEVER mints token
        let ap_res = verify_firmware_and_ekf_gate(
            AutopilotStack::ArduPlane {
                major,
                minor,
                patch,
            },
            ekf_aligned,
        );
        let ap_too_old = major < 4 || (major == 4 && minor < 7) || (major == 4 && minor == 7 && patch < 2);
        if ap_too_old || !ekf_aligned {
            assert!(ap_res == Err(HalError::FirmwareGateFailed));
        } else {
            assert!(ap_res.is_ok());
        }

        // Part B: FixedWing rejects velocity-only / direct-pitch / direct-throttle
        let fw_cmd = AvoidanceVector {
            target_course_cdeg: kani::any(),
            target_alt_mm: kani::any(),
            target_eas_cm_s: kani::any(),
            is_velocity_only: kani::any(),
            is_multicopter_only: kani::any(),
            pitch_direct_cdeg: if kani::any() { Some(kani::any()) } else { None },
            throttle_direct_permille: if kani::any() { Some(kani::any()) } else { None },
        };
        if fw_cmd.is_velocity_only
            || fw_cmd.is_multicopter_only
            || fw_cmd.pitch_direct_cdeg.is_some()
            || fw_cmd.throttle_direct_permille.is_some()
            || fw_cmd.target_eas_cm_s == 0
        {
            assert!(fw_cmd.validate_for_airframe(AirframeClass::FixedWing) == Err(HalError::InvalidAirframeCmd));
        }

        // Part C: AF447 airdata export ban & > 18,432 B/s UART saturation guard
        let sched = Mavlink2UartSchedule {
            odom_hz: kani::any(),
            daa_summary_hz: kani::any(),
            odd_state_hz: kani::any(),
            vlm_veto_hz: kani::any(),
            plugin_health_hz: kani::any(),
            command_int_hz: kani::any(),
            adsb_tracks_hz: kani::any(),
            signed_mavlink2: kani::any(),
        };
        if let Ok(bps) = sched.checked_bytes_per_sec() {
            assert!(bps <= MAX_UART_921K_20PCT_BPS);
        }

        let mut hal = ReferenceAutopilotHal::new(
            CanonicalVehicleState {
                stack: AutopilotStack::ArduPlane {
                    major: 4,
                    minor: 7,
                    patch: 3,
                },
                ekf_source_aligned: true,
                armed: true,
            },
            Mavlink2UartSchedule::canonical_v3(true),
        );
        let odd_violation = OddState32B {
            health_bitmask: kani::any(),
            airdata_export_blocked: false,
            ieee_crc32: 0,
        };
        let veto = VlmVeto32B {
            hazard_mask: kani::any(),
            actuator_authority: false,
            ieee_crc32: 0,
        };
        let health = PluginHealth48B {
            plugin_id_hash: 0,
            timestamp_mono_us: 0,
            last_seq_no: 0,
            lifecycle_state: 0,
            civil_mission_class: 1,
            missed_heartbeats: 0,
            rss_mb: 0,
            cpu_permille: 0,
            reserved_zero: [0u8; 12],
            ieee_crc32: 0,
        };
        assert!(hal.send_safety_telemetry(&odd_violation, &veto, &health) == Err(HalError::AirdataExportBanned));
    }

    /// Proof 4: Monotonic Pessimistic Union — for ALL symbolic `base_mode`, `base_odd_mask`,
    /// and symbolic `Ring1AdvisoryPacket`, Ring-1 NEVER lowers `SimplexMode`, NEVER clears any
    /// Ring-0 ODD fault bit, and NEVER grants actuator authority.
    #[kani::proof]
    fn kani_monotonic_pessimistic_union_never_relaxes() {
        let mut sup = PluginSupervisorState::new(DEFAULT_MAX_HEARTBEAT_MISS_MS);
        let base_mode = any_simplex_mode();
        let base_odd_mask: u32 = kani::any();
        let now_ns: u64 = kani::any();
        let pkt = any_advisory();

        let out = sup.ingest_advisory(&pkt, now_ns, base_mode, base_odd_mask);
        assert!(out.out_mode >= base_mode);
        assert!((out.out_odd_mask & base_odd_mask) == base_odd_mask);
        assert!(!out.actuator_authority);
    }
}
