// Copyright 2026 OpenXVS Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Exact 64-byte (`[u8; 64]`) little-endian wire codec for `ActiveRangeTelemetry`
//! with IEEE-802.3 CRC32 over bytes `0..60`.

use crate::NUM_BANKS;

/// Wire size in bytes (exact cache-line aligned 64 bytes, zero padding).
pub const ACTIVE_RANGE_WIRE_BYTES: usize = 64;

/// Telemetry status flag bits (`ActiveRangeTelemetry::flags`).
pub const FLAG_TAU_FUSED: u16 = 1 << 0;
pub const FLAG_KOSCH_FUSED: u16 = 1 << 1;
pub const FLAG_SC3BF_ALL_SAFE: u16 = 1 << 2;
pub const FLAG_FOV_SAFE: u16 = 1 << 3;

/// Operational mode of the `XVS-ActiveRange` controller.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveRangeMode {
    NominalHold = 0,
    ActiveSwerve = 1,
    Locked = 2,
    ImmediateEvade = 3,
}

impl ActiveRangeMode {
    /// Convert a raw wire `u8` into `ActiveRangeMode`, returning `None` if out of range.
    pub const fn from_u8(raw: u8) -> Option<Self> {
        match raw {
            0 => Some(Self::NominalHold),
            1 => Some(Self::ActiveSwerve),
            2 => Some(Self::Locked),
            3 => Some(Self::ImmediateEvade),
            _ => None,
        }
    }
}

/// Exact 64-byte cache-line-aligned telemetry packet (`struct.pack("<IBBH7f6fI", ...)`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActiveRangeTelemetry {
    pub seq: u32,                // bytes  0..4
    pub mode: u8,                // byte   4    (ActiveRangeMode as u8)
    pub active_bank_idx: u8,     // byte   5    (argmax w_m in 0..6)
    pub flags: u16,              // bytes  6..8 (bit0..bit3)
    pub r_hat_m: f32,            // bytes  8..12
    pub sigma_r_m: f32,          // bytes 12..16
    pub r_safe_lower_m: f32,     // bytes 16..20
    pub rho_dot_inv_s: f32,      // bytes 20..24
    pub cmd_bank_rad: f32,       // bytes 24..28
    pub min_sc3bf_margin: f32,   // bytes 28..32
    pub fim_gain: f32,           // bytes 32..36
    pub weights: [f32; NUM_BANKS], // bytes 36..60
    pub crc32: u32,              // bytes 60..64 (IEEE-802.3 CRC32 over bytes 0..60)
}

const _: () = {
    assert!(core::mem::size_of::<ActiveRangeTelemetry>() == ACTIVE_RANGE_WIRE_BYTES);
    assert!(core::mem::align_of::<ActiveRangeTelemetry>() == 4);
};

/// Compute standard IEEE-802.3 CRC32 (`zlib.crc32` compatible; check value `0xCBF43926`).
/// Uses branchless bitwise polynomial division (`0xEDB8_8320`) for compact `#![no_std]` footprint
/// and tractable Kani/CBMC SAT solving (avoids 256-way symbolic LUT multiplexers).
pub fn ieee_crc32(bytes: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    let mut i = 0usize;
    while i < bytes.len() {
        crc ^= bytes[i] as u32;
        let mut bit = 0u8;
        while bit < 8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
            bit += 1;
        }
        i += 1;
    }
    !crc
}

/// Payload length in bytes preceding the 4-byte IEEE-802.3 CRC32 trailer.
pub const ACTIVE_RANGE_PAYLOAD_BYTES: usize = 60;

#[inline]
fn write_u32_le(buf: &mut [u8], offset: usize, val: u32) {
    let b = val.to_le_bytes();
    buf[offset] = b[0];
    buf[offset + 1] = b[1];
    buf[offset + 2] = b[2];
    buf[offset + 3] = b[3];
}

#[inline]
fn write_f32_le(buf: &mut [u8], offset: usize, val: f32) {
    let b = val.to_le_bytes();
    buf[offset] = b[0];
    buf[offset + 1] = b[1];
    buf[offset + 2] = b[2];
    buf[offset + 3] = b[3];
}

#[inline]
fn read_u32_le(buf: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([buf[offset], buf[offset + 1], buf[offset + 2], buf[offset + 3]])
}

#[inline]
fn read_f32_le(buf: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes([buf[offset], buf[offset + 1], buf[offset + 2], buf[offset + 3]])
}

/// Encode the 60-byte little-endian payload preceding the IEEE-802.3 CRC32 trailer.
pub fn encode_payload(t: &ActiveRangeTelemetry) -> [u8; ACTIVE_RANGE_PAYLOAD_BYTES] {
    let mut out = [0u8; ACTIVE_RANGE_PAYLOAD_BYTES];
    write_u32_le(&mut out, 0, t.seq);
    out[4] = t.mode;
    out[5] = t.active_bank_idx;
    let fb = t.flags.to_le_bytes();
    out[6] = fb[0];
    out[7] = fb[1];
    write_f32_le(&mut out, 8, t.r_hat_m);
    write_f32_le(&mut out, 12, t.sigma_r_m);
    write_f32_le(&mut out, 16, t.r_safe_lower_m);
    write_f32_le(&mut out, 20, t.rho_dot_inv_s);
    write_f32_le(&mut out, 24, t.cmd_bank_rad);
    write_f32_le(&mut out, 28, t.min_sc3bf_margin);
    write_f32_le(&mut out, 32, t.fim_gain);
    let mut m = 0usize;
    while m < NUM_BANKS {
        write_f32_le(&mut out, 36 + 4 * m, t.weights[m]);
        m += 1;
    }
    out
}

/// Decode and validate the 60-byte payload (rejects invalid `mode`, `active_bank_idx`, or non-finite floats).
pub fn decode_payload(
    payload: &[u8; ACTIVE_RANGE_PAYLOAD_BYTES],
    wire_crc: u32,
) -> Option<ActiveRangeTelemetry> {
    let mode = payload[4];
    ActiveRangeMode::from_u8(mode)?;
    let active_bank_idx = payload[5];
    if (active_bank_idx as usize) >= NUM_BANKS {
        return None;
    }

    let seq = read_u32_le(payload, 0);
    let flags = u16::from_le_bytes([payload[6], payload[7]]);
    let r_hat_m = read_f32_le(payload, 8);
    let sigma_r_m = read_f32_le(payload, 12);
    let r_safe_lower_m = read_f32_le(payload, 16);
    let rho_dot_inv_s = read_f32_le(payload, 20);
    let cmd_bank_rad = read_f32_le(payload, 24);
    let min_sc3bf_margin = read_f32_le(payload, 28);
    let fim_gain = read_f32_le(payload, 32);

    if !(r_hat_m.is_finite()
        && sigma_r_m.is_finite()
        && r_safe_lower_m.is_finite()
        && rho_dot_inv_s.is_finite()
        && cmd_bank_rad.is_finite()
        && min_sc3bf_margin.is_finite()
        && fim_gain.is_finite())
    {
        return None;
    }

    let mut weights = [0.0f32; NUM_BANKS];
    let mut m = 0usize;
    while m < NUM_BANKS {
        let w = read_f32_le(payload, 36 + 4 * m);
        if !w.is_finite() {
            return None;
        }
        weights[m] = w;
        m += 1;
    }

    Some(ActiveRangeTelemetry {
        seq,
        mode,
        active_bank_idx,
        flags,
        r_hat_m,
        sigma_r_m,
        r_safe_lower_m,
        rho_dot_inv_s,
        cmd_bank_rad,
        min_sc3bf_margin,
        fim_gain,
        weights,
        crc32: wire_crc,
    })
}

/// Encode `ActiveRangeTelemetry` into an exact 64-byte little-endian frame with IEEE CRC32 over `0..60`.
pub fn encode_wire(t: &ActiveRangeTelemetry) -> [u8; ACTIVE_RANGE_WIRE_BYTES] {
    let payload = encode_payload(t);
    let mut out = [0u8; ACTIVE_RANGE_WIRE_BYTES];
    let mut i = 0usize;
    while i < ACTIVE_RANGE_PAYLOAD_BYTES {
        out[i] = payload[i];
        i += 1;
    }
    let crc = ieee_crc32(&payload);
    write_u32_le(&mut out, 60, crc);
    out
}

/// Decode and validate a 64-byte little-endian `ActiveRangeTelemetry` wire frame.
///
/// Rejects CRC32 mismatch, invalid `mode` (`> 3`), invalid `active_bank_idx` (`>= 6`),
/// or any non-finite (`NaN`/`Inf`) floating-point field.
pub fn decode_wire(raw: &[u8; ACTIVE_RANGE_WIRE_BYTES]) -> Option<ActiveRangeTelemetry> {
    let mut payload = [0u8; ACTIVE_RANGE_PAYLOAD_BYTES];
    let mut i = 0usize;
    while i < ACTIVE_RANGE_PAYLOAD_BYTES {
        payload[i] = raw[i];
        i += 1;
    }
    let expected_crc = ieee_crc32(&payload);
    let wire_crc = read_u32_le(raw, 60);
    if wire_crc != expected_crc {
        return None;
    }
    decode_payload(&payload, wire_crc)
}
