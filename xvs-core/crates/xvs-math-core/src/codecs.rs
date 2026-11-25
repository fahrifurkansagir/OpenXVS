//! Bit-exact `#![no_std]` FAA GDL-90 Public ICD Rev A (`#0`, `#10`, `#11`, `#20`,
//! CRC-16-CCITT `0x1021`, `0x7E`/`0x7D` byte-stuffing, optional `SIM` callsign prefix guard)
//! and MAVLink v2 (`#246 ADSB_VEHICLE` with `CRC_EXTRA = 184` + `#52001 OPENXVS_BEARING_TRACK`
//! 77-byte SHA-256 signed frame), matching `src/xvs/avionics/{gdl90.py,mavlink_adsb.py}` bit-for-bit.

use crate::det_f64::{clamp, round_ties_even};
use crate::mesh_daa::{OpenXvsBearingTrack, OPENXVS_BEARING_TRACK_PAYLOAD_LEN};

// ============================================================================
// FAA GDL-90 Public ICD Rev A Constants & CRC-16-CCITT (poly = 0x1021)
// ============================================================================

pub const GDL90_FLAG_BYTE: u8 = 0x7E;
pub const GDL90_ESCAPE_BYTE: u8 = 0x7D;
pub const GDL90_STUFF_XOR: u8 = 0x20;

pub const GDL90_MSG_ID_HEARTBEAT: u8 = 0x00;
pub const GDL90_MSG_ID_OWNSHIP_REPORT: u8 = 0x0A;
pub const GDL90_MSG_ID_OWNSHIP_GEO_ALT: u8 = 0x0B;
pub const GDL90_MSG_ID_TRAFFIC_REPORT: u8 = 0x14;

/// Official FAA GDL-90 Public ICD §2.2.3 Heartbeat test vector.
pub const GDL90_ICD_HEARTBEAT_UNSTUFFED: [u8; 7] = [0x00, 0x81, 0x41, 0xDB, 0xD0, 0x08, 0x02];
pub const GDL90_ICD_HEARTBEAT_CRC_U16: u16 = 0x8BB3;
pub const GDL90_ICD_HEARTBEAT_FRAMED: [u8; 11] = [
    0x7E, 0x00, 0x81, 0x41, 0xDB, 0xD0, 0x08, 0x02, 0xB3, 0x8B, 0x7E,
];

const fn build_gdl90_crc_table() -> [u16; 256] {
    let mut table = [0u16; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = (i as u16) << 8;
        let mut bit = 0;
        while bit < 8 {
            if (crc & 0x8000) != 0 {
                crc = (crc << 1) ^ 0x1021;
            } else {
                crc <<= 1;
            }
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

const GDL90_CRC16_TABLE: [u16; 256] = build_gdl90_crc_table();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecError {
    FrameTooShort,
    MissingFlags,
    InvalidEscape,
    UnexpectedFlag,
    BufferTooSmall,
    CrcMismatch,
    InvalidStx,
    UnexpectedMsgId,
    InvalidLength,
    SignatureMismatch,
}

/// Compute FAA GDL-90 ICD §2.2.2 CRC-16-CCITT (wire order: LSB first).
#[inline]
pub fn compute_gdl90_crc(data: &[u8]) -> u16 {
    let mut crc: u16 = 0;
    for &b in data {
        crc = GDL90_CRC16_TABLE[((crc >> 8) & 0xFF) as usize] ^ (crc << 8) ^ (b as u16);
    }
    crc
}

/// Append LSB-first CRC-16, byte-stuff `0x7E`/`0x7D`, wrap in `0x7E ... 0x7E`.
/// Returns total framed byte count written into `out`.
pub fn frame_gdl90_message(payload: &[u8], out: &mut [u8]) -> usize {
    let crc = compute_gdl90_crc(payload);
    let fcs = [(crc & 0xFF) as u8, ((crc >> 8) & 0xFF) as u8];
    let mut pos = 0usize;
    if pos < out.len() {
        out[pos] = GDL90_FLAG_BYTE;
        pos += 1;
    }
    for &b in payload.iter().chain(fcs.iter()) {
        if b == GDL90_FLAG_BYTE || b == GDL90_ESCAPE_BYTE {
            if pos + 1 < out.len() {
                out[pos] = GDL90_ESCAPE_BYTE;
                out[pos + 1] = b ^ GDL90_STUFF_XOR;
                pos += 2;
            }
        } else if pos < out.len() {
            out[pos] = b;
            pos += 1;
        }
    }
    if pos < out.len() {
        out[pos] = GDL90_FLAG_BYTE;
        pos += 1;
    }
    pos
}

/// Strip `0x7E` flags, unstuff bytes, verify CRC-16, and write payload into `out`.
pub fn decode_gdl90_frame(frame: &[u8], out: &mut [u8]) -> Result<usize, CodecError> {
    if frame.len() < 5 {
        return Err(CodecError::FrameTooShort);
    }
    if frame[0] != GDL90_FLAG_BYTE || frame[frame.len() - 1] != GDL90_FLAG_BYTE {
        return Err(CodecError::MissingFlags);
    }
    let body = &frame[1..frame.len() - 1];
    let mut unstuffed = [0u8; 64];
    let mut ulen = 0usize;
    let mut i = 0usize;
    while i < body.len() {
        let b = body[i];
        if b == GDL90_ESCAPE_BYTE {
            if i + 1 >= body.len() {
                return Err(CodecError::InvalidEscape);
            }
            if ulen >= unstuffed.len() {
                return Err(CodecError::BufferTooSmall);
            }
            unstuffed[ulen] = body[i + 1] ^ GDL90_STUFF_XOR;
            ulen += 1;
            i += 2;
        } else if b == GDL90_FLAG_BYTE {
            return Err(CodecError::UnexpectedFlag);
        } else {
            if ulen >= unstuffed.len() {
                return Err(CodecError::BufferTooSmall);
            }
            unstuffed[ulen] = b;
            ulen += 1;
            i += 1;
        }
    }
    if ulen < 3 {
        return Err(CodecError::FrameTooShort);
    }
    let plen = ulen - 2;
    let rx_crc = (unstuffed[plen] as u16) | ((unstuffed[plen + 1] as u16) << 8);
    let calc_crc = compute_gdl90_crc(&unstuffed[..plen]);
    if rx_crc != calc_crc {
        return Err(CodecError::CrcMismatch);
    }
    if plen > out.len() {
        return Err(CodecError::BufferTooSmall);
    }
    out[..plen].copy_from_slice(&unstuffed[..plen]);
    Ok(plen)
}

#[inline]
pub fn encode_lat_lon_semicircles(angle_deg: f64) -> [u8; 3] {
    let clamped = clamp(angle_deg, -180.0, 180.0);
    let scale = (0x80_0000_u32 as f64) / 180.0;
    let raw_f = round_ties_even(clamped * scale);
    let raw = (raw_f as i32).clamp(-0x80_0000, 0x7F_FFFF);
    let u24 = (raw as u32) & 0x00FF_FFFF;
    [
        ((u24 >> 16) & 0xFF) as u8,
        ((u24 >> 8) & 0xFF) as u8,
        (u24 & 0xFF) as u8,
    ]
}

#[inline]
pub fn encode_altitude_25ft(alt_ft: Option<f64>) -> u16 {
    match alt_ft {
        Some(v) if v.is_finite() => {
            let code = round_ties_even((v + 1000.0) / 25.0) as i32;
            code.clamp(0x000, 0xFFE) as u16
        }
        _ => 0xFFF,
    }
}

/// Encode and frame GDL-90 Heartbeat (Message ID `0x00`, 7-byte payload).
pub fn encode_gdl90_heartbeat(
    utc_time_s: u32,
    gps_pos_valid: bool,
    maint_reqd: bool,
    utc_ok: bool,
    uplink_count: u8,
    basic_and_long_count: u16,
    out: &mut [u8],
) -> usize {
    let ts = utc_time_s % 86_400;
    let status1 = (u8::from(gps_pos_valid) << 7) | (u8::from(maint_reqd) << 6) | 0x01; // uat_initialized=true
    let ts_bit16 = ((ts >> 16) & 0x01) as u8;
    let status2 = (ts_bit16 << 7) | u8::from(utc_ok);
    let ts_lsb = (ts & 0xFF) as u8;
    let ts_msb = ((ts >> 8) & 0xFF) as u8;
    let ul = uplink_count.min(31);
    let bl = basic_and_long_count.min(1023);
    let msg_cnt_hi = ((ul & 0x1F) << 3) | (((bl >> 8) & 0x03) as u8);
    let msg_cnt_lo = (bl & 0xFF) as u8;
    let payload = [
        GDL90_MSG_ID_HEARTBEAT,
        status1,
        status2,
        ts_lsb,
        ts_msb,
        msg_cnt_hi,
        msg_cnt_lo,
    ];
    frame_gdl90_message(&payload, out)
}

/// Encode and frame GDL-90 Ownship Geometric Altitude (Message ID `0x0B`, 5-byte payload).
pub fn encode_gdl90_ownship_geo_alt(
    alt_ft: f64,
    vfom_m: f64,
    vertical_warning: bool,
    out: &mut [u8],
) -> usize {
    let alt_5ft = (round_ties_even(alt_ft / 5.0) as i32).clamp(-32768, 32767) as i16;
    let u16_alt = alt_5ft as u16;
    let vfom_int: u16 = if !vfom_m.is_finite() || vfom_m < 0.0 {
        0x7FFF
    } else if vfom_m > 32766.0 {
        0x7FFE
    } else {
        (round_ties_even(vfom_m) as u16) & 0x7FFF
    };
    let metrics = (u16::from(vertical_warning) << 15) | vfom_int;
    let payload = [
        GDL90_MSG_ID_OWNSHIP_GEO_ALT,
        ((u16_alt >> 8) & 0xFF) as u8,
        (u16_alt & 0xFF) as u8,
        ((metrics >> 8) & 0xFF) as u8,
        (metrics & 0xFF) as u8,
    ];
    frame_gdl90_message(&payload, out)
}

#[derive(Debug, Clone, Copy)]
pub struct Gdl90TargetParams {
    pub alert_status: u8,
    pub address_type: u8,
    pub participant_address: u32,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub altitude_ft: Option<f64>,
    pub misc_indicators: u8,
    pub nic: u8,
    pub nacp: u8,
    pub horiz_vel_kt: Option<f64>,
    pub vert_vel_fpm: Option<f64>,
    pub track_heading_deg: f64,
    pub emitter_category: u8,
    pub callsign: [u8; 8],
    pub emergency_code: u8,
}

fn encode_target_payload(msg_id: u8, p: &Gdl90TargetParams, force_sim_prefix: bool) -> [u8; 28] {
    let mut out = [0u8; 28];
    out[0] = msg_id;
    out[1] = ((p.alert_status & 0x0F) << 4) | (p.address_type & 0x0F);
    let addr = p.participant_address & 0x00FF_FFFF;
    out[2] = ((addr >> 16) & 0xFF) as u8;
    out[3] = ((addr >> 8) & 0xFF) as u8;
    out[4] = (addr & 0xFF) as u8;

    let lat = encode_lat_lon_semicircles(p.lat_deg);
    let lon = encode_lat_lon_semicircles(p.lon_deg);
    out[5..8].copy_from_slice(&lat);
    out[8..11].copy_from_slice(&lon);

    let alt_code = encode_altitude_25ft(p.altitude_ft);
    out[11] = ((alt_code >> 4) & 0xFF) as u8;
    out[12] = (((alt_code & 0x0F) << 4) as u8) | (p.misc_indicators & 0x0F);
    out[13] = ((p.nic & 0x0F) << 4) | (p.nacp & 0x0F);

    let h_code: u16 = match p.horiz_vel_kt {
        Some(v) if v.is_finite() && v >= 0.0 => (round_ties_even(v) as i32).clamp(0, 0xFFE) as u16,
        _ => 0xFFF,
    };
    let v_code: u16 = match p.vert_vel_fpm {
        Some(v) if v.is_finite() => {
            let steps = (round_ties_even(v / 64.0) as i32).clamp(-510, 510);
            (steps as u16) & 0x0FFF
        }
        _ => 0x800,
    };
    out[14] = ((h_code >> 4) & 0xFF) as u8;
    out[15] = (((h_code & 0x0F) << 4) as u8) | (((v_code >> 8) & 0x0F) as u8);
    out[16] = (v_code & 0xFF) as u8;

    let mut trk = p.track_heading_deg % 360.0;
    if trk < 0.0 {
        trk += 360.0;
    }
    out[17] = ((round_ties_even(trk * (256.0 / 360.0)) as i32) & 0xFF) as u8;
    out[18] = p.emitter_category;

    let mut cs = [b' '; 8];
    let mut dst = 0usize;
    if force_sim_prefix {
        cs[0] = b'S';
        cs[1] = b'I';
        cs[2] = b'M';
        dst = 3;
    }
    for &ch_raw in &p.callsign {
        if dst >= 8 {
            break;
        }
        let ch = ch_raw.to_ascii_uppercase();
        if ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == b' ' {
            cs[dst] = ch;
            dst += 1;
        }
    }
    out[19..27].copy_from_slice(&cs);
    out[27] = (p.emergency_code & 0x0F) << 4;
    out
}

/// Encode and frame GDL-90 Ownship Report (Message ID `0x0A`, 28-byte payload).
pub fn encode_gdl90_ownship(p: &Gdl90TargetParams, out: &mut [u8]) -> usize {
    let payload = encode_target_payload(GDL90_MSG_ID_OWNSHIP_REPORT, p, false);
    frame_gdl90_message(&payload, out)
}

/// Encode and frame GDL-90 Traffic Report (Message ID `0x14`, 28-byte payload).
/// When `sim_prefix` is `true`, strictly forces `"SIM"` prefix on the 8-byte callsign.
pub fn encode_gdl90_traffic(p: &Gdl90TargetParams, sim_prefix: bool, out: &mut [u8]) -> usize {
    let payload = encode_target_payload(GDL90_MSG_ID_TRAFFIC_REPORT, p, sim_prefix);
    frame_gdl90_message(&payload, out)
}

// ============================================================================
// MAVLink v2 X.25 CRC-16/MCRF4XX, #246 ADSB_VEHICLE, and Signed #52001
// ============================================================================

pub const MAVLINK_V2_STX: u8 = 0xFD;
pub const MAVLINK_V2_HEADER_LEN: usize = 10;
pub const MAVLINK_INCOMPAT_FLAG_SIGNED: u8 = 0x01;
pub const MAVLINK_V2_SIG_LEN: usize = 13;

pub const ADSB_VEHICLE_MSG_ID: u32 = 246;
pub const ADSB_VEHICLE_CRC_EXTRA: u8 = 184;
pub const ADSB_VEHICLE_PAYLOAD_LEN: usize = 38;
pub const ADSB_VEHICLE_FRAME_LEN: usize = 50;

pub const OPENXVS_BEARING_TRACK_MSG_ID: u32 = 52_001;
pub const OPENXVS_BEARING_TRACK_CRC_EXTRA: u8 = 73;
/// `10 B` header + `52 B` payload + `2 B` CRC + `13 B` SHA-256 signature = `77 B` exact.
pub const MAVLINK_BEARING_TRACK_FRAME_LEN: usize =
    MAVLINK_V2_HEADER_LEN + OPENXVS_BEARING_TRACK_PAYLOAD_LEN + 2 + MAVLINK_V2_SIG_LEN;

#[inline]
fn x25_accumulate(byte_val: u8, crc: u16) -> u16 {
    let mut tmp = byte_val ^ ((crc & 0xFF) as u8);
    tmp ^= tmp << 4;
    let t = tmp as u16;
    ((crc >> 8) & 0xFF) ^ (t << 8) ^ (t << 3) ^ (t >> 4)
}

pub fn compute_mavlink_x25_crc(header_no_stx: &[u8], payload: &[u8], crc_extra: u8) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for &b in header_no_stx {
        crc = x25_accumulate(b, crc);
    }
    for &b in payload {
        crc = x25_accumulate(b, crc);
    }
    x25_accumulate(crc_extra, crc)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MavlinkAdsb246Params {
    pub icao_address: u32,
    pub lat_deg_e7: i32,
    pub lon_deg_e7: i32,
    pub altitude_mm: i32,
    pub heading_cdeg: u16,
    pub hor_velocity_cms: u16,
    pub ver_velocity_cms: i16,
    pub flags: u16,
    pub squawk: u16,
    pub altitude_type: u8,
    pub callsign: [u8; 9],
    pub emitter_type: u8,
    pub tslc: u8,
    pub seq: u8,
    pub sysid: u8,
    pub compid: u8,
}

/// Pack exact 50-byte MAVLink v2 `#246 ADSB_VEHICLE` frame (`CRC_EXTRA = 184`).
pub fn encode_mavlink_v2_adsb_246(m: &MavlinkAdsb246Params) -> [u8; ADSB_VEHICLE_FRAME_LEN] {
    let mut frame = [0u8; ADSB_VEHICLE_FRAME_LEN];
    frame[0] = MAVLINK_V2_STX;
    frame[1] = ADSB_VEHICLE_PAYLOAD_LEN as u8;
    frame[2] = 0; // incompat_flags
    frame[3] = 0; // compat_flags
    frame[4] = m.seq;
    frame[5] = m.sysid;
    frame[6] = m.compid;
    frame[7] = (ADSB_VEHICLE_MSG_ID & 0xFF) as u8;
    frame[8] = ((ADSB_VEHICLE_MSG_ID >> 8) & 0xFF) as u8;
    frame[9] = ((ADSB_VEHICLE_MSG_ID >> 16) & 0xFF) as u8;

    let p = &mut frame[10..48];
    p[0..4].copy_from_slice(&m.icao_address.to_le_bytes());
    p[4..8].copy_from_slice(&m.lat_deg_e7.to_le_bytes());
    p[8..12].copy_from_slice(&m.lon_deg_e7.to_le_bytes());
    p[12..16].copy_from_slice(&m.altitude_mm.to_le_bytes());
    p[16..18].copy_from_slice(&m.heading_cdeg.to_le_bytes());
    p[18..20].copy_from_slice(&m.hor_velocity_cms.to_le_bytes());
    p[20..22].copy_from_slice(&m.ver_velocity_cms.to_le_bytes());
    p[22..24].copy_from_slice(&m.flags.to_le_bytes());
    p[24..26].copy_from_slice(&m.squawk.to_le_bytes());
    p[26] = m.altitude_type;
    p[27..36].copy_from_slice(&m.callsign);
    p[36] = m.emitter_type;
    p[37] = m.tslc;

    let crc = compute_mavlink_x25_crc(&frame[1..10], &frame[10..48], ADSB_VEHICLE_CRC_EXTRA);
    frame[48] = (crc & 0xFF) as u8;
    frame[49] = ((crc >> 8) & 0xFF) as u8;
    frame
}

/// Unpack and verify a 50-byte MAVLink v2 `#246 ADSB_VEHICLE` frame.
pub fn decode_mavlink_v2_adsb_246(frame: &[u8]) -> Result<MavlinkAdsb246Params, CodecError> {
    if frame.len() != ADSB_VEHICLE_FRAME_LEN {
        return Err(CodecError::InvalidLength);
    }
    if frame[0] != MAVLINK_V2_STX {
        return Err(CodecError::InvalidStx);
    }
    let msgid = (frame[7] as u32) | ((frame[8] as u32) << 8) | ((frame[9] as u32) << 16);
    if msgid != ADSB_VEHICLE_MSG_ID {
        return Err(CodecError::UnexpectedMsgId);
    }
    let rx_crc = (frame[48] as u16) | ((frame[49] as u16) << 8);
    let calc_crc = compute_mavlink_x25_crc(&frame[1..10], &frame[10..48], ADSB_VEHICLE_CRC_EXTRA);
    if rx_crc != calc_crc {
        return Err(CodecError::CrcMismatch);
    }
    let p = &frame[10..48];
    let mut callsign = [0u8; 9];
    callsign.copy_from_slice(&p[27..36]);
    Ok(MavlinkAdsb246Params {
        icao_address: u32::from_le_bytes([p[0], p[1], p[2], p[3]]),
        lat_deg_e7: i32::from_le_bytes([p[4], p[5], p[6], p[7]]),
        lon_deg_e7: i32::from_le_bytes([p[8], p[9], p[10], p[11]]),
        altitude_mm: i32::from_le_bytes([p[12], p[13], p[14], p[15]]),
        heading_cdeg: u16::from_le_bytes([p[16], p[17]]),
        hor_velocity_cms: u16::from_le_bytes([p[18], p[19]]),
        ver_velocity_cms: i16::from_le_bytes([p[20], p[21]]),
        flags: u16::from_le_bytes([p[22], p[23]]),
        squawk: u16::from_le_bytes([p[24], p[25]]),
        altitude_type: p[26],
        callsign,
        emitter_type: p[36],
        tslc: p[37],
        seq: frame[4],
        sysid: frame[5],
        compid: frame[6],
    })
}

/// Pure-Rust `#![no_std]` FIPS 180-4 SHA-256 (zero external crate dependencies) for MAVLink v2 signing.
fn sha256_bytes(chunks: &[&[u8]]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let compress = |state: &mut [u32; 8], block: &[u8; 64]| {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[4 * i],
                block[4 * i + 1],
                block[4 * i + 2],
                block[4 * i + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = *state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(hh);
    };

    let mut block = [0u8; 64];
    let mut bpos = 0usize;
    let mut total_bytes: u64 = 0;

    for chunk in chunks {
        for &byte in *chunk {
            block[bpos] = byte;
            bpos += 1;
            total_bytes += 1;
            if bpos == 64 {
                compress(&mut h, &block);
                bpos = 0;
            }
        }
    }

    // Padding: 0x80, zeros, 64-bit big-endian bit length
    block[bpos] = 0x80;
    bpos += 1;
    if bpos > 56 {
        for slot in &mut block[bpos..64] {
            *slot = 0;
        }
        compress(&mut h, &block);
        bpos = 0;
    }
    for slot in &mut block[bpos..56] {
        *slot = 0;
    }
    let bit_len = total_bytes * 8;
    block[56..64].copy_from_slice(&bit_len.to_be_bytes());
    compress(&mut h, &block);

    let mut digest = [0u8; 32];
    for i in 0..8 {
        digest[4 * i..4 * i + 4].copy_from_slice(&h[i].to_be_bytes());
    }
    digest
}

/// Pack exact 77-byte SHA-256 signed MAVLink v2 `#52001 OPENXVS_BEARING_TRACK` frame.
pub fn encode_mavlink_v2_bearing_track(
    track: &OpenXvsBearingTrack,
    seq: u8,
    sysid: u8,
    compid: u8,
    link_id: u8,
    timestamp_48: u64,
    secret_key: &[u8; 32],
) -> [u8; MAVLINK_BEARING_TRACK_FRAME_LEN] {
    let mut frame = [0u8; MAVLINK_BEARING_TRACK_FRAME_LEN];
    frame[0] = MAVLINK_V2_STX;
    frame[1] = OPENXVS_BEARING_TRACK_PAYLOAD_LEN as u8;
    frame[2] = MAVLINK_INCOMPAT_FLAG_SIGNED;
    frame[3] = 0;
    frame[4] = seq;
    frame[5] = sysid;
    frame[6] = compid;
    frame[7] = (OPENXVS_BEARING_TRACK_MSG_ID & 0xFF) as u8;
    frame[8] = ((OPENXVS_BEARING_TRACK_MSG_ID >> 8) & 0xFF) as u8;
    frame[9] = ((OPENXVS_BEARING_TRACK_MSG_ID >> 16) & 0xFF) as u8;

    let payload = track.to_le_bytes();
    frame[10..62].copy_from_slice(&payload);

    let crc = compute_mavlink_x25_crc(
        &frame[1..10],
        &frame[10..62],
        OPENXVS_BEARING_TRACK_CRC_EXTRA,
    );
    frame[62] = (crc & 0xFF) as u8;
    frame[63] = ((crc >> 8) & 0xFF) as u8;

    // MAVLink v2 13-byte signature: link_id (1 B) + timestamp_48 LE (6 B) + sha256[0..6]
    let ts_bytes = timestamp_48.to_le_bytes();
    frame[64] = link_id;
    frame[65..71].copy_from_slice(&ts_bytes[0..6]);
    let digest = sha256_bytes(&[secret_key, &frame[0..71]]);
    frame[71..77].copy_from_slice(&digest[0..6]);
    frame
}

/// Verify and unpack a 77-byte signed MAVLink v2 `#52001 OPENXVS_BEARING_TRACK` frame.
pub fn decode_mavlink_v2_bearing_track(
    frame: &[u8],
    secret_key: &[u8; 32],
) -> Result<(OpenXvsBearingTrack, u8, u64), CodecError> {
    if frame.len() != MAVLINK_BEARING_TRACK_FRAME_LEN {
        return Err(CodecError::InvalidLength);
    }
    if frame[0] != MAVLINK_V2_STX {
        return Err(CodecError::InvalidStx);
    }
    let msgid = (frame[7] as u32) | ((frame[8] as u32) << 8) | ((frame[9] as u32) << 16);
    if msgid != OPENXVS_BEARING_TRACK_MSG_ID {
        return Err(CodecError::UnexpectedMsgId);
    }
    if (frame[2] & MAVLINK_INCOMPAT_FLAG_SIGNED) == 0 {
        return Err(CodecError::SignatureMismatch);
    }
    let rx_crc = (frame[62] as u16) | ((frame[63] as u16) << 8);
    let calc_crc = compute_mavlink_x25_crc(
        &frame[1..10],
        &frame[10..62],
        OPENXVS_BEARING_TRACK_CRC_EXTRA,
    );
    if rx_crc != calc_crc {
        return Err(CodecError::CrcMismatch);
    }
    let digest = sha256_bytes(&[secret_key, &frame[0..71]]);
    if frame[71..77] != digest[0..6] {
        return Err(CodecError::SignatureMismatch);
    }
    let mut payload = [0u8; OPENXVS_BEARING_TRACK_PAYLOAD_LEN];
    payload.copy_from_slice(&frame[10..62]);
    let trk = OpenXvsBearingTrack::from_le_bytes(&payload);
    let link_id = frame[64];
    let mut ts_le = [0u8; 8];
    ts_le[0..6].copy_from_slice(&frame[65..71]);
    let ts48 = u64::from_le_bytes(ts_le);
    Ok((trk, link_id, ts48))
}
