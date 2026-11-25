// Copyright 2026 OpenXVS Contributors
// SPDX-License-Identifier: Apache-2.0
//! GNSS-Denied Two-Way RF `PeerClockOffsetEstimator` (`#![no_std]`, zero heap, zero `unsafe`).
//!
//! Closes `REALWORLD_DEPLOYABILITY_AUDIT_V5.md` Failure Mode 7 (GNSS-denied peer clock skew):
//! during GNSS-denied flight (`XVS-Nav` active), COTS TCXOs drift `7-72 ms/hr` (`±2-20 ppm`),
//! pushing independent aircraft clocks past `StatefulMeshDaaGuard`'s `50 ms` `MAX_FUTURE_SKEW_US`
//! and silently dropping valid `#52001` peer packets.
//!
//! This module estimates each peer's clock offset `offset_us ≈ t_peer - t_local` from bounded
//! two-way RF timestamp exchanges `(t1_local_tx, t2_peer_rx, t3_peer_tx, t4_local_rx)`:
//!   - `offset_us = ((t2 - t1) + (t3 - t4)) / 2` (exact `i128` difference arithmetic, zero overflow)
//!   - `rtt_us    = (t4 - t1) - (t3 - t2)`
//!   - Hard max RTT bound (`MAX_SYNC_RTT_US = 150 ms`) and offset bound (`±MAX_ABS_CLOCK_OFFSET_US`)
//!   - Median-of-5 outlier rejection (optimal 9-comparator sorting network, zero loops/heap)
//!   - Bounded rate-limited offset slew (`<= min(dt_sync / 10, MAX_SLEW_STEP_US)`)
//!   - Fail-closed conservative uncertainty ceiling (`MAX_OFFSET_UNCERT_US = 20 ms`), combining
//!     path-asymmetry bound (`median_rtt / 2`), trimmed sample dispersion, holdover TCXO drift
//!     (`40 ppm`), and any active slew residual.

/// Rolling median window size for outlier-resistant clock offset estimation.
pub const SYNC_WINDOW_SIZE: usize = 5;

/// Minimum accepted two-way RF exchanges required before declaring a peer synchronized.
pub const MIN_SYNC_SAMPLES: usize = 3;

/// Maximum tracked peer sender IDs in static zero-heap clock-sync table.
pub const MAX_SYNC_PEERS: usize = 16;

/// Maximum accepted two-way RF round-trip time (`150 ms`). Exchanges above this fail closed.
pub const MAX_SYNC_RTT_US: u64 = 150_000;

/// Fail-closed upper bound on peer clock offset uncertainty (`20 ms`).
pub const MAX_OFFSET_UNCERT_US: u64 = 20_000;

/// Bounded absolute clock offset accepted and emitted (`±5.0 s`, 10x above multi-hour TCXO drift).
pub const MAX_ABS_CLOCK_OFFSET_US: i64 = 5_000_000;

/// Maximum per-update clock offset slew step (`5 ms`) after initial calibration lock.
pub const MAX_SLEW_STEP_US: u64 = 5_000;

/// Slew rate shift (`dt_local >> 4` = `dt_local / 16` = max `6.25%` slew rate = `62,500 ppm`,
/// over 1,500x faster than TCXO drift yet strictly `< 100%` so monotonic forward local time
/// always outpaces offset slew; exact power-of-2 avoids divider circuits on bare-metal MCUs).
pub const SLEW_RATE_SHIFT: u32 = 4;

/// Equivalent divisor (`1 << SLEW_RATE_SHIFT = 16`) for slew-rate bound assertions.
pub const SLEW_RATE_DIVISOR: u64 = 1u64 << SLEW_RATE_SHIFT;

/// Conservative worst-case relative TCXO frequency drift between two independent ±20 ppm
/// oscillators (`40 ppm` physical spec; bounded conservatively via shift sum `1/2^15 + 1/2^16 ≈ 45.78 ppm`).
pub const MAX_REL_TCXO_DRIFT_PPM: u64 = 40;

/// Explicit fail-closed rejection reason for an invalid two-way RF sync exchange.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockSyncRejectReason {
    /// Non-causal exchange timestamps (`t4 < t1`, `t3 < t2`, or backward `t4 < last_sync`).
    NonCausalTimestamps,
    /// Peer turnaround exceeded local span (`(t4 - t1) < (t3 - t2)`).
    NegativeRtt,
    /// Two-way RF round-trip time exceeded `max_rtt_us` (`150 ms`).
    RttExceeded,
    /// Estimated offset exceeded `±MAX_ABS_CLOCK_OFFSET_US`.
    OffsetOutOfBounds,
    /// Static peer table (`MAX_SYNC_PEERS = 16`) is full.
    PeerTableFull,
}

/// One bounded two-way RF timestamp exchange with a cooperative peer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TwoWaySyncExchange {
    pub sender_id: u8,
    pub t1_local_tx_us: u64,
    pub t2_peer_rx_us: u64,
    pub t3_peer_tx_us: u64,
    pub t4_local_rx_us: u64,
}

/// Snapshot of a peer's clock discipline at a given local evaluation timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerClockEstimate {
    /// `true` iff calibrated with `>= MIN_SYNC_SAMPLES`, causal local time, and
    /// `offset_uncert_us <= max_offset_uncert_us` (`<= 20 ms`).
    pub synchronized: bool,
    /// Estimated `t_peer - t_local` in microseconds, bounded to `[-MAX_ABS_CLOCK_OFFSET_US, +MAX_ABS_CLOCK_OFFSET_US]`.
    pub offset_us: i64,
    /// Conservative upper bound on offset estimation error (`us`). `u64::MAX` when uncalibrated.
    pub offset_uncert_us: u64,
    /// Median two-way RF round-trip time (`us`) across the rolling window.
    pub median_rtt_us: u64,
    /// Number of valid samples stored in the rolling window (`0..=5`).
    pub sample_count: u8,
    /// Local timestamp (`t4_local_rx_us`) of the latest accepted sync exchange.
    pub last_sync_local_us: u64,
}

impl PeerClockEstimate {
    pub const UNCALIBRATED: Self = Self {
        synchronized: false,
        offset_us: 0,
        offset_uncert_us: u64::MAX,
        median_rtt_us: u64::MAX,
        sample_count: 0,
        last_sync_local_us: 0,
    };
}

/// Result returned by `PeerClockOffsetEstimator::record_exchange`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClockSyncSampleOutcome {
    pub accepted: bool,
    pub reject_reason: Option<ClockSyncRejectReason>,
    pub raw_offset_us: i64,
    pub rtt_us: u64,
    pub estimate: PeerClockEstimate,
}

/// Single-peer zero-heap clock-offset channel (rolling median-of-5 window + bounded slew).
#[derive(Debug, Clone, Copy)]
pub(crate) struct PeerSyncSlot {
    pub(crate) used: bool,
    pub(crate) sender_id: u8,
    offsets_us: [i64; SYNC_WINDOW_SIZE],
    rtts_us: [u64; SYNC_WINDOW_SIZE],
    sample_count: u8,
    write_idx: u8,
    calibrated: bool,
    slewed_offset_us: i64,
    median_offset_us: i64,
    median_rtt_us: u64,
    dispersion_us: u64,
    last_sync_local_us: u64,
}

impl PeerSyncSlot {
    pub(crate) const EMPTY: Self = Self {
        used: false,
        sender_id: 0,
        offsets_us: [0; SYNC_WINDOW_SIZE],
        rtts_us: [0; SYNC_WINDOW_SIZE],
        sample_count: 0,
        write_idx: 0,
        calibrated: false,
        slewed_offset_us: 0,
        median_offset_us: 0,
        median_rtt_us: u64::MAX,
        dispersion_us: u64::MAX,
        last_sync_local_us: 0,
    };

    /// Pure single-peer update on an already-validated `(raw_offset_us, rtt_us)` sample.
    /// Operates strictly in `i64`/`u64` (zero `i128`/`u128`, zero array-of-peers indexing).
    pub(crate) fn ingest_validated(
        &mut self,
        raw_offset_us: i64,
        rtt_us: u64,
        t4_local_rx_us: u64,
        max_offset_uncert_us: u64,
        max_slew_step_us: u64,
    ) -> Result<(), ClockSyncRejectReason> {
        if self.sample_count > 0 && t4_local_rx_us < self.last_sync_local_us {
            return Err(ClockSyncRejectReason::NonCausalTimestamps);
        }

        let widx = if (self.write_idx as usize) < SYNC_WINDOW_SIZE {
            self.write_idx as usize
        } else {
            0
        };
        self.offsets_us[widx] = raw_offset_us;
        self.rtts_us[widx] = rtt_us;
        self.write_idx = if widx + 1 >= SYNC_WINDOW_SIZE {
            0
        } else {
            (widx + 1) as u8
        };
        if (self.sample_count as usize) < SYNC_WINDOW_SIZE {
            self.sample_count += 1;
        }

        let count = self.sample_count as usize;
        let (median_offset_us, dispersion_us) =
            median_and_dispersion_i64(&self.offsets_us, count);
        let median_rtt_us = median_u64(&self.rtts_us, count);

        if !self.calibrated {
            self.slewed_offset_us = median_offset_us;
            if count >= MIN_SYNC_SAMPLES {
                let init_uncert = (median_rtt_us >> 1).saturating_add(dispersion_us);
                if init_uncert <= max_offset_uncert_us {
                    self.calibrated = true;
                }
            }
        } else {
            let dt_us = t4_local_rx_us.saturating_sub(self.last_sync_local_us);
            let rate_cap = dt_us >> SLEW_RATE_SHIFT;
            let max_slew_us = if rate_cap < max_slew_step_us {
                rate_cap
            } else {
                max_slew_step_us
            };
            // Both median_offset_us and slewed_offset_us are in [-5_000_000, +5_000_000],
            // so their difference fits in i64 with > 10^11x headroom.
            let delta = median_offset_us - self.slewed_offset_us;
            let max_slew_i64 = max_slew_us as i64;
            let clamped_delta = if delta > max_slew_i64 {
                max_slew_i64
            } else if delta < -max_slew_i64 {
                -max_slew_i64
            } else {
                delta
            };
            self.slewed_offset_us = clamp_offset_i64(self.slewed_offset_us + clamped_delta);
        }

        self.median_offset_us = median_offset_us;
        self.median_rtt_us = median_rtt_us;
        self.dispersion_us = dispersion_us;
        self.last_sync_local_us = t4_local_rx_us;
        Ok(())
    }

    pub(crate) fn estimate(
        &self,
        now_local_us: u64,
        max_offset_uncert_us: u64,
    ) -> PeerClockEstimate {
        if (self.sample_count as usize) < MIN_SYNC_SAMPLES
            || !self.calibrated
            || now_local_us < self.last_sync_local_us
        {
            return PeerClockEstimate {
                synchronized: false,
                offset_us: self.slewed_offset_us,
                offset_uncert_us: u64::MAX,
                median_rtt_us: self.median_rtt_us,
                sample_count: self.sample_count,
                last_sync_local_us: self.last_sync_local_us,
            };
        }

        let elapsed_us = now_local_us - self.last_sync_local_us;
        let holdover_drift_us = holdover_drift_bound_us(elapsed_us);
        let rtt_half_us = self.median_rtt_us >> 1;
        let slew_residual_us = (self.median_offset_us - self.slewed_offset_us).unsigned_abs();

        let total_uncert_us = rtt_half_us
            .saturating_add(self.dispersion_us)
            .saturating_add(holdover_drift_us)
            .saturating_add(slew_residual_us);

        PeerClockEstimate {
            synchronized: total_uncert_us <= max_offset_uncert_us,
            offset_us: self.slewed_offset_us,
            offset_uncert_us: total_uncert_us,
            median_rtt_us: self.median_rtt_us,
            sample_count: self.sample_count,
            last_sync_local_us: self.last_sync_local_us,
        }
    }
}

/// Pure, overflow-free validation of one two-way RF timestamp exchange over all `2^256` `u64` tuples.
///
/// Uses the exact algebraic identity `(t3 - t4) == (t2 - t1) - rtt_us` (where `rtt_us = (t4 - t1) - (t3 - t2)`),
/// eliminating all 128-bit arithmetic while remaining bit-exact to `((t2 - t1) + (t3 - t4)) / 2`.
pub(crate) fn validate_two_way_exchange(
    ex: &TwoWaySyncExchange,
    max_rtt_us: u64,
) -> Result<(i64, u64), (ClockSyncRejectReason, i64, u64)> {
    if ex.t4_local_rx_us < ex.t1_local_tx_us || ex.t3_peer_tx_us < ex.t2_peer_rx_us {
        return Err((ClockSyncRejectReason::NonCausalTimestamps, 0, u64::MAX));
    }
    let local_span_us = ex.t4_local_rx_us - ex.t1_local_tx_us;
    let peer_turnaround_us = ex.t3_peer_tx_us - ex.t2_peer_rx_us;
    if local_span_us < peer_turnaround_us {
        return Err((ClockSyncRejectReason::NegativeRtt, 0, u64::MAX));
    }
    let rtt_us = local_span_us - peer_turnaround_us;
    if rtt_us > max_rtt_us {
        return Err((ClockSyncRejectReason::RttExceeded, 0, rtt_us));
    }
    // Since d2 = d1 - rtt_us and 0 <= rtt_us <= MAX_SYNC_RTT_US, |d1| > MAX_ABS + MAX_SYNC_RTT_US
    // guarantees |offset| = |(2*d1 - rtt_us)/2| > MAX_ABS_CLOCK_OFFSET_US.
    let max_d1_mag = (MAX_ABS_CLOCK_OFFSET_US as u64) + MAX_SYNC_RTT_US;
    let d1: i64 = if ex.t2_peer_rx_us >= ex.t1_local_tx_us {
        let mag = ex.t2_peer_rx_us - ex.t1_local_tx_us;
        if mag > max_d1_mag {
            return Err((
                ClockSyncRejectReason::OffsetOutOfBounds,
                MAX_ABS_CLOCK_OFFSET_US,
                rtt_us,
            ));
        }
        mag as i64
    } else {
        let mag = ex.t1_local_tx_us - ex.t2_peer_rx_us;
        if mag > max_d1_mag {
            return Err((
                ClockSyncRejectReason::OffsetOutOfBounds,
                -MAX_ABS_CLOCK_OFFSET_US,
                rtt_us,
            ));
        }
        -(mag as i64)
    };
    let d2 = d1 - (rtt_us as i64);
    let offset_us = div2_trunc_i64(d1 + d2);
    if offset_us < -MAX_ABS_CLOCK_OFFSET_US {
        return Err((
            ClockSyncRejectReason::OffsetOutOfBounds,
            -MAX_ABS_CLOCK_OFFSET_US,
            rtt_us,
        ));
    }
    if offset_us > MAX_ABS_CLOCK_OFFSET_US {
        return Err((
            ClockSyncRejectReason::OffsetOutOfBounds,
            MAX_ABS_CLOCK_OFFSET_US,
            rtt_us,
        ));
    }
    Ok((offset_us, rtt_us))
}

/// `#![no_std]` zero-heap two-way RF peer clock-offset estimator with median-of-5 outlier
/// rejection, bounded slew rate, and fail-closed `<= 20 ms` uncertainty gating.
#[derive(Debug, Clone, Copy)]
pub struct PeerClockOffsetEstimator {
    peers: [PeerSyncSlot; MAX_SYNC_PEERS],
    max_rtt_us: u64,
    max_offset_uncert_us: u64,
    max_slew_step_us: u64,
}

impl Default for PeerClockOffsetEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl PeerClockOffsetEstimator {
    /// Construct an estimator with canonical OpenXVS V5.2 safety bounds (`150 ms` max RTT,
    /// `20 ms` max offset uncertainty, `5 ms` max slew step).
    pub const fn new() -> Self {
        Self {
            peers: [PeerSyncSlot::EMPTY; MAX_SYNC_PEERS],
            max_rtt_us: MAX_SYNC_RTT_US,
            max_offset_uncert_us: MAX_OFFSET_UNCERT_US,
            max_slew_step_us: MAX_SLEW_STEP_US,
        }
    }

    /// Construct an estimator with custom bounds, clamped fail-closed to canonical ceilings.
    pub const fn with_bounds(
        max_rtt_us: u64,
        max_offset_uncert_us: u64,
        max_slew_step_us: u64,
    ) -> Self {
        let clamped_rtt = if max_rtt_us <= MAX_SYNC_RTT_US {
            max_rtt_us
        } else {
            MAX_SYNC_RTT_US
        };
        let clamped_uncert = if max_offset_uncert_us <= MAX_OFFSET_UNCERT_US {
            max_offset_uncert_us
        } else {
            MAX_OFFSET_UNCERT_US
        };
        let clamped_slew = if max_slew_step_us <= MAX_SLEW_STEP_US {
            max_slew_step_us
        } else {
            MAX_SLEW_STEP_US
        };
        Self {
            peers: [PeerSyncSlot::EMPTY; MAX_SYNC_PEERS],
            max_rtt_us: clamped_rtt,
            max_offset_uncert_us: clamped_uncert,
            max_slew_step_us: clamped_slew,
        }
    }

    #[inline]
    pub const fn max_rtt_us(&self) -> u64 {
        self.max_rtt_us
    }

    #[inline]
    pub const fn max_offset_uncert_us(&self) -> u64 {
        self.max_offset_uncert_us
    }

    #[inline]
    pub const fn max_slew_step_us(&self) -> u64 {
        self.max_slew_step_us
    }

    /// Reset all tracked peer clock estimates back to uncalibrated fail-closed state.
    pub fn reset(&mut self) {
        self.peers = [PeerSyncSlot::EMPTY; MAX_SYNC_PEERS];
    }

    /// Invalidate a single peer's clock discipline (e.g., on link reset / reboot detection).
    pub fn invalidate_peer(&mut self, sender_id: u8) {
        let mut i = 0usize;
        while i < MAX_SYNC_PEERS {
            if self.peers[i].used && self.peers[i].sender_id == sender_id {
                self.peers[i] = PeerSyncSlot::EMPTY;
                return;
            }
            i += 1;
        }
    }

    /// Validate and ingest one two-way RF timestamp exchange `(t1, t2, t3, t4)`.
    ///
    /// Strictly overflow-free across all `u64` inputs (`2^256` timestamp space). Any invalid
    /// causality, negative RTT, `rtt_us > max_rtt_us`, or `|offset_us| > MAX_ABS_CLOCK_OFFSET_US`
    /// is rejected fail-closed without mutating the peer's existing rolling window.
    pub fn record_exchange(&mut self, ex: TwoWaySyncExchange) -> ClockSyncSampleOutcome {
        let eval_now = ex.t4_local_rx_us;

        let (raw_offset_us, rtt_us) = match validate_two_way_exchange(&ex, self.max_rtt_us) {
            Ok(pair) => pair,
            Err((reason, raw_off, rtt)) => {
                return ClockSyncSampleOutcome {
                    accepted: false,
                    reject_reason: Some(reason),
                    raw_offset_us: raw_off,
                    rtt_us: rtt,
                    estimate: self.estimate_peer(ex.sender_id, eval_now),
                };
            }
        };

        let slot_idx = match self.find_or_alloc_slot(ex.sender_id) {
            Some(idx) => idx,
            None => {
                return ClockSyncSampleOutcome {
                    accepted: false,
                    reject_reason: Some(ClockSyncRejectReason::PeerTableFull),
                    raw_offset_us,
                    rtt_us,
                    estimate: PeerClockEstimate::UNCALIBRATED,
                };
            }
        };

        // Single local copy avoids repeated symbolic array-mux writes in CBMC
        let mut slot = self.peers[slot_idx];
        match slot.ingest_validated(
            raw_offset_us,
            rtt_us,
            eval_now,
            self.max_offset_uncert_us,
            self.max_slew_step_us,
        ) {
            Ok(()) => {
                let estimate = slot.estimate(eval_now, self.max_offset_uncert_us);
                self.peers[slot_idx] = slot;
                ClockSyncSampleOutcome {
                    accepted: true,
                    reject_reason: None,
                    raw_offset_us,
                    rtt_us,
                    estimate,
                }
            }
            Err(reason) => ClockSyncSampleOutcome {
                accepted: false,
                reject_reason: Some(reason),
                raw_offset_us,
                rtt_us,
                estimate: slot.estimate(eval_now, self.max_offset_uncert_us),
            },
        }
    }

    /// Evaluate a peer's calibrated clock offset and fail-closed uncertainty bound at `now_local_us`.
    pub fn estimate_peer(&self, sender_id: u8, now_local_us: u64) -> PeerClockEstimate {
        match self.find_slot(sender_id) {
            Some(slot) => slot.estimate(now_local_us, self.max_offset_uncert_us),
            None => PeerClockEstimate::UNCALIBRATED,
        }
    }

    /// Translate a peer's observation timestamp `peer_ts_us` into local clock time
    /// (`peer_ts_us - offset_us`). Returns `None` fail-closed if the peer is not synchronized
    /// (`offset_uncert_us > max_offset_uncert_us`) or if translation would underflow/overflow `u64`.
    pub fn translate_peer_to_local_us(
        &self,
        sender_id: u8,
        peer_ts_us: u64,
        now_local_us: u64,
    ) -> Option<(u64, PeerClockEstimate)> {
        let est = self.estimate_peer(sender_id, now_local_us);
        translate_ts_with_estimate(peer_ts_us, est)
    }

    fn find_slot(&self, sender_id: u8) -> Option<&PeerSyncSlot> {
        let mut i = 0usize;
        while i < MAX_SYNC_PEERS {
            if self.peers[i].used && self.peers[i].sender_id == sender_id {
                return Some(&self.peers[i]);
            }
            i += 1;
        }
        None
    }

    fn find_or_alloc_slot(&mut self, sender_id: u8) -> Option<usize> {
        let mut free_idx: Option<usize> = None;
        let mut i = 0usize;
        while i < MAX_SYNC_PEERS {
            if self.peers[i].used && self.peers[i].sender_id == sender_id {
                return Some(i);
            }
            if !self.peers[i].used && free_idx.is_none() {
                free_idx = Some(i);
            }
            i += 1;
        }
        if let Some(idx) = free_idx {
            self.peers[idx] = PeerSyncSlot {
                used: true,
                sender_id,
                ..PeerSyncSlot::EMPTY
            };
            Some(idx)
        } else {
            None
        }
    }
}

/// Pure `u64` timestamp translation `peer_ts_us - est.offset_us` with exact underflow/overflow guard.
#[inline]
pub(crate) fn translate_ts_with_estimate(
    peer_ts_us: u64,
    est: PeerClockEstimate,
) -> Option<(u64, PeerClockEstimate)> {
    if !est.synchronized {
        return None;
    }
    let aligned = if est.offset_us >= 0 {
        peer_ts_us.checked_sub(est.offset_us as u64)?
    } else {
        peer_ts_us.checked_add(est.offset_us.unsigned_abs())?
    };
    Some((aligned, est))
}

/// Optimal 9-comparator pure-SSA sorting network for 5 signed `i64` values (zero loops, zero pointers).
#[inline]
pub(crate) const fn sort5_i64(a: [i64; 5]) -> [i64; 5] {
    let (x0, x1) = minmax_i64(a[0], a[1]);
    let (x3, x4) = minmax_i64(a[3], a[4]);
    let (x2, x4) = minmax_i64(a[2], x4);
    let (x2, x3) = minmax_i64(x2, x3);
    let (x1, x4) = minmax_i64(x1, x4);
    let (x0, x3) = minmax_i64(x0, x3);
    let (x0, x2) = minmax_i64(x0, x2);
    let (x1, x3) = minmax_i64(x1, x3);
    let (x1, x2) = minmax_i64(x1, x2);
    [x0, x1, x2, x3, x4]
}

/// Optimal 9-comparator pure-SSA sorting network for 5 unsigned `u64` values (zero loops, zero pointers).
#[inline]
pub(crate) const fn sort5_u64(a: [u64; 5]) -> [u64; 5] {
    let (x0, x1) = minmax_u64(a[0], a[1]);
    let (x3, x4) = minmax_u64(a[3], a[4]);
    let (x2, x4) = minmax_u64(a[2], x4);
    let (x2, x3) = minmax_u64(x2, x3);
    let (x1, x4) = minmax_u64(x1, x4);
    let (x0, x3) = minmax_u64(x0, x3);
    let (x0, x2) = minmax_u64(x0, x2);
    let (x1, x3) = minmax_u64(x1, x3);
    let (x1, x2) = minmax_u64(x1, x2);
    [x0, x1, x2, x3, x4]
}

#[inline]
const fn minmax_i64(a: i64, b: i64) -> (i64, i64) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

#[inline]
const fn minmax_u64(a: u64, b: u64) -> (u64, u64) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

#[inline]
const fn div2_trunc_i64(x: i64) -> i64 {
    if x >= 0 {
        x >> 1
    } else {
        -((x.wrapping_neg() as u64 >> 1) as i64)
    }
}

#[inline]
fn median_and_dispersion_i64(buf: &[i64; SYNC_WINDOW_SIZE], count: usize) -> (i64, u64) {
    // All stored offsets are validated within [-MAX_ABS_CLOCK_OFFSET_US, +MAX_ABS_CLOCK_OFFSET_US]
    // (±5e6), so signed sums/differences fit comfortably in i64 without overflow.
    match count {
        0 => (0, u64::MAX),
        1 => (buf[0], u64::MAX),
        2 => (clamp_offset_i64(div2_trunc_i64(buf[0] + buf[1])), u64::MAX),
        3 => {
            let (x0, x1) = minmax_i64(buf[0], buf[1]);
            let (x1, x2) = minmax_i64(x1, buf[2]);
            let (x0, x1) = minmax_i64(x0, x1);
            let spread = (x2 - x0).unsigned_abs() >> 1;
            (x1, spread)
        }
        4 => {
            let (x0, x1) = minmax_i64(buf[0], buf[1]);
            let (x2, x3) = minmax_i64(buf[2], buf[3]);
            let (_x0, x2) = minmax_i64(x0, x2);
            let (x1, _x3) = minmax_i64(x1, x3);
            let (x1, x2) = minmax_i64(x1, x2);
            let med = div2_trunc_i64(x1 + x2);
            let spread = (x2 - x1).unsigned_abs() >> 1;
            (clamp_offset_i64(med), spread)
        }
        _ => {
            let s = sort5_i64(*buf);
            // Trimmed inner-3 half-spread rejects both top-1 and bottom-1 outliers cleanly
            let spread = (s[3] - s[1]).unsigned_abs() >> 1;
            (s[2], spread)
        }
    }
}

#[inline]
fn median_u64(buf: &[u64; SYNC_WINDOW_SIZE], count: usize) -> u64 {
    // All stored RTTs are validated <= MAX_SYNC_RTT_US (150_000), so u64 sums cannot overflow.
    match count {
        0 => u64::MAX,
        1 => buf[0],
        2 => (buf[0] + buf[1]) >> 1,
        3 => {
            let (x0, x1) = minmax_u64(buf[0], buf[1]);
            let (x1, _x2) = minmax_u64(x1, buf[2]);
            let (_x0, x1) = minmax_u64(x0, x1);
            x1
        }
        4 => {
            let (x0, x1) = minmax_u64(buf[0], buf[1]);
            let (x2, x3) = minmax_u64(buf[2], buf[3]);
            let (_x0, x2) = minmax_u64(x0, x2);
            let (x1, _x3) = minmax_u64(x1, x3);
            let (x1, x2) = minmax_u64(x1, x2);
            (x1 + x2) >> 1
        }
        _ => {
            let s = sort5_u64(*buf);
            s[2]
        }
    }
}

#[inline]
fn clamp_offset_i64(val: i64) -> i64 {
    val.clamp(-MAX_ABS_CLOCK_OFFSET_US, MAX_ABS_CLOCK_OFFSET_US)
}

#[inline]
fn holdover_drift_bound_us(elapsed_us: u64) -> u64 {
    // Strictly conservative upper bound on 40 ppm relative TCXO drift using pure bitshifts
    // (`(1/32768 + 1/65536) = 3/65536 ≈ 45.78 ppm > 40 ppm`, zero division, zero overflow).
    if elapsed_us == 0 {
        0
    } else {
        (elapsed_us >> 15)
            .saturating_add(elapsed_us >> 16)
            .saturating_add(1)
    }
}
