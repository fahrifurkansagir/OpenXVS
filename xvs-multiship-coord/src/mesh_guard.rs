// Copyright 2026 OpenXVS Contributors
// SPDX-License-Identifier: Apache-2.0
//! DR-22 Finding F4 Fix: Stateful `XVS-MeshDAA` `200 ms` Monotonic Staleness Gate & Sticky
//! Triple-Ray Ghost-Suppression Hysteresis Wrapper (`#![no_std]`, zero heap).
//!
//! Wraps the unmodified frozen `xvs_math_core::mesh_daa::MeshDaaFuser` from above:
//!   1. Rejects stale (`> 200 ms`) and future-skewed (`> 50 ms`) `#52001` peer frames.
//!   2. Rejects per-sender non-monotonic timestamp replays (`ts < last_ts[sender_id]`).
//!   3. Retains triple-ray-confirmed track anchors across transient `150–250 ms` `3 <-> 2`
//!      peer packet drops (`TRIPLE_RAY_HYSTERESIS_US = 500_000` us = `500 ms`), suppressing
//!      newly spawned coplanar 2-ship ghosts (`0 / 100` re-admission) while keeping every
//!      previously confirmed real intruder track continuous.

use crate::clock_sync::{
    translate_ts_with_estimate, PeerClockEstimate, PeerClockOffsetEstimator,
};
use xvs_math_core::det_f64::{dot3, sub3, Vec3};
use xvs_math_core::mesh_daa::{
    FusedMeshTrack, MeshDaaFuser, MeshFusionResult, OpenXvsBearingTrack, MAX_FUSED_TRACKS,
};

/// Maximum acceptable observation age (`200 ms`, aligned with `xvs-os-hal` & `xvs-odd-vlm`).
pub const MAX_MESH_AGE_US: u64 = 200_000;

/// Maximum tolerated clock skew into the future (`50 ms`).
pub const MAX_FUTURE_SKEW_US: u64 = 50_000;

/// Sticky triple-ray ghost-suppression hysteresis window (`500 ms` covers transient `150-250 ms` drops).
pub const TRIPLE_RAY_HYSTERESIS_US: u64 = 500_000;

/// Maximum bounded per-tick observation batch size (zero heap allocation).
pub const MAX_GUARD_OBS: usize = 32;

/// Maximum tracked peer sender IDs in static monotonic-watermark table.
pub const MAX_TRACKED_PEERS: usize = 16;

/// Assumed upper bound on civil intruder cruise speed (`m/s`) for anchor gating during brief drops.
const MAX_CIVIL_INTRUDER_SPEED_MPS: f64 = 120.0;

#[derive(Debug, Clone, Copy)]
struct PeerWatermark {
    used: bool,
    sender_id: u8,
    last_ts_utc_us: u64,
}

impl PeerWatermark {
    const EMPTY: Self = Self {
        used: false,
        sender_id: 0,
        last_ts_utc_us: 0,
    };
}

#[derive(Debug, Clone, Copy)]
struct ConfirmedAnchor {
    valid: bool,
    pos_ecef: Vec3,
    confirmed_utc_us: u64,
}

impl ConfirmedAnchor {
    const EMPTY: Self = Self {
        valid: false,
        pos_ecef: [0.0; 3],
        confirmed_utc_us: 0,
    };
}

/// Single-observation decision emitted by the clock-synced freshness & replay gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ObsClockSyncVerdict {
    /// Observation is fresh in local-aligned time and monotonic in the peer's raw clock domain.
    Accept(u64),
    /// GNSS-denied peer lacks synchronized clock discipline (`offset_uncert_us > 20 ms`).
    RejectUncalibrated,
    /// Aligned observation violates `> 200 ms` age or `> 50 ms` future skew.
    RejectStaleOrFuture,
    /// Observation raw peer timestamp is strictly older than last committed watermark (replay).
    RejectReplay,
}

/// Pure single-observation clock-sync freshness & slew-invariant monotonic replay kernel
/// executed by `StatefulMeshDaaGuard::filter_fresh_observations_with_clock_sync`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn evaluate_obs_clock_synced(
    obs_sender_id: u8,
    obs_raw_ts_us: u64,
    now_local_us: u64,
    own_sender_id: u8,
    gnss_denied: bool,
    peer_est: PeerClockEstimate,
    prev_raw_ts: Option<u64>,
    max_age_us: u64,
    max_future_skew_us: u64,
) -> ObsClockSyncVerdict {
    // 1. Apply peer clock offset calibration when synchronized; fail-closed if GNSS-denied & uncalibrated
    let aligned_ts_us = if obs_sender_id == own_sender_id {
        obs_raw_ts_us
    } else if let Some((aligned, _est)) = translate_ts_with_estimate(obs_raw_ts_us, peer_est) {
        aligned
    } else if gnss_denied {
        return ObsClockSyncVerdict::RejectUncalibrated;
    } else {
        obs_raw_ts_us
    };

    // 2. Future-skew (<= 50 ms) & staleness (<= 200 ms) gate in local-aligned time
    let future_viol = aligned_ts_us > now_local_us.saturating_add(max_future_skew_us);
    let stale_viol = now_local_us > aligned_ts_us.saturating_add(max_age_us);
    if future_viol || stale_viol {
        return ObsClockSyncVerdict::RejectStaleOrFuture;
    }

    // 3. Per-sender monotonic replay gate in peer's raw clock domain (slew-invariant!)
    if let Some(prev_ts) = prev_raw_ts {
        if obs_raw_ts_us < prev_ts {
            return ObsClockSyncVerdict::RejectReplay;
        }
    }

    ObsClockSyncVerdict::Accept(aligned_ts_us)
}

/// Output of one deterministic `StatefulMeshDaaGuard` tick (`fuse_guarded` or
/// `fuse_guarded_with_clock_sync`).
#[derive(Debug, Clone, Copy)]
pub struct GuardedMeshFusionResult {
    pub tracks: [FusedMeshTrack; MAX_FUSED_TRACKS],
    pub num_fused: usize,
    pub ghosts_rejected_3ship: u32,
    pub sticky_ghosts_suppressed: u32,
    pub stale_obs_rejected: u32,
    pub replay_obs_rejected: u32,
    pub uncalibrated_obs_rejected: u32,
    pub fresh_sender_count: usize,
    pub sticky_triple_ray_active: bool,
    pub fallback_to_tier_b_optical: bool,
}

/// Stateful `#![no_std]` wrapper over frozen `MeshDaaFuser` closing DR-22 Finding F4.
#[derive(Debug, Clone, Copy)]
pub struct StatefulMeshDaaGuard {
    fuser: MeshDaaFuser,
    max_age_us: u64,
    max_future_skew_us: u64,
    hysteresis_us: u64,
    peers: [PeerWatermark; MAX_TRACKED_PEERS],
    last_3ship_utc_us: Option<u64>,
    anchors: [ConfirmedAnchor; MAX_FUSED_TRACKS],
}

impl Default for StatefulMeshDaaGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl StatefulMeshDaaGuard {
    pub const fn new() -> Self {
        Self {
            fuser: MeshDaaFuser::new(),
            max_age_us: MAX_MESH_AGE_US,
            max_future_skew_us: MAX_FUTURE_SKEW_US,
            hysteresis_us: TRIPLE_RAY_HYSTERESIS_US,
            peers: [PeerWatermark::EMPTY; MAX_TRACKED_PEERS],
            last_3ship_utc_us: None,
            anchors: [ConfirmedAnchor::EMPTY; MAX_FUSED_TRACKS],
        }
    }

    pub const fn with_fuser(fuser: MeshDaaFuser) -> Self {
        Self {
            fuser,
            max_age_us: MAX_MESH_AGE_US,
            max_future_skew_us: MAX_FUTURE_SKEW_US,
            hysteresis_us: TRIPLE_RAY_HYSTERESIS_US,
            peers: [PeerWatermark::EMPTY; MAX_TRACKED_PEERS],
            last_3ship_utc_us: None,
            anchors: [ConfirmedAnchor::EMPTY; MAX_FUSED_TRACKS],
        }
    }

    /// Filter incoming `#52001` observations for freshness (`<= 200 ms` age, `<= 50 ms` future skew)
    /// and per-sender monotonic timestamp non-replay, committing accepted peer watermarks.
    pub(crate) fn filter_fresh_observations(
        &mut self,
        obs: &[OpenXvsBearingTrack],
        now_utc_us: u64,
    ) -> ([OpenXvsBearingTrack; MAX_GUARD_OBS], usize, u32, u32, usize) {
        let mut fresh_buf = [empty_bearing_track(); MAX_GUARD_OBS];
        let mut n_fresh = 0usize;
        let mut stale_obs_rejected = 0u32;
        let mut replay_obs_rejected = 0u32;

        // Track maximum accepted timestamp per sender within this tick (allows multiple
        // simultaneous target rays from the same sender at identical `time_utc_us`).
        let mut batch_senders = [0u8; MAX_TRACKED_PEERS];
        let mut batch_max_ts = [0u64; MAX_TRACKED_PEERS];
        let mut n_batch_senders = 0usize;

        let commit_limit = if obs.len() < MAX_TRACKED_PEERS {
            obs.len()
        } else {
            MAX_TRACKED_PEERS
        };

        for o in obs {
            // 1. Future-skew & >200 ms staleness gate
            let future_viol = o.time_utc_us > now_utc_us.saturating_add(self.max_future_skew_us);
            let stale_viol = now_utc_us > o.time_utc_us.saturating_add(self.max_age_us);
            if future_viol || stale_viol {
                stale_obs_rejected = stale_obs_rejected.saturating_add(1);
                continue;
            }

            // 2. Per-sender monotonic replay gate (must be >= last committed tick timestamp)
            if let Some(prev_ts) = self.peer_last_ts(o.sender_id) {
                if o.time_utc_us < prev_ts {
                    replay_obs_rejected = replay_obs_rejected.saturating_add(1);
                    continue;
                }
            }

            if n_fresh < MAX_GUARD_OBS {
                fresh_buf[n_fresh] = *o;
                n_fresh += 1;
            }

            // Register distinct fresh sender
            let mut found = false;
            let mut k = 0usize;
            while k < n_batch_senders && k < commit_limit {
                if batch_senders[k] == o.sender_id {
                    if o.time_utc_us > batch_max_ts[k] {
                        batch_max_ts[k] = o.time_utc_us;
                    }
                    found = true;
                    break;
                }
                k += 1;
            }
            if !found && n_batch_senders < commit_limit {
                batch_senders[n_batch_senders] = o.sender_id;
                batch_max_ts[n_batch_senders] = o.time_utc_us;
                n_batch_senders += 1;
            }
        }

        // Commit updated monotonic timestamps for accepted fresh senders
        let mut k = 0usize;
        while k < n_batch_senders && k < commit_limit {
            self.update_peer_ts(batch_senders[k], batch_max_ts[k]);
            k += 1;
        }

        (
            fresh_buf,
            n_fresh,
            stale_obs_rejected,
            replay_obs_rejected,
            n_batch_senders,
        )
    }

    /// Enforce sticky triple-ray ghost suppression across transient `3 -> 2` peer drops
    /// within `TRIPLE_RAY_HYSTERESIS_US` (`500 ms`) after the latest 3-ship confirmation tick.
    pub(crate) fn apply_sticky_hysteresis(
        &mut self,
        raw: &MeshFusionResult,
        n_batch_senders: usize,
        stale_obs_rejected: u32,
        replay_obs_rejected: u32,
        now_utc_us: u64,
    ) -> GuardedMeshFusionResult {
        if n_batch_senders >= 3 {
            self.last_3ship_utc_us = Some(now_utc_us);
            let mut a_idx = 0usize;
            while a_idx < MAX_FUSED_TRACKS {
                self.anchors[a_idx] = ConfirmedAnchor::EMPTY;
                a_idx += 1;
            }
            let mut out_idx = 0usize;
            let mut i = 0usize;
            while i < raw.num_fused && i < MAX_FUSED_TRACKS && out_idx < MAX_FUSED_TRACKS {
                if raw.tracks[i].valid && raw.tracks[i].triple_ray_confirmed {
                    self.anchors[out_idx] = ConfirmedAnchor {
                        valid: true,
                        pos_ecef: raw.tracks[i].pos_ecef,
                        confirmed_utc_us: now_utc_us,
                    };
                    out_idx += 1;
                }
                i += 1;
            }
            return GuardedMeshFusionResult {
                tracks: raw.tracks,
                num_fused: raw.num_fused,
                ghosts_rejected_3ship: raw.ghosts_rejected,
                sticky_ghosts_suppressed: 0,
                stale_obs_rejected,
                replay_obs_rejected,
                uncalibrated_obs_rejected: 0,
                fresh_sender_count: n_batch_senders,
                sticky_triple_ray_active: false,
                fallback_to_tier_b_optical: false,
            };
        }

        let sticky_active = match self.last_3ship_utc_us {
            Some(t3) => now_utc_us >= t3 && (now_utc_us - t3) <= self.hysteresis_us,
            None => false,
        };

        if !sticky_active {
            return GuardedMeshFusionResult {
                tracks: raw.tracks,
                num_fused: raw.num_fused,
                ghosts_rejected_3ship: raw.ghosts_rejected,
                sticky_ghosts_suppressed: 0,
                stale_obs_rejected,
                replay_obs_rejected,
                uncalibrated_obs_rejected: 0,
                fresh_sender_count: n_batch_senders,
                sticky_triple_ray_active: false,
                fallback_to_tier_b_optical: false,
            };
        }

        // Filter 2-ship candidates against recently triple-ray-confirmed anchors
        let mut filtered_tracks = [FusedMeshTrack::default(); MAX_FUSED_TRACKS];
        let mut num_kept = 0usize;
        let mut sticky_ghosts_suppressed = 0u32;

        let mut i = 0usize;
        while i < raw.num_fused && i < MAX_FUSED_TRACKS {
            let cand = raw.tracks[i];
            if cand.valid && self.matches_confirmed_anchor(cand.pos_ecef, now_utc_us) {
                if num_kept < MAX_FUSED_TRACKS {
                    filtered_tracks[num_kept] = FusedMeshTrack {
                        triple_ray_confirmed: true,
                        ..cand
                    };
                    num_kept += 1;
                }
            } else if cand.valid {
                sticky_ghosts_suppressed = sticky_ghosts_suppressed.saturating_add(1);
            }
            i += 1;
        }

        GuardedMeshFusionResult {
            tracks: filtered_tracks,
            num_fused: num_kept,
            ghosts_rejected_3ship: raw.ghosts_rejected,
            sticky_ghosts_suppressed,
            stale_obs_rejected,
            replay_obs_rejected,
            uncalibrated_obs_rejected: 0,
            fresh_sender_count: n_batch_senders,
            sticky_triple_ray_active: true,
            fallback_to_tier_b_optical: false,
        }
    }

    /// Filter incoming `#52001` observations using GNSS-denied two-way RF peer clock discipline
    /// (`PeerClockOffsetEstimator`), enforcing freshness in local-aligned time and monotonic
    /// replay rejection in each peer's raw clock domain (strictly invariant to offset slewing).
    pub(crate) fn filter_fresh_observations_with_clock_sync(
        &mut self,
        obs: &[OpenXvsBearingTrack],
        now_local_us: u64,
        own_sender_id: u8,
        gnss_denied: bool,
        clock_sync: &PeerClockOffsetEstimator,
    ) -> (
        [OpenXvsBearingTrack; MAX_GUARD_OBS],
        usize,
        u32,
        u32,
        u32,
        usize,
        bool,
    ) {
        let mut fresh_buf = [empty_bearing_track(); MAX_GUARD_OBS];
        let mut n_fresh = 0usize;
        let mut stale_obs_rejected = 0u32;
        let mut replay_obs_rejected = 0u32;
        let mut uncalibrated_obs_rejected = 0u32;
        let mut any_peer_uncalibrated = false;

        let mut batch_senders = [0u8; MAX_TRACKED_PEERS];
        let mut batch_max_raw_ts = [0u64; MAX_TRACKED_PEERS];
        let mut n_batch_senders = 0usize;

        let commit_limit = if obs.len() < MAX_TRACKED_PEERS {
            obs.len()
        } else {
            MAX_TRACKED_PEERS
        };

        for o in obs {
            let peer_est = if o.sender_id == own_sender_id {
                PeerClockEstimate::UNCALIBRATED
            } else {
                clock_sync.estimate_peer(o.sender_id, now_local_us)
            };
            let prev_ts = self.peer_last_ts(o.sender_id);
            let aligned_ts_us = match evaluate_obs_clock_synced(
                o.sender_id,
                o.time_utc_us,
                now_local_us,
                own_sender_id,
                gnss_denied,
                peer_est,
                prev_ts,
                self.max_age_us,
                self.max_future_skew_us,
            ) {
                ObsClockSyncVerdict::Accept(ts) => ts,
                ObsClockSyncVerdict::RejectUncalibrated => {
                    uncalibrated_obs_rejected = uncalibrated_obs_rejected.saturating_add(1);
                    any_peer_uncalibrated = true;
                    continue;
                }
                ObsClockSyncVerdict::RejectStaleOrFuture => {
                    stale_obs_rejected = stale_obs_rejected.saturating_add(1);
                    continue;
                }
                ObsClockSyncVerdict::RejectReplay => {
                    replay_obs_rejected = replay_obs_rejected.saturating_add(1);
                    continue;
                }
            };

            if n_fresh < MAX_GUARD_OBS {
                let mut aligned_obs = *o;
                aligned_obs.time_utc_us = aligned_ts_us;
                fresh_buf[n_fresh] = aligned_obs;
                n_fresh += 1;
            }

            // Register distinct fresh sender and raw high-water mark
            let mut found = false;
            let mut k = 0usize;
            while k < n_batch_senders && k < commit_limit {
                if batch_senders[k] == o.sender_id {
                    if o.time_utc_us > batch_max_raw_ts[k] {
                        batch_max_raw_ts[k] = o.time_utc_us;
                    }
                    found = true;
                    break;
                }
                k += 1;
            }
            if !found && n_batch_senders < commit_limit {
                batch_senders[n_batch_senders] = o.sender_id;
                batch_max_raw_ts[n_batch_senders] = o.time_utc_us;
                n_batch_senders += 1;
            }
        }

        // Commit updated monotonic raw peer timestamps for accepted fresh senders
        let mut k = 0usize;
        while k < n_batch_senders && k < commit_limit {
            self.update_peer_ts(batch_senders[k], batch_max_raw_ts[k]);
            k += 1;
        }

        (
            fresh_buf,
            n_fresh,
            stale_obs_rejected,
            replay_obs_rejected,
            uncalibrated_obs_rejected,
            n_batch_senders,
            any_peer_uncalibrated,
        )
    }

    /// Fail-closed sticky triple-ray hysteresis for `fuse_guarded_with_clock_sync`:
    /// NEVER admits unguarded 2-ship coplanar tracks when `< 3` calibrated senders are present
    /// outside the `500 ms` `TRIPLE_RAY_HYSTERESIS_US` window, and asserts
    /// `fallback_to_tier_b_optical = true` whenever clock discipline is uncalibrated or 3-ship
    /// dropout exceeds `TRIPLE_RAY_HYSTERESIS_US`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_sticky_hysteresis_clock_synced(
        &mut self,
        raw: &MeshFusionResult,
        n_batch_senders: usize,
        stale_obs_rejected: u32,
        replay_obs_rejected: u32,
        uncalibrated_obs_rejected: u32,
        any_peer_uncalibrated: bool,
        now_local_us: u64,
    ) -> GuardedMeshFusionResult {
        if n_batch_senders >= 3 {
            self.last_3ship_utc_us = Some(now_local_us);
            let mut a_idx = 0usize;
            while a_idx < MAX_FUSED_TRACKS {
                self.anchors[a_idx] = ConfirmedAnchor::EMPTY;
                a_idx += 1;
            }
            let mut out_idx = 0usize;
            let mut i = 0usize;
            while i < raw.num_fused && i < MAX_FUSED_TRACKS && out_idx < MAX_FUSED_TRACKS {
                if raw.tracks[i].valid && raw.tracks[i].triple_ray_confirmed {
                    self.anchors[out_idx] = ConfirmedAnchor {
                        valid: true,
                        pos_ecef: raw.tracks[i].pos_ecef,
                        confirmed_utc_us: now_local_us,
                    };
                    out_idx += 1;
                }
                i += 1;
            }
            return GuardedMeshFusionResult {
                tracks: raw.tracks,
                num_fused: raw.num_fused,
                ghosts_rejected_3ship: raw.ghosts_rejected,
                sticky_ghosts_suppressed: 0,
                stale_obs_rejected,
                replay_obs_rejected,
                uncalibrated_obs_rejected,
                fresh_sender_count: n_batch_senders,
                sticky_triple_ray_active: false,
                fallback_to_tier_b_optical: any_peer_uncalibrated,
            };
        }

        let sticky_active = match self.last_3ship_utc_us {
            Some(t3) => now_local_us >= t3 && (now_local_us - t3) <= self.hysteresis_us,
            None => false,
        };

        if sticky_active {
            let mut filtered_tracks = [FusedMeshTrack::default(); MAX_FUSED_TRACKS];
            let mut num_kept = 0usize;
            let mut sticky_ghosts_suppressed = 0u32;

            let mut i = 0usize;
            while i < raw.num_fused && i < MAX_FUSED_TRACKS {
                let cand = raw.tracks[i];
                if cand.valid && self.matches_confirmed_anchor(cand.pos_ecef, now_local_us) {
                    if num_kept < MAX_FUSED_TRACKS {
                        filtered_tracks[num_kept] = FusedMeshTrack {
                            triple_ray_confirmed: true,
                            ..cand
                        };
                        num_kept += 1;
                    }
                } else if cand.valid {
                    sticky_ghosts_suppressed = sticky_ghosts_suppressed.saturating_add(1);
                }
                i += 1;
            }

            return GuardedMeshFusionResult {
                tracks: filtered_tracks,
                num_fused: num_kept,
                ghosts_rejected_3ship: raw.ghosts_rejected,
                sticky_ghosts_suppressed,
                stale_obs_rejected,
                replay_obs_rejected,
                uncalibrated_obs_rejected,
                fresh_sender_count: n_batch_senders,
                sticky_triple_ray_active: true,
                fallback_to_tier_b_optical: any_peer_uncalibrated,
            };
        }

        // Fail-closed: < 3 senders and outside 500 ms triple-ray hysteresis window.
        // Suppress ALL unconfirmed 2-ship tracks to eliminate coplanar ghost re-admission
        // and signal mandatory fallback to Tier-B optical-only ActiveRange!
        let mut unconfirmed_suppressed = 0u32;
        let mut i = 0usize;
        while i < raw.num_fused && i < MAX_FUSED_TRACKS {
            if raw.tracks[i].valid {
                unconfirmed_suppressed = unconfirmed_suppressed.saturating_add(1);
            }
            i += 1;
        }

        GuardedMeshFusionResult {
            tracks: [FusedMeshTrack::default(); MAX_FUSED_TRACKS],
            num_fused: 0,
            ghosts_rejected_3ship: raw.ghosts_rejected,
            sticky_ghosts_suppressed: unconfirmed_suppressed,
            stale_obs_rejected,
            replay_obs_rejected,
            uncalibrated_obs_rejected,
            fresh_sender_count: n_batch_senders,
            sticky_triple_ray_active: false,
            fallback_to_tier_b_optical: true,
        }
    }

    /// GNSS-denied clock-synchronized guarded MeshDAA fusion (`REALWORLD_DEPLOYABILITY_AUDIT_V5.md`
    /// Failure Mode 7 fix).
    ///
    /// Applies calibrated two-way RF peer clock offsets from `clock_sync` when synchronized
    /// (`offset_uncert_us <= 20 ms`), preserving 100% rejection of genuine `> 200 ms` stale /
    /// replay attacks while eliminating false peer dropouts under multi-hour TCXO drift (`+120 ms+`).
    /// Whenever any peer clock is uncalibrated during `gnss_denied == true`, or whenever 3-ship
    /// dropout exceeds `TRIPLE_RAY_HYSTERESIS_US` (`500 ms`), sets `fallback_to_tier_b_optical = true`
    /// and NEVER admits unguarded 2-ship coplanar ghosts.
    pub fn fuse_guarded_with_clock_sync(
        &mut self,
        obs: &[OpenXvsBearingTrack],
        now_local_us: u64,
        own_sender_id: u8,
        gnss_denied: bool,
        clock_sync: &PeerClockOffsetEstimator,
    ) -> GuardedMeshFusionResult {
        let (
            fresh_buf,
            n_fresh,
            stale_obs_rejected,
            replay_obs_rejected,
            uncalibrated_obs_rejected,
            n_batch_senders,
            any_peer_uncalibrated,
        ) = self.filter_fresh_observations_with_clock_sync(
            obs,
            now_local_us,
            own_sender_id,
            gnss_denied,
            clock_sync,
        );

        let raw = if obs.len() < 2 || n_fresh < 2 {
            MeshFusionResult {
                tracks: [FusedMeshTrack::default(); MAX_FUSED_TRACKS],
                num_fused: 0,
                ghosts_rejected: 0,
            }
        } else {
            self.fuser.fuse(&fresh_buf[..n_fresh])
        };

        self.apply_sticky_hysteresis_clock_synced(
            &raw,
            n_batch_senders,
            stale_obs_rejected,
            replay_obs_rejected,
            uncalibrated_obs_rejected,
            any_peer_uncalibrated,
            now_local_us,
        )
    }

    /// Filter incoming `#52001` observations for freshness & monotonicity, execute frozen
    /// `MeshDaaFuser::fuse`, and enforce sticky triple-ray ghost suppression across brief
    /// `3 <-> 2` peer drops.
    pub fn fuse_guarded(
        &mut self,
        obs: &[OpenXvsBearingTrack],
        now_utc_us: u64,
    ) -> GuardedMeshFusionResult {
        let (fresh_buf, n_fresh, stale_obs_rejected, replay_obs_rejected, n_batch_senders) =
            self.filter_fresh_observations(obs, now_utc_us);

        // 3. Invoke unmodified frozen MeshDaaFuser::fuse on fresh, monotonic subset
        //    (triangulation requires >= 2 rays; short-circuit avoids empty-batch overhead)
        let raw = if obs.len() < 2 || n_fresh < 2 {
            MeshFusionResult {
                tracks: [FusedMeshTrack::default(); MAX_FUSED_TRACKS],
                num_fused: 0,
                ghosts_rejected: 0,
            }
        } else {
            self.fuser.fuse(&fresh_buf[..n_fresh])
        };

        // 4. Sticky triple-ray ghost hysteresis
        self.apply_sticky_hysteresis(
            &raw,
            n_batch_senders,
            stale_obs_rejected,
            replay_obs_rejected,
            now_utc_us,
        )
    }

    fn matches_confirmed_anchor(&self, pos_ecef: Vec3, now_utc_us: u64) -> bool {
        let mut i = 0usize;
        while i < MAX_FUSED_TRACKS {
            let anc = self.anchors[i];
            if anc.valid && now_utc_us >= anc.confirmed_utc_us {
                let dt_us = now_utc_us - anc.confirmed_utc_us;
                if dt_us <= self.hysteresis_us {
                    let dt_s = (dt_us as f64) * 1.0e-6;
                    let gate_m = self.fuser.merge_radius_m + MAX_CIVIL_INTRUDER_SPEED_MPS * dt_s;
                    let delta = sub3(pos_ecef, anc.pos_ecef);
                    let dist_sq = dot3(delta, delta);
                    if gate_m >= 0.0 && dist_sq.is_finite() && dist_sq <= gate_m * gate_m {
                        return true;
                    }
                }
            }
            i += 1;
        }
        false
    }

    fn peer_last_ts(&self, sender_id: u8) -> Option<u64> {
        let mut i = 0usize;
        while i < MAX_TRACKED_PEERS {
            if self.peers[i].used && self.peers[i].sender_id == sender_id {
                return Some(self.peers[i].last_ts_utc_us);
            }
            i += 1;
        }
        None
    }

    fn update_peer_ts(&mut self, sender_id: u8, ts_utc_us: u64) {
        let mut free_idx: Option<usize> = None;
        let mut i = 0usize;
        while i < MAX_TRACKED_PEERS {
            if self.peers[i].used && self.peers[i].sender_id == sender_id {
                if ts_utc_us > self.peers[i].last_ts_utc_us {
                    self.peers[i].last_ts_utc_us = ts_utc_us;
                }
                return;
            }
            if !self.peers[i].used && free_idx.is_none() {
                free_idx = Some(i);
            }
            i += 1;
        }
        if let Some(idx) = free_idx {
            self.peers[idx] = PeerWatermark {
                used: true,
                sender_id,
                last_ts_utc_us: ts_utc_us,
            };
        }
    }
}

pub(crate) const fn empty_bearing_track() -> OpenXvsBearingTrack {
    OpenXvsBearingTrack {
        time_utc_us: 0,
        own_ecef_cm: [0; 3],
        bearing_ecef_q15: [0; 3],
        bearing_rate_urad_s: [0; 3],
        cone_half_angle_urad: 0,
        own_pos_std_cm: 0,
        track_id: 0,
        sender_id: 0,
        flags: 0,
        cov_diag_packed: [0; 6],
    }
}
