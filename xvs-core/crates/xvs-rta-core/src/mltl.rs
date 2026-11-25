//! Past-time Mission-time Linear Temporal Logic (MLTL) Visual-ODD monitor (`SP1_LIVEDECK_DESIGN.md` §4.1.2).
//!
//! Zero heap allocation, strictly causal (0-tick preemption), evaluated over fixed `[bool; 64]`
//! ring buffers in integer ticks (`H[0, n]`, `O[0, n]`, `S[a, b]`).

use crate::odd_generated::{
    CAM_COUNT_HISTORY_WINDOW, EPIPOLAR_CAM_MAX_URAD, EPI_ONCE_WINDOW, MIN_HEALTHY_CAMS,
    MLTL_RING_CAPACITY, SPOOF_ONCE_WINDOW, WINDOW_DROPOUT_20_MAX_TICKS, WINDOW_DROPOUT_LEN_TICKS,
};

const RING_MASK: usize = MLTL_RING_CAPACITY - 1;

/// Fixed 64-tick boolean ring buffer with saturating tick counter (never panics or wraps).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoolRing64 {
    buf: [bool; MLTL_RING_CAPACITY],
    head: usize,
    len: u64,
}

impl Default for BoolRing64 {
    fn default() -> Self {
        Self::new()
    }
}

impl BoolRing64 {
    pub const fn new() -> Self {
        Self {
            buf: [false; MLTL_RING_CAPACITY],
            head: 0,
            len: 0,
        }
    }

    /// Push a new boolean sample for current tick `t` (becomes `k = 0` ago).
    pub fn push(&mut self, val: bool) {
        let slot = self.head & RING_MASK;
        self.buf[slot] = val;
        self.head = self.head.wrapping_add(1) & RING_MASK;
        self.len = self.len.saturating_add(1);
    }

    pub const fn len(&self) -> u64 {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of valid retained samples in `[0, 64]`.
    #[inline]
    pub const fn available(&self) -> usize {
        if self.len >= (MLTL_RING_CAPACITY as u64) {
            MLTL_RING_CAPACITY
        } else {
            self.len as usize
        }
    }

    /// Return the boolean sample recorded `k` ticks ago (`k = 0` is most recent).
    #[inline]
    pub fn get_ago(&self, k: usize) -> Option<bool> {
        if k >= self.available() {
            return None;
        }
        let idx = self
            .head
            .wrapping_add(MLTL_RING_CAPACITY)
            .wrapping_sub(1)
            .wrapping_sub(k)
            & RING_MASK;
        Some(self.buf[idx])
    }

    /// Past-time **Historically** `H[0, n](p)`: `p` held at every tick in `[t - n, t]`.
    /// When `t < n` (`available <= n`), evaluates over `[0, t]` unless `require_warmup` is true.
    pub fn historically(&self, n: usize, require_warmup: bool) -> bool {
        let avail = self.available();
        if avail == 0 {
            return false;
        }
        if require_warmup && avail <= n {
            return false;
        }
        let max_k = if n < avail { n } else { avail.saturating_sub(1) };
        let mut k: usize = 0;
        while k <= max_k {
            if let Some(v) = self.get_ago(k) {
                if !v {
                    return false;
                }
            } else {
                return false;
            }
            if k == usize::MAX {
                break;
            }
            k = k.saturating_add(1);
        }
        true
    }

    /// Past-time **Once** `O[0, n](p)`: `p` held at least once in `[t - n, t]`.
    pub fn once(&self, n: usize) -> bool {
        let avail = self.available();
        if avail == 0 {
            return false;
        }
        let max_k = if n < avail { n } else { avail.saturating_sub(1) };
        let mut k: usize = 0;
        while k <= max_k {
            if let Some(true) = self.get_ago(k) {
                return true;
            }
            if k == usize::MAX {
                break;
            }
            k = k.saturating_add(1);
        }
        false
    }

    /// Past-time **Since** `S[a, b](p, q)`: there exists `k in [a, b]` such that `q` held `k`
    /// ticks ago and `p` held at every tick `j in [0, k)`.
    pub fn since(p: &Self, q: &Self, a: usize, b: usize) -> bool {
        if a > b {
            return false;
        }
        let avail_p = p.available();
        let avail_q = q.available();
        let avail = if avail_p < avail_q { avail_p } else { avail_q };
        if avail == 0 || a >= avail {
            return false;
        }
        let max_k = if b < avail { b } else { avail.saturating_sub(1) };

        let mut k: usize = 0;
        let mut p_all_since = true;
        while k <= max_k {
            if k >= a && p_all_since {
                if let Some(true) = q.get_ago(k) {
                    return true;
                }
            }
            match p.get_ago(k) {
                Some(true) => {}
                _ => {
                    p_all_since = false;
                }
            }
            if !p_all_since {
                break;
            }
            if k == usize::MAX {
                break;
            }
            k = k.saturating_add(1);
        }
        false
    }

    /// Count how many ticks were `true` within the last `window` ticks (`[t - window + 1, t]`).
    pub fn count_true_last(&self, window: usize) -> u8 {
        let avail = self.available();
        let limit = if window < avail { window } else { avail };
        let mut count: u8 = 0;
        let mut k: usize = 0;
        while k < limit {
            if let Some(true) = self.get_ago(k) {
                count = count.saturating_add(1);
            }
            k = k.saturating_add(1);
        }
        count
    }
}

/// Alias exported per Task 1 interface specification.
pub type PastMltlMonitor = OddMltlMonitor;

/// Per-tick output of the compiled past-time MLTL ODD monitor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OddMltlOutcome {
    pub mltl_cam_fault_mask: u8,
    pub healthy_mask: u8,
    pub n_cams: u8,
    pub r_cam_count_ok: bool,
    pub spoof_tripped: bool,
    pub window_dropout_20: u8,
    pub critical_violation: bool,
}

/// Compiled Visual-ODD past-time MLTL monitor holding bounded 64-tick ring buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OddMltlMonitor {
    cam_count_ge_4: BoolRing64,
    epipolar_exceeded: [BoolRing64; 6],
    spoof_ring: BoolRing64,
    dropout_ring: BoolRing64,
}

impl Default for OddMltlMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl OddMltlMonitor {
    pub const fn new() -> Self {
        Self {
            cam_count_ge_4: BoolRing64::new(),
            epipolar_exceeded: [BoolRing64::new(); 6],
            spoof_ring: BoolRing64::new(),
            dropout_ring: BoolRing64::new(),
        }
    }

    /// Evaluate all compiled `v2_m1_odd.yaml` past-time MLTL rules for the current tick.
    pub fn eval_tick(
        &mut self,
        cam_health_mask: u8,
        ood_exceeded_mask: u8,
        epipolar_max_urad: &[i64; 6],
        spoof_alarm: bool,
        transient_dropout: bool,
    ) -> OddMltlOutcome {
        // 1. R_EPI_SINGLE_CAM[i]: O[0, 1](epipolar_max_urad[i] > 6_109)
        let mut mltl_cam_fault_mask: u8 = 0;
        let mut cam_i: usize = 0;
        while cam_i < 6 {
            let over = epipolar_max_urad[cam_i] > EPIPOLAR_CAM_MAX_URAD;
            self.epipolar_exceeded[cam_i].push(over);
            if self.epipolar_exceeded[cam_i].once(EPI_ONCE_WINDOW) {
                mltl_cam_fault_mask |= 1_u8 << cam_i;
            }
            cam_i = cam_i.saturating_add(1);
        }

        // 2. Effective 6-bit camera health mask (bits 0..5 ordered F, R, B, L, U, D)
        let healthy_mask = cam_health_mask & !ood_exceeded_mask & !mltl_cam_fault_mask & 0x3F;
        let n_cams = healthy_mask.count_ones() as u8;

        // 3. R_CAM_COUNT: H[0, 2](active_healthy_cams >= 4)
        self.cam_count_ge_4.push(n_cams >= MIN_HEALTHY_CAMS);
        let r_cam_count_ok = self
            .cam_count_ge_4
            .historically(CAM_COUNT_HISTORY_WINDOW, false);

        // 4. R_SPOOF_LOCK: O[0, 0](spoof_alarm)
        self.spoof_ring.push(spoof_alarm);
        let spoof_tripped = self.spoof_ring.once(SPOOF_ONCE_WINDOW);

        // 5. R_DROPOUT_DUTY_20: count(transient_dropout, 20) <= 10 (<= 50% vs >= 55% -> P0)
        self.dropout_ring.push(transient_dropout);
        let window_dropout_20 = self.dropout_ring.count_true_last(WINDOW_DROPOUT_LEN_TICKS);

        let critical_violation =
            !r_cam_count_ok || (window_dropout_20 > WINDOW_DROPOUT_20_MAX_TICKS);

        OddMltlOutcome {
            mltl_cam_fault_mask,
            healthy_mask,
            n_cams,
            r_cam_count_ok,
            spoof_tripped,
            window_dropout_20,
            critical_violation,
        }
    }

    /// Expose tick count of the underlying dropout ring for diagnostics and Kani proofs.
    pub const fn ticks_seen(&self) -> u64 {
        self.dropout_ring.len()
    }
}
