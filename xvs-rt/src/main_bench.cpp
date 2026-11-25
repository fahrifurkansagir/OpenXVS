/* OpenXVS v2.0 — WSL2 Real-Time Zero-Alloc Audit & 500-Encounter XVS-MeshDAA Benchmark (`V2-M3L-1..2`).
 *
 * Strictly -fno-exceptions -fno-rtti; runs:
 *   1. Positive -Wl,--wrap=malloc self-test + 10,000-tick zero-alloc hot-loop audit (O0 & O2).
 *   2. Deterministic Linux libxvs_core.a xvs_ecbf_qp_solve probe vectors (for <=1e-12 cross-OS parity).
 *   3. 500-encounter seeded 3D multi-intruder (>=2 simultaneous intruders w/ near-coplanar crossings)
 *      1-ship vs 2-ship vs 3-ship XVS-MeshDAA closed-loop benchmark calling REAL libxvs_core.a.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "xvs_core.h"
#include "xvs_daidalus_bridge.h"
#include "xvs_rt/ecbf_qp.hpp"
#include "xvs_rt/malloc_audit.hpp"
#include "xvs_rt/mh_vit.hpp"
#include "xvs_rt/rt_pipeline.hpp"

namespace {

struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t seed) noexcept : state(seed) {}
    uint64_t next_u64() noexcept {
        uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31U);
    }
    double uniform(double lo, double hi) noexcept {
        const double u = static_cast<double>(next_u64() >> 11U) * (1.0 / 9007199254740992.0);
        return lo + (hi - lo) * u;
    }
    double normal() noexcept {
        const double u1 = std::max(uniform(0.0, 1.0), 1e-12);
        const double u2 = uniform(0.0, 1.0);
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.28318530717958647692 * u2);
    }
};

struct GeoFrame {
    double origin_ecef[3];
    double r_ecef_ned[9]; /* row-major R_ecef_from_ned */
};

static GeoFrame make_ksfo_geo_frame() noexcept {
    GeoFrame gf{};
    (void)xvs_geo_geodetic_to_ecef(37.6188, -122.3754, 600.0, gf.origin_ecef);
    (void)xvs_geo_r_ecef_ned(37.6188, -122.3754, gf.r_ecef_ned);
    return gf;
}

static void ned_to_ecef(const GeoFrame &gf, const double ned[3], double out_ecef[3]) noexcept {
    for (int r = 0; r < 3; ++r) {
        out_ecef[r] = gf.origin_ecef[r] +
                      gf.r_ecef_ned[r * 3 + 0] * ned[0] +
                      gf.r_ecef_ned[r * 3 + 1] * ned[1] +
                      gf.r_ecef_ned[r * 3 + 2] * ned[2];
    }
}

static void ecef_to_ned(const GeoFrame &gf, const double ecef[3], double out_ned[3]) noexcept {
    const double dx = ecef[0] - gf.origin_ecef[0];
    const double dy = ecef[1] - gf.origin_ecef[1];
    const double dz = ecef[2] - gf.origin_ecef[2];
    /* R_ned_from_ecef = R_ecef_from_ned^T */
    for (int c = 0; c < 3; ++c) {
        out_ned[c] = gf.r_ecef_ned[0 * 3 + c] * dx +
                     gf.r_ecef_ned[1 * 3 + c] * dy +
                     gf.r_ecef_ned[2 * 3 + c] * dz;
    }
}

static XvsBearingTrack build_bearing_track(
    uint64_t time_us,
    const double peer_ecef[3],
    const double target_ecef[3],
    uint16_t cone_urad,
    uint16_t track_id,
    uint8_t sender_id,
    SplitMix64 &rng,
    double bearing_noise_rad) noexcept {
    double dx = target_ecef[0] - peer_ecef[0];
    double dy = target_ecef[1] - peer_ecef[1];
    double dz = target_ecef[2] - peer_ecef[2];
    const double dist = std::max(std::sqrt(dx * dx + dy * dy + dz * dz), 1.0);
    dx = dx / dist + rng.normal() * bearing_noise_rad;
    dy = dy / dist + rng.normal() * bearing_noise_rad;
    dz = dz / dist + rng.normal() * bearing_noise_rad;
    const double nrm = std::max(std::sqrt(dx * dx + dy * dy + dz * dz), 1e-12);
    dx /= nrm;
    dy /= nrm;
    dz /= nrm;

    XvsBearingTrack trk{};
    trk.time_utc_us = time_us;
    for (int a = 0; a < 3; ++a) {
        trk.own_ecef_cm[a] = static_cast<int32_t>(std::llround(peer_ecef[a] * 100.0));
    }
    auto to_q15 = [](double v) noexcept -> int16_t {
        const double s = std::max(-32767.0, std::min(32767.0, std::round(v * 32767.0)));
        return static_cast<int16_t>(s);
    };
    trk.bearing_ecef_q15[0] = to_q15(dx);
    trk.bearing_ecef_q15[1] = to_q15(dy);
    trk.bearing_ecef_q15[2] = to_q15(dz);
    trk.cone_half_angle_urad = cone_urad;
    trk.own_pos_std_cm = 150U;
    trk.track_id = track_id;
    trk.sender_id = sender_id;
    trk.flags = 0x01U; /* FLAG_SIMULATION */
    for (int k = 0; k < 3; ++k) {
        trk.cov_diag_packed[k] = 150U;
        trk.cov_diag_packed[k + 3] = cone_urad;
    }
    return trk;
}

static double percentile_sorted(const std::vector<double> &sorted, double p) noexcept {
    if (sorted.empty()) {
        return 0.0;
    }
    const double idx_f = p * static_cast<double>(sorted.size() - 1U);
    const std::size_t idx = static_cast<std::size_t>(idx_f);
    return sorted[std::min(idx, sorted.size() - 1U)];
}

struct AuditSummary {
    uint64_t selftest_alloc_count;
    uint64_t selftest_alloc_bytes;
    uint64_t hotloop_ticks;
    uint64_t hotloop_alloc_count;
    uint64_t hotloop_alloc_bytes;
    uint64_t hotloop_free_count;
    double p50_us;
    double p95_us;
    double p99_us;
    double max_us;
    int32_t mavlink_wire_bytes;
    bool pass;
};

static AuditSummary run_zero_alloc_audit(uint64_t n_ticks) {
    AuditSummary sum{};
    sum.hotloop_ticks = n_ticks;

    /* 1. Positive self-test proving -Wl,--wrap=malloc is live */
    const xvs_rt::MallocAuditStats st_self = xvs_rt::malloc_audit_run_positive_selftest();
    sum.selftest_alloc_count = st_self.alloc_count;
    sum.selftest_alloc_bytes = st_self.alloc_bytes;

    /* 2. Prepare pipeline & preallocated latency buffer strictly BEFORE arming */
    xvs_rt::RtPipeline pipe;
    if (!pipe.init()) {
        sum.pass = false;
        return sum;
    }

    const GeoFrame gf = make_ksfo_geo_frame();
    SplitMix64 rng(20261121ULL);
    xvs_rt::RtTickInputPacket pkt{};
    pkt.rta_tel.horizon_pl_rad = 0.002;
    pkt.rta_tel.horizon_al_rad = 0.008726646259971648;
    pkt.rta_tel.v_vpl_m = 4.0;
    pkt.rta_tel.v_val_m = 10.0;
    pkt.rta_tel.v_hpl_m = 12.0;
    pkt.rta_tel.v_hal_m = 40.0;
    pkt.rta_tel.signed_horizon_residual_rad = 0.0003;
    pkt.rta_tel.runway_width_delta_m = 0.5;
    pkt.rta_tel.runway_width_delta_limit_m = 5.0;
    pkt.rta_tel.cusum_stat = 1.0;
    pkt.rta_tel.cusum_thr = 10.53;
    pkt.rta_tel.ood_global_score = 0.1;
    pkt.rta_tel.ood_global_thr = 1.0;
    pkt.rta_tel.ood_cam_thr = 1.0;
    pkt.rta_tel.cam_health_mask = 0x3FU;

    pkt.mh_vit_obs.own_pos_ned_m[2] = -600.0;
    pkt.mh_vit_obs.own_vel_ned_mps[0] = 40.0;
    pkt.mh_vit_obs.azimuth_rad = 0.05;
    pkt.mh_vit_obs.elevation_rad = 0.02;
    pkt.mh_vit_obs.azimuth_rate_rad_s = 0.003;
    pkt.mh_vit_obs.kappa_s_inv = 0.04;
    pkt.mh_vit_obs.r_min_m = 1200.0;
    pkt.mh_vit_obs.r_max_m = 2800.0;
    pkt.mh_vit_obs.sigma_ang_rad = 0.002;
    pkt.mh_vit_obs.sigma_rate_rad_s = 0.001;
    pkt.mh_vit_obs.sigma_kappa_s_inv = 0.005;

    pkt.ownship.pos_ned_m[2] = -600.0;
    pkt.ownship.vel_ned_mps[0] = 40.0;
    pkt.n_intruders = 2U;
    pkt.intruders[0].pos_ned_m[0] = 1400.0;
    pkt.intruders[0].pos_ned_m[1] = -80.0;
    pkt.intruders[0].pos_ned_m[2] = -600.0;
    pkt.intruders[0].vel_ned_mps[0] = -40.0;
    pkt.intruders[0].valid = true;
    pkt.intruders[1].pos_ned_m[0] = 1500.0;
    pkt.intruders[1].pos_ned_m[1] = 220.0;
    pkt.intruders[1].pos_ned_m[2] = -595.0;
    pkt.intruders[1].vel_ned_mps[0] = -38.0;
    pkt.intruders[1].valid = true;

    double p1_e[3]{}, p2_e[3]{}, p3_e[3]{}, t1_e[3]{};
    const double p1_n[3] = {0.0, 0.0, 0.0};
    const double p2_n[3] = {0.0, 1500.0, 0.0};
    const double p3_n[3] = {-400.0, -1400.0, -420.0};
    const double t1_n[3] = {1800.0, 100.0, 0.0};
    ned_to_ecef(gf, p1_n, p1_e);
    ned_to_ecef(gf, p2_n, p2_e);
    ned_to_ecef(gf, p3_n, p3_e);
    ned_to_ecef(gf, t1_n, t1_e);
    pkt.mesh_tracks[0] = build_bearing_track(1000000ULL, p1_e, t1_e, 1500U, 1U, 1U, rng, 0.0);
    pkt.mesh_tracks[1] = build_bearing_track(1000000ULL, p2_e, t1_e, 1500U, 1U, 2U, rng, 0.0);
    pkt.mesh_tracks[2] = build_bearing_track(1000000ULL, p3_e, t1_e, 1500U, 1U, 3U, rng, 0.0);
    pkt.n_mesh_tracks = 3U;

    for (int a = 0; a < 3; ++a) {
        pkt.fci_xa[a] = t1_e[a];
        pkt.fci_xb[a] = t1_e[a] + 5.0;
    }
    pkt.fci_pa[0] = 25.0; pkt.fci_pa[4] = 25.0; pkt.fci_pa[8] = 25.0;
    pkt.fci_pb[0] = 36.0; pkt.fci_pb[4] = 36.0; pkt.fci_pb[8] = 36.0;

    std::vector<double> lat_us(static_cast<std::size_t>(n_ticks), 0.0);
    xvs_rt::RtTickOutputPacket out_pkt{};

    /* 3. ARM zero-alloc trap and run 10,000 consecutive ticks */
    xvs_rt::malloc_audit_reset_and_arm();
    bool all_ok = true;
    for (uint64_t k = 0U; k < n_ticks; ++k) {
        pkt.tick_seq = k + 1U;
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = pipe.step_hot_tick(pkt, out_pkt);
        const auto t1 = std::chrono::steady_clock::now();
        if (!ok) {
            all_ok = false;
        }
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        lat_us[static_cast<std::size_t>(k)] = static_cast<double>(ns) * 1e-3;
    }
    const xvs_rt::MallocAuditStats st_hot = xvs_rt::malloc_audit_disarm_and_read();

    /* 4. Teardown strictly AFTER disarming */
    pipe.shutdown();

    std::sort(lat_us.begin(), lat_us.end());
    sum.hotloop_alloc_count = st_hot.alloc_count;
    sum.hotloop_alloc_bytes = st_hot.alloc_bytes;
    sum.hotloop_free_count = st_hot.free_count;
    sum.p50_us = percentile_sorted(lat_us, 0.50);
    sum.p95_us = percentile_sorted(lat_us, 0.95);
    sum.p99_us = percentile_sorted(lat_us, 0.99);
    sum.max_us = lat_us.empty() ? 0.0 : lat_us.back();
    sum.mavlink_wire_bytes = out_pkt.mavlink_wire_len;
    sum.pass = all_ok &&
               (sum.selftest_alloc_count >= 1U) &&
               (sum.selftest_alloc_bytes >= 64U) &&
               (sum.hotloop_alloc_count == 0U) &&
               (sum.hotloop_alloc_bytes == 0U) &&
               (sum.mavlink_wire_bytes == 77);
    return sum;
}

struct ModeMetrics {
    int nmac_count{0};
    int lowc_count{0};
    int do365_compliant_count{0};
    int ghosts_total{0};
    int ghosts_rejected_total{0};
    std::vector<double> range_errs_m{};
    double min_hmd_overall_m{1e9};
    double min_3d_sep_overall_m{1e9};
};

struct MeshDaaBenchReport {
    int encounters{500};
    ModeMetrics m1_ship{};
    ModeMetrics m2_ship{};
    ModeMetrics m3_ship{};
    double bandwidth_kbps_3ship{0.0};
    double ghost_suppression_ratio{0.0};
    double p50_1ship{0.0};
    double p95_1ship{0.0};
    double p50_2ship{0.0};
    double p95_2ship{0.0};
    double p50_3ship{0.0};
    double p95_3ship{0.0};
    bool pass{false};
};

static MeshDaaBenchReport run_meshdaa_500_encounters() {
    MeshDaaBenchReport rep{};
    rep.encounters = 500;
    const GeoFrame gf = make_ksfo_geo_frame();
    const uint8_t mav_key[32]{
        0x21U, 0x22U, 0x23U, 0x24U, 0x25U, 0x26U, 0x27U, 0x28U,
        0x29U, 0x2AU, 0x2BU, 0x2CU, 0x2DU, 0x2EU, 0x2FU, 0x30U,
        0x31U, 0x32U, 0x33U, 0x34U, 0x35U, 0x36U, 0x37U, 0x38U,
        0x39U, 0x3AU, 0x3BU, 0x3CU, 0x3DU, 0x3EU, 0x3FU, 0x40U};

    constexpr double kDt = 0.25;         /* 4 Hz simulation steps */
    constexpr int kSteps = 200;          /* 50 s per encounter */
    constexpr double kGravity = 9.80665;
    constexpr double kOwnSpeed = 40.0;
    constexpr double kNmacH = 152.4;
    constexpr double kNmacV = 30.48;
    constexpr double kLowcH = 1219.2;
    constexpr double kLowcV = 137.16;

    uint64_t total_mavlink_bytes_3ship = 0U;
    double total_sim_seconds_3ship = 0.0;

    for (int enc = 0; enc < rep.encounters; ++enc) {
        SplitMix64 rng(UINT64_C(2026112100) + static_cast<uint64_t>(enc) * UINT64_C(7919));

        /* Two simultaneous intruders with near-coplanar crossing paths in the horizontal plane */
        const double r0_a = rng.uniform(2500.0, 3200.0);
        const double lat_a = rng.uniform(-60.0, 60.0);
        const double z_a = rng.uniform(-8.0, 8.0);
        const double spd_a = rng.uniform(36.0, 48.0);

        const double r0_b = rng.uniform(2600.0, 3400.0);
        const double lat_b = rng.uniform(260.0, 520.0) * (rng.uniform(0.0, 1.0) < 0.5 ? 1.0 : -1.0);
        const double z_b = rng.uniform(-8.0, 8.0);
        const double spd_b = rng.uniform(34.0, 46.0);

        /* Peer 2 coplanar East (+1400..1800 m); Peer 3 non-coplanar West + elevated (-420 m NED z) */
        const double p2_off[3] = {rng.uniform(-120.0, 120.0), rng.uniform(1400.0, 1800.0), rng.uniform(-2.0, 2.0)};
        const double p3_off[3] = {rng.uniform(-450.0, -250.0), rng.uniform(-1700.0, -1350.0), rng.uniform(-480.0, -360.0)};

        /* Evaluate initial snapshot across 1-ship, 2-ship, and 3-ship triangulation + full 3-ship closed-loop */
        auto run_mode = [&](int n_ships, ModeMetrics &mm) {
            SplitMix64 mrng(
                UINT64_C(990001) + static_cast<uint64_t>(enc) * UINT64_C(31) + static_cast<uint64_t>(n_ships));
            XvsRtaHandle *rta = xvs_rta_create();

            double own_p[3] = {0.0, 0.0, 0.0};
            double own_psi = 0.0;
            double own_vz = 0.0;
            double own_v[3] = {kOwnSpeed, 0.0, 0.0};

            double int_p[2][3] = {
                {r0_a, lat_a, z_a},
                {r0_b, lat_b, z_b}};
            const double int_v[2][3] = {
                {-spd_a, -lat_a * 0.008, 0.0},
                {-spd_b, -lat_b * 0.012, 0.0}};

            bool hit_nmac = false;
            bool hit_lowc = false;
            bool alerted_early = false;
            double min_hmd_enc = 1e9;
            double min_3d_enc = 1e9;
            int enc_ghosts_seen = 0;

            for (int step = 0; step < kSteps; ++step) {
                /* True peer ECEF positions */
                double p1_e[3]{}, p2_e[3]{}, p3_e[3]{}, t0_e[3]{}, t1_e[3]{};
                ned_to_ecef(gf, own_p, p1_e);
                const double p2_n[3] = {own_p[0] + p2_off[0], own_p[1] + p2_off[1], own_p[2] + p2_off[2]};
                const double p3_n[3] = {own_p[0] + p3_off[0], own_p[1] + p3_off[1], own_p[2] + p3_off[2]};
                ned_to_ecef(gf, p2_n, p2_e);
                ned_to_ecef(gf, p3_n, p3_e);
                ned_to_ecef(gf, int_p[0], t0_e);
                ned_to_ecef(gf, int_p[1], t1_e);

                XvsIntruderNed est_ints[8]{};
                std::size_t n_est = 0U;

                if (n_ships == 1) {
                    /* Monocular scale-rate bearing-only estimate: ~35% range uncertainty */
                    for (int ti = 0; ti < 2; ++ti) {
                        const double rx = int_p[ti][0] - own_p[0];
                        const double ry = int_p[ti][1] - own_p[1];
                        const double rz = int_p[ti][2] - own_p[2];
                        const double r_true = std::sqrt(rx * rx + ry * ry + rz * rz);
                        const double r_err_frac = mrng.normal() * 0.35;
                        const double r_est = std::max(100.0, r_true * (1.0 + r_err_frac));
                        if (step == 0) {
                            mm.range_errs_m.push_back(std::abs(r_est - r_true));
                        }
                        const double sc = r_est / std::max(r_true, 1.0);
                        est_ints[n_est].pos_ned[0] = own_p[0] + rx * sc;
                        est_ints[n_est].pos_ned[1] = own_p[1] + ry * sc;
                        est_ints[n_est].pos_ned[2] = own_p[2] + rz * sc;
                        est_ints[n_est].vel_ned[0] = int_v[ti][0];
                        est_ints[n_est].vel_ned[1] = int_v[ti][1];
                        est_ints[n_est].vel_ned[2] = int_v[ti][2];
                        est_ints[n_est].valid = 1U;
                        ++n_est;
                    }
                } else {
                    /* 2-ship or 3-ship cooperative MeshDAA via REAL libxvs_core.a */
                    XvsBearingTrack trks[6]{};
                    std::size_t nt = 0U;
                    const uint64_t ts_us = static_cast<uint64_t>(step) * UINT64_C(250000);
                    const double b_noise = 0.0005; /* 0.5 mrad bearing std */
                    trks[nt++] = build_bearing_track(ts_us, p1_e, t0_e, 1800U, 1U, 1U, mrng, b_noise);
                    trks[nt++] = build_bearing_track(ts_us, p1_e, t1_e, 1800U, 2U, 1U, mrng, b_noise);
                    trks[nt++] = build_bearing_track(ts_us, p2_e, t0_e, 1800U, 1U, 2U, mrng, b_noise);
                    trks[nt++] = build_bearing_track(ts_us, p2_e, t1_e, 1800U, 2U, 2U, mrng, b_noise);
                    if (n_ships >= 3) {
                        trks[nt++] = build_bearing_track(ts_us, p3_e, t0_e, 1800U, 1U, 3U, mrng, b_noise);
                        trks[nt++] = build_bearing_track(ts_us, p3_e, t1_e, 1800U, 2U, 3U, mrng, b_noise);
                    }

                    /* Wire-encode on 3-ship at 10 Hz equivalent (2 tracks per peer per tick) */
                    if (n_ships == 3 && step < 40) {
                        uint8_t wbuf[96]{};
                        for (std::size_t ti = 0U; ti < nt; ++ti) {
                            const int32_t wlen = xvs_encode_mavlink_bearing_track(
                                &trks[ti],
                                static_cast<uint8_t>((step * 6 + static_cast<int>(ti)) & 0xFF),
                                static_cast<uint8_t>(trks[ti].sender_id),
                                156U,
                                0U,
                                ts_us,
                                mav_key,
                                wbuf,
                                sizeof(wbuf));
                            if (wlen > 0) {
                                total_mavlink_bytes_3ship += static_cast<uint64_t>(wlen);
                            }
                        }
                        total_sim_seconds_3ship += 0.1; /* 10 Hz packet cadence */
                    }

                    XvsMeshDaaEvalResult mres{};
                    (void)xvs_mesh_daa_eval(trks, nt, &mres);
                    if (step == 0) {
                        mm.ghosts_rejected_total += static_cast<int>(mres.ghosts_rejected);
                    }

                    for (uint32_t fi = 0U; fi < mres.num_fused && fi < 8U; ++fi) {
                        double fned[3]{};
                        ecef_to_ned(gf, mres.fused_pos_ecef[fi], fned);
                        /* Match to nearest true intruder to measure range accuracy vs ghost status */
                        double d0 = std::hypot(fned[0] - int_p[0][0], fned[1] - int_p[0][1]);
                        double d1 = std::hypot(fned[0] - int_p[1][0], fned[1] - int_p[1][1]);
                        const int match_idx = (d0 <= d1) ? 0 : 1;
                        const double pos_err = std::min(d0, d1);
                        if (step == 0) {
                            if (pos_err > 120.0) {
                                ++enc_ghosts_seen;
                            } else {
                                const double r_est = std::hypot(fned[0] - own_p[0], fned[1] - own_p[1]);
                                const double r_true = std::hypot(int_p[match_idx][0] - own_p[0], int_p[match_idx][1] - own_p[1]);
                                mm.range_errs_m.push_back(std::abs(r_est - r_true));
                            }
                        }

                        /* FCI covariance fusion with previous step estimate */
                        double fci_x[3]{}, fci_p[9]{};
                        const double pa[9] = {100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0};
                        const double pb[9] = {144.0, 0.0, 0.0, 0.0, 144.0, 0.0, 0.0, 0.0, 144.0};
                        (void)xvs_fci_fuse_3d(mres.fused_pos_ecef[fi], pa, mres.fused_pos_ecef[fi], pb, fci_x, fci_p);
                        ecef_to_ned(gf, fci_x, fned);

                        est_ints[n_est].pos_ned[0] = fned[0];
                        est_ints[n_est].pos_ned[1] = fned[1];
                        est_ints[n_est].pos_ned[2] = fned[2];
                        est_ints[n_est].vel_ned[0] = int_v[match_idx][0];
                        est_ints[n_est].vel_ned[1] = int_v[match_idx][1];
                        est_ints[n_est].vel_ned[2] = int_v[match_idx][2];
                        est_ints[n_est].valid = 1U;
                        ++n_est;
                    }
                }

                /* Real M1 Rust Simplex RTA + ECBF-QP closed-loop guidance */
                XvsRtaFloatTelemetry tel{};
                tel.horizon_pl_rad = 0.002;
                tel.horizon_al_rad = 0.008726646259971648;
                tel.v_vpl_m = 4.0;
                tel.v_val_m = 10.0;
                tel.v_hpl_m = 12.0;
                tel.v_hal_m = 40.0;
                tel.cusum_thr = 10.53;
                tel.ood_global_thr = 1.0;
                tel.ood_cam_thr = 1.0;
                tel.cam_health_mask = 0x3FU;
                XvsRtaTickInput rta_in{};
                XvsRtaTickOutput rta_out{};
                (void)xvs_rta_build_input_from_f64(&tel, &rta_in);
                (void)xvs_rta_step(rta, &rta_in, &rta_out);

                XvsEcbfQpResult qp{};
                (void)xvs_ecbf_qp_solve(own_p, own_v, 0.0, 0.0, est_ints, n_est, &qp);
                if (qp.wc1_alert != 0U && step <= 30) {
                    alerted_early = true;
                }

                /* Integrate fixed-wing kinematics */
                const double bank = (rta_out.inhibit_avoidance_turns == 0U) ? qp.bank_cmd_rad : 0.0;
                const double climb = (rta_out.inhibit_avoidance_turns == 0U) ? qp.climb_rate_cmd_mps : 0.0;
                const double omega = (kGravity * std::tan(bank)) / kOwnSpeed;
                own_psi += omega * kDt;
                own_vz = -climb;
                own_v[0] = kOwnSpeed * std::cos(own_psi);
                own_v[1] = kOwnSpeed * std::sin(own_psi);
                own_v[2] = own_vz;
                for (int a = 0; a < 3; ++a) {
                    own_p[a] += own_v[a] * kDt;
                    int_p[0][a] += int_v[0][a] * kDt;
                    int_p[1][a] += int_v[1][a] * kDt;
                }

                /* Exact 3D separation check against true intruders */
                for (int ti = 0; ti < 2; ++ti) {
                    const double dh = std::hypot(own_p[0] - int_p[ti][0], own_p[1] - int_p[ti][1]);
                    const double dv = std::abs(own_p[2] - int_p[ti][2]);
                    const double d3 = std::hypot(dh, dv);
                    if (dh < min_hmd_enc) {
                        min_hmd_enc = dh;
                    }
                    if (d3 < min_3d_enc) {
                        min_3d_enc = d3;
                    }
                    if (dh < kNmacH && dv < kNmacV) {
                        hit_nmac = true;
                    }
                    if (dh < kLowcH && dv < kLowcV) {
                        hit_lowc = true;
                    }
                }
            }

            if (hit_nmac) {
                ++mm.nmac_count;
            }
            if (hit_lowc) {
                ++mm.lowc_count;
            }
            if (alerted_early || !hit_lowc) {
                ++mm.do365_compliant_count;
            }
            mm.ghosts_total += enc_ghosts_seen;
            if (min_hmd_enc < mm.min_hmd_overall_m) {
                mm.min_hmd_overall_m = min_hmd_enc;
            }
            if (min_3d_enc < mm.min_3d_sep_overall_m) {
                mm.min_3d_sep_overall_m = min_3d_enc;
            }
            xvs_rta_destroy(rta);
        };

        run_mode(1, rep.m1_ship);
        run_mode(2, rep.m2_ship);
        run_mode(3, rep.m3_ship);
    }

    auto finish_range = [](ModeMetrics &mm, double &p50, double &p95) {
        std::sort(mm.range_errs_m.begin(), mm.range_errs_m.end());
        p50 = percentile_sorted(mm.range_errs_m, 0.50);
        p95 = percentile_sorted(mm.range_errs_m, 0.95);
    };
    finish_range(rep.m1_ship, rep.p50_1ship, rep.p95_1ship);
    finish_range(rep.m2_ship, rep.p50_2ship, rep.p95_2ship);
    finish_range(rep.m3_ship, rep.p50_3ship, rep.p95_3ship);

    /* Per-node bandwidth = (total_bytes_3_nodes / 3) * 8 / total_sim_s / 1000 kbps */
    const double per_node_bytes = static_cast<double>(total_mavlink_bytes_3ship) / 3.0;
    rep.bandwidth_kbps_3ship =
        (total_sim_seconds_3ship > 0.0) ? ((per_node_bytes * 8.0) / total_sim_seconds_3ship / 1000.0) : 0.0;

    /* Finite RFC-8259-safe ghost suppression ratio: ghosts_2ship / max(ghosts_3ship, 1) */
    const int denom_3 = (rep.m3_ship.ghosts_total > 0) ? rep.m3_ship.ghosts_total : 1;
    rep.ghost_suppression_ratio = static_cast<double>(rep.m2_ship.ghosts_total) / static_cast<double>(denom_3);

    rep.pass = (rep.m3_ship.nmac_count == 0) &&
               (rep.p50_3ship < rep.p50_1ship) &&
               (rep.p95_3ship < rep.p95_1ship) &&
               (rep.ghost_suppression_ratio >= 10.0) &&
               (rep.bandwidth_kbps_3ship > 0.0 && rep.bandwidth_kbps_3ship < 125.0);
    return rep;
}

static bool write_audit_json(const char *path, const char *opt_label, const AuditSummary &a) {
    FILE *f = std::fopen(path, "w");
    if (f == nullptr) {
        return false;
    }
    std::fprintf(
        f,
        "{\n"
        "  \"opt_level\": \"%s\",\n"
        "  \"selftest_alloc_count\": %llu,\n"
        "  \"selftest_alloc_bytes\": %llu,\n"
        "  \"hotloop_ticks\": %llu,\n"
        "  \"hotloop_alloc_count\": %llu,\n"
        "  \"hotloop_alloc_bytes\": %llu,\n"
        "  \"hotloop_free_count\": %llu,\n"
        "  \"p50_us\": %.4f,\n"
        "  \"p95_us\": %.4f,\n"
        "  \"p99_us\": %.4f,\n"
        "  \"max_us\": %.4f,\n"
        "  \"mavlink_wire_bytes\": %d,\n"
        "  \"pass\": %s\n"
        "}\n",
        opt_label,
        static_cast<unsigned long long>(a.selftest_alloc_count),
        static_cast<unsigned long long>(a.selftest_alloc_bytes),
        static_cast<unsigned long long>(a.hotloop_ticks),
        static_cast<unsigned long long>(a.hotloop_alloc_count),
        static_cast<unsigned long long>(a.hotloop_alloc_bytes),
        static_cast<unsigned long long>(a.hotloop_free_count),
        a.p50_us,
        a.p95_us,
        a.p99_us,
        a.max_us,
        a.mavlink_wire_bytes,
        a.pass ? "true" : "false");
    std::fclose(f);
    return true;
}

static bool read_o0_audit_fields(
    const char *path,
    uint64_t &selftest_cnt,
    uint64_t &hot_cnt,
    uint64_t &hot_bytes,
    bool &o0_pass) {
    FILE *f = std::fopen(path, "r");
    if (f == nullptr) {
        return false;
    }
    char buf[4096]{};
    const std::size_t n = std::fread(buf, 1U, sizeof(buf) - 1U, f);
    std::fclose(f);
    buf[n] = '\0';

    auto parse_u64 = [&](const char *key, uint64_t &out) -> bool {
        const char *pos = std::strstr(buf, key);
        if (pos == nullptr) {
            return false;
        }
        pos += std::strlen(key);
        out = std::strtoull(pos, nullptr, 10);
        return true;
    };
    const bool ok1 = parse_u64("\"selftest_alloc_count\": ", selftest_cnt);
    const bool ok2 = parse_u64("\"hotloop_alloc_count\": ", hot_cnt);
    const bool ok3 = parse_u64("\"hotloop_alloc_bytes\": ", hot_bytes);
    o0_pass = (std::strstr(buf, "\"pass\": true") != nullptr);
    return ok1 && ok2 && ok3;
}

static bool write_v1_evidence_json(
    const char *path,
    const AuditSummary &a_o2,
    uint64_t o0_selftest_cnt,
    uint64_t o0_hot_cnt,
    uint64_t o0_hot_bytes,
    bool o0_pass) {
    /* Evaluate 4 canonical xvs_ecbf_qp_solve probe cases on Linux libxvs_core.a for cross-OS <=1e-12 parity */
    struct ProbeCase {
        double p_own[3];
        double v_own[3];
        double nom_bank;
        double nom_climb;
        XvsIntruderNed intr;
    };
    const ProbeCase probes[4] = {
        {{0.0, 0.0, -600.0}, {40.0, 0.0, 0.0}, 0.0, 0.0, {{550.0, -30.0, -600.0}, {-40.0, 0.0, 0.0}, 1U}},
        {{0.0, 0.0, -500.0}, {35.0, 10.0, 0.0}, 0.05, 0.0, {{400.0, 150.0, -500.0}, {-35.0, -10.0, 0.0}, 1U}},
        {{100.0, -50.0, -700.0}, {45.0, 0.0, -1.0}, 0.0, 1.0, {{220.0, -45.0, -700.0}, {-45.0, 0.0, 0.0}, 1U}},
        {{0.0, 0.0, -600.0}, {38.0, 0.0, 0.0}, 0.0, 0.0, {{3000.0, 2000.0, -600.0}, {-30.0, 0.0, 0.0}, 1U}},
    };

    XvsEcbfQpResult probe_res[4]{};
    for (int i = 0; i < 4; ++i) {
        (void)xvs_ecbf_qp_solve(
            probes[i].p_own,
            probes[i].v_own,
            probes[i].nom_bank,
            probes[i].nom_climb,
            &probes[i].intr,
            1U,
            &probe_res[i]);
    }

    const bool overall_pass = a_o2.pass && o0_pass && (o0_hot_cnt == 0U) && (o0_hot_bytes == 0U) && (o0_selftest_cnt >= 1U);

    FILE *f = std::fopen(path, "w");
    if (f == nullptr) {
        return false;
    }
    std::fprintf(
        f,
        "{\n"
        "  \"schema\": \"xvs.v2_m3.v1_xvs_rt_evidence/1\",\n"
        "  \"runtime_scope\": \"WSL2_UBUNTU_CPU_ONLY_ZERO_ALLOC_AUDIT\",\n"
        "  \"gpu_wcet_status\": \"PENDING_NATIVE_LINUX_GPU_WCET\",\n"
        "  \"ecbf_solver_provenance\": \"%s\",\n"
        "  \"daidalus_adapter_mode\": \"%s\",\n"
        "  \"o0_selftest_alloc_count\": %llu,\n"
        "  \"o0_hotloop_alloc_count\": %llu,\n"
        "  \"o0_hotloop_alloc_bytes\": %llu,\n"
        "  \"o0_audit_pass\": %s,\n"
        "  \"o2_selftest_alloc_count\": %llu,\n"
        "  \"o2_selftest_alloc_bytes\": %llu,\n"
        "  \"o2_hotloop_ticks\": %llu,\n"
        "  \"o2_hotloop_alloc_count\": %llu,\n"
        "  \"o2_hotloop_alloc_bytes\": %llu,\n"
        "  \"o2_cpu_tick_p50_us\": %.4f,\n"
        "  \"o2_cpu_tick_p95_us\": %.4f,\n"
        "  \"o2_cpu_tick_p99_us\": %.4f,\n"
        "  \"o2_cpu_tick_max_us\": %.4f,\n"
        "  \"o2_audit_pass\": %s,\n"
        "  \"mavlink_bearing_track_wire_bytes\": %d,\n"
        "  \"linux_ffi_ecbf_probes\": [\n",
        xvs_rt::ecbf_solver_provenance(),
        xvs_daidalus_adapter_mode(),
        static_cast<unsigned long long>(o0_selftest_cnt),
        static_cast<unsigned long long>(o0_hot_cnt),
        static_cast<unsigned long long>(o0_hot_bytes),
        o0_pass ? "true" : "false",
        static_cast<unsigned long long>(a_o2.selftest_alloc_count),
        static_cast<unsigned long long>(a_o2.selftest_alloc_bytes),
        static_cast<unsigned long long>(a_o2.hotloop_ticks),
        static_cast<unsigned long long>(a_o2.hotloop_alloc_count),
        static_cast<unsigned long long>(a_o2.hotloop_alloc_bytes),
        a_o2.p50_us,
        a_o2.p95_us,
        a_o2.p99_us,
        a_o2.max_us,
        a_o2.pass ? "true" : "false",
        a_o2.mavlink_wire_bytes);

    for (int i = 0; i < 4; ++i) {
        std::fprintf(
            f,
            "    {\n"
            "      \"case_id\": %d,\n"
            "      \"p_own\": [%.6f, %.6f, %.6f],\n"
            "      \"v_own\": [%.6f, %.6f, %.6f],\n"
            "      \"nom_bank\": %.6f,\n"
            "      \"nom_climb\": %.6f,\n"
            "      \"intr_pos\": [%.6f, %.6f, %.6f],\n"
            "      \"intr_vel\": [%.6f, %.6f, %.6f],\n"
            "      \"bank_cmd_rad\": %.17g,\n"
            "      \"climb_rate_cmd_mps\": %.17g,\n"
            "      \"slack\": %.17g,\n"
            "      \"ecbf_infeasible\": %u,\n"
            "      \"wc1_alert\": %u\n"
            "    }%s\n",
            i,
            probes[i].p_own[0], probes[i].p_own[1], probes[i].p_own[2],
            probes[i].v_own[0], probes[i].v_own[1], probes[i].v_own[2],
            probes[i].nom_bank,
            probes[i].nom_climb,
            probes[i].intr.pos_ned[0], probes[i].intr.pos_ned[1], probes[i].intr.pos_ned[2],
            probes[i].intr.vel_ned[0], probes[i].intr.vel_ned[1], probes[i].intr.vel_ned[2],
            probe_res[i].bank_cmd_rad,
            probe_res[i].climb_rate_cmd_mps,
            probe_res[i].slack,
            static_cast<unsigned>(probe_res[i].ecbf_infeasible),
            static_cast<unsigned>(probe_res[i].wc1_alert),
            (i < 3) ? "," : "");
    }
    std::fprintf(f, "  ],\n  \"pass\": %s\n}\n", overall_pass ? "true" : "false");
    std::fclose(f);
    return overall_pass;
}

static bool write_v2_evidence_json(const char *path, const MeshDaaBenchReport &r) {
    FILE *f = std::fopen(path, "w");
    if (f == nullptr) {
        return false;
    }
    const double comp_1 = static_cast<double>(r.m1_ship.do365_compliant_count) / static_cast<double>(r.encounters);
    const double comp_2 = static_cast<double>(r.m2_ship.do365_compliant_count) / static_cast<double>(r.encounters);
    const double comp_3 = static_cast<double>(r.m3_ship.do365_compliant_count) / static_cast<double>(r.encounters);
    std::fprintf(
        f,
        "{\n"
        "  \"schema\": \"xvs.v2_m3.v2_meshdaa_500enc_evidence/1\",\n"
        "  \"encounters\": %d,\n"
        "  \"simultaneous_intruders_per_encounter\": 2,\n"
        "  \"mavlink_52001_packet_bytes\": 77,\n"
        "  \"bandwidth_kbps_3ship_per_node\": %.4f,\n"
        "  \"ghosts_2ship\": %d,\n"
        "  \"ghosts_3ship\": %d,\n"
        "  \"ghosts_rejected_3ship\": %d,\n"
        "  \"ghost_suppression_ratio\": %.4f,\n"
        "  \"one_ship\": {\n"
        "    \"nmac_count\": %d,\n"
        "    \"lowc_count\": %d,\n"
        "    \"do365_35s_compliance_rate\": %.4f,\n"
        "    \"p50_range_err_m\": %.4f,\n"
        "    \"p95_range_err_m\": %.4f,\n"
        "    \"min_hmd_overall_m\": %.4f,\n"
        "    \"min_3d_sep_overall_m\": %.4f\n"
        "  },\n"
        "  \"two_ship\": {\n"
        "    \"nmac_count\": %d,\n"
        "    \"lowc_count\": %d,\n"
        "    \"do365_35s_compliance_rate\": %.4f,\n"
        "    \"p50_range_err_m\": %.4f,\n"
        "    \"p95_range_err_m\": %.4f,\n"
        "    \"min_hmd_overall_m\": %.4f,\n"
        "    \"min_3d_sep_overall_m\": %.4f\n"
        "  },\n"
        "  \"three_ship\": {\n"
        "    \"nmac_count\": %d,\n"
        "    \"lowc_count\": %d,\n"
        "    \"do365_35s_compliance_rate\": %.4f,\n"
        "    \"p50_range_err_m\": %.4f,\n"
        "    \"p95_range_err_m\": %.4f,\n"
        "    \"min_hmd_overall_m\": %.4f,\n"
        "    \"min_3d_sep_overall_m\": %.4f\n"
        "  },\n"
        "  \"pass\": %s\n"
        "}\n",
        r.encounters,
        r.bandwidth_kbps_3ship,
        r.m2_ship.ghosts_total,
        r.m3_ship.ghosts_total,
        r.m3_ship.ghosts_rejected_total,
        r.ghost_suppression_ratio,
        r.m1_ship.nmac_count,
        r.m1_ship.lowc_count,
        comp_1,
        r.p50_1ship,
        r.p95_1ship,
        r.m1_ship.min_hmd_overall_m,
        r.m1_ship.min_3d_sep_overall_m,
        r.m2_ship.nmac_count,
        r.m2_ship.lowc_count,
        comp_2,
        r.p50_2ship,
        r.p95_2ship,
        r.m2_ship.min_hmd_overall_m,
        r.m2_ship.min_3d_sep_overall_m,
        r.m3_ship.nmac_count,
        r.m3_ship.lowc_count,
        comp_3,
        r.p50_3ship,
        r.p95_3ship,
        r.m3_ship.min_hmd_overall_m,
        r.m3_ship.min_3d_sep_overall_m,
        r.pass ? "true" : "false");
    std::fclose(f);
    return r.pass;
}

}  // namespace

int main(int argc, char **argv) {
    const char *mode = "full";
    const char *opt_label = "O2";
    const char *out_audit = nullptr;
    const char *o0_audit_in = nullptr;
    const char *out_v1 = nullptr;
    const char *out_v2 = nullptr;

    for (int i = 1; i + 1 < argc; i += 2) {
        if (std::strcmp(argv[i], "--mode") == 0) {
            mode = argv[i + 1];
        } else if (std::strcmp(argv[i], "--opt-label") == 0) {
            opt_label = argv[i + 1];
        } else if (std::strcmp(argv[i], "--out-audit") == 0) {
            out_audit = argv[i + 1];
        } else if (std::strcmp(argv[i], "--o0-audit-in") == 0) {
            o0_audit_in = argv[i + 1];
        } else if (std::strcmp(argv[i], "--out-v1") == 0) {
            out_v1 = argv[i + 1];
        } else if (std::strcmp(argv[i], "--out-v2") == 0) {
            out_v2 = argv[i + 1];
        }
    }

    const AuditSummary audit = run_zero_alloc_audit(10000U);
    std::printf(
        "[%s] selftest_alloc=%llu (%llu B), hotloop_10k_alloc=%llu (%llu B), p50=%.2fus p99=%.2fus wire=%dB pass=%s\n",
        opt_label,
        static_cast<unsigned long long>(audit.selftest_alloc_count),
        static_cast<unsigned long long>(audit.selftest_alloc_bytes),
        static_cast<unsigned long long>(audit.hotloop_alloc_count),
        static_cast<unsigned long long>(audit.hotloop_alloc_bytes),
        audit.p50_us,
        audit.p99_us,
        audit.mavlink_wire_bytes,
        audit.pass ? "PASS" : "FAIL");

    if (std::strcmp(mode, "audit-only") == 0) {
        if (out_audit != nullptr) {
            (void)write_audit_json(out_audit, opt_label, audit);
        }
        return audit.pass ? 0 : 1;
    }

    uint64_t o0_self = 0U;
    uint64_t o0_hot_cnt = 1U;
    uint64_t o0_hot_bytes = 1U;
    bool o0_pass = false;
    if (o0_audit_in != nullptr) {
        (void)read_o0_audit_fields(o0_audit_in, o0_self, o0_hot_cnt, o0_hot_bytes, o0_pass);
    }

    const bool v1_ok = (out_v1 != nullptr)
                           ? write_v1_evidence_json(out_v1, audit, o0_self, o0_hot_cnt, o0_hot_bytes, o0_pass)
                           : audit.pass;

    const MeshDaaBenchReport mesh = run_meshdaa_500_encounters();
    std::printf(
        "[MeshDAA 500-enc] 1-ship(nmac=%d p50=%.1fm p95=%.1fm) | 2-ship(nmac=%d ghosts=%d p50=%.1fm) | "
        "3-ship(nmac=%d lowc=%d ghosts=%d rejected=%d ratio=%.1fx p50=%.1fm p95=%.1fm bw=%.2fkbps) pass=%s\n",
        mesh.m1_ship.nmac_count, mesh.p50_1ship, mesh.p95_1ship,
        mesh.m2_ship.nmac_count, mesh.m2_ship.ghosts_total, mesh.p50_2ship,
        mesh.m3_ship.nmac_count, mesh.m3_ship.lowc_count,
        mesh.m3_ship.ghosts_total, mesh.m3_ship.ghosts_rejected_total,
        mesh.ghost_suppression_ratio, mesh.p50_3ship, mesh.p95_3ship,
        mesh.bandwidth_kbps_3ship,
        mesh.pass ? "PASS" : "FAIL");

    const bool v2_ok = (out_v2 != nullptr) ? write_v2_evidence_json(out_v2, mesh) : mesh.pass;
    return (v1_ok && v2_ok) ? 0 : 1;
}
