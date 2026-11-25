/* OpenXVS v2.0 — Native C++20 Unit & Fail-Closed Test Suite (`xvs-rt`, V2-M3L-1).
 *
 * Compiled with -std=c++20 -fno-exceptions -fno-rtti -Wall -Wextra -Werror -Wconversion
 * -Wsign-conversion -Wshadow -Wdouble-promotion and linked against real libxvs_core.a.
 */

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <thread>

#include "xvs_core.h"
#include "xvs_daidalus_bridge.h"
#include "xvs_rt/ecbf_qp.hpp"
#include "xvs_rt/mh_vit.hpp"
#include "xvs_rt/rt_pipeline.hpp"
#include "xvs_rt/spsc_ring.hpp"

namespace {

static int g_failures = 0;

static void check_cond(bool cond, const char *msg) {
    if (!cond) {
        std::fprintf(stderr, "[FAIL] %s\n", msg);
        ++g_failures;
    }
}

static void test_spsc_ring_single_and_multithread() {
    xvs_rt::SpscRing<uint64_t, 64U> ring;
    check_cond(ring.empty(), "SPSC starts empty");

    /* Fill to usable capacity (63), verify push fails on 64th, then overwrite works */
    for (uint64_t i = 0U; i < 63U; ++i) {
        check_cond(ring.try_push(i + 100U), "SPSC try_push within capacity");
    }
    check_cond(!ring.try_push(999U), "SPSC try_push rejects when full");
    const bool dropped = ring.push_overwrite_latest(999U);
    check_cond(dropped, "SPSC push_overwrite_latest signals drop when full");

    uint64_t first = 0U;
    check_cond(ring.try_pop(first) && first == 101U, "SPSC overwrite dropped oldest item (100)");

    /* Concurrent producer / consumer stress across 20,000 items */
    ring.reset();
    constexpr uint64_t kCount = 20000U;
    std::atomic<bool> done{false};
    std::thread producer([&]() {
        for (uint64_t v = 1U; v <= kCount; ++v) {
            while (!ring.try_push(v)) {
                std::this_thread::yield();
            }
        }
        done.store(true, std::memory_order_release);
    });

    uint64_t expected = 1U;
    uint64_t val = 0U;
    while (expected <= kCount) {
        if (ring.try_pop(val)) {
            if (val != expected) {
                check_cond(false, "SPSC FIFO ordering mismatch under concurrency");
                break;
            }
            ++expected;
        } else if (!done.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }
    producer.join();
    check_cond(expected == kCount + 1U && ring.empty(), "SPSC 20k concurrent FIFO exact");
}

static void test_mh_vit_ned_and_fail_closed() {
    xvs_rt::MhVitObservationNed obs{};
    obs.own_pos_ned_m[0] = 0.0;
    obs.own_pos_ned_m[1] = 0.0;
    obs.own_pos_ned_m[2] = -500.0;
    obs.own_vel_ned_mps[0] = 35.0;
    obs.own_vel_ned_mps[1] = 0.0;
    obs.own_vel_ned_mps[2] = 0.0;
    obs.azimuth_rad = 0.0;           /* Due North */
    obs.elevation_rad = 0.25;        /* Positive above horizon -> NED z < own_z */
    obs.azimuth_rate_rad_s = 0.01;
    obs.elevation_rate_rad_s = 0.0;
    obs.kappa_s_inv = 0.05;          /* Closing */
    /* +/-40% around 2000 m -> [1200, 2800] */
    obs.r_min_m = 1200.0;
    obs.r_max_m = 2800.0;
    obs.sigma_ang_rad = 0.002;
    obs.sigma_rate_rad_s = 0.001;
    obs.sigma_kappa_s_inv = 0.005;

    const xvs_rt::MhVitResult res = xvs_rt::generate_mh_vit_ned(obs);
    check_cond(res.valid, "MH-VIT nominal valid");
    check_cond(res.max_rel_half_width <= 0.085, "MH-VIT dr_k/r_k <= 0.085 at +/-40%");
    /* 0% inward overshoot: a_0 == r_min, b_4 == r_max, strictly interior midpoints */
    check_cond(std::abs(res.hypotheses[0].a_k_m - 1200.0) < 1e-9, "MH-VIT a_0 == r_min");
    check_cond(std::abs(res.hypotheses[4].b_k_m - 2800.0) < 1e-9, "MH-VIT b_4 == r_max");
    /* NED elevation sign: positive elevation -> z_ned strictly LESS than own_z */
    check_cond(res.hypotheses[2].pos_ned_m[2] < obs.own_pos_ned_m[2], "MH-VIT NED +el -> -z_D");

    /* Fail-closed checks: NaN, +Inf, pole |el| -> pi/2, inverted range bounds */
    xvs_rt::MhVitObservationNed bad = obs;
    bad.azimuth_rad = std::numeric_limits<double>::quiet_NaN();
    check_cond(!xvs_rt::generate_mh_vit_ned(bad).valid, "MH-VIT rejects NaN azimuth");

    bad = obs;
    bad.r_max_m = std::numeric_limits<double>::infinity();
    check_cond(!xvs_rt::generate_mh_vit_ned(bad).valid, "MH-VIT rejects +Inf r_max");

    bad = obs;
    bad.elevation_rad = 1.5707963267948966;
    check_cond(!xvs_rt::generate_mh_vit_ned(bad).valid, "MH-VIT rejects pole elevation");

    bad = obs;
    bad.r_min_m = 3000.0;
    check_cond(!xvs_rt::generate_mh_vit_ned(bad).valid, "MH-VIT rejects r_min >= r_max");
}

static void test_ecbf_qp_ned_safety_agreement_and_fail_closed() {
    const xvs_rt::EcbfConfig cfg = xvs_rt::default_ecbf_config();
    check_cond(
        std::strcmp(xvs_rt::ecbf_solver_provenance(), "STANDALONE_ACTIVE_SET_SOLVER_DAQP_ABI_READY") == 0,
        "ECBF-QP solver provenance tag exact");

    /* Head-on / slight-left conflict: ownship flying North at 40 m/s, intruder approaching from North-West */
    xvs_rt::EcbfOwnshipNed own{};
    own.pos_ned_m[0] = 0.0;
    own.pos_ned_m[1] = 0.0;
    own.pos_ned_m[2] = -600.0;
    own.vel_ned_mps[0] = 40.0;
    own.vel_ned_mps[1] = 0.0;
    own.vel_ned_mps[2] = 0.0;

    xvs_rt::EcbfIntruderNed intr{};
    intr.pos_ned_m[0] = 550.0;
    intr.pos_ned_m[1] = -30.0; /* slightly West -> right turn (+bank) opens separation */
    intr.pos_ned_m[2] = -600.0;
    intr.vel_ned_mps[0] = -40.0;
    intr.vel_ned_mps[1] = 0.0;
    intr.vel_ned_mps[2] = 0.0;
    intr.cooperative = false;
    intr.valid = true;

    const xvs_rt::EcbfSolutionNed cpp_sol = xvs_rt::solve_ecbf_qp_ned(own, &intr, 1U, cfg);
    check_cond(std::isfinite(cpp_sol.bank_cmd_rad) && std::abs(cpp_sol.bank_cmd_rad) <= cfg.phi_max_rad + 1e-9,
               "ECBF-QP bank within +/-30 deg");
    check_cond(cpp_sol.bank_cmd_rad > 0.02, "ECBF-QP turns right away from West-side threat");

    /* Cross-check directional safety agreement with M1 Rust xvs_ecbf_qp_solve */
    XvsIntruderNed rust_int{};
    rust_int.pos_ned[0] = intr.pos_ned_m[0];
    rust_int.pos_ned[1] = intr.pos_ned_m[1];
    rust_int.pos_ned[2] = intr.pos_ned_m[2];
    rust_int.vel_ned[0] = intr.vel_ned_mps[0];
    rust_int.vel_ned[1] = intr.vel_ned_mps[1];
    rust_int.vel_ned[2] = intr.vel_ned_mps[2];
    rust_int.valid = 1U;
    XvsEcbfQpResult rust_sol{};
    check_cond(xvs_ecbf_qp_solve(own.pos_ned_m, own.vel_ned_mps, 0.0, 0.0, &rust_int, 1U, &rust_sol) == 0,
               "Rust xvs_ecbf_qp_solve succeeded");
    check_cond(rust_sol.bank_cmd_rad > 0.0 && cpp_sol.bank_cmd_rad > 0.0,
               "C++20 & Rust ECBF-QP agree on positive right-turn avoidance direction");

    /* Sandwich infeasibility -> safe escape active */
    xvs_rt::EcbfIntruderNed sandwich[2]{};
    sandwich[0] = intr;
    sandwich[0].pos_ned_m[0] = 90.0;
    sandwich[0].pos_ned_m[1] = -80.0;
    sandwich[1] = intr;
    sandwich[1].pos_ned_m[0] = 90.0;
    sandwich[1].pos_ned_m[1] = 80.0;
    const xvs_rt::EcbfSolutionNed esc_sol = xvs_rt::solve_ecbf_qp_ned(own, sandwich, 2U, cfg);
    check_cond(esc_sol.safe_escape_active && !esc_sol.feasible,
               "ECBF-QP activates deterministic safe escape on infeasible NMAC sandwich");
    check_cond(std::abs(esc_sol.bank_cmd_rad) <= cfg.phi_max_rad + 1e-9, "Safe escape bank clamped <= 30 deg");

    /* Fail-closed on NaN / +Inf */
    xvs_rt::EcbfOwnshipNed nan_own = own;
    nan_own.pos_ned_m[1] = std::numeric_limits<double>::quiet_NaN();
    const xvs_rt::EcbfSolutionNed nan_sol = xvs_rt::solve_ecbf_qp_ned(nan_own, &intr, 1U, cfg);
    check_cond(nan_sol.fail_closed && nan_sol.safe_escape_active && !nan_sol.feasible,
               "ECBF-QP fails closed on NaN ownship");

    xvs_rt::EcbfIntruderNed inf_intr = intr;
    inf_intr.vel_ned_mps[0] = std::numeric_limits<double>::infinity();
    const xvs_rt::EcbfSolutionNed inf_sol = xvs_rt::solve_ecbf_qp_ned(own, &inf_intr, 1U, cfg);
    check_cond(inf_sol.fail_closed && inf_sol.safe_escape_active && !inf_sol.feasible,
               "ECBF-QP fails closed on +Inf intruder");

    /* D16-M3-M2 adversarial probes: corrupted EcbfConfig fields + exact dead-ahead head-on */
    auto check_finite_fail_closed = [](const xvs_rt::EcbfSolutionNed &s, const char *label) {
        check_cond(s.fail_closed && s.safe_escape_active && !s.feasible, label);
        check_cond(std::isfinite(s.omega_cmd_rad_s) && std::isfinite(s.a_z_down_cmd_mps2) &&
                       std::isfinite(s.a_t_cmd_mps2) && std::isfinite(s.delta_wc) &&
                       std::isfinite(s.bank_cmd_rad) && std::isfinite(s.climb_rate_cmd_mps),
                   label);
        check_cond(std::abs(s.bank_cmd_rad) <= 0.5235987755982988 + 1e-6, label);
    };

    /* PROBE_PHI_NAN */
    xvs_rt::EcbfConfig cfg_phi_nan = cfg;
    cfg_phi_nan.phi_max_rad = std::numeric_limits<double>::quiet_NaN();
    check_finite_fail_closed(
        xvs_rt::solve_ecbf_qp_ned(own, &intr, 1U, cfg_phi_nan),
        "PROBE_PHI_NAN: fail-closed with strictly finite escape commands");

    /* PROBE_RNMAC_NAN */
    xvs_rt::EcbfConfig cfg_rnmac_nan = cfg;
    cfg_rnmac_nan.r_nmac_eff_m = std::numeric_limits<double>::quiet_NaN();
    check_finite_fail_closed(
        xvs_rt::solve_ecbf_qp_ned(own, &intr, 1U, cfg_rnmac_nan),
        "PROBE_RNMAC_NAN: fail-closed with strictly finite escape commands");

    /* PROBE_AZMAX_NAN */
    xvs_rt::EcbfConfig cfg_azmax_nan = cfg;
    cfg_azmax_nan.a_z_max_mps2 = std::numeric_limits<double>::quiet_NaN();
    check_finite_fail_closed(
        xvs_rt::solve_ecbf_qp_ned(own, &intr, 1U, cfg_azmax_nan),
        "PROBE_AZMAX_NAN: fail-closed with strictly finite escape commands");

    /* PROBE_HEADON_ZERO_Y: exact dead-ahead (cross_z == 0.0) -> 14 CFR 91.113 RIGHT (+30 deg) */
    xvs_rt::EcbfIntruderNed headon = intr;
    headon.pos_ned_m[0] = 50.0;
    headon.pos_ned_m[1] = 0.0;
    headon.pos_ned_m[2] = -600.0;
    headon.vel_ned_mps[0] = -40.0;
    headon.vel_ned_mps[1] = 0.0;
    headon.vel_ned_mps[2] = 0.0;
    const xvs_rt::EcbfSolutionNed headon_sol = xvs_rt::solve_ecbf_qp_ned(own, &headon, 1U, cfg);
    check_cond(headon_sol.safe_escape_active && !headon_sol.fail_closed && !headon_sol.feasible,
               "PROBE_HEADON_ZERO_Y: activates deterministic safe escape");
    check_cond(std::isfinite(headon_sol.bank_cmd_rad) &&
                   std::abs(headon_sol.bank_cmd_rad - cfg.phi_max_rad) < 1e-6,
               "PROBE_HEADON_ZERO_Y: dead-ahead escapes RIGHT (+30 deg, 14 CFR 91.113)");
}

static void test_daidalus_boundary_ned_enu_and_fail_closed() {
    check_cond(
        std::strcmp(xvs_daidalus_adapter_mode(), "STANDALONE_DO365_BOUNDARY_EMULATOR_READY_FOR_UPSTREAM_NOSA") == 0,
        "DAIDALUS adapter mode tag exact");

    const XvsDaidalusConfig cfg = xvs_daidalus_default_do365a_config();
    XvsDaidalusAircraftNed own{};
    own.pos_ned_m[0] = 0.0;
    own.pos_ned_m[1] = 0.0;
    own.pos_ned_m[2] = -600.0;
    own.vel_ned_mps[0] = 40.0;
    own.vel_ned_mps[1] = 0.0;
    own.vel_ned_mps[2] = 0.0;
    own.valid = 1U;

    /* Head-on threat 1600 m North closing at 80 m/s -> ~20 s to CPA -> WARNING (3) */
    XvsDaidalusAircraftNed intr{};
    intr.pos_ned_m[0] = 1600.0;
    intr.pos_ned_m[1] = 0.0;
    intr.pos_ned_m[2] = -600.0;
    intr.vel_ned_mps[0] = -40.0;
    intr.vel_ned_mps[1] = 0.0;
    intr.vel_ned_mps[2] = 0.0;
    intr.valid = 1U;

    XvsDaidalusBandResult res{};
    check_cond(xvs_daidalus_eval_ned(&own, &intr, 1U, &cfg, &res) == 0, "DAIDALUS eval nominal OK");
    check_cond(res.fail_closed == 0U && res.alert_level == 3, "DAIDALUS raises WARNING on 20s head-on");
    check_cond(res.min_hmd_m < 1.0 && res.min_vmd_m < 1.0, "DAIDALUS exact CPA HMD/VMD");

    /* Vertically separated by 500 m (> 137.16 m ZTHR) -> ALERT NONE (0) */
    intr.pos_ned_m[2] = -1100.0;
    check_cond(xvs_daidalus_eval_ned(&own, &intr, 1U, &cfg, &res) == 0, "DAIDALUS eval separated OK");
    check_cond(res.alert_level == 0, "DAIDALUS returns NONE when vertically separated by 500 m");

    /* Fail-closed on NaN / +/-Inf */
    intr.pos_ned_m[0] = std::numeric_limits<double>::quiet_NaN();
    check_cond(xvs_daidalus_eval_ned(&own, &intr, 1U, &cfg, &res) == -1, "DAIDALUS rejects NaN intruder");
    check_cond(res.fail_closed == 1U && res.alert_level == -1, "DAIDALUS fail_closed flag set on NaN");

    own.vel_ned_mps[2] = -std::numeric_limits<double>::infinity();
    intr.pos_ned_m[0] = 1600.0;
    check_cond(xvs_daidalus_eval_ned(&own, &intr, 1U, &cfg, &res) == -1, "DAIDALUS rejects -Inf ownship");
    check_cond(res.fail_closed == 1U && res.alert_level == -1, "DAIDALUS fail_closed flag set on -Inf");
}

}  // namespace

int main() {
    test_spsc_ring_single_and_multithread();
    test_mh_vit_ned_and_fail_closed();
    test_ecbf_qp_ned_safety_agreement_and_fail_closed();
    test_daidalus_boundary_ned_enu_and_fail_closed();

    if (g_failures == 0) {
        std::printf("[PASS] test_xvs_rt: all 4 native C++20 unit & fail-closed suites passed!\n");
        return 0;
    }
    std::fprintf(stderr, "[FAIL] test_xvs_rt: %d check(s) failed.\n", g_failures);
    return 1;
}
