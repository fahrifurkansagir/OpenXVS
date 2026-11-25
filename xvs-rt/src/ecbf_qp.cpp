/* OpenXVS v2.0 — 3D Fixed-Wing ECBF-QP in Strict NED (`DR-05:26-39`, `BC-M3-M3`).
 *
 * Solves:
 *   min_u  0.5 * (w_omega*(omega - nom_omega)^2 + w_az*(a_z - nom_az)^2 + w_at*(a_T - nom_aT)^2 + w_delta*delta_wc^2)
 *   s.t.   box envelope constraints on [omega, a_z_down, a_T, delta_wc >= 0]
 *          hard inflated NMAC barrier constraints (R_nm_eff = 220 m, Z_nm_eff = 50 m)
 *          soft Well-Clear barrier constraints relaxed by delta_wc >= 0
 * Strictly zero malloc, zero exceptions, fail-closed on NaN / +/-Inf.
 */

#include "xvs_rt/ecbf_qp.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace xvs_rt {

namespace {

static constexpr double kGravityMps2 = 9.80665;
static constexpr double kDefaultPhiMaxRad = 0.5235987755982988; /* 30 deg */
static constexpr double kMaxSafePhiRad = 1.5707963267948966 - 1e-3;
static constexpr double kDefaultVMinMps = 22.0;
static constexpr double kDefaultAzMaxMps2 = 3.5;
static constexpr double kDefaultVzMaxMps = 6.0;
static constexpr double kDefaultDtEffS = 0.5;
static constexpr int kMaxSolverIters = 15;
static constexpr std::size_t kMaxConstraints = kMaxEcbfIntruders * 2U;

struct LinearIneq {
    /* a[0]*omega + a[1]*a_z + a[2]*a_T + a[3]*delta_wc <= b */
    double a[4];
    double b;
    bool is_hard;
};

static bool is_finite3(const double v[3]) noexcept {
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

static bool is_pos_finite(double x) noexcept {
    return std::isfinite(x) && x > 0.0;
}

static bool is_valid_config(const EcbfConfig &cfg) noexcept {
    return std::isfinite(cfg.phi_max_rad) && cfg.phi_max_rad > 0.0 && cfg.phi_max_rad <= kMaxSafePhiRad &&
           is_pos_finite(cfg.v_min_mps) && is_pos_finite(cfg.v_max_mps) && cfg.v_max_mps > cfg.v_min_mps &&
           is_pos_finite(cfg.v_z_max_mps) &&
           is_pos_finite(cfg.a_z_max_mps2) &&
           is_pos_finite(cfg.a_t_max_mps2) &&
           is_pos_finite(cfg.dt_eff_s) &&
           is_pos_finite(cfg.r_wc_noncoop_m) &&
           is_pos_finite(cfg.r_wc_coop_m) &&
           is_pos_finite(cfg.z_wc_m) &&
           is_pos_finite(cfg.r_nmac_eff_m) &&
           is_pos_finite(cfg.z_nmac_eff_m) &&
           is_pos_finite(cfg.w_omega) &&
           is_pos_finite(cfg.w_az) &&
           is_pos_finite(cfg.w_at) &&
           is_pos_finite(cfg.w_delta);
}

static double clamp_val(double x, double lo, double hi) noexcept {
    if (x < lo) {
        return lo;
    }
    if (x > hi) {
        return hi;
    }
    return x;
}

static EcbfSolutionNed make_safe_escape(
    const EcbfOwnshipNed &own,
    const EcbfConfig &cfg,
    double turn_sign,
    double az_down_escape,
    double min_h_nmac,
    double min_h_wc,
    uint8_t iters,
    bool fail_closed) noexcept {
    EcbfSolutionNed sol{};
    const double safe_phi_max =
        (std::isfinite(cfg.phi_max_rad) && cfg.phi_max_rad > 0.0 && cfg.phi_max_rad <= kMaxSafePhiRad)
            ? cfg.phi_max_rad
            : kDefaultPhiMaxRad;
    const double safe_v_min = is_pos_finite(cfg.v_min_mps) ? cfg.v_min_mps : kDefaultVMinMps;
    const double safe_az_max = is_pos_finite(cfg.a_z_max_mps2) ? cfg.a_z_max_mps2 : kDefaultAzMaxMps2;
    const double safe_vz_max = is_pos_finite(cfg.v_z_max_mps) ? cfg.v_z_max_mps : kDefaultVzMaxMps;
    const double safe_dt_eff = is_pos_finite(cfg.dt_eff_s) ? cfg.dt_eff_s : kDefaultDtEffS;

    const double vx = std::isfinite(own.vel_ned_mps[0]) ? own.vel_ned_mps[0] : 30.0;
    const double vy = std::isfinite(own.vel_ned_mps[1]) ? own.vel_ned_mps[1] : 0.0;
    const double vz = std::isfinite(own.vel_ned_mps[2]) ? own.vel_ned_mps[2] : 0.0;
    const double v_h = std::max(std::hypot(vx, vy), safe_v_min);
    const double omega_max = (kGravityMps2 * std::tan(safe_phi_max)) / v_h;
    const double safe_turn_sign = (std::isfinite(turn_sign) && turn_sign < 0.0) ? -1.0 : 1.0;
    const double safe_az_esc = std::isfinite(az_down_escape) ? az_down_escape : -safe_az_max;

    sol.omega_cmd_rad_s = clamp_val(safe_turn_sign * omega_max, -omega_max, omega_max);
    sol.a_z_down_cmd_mps2 = clamp_val(safe_az_esc, -safe_az_max, safe_az_max);
    sol.a_t_cmd_mps2 = 0.0;
    sol.delta_wc = 1.0;
    sol.bank_cmd_rad = clamp_val(std::atan((sol.omega_cmd_rad_s * v_h) / kGravityMps2), -safe_phi_max, safe_phi_max);
    const double vz_next = clamp_val(vz + sol.a_z_down_cmd_mps2 * safe_dt_eff, -safe_vz_max, safe_vz_max);
    sol.climb_rate_cmd_mps = -vz_next; /* NED Down -> +up climb rate */
    sol.min_h_nmac = std::isfinite(min_h_nmac) ? min_h_nmac : -1.0;
    sol.min_h_wc = std::isfinite(min_h_wc) ? min_h_wc : -1.0;
    sol.iterations = iters;
    sol.feasible = false;
    sol.safe_escape_active = true;
    sol.fail_closed = fail_closed;
    sol.solver_provenance = ecbf_solver_provenance();
    return sol;
}

}  // namespace

const char *ecbf_solver_provenance() noexcept {
    return "STANDALONE_ACTIVE_SET_SOLVER_DAQP_ABI_READY";
}

EcbfConfig default_ecbf_config() noexcept {
    EcbfConfig cfg{};
    cfg.phi_max_rad = 0.5235987755982988;  /* 30 deg */
    cfg.v_min_mps = 22.0;
    cfg.v_max_mps = 65.0;
    cfg.v_z_max_mps = 6.0;
    cfg.a_z_max_mps2 = 3.5;
    cfg.a_t_max_mps2 = 2.5;
    cfg.dt_eff_s = 0.5;
    cfg.r_wc_noncoop_m = 670.56;
    cfg.r_wc_coop_m = 1219.2;
    cfg.z_wc_m = 137.16;
    cfg.r_nmac_eff_m = 220.0;
    cfg.z_nmac_eff_m = 50.0;
    cfg.w_omega = 10.0;
    cfg.w_az = 4.0;
    cfg.w_at = 1.0;
    cfg.w_delta = 20000.0;
    return cfg;
}

EcbfSolutionNed solve_ecbf_qp_ned(
    const EcbfOwnshipNed &own,
    const EcbfIntruderNed *intruders,
    std::size_t n_intruders,
    const EcbfConfig &cfg) noexcept {
    /* Strict fail-closed validation on all ownship & config floats */
    if (!is_finite3(own.pos_ned_m) || !is_finite3(own.vel_ned_mps) ||
        !std::isfinite(own.nom_omega_rad_s) || !std::isfinite(own.nom_az_down_mps2) ||
        !std::isfinite(own.nom_a_t_mps2) ||
        !is_valid_config(cfg)) {
        return make_safe_escape(own, cfg, 1.0, -kDefaultAzMaxMps2, -1.0, -1.0, 0U, true);
    }

    if (n_intruders > kMaxEcbfIntruders || (n_intruders > 0U && intruders == nullptr)) {
        return make_safe_escape(own, cfg, 1.0, -cfg.a_z_max_mps2, -1.0, -1.0, 0U, true);
    }

    for (std::size_t i = 0U; i < n_intruders; ++i) {
        if (intruders[i].valid) {
            if (!is_finite3(intruders[i].pos_ned_m) || !is_finite3(intruders[i].vel_ned_mps)) {
                return make_safe_escape(own, cfg, 1.0, -cfg.a_z_max_mps2, -1.0, -1.0, 0U, true);
            }
        }
    }

    const double vx = own.vel_ned_mps[0];
    const double vy = own.vel_ned_mps[1];
    const double vz = own.vel_ned_mps[2];
    const double v_h = std::sqrt(vx * vx + vy * vy);
    const double v_eff = std::max(v_h, cfg.v_min_mps);
    const double psi = (v_h > 1e-6) ? std::atan2(vy, vx) : 0.0;
    const double cos_psi = std::cos(psi);
    const double sin_psi = std::sin(psi);

    /* Box bounds enforcing bank, airspeed, and vertical speed viability */
    const double omega_max = (kGravityMps2 * std::tan(cfg.phi_max_rad)) / v_eff;
    const double az_lo = std::max(-cfg.a_z_max_mps2, (-cfg.v_z_max_mps - vz) / cfg.dt_eff_s);
    const double az_hi = std::min(cfg.a_z_max_mps2, (cfg.v_z_max_mps - vz) / cfg.dt_eff_s);
    const double at_lo = std::max(-cfg.a_t_max_mps2, (cfg.v_min_mps - v_h) / cfg.dt_eff_s);
    const double at_hi = std::min(cfg.a_t_max_mps2, (cfg.v_max_mps - v_h) / cfg.dt_eff_s);

    const double u_lo[4] = {-omega_max, std::min(az_lo, az_hi), std::min(at_lo, at_hi), 0.0};
    const double u_hi[4] = {omega_max, std::max(az_lo, az_hi), std::max(at_lo, at_hi), 100.0};
    const double u_nom[4] = {
        clamp_val(own.nom_omega_rad_s, u_lo[0], u_hi[0]),
        clamp_val(own.nom_az_down_mps2, u_lo[1], u_hi[1]),
        clamp_val(own.nom_a_t_mps2, u_lo[2], u_hi[2]),
        0.0};
    const double w_diag[4] = {cfg.w_omega, cfg.w_az, cfg.w_at, cfg.w_delta};

    /* Assemble 2nd-order ellipsoidal ECBF linear constraints in u = [omega, a_z_down, a_T, delta_wc] */
    LinearIneq ineqs[kMaxConstraints]{};
    std::size_t n_ineqs = 0U;
    double min_h_nmac = std::numeric_limits<double>::infinity();
    double min_h_wc = std::numeric_limits<double>::infinity();
    double best_escape_turn_sign = 1.0;
    double best_escape_az = -cfg.a_z_max_mps2;
    double closest_dist = std::numeric_limits<double>::infinity();

    /* Ownship horizontal acceleration Jacobian wrt [omega, a_T] in NED:
     *   a_N = a_T*cos(psi) - v_eff*omega*sin(psi)
     *   a_E = a_T*sin(psi) + v_eff*omega*cos(psi)
     *   a_D = a_z_down
     */
    constexpr double k1 = 0.65;
    constexpr double k2 = 0.40;

    for (std::size_t i = 0U; i < n_intruders; ++i) {
        if (!intruders[i].valid) {
            continue;
        }
        const double rx = own.pos_ned_m[0] - intruders[i].pos_ned_m[0];
        const double ry = own.pos_ned_m[1] - intruders[i].pos_ned_m[1];
        const double rz = own.pos_ned_m[2] - intruders[i].pos_ned_m[2];
        const double rvx = own.vel_ned_mps[0] - intruders[i].vel_ned_mps[0];
        const double rvy = own.vel_ned_mps[1] - intruders[i].vel_ned_mps[1];
        const double rvz = own.vel_ned_mps[2] - intruders[i].vel_ned_mps[2];

        const double dist = std::sqrt(rx * rx + ry * ry + rz * rz);
        if (dist < closest_dist) {
            closest_dist = dist;
            /* Right-of-way escape turn away from intruder bearing (14 CFR 91.113: exact head-on cross_z==0 -> RIGHT +1.0) */
            const double cross_z = cos_psi * (-ry) - sin_psi * (-rx);
            best_escape_turn_sign = (cross_z > 0.0) ? -1.0 : 1.0;
            /* Vertical escape: if ownship is above intruder (rz < 0 in NED), climb (a_z_down < 0) */
            best_escape_az = (rz <= 0.0) ? -cfg.a_z_max_mps2 : cfg.a_z_max_mps2;
        }

        const double r_wc = intruders[i].cooperative ? cfg.r_wc_coop_m : cfg.r_wc_noncoop_m;
        const double inv_r_nm2 = 1.0 / (cfg.r_nmac_eff_m * cfg.r_nmac_eff_m);
        const double inv_z_nm2 = 1.0 / (cfg.z_nmac_eff_m * cfg.z_nmac_eff_m);
        const double inv_r_wc2 = 1.0 / (r_wc * r_wc);
        const double inv_z_wc2 = 1.0 / (cfg.z_wc_m * cfg.z_wc_m);

        const double h_nm = (rx * rx + ry * ry) * inv_r_nm2 + (rz * rz) * inv_z_nm2 - 1.0;
        const double h_wc = (rx * rx + ry * ry) * inv_r_wc2 + (rz * rz) * inv_z_wc2 - 1.0;
        if (h_nm < min_h_nmac) {
            min_h_nmac = h_nm;
        }
        if (h_wc < min_h_wc) {
            min_h_wc = h_wc;
        }

        auto add_barrier = [&](double inv_r2, double inv_z2, double h_val, bool is_hard) {
            if (n_ineqs >= kMaxConstraints) {
                return;
            }
            const double h_dot = 2.0 * ((rx * rvx + ry * rvy) * inv_r2 + (rz * rvz) * inv_z2);
            const double drift_ddot = 2.0 * ((rvx * rvx + rvy * rvy) * inv_r2 + (rvz * rvz) * inv_z2);
            /* Only activate barrier when approaching or within 12x barrier volume */
            if (h_val > 12.0 && h_dot >= 0.0) {
                return;
            }
            /* L_g L_f h * u >= -(drift_ddot + k1*h_dot + k2*h_val) [- delta_wc if soft] */
            const double lg_omega = 2.0 * inv_r2 * (-rx * v_eff * sin_psi + ry * v_eff * cos_psi);
            const double lg_az = 2.0 * inv_z2 * rz;
            const double lg_at = 2.0 * inv_r2 * (rx * cos_psi + ry * sin_psi);
            const double rhs = drift_ddot + k1 * h_dot + k2 * h_val;

            /* Rewrite as a^T u <= b */
            LinearIneq &c = ineqs[n_ineqs++];
            c.a[0] = -lg_omega;
            c.a[1] = -lg_az;
            c.a[2] = -lg_at;
            c.a[3] = is_hard ? 0.0 : -1.0;
            c.b = rhs;
            c.is_hard = is_hard;
        };

        add_barrier(inv_r_nm2, inv_z_nm2, h_nm, true);
        add_barrier(inv_r_wc2, inv_z_wc2, h_wc, false);
    }

    /* Zero-malloc dual active-set / weighted Hildreth-Dykstra solver (<= 15 iterations) */
    double u[4] = {u_nom[0], u_nom[1], u_nom[2], u_nom[3]};
    double dual[kMaxConstraints]{};
    uint8_t iters_used = 0U;

    for (int iter = 0; iter < kMaxSolverIters; ++iter) {
        iters_used = static_cast<uint8_t>(iter + 1);
        double max_viol = 0.0;

        for (std::size_t ci = 0U; ci < n_ineqs; ++ci) {
            const LinearIneq &c = ineqs[ci];
            const double val = c.a[0] * u[0] + c.a[1] * u[1] + c.a[2] * u[2] + c.a[3] * u[3];
            const double viol = val - c.b;
            if (viol > max_viol) {
                max_viol = viol;
            }
            const double denom =
                (c.a[0] * c.a[0]) / w_diag[0] +
                (c.a[1] * c.a[1]) / w_diag[1] +
                (c.a[2] * c.a[2]) / w_diag[2] +
                (c.a[3] * c.a[3]) / w_diag[3];
            if (denom <= 1e-12) {
                continue;
            }
            const double step = viol / denom;
            const double new_dual = std::max(0.0, dual[ci] + step);
            dual[ci] = new_dual;

            /* Exact KKT primal projection onto box bounds (matches M1 Rust daa_ecbf.rs:291-297) */
            for (int k = 0; k < 4; ++k) {
                double grad_sum = 0.0;
                for (std::size_t j = 0U; j < n_ineqs; ++j) {
                    if (dual[j] > 0.0) {
                        grad_sum += dual[j] * ineqs[j].a[k];
                    }
                }
                u[k] = clamp_val(u_nom[k] - grad_sum / w_diag[k], u_lo[k], u_hi[k]);
            }
        }
        if (max_viol <= 1e-5) {
            break;
        }
    }

    /* Verify hard NMAC feasibility after bounded active-set solve (NaN-safe negation) */
    bool hard_feasible = true;
    for (std::size_t ci = 0U; ci < n_ineqs; ++ci) {
        if (!ineqs[ci].is_hard) {
            continue;
        }
        const LinearIneq &c = ineqs[ci];
        const double viol = (c.a[0] * u[0] + c.a[1] * u[1] + c.a[2] * u[2]) - c.b;
        if (!(viol <= 1e-2)) {
            hard_feasible = false;
            break;
        }
    }

    if (!hard_feasible) {
        return make_safe_escape(
            own, cfg, best_escape_turn_sign, best_escape_az, min_h_nmac, min_h_wc, iters_used, false);
    }

    const double bank_cmd = clamp_val(std::atan((u[0] * v_eff) / kGravityMps2), -cfg.phi_max_rad, cfg.phi_max_rad);
    const double vz_next = clamp_val(vz + u[1] * cfg.dt_eff_s, -cfg.v_z_max_mps, cfg.v_z_max_mps);
    const double climb_cmd = -vz_next;
    if (!std::isfinite(u[0]) || !std::isfinite(u[1]) || !std::isfinite(u[2]) || !std::isfinite(u[3]) ||
        !std::isfinite(bank_cmd) || !std::isfinite(climb_cmd)) {
        return make_safe_escape(
            own, cfg, best_escape_turn_sign, best_escape_az, min_h_nmac, min_h_wc, iters_used, true);
    }

    EcbfSolutionNed sol{};
    sol.omega_cmd_rad_s = u[0];
    sol.a_z_down_cmd_mps2 = u[1];
    sol.a_t_cmd_mps2 = u[2];
    sol.delta_wc = u[3];
    sol.bank_cmd_rad = bank_cmd;
    sol.climb_rate_cmd_mps = climb_cmd;
    sol.min_h_nmac = min_h_nmac;
    sol.min_h_wc = min_h_wc;
    sol.iterations = iters_used;
    sol.feasible = true;
    sol.safe_escape_active = false;
    sol.fail_closed = false;
    sol.solver_provenance = ecbf_solver_provenance();
    return sol;
}

}  // namespace xvs_rt
