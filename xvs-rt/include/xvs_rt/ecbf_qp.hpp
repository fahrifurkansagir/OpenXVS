/* OpenXVS v2.0 — 3D Fixed-Wing ECBF-QP in Strict NED (`xvs-rt`, DR-05:26-39 / SP3 §3.1 / BC-M3-M3).
 *
 * 4-variable QP: u = [omega, a_z_down, a_T, delta_wc]^T
 *   - omega      : turn rate (rad/s, +right / clockwise NED yaw rate)
 *   - a_z_down   : NED Down vertical acceleration (m/s^2, +Down; climb = -(v_z_ned + a_z_down * dt))
 *   - a_T        : tangential airspeed acceleration (m/s^2)
 *   - delta_wc   : non-negative Well-Clear soft slack variable (w_delta = 2e4)
 * Strictly zero heap allocation, zero exceptions, fail-closed on NaN / +/-Inf.
 */

#ifndef XVS_RT_ECBF_QP_HPP
#define XVS_RT_ECBF_QP_HPP

#pragma once

#include <cstddef>
#include <cstdint>

namespace xvs_rt {

static constexpr std::size_t kMaxEcbfIntruders = 16U;

struct EcbfIntruderNed {
    double pos_ned_m[3];
    double vel_ned_mps[3];
    bool cooperative;
    bool valid;
};

struct EcbfOwnshipNed {
    double pos_ned_m[3];
    double vel_ned_mps[3];
    double nom_omega_rad_s;
    double nom_az_down_mps2;
    double nom_a_t_mps2;
};

struct EcbfConfig {
    double phi_max_rad;         /* 30 deg = 0.5235987755982988 rad */
    double v_min_mps;           /* 22.0 m/s stall margin */
    double v_max_mps;           /* 65.0 m/s max cruise */
    double v_z_max_mps;         /* 6.0 m/s max climb/descent rate */
    double a_z_max_mps2;        /* 3.5 m/s^2 vertical accel limit */
    double a_t_max_mps2;        /* 2.5 m/s^2 longitudinal accel limit */
    double dt_eff_s;            /* 0.5 s command lookahead */
    double r_wc_noncoop_m;      /* 670.56 m (2200 ft) */
    double r_wc_coop_m;         /* 1219.2 m (4000 ft) */
    double z_wc_m;              /* 137.16 m (450 ft) */
    double r_nmac_eff_m;        /* 220.0 m inflated hard NMAC */
    double z_nmac_eff_m;        /* 50.0 m inflated hard NMAC */
    double w_omega;             /* 10.0 */
    double w_az;                /* 4.0 */
    double w_at;                /* 1.0 */
    double w_delta;             /* 20000.0 */
};

struct EcbfSolutionNed {
    double omega_cmd_rad_s;
    double a_z_down_cmd_mps2;
    double a_t_cmd_mps2;
    double delta_wc;
    double bank_cmd_rad;
    double climb_rate_cmd_mps;  /* +up m/s = -(v_z_ned + a_z_down * dt_eff) */
    double min_h_nmac;
    double min_h_wc;
    uint8_t iterations;
    bool feasible;
    bool safe_escape_active;
    bool fail_closed;
    const char *solver_provenance;
};

const char *ecbf_solver_provenance() noexcept;
EcbfConfig default_ecbf_config() noexcept;

EcbfSolutionNed solve_ecbf_qp_ned(
    const EcbfOwnshipNed &own,
    const EcbfIntruderNed *intruders,
    std::size_t n_intruders,
    const EcbfConfig &cfg) noexcept;

}  // namespace xvs_rt

#endif  // XVS_RT_ECBF_QP_HPP
