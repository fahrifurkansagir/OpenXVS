/* OpenXVS v2.0 — K=5 Log-Spaced Interior Multi-Hypothesis Virtual Intruder Tracking (`MH-VIT`).
 *
 * Strictly CONVENTIONS.md NED {+N, +E, +D} (ZERO internal ENU!), DR-05:73-75 & SP3 §3.1 (BC-M3-m2):
 *   u_hat   = [ cos(el) * cos(theta),  cos(el) * sin(theta), -sin(el) ]
 *   e_theta = [-sin(theta),            cos(theta),            0.0     ]
 *   e_el    = [-sin(el) * cos(theta), -sin(el) * sin(theta), -cos(el) ]
 * Pole-guarded (|el| <= pi/2 - 1e-6) and fail-closed on NaN / +/-Inf / invalid range bounds.
 */

#ifndef XVS_RT_MH_VIT_HPP
#define XVS_RT_MH_VIT_HPP

#pragma once

#include <array>
#include <cstdint>

namespace xvs_rt {

static constexpr int kMhVitHypotheses = 5;

struct MhVitObservationNed {
    double own_pos_ned_m[3];
    double own_vel_ned_mps[3];
    double azimuth_rad;        /* theta: clockwise from North */
    double elevation_rad;      /* el: positive above horizon (-z_D) */
    double azimuth_rate_rad_s; /* theta_dot */
    double elevation_rate_rad_s; /* el_dot */
    double kappa_s_inv;        /* scale-rate closure kappa = -r_dot / r (s^-1) */
    double r_min_m;            /* conservative lower range bound (> 0) */
    double r_max_m;            /* conservative upper range bound (> r_min_m) */
    double sigma_ang_rad;      /* bearing angular 1-sigma (rad) */
    double sigma_rate_rad_s;   /* angular rate 1-sigma (rad/s) */
    double sigma_kappa_s_inv;  /* scale-rate 1-sigma (s^-1) */
};

struct MhVitHypothesisNed {
    double a_k_m;
    double b_k_m;
    double r_k_m;
    double dr_k_m;
    double pos_ned_m[3];
    double vel_ned_mps[3];
    double sigma_p_m;
    double sigma_z_m;
    double sigma_v_mps;
};

struct MhVitResult {
    bool valid;
    double q_ratio;
    double max_rel_half_width;
    std::array<MhVitHypothesisNed, kMhVitHypotheses> hypotheses;
};

MhVitResult generate_mh_vit_ned(const MhVitObservationNed &obs) noexcept;

}  // namespace xvs_rt

#endif  // XVS_RT_MH_VIT_HPP
