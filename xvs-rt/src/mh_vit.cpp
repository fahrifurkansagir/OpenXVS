/* OpenXVS v2.0 — K=5 Log-Spaced Interior MH-VIT in Strict NED (`DR-05:73-75`, `BC-M3-m2`). */

#include "xvs_rt/mh_vit.hpp"

#include <cmath>

namespace xvs_rt {

namespace {

static constexpr double kHalfPiPoleGuard = 1.5707963267948966 - 1e-6;

static bool is_finite3(const double v[3]) noexcept {
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

}  // namespace

MhVitResult generate_mh_vit_ned(const MhVitObservationNed &obs) noexcept {
    MhVitResult res{};
    res.valid = false;
    res.q_ratio = 0.0;
    res.max_rel_half_width = 0.0;

    if (!is_finite3(obs.own_pos_ned_m) || !is_finite3(obs.own_vel_ned_mps) ||
        !std::isfinite(obs.azimuth_rad) || !std::isfinite(obs.elevation_rad) ||
        !std::isfinite(obs.azimuth_rate_rad_s) || !std::isfinite(obs.elevation_rate_rad_s) ||
        !std::isfinite(obs.kappa_s_inv) ||
        !std::isfinite(obs.r_min_m) || !std::isfinite(obs.r_max_m) ||
        !std::isfinite(obs.sigma_ang_rad) || !std::isfinite(obs.sigma_rate_rad_s) ||
        !std::isfinite(obs.sigma_kappa_s_inv)) {
        return res;
    }

    if (obs.r_min_m <= 0.0 || obs.r_max_m <= obs.r_min_m) {
        return res;
    }
    if (obs.sigma_ang_rad < 0.0 || obs.sigma_rate_rad_s < 0.0 || obs.sigma_kappa_s_inv < 0.0) {
        return res;
    }
    if (std::abs(obs.elevation_rad) > kHalfPiPoleGuard) {
        return res;
    }

    const double cos_el = std::cos(obs.elevation_rad);
    const double sin_el = std::sin(obs.elevation_rad);
    const double cos_th = std::cos(obs.azimuth_rad);
    const double sin_th = std::sin(obs.azimuth_rad);

    /* Exact CONVENTIONS.md NED spherical orthonormal triad (BC-M3-m2) */
    const double u_hat[3] = {cos_el * cos_th, cos_el * sin_th, -sin_el};
    const double e_th[3] = {-sin_th, cos_th, 0.0};
    const double e_el[3] = {-sin_el * cos_th, -sin_el * sin_th, -cos_el};

    const double q = std::pow(obs.r_max_m / obs.r_min_m, 1.0 / static_cast<double>(kMhVitHypotheses));
    if (!std::isfinite(q) || q <= 1.0) {
        return res;
    }

    res.q_ratio = q;
    double max_rel_hw = 0.0;

    for (int k = 0; k < kMhVitHypotheses; ++k) {
        const double a_k = obs.r_min_m * std::pow(q, static_cast<double>(k));
        const double b_k = obs.r_min_m * std::pow(q, static_cast<double>(k + 1));
        const double r_k = 0.5 * (a_k + b_k);
        const double dr_k = 0.5 * (b_k - a_k);
        const double rel_hw = dr_k / r_k;
        if (rel_hw > max_rel_hw) {
            max_rel_hw = rel_hw;
        }

        MhVitHypothesisNed &h = res.hypotheses[static_cast<std::size_t>(k)];
        h.a_k_m = a_k;
        h.b_k_m = b_k;
        h.r_k_m = r_k;
        h.dr_k_m = dr_k;

        for (int axis = 0; axis < 3; ++axis) {
            h.pos_ned_m[axis] = obs.own_pos_ned_m[axis] + r_k * u_hat[axis];
            h.vel_ned_mps[axis] =
                obs.own_vel_ned_mps[axis] +
                r_k * (-obs.kappa_s_inv * u_hat[axis] +
                       obs.azimuth_rate_rad_s * cos_el * e_th[axis] +
                       obs.elevation_rate_rad_s * e_el[axis]);
        }

        /* Conservative SUM positional & velocity uncertainties */
        const double transverse_pos = r_k * obs.sigma_ang_rad;
        h.sigma_p_m = std::sqrt(dr_k * dr_k * (cos_el * cos_el) + transverse_pos * transverse_pos);
        h.sigma_z_m = std::sqrt(dr_k * dr_k * (sin_el * sin_el) + transverse_pos * transverse_pos);
        const double v_radial_sig = std::sqrt(
            (r_k * obs.sigma_kappa_s_inv) * (r_k * obs.sigma_kappa_s_inv) +
            (dr_k * std::abs(obs.kappa_s_inv)) * (dr_k * std::abs(obs.kappa_s_inv)));
        const double v_tan_sig = r_k * obs.sigma_rate_rad_s;
        h.sigma_v_mps = std::sqrt(v_radial_sig * v_radial_sig + v_tan_sig * v_tan_sig);
    }

    res.max_rel_half_width = max_rel_hw;
    res.valid = true;
    return res;
}

}  // namespace xvs_rt
