/* OpenXVS v2.0 — Isolated DAIDALUS C-ABI Translation Unit.
 *
 * AUTHORIZED NED <-> ENU BOUNDARY:
 * This translation unit is the SOLE file in the entire xvs-rt build permitted to convert
 * between OpenXVS NED {+N, +E, +D} and NASA DAIDALUS ENU {+E, +N, +U}.
 * Zero dynamic heap allocations; zero exceptions; fail-closed on NaN / +/-Inf.
 */

#include "xvs_daidalus_bridge.h"

#include <cmath>
#include <limits>

namespace {

struct Vec3Enu {
    double e;
    double n;
    double u;
};

static bool is_finite_vec3(const double v[3]) {
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

/* Exact orthogonal NED -> ENU conversion: E = y_ned, N = x_ned, U = -z_ned. */
static Vec3Enu ned_to_enu(const double ned[3]) {
    return Vec3Enu{ned[1], ned[0], -ned[2]};
}

static bool config_valid(const XvsDaidalusConfig &cfg) {
    return std::isfinite(cfg.dthr_m) && cfg.dthr_m > 0.0 &&
           std::isfinite(cfg.zthr_m) && cfg.zthr_m > 0.0 &&
           std::isfinite(cfg.tthr_s) && cfg.tthr_s >= 0.0 &&
           std::isfinite(cfg.lookahead_s) && cfg.lookahead_s > 0.0 &&
           std::isfinite(cfg.nmac_h_m) && cfg.nmac_h_m > 0.0 &&
           std::isfinite(cfg.nmac_v_m) && cfg.nmac_v_m > 0.0;
}

static double eval_candidate_min_sep_enu(
    const Vec3Enu &own_p,
    const Vec3Enu &own_v,
    const XvsDaidalusAircraftNed *intruders_ned,
    size_t n_intruders,
    double lookahead_s,
    double dthr_m,
    double zthr_m) {
    double worst_ratio = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < n_intruders; ++i) {
        if (intruders_ned[i].valid == 0U) {
            continue;
        }
        const Vec3Enu ip = ned_to_enu(intruders_ned[i].pos_ned_m);
        const Vec3Enu iv = ned_to_enu(intruders_ned[i].vel_ned_mps);
        const double s_e = ip.e - own_p.e;
        const double s_n = ip.n - own_p.n;
        const double s_u = ip.u - own_p.u;
        const double v_e = iv.e - own_v.e;
        const double v_n = iv.n - own_v.n;
        const double v_u = iv.u - own_v.u;

        const double v_h2 = v_e * v_e + v_n * v_n;
        double t_cpa = 0.0;
        if (v_h2 > 1e-9) {
            t_cpa = -(s_e * v_e + s_n * v_n) / v_h2;
            if (t_cpa < 0.0) {
                t_cpa = 0.0;
            }
            if (t_cpa > lookahead_s) {
                t_cpa = lookahead_s;
            }
        }
        const double de = s_e + v_e * t_cpa;
        const double dn = s_n + v_n * t_cpa;
        const double du = s_u + v_u * t_cpa;
        const double hmd = std::sqrt(de * de + dn * dn);
        const double vmd = std::abs(du);
        const double ratio = std::max(hmd / dthr_m, vmd / zthr_m);
        if (ratio < worst_ratio) {
            worst_ratio = ratio;
        }
    }
    return worst_ratio;
}

}  // namespace

extern "C" {

const char *xvs_daidalus_adapter_mode(void) {
    return "STANDALONE_DO365_BOUNDARY_EMULATOR_READY_FOR_UPSTREAM_NOSA";
}

XvsDaidalusConfig xvs_daidalus_default_do365a_config(void) {
    XvsDaidalusConfig cfg{};
    cfg.dthr_m = 1219.2;      /* 4000 ft */
    cfg.zthr_m = 137.16;      /* 450 ft */
    cfg.tthr_s = 35.0;        /* 35 s modified tau */
    cfg.lookahead_s = 60.0;   /* 60 s alert lookahead */
    cfg.nmac_h_m = 152.4;     /* 500 ft */
    cfg.nmac_v_m = 30.48;     /* 100 ft */
    return cfg;
}

int32_t xvs_daidalus_eval_ned(
    const XvsDaidalusAircraftNed *ownship_ned,
    const XvsDaidalusAircraftNed *intruders_ned,
    size_t n_intruders,
    const XvsDaidalusConfig *cfg,
    XvsDaidalusBandResult *out_res) {
    if (out_res == nullptr) {
        return -1;
    }

    /* Fail-closed default output */
    out_res->alert_level = -1;
    out_res->min_time_to_wcv_s = std::numeric_limits<double>::quiet_NaN();
    out_res->min_hmd_m = std::numeric_limits<double>::quiet_NaN();
    out_res->min_vmd_m = std::numeric_limits<double>::quiet_NaN();
    out_res->preferred_turn_rad = 0.0;
    out_res->preferred_climb_mps = 0.0;
    out_res->left_band_conflict = 1U;
    out_res->right_band_conflict = 1U;
    out_res->fail_closed = 1U;

    if (ownship_ned == nullptr || cfg == nullptr || !config_valid(*cfg)) {
        return -1;
    }
    if (ownship_ned->valid == 0U ||
        !is_finite_vec3(ownship_ned->pos_ned_m) ||
        !is_finite_vec3(ownship_ned->vel_ned_mps)) {
        return -1;
    }
    if (n_intruders > 0U && intruders_ned == nullptr) {
        return -1;
    }

    for (size_t i = 0; i < n_intruders; ++i) {
        if (intruders_ned[i].valid != 0U) {
            if (!is_finite_vec3(intruders_ned[i].pos_ned_m) ||
                !is_finite_vec3(intruders_ned[i].vel_ned_mps)) {
                return -1;
            }
        }
    }

    const Vec3Enu own_p = ned_to_enu(ownship_ned->pos_ned_m);
    const Vec3Enu own_v = ned_to_enu(ownship_ned->vel_ned_mps);

    int32_t worst_alert = 0;
    double best_t_wcv = std::numeric_limits<double>::infinity();
    double best_hmd = std::numeric_limits<double>::infinity();
    double best_vmd = std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < n_intruders; ++i) {
        if (intruders_ned[i].valid == 0U) {
            continue;
        }
        const Vec3Enu ip = ned_to_enu(intruders_ned[i].pos_ned_m);
        const Vec3Enu iv = ned_to_enu(intruders_ned[i].vel_ned_mps);

        const double s_e = ip.e - own_p.e;
        const double s_n = ip.n - own_p.n;
        const double s_u = ip.u - own_p.u;
        const double v_e = iv.e - own_v.e;
        const double v_n = iv.n - own_v.n;
        const double v_u = iv.u - own_v.u;

        const double d_h = std::sqrt(s_e * s_e + s_n * s_n);
        const double d_v = std::abs(s_u);
        const double s_dot_v = s_e * v_e + s_n * v_n;
        const double v_h2 = v_e * v_e + v_n * v_n;

        /* Modified Tau (DO-365A TAUMOD) */
        double taumod = std::numeric_limits<double>::infinity();
        if (d_h <= cfg->dthr_m) {
            taumod = 0.0;
        } else if (s_dot_v < -1e-6) {
            taumod = (cfg->dthr_m * cfg->dthr_m - d_h * d_h) / s_dot_v;
            if (taumod < 0.0) {
                taumod = 0.0;
            }
        }

        /* Horizontal & vertical CPA inside [0, lookahead_s] */
        double t_cpa = 0.0;
        if (v_h2 > 1e-9) {
            t_cpa = -s_dot_v / v_h2;
            if (t_cpa < 0.0) {
                t_cpa = 0.0;
            }
            if (t_cpa > cfg->lookahead_s) {
                t_cpa = cfg->lookahead_s;
            }
        }
        const double de_cpa = s_e + v_e * t_cpa;
        const double dn_cpa = s_n + v_n * t_cpa;
        const double du_cpa = s_u + v_u * t_cpa;
        const double hmd = std::sqrt(de_cpa * de_cpa + dn_cpa * dn_cpa);
        const double vmd = std::abs(du_cpa);

        /* Time to co-altitude */
        double t_coa = std::numeric_limits<double>::infinity();
        if (d_v <= cfg->zthr_m) {
            t_coa = 0.0;
        } else if (s_u * v_u < -1e-6) {
            t_coa = -(s_u - (s_u > 0.0 ? cfg->zthr_m : -cfg->zthr_m)) / v_u;
            if (t_coa < 0.0) {
                t_coa = 0.0;
            }
        }

        if (hmd < best_hmd) {
            best_hmd = hmd;
        }
        if (vmd < best_vmd) {
            best_vmd = vmd;
        }

        const bool hor_wcv = (d_h <= cfg->dthr_m) || ((hmd <= cfg->dthr_m) && (taumod <= cfg->tthr_s));
        const bool ver_wcv = (d_v <= cfg->zthr_m) || (vmd <= cfg->zthr_m) || (t_coa <= cfg->tthr_s);

        int32_t level = 0;
        double t_wcv = std::numeric_limits<double>::infinity();
        if (hor_wcv && ver_wcv) {
            t_wcv = std::min(taumod, t_cpa);
            if (d_h <= cfg->dthr_m && d_v <= cfg->zthr_m) {
                level = 3; /* WARNING (active LoWC) */
                t_wcv = 0.0;
            } else if (t_wcv <= 25.0) {
                level = 3; /* WARNING */
            } else if (t_wcv <= 40.0) {
                level = 2; /* CORRECTIVE */
            } else if (t_wcv <= cfg->lookahead_s) {
                level = 1; /* PREVENTIVE */
            }
        } else if ((hmd <= 1.25 * cfg->dthr_m) && (vmd <= 1.25 * cfg->zthr_m) && (t_cpa <= cfg->lookahead_s)) {
            level = 1; /* PREVENTIVE */
            t_wcv = t_cpa;
        }

        if (level > worst_alert) {
            worst_alert = level;
        }
        if (t_wcv < best_t_wcv) {
            best_t_wcv = t_wcv;
        }
    }

    /* Evaluate +25 deg right vs -25 deg left heading resolution bands in ENU */
    const double speed_h = std::sqrt(own_v.e * own_v.e + own_v.n * own_v.n);
    const double hdg_enu = std::atan2(own_v.e, own_v.n); /* clockwise from North */
    const double turn_test = 0.4363323129985824;         /* 25 deg */

    Vec3Enu v_right{
        speed_h * std::sin(hdg_enu + turn_test),
        speed_h * std::cos(hdg_enu + turn_test),
        own_v.u};
    Vec3Enu v_left{
        speed_h * std::sin(hdg_enu - turn_test),
        speed_h * std::cos(hdg_enu - turn_test),
        own_v.u};

    const double sep_right = eval_candidate_min_sep_enu(
        own_p, v_right, intruders_ned, n_intruders, cfg->lookahead_s, cfg->dthr_m, cfg->zthr_m);
    const double sep_left = eval_candidate_min_sep_enu(
        own_p, v_left, intruders_ned, n_intruders, cfg->lookahead_s, cfg->dthr_m, cfg->zthr_m);

    out_res->alert_level = worst_alert;
    out_res->min_time_to_wcv_s = best_t_wcv;
    out_res->min_hmd_m = best_hmd;
    out_res->min_vmd_m = best_vmd;
    out_res->right_band_conflict = (sep_right < 1.0) ? 1U : 0U;
    out_res->left_band_conflict = (sep_left < 1.0) ? 1U : 0U;
    out_res->preferred_turn_rad = (worst_alert == 0) ? 0.0 : ((sep_right >= sep_left) ? turn_test : -turn_test);
    out_res->preferred_climb_mps = (worst_alert >= 2) ? 2.5 : 0.0;
    out_res->fail_closed = 0U;
    return 0;
}

}  // extern "C"
