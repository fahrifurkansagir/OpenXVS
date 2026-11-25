/* OpenXVS v2.0 — Zero-Alloc Real-Time Flight Loop Binding M1 `libxvs_core.a` (`DR-06`). */

#include "xvs_rt/rt_pipeline.hpp"

#include <cmath>

namespace xvs_rt {

bool RtPipeline::init() noexcept {
    if (rta_handle_ != nullptr) {
        xvs_rta_destroy(rta_handle_);
        rta_handle_ = nullptr;
    }
    ingress_ring_.reset();
    egress_ring_.reset();
    rta_handle_ = xvs_rta_create();
    return rta_handle_ != nullptr;
}

void RtPipeline::shutdown() noexcept {
    if (rta_handle_ != nullptr) {
        xvs_rta_destroy(rta_handle_);
        rta_handle_ = nullptr;
    }
}

bool RtPipeline::step_hot_tick(const RtTickInputPacket &in_pkt, RtTickOutputPacket &out_pkt) noexcept {
    if (rta_handle_ == nullptr) {
        return false;
    }

    /* 1. Wait-free SPSC ingress */
    (void)ingress_ring_.push_overwrite_latest(in_pkt);
    RtTickInputPacket work{};
    if (!ingress_ring_.try_pop(work)) {
        return false;
    }

    RtTickOutputPacket res{};
    res.tick_seq = work.tick_seq;

    /* 2. Fail-closed f64 -> i64 quantizer + Simplex RTA + past-time MLTL (libxvs_core.a) */
    XvsRtaTickInput rta_in{};
    if (xvs_rta_build_input_from_f64(&work.rta_tel, &rta_in) != 0) {
        return false;
    }
    if (xvs_rta_step(rta_handle_, &rta_in, &res.rta_out) != 0) {
        return false;
    }

    /* 3. K=5 log-spaced interior MH-VIT strictly in NED */
    const MhVitResult mh = generate_mh_vit_ned(work.mh_vit_obs);
    res.mh_vit_valid = mh.valid;

    /* 4. C++20 4-var NED ECBF-QP */
    const std::size_t n_int = (work.n_intruders <= 4U) ? static_cast<std::size_t>(work.n_intruders) : 4U;
    res.ecbf_cpp = solve_ecbf_qp_ned(work.ownship, work.intruders, n_int, ecbf_cfg_);

    /* 5. M1 Rust xvs_ecbf_qp_solve via libxvs_core.a */
    XvsIntruderNed rust_ints[4]{};
    XvsDaidalusAircraftNed daid_ints[4]{};
    for (std::size_t i = 0U; i < n_int; ++i) {
        for (int a = 0; a < 3; ++a) {
            rust_ints[i].pos_ned[a] = work.intruders[i].pos_ned_m[a];
            rust_ints[i].vel_ned[a] = work.intruders[i].vel_ned_mps[a];
            daid_ints[i].pos_ned_m[a] = work.intruders[i].pos_ned_m[a];
            daid_ints[i].vel_ned_mps[a] = work.intruders[i].vel_ned_mps[a];
        }
        rust_ints[i].valid = work.intruders[i].valid ? 1U : 0U;
        daid_ints[i].valid = work.intruders[i].valid ? 1U : 0U;
    }
    const double nom_climb_mps = -work.ownship.vel_ned_mps[2];
    if (xvs_ecbf_qp_solve(
            work.ownship.pos_ned_m,
            work.ownship.vel_ned_mps,
            0.0,
            nom_climb_mps,
            rust_ints,
            n_int,
            &res.ecbf_rust) != 0) {
        return false;
    }

    /* 6. Isolated DAIDALUS C-ABI boundary (NED <-> ENU happens strictly inside bridge.cpp) */
    XvsDaidalusAircraftNed daid_own{};
    for (int a = 0; a < 3; ++a) {
        daid_own.pos_ned_m[a] = work.ownship.pos_ned_m[a];
        daid_own.vel_ned_mps[a] = work.ownship.vel_ned_mps[a];
    }
    daid_own.valid = 1U;
    if (xvs_daidalus_eval_ned(&daid_own, daid_ints, n_int, &daidalus_cfg_, &res.daidalus_res) != 0) {
        return false;
    }

    /* 7. M1 Rust MeshDAA triple-ray fusion (zero-alloc stack buffer after BC-M3-B1!) */
    const std::size_t n_trk = (work.n_mesh_tracks <= 6U) ? static_cast<std::size_t>(work.n_mesh_tracks) : 6U;
    if (xvs_mesh_daa_eval(work.mesh_tracks, n_trk, &res.mesh_res) != 0) {
        return false;
    }

    /* 8. M1 Rust Fast Covariance Intersection (FCI) */
    if (xvs_fci_fuse_3d(
            work.fci_xa,
            work.fci_pa,
            work.fci_xb,
            work.fci_pb,
            res.fci_x,
            res.fci_p) != 0) {
        return false;
    }

    /* 9. M1 Rust signed 77-byte MAVLink v2 #52001 OPENXVS_BEARING_TRACK encoder */
    if (n_trk > 0U) {
        res.mavlink_wire_len = xvs_encode_mavlink_bearing_track(
            &work.mesh_tracks[0],
            static_cast<uint8_t>(work.tick_seq & 0xFFU),
            1U,
            156U,
            0U,
            work.tick_seq * 10000U,
            mav_secret_key_,
            res.mavlink_wire_buf,
            sizeof(res.mavlink_wire_buf));
    } else {
        res.mavlink_wire_len = 0;
    }

    /* 10. Wait-free SPSC egress */
    (void)egress_ring_.push_overwrite_latest(res);
    return egress_ring_.try_pop(out_pkt);
}

}  // namespace xvs_rt
