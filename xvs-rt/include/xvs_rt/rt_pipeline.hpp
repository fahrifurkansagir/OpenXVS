/* OpenXVS v2.0 — Real-Time Flight Loop Pipeline Binding M1 `libxvs_core.a` (`xvs_core.h`).
 *
 * Strictly zero heap allocation inside step_hot_tick():
 *   SPSC ingress -> xvs_rta_build_input_from_f64 -> xvs_rta_step -> K=5 MH-VIT ->
 *   C++20 3D ECBF-QP -> xvs_mesh_daa_eval -> xvs_fci_fuse_3d -> xvs_ecbf_qp_solve ->
 *   xvs_encode_mavlink_bearing_track -> DAIDALUS C-ABI -> SPSC egress.
 */

#ifndef XVS_RT_PIPELINE_HPP
#define XVS_RT_PIPELINE_HPP

#pragma once

#include <cstdint>

#include "xvs_core.h"
#include "xvs_daidalus_bridge.h"
#include "xvs_rt/ecbf_qp.hpp"
#include "xvs_rt/mh_vit.hpp"
#include "xvs_rt/spsc_ring.hpp"

namespace xvs_rt {

struct RtTickInputPacket {
    uint64_t tick_seq;
    XvsRtaFloatTelemetry rta_tel;
    MhVitObservationNed mh_vit_obs;
    EcbfOwnshipNed ownship;
    EcbfIntruderNed intruders[4];
    uint32_t n_intruders;
    XvsBearingTrack mesh_tracks[6];
    uint32_t n_mesh_tracks;
    double fci_xa[3];
    double fci_pa[9];
    double fci_xb[3];
    double fci_pb[9];
};

struct RtTickOutputPacket {
    uint64_t tick_seq;
    XvsRtaTickOutput rta_out;
    EcbfSolutionNed ecbf_cpp;
    XvsEcbfQpResult ecbf_rust;
    XvsMeshDaaEvalResult mesh_res;
    XvsDaidalusBandResult daidalus_res;
    double fci_x[3];
    double fci_p[9];
    int32_t mavlink_wire_len;
    uint8_t mavlink_wire_buf[96];
    bool mh_vit_valid;
};

class RtPipeline {
public:
    RtPipeline() noexcept = default;
    RtPipeline(const RtPipeline &) = delete;
    RtPipeline &operator=(const RtPipeline &) = delete;

    /* Must be called BEFORE arming malloc audit (calls xvs_rta_create()). */
    bool init() noexcept;

    /* Must be called AFTER disarming malloc audit (calls xvs_rta_destroy()). */
    void shutdown() noexcept;

    /* Zero-alloc hot-loop execution of one 10 Hz flight tick. */
    bool step_hot_tick(const RtTickInputPacket &in_pkt, RtTickOutputPacket &out_pkt) noexcept;

private:
    XvsRtaHandle *rta_handle_{nullptr};
    EcbfConfig ecbf_cfg_{default_ecbf_config()};
    XvsDaidalusConfig daidalus_cfg_{xvs_daidalus_default_do365a_config()};
    SpscRing<RtTickInputPacket, 16U> ingress_ring_{};
    SpscRing<RtTickOutputPacket, 16U> egress_ring_{};
    uint8_t mav_secret_key_[32]{
        0x11U, 0x22U, 0x33U, 0x44U, 0x55U, 0x66U, 0x77U, 0x88U,
        0x99U, 0xAAU, 0xBBU, 0xCCU, 0xDDU, 0xEEU, 0xFFU, 0x01U,
        0x12U, 0x23U, 0x34U, 0x45U, 0x56U, 0x67U, 0x78U, 0x89U,
        0x9AU, 0xABU, 0xBCU, 0xCDU, 0xDEU, 0xEFU, 0xF0U, 0x02U};
};

}  // namespace xvs_rt

#endif  // XVS_RT_PIPELINE_HPP
