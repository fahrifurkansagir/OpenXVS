/* OpenXVS v2.0 — Isolated Pure C-ABI Boundary for NASA DAIDALUS v2.0.4 (NOSA 1.3).
 *
 * Strictly POD structs + extern "C" functions only (zero STL / C++ exceptions across ABI).
 * Callers supply strict CONVENTIONS.md NED coordinates; NED <-> ENU conversion occurs
 * EXCLUSIVELY inside xvs_daidalus_bridge.cpp.
 */

#ifndef XVS_DAIDALUS_BRIDGE_H
#define XVS_DAIDALUS_BRIDGE_H

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct XvsDaidalusAircraftNed {
    double pos_ned_m[3];
    double vel_ned_mps[3];
    uint8_t valid;
    uint8_t _pad[7];
} XvsDaidalusAircraftNed;

typedef struct XvsDaidalusConfig {
    double dthr_m;         /* DO-365A horizontal WCV threshold (e.g. 1219.2 m = 4000 ft) */
    double zthr_m;         /* DO-365A vertical WCV threshold (e.g. 137.16 m = 450 ft) */
    double tthr_s;         /* DO-365A modified tau threshold (e.g. 35.0 s) */
    double lookahead_s;    /* Lookahead horizon (e.g. 60.0 s) */
    double nmac_h_m;       /* NMAC horizontal radius (152.4 m = 500 ft) */
    double nmac_v_m;       /* NMAC vertical half-height (30.48 m = 100 ft) */
} XvsDaidalusConfig;

typedef struct XvsDaidalusBandResult {
    int32_t alert_level;          /* 0=NONE, 1=PREVENTIVE, 2=CORRECTIVE, 3=WARNING, -1=FAULT */
    double min_time_to_wcv_s;     /* Time to Well-Clear violation (s), +inf if clear, NaN on fault */
    double min_hmd_m;             /* Predicted horizontal miss distance (m) */
    double min_vmd_m;             /* Predicted vertical miss distance (m) */
    double preferred_turn_rad;    /* Recommended horizontal heading change (rad, +right / -left) */
    double preferred_climb_mps;   /* Recommended vertical rate (+up m/s; NED vz = -climb) */
    uint8_t left_band_conflict;   /* 1 if left turn conflicts */
    uint8_t right_band_conflict;  /* 1 if right turn conflicts */
    uint8_t fail_closed;          /* 1 on any NaN/Inf/invalid input */
    uint8_t _pad[5];
} XvsDaidalusBandResult;

/* Return static adapter mode string identifying standalone emulator vs upstream NOSA drop-in. */
const char *xvs_daidalus_adapter_mode(void);

/* Populate default RTCA DO-365A non-cooperative / cooperative Well-Clear parameters. */
XvsDaidalusConfig xvs_daidalus_default_do365a_config(void);

/* Evaluate DO-365A Well-Clear conflict & maneuver bands in strict NED (converted to ENU inside). */
int32_t xvs_daidalus_eval_ned(
    const XvsDaidalusAircraftNed *ownship_ned,
    const XvsDaidalusAircraftNed *intruders_ned,
    size_t n_intruders,
    const XvsDaidalusConfig *cfg,
    XvsDaidalusBandResult *out_res);

#ifdef __cplusplus
}
#endif

#endif /* XVS_DAIDALUS_BRIDGE_H */
