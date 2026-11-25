#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

//! OpenXVS v2.0 `xvs-math-core` — `#![no_std]` deterministic `DetF64` flight dynamics,
//! WGS-84 & `SO(3)` geometry, `XVS6-CUBE-v1`/`TRAFFIC-RING-v1` rig models, dual-bank 15-state
//! `ES-EKF` with Bonferroni 3-layer spoofing guard, multi-cam Vision-RAIM, `XVS-MeshDAA`
//! 3-ship triple-ray + `FCI`, slack-augmented 3D `ECBF-QP` with deterministic escape fallback,
//! OOD write-gated `GuardedDtAci`, bit-exact GDL-90 & MAVLink v2 codecs, and `LiveDeckEngine`.

pub mod codecs;
pub mod daa_ecbf;
pub mod det_f64;
pub mod dtaci;
pub mod es_ekf;
pub mod flight6dof;
pub mod geo;
pub mod mesh_daa;
pub mod rig;
pub mod sim_engine;
pub mod vraim;

pub use codecs::{
    compute_gdl90_crc, decode_gdl90_frame, decode_mavlink_v2_adsb_246,
    decode_mavlink_v2_bearing_track, encode_gdl90_heartbeat, encode_gdl90_ownship,
    encode_gdl90_ownship_geo_alt, encode_gdl90_traffic, encode_mavlink_v2_adsb_246,
    encode_mavlink_v2_bearing_track, frame_gdl90_message, CodecError, Gdl90TargetParams,
    MavlinkAdsb246Params,
};
pub use daa_ecbf::{EcbfQpSolution, EcbfQpSolver, IntruderStateNed, Wc1AlertStatus};
pub use det_f64::{sanitize_nan, DetF64, Mat3, SplitMix64, Vec3};
pub use dtaci::GuardedDtAci;
pub use es_ekf::{DualBankEsEkf, SpoofCheckInput, SpoofCheckOutput};
pub use flight6dof::{FlightControlCmd, FlightModel6Dof, FlightState};
pub use geo::{
    ecef_to_geodetic, euler_to_r, euler_zyx_to_quat, geodetic_to_ecef, quat_to_r, r_ecef_ned,
    r_to_euler, r_to_quat, so3_exp, so3_log,
};
pub use mesh_daa::{
    fci_fuse_3d, FusedMeshTrack, MeshDaaFuser, MeshFusionResult, OpenXvsBearingTrack,
    OPENXVS_BEARING_TRACK_PAYLOAD_LEN,
};
pub use rig::{
    camera_pose_ecef, project_ecef_to_cam, PinholeCamera, RigCamera, TRAFFIC_RING_V1, XVS6_CUBE_V1,
};
pub use sim_engine::{FaultPerturbations, LiveDeckEngine, MissionId, TelemetrySnapshot};
pub use vraim::{
    HorizonIsolationResult, RunwayVraimResult, VisionRaimEvaluator, KSFO_28L_WIDTH_M,
    KSFO_TAXIWAY_C_WIDTH_M,
};
