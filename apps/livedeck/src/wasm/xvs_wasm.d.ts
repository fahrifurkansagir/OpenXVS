/* tslint:disable */
/* eslint-disable */

export class WasmLiveDeckSim {
    free(): void;
    [Symbol.dispose](): void;
    clear_tx_ring(): void;
    /**
     * Drain queued GDL-90 (`proto=0x01`) and MAVLink v2 (`proto=0x02`) packets into the flat
     * `tx_ring` buffer using `[proto: u8, len_lo: u8, len_hi: u8, payload...]` framing.
     */
    drain_tx_into_ring(): number;
    constructor(seed: number);
    reset_mission(mission_id: number, seed: number): void;
    set_faults(cam_r_bias_rad: number, spoof_vn_mps: number, spoof_ve_mps: number, spoof_vd_mps: number, runway_corner_bias_px: number, runway_observed_width_m: number, fog_contrast: number, ood_global_score: number, lwir_efvs_enabled: boolean, transient_dropout: boolean, ml_detector_fault: boolean, go_around_ack_reset: boolean, peer3_active: boolean, manual_override: boolean, manual_bank_cmd_rad: number, manual_climb_cmd_mps: number, foreflight_sim_prefix: boolean): void;
    snapshot_f32_len(): number;
    snapshot_f32_ptr(): number;
    snapshot_f64_len(): number;
    snapshot_f64_ptr(): number;
    /**
     * Advance `dt_ticks` physics substeps (`100 Hz`, `0.01 s` each) and update zero-copy views.
     */
    step_100hz(dt_ticks: number): void;
    /**
     * Advance exactly one 10 Hz RTA tick (`10` physics substeps = `0.10 s`).
     */
    step_rta_tick(): void;
    tx_ring_len(): number;
    tx_ring_ptr(): number;
}

export function wasm_memory(): any;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmlivedecksim_free: (a: number, b: number) => void;
    readonly wasm_memory: () => any;
    readonly wasmlivedecksim_clear_tx_ring: (a: number) => void;
    readonly wasmlivedecksim_drain_tx_into_ring: (a: number) => number;
    readonly wasmlivedecksim_new: (a: number) => number;
    readonly wasmlivedecksim_reset_mission: (a: number, b: number, c: number) => void;
    readonly wasmlivedecksim_set_faults: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number) => void;
    readonly wasmlivedecksim_snapshot_f32_len: (a: number) => number;
    readonly wasmlivedecksim_snapshot_f32_ptr: (a: number) => number;
    readonly wasmlivedecksim_snapshot_f64_len: (a: number) => number;
    readonly wasmlivedecksim_snapshot_f64_ptr: (a: number) => number;
    readonly wasmlivedecksim_step_100hz: (a: number, b: number) => void;
    readonly wasmlivedecksim_step_rta_tick: (a: number) => void;
    readonly wasmlivedecksim_tx_ring_len: (a: number) => number;
    readonly wasmlivedecksim_tx_ring_ptr: (a: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
