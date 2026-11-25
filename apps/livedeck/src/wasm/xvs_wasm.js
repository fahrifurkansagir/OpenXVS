/* @ts-self-types="./xvs_wasm.d.ts" */

export class WasmLiveDeckSim {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmLiveDeckSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmlivedecksim_free(ptr, 0);
    }
    clear_tx_ring() {
        wasm.wasmlivedecksim_clear_tx_ring(this.__wbg_ptr);
    }
    /**
     * Drain queued GDL-90 (`proto=0x01`) and MAVLink v2 (`proto=0x02`) packets into the flat
     * `tx_ring` buffer using `[proto: u8, len_lo: u8, len_hi: u8, payload...]` framing.
     * @returns {number}
     */
    drain_tx_into_ring() {
        const ret = wasm.wasmlivedecksim_drain_tx_into_ring(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} seed
     */
    constructor(seed) {
        const ret = wasm.wasmlivedecksim_new(seed);
        this.__wbg_ptr = ret;
        WasmLiveDeckSimFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {number} mission_id
     * @param {number} seed
     */
    reset_mission(mission_id, seed) {
        wasm.wasmlivedecksim_reset_mission(this.__wbg_ptr, mission_id, seed);
    }
    /**
     * @param {number} cam_r_bias_rad
     * @param {number} spoof_vn_mps
     * @param {number} spoof_ve_mps
     * @param {number} spoof_vd_mps
     * @param {number} runway_corner_bias_px
     * @param {number} runway_observed_width_m
     * @param {number} fog_contrast
     * @param {number} ood_global_score
     * @param {boolean} lwir_efvs_enabled
     * @param {boolean} transient_dropout
     * @param {boolean} ml_detector_fault
     * @param {boolean} go_around_ack_reset
     * @param {boolean} peer3_active
     * @param {boolean} manual_override
     * @param {number} manual_bank_cmd_rad
     * @param {number} manual_climb_cmd_mps
     * @param {boolean} foreflight_sim_prefix
     */
    set_faults(cam_r_bias_rad, spoof_vn_mps, spoof_ve_mps, spoof_vd_mps, runway_corner_bias_px, runway_observed_width_m, fog_contrast, ood_global_score, lwir_efvs_enabled, transient_dropout, ml_detector_fault, go_around_ack_reset, peer3_active, manual_override, manual_bank_cmd_rad, manual_climb_cmd_mps, foreflight_sim_prefix) {
        wasm.wasmlivedecksim_set_faults(this.__wbg_ptr, cam_r_bias_rad, spoof_vn_mps, spoof_ve_mps, spoof_vd_mps, runway_corner_bias_px, runway_observed_width_m, fog_contrast, ood_global_score, lwir_efvs_enabled, transient_dropout, ml_detector_fault, go_around_ack_reset, peer3_active, manual_override, manual_bank_cmd_rad, manual_climb_cmd_mps, foreflight_sim_prefix);
    }
    /**
     * @returns {number}
     */
    snapshot_f32_len() {
        const ret = wasm.wasmlivedecksim_snapshot_f32_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    snapshot_f32_ptr() {
        const ret = wasm.wasmlivedecksim_snapshot_f32_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    snapshot_f64_len() {
        const ret = wasm.wasmlivedecksim_snapshot_f64_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    snapshot_f64_ptr() {
        const ret = wasm.wasmlivedecksim_snapshot_f64_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Advance `dt_ticks` physics substeps (`100 Hz`, `0.01 s` each) and update zero-copy views.
     * @param {number} dt_ticks
     */
    step_100hz(dt_ticks) {
        wasm.wasmlivedecksim_step_100hz(this.__wbg_ptr, dt_ticks);
    }
    /**
     * Advance exactly one 10 Hz RTA tick (`10` physics substeps = `0.10 s`).
     */
    step_rta_tick() {
        wasm.wasmlivedecksim_step_rta_tick(this.__wbg_ptr);
    }
    /**
     * @returns {number}
     */
    tx_ring_len() {
        const ret = wasm.wasmlivedecksim_tx_ring_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    tx_ring_ptr() {
        const ret = wasm.wasmlivedecksim_tx_ring_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) WasmLiveDeckSim.prototype[Symbol.dispose] = WasmLiveDeckSim.prototype.free;

/**
 * @returns {any}
 */
export function wasm_memory() {
    const ret = wasm.wasm_memory();
    return ret;
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_memory_17a1178bc3328ea3: function() {
            const ret = wasm.memory;
            return ret;
        },
        __wbg___wbindgen_throw_4c88d8bf264fd984: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            table.set(0, undefined);
            table.set(1024, undefined);
            table.set(1025, null);
            table.set(1026, true);
            table.set(1027, false);
        },
    };
    return {
        __proto__: null,
        "./xvs_wasm_bg.js": import0,
    };
}

const WasmLiveDeckSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmlivedecksim_free(ptr, 1));

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(bytes) {
    numBytesDecoded += bytes.length;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true });
        cachedTextDecoder.decode();
        numBytesDecoded = bytes.length;
    }
    return cachedTextDecoder.decode(bytes);
}

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (!module.ok) {
            throw new Error(`failed to fetch Wasm: ${module.status} ${module.statusText} fetching '${module.url}'`);
        }

        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

let __wbg_init_promise;
async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;
    if (__wbg_init_promise === undefined) {
        __wbg_init_promise = __wbg_init_once(module_or_path);
        __wbg_init_promise.catch(() => { __wbg_init_promise = undefined; });
    }
    return __wbg_init_promise;
}

async function __wbg_init_once(module_or_path) {

    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('xvs_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
