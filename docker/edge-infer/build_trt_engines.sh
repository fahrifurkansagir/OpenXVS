#!/usr/bin/env bash
# ==============================================================================
# OpenXVS Jetson Orin Day-1 TensorRT Engine Builder & Benchmark Kit
# ==============================================================================
# Compiles FP16 (and optional INT8 calibration) TensorRT `.plan` engines via
# `trtexec` for all 6 OpenXVS ONNX models on NVIDIA Jetson Orin (Orin Nano Super
# 8GB / Orin NX / Orin AGX), supporting BOTH:
#   - `v1.1` profile: 256x256 XVS-D2 + 6-output XVS-Land + 1.76 MiB Thermal
#     + XVS-Nav + XFeat 320x256 + LighterGlue K=256 (all SHA-256 pinned)
#   - `extended` profile: 640x640 XVS-D2 + 7-output XVS-Land (`runway_present_logit`)
#     + 1.13 MiB XVS-ThermalNet (disagreement vacuity) + Nav/XFeat/LighterGlue
#
# MANDATORY NVIDIA TENSORRT SLA & NON-FLIGHT BOUNDARY:
#   SIMULATION & BENCHMARK EVALUATION ONLY -- ZERO ACTUATION -- NOT FOR FLIGHT
#   - NON_FLIGHT_SIM_ONLY=1 (ACTUATOR_AUTHORITY=false)
#   - Subject to NVIDIA TensorRT SLA Sec. 1.2 & Sec. 2 (Critical Applications):
#     compiled `.plan` files are machine-local evaluation caches and MUST NOT be
#     redistributed or deployed in certified/life-critical flight control.
#   - Strictly civil/dual-use, zero GPL/AGPL, strictly NED/FRD conventions.
# ==============================================================================

set -euo pipefail

export NON_FLIGHT_SIM_ONLY="${NON_FLIGHT_SIM_ONLY:-1}"
export OPENXVS_NON_FLIGHT_SIM_ONLY="${OPENXVS_NON_FLIGHT_SIM_ONLY:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PROFILE="v1.1"
BOARD="orin-nano-super"
PRECISION="fp16"
MODEL_DIR="${OPENXVS_MODEL_DIR:-${REPO_ROOT}/models}"
OUT_DIR="${OPENXVS_TRT_ENGINE_CACHE:-${REPO_ROOT}/outputs/jetson_trt_plans}"
CALIB_DIR=""
MANIFEST_PATH=""
WORKSPACE_MB=""
WARMUP_MS=1000
ITERATIONS=200
DRY_RUN=0
ALLOW_UNPINNED_V53=0
SKIP_BENCHMARK=0
TRTEXEC_BIN="${TRTEXEC_BIN:-trtexec}"

# ---- Fail-Closed Pinned SHA-256 Digests (`v1.1` / `v5.2` + Faz-B Nav Triad) ----
PIN_D2_V11_SHA256="2c9578f362f6bbbcc0f9ce1de925bf5dbce5459d75dfb387c79c810023750d0d"
PIN_LAND_V52_SHA256="834c8e270f237f044d3920c98d5c0818363ab781be78071884e9365a221ac799"
PIN_THERMAL_V52_SHA256="bc6ceac6252f48dc52305c4e7955cb7841bc5adc7c191a59ce9b0e36ef10fcac"
PIN_NAV_FAZB_SHA256="e48f0ed5a234dcf1661031b80db5028d8c5b926b7b54e06575b4ed20bd6fc46f"
PIN_XFEAT_FP16_SHA256="e28a2f3afd7e23fe9818ecea34b57185708ff2e2b45f24cb514628f49c3d8182"
PIN_LIGHTERGLUE_FP16_SHA256="e63121ce24b840dba04acd490133eba4175a2ae43b087db3dac6e9d2f3220f3b"

TEGRASTATS_PID=""

cleanup() {
  if [ -n "${TEGRASTATS_PID}" ]; then
    if kill -0 "${TEGRASTATS_PID}" 2>/dev/null; then
      kill -TERM "${TEGRASTATS_PID}" 2>/dev/null || true
      wait "${TEGRASTATS_PID}" 2>/dev/null || true
    fi
    TEGRASTATS_PID=""
  fi
}
trap cleanup EXIT INT TERM

usage() {
  cat <<'EOF'
Usage: build_trt_engines.sh [OPTIONS]

Options:
  --profile <v1.1|extended>      Model suite profile (default: v1.1)
                                   v1.1: 256x256 D2 + 6-out Land + Thermal + Nav/XFeat/LG
                                   extended: 640x640 D2 + 7-out Land + 1.13 MiB Thermal + Nav/XFeat/LG
  --board <orin-nano-super|orin-nx|orin-agx>
                                 Target Jetson Orin RAM/workspace profile (default: orin-nano-super)
  --precision <fp16|int8>        TensorRT precision mode (default: fp16; int8 enables --int8 --fp16)
  --calib-dir <DIR>              INT8 calibration cache directory (optional with --precision int8)
  --model-dir <DIR>              Directory containing ONNX models (default: $OPENXVS_MODEL_DIR or ./models)
  --out-dir <DIR>                Output directory for .plan engines & logs (default: ./outputs/jetson_trt_plans)
  --manifest <FILE>              Optional JSON or sha256sum manifest pinning extended ONNX SHA-256 digests
  --workspace-mb <INT>           Override TensorRT workspace pool MiB (default: board-specific cap)
  --warmup-ms <INT>              trtexec warmup duration ms (default: 1000)
  --iterations <INT>             trtexec timed iterations for p50/p95/p99 (default: 200)
  --skip-benchmark               Build engines only; skip p50/p95/p99 timing pass
  --allow-unpinned-v53           Permit building unpinned local extended ONNX exports (records computed SHA-256)
  --dry-run                      Verify paths/digests and print exact trtexec commands without invoking GPU
  -h, --help                     Show this help message
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --profile)
      PROFILE="${2:?Missing value for --profile}"
      shift 2
      ;;
    --board)
      BOARD="${2:?Missing value for --board}"
      shift 2
      ;;
    --precision)
      PRECISION="${2:?Missing value for --precision}"
      shift 2
      ;;
    --calib-dir)
      CALIB_DIR="${2:?Missing value for --calib-dir}"
      shift 2
      ;;
    --model-dir)
      MODEL_DIR="${2:?Missing value for --model-dir}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:?Missing value for --out-dir}"
      shift 2
      ;;
    --manifest)
      MANIFEST_PATH="${2:?Missing value for --manifest}"
      shift 2
      ;;
    --workspace-mb)
      WORKSPACE_MB="${2:?Missing value for --workspace-mb}"
      shift 2
      ;;
    --warmup-ms)
      WARMUP_MS="${2:?Missing value for --warmup-ms}"
      shift 2
      ;;
    --iterations)
      ITERATIONS="${2:?Missing value for --iterations}"
      shift 2
      ;;
    --skip-benchmark)
      SKIP_BENCHMARK=1
      shift
      ;;
    --allow-unpinned-v53)
      ALLOW_UNPINNED_V53=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${PROFILE}" in
  v1.1|extended) ;;
  *)
    echo "[ERROR] Unsupported --profile '${PROFILE}' (expected v1.1 or extended)" >&2
    exit 2
    ;;
esac

case "${PRECISION}" in
  fp16|int8) ;;
  *)
    echo "[ERROR] Unsupported --precision '${PRECISION}' (expected fp16 or int8)" >&2
    exit 2
    ;;
esac

if [ -z "${WORKSPACE_MB}" ]; then
  case "${BOARD}" in
    orin-nano-super) WORKSPACE_MB=1024 ;;
    orin-nx)         WORKSPACE_MB=2048 ;;
    orin-agx)        WORKSPACE_MB=4096 ;;
    *)
      echo "[ERROR] Unsupported --board '${BOARD}' (expected orin-nano-super, orin-nx, or orin-agx)" >&2
      exit 2
      ;;
  esac
fi

if [ "${NON_FLIGHT_SIM_ONLY}" != "1" ]; then
  echo "[ERROR] Refusing TensorRT build: NON_FLIGHT_SIM_ONLY must remain 1 (NVIDIA TRT SLA Sec. 2)" >&2
  exit 3
fi

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/timing"

echo "================================================================================"
echo "OPENXVS JETSON ORIN DAY-1 TENSORRT ENGINE BUILDER (NON_FLIGHT_SIM_ONLY=1)"
echo "================================================================================"
echo "WATERMARK : SIMULATION & BENCHMARK EVALUATION ONLY -- ZERO ACTUATION -- NOT FOR FLIGHT"
echo "EULA LOCK : NVIDIA TensorRT SLA Sec. 1.2 & 2 (local eval .plan cache only; DO NOT redistribute)"
echo "PROFILE   : ${PROFILE} | BOARD: ${BOARD} | PRECISION: ${PRECISION} | WORKSPACE: ${WORKSPACE_MB} MiB"
echo "MODEL_DIR : ${MODEL_DIR}"
echo "OUT_DIR   : ${OUT_DIR}"
echo "================================================================================"

if [ "${DRY_RUN}" -eq 0 ] && ! command -v "${TRTEXEC_BIN}" >/dev/null 2>&1; then
  if [ -x "/usr/src/tensorrt/bin/trtexec" ]; then
    TRTEXEC_BIN="/usr/src/tensorrt/bin/trtexec"
  else
    echo "[ERROR] trtexec not found in PATH or /usr/src/tensorrt/bin/trtexec (use --dry-run on non-Jetson hosts)" >&2
    exit 4
  fi
fi

sha256_file() {
  local target="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${target}" | awk '{print tolower($1)}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${target}" | awk '{print tolower($1)}'
  elif command -v python3 >/dev/null 2>&1; then
    python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${target}"
  else
    echo "[ERROR] No sha256sum/shasum/python3 utility available for fail-closed digest verification" >&2
    exit 5
  fi
}

lookup_manifest_sha256() {
  local filename="$1"
  if [ -z "${MANIFEST_PATH}" ] || [ ! -f "${MANIFEST_PATH}" ]; then
    return 0
  fi
  # Supports both `sha256sum` text files and simple JSON manifests containing filename + sha256
  local hit=""
  hit="$(awk -v fn="${filename}" '$2 == fn || $2 ~ ("/" fn "$") {print tolower($1); exit}' "${MANIFEST_PATH}" 2>/dev/null || true)"
  if [ -n "${hit}" ] && [[ "${hit}" =~ ^[0-9a-f]{64}$ ]]; then
    printf '%s' "${hit}"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 - "${MANIFEST_PATH}" "${filename}" <<'PYEOF' 2>/dev/null || true
import json, sys
p, fn = sys.argv[1], sys.argv[2]
try:
    doc = json.load(open(p, encoding="utf-8"))
except Exception:
    sys.exit(0)
def find_sha(obj):
    if isinstance(obj, dict):
        if fn in obj and isinstance(obj[fn], str) and len(obj[fn]) == 64:
            return obj[fn].lower()
        if fn in obj and isinstance(obj[fn], dict):
            s = obj[fn].get("sha256")
            if isinstance(s, str) and len(s) == 64:
                return s.lower()
        for k, v in obj.items():
            if k.endswith(fn) and isinstance(v, str) and len(v) == 64:
                return v.lower()
            r = find_sha(v)
            if r:
                return r
    elif isinstance(obj, list):
        for item in obj:
            r = find_sha(item)
            if r:
                return r
    return ""
res = find_sha(doc)
if res:
    print(res)
PYEOF
  fi
}

resolve_onnx_path() {
  local filename="$1"
  shift
  if [ -f "${MODEL_DIR}/${filename}" ]; then
    printf '%s' "${MODEL_DIR}/${filename}"
    return 0
  fi
  local fallback
  for fallback in "$@"; do
    if [ -f "${fallback}" ]; then
      printf '%s' "${fallback}"
      return 0
    fi
  done
  printf '%s' "${MODEL_DIR}/${filename}"
}

verify_model_sha256() {
  local key="$1"
  local onnx_path="$2"
  local expected_sha="$3"
  local filename
  filename="$(basename "${onnx_path}")"

  if [ ! -f "${onnx_path}" ]; then
    echo "[ERROR] [${key}] Missing ONNX model: ${onnx_path}" >&2
    return 1
  fi

  local manifest_sha
  manifest_sha="$(lookup_manifest_sha256 "${filename}")"
  if [ -n "${manifest_sha}" ]; then
    expected_sha="${manifest_sha}"
  fi

  local actual_sha
  actual_sha="$(sha256_file "${onnx_path}")"

  if [ -z "${expected_sha}" ]; then
    if [ "${PROFILE}" = "extended" ] && [ "${ALLOW_UNPINNED_V53}" -eq 1 ]; then
      echo "[WARN]  [${key}] Unpinned extended model allowed via --allow-unpinned-v53; recorded sha256=${actual_sha}" >&2
      printf '%s' "${actual_sha}"
      return 0
    fi
    echo "[ERROR] [${key}] SHA256_UNPINNED_REJECTED for ${filename} (supply --manifest <file> or --allow-unpinned-v53)" >&2
    return 1
  fi

  if [ "${actual_sha}" != "${expected_sha}" ]; then
    echo "[ERROR] [${key}] SHA256_MISMATCH_REJECTED for ${filename}: expected=${expected_sha}, actual=${actual_sha}" >&2
    return 1
  fi
  echo "[OK]    [${key}] SHA-256 verified: ${actual_sha} (${filename})" >&2
  printf '%s' "${actual_sha}"
}

probe_gpu_vram_mib() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || true
  else
    printf 'UNAVAILABLE'
  fi
}

parse_trtexec_metric() {
  local log_file="$1"
  local pattern="$2"
  awk -v pat="${pattern}" '
    index($0, pat) > 0 {
      for (i = 1; i <= NF; i++) {
        if ($i == "ms" && i > 1) { val = $(i-1); }
      }
    }
    END { if (val != "") print val; else print "null"; }
  ' "${log_file}"
}

TEGRASTATS_LOG="${OUT_DIR}/logs/tegrastats_${BOARD}_${PROFILE}_${PRECISION}.log"
if [ "${DRY_RUN}" -eq 0 ] && command -v tegrastats >/dev/null 2>&1; then
  echo "[INFO]  Starting tegrastats power/thermal sampler -> ${TEGRASTATS_LOG}"
  tegrastats --interval 500 --logfile "${TEGRASTATS_LOG}" >/dev/null 2>&1 &
  TEGRASTATS_PID=$!
else
  echo "[INFO]  tegrastats unavailable or --dry-run active; skipping live Jetson power telemetry capture"
fi

# ---- Model Matrix Selection (`v1.1` vs `extended`) ----
if [ "${PROFILE}" = "extended" ]; then
  D2_FILE="xvs_d2_v53_fp16.onnx"
  D2_SHAPES="x_t:1x6x640x640,h_prev_warped:1x24x160x160"
  D2_SHA=""
  D2_NOTE="extended_640x640_horizon_tile"

  LAND_FILE="xvs_land_v53_fp16.onnx"
  LAND_SHAPES="roi_crop:1x3x384x384"
  LAND_SHA=""
  LAND_NOTE="extended_7output_runway_present_logit"

  THERMAL_FILE="xvs_thermal_v53_fp16.onnx"
  THERMAL_SHAPES="x_eo:1x3x256x320,x_lwir:1x2x128x160,q_eo:1,q_lwir:1"
  THERMAL_SHA=""
  THERMAL_NOTE="extended_1.13MiB_disagreement_vacuity"
else
  D2_FILE="xvs_d2_web_fp16.onnx"
  D2_SHAPES="x_t:1x6x256x256,h_prev_warped:1x24x64x64"
  D2_SHA="${PIN_D2_V11_SHA256}"
  D2_NOTE="v1.1_256x256_convgru"

  LAND_FILE="xvs_land_v52_fp16.onnx"
  LAND_SHAPES="roi_crop:1x3x384x384"
  LAND_SHA="${PIN_LAND_V52_SHA256}"
  LAND_NOTE="v1.1_6output_keyline_sigma"

  THERMAL_FILE="xvs_thermal_v52_fp16.onnx"
  THERMAL_SHAPES="x_eo:1x3x256x320,x_lwir:1x2x128x160,q_eo:1,q_lwir:1"
  THERMAL_SHA="${PIN_THERMAL_V52_SHA256}"
  THERMAL_NOTE="v1.1_1.76MiB_evidential_fpn"
fi

NAV_FILE="xvs_nav_v5_fazb_fp16.onnx"
NAV_SHAPES="img_s1:1x3x224x224"
NAV_SHA="${PIN_NAV_FAZB_SHA256}"
NAV_NOTE="stage1_mnv4_mixvpr"

XFEAT_FILE="xfeat_320x256_fp16.onnx"
XFEAT_SHAPES="img_s2:1x3x256x320"
XFEAT_SHA="${PIN_XFEAT_FP16_SHA256}"
XFEAT_NOTE="stage2a_xfeat_320x256"

LG_FILE="lighterglue_k256_fp16.onnx"
LG_SHAPES="kpts0:1x256x2,kpts1:1x256x2,desc0:1x256x64,desc1:1x256x64"
LG_SHA="${PIN_LIGHTERGLUE_FP16_SHA256}"
LG_NOTE="stage2b_lighterglue_k256"

# Resolve paths (MODEL_DIR primary, repo model directory as fallback)
D2_PATH="$(resolve_onnx_path "${D2_FILE}" \
  "${REPO_ROOT}/apps/livedeck/public/models/${D2_FILE}")"
LAND_PATH="$(resolve_onnx_path "${LAND_FILE}")"
THERMAL_PATH="$(resolve_onnx_path "${THERMAL_FILE}")"
NAV_PATH="$(resolve_onnx_path "${NAV_FILE}")"
XFEAT_PATH="$(resolve_onnx_path "${XFEAT_FILE}")"
LG_PATH="$(resolve_onnx_path "${LG_FILE}")"

MODEL_KEYS=("xvs_d2" "xvs_land" "xvs_thermal" "xvs_nav" "xfeat" "lighterglue")
MODEL_PATHS=("${D2_PATH}" "${LAND_PATH}" "${THERMAL_PATH}" "${NAV_PATH}" "${XFEAT_PATH}" "${LG_PATH}")
MODEL_SHAPES=("${D2_SHAPES}" "${LAND_SHAPES}" "${THERMAL_SHAPES}" "${NAV_SHAPES}" "${XFEAT_SHAPES}" "${LG_SHAPES}")
MODEL_SHAS=("${D2_SHA}" "${LAND_SHA}" "${THERMAL_SHA}" "${NAV_SHA}" "${XFEAT_SHA}" "${LG_SHA}")
MODEL_NOTES=("${D2_NOTE}" "${LAND_NOTE}" "${THERMAL_NOTE}" "${NAV_NOTE}" "${XFEAT_NOTE}" "${LG_NOTE}")

SUMMARY_JSON="${OUT_DIR}/trt_build_benchmark_summary.json"
MODELS_JSON_ENTRIES=()

build_one_model() {
  local key="$1"
  local onnx_path="$2"
  local shapes="$3"
  local expected_sha="$4"
  local note="$5"

  local verified_sha
  verified_sha="$(verify_model_sha256 "${key}" "${onnx_path}" "${expected_sha}")"

  local base
  base="$(basename "${onnx_path}" .onnx)"
  local plan_path="${OUT_DIR}/${base}_${BOARD}_${PRECISION}.plan"
  local log_path="${OUT_DIR}/logs/${base}_${BOARD}_${PRECISION}.trtexec.log"
  local timing_path="${OUT_DIR}/timing/${base}_${BOARD}_${PRECISION}.times.json"

  local cmd=(
    "${TRTEXEC_BIN}"
    "--onnx=${onnx_path}"
    "--saveEngine=${plan_path}"
    "--shapes=${shapes}"
    "--memPoolSize=workspace:${WORKSPACE_MB}M"
    "--fp16"
    "--useCudaGraph"
    "--separateProfileRun"
  )

  if [ "${PRECISION}" = "int8" ]; then
    cmd+=("--int8")
    if [ -n "${CALIB_DIR}" ] && [ -f "${CALIB_DIR}/${key}.calib" ]; then
      cmd+=("--calib=${CALIB_DIR}/${key}.calib")
    fi
  fi

  if [ "${SKIP_BENCHMARK}" -eq 1 ]; then
    cmd+=("--skipInference")
  else
    cmd+=(
      "--warmUp=${WARMUP_MS}"
      "--iterations=${ITERATIONS}"
      "--percentile=50,95,99"
      "--exportTimes=${timing_path}"
    )
  fi

  local vram_before
  vram_before="$(probe_gpu_vram_mib)"

  local p50_ms="null"
  local p95_ms="null"
  local p99_ms="null"
  local plan_bytes=0
  local status="built"

  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "[DRY-RUN] [${key}] ${cmd[*]}"
    status="dry_run_verified"
  else
    echo "[BUILD] [${key}] Compiling ${onnx_path} -> ${plan_path}"
    "${cmd[@]}" 2>&1 | tee "${log_path}"
    if [ ! -s "${plan_path}" ]; then
      echo "[ERROR] [${key}] Expected TensorRT plan was not created: ${plan_path}" >&2
      return 1
    fi
    plan_bytes="$(wc -c < "${plan_path}" | tr -d ' ')"
    if [ "${SKIP_BENCHMARK}" -eq 0 ]; then
      p50_ms="$(parse_trtexec_metric "${log_path}" "percentile(50%")"
      [ "${p50_ms}" = "null" ] && p50_ms="$(parse_trtexec_metric "${log_path}" "median =")"
      p95_ms="$(parse_trtexec_metric "${log_path}" "percentile(95%")"
      p99_ms="$(parse_trtexec_metric "${log_path}" "percentile(99%")"
    fi
    echo "[DONE]  [${key}] plan=${plan_bytes} B | p50=${p50_ms} ms | p95=${p95_ms} ms | p99=${p99_ms} ms"
  fi

  local vram_after
  vram_after="$(probe_gpu_vram_mib)"

  MODELS_JSON_ENTRIES+=(
    "    {\"model\": \"${key}\", \"variant\": \"${note}\", \"onnx\": \"$(basename "${onnx_path}")\", \"sha256\": \"${verified_sha}\", \"shapes\": \"${shapes}\", \"plan\": \"$(basename "${plan_path}")\", \"plan_bytes\": ${plan_bytes}, \"status\": \"${status}\", \"latency_p50_ms\": ${p50_ms}, \"latency_p95_ms\": ${p95_ms}, \"latency_p99_ms\": ${p99_ms}, \"vram_before\": \"${vram_before}\", \"vram_after\": \"${vram_after}\"}"
  )
}

for idx in "${!MODEL_KEYS[@]}"; do
  build_one_model \
    "${MODEL_KEYS[$idx]}" \
    "${MODEL_PATHS[$idx]}" \
    "${MODEL_SHAPES[$idx]}" \
    "${MODEL_SHAS[$idx]}" \
    "${MODEL_NOTES[$idx]}"
done

cleanup

POWER_SUMMARY="UNAVAILABLE"
if [ -s "${TEGRASTATS_LOG}" ]; then
  POWER_SUMMARY="$(awk '
    match($0, /VDD_IN [0-9]+mW/) {
      s = substr($0, RSTART + 7, RLENGTH - 9) + 0;
      if (s > max_mw) max_mw = s;
      sum_mw += s; cnt++;
    }
    END {
      if (cnt > 0) printf "avg_vdd_in_mw=%.1f,peak_vdd_in_mw=%d,samples=%d", sum_mw / cnt, max_mw, cnt;
      else print "PARSED_0_VDD_IN_SAMPLES";
    }
  ' "${TEGRASTATS_LOG}")"
fi

{
  echo "{"
  echo "  \"schema\": \"openxvs.jetson_trt_build_summary/1\","
  echo "  \"non_flight_sim_only\": true,"
  echo "  \"actuator_authority\": false,"
  echo "  \"watermark\": \"SIMULATION & BENCHMARK EVALUATION ONLY -- ZERO ACTUATION -- NOT FOR FLIGHT\","
  echo "  \"eula_boundary\": \"NVIDIA TensorRT SLA Sec. 1.2 & Sec. 2 (local non-redistributable eval cache)\","
  echo "  \"profile\": \"${PROFILE}\","
  echo "  \"board\": \"${BOARD}\","
  echo "  \"precision\": \"${PRECISION}\","
  echo "  \"workspace_mb\": ${WORKSPACE_MB},"
  echo "  \"dry_run\": $([ "${DRY_RUN}" -eq 1 ] && echo true || echo false),"
  echo "  \"tegrastats_power_envelope\": \"${POWER_SUMMARY}\","
  echo "  \"models\": ["
  for i in "${!MODELS_JSON_ENTRIES[@]}"; do
    if [ "$i" -lt $((${#MODELS_JSON_ENTRIES[@]} - 1)) ]; then
      echo "${MODELS_JSON_ENTRIES[$i]},"
    else
      echo "${MODELS_JSON_ENTRIES[$i]}"
    fi
  done
  echo "  ]"
  echo "}"
} > "${SUMMARY_JSON}"

echo "[OPENXVS-TRT] Complete (${#MODEL_KEYS[@]}/6 models). Summary -> ${SUMMARY_JSON}"
