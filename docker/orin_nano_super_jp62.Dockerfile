# syntax=docker/dockerfile:1.7
# ==============================================================================
# OpenXVS Edge ARM64 OEM Recipe — NVIDIA Jetson Orin Nano Super 8GB (67 TOPS)
# JetPack 6.2 / L4T R36.4.3+ (`MAXN_SUPER` 25W, 67 Sparse INT8 TOPS, 8 GB RAM)
# Base: nvcr.io/nvidia/l4t-tensorrt:r10.3.0-runtime (CUDA 12.6, TensorRT 10.3)
# ==============================================================================
# MANDATORY NVIDIA TENSORRT SLA & NON-FLIGHT OEM EVALUATION EULA NOTICE:
#   1. NVIDIA TensorRT Software License Agreement (SLA) Section 1.2 & Section 2
#      explicitly prohibit:
#        (a) OSS license contamination / unauthorized redistribution of TensorRT
#            runtime libraries or compiled `.engine` plan files; and
#        (b) Use in "Critical Applications" (including life-critical avionics,
#            aircraft flight control, or certified aviation navigation systems).
#   2. Accordingly, OpenXVS distributes ONLY this OEM-side build recipe and
#      strictly restricts execution to:
#        SIMULATION & BENCHMARK EVALUATION ONLY — ZERO ACTUATION — NOT FOR FLIGHT
#      under EU PLD Directive (EU) 2024/2853 Art. 4(7) and NON_FLIGHT_SIM_ONLY=1.
#   3. Any future safety-critical flight RTA (`xvs-rta-core`, `xvs-active-range`,
#      `xvs-odd-vlm`, `xvs-os-hal`, `xvs-multiship-coord`) remains isolated
#      bare-metal `#![no_std]` Rust with zero TensorRT/CUDA dependency.
#   4. 8 GB Unified LPDDR5 Envelope: defaults cap TensorRT workspace at 1024 MiB,
#      glibc arenas (`MALLOC_ARENA_MAX=2`), and app budget at 6144 MiB to leave
#      >= 2 GiB headroom for Linux + L4T unified GPU allocations. Zero Python.
#   5. Hardware latency on physical Jetson Orin Nano Super 8GB remains UNVERIFIED
#      until physical silicon bench test.
# ==============================================================================

ARG RUST_BUILDER_IMAGE=rust:1.92-bookworm
ARG ORT_VERSION=1.24.4
ARG ORT_AARCH64_URL=https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-aarch64-${ORT_VERSION}.tgz
ARG JETPACK62_TRT_IMAGE=nvcr.io/nvidia/l4t-tensorrt:r10.3.0-runtime

# ------------------------------------------------------------------------------
# Stage 1: Native ARM64 Pure-Rust `ort` (`load-dynamic`) Builder (Zero Python)
# ------------------------------------------------------------------------------
FROM --platform=linux/arm64 ${RUST_BUILDER_IMAGE} AS builder

ARG ORT_VERSION
ARG ORT_AARCH64_URL

WORKDIR /build

COPY xvs-core/ /build/xvs-core/
COPY xvs-active-range/ /build/xvs-active-range/
COPY xvs-multiship-coord/ /build/xvs-multiship-coord/
COPY docker/edge-infer/ /build/docker/edge-infer/

ENV CARGO_INCREMENTAL=0 \
    NON_FLIGHT_SIM_ONLY=1 \
    OPENXVS_NON_FLIGHT_SIM_ONLY=1 \
    RUSTFLAGS="--remap-path-prefix=/build=. --remap-path-prefix=/usr/local/cargo=/cargo --remap-path-prefix=/root=/home/builder"

RUN cargo build \
      --manifest-path /build/docker/edge-infer/Cargo.toml \
      --release \
      --features non_flight_sim_only && \
    strip /build/docker/edge-infer/target/release/openxvs-edge-infer

# Stage MIT-licensed official ONNX Runtime Linux aarch64 dynamic library (glibc >= 2.28)
RUN mkdir -p /opt/openxvs/lib && \
    curl -fsSL "${ORT_AARCH64_URL}" -o /tmp/ort.tgz && \
    tar -xzf /tmp/ort.tgz -C /tmp && \
    cp -a /tmp/onnxruntime-linux-aarch64-${ORT_VERSION}/lib/libonnxruntime.so* /opt/openxvs/lib/ && \
    rm -rf /tmp/ort.tgz /tmp/onnxruntime-linux-aarch64-${ORT_VERSION}

# ------------------------------------------------------------------------------
# Stage 2: Jetson Orin Nano Super 8GB (67 TOPS) JetPack 6.2 Runtime (Zero Python)
# ------------------------------------------------------------------------------
FROM --platform=linux/arm64 ${JETPACK62_TRT_IMAGE} AS runtime

LABEL org.opencontainers.image.title="OpenXVS Edge OEM Evaluator (Jetson Orin Nano Super 8GB / 67 TOPS / JP 6.2)" \
      org.opencontainers.image.description="OEM-side non-flight evaluation recipe for NVIDIA Jetson Orin Nano Super 8GB (67 TOPS, CUDA 12.6, TensorRT 10.3, zero Python, MIT ORT dynamic lib). Subject to NVIDIA TensorRT SLA Sec. 1.2 & Sec. 2." \
      org.opencontainers.image.licenses="Apache-2.0 OR MIT" \
      org.openxvs.target_board="jetson-orin-nano-super-8gb" \
      org.openxvs.compute_tops="67" \
      org.openxvs.ram_cap_gb="8" \
      org.openxvs.jetpack="6.2-r36.4.3-cuda12.6-trt10.3-maxn-super" \
      org.openxvs.non_flight_sim_only="true" \
      org.openxvs.actuator_authority="false" \
      org.openxvs.nvidia_eula_boundary="NVIDIA TensorRT SLA Sec. 1.2 & 2 Critical Applications — Non-Flight Bench/Sim Evaluation Only" \
      org.openxvs.watermark="SIMULATION & BENCHMARK EVALUATION ONLY — ZERO ACTUATION — NOT FOR FLIGHT"

RUN mkdir -p /opt/openxvs/bin /opt/openxvs/lib /opt/openxvs/models /var/cache/openxvs/engines /etc/openxvs

COPY --from=builder /build/docker/edge-infer/target/release/openxvs-edge-infer /opt/openxvs/bin/openxvs-edge-infer
COPY --from=builder /opt/openxvs/lib/ /opt/openxvs/lib/

RUN cat <<'EOF' > /etc/openxvs/NVIDIA_TENSORRT_EULA_NOTICE.txt
================================================================================
OPENXVS JETSON ORIN NANO SUPER 8GB (67 TOPS / JP 6.2 / TRT 10.3) — SAFETY LOCK
================================================================================
WATERMARK : SIMULATION & BENCHMARK EVALUATION ONLY — ZERO ACTUATION — NOT FOR FLIGHT
LOCK      : NON_FLIGHT_SIM_ONLY=1 (ACTUATOR_AUTHORITY=false)
PROFILE   : Jetson Orin Nano Super 8GB (67 TOPS MAXN_SUPER, 8 GB RAM cap profile)
EULA      : Subject to NVIDIA TensorRT SLA Sections 1.2 & 2 (Critical
            Applications restriction: NOT licensed for flight-critical avionics,
            certified navigation, or closed-loop aircraft actuation).
ENGINES   : Compiled TensorRT `.engine` plan caches are built locally on first
            boot in /var/cache/openxvs/engines (1024 MiB workspace cap) and
            MUST NOT be redistributed.
================================================================================
EOF

RUN cat <<'EOF' > /usr/local/bin/openxvs-orin-nano-super-entrypoint.sh
#!/usr/bin/env sh
set -eu
cat /etc/openxvs/NVIDIA_TENSORRT_EULA_NOTICE.txt >&2
if [ "${OPENXVS_BUILD_TRT_ENGINES:-0}" = "1" ] && command -v trtexec >/dev/null 2>&1; then
  ws_mb="${OPENXVS_TRT_WORKSPACE_MB:-1024}"
  for onnx in /opt/openxvs/models/*.onnx; do
    [ -f "$onnx" ] || continue
    base="$(basename "$onnx" .onnx)"
    plan="/var/cache/openxvs/engines/${base}_orin_nano_super.engine"
    if [ ! -f "$plan" ]; then
      echo "[OPENXVS-ORIN-NANO-SUPER-8GB] Building local non-redistributable TRT 10.3 plan (ws=${ws_mb}MiB): ${plan}" >&2
      trtexec --onnx="$onnx" --saveEngine="$plan" --fp16 --stronglyTyped --memPoolSize="workspace:${ws_mb}M" || true
    fi
  done
fi
exec /opt/openxvs/bin/openxvs-edge-infer "$@"
EOF
RUN chmod +x /usr/local/bin/openxvs-orin-nano-super-entrypoint.sh

ENV NON_FLIGHT_SIM_ONLY=1 \
    OPENXVS_NON_FLIGHT_SIM_ONLY=1 \
    OPENXVS_TARGET_BOARD=jetson-orin-nano-super-8gb \
    OPENXVS_COMPUTE_TOPS=67 \
    OPENXVS_RAM_CAP_GB=8 \
    OPENXVS_MEMORY_CAP_MB=6144 \
    OPENXVS_TRT_WORKSPACE_MB=1024 \
    ORT_INTRA_OP_NUM_THREADS=4 \
    MALLOC_ARENA_MAX=2 \
    ORT_DYLIB_PATH=/opt/openxvs/lib/libonnxruntime.so \
    LD_LIBRARY_PATH=/opt/openxvs/lib:${LD_LIBRARY_PATH} \
    OPENXVS_MODEL_DIR=/opt/openxvs/models \
    OPENXVS_TRT_ENGINE_CACHE=/var/cache/openxvs/engines

ENTRYPOINT ["/usr/local/bin/openxvs-orin-nano-super-entrypoint.sh"]
