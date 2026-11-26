# syntax=docker/dockerfile:1.7
# ==============================================================================
# OpenXVS Edge ARM64 OCI Recipe — Raspberry Pi 5 (Cortex-A76, linux/arm64)
# ==============================================================================
# NON-FLIGHT EVALUATION NOTICE (EU PLD Directive (EU) 2024/2853 Art. 4(7)):
#   SIMULATION & BENCHMARK EVALUATION ONLY — ZERO ACTUATION — NOT FOR FLIGHT.
#   Compiled with `non_flight_sim_only` default feature (`ACTUATOR_AUTHORITY=false`)
#   and `NON_FLIGHT_SIM_ONLY=1` runtime lock.
#
# CRITICAL ABI LOCK:
#   - Official ONNX Runtime `linux-aarch64` requires glibc >= 2.28.
#   - Alpine Linux (musl) is STRICTLY PROHIBITED for this target.
#   - Runtime stages:
#       * `runtime-wolfi` (default): `cgr.dev/chainguard/wolfi-base` (glibc, ~9.4 MB, 0-3 CVEs)
#       * `runtime-distroless`     : `gcr.io/distroless/cc-debian12:nonroot` (glibc 2.36, no shell)
#   - Hardware latency on physical Raspberry Pi 5 remains UNVERIFIED until silicon test.
# ==============================================================================

ARG RUST_BUILDER_IMAGE=rust:1.92-bookworm
ARG ORT_VERSION=1.24.4
ARG ORT_AARCH64_URL=https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-aarch64-${ORT_VERSION}.tgz
ARG WOLFI_BASE_IMAGE=cgr.dev/chainguard/wolfi-base:latest
ARG DISTROLESS_CC_IMAGE=gcr.io/distroless/cc-debian12:nonroot

# ------------------------------------------------------------------------------
# Stage 1: Native ARM64 Pure-Rust `ort` (`load-dynamic`) Builder + ORT Staging
# ------------------------------------------------------------------------------
FROM --platform=linux/arm64 ${RUST_BUILDER_IMAGE} AS builder

ARG ORT_VERSION
ARG ORT_AARCH64_URL

WORKDIR /build

# Copy only read-only formal crates and headless edge-infer crate (zero Stratum-A, zero Python)
COPY xvs-core/ /build/xvs-core/
COPY xvs-active-range/ /build/xvs-active-range/
COPY xvs-multiship-coord/ /build/xvs-multiship-coord/
COPY docker/edge-infer/ /build/docker/edge-infer/

# REL1 path-leak prevention + compile-time NON_FLIGHT_SIM_ONLY=1
ENV CARGO_INCREMENTAL=0 \
    NON_FLIGHT_SIM_ONLY=1 \
    OPENXVS_NON_FLIGHT_SIM_ONLY=1 \
    RUSTFLAGS="--remap-path-prefix=/build=. --remap-path-prefix=/usr/local/cargo=/cargo --remap-path-prefix=/root=/home/builder"

RUN cargo build \
      --manifest-path /build/docker/edge-infer/Cargo.toml \
      --release \
      --features non_flight_sim_only && \
    strip /build/docker/edge-infer/target/release/openxvs-edge-infer

# Stage MIT-licensed official ONNX Runtime Linux aarch64 shared library (glibc >= 2.28)
RUN mkdir -p /opt/openxvs/lib /opt/openxvs/models && \
    if curl -fsSL "${ORT_AARCH64_URL}" -o /tmp/ort.tgz; then \
      tar -xzf /tmp/ort.tgz -C /tmp && \
      cp -a /tmp/onnxruntime-linux-aarch64-${ORT_VERSION}/lib/libonnxruntime.so* /opt/openxvs/lib/ && \
      rm -rf /tmp/ort.tgz /tmp/onnxruntime-linux-aarch64-${ORT_VERSION}; \
    else \
      apt-get update && apt-get install -y --no-install-recommends unzip && rm -rf /var/lib/apt/lists/* && \
      curl -fsSL "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/${ORT_VERSION}" -o /tmp/ort.nupkg && \
      unzip -j /tmp/ort.nupkg "runtimes/linux-arm64/native/libonnxruntime*.so" -d /opt/openxvs/lib/ && \
      chmod 755 /opt/openxvs/lib/libonnxruntime*.so && \
      rm -f /tmp/ort.nupkg; \
    fi

# ------------------------------------------------------------------------------
# Stage 2A: Minimal Distroless glibc Runtime (optional `--target runtime-distroless`)
# ------------------------------------------------------------------------------
FROM --platform=linux/arm64 ${DISTROLESS_CC_IMAGE} AS runtime-distroless

LABEL org.opencontainers.image.title="OpenXVS Edge Evaluator (RPi 5 ARM64 Distroless)" \
      org.opencontainers.image.description="Pure-Rust ort load-dynamic civil aviation benchmark evaluator (NON_FLIGHT_SIM_ONLY=1)" \
      org.opencontainers.image.licenses="Apache-2.0 OR MIT" \
      org.openxvs.non_flight_sim_only="true" \
      org.openxvs.actuator_authority="false" \
      org.openxvs.watermark="SIMULATION & BENCHMARK EVALUATION ONLY — ZERO ACTUATION — NOT FOR FLIGHT"

COPY --from=builder /build/docker/edge-infer/target/release/openxvs-edge-infer /opt/openxvs/bin/openxvs-edge-infer
COPY --from=builder /opt/openxvs/lib/ /opt/openxvs/lib/

ENV NON_FLIGHT_SIM_ONLY=1 \
    OPENXVS_NON_FLIGHT_SIM_ONLY=1 \
    ORT_DYLIB_PATH=/opt/openxvs/lib/libonnxruntime.so \
    LD_LIBRARY_PATH=/opt/openxvs/lib \
    OPENXVS_MODEL_DIR=/opt/openxvs/models

ENTRYPOINT ["/opt/openxvs/bin/openxvs-edge-infer"]

# ------------------------------------------------------------------------------
# Stage 2B: Default Chainguard Wolfi glibc Runtime (0-3 CVEs, NO Alpine musl)
# ------------------------------------------------------------------------------
FROM --platform=linux/arm64 ${WOLFI_BASE_IMAGE} AS runtime-wolfi

LABEL org.opencontainers.image.title="OpenXVS Edge Evaluator (RPi 5 ARM64 Wolfi)" \
      org.opencontainers.image.description="Pure-Rust ort load-dynamic civil aviation benchmark evaluator (NON_FLIGHT_SIM_ONLY=1, glibc >= 2.28)" \
      org.opencontainers.image.licenses="Apache-2.0 OR MIT" \
      org.openxvs.non_flight_sim_only="true" \
      org.openxvs.actuator_authority="false" \
      org.openxvs.abi="glibc>=2.28-no-musl" \
      org.openxvs.watermark="SIMULATION & BENCHMARK EVALUATION ONLY — ZERO ACTUATION — NOT FOR FLIGHT"

RUN apk add --no-cache libstdc++ libgcc && \
    adduser -D -u 10001 openxvs && \
    mkdir -p /opt/openxvs/bin /opt/openxvs/lib /opt/openxvs/models && \
    chown -R openxvs:openxvs /opt/openxvs

COPY --from=builder /build/docker/edge-infer/target/release/openxvs-edge-infer /opt/openxvs/bin/openxvs-edge-infer
COPY --from=builder /opt/openxvs/lib/ /opt/openxvs/lib/

USER openxvs:openxvs

ENV NON_FLIGHT_SIM_ONLY=1 \
    OPENXVS_NON_FLIGHT_SIM_ONLY=1 \
    ORT_DYLIB_PATH=/opt/openxvs/lib/libonnxruntime.so \
    LD_LIBRARY_PATH=/opt/openxvs/lib \
    OPENXVS_MODEL_DIR=/opt/openxvs/models

ENTRYPOINT ["/opt/openxvs/bin/openxvs-edge-infer"]
