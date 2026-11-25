# OpenXVS Edge ARM64 OCI Kit (`$0` Hardware-Free Build & OEM Recipes)

> **MANDATORY NON-FLIGHT BOUNDARY (`EU PLD Directive (EU) 2024/2853 Art. 4(7)`):**
> `SIMULATION & BENCHMARK EVALUATION ONLY — ZERO ACTUATION — NOT FOR FLIGHT`
>
> All containers compile `openxvs-edge-infer` with `non_flight_sim_only` enabled (`ACTUATOR_AUTHORITY = false`), zero Python, zero TLS/AES confidentiality crypto crates (`5D002` private-first guardrail), and zero GPL/AGPL dependencies.

## Target Matrix

| Recipe | Target Hardware | Runtime Base Image | ABI / EULA Boundary | Silicon Latency |
| :--- | :--- | :--- | :--- | :--- |
| [`rpi5.Dockerfile`](./rpi5.Dockerfile) | Raspberry Pi 5 (Cortex-A76, `linux/arm64`) | `cgr.dev/chainguard/wolfi-base` (default) or `gcr.io/distroless/cc-debian12:nonroot` (`--target runtime-distroless`) | **glibc >= 2.28 ONLY** — Alpine `musl` is strictly banned (official ORT `linux-aarch64` requires glibc >= 2.28). | `UNVERIFIED` until physical RPi 5 benchmark |
| [`orin_nx_jp62.Dockerfile`](./orin_nx_jp62.Dockerfile) | Jetson Orin NX (JetPack 6.2, L4T R36.4.3/36.5) | `nvcr.io/nvidia/l4t-tensorrt:r10.3.0-runtime` (CUDA 12.6, TensorRT 10.3) | **OEM-side recipe only.** Subject to **NVIDIA TensorRT SLA Sec. 1.2 & Sec. 2 (Critical Applications)** — non-flight eval only; `.engine` plans built locally on first boot, never redistributed. | `UNVERIFIED` until physical Orin NX benchmark |
| [`orin_nx_jp72.Dockerfile`](./orin_nx_jp72.Dockerfile) | Jetson Orin NX / Thor (JetPack 7.2/7.2.1 SBSA) | `nvcr.io/nvidia/cuda:13.0.0-runtime-ubuntu24.04` + TensorRT runtime (`tensorrt-libs`, `libnvinfer-bin`) | NVIDIA discontinued legacy `l4t-jetpack`/`l4t-base` for JP 7.x. Subject to **NVIDIA TensorRT SLA Sec. 1.2 & Sec. 2 (Critical Applications)**. | `UNVERIFIED` until physical JP 7.2 silicon benchmark |

## `$0` Hardware-Free Native ARM64 CI (`.github/workflows/xvs_edge_arm64_oci.yml`)

- Runs natively on GitHub-hosted **`ubuntu-24.04-arm`** runners (private repos receive 2,400 native ARM64 min/mo within the free `$12/mo` runner allowance — zero QEMU slowdown).
- Generates an **SPDX 2.3 JSON SBOM** with **`syft`** and enforces zero GPL/AGPL/LGPL-static contamination.
- Signs private evaluation digests strictly with **key-based Private-First Cosign (`--tlog-upload=false`)** to guarantee **zero Rekor transparency-log metadata leakage** prior to the May 2027 NeurIPS E&D double-blind submission window:
  ```bash
  cosign sign --key env://COSIGN_PRIVATE_KEY --tlog-upload=false --yes "${IMAGE_REF}@${DIGEST}"
  ```
- Only publish clean-room genesis exports (zero Stratum-A history, zero un-cleared `XVS-Nav` source) to private CI repositories.

## Local Build & Evaluation Commands

```bash
# Raspberry Pi 5 (Chainguard Wolfi glibc runtime)
docker buildx build --platform linux/arm64 -f docker/rpi5.Dockerfile --target runtime-wolfi -t openxvs-edge-rpi5:eval .

# Raspberry Pi 5 (Distroless Debian 12 glibc runtime, zero shell)
docker buildx build --platform linux/arm64 -f docker/rpi5.Dockerfile --target runtime-distroless -t openxvs-edge-rpi5:distroless .

# Jetson Orin NX JetPack 6.2 (OEM local build; requires NGC access)
docker buildx build --platform linux/arm64 -f docker/orin_nx_jp62.Dockerfile -t openxvs-edge-orin-jp62:eval .

# Jetson Orin NX / Thor JetPack 7.2 SBSA (OEM local build)
docker buildx build --platform linux/arm64 -f docker/orin_nx_jp72.Dockerfile -t openxvs-edge-orin-jp72:eval .

# Run offline benchmark self-check with mounted ONNX weights (NON_FLIGHT_SIM_ONLY=1)
docker run --rm --read-only \
  -v "$(pwd)/models:/opt/openxvs/models:ro" \
  openxvs-edge-rpi5:eval
```
