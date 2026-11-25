# OpenXVS v2.0 — Native WSL2 Ubuntu Makefile for `xvs-rt` (g++ 15.2.0 + Cargo `--locked` staticlib)
# Uses relative parent paths (`..`) to stay 100% portable and space-safe under any WSL2 checkout path.
# Builds strictly under /tmp/openxvs_rt_build and /tmp/openxvs_wsl_target so zero build artifacts
# pollute the repository working tree.

CXX ?= g++
CARGO ?= cargo

XVS_CORE_DIR := ../xvs-core
DAIDALUS_DIR := ../third_party/daidalus_boundary
OUT_M3_DIR := ../output/xvs/v2_m3

WSL_RUST_TARGET ?= /tmp/openxvs_wsl_target
BUILD_DIR ?= /tmp/openxvs_rt_build
LIB_XVS_CORE := $(WSL_RUST_TARGET)/release/libxvs_core.a

COMMON_FLAGS := -std=c++20 -fno-exceptions -fno-rtti \
	-Wall -Wextra -Werror -Wconversion -Wsign-conversion -Wshadow -Wdouble-promotion \
	-Iinclude -I$(XVS_CORE_DIR)/crates/xvs-ffi/include -I$(DAIDALUS_DIR)

WRAP_LDFLAGS := -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=realloc -Wl,--wrap=free
LDLIBS := $(LIB_XVS_CORE) -lm -ldl -lpthread

CORE_SRCS := \
	src/mh_vit.cpp \
	src/ecbf_qp.cpp \
	src/rt_pipeline.cpp \
	$(DAIDALUS_DIR)/xvs_daidalus_bridge.cpp

AUDIT_SRC := src/malloc_audit.cpp
BENCH_SRC := src/main_bench.cpp
TEST_SRC := tests/test_xvs_rt.cpp

BIN_TEST := $(BUILD_DIR)/test_xvs_rt
BIN_BENCH_O0 := $(BUILD_DIR)/xvs_rt_bench_O0
BIN_BENCH_O2 := $(BUILD_DIR)/xvs_rt_bench_O2

.PHONY: all rust build test bench run-all clean

all: build

rust:
	cd $(XVS_CORE_DIR) && CARGO_TARGET_DIR=$(WSL_RUST_TARGET) $(CARGO) build -p xvs-ffi --release --locked

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_TEST): rust $(BUILD_DIR) $(CORE_SRCS) $(TEST_SRC)
	$(CXX) -O2 $(COMMON_FLAGS) $(CORE_SRCS) $(TEST_SRC) -o $(BIN_TEST) $(LDLIBS)

$(BIN_BENCH_O0): rust $(BUILD_DIR) $(CORE_SRCS) $(AUDIT_SRC) $(BENCH_SRC)
	$(CXX) -O0 -g $(COMMON_FLAGS) $(CORE_SRCS) $(AUDIT_SRC) $(BENCH_SRC) -o $(BIN_BENCH_O0) $(WRAP_LDFLAGS) $(LDLIBS)

$(BIN_BENCH_O2): rust $(BUILD_DIR) $(CORE_SRCS) $(AUDIT_SRC) $(BENCH_SRC)
	$(CXX) -O2 $(COMMON_FLAGS) $(CORE_SRCS) $(AUDIT_SRC) $(BENCH_SRC) -o $(BIN_BENCH_O2) $(WRAP_LDFLAGS) $(LDLIBS)

build: $(BIN_TEST) $(BIN_BENCH_O0) $(BIN_BENCH_O2)

test: $(BIN_TEST)
	$(BIN_TEST)

bench: $(BIN_BENCH_O0) $(BIN_BENCH_O2)
	mkdir -p $(OUT_M3_DIR)
	$(BIN_BENCH_O0) --mode audit-only --opt-label O0 --out-audit $(BUILD_DIR)/audit_O0.json
	$(BIN_BENCH_O2) --mode full --opt-label O2 \
		--o0-audit-in $(BUILD_DIR)/audit_O0.json \
		--out-v1 $(OUT_M3_DIR)/v1_xvs_rt_evidence.json \
		--out-v2 $(OUT_M3_DIR)/v2_meshdaa_500enc_evidence.json

run-all: test bench

clean:
	rm -rf $(BUILD_DIR)
