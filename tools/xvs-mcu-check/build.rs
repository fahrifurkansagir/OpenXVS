// Copyright 2026 OpenXVS Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Stage `memory.x` into Cargo `OUT_DIR` so `cortex-m-rt`'s `link.x` resolves
//! the ARM MPS2-AN500 (Cortex-M7) memory map for `thumbv7em-none-eabi` binaries.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let target = env::var("TARGET").unwrap_or_default();
    if target.starts_with("thumbv") {
        let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR not set"));
        let memory_x = include_bytes!("memory.x");
        fs::write(out.join("memory.x"), memory_x).expect("failed to write memory.x to OUT_DIR");
        println!("cargo:rustc-link-search={}", out.display());
    }
    println!("cargo:rerun-if-changed=memory.x");
    println!("cargo:rerun-if-changed=build.rs");
}
