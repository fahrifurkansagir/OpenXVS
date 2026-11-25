// Prevents additional console window on Windows in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
#![forbid(unsafe_code)]

fn main() {
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("error while running OpenXVS v3.0 Observatory");
}
