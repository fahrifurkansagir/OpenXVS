import { copyFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { defineConfig, type Plugin } from "vite";

const COOP_COEP_HEADERS = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

const ORT_RUNTIME_FILES = [
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
  "ort-wasm-simd-threaded.jsep.mjs",
  "ort-wasm-simd-threaded.jsep.wasm",
  "ort-wasm-simd-threaded.jspi.mjs",
  "ort-wasm-simd-threaded.jspi.wasm",
  "ort-wasm-simd-threaded.asyncify.mjs",
  "ort-wasm-simd-threaded.asyncify.wasm",
];

/**
 * Serves & copies pinned `onnxruntime-web@1.31.1` WASM/MJS binaries into `dist/ort/`
 * so Stage-1 AI loads strictly from same-origin (ZERO external CDN requests under allowlist v1.4)
 * while keeping Stage-0 bundle <= 700 KB Brotli.
 */
function stage1OrtRuntimePlugin(): Plugin {
  const ortDistDir = resolve(__dirname, "node_modules/onnxruntime-web/dist");
  let outDir = resolve(__dirname, "dist");

  return {
    name: "openxvs-stage1-ort-runtime",
    configResolved(cfg) {
      outDir = resolve(cfg.root, cfg.build.outDir);
    },
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = req.url?.split("?")[0] ?? "";
        if (!url.startsWith("/ort/")) {
          next();
          return;
        }
        const filename = url.slice("/ort/".length);
        if (!ORT_RUNTIME_FILES.includes(filename)) {
          next();
          return;
        }
        const filePath = join(ortDistDir, filename);
        if (!existsSync(filePath)) {
          next();
          return;
        }
        res.setHeader(
          "Content-Type",
          filename.endsWith(".wasm") ? "application/wasm" : "application/javascript",
        );
        res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
        res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
        res.end(readFileSync(filePath));
      });
    },
    closeBundle() {
      const targetDir = join(outDir, "ort");
      mkdirSync(targetDir, { recursive: true });
      for (const f of ORT_RUNTIME_FILES) {
        const src = join(ortDistDir, f);
        if (existsSync(src)) {
          copyFileSync(src, join(targetDir, f));
        }
      }
    },
  };
}

export default defineConfig({
  base: "./",
  plugins: [stage1OrtRuntimePlugin()],
  server: {
    host: "127.0.0.1",
    port: 4173,
    headers: COOP_COEP_HEADERS,
  },
  preview: {
    host: "127.0.0.1",
    port: 4173,
    headers: COOP_COEP_HEADERS,
  },
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  build: {
    target: "es2022",
    sourcemap: false,
    assetsInlineLimit: 0,
  },
});
