import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  server: {
    host: "127.0.0.1",
    port: 4174,
    strictPort: false,
  },
  preview: {
    host: "127.0.0.1",
    port: 4174,
    strictPort: false,
  },
  build: {
    target: "es2022",
    sourcemap: false,
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        manualChunks: {
          three: ["three"],
        },
      },
    },
  },
});
