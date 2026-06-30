import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Forward /health and /api/* to the local FastAPI server so the browser
      // sees same-origin requests (no CORS headers needed in dev).
      "/health": "http://localhost:8080",
      "/api": "http://localhost:8080",
    },
  },
  build: { outDir: "dist", sourcemap: true, chunkSizeWarningLimit: 600 },
});
