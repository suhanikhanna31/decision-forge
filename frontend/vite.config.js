import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev-time proxy so the browser only ever talks to one origin; in
// production each service is reached via its own VITE_*_URL env var
// (see src/api/client.js) instead of this proxy.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/auth-api": {
        target: "http://localhost:8001",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/auth-api/, "/api"),
      },
      "/ml-api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/ml-api/, ""),
      },
      "/audit-api": {
        target: "http://localhost:5000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/audit-api/, "/api"),
      },
    },
  },
});
