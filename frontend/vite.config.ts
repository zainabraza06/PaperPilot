import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

const API_TARGET =
  loadEnv('development', process.cwd(), 'VITE_')['VITE_API_PROXY'] ??
  'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  resolve: {
    // import.meta.dirname rather than __dirname: this config is ESM, and
    // Vite's native config loader does not provide the CommonJS globals.
    alias: { '@': new URL('./src', import.meta.url).pathname },
  },
  server: {
    port: 5173,
    // The backend already allows this origin via CORS, but proxying keeps
    // the app same-origin in development, so nothing depends on CORS being
    // configured correctly to work locally.
    //
    // The target is overridable because port 8000 is popular: if something
    // else already owns it, `VITE_API_PROXY=http://127.0.0.1:8010 npm run
    // dev` is a one-line fix rather than an edit to a tracked file.
    proxy: {
      '/api': { target: API_TARGET, changeOrigin: true },
      '/health': { target: API_TARGET, changeOrigin: true },
    },
  },
})
