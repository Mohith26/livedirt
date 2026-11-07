import { defineConfig } from 'vite';
export default defineConfig({
  base: '/',
  server: { host: '127.0.0.1', port: 5199, strictPort: true, watch: { ignored: ['**/bench/**', '**/dist/**'] } },
  build: { target: 'es2022', sourcemap: true, chunkSizeWarningLimit: 4000 },
  worker: { format: 'es' },
  test: { environment: 'node' },
} as any);
