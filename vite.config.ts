import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: process.env.OVB_BASE ?? '/open-vision-builder/',
  worker: { format: 'es' },
  test: { include: ['tests/**/*.test.ts'], testTimeout: 30_000 },
  build: { target: 'es2022', license: { fileName: 'THIRD_PARTY_LICENSES.md' } },
});
