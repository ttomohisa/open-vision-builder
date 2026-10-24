import { defineConfig } from '@playwright/test';

const base = process.env.OVB_BASE ?? '/open-vision-builder/';
export default defineConfig({
  testDir: './tests',
  testMatch: ['**/browser.spec.ts', '**/decode.browser.spec.ts'],
  fullyParallel: false,
  workers: 1,
  timeout: 180_000,
  expect: { timeout: 10_000 },
  use: { baseURL: `http://127.0.0.1:4173${base}`, headless: true, acceptDownloads: true, trace: 'retain-on-failure' },
  webServer: { command: 'npm run preview -- --port 4173', url: `http://127.0.0.1:4173${base}`, reuseExistingServer: false, timeout: 30_000 },
});
