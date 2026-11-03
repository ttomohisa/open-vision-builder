import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { strFromU8, strToU8, unzipSync, zipSync } from 'fflate';

test('classification project backup restores bytes, classes and settings without server writes', async ({ page, context }) => {
  const unexpected: string[] = [];
  const pageErrors: string[] = [];
  context.on('request', request => {
    const url = new URL(request.url());
    if (url.protocol.startsWith('http') && url.origin !== 'http://127.0.0.1:4173') unexpected.push(request.url());
    if (request.method() !== 'GET') unexpected.push(request.method());
  });
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('./');
  await page.getByRole('button', { name: 'サンプルで試す' }).click();
  await expect(page.getByText('12 / 64 枚')).toBeVisible();
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'プロジェクトをバックアップ' }).click();
  const downloaded = await downloadPromise;
  expect(downloaded.suggestedFilename()).toBe('open-vision-project.zip');
  const bytes = new Uint8Array(await readFile((await downloaded.path())!));
  const archive = unzipSync(bytes);
  const manifest = JSON.parse(strFromU8(archive['manifest.json']));
  expect(manifest.kind).toBe('open-vision-classification-project');
  expect(manifest.images).toHaveLength(12);
  expect(manifest.labels).toEqual(['暖色', '寒色']);
  await page.getByLabel('クラス1の名前').fill('別のラベル');
  await expect(page.getByLabel('クラス1の名前')).toHaveValue('別のラベル');
  page.once('dialog', dialog => dialog.accept());
  await page.getByLabel('プロジェクトZIPを選択').setInputFiles({ name: 'project.zip', mimeType: 'application/zip', buffer: Buffer.from(bytes) });
  await expect(page.getByLabel('クラス1の名前')).toHaveValue('暖色');
  await expect(page.getByText('12 / 64 枚')).toBeVisible();
  await expect(page.getByText('プロジェクトを復元しました。', { exact: false })).toBeVisible();
  expect(unexpected).toEqual([]);
  expect(pageErrors).toEqual([]);
});

test('invalid backup cannot replace current project or erase dataset', async ({ page }) => {
  await page.goto('./');
  await page.getByRole('button', { name: 'サンプルで試す' }).click();
  const downloaded = page.waitForEvent('download');
  await page.getByRole('button', { name: 'プロジェクトをバックアップ' }).click();
  const zip = unzipSync(new Uint8Array(await readFile((await (await downloaded).path())!)));
  const manifest = JSON.parse(strFromU8(zip['manifest.json']));
  manifest.images[0].sha256 = '0'.repeat(64);
  zip['manifest.json'] = strToU8(JSON.stringify(manifest));
  const corrupt = zipSync(zip, { level: 0 });
  await page.getByLabel('プロジェクトZIPを選択').setInputFiles({
    name: 'bad.zip', mimeType: 'application/zip', buffer: Buffer.from(corrupt),
  });
  await expect(page.getByText('12 / 64 枚')).toBeVisible();
  await expect(page.getByLabel('クラス1の名前')).toHaveValue('暖色');
  await expect(page.getByText('画像ファイルが欠損', { exact: false })).toBeVisible();
});
