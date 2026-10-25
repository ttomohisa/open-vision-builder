import { test, expect } from '@playwright/test';
import { mkdir, writeFile, readFile } from 'node:fs/promises';
import { unzipSync, strFromU8 } from 'fflate';
import { createHash } from 'node:crypto';

const sha256 = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');

test('trains real pretrained features, exports a complete verified model, and invalidates edited data', async ({ page, context }) => {
  const external: string[] = [];
  const errors: string[] = [];
  context.on('request', request => {
    const url = new URL(request.url());
    if (url.protocol.startsWith('http') && url.origin !== 'http://127.0.0.1:4173') external.push(request.url());
    if (request.method() !== 'GET') external.push(`${request.method()} ${request.url()}`);
  });
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('./');
  expect(await page.evaluate(() => crossOriginIsolated)).toBe(false);
  await expect(page.getByRole('button', { name: '学習してONNXを作成' })).toBeDisabled();
  await page.getByRole('button', { name: 'サンプルで試す' }).click();
  await expect(page.getByRole('button', { name: '学習してONNXを作成' })).toBeEnabled();
  await page.getByRole('button', { name: '学習してONNXを作成' }).click();
  await expect(page.getByRole('heading', { name: 'モデルを書き出せます' })).toBeVisible({ timeout: 150_000 });
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'モデル一式をダウンロード' }).click();
  const download = await downloadPromise;
  const path = await download.path();
  expect(path).not.toBeNull();
  const files = unzipSync(new Uint8Array(await readFile(path!)));
  for (const required of ['model.onnx', 'labels.json', 'manifest.json', 'report.json', 'LICENSE.txt', 'NOTICE.txt', 'MODEL_CARD.md', 'predict.py', 'README.txt']) expect(files[required]?.length).toBeGreaterThan(0);
  const report = JSON.parse(strFromU8(files['report.json']));
  expect(report.training.epochsCompleted).toBeGreaterThan(0);
  expect(report.training.weightDeltaL2).toBeGreaterThan(0);
  expect(report.metrics.testCount).toBe(4);
  expect(report.metrics.accuracy).toBeGreaterThanOrEqual(0.5);
  expect(report.verification.native.passed).toBe(true);
  expect(report.verification.export.passed).toBe(true);
  expect(report.training.scope).toBe('classifier-head-only');
  expect(files['model.onnx'].byteLength).toBeGreaterThan(9_000_000);
  await mkdir('artifacts/browser', { recursive: true });
  await writeFile('artifacts/browser/model.onnx', files['model.onnx']);
  await writeFile('artifacts/browser/report.json', files['report.json']);
  await writeFile('artifacts/browser/labels.json', files['labels.json']);
  await writeFile('artifacts/browser/manifest.json', files['manifest.json']);
  await page.screenshot({ path: 'artifacts/browser/workflow.png', fullPage: true });
  await expect(page.getByTestId('project-save-state')).toContainText('自動保存済み');
  const originalModelSha = sha256(files['model.onnx']);
  await page.reload();
  await expect(page.getByRole('heading', { name: 'モデルを書き出せます' })).toBeVisible();
  const restoredDownloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'モデル一式をダウンロード' }).click();
  const restoredDownload = await restoredDownloadPromise;
  const restoredFiles = unzipSync(new Uint8Array(await readFile((await restoredDownload.path())!)));
  expect(sha256(restoredFiles['model.onnx'])).toBe(originalModelSha);
  await page.getByLabel('クラス1の名前').fill('変更したクラス');
  await expect(page.getByRole('button', { name: 'モデル一式をダウンロード' })).toHaveCount(0);
  expect(external).toEqual([]);
  expect(errors).toEqual([]);
});

test('cancelling a run does not expose a stale model', async ({ page }) => {
  await page.goto('./');
  await page.getByRole('button', { name: 'サンプルで試す' }).click();
  await page.getByRole('button', { name: '学習してONNXを作成' }).click();
  await page.getByRole('button', { name: '中断' }).click();
  await expect(page.getByText('学習を中断しました。')).toBeVisible();
  await expect(page.getByRole('button', { name: 'モデル一式をダウンロード' })).toHaveCount(0);
  await expect(page.getByRole('button', { name: '学習してONNXを作成' })).toBeEnabled();
});


test('autosaves dataset and settings and restores them after reload', async ({ page }) => {
  await page.goto('./');
  await expect(page.getByTestId('project-save-state')).not.toContainText('読み込み中');
  await page.getByRole('button', { name: 'サンプルで試す' }).click();
  await page.getByLabel('クラス1の名前').fill('保存テスト');
  await page.locator('details.settings').click();
  await page.getByLabel('エポック数').fill('23');
  await page.getByLabel('学習率').fill('0.02');
  await expect(page.getByTestId('project-save-state')).toContainText('自動保存済み');
  expect(await page.locator('.thumbnail-list li').count()).toBe(12);

  await page.reload();
  await expect(page.getByLabel('クラス1の名前')).toHaveValue('保存テスト');
  await expect(page.getByLabel('エポック数')).toHaveValue('23');
  await expect(page.getByLabel('学習率')).toHaveValue('0.02');
  expect(await page.locator('.thumbnail-list li').count()).toBe(12);
  await expect(page.getByRole('button', { name: '学習してONNXを作成' })).toBeEnabled();
});

test('backup import is atomic and clear removes the local project', async ({ page }) => {
  await page.goto('./');
  await page.getByRole('button', { name: 'サンプルで試す' }).click();
  await page.getByLabel('クラス1の名前').fill('バックアップ元');
  await expect(page.getByTestId('project-save-state')).toContainText('自動保存済み');

  const backupDownloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'バックアップを書き出す' }).click();
  const backup = await backupDownloadPromise;
  const backupPath = await backup.path();
  expect(backupPath).not.toBeNull();
  const original = new Uint8Array(await readFile(backupPath!));

  await page.getByLabel('クラス1の名前').fill('現在を守る');
  await expect(page.getByTestId('project-save-state')).toContainText('自動保存済み');
  const corruptPath = 'artifacts/browser/corrupt-project.ovb.zip';
  await mkdir('artifacts/browser', { recursive: true });
  await writeFile(corruptPath, original.slice(0, Math.max(0, original.length - 7)));
  await page.getByLabel('プロジェクトのバックアップを読み込む').setInputFiles(corruptPath);
  await expect(page.getByRole('alert')).toContainText('バックアップを読み込めませんでした');
  await expect(page.getByLabel('クラス1の名前')).toHaveValue('現在を守る');
  expect(await page.locator('.thumbnail-list li').count()).toBe(12);

  page.once('dialog', dialog => dialog.accept());
  await page.getByRole('button', { name: '端末内のプロジェクトを削除' }).click();
  await expect(page.getByLabel('クラス1の名前')).toHaveValue('クラス1');
  expect(await page.locator('.thumbnail-list li').count()).toBe(0);

  await page.getByLabel('プロジェクトのバックアップを読み込む').setInputFiles(backupPath!);
  await expect(page.getByText('バックアップを読み込みました。')).toBeVisible();
  await expect(page.getByLabel('クラス1の名前')).toHaveValue('バックアップ元');
  expect(await page.locator('.thumbnail-list li').count()).toBe(12);
  await expect(page.getByTestId('project-save-state')).toContainText('自動保存済み');
});
