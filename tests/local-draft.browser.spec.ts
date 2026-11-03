import { test, expect } from '@playwright/test';

const saved = 'このブラウザに保存済みです。ZIPバックアップも定期的に保存してください。';

test('autosave survives reload with source images, class names and unfinished settings', async ({ page }) => {
  await page.goto('./');
  await page.getByRole('button', { name: 'サンプルで試す' }).click();
  await expect(page.getByText('12 / 64 枚')).toBeVisible();
  await page.getByLabel('クラス1の名前').fill('ローカル保存');
  await page.locator('details.settings summary').click();
  await page.locator('input[type="number"]').first().fill('27');
  await expect(page.getByText(saved)).toBeVisible({ timeout: 20_000 });
  await page.reload();
  await expect(page.getByLabel('クラス1の名前')).toHaveValue('ローカル保存');
  await page.locator('details.settings summary').click();
  await expect(page.locator('input[type="number"]').first()).toHaveValue('27');
  await expect(page.getByText('12 / 64 枚')).toBeVisible();
  await page.getByLabel('クラス1の名前').fill(''); // incomplete drafts must still autosave
  await expect(page.getByText(saved)).toBeVisible({ timeout: 20_000 });
  await page.reload();
  await expect(page.getByLabel('クラス1の名前')).toHaveValue('');
  await expect(page.getByText('12 / 64 枚')).toBeVisible();
});

test('stale second tab cannot silently overwrite a newer local draft', async ({ page, context }) => {
  await page.goto('./');
  await page.getByRole('button', { name: 'サンプルで試す' }).click();
  await expect(page.getByText(saved)).toBeVisible({ timeout: 20_000 });
  const second = await context.newPage();
  await second.goto('./');
  await expect(second.getByText('12 / 64 枚')).toBeVisible();
  await page.getByLabel('クラス1の名前').fill('先に更新したタブ');
  await expect(page.getByText(saved)).toBeVisible({ timeout: 20_000 });
  await second.getByLabel('クラス1の名前').fill('古いタブからの変更');
  await expect(second.getByText('別のタブで保存内容が更新されました。', { exact: false })).toBeVisible({ timeout: 20_000 });
  await second.reload();
  await expect(second.getByLabel('クラス1の名前')).toHaveValue('先に更新したタブ');
  await second.close();
});

test('aborted IndexedDB write leaves the previous successful draft recoverable', async ({ page }) => {
  await page.goto('./');
  await page.getByRole('button', { name: 'サンプルで試す' }).click();
  await expect(page.getByText(saved)).toBeVisible({ timeout: 20_000 });
  await page.evaluate(() => {
    const original = IDBObjectStore.prototype.put;
    let abortOnce = true;
    IDBObjectStore.prototype.put = function (...args) {
      const request = original.apply(this, args);
      if (this.name === 'drafts' && abortOnce) {
        abortOnce = false;
        this.transaction.abort();
      }
      return request;
    };
  });
  await page.getByLabel('クラス1の名前').fill('未保存の変更');
  await expect(page.locator('.local-storage-status')).toContainText(/abort|中断|保存できません|transaction/i, { timeout: 20_000 });
  await page.reload();
  await expect(page.getByLabel('クラス1の名前')).toHaveValue('暖色');
  await expect(page.getByText('12 / 64 枚')).toBeVisible();
});

test('corrupted saved image hash blocks autoload rather than silently overwriting the record', async ({ page }) => {
  await page.goto('./');
  await page.getByRole('button', { name: 'サンプルで試す' }).click();
  await expect(page.getByText(saved)).toBeVisible({ timeout: 20_000 });
  await page.evaluate(async () => {
    await new Promise<void>((resolve, reject) => {
      const request = indexedDB.open('open-vision-builder-classification-draft-v1', 1);
      request.onsuccess = () => {
        const db = request.result;
        const tx = db.transaction('drafts', 'readwrite');
        const item = tx.objectStore('drafts').get('current');
        item.onsuccess = () => {
          const record = item.result;
          record.images[0].sha256 = '0'.repeat(64);
          tx.objectStore('drafts').put(record, 'current');
        };
        tx.oncomplete = () => { db.close(); resolve(); };
        tx.onabort = () => { db.close(); reject(tx.error); };
      };
      request.onerror = () => reject(request.error);
    });
  });
  await page.reload();
  await expect(page.getByText('保存した画像のSHA-256が一致しません。')).toBeVisible();
  await expect(page.getByText('0 / 64 枚')).toBeVisible();
  // The unchanged browser record is recoverable through an external ZIP backup;
  // the app must not write a blank draft over the corrupted record.
  const count = await page.evaluate(async () => new Promise<number>((resolve, reject) => {
    const request = indexedDB.open('open-vision-builder-classification-draft-v1', 1);
    request.onsuccess = () => {
      const db = request.result;
      const tx = db.transaction('drafts');
      const value = tx.objectStore('drafts').get('current');
      value.onsuccess = () => resolve(value.result.images.length);
      tx.oncomplete = () => db.close();
    };
    request.onerror = () => reject(request.error);
  }));
  expect(count).toBe(12);
});
