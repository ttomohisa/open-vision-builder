# Classification Project Persistence Design

Date: 2026-10-25 JST

## Intent

分類プロジェクトをブラウザを閉じても失わず、同一 origin で自動復元し、明示的なバックアップ ZIP で別ブラウザ・別端末へ移せるようにする。ユーザー画像・ラベル・設定・学習後モデルは運営者へ送信しない。

## Scope

Goals:
- クラス名、画像、train/test 分割、学習設定を編集後に自動保存する。
- ページ再読み込み時に直近の分類プロジェクトを自動復元する。
- 学習済みの model kit ZIP と評価 summary も保存し、再読み込み後に再ダウンロードできる。
- プロジェクト全体を schema-versioned ZIP に書き出し、checksum 検証後に読み込める。
- 保存容量の使用量/上限と persistent-storage 状態を表示し、ユーザー操作で persist() を要求できる。
- 保存データの削除を明示操作で行える。
- 保存失敗・容量不足・破損バックアップを、作業中データを壊さず説明する。

Non-goals:
- 学習途中の optimizer / checkpoint / epoch 再開。
- 複数プロジェクト一覧、名前変更、履歴管理。
- Service Worker / オフラインアプリ化。
- 検出プロジェクトの保存。
- クラウド同期、API、アカウント。

## Approaches considered

1. IndexedDB に画像 Blob も含めてすべて保存: 実装は最短だが、大きい画像・モデルをメタデータと同じトランザクションに詰め込み、既存 ARCHITECTURE の境界とも合わない。
2. IndexedDB + OPFS（採用）: IndexedDB は schema/version/labels/settings/sample metadata/result summary、OPFS は画像と model-kit ZIP。大きい byte 列を分離でき、後続 checkpoint 実装へ自然に拡張できる。
3. バックアップ ZIP だけ: サイトデータ消去への備えにはなるが、通常の再読み込みで復元できず Custom Vision 的な作業体験にならない。

## Storage contract

- IndexedDB: open-vision-builder / projects / classification-default.
- OPFS: projects/classification-default/images/<sample-id> と projects/classification-default/result/model-kit.zip.
- manifest schemaVersion: 1; task: classification; model: mobilenetv4-conv-small.
- sample metadata: id/name/mime/bytes/label/split/synthetic/sha256.
- settings: epochs / learningRate.
- result metadata: RunSummary + model-kit bytes/hash when present.

保存は OPFS の必要ファイルを書き終えてから IndexedDB manifest を確定する。manifest だけが復元可能な世代を指す。manifest 確定後に不要 OPFS ファイルを掃除する。自動保存に失敗しても現在の React state は消さない。

## Backup format

ZIP root:
- project.json
- images/<sample-id>
- optional result/model-kit.zip

project.json は schemaVersion/task/model/labels/settings/samples/result を持ち、各 binary entry の path/bytes/sha256 を持つ。import 時は path traversal を拒否し、entry 数・画像数・各画像20 MiB・合計サイズを上限化し、mime/label/split/settings/checksum を全検証してから現在プロジェクトを置換する。ファイル名は open-vision-project.ovb.zip とする。

## UI

データセット見出し付近にプロジェクト操作を追加する。保存状態、バックアップ書き出し/読み込み、端末保存の削除、storage estimate、persistent 状態、保存を保護ボタンを表示する。起動時復元が終わるまでは編集/学習を無効にし、空データを先に自動保存して既存データを消さない。

## Error handling

- IndexedDB / OPFS 非対応: 自動保存を無効化し、理由を表示。モデル学習自体は継続利用可能。
- QuotaExceededError: 容量不足として明示し、バックアップ書き出しを案内。
- backup checksum / schema / path / sample contract failure: import を中断し、現在プロジェクトを変更しない。
- manifest にある OPFS file が欠落: 自動復元を失敗として扱い、壊れた保存を勝手に空データで上書きしない。

## Verification

- Pure unit tests: manifest/backup encode-decode, checksum, traversal, limits, malformed metadata.
- Chromium: sample を準備 → 自動保存待ち → reload → labels/images/settings 復元。
- Chromium: 学習完了 → reload → result 復元 → model kit を再ダウンロード。
- Chromium: backup export → local project clear → import → dataset/result 復元。
- Chromium: corrupted backup import leaves existing project unchanged.
- Existing classification browser/Python ONNX regression stays green.

## Later

checkpoint の粒度、optimizer state、複数プロジェクト、OPFS temp/atomic rename recipe、Service Worker は別PRで扱う。