# 分類: 実画像での学習・品質・ONNX 比較

- 日付: 2026-10-25 JST。
- 根拠: PR #6 の次作業、STATUS、ADR 0005。ユーザーが PR #6 をマージして継続を指示した。
- 比較元: `c088e211400e2db91e488858382fca39971cf41d`。専用ブランチ `test/classification-real-images`。
- 実装上の選択は以下に明示する。製品要件の変更ではない。

## 答える問い

現在の分類画面の既定設定で、実写真の新しい5クラスを学習できるか。同じ画像・前処理・固定 backbone・初期 head・学習レシピの PyTorch と比べて品質や出力が悪化しないか。ブラウザが生成した完全 ONNX を、学習状態のない別 context と Python で使えるか。

## 方法と選択理由

既存の App → Worker → ZIP の実経路を Playwright で操作する。比較用 fixture は既存の decode / preprocess / predictHead を呼び、独立 PyTorch は元 timm の特徴を自分で計算する。新しい製品エンジンや診断 API は作らない。

検討した方法は、(1) 既存画面を使う、(2) Worker だけを直接起動する、(3) 学習処理を別実装する、の3つ。今回は画像取り込みから持ち出しまでの実経路も確かめられる (1) を選ぶ。補助 fixture による追加の推論は測定に含めるが、ブラウザの学習は一回だけとする。

公開 TensorFlow flower_photos の5クラスから60画像を、推論前の固定規則で選ぶ。各クラス6 train / 2 tuning reserved / 4 evaluation。tuning の10枚は今回の学習・推論・設定選択に使わず、実行対象は50枚。写真家名を正規化して全60枚で重複を避け、元画像 hash と Flickr photo ID も一意にする。明らかなラベル誤り・花のない画像・写真ではない図版や人工の花の除外理由を保存する。難しい構図・写真加工・背景物体を理由に除外しない。

JPEG の codec 差が学習系比較へ混ざらないよう、固定 Pillow 11.1.0 で EXIF 適用・RGBA 化した lossless PNG を入力にする。この選択は測定前に固定し、元 JPEG と PNG の hash・変換内容・著作者と利用条件を記録する。アプリと Python は同じ PNG をそれぞれ decode / preprocess する。JPEG 全般の互換性はこの結果から認定しない。

## 固定する条件

- 既存の MobileNetV4 Conv Small の正確な公開 checkpoint・準備 ONNX・ライセンスを継続使用する。
- 5クラスの head W `[1280,5]` / b `[5]` をゼロ初期化。全 backbone / BatchNorm は固定。
- 現在の UI 既定値: full batch 30枚、40 SGD updates、learning rate 0.01、平均 softmax cross entropy。shuffle・augmentation・momentum・decay・schedule なし。
- 同一クラス順・画像順・入力サイズ224・前処理を使用。PyTorch の特徴抽出は batch1、head だけ full batch。
- 元の数値条件 `abs(actual-expected) <= 2e-4 + 1e-4*abs(expected)` を features / 元 logits / loss trajectory / final head / 学習後 logits / export に使用。入力の最大絶対差は既存 decoder 条件の `< 1e-6`。
- 最後の保存 loss は update 40 の直前である。更新後の train CE / accuracy は別途最終重みで測定する。
- この小規模試験の最低条件は、両実装の train CE 改善率20%以上・train accuracy90%以上・evaluation accuracy70%以上。両実装の evaluation accuracy と macro F1 の差はそれぞれ0.05以下。これは実装者が定める初期の実用性点検であり、製品全体の品質保証・ユーザーとの精度契約ではない。
- per-image logits / predicted label、confusion matrix、クラス別 recall、macro F1、誤分類一覧を保存。平均値だけで失敗例を隠さない。

## 完了と打切り

固定した計画・データ・コードを PR に保存してから実画像のモデル測定を開始する。一回の完全実行、controller 最大10分、実作業90分を初期上限とし、結果に合わせて画像・設定・条件を変えない。明確な実行基盤の不具合は、最初の証拠を保持して、別コミットで一回だけ修復可能。数値・品質結果を変える目的の再試行は禁止。

非有限値・入力不一致・破損・予算超過は保存して停止する。モデル数値条件または品質条件が外れても、残る独立した出力検証を予算内で保存し、当該実験は FAIL のまま終了する。

| 結果 | 次の判断 |
| --- | --- |
| 学習・品質・独立 export の条件内 | 現分類レシピを継続する根拠とし、次は対象環境の資源・中断/保存へ進む |
| PyTorch も学習・品質の条件未達 | 既定レシピと少量データ適合性を再評価。ブラウザ runtime を即断しない |
| PyTorch は満たすが browser が未達 | 保存した入力・loss・最終 head の比較から具体的な修正仮説を別作業にする |
| export が条件未達 | 完全モデルと比較入力を保存し、利用可能という判定を保留。許容差を緩めない |

5種類の花、少量・写真家名単位の分割、固定 backbone、WASM、当該 Linux/Chromium 環境に限る。元 ImageNet 学習との画像重複やドメイン独立は保証しない。Medium、backbone fine-tuning、検出、WebGPU、一般 PC の時間・メモリ・edge・保存再開の認定を増やさない。成功しても G4 全体・P1 全体・初期版完成とはしない。

## 保存するもの

Git: 固定入力 manifest と出典、計画、比較コードと小さい単体テスト、実測集計・全画像の予測・条件外結果、STATUS 等の更新。大きな画像・input tensor・model・trace は ignored artifacts。生 artifact を恒久保管済みと呼ばない。運営 API / DB や画像アップロード経路は追加しない。マージはユーザーが行う。
