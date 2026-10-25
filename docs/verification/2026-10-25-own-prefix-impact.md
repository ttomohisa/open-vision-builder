# D-FINE: 実画像での own-prefix 学習・出力影響

- 日付: 2026-10-25 JST。
- 判定: **測定完了、事前条件は未達。検出器の採用・公開は保留。**
- 事前登録: `f5b07fc82d950106c2e4165fea3869236c488b2f` / [PR #6](https://github.com/ttomohisa/open-vision-builder/pull/6)。
- 完了した実験のコード: `34e1f2b8cd957c52c4fe4c3322b478bc8db8a86d`。
- [実行計画](../superpowers/plans/2026-10-25-dfine-own-prefix-impact.md)、[データ出典](2026-10-25-impact-data.md)、[集計・対照・export の証拠](evidence/2026-10-25-own-prefix-impact.json)、[評価画像の検出結果](evidence/2026-10-25-own-prefix-detections.json)。

## 結論とここで止める理由

ブラウザと PyTorch がそれぞれ自身の frozen-prefix 特徴を使った12 stepでは、matching・最終 logits/boxes・損失・全学習対象の勾配・更新値の比較は通った。同じブラウザ特徴と現在重みを使う0/6/12 step の対照も通った。一方、中間の `corners` には不一致が残る。

**両実装の平均学習損失はわずかに下がったが、事前指定した1%の改善に届かなかった。** ブラウザは約0.004070%、PyTorch は約0.003928%である。この条件では学習効果の比較に必要な変化を確保できず、「有効な追加学習が成立」「ブラウザの差は無害」とは認定しない。

完全 ONNX の再読込は別ブラウザ context で10/10画像が一致したが、Python CPU では2画像が元の raw-output 条件を外れた。出力ファイルに更新重みが正しく入ることと、異なる runtime の出力一致は別の確認である。**学習効果の条件未達と Python 出力の不一致の両方を残し、この実験を終了する。** 条件・画像・step数を変える再試行や decoder 分解は追加しない。

## 固定条件と実行範囲

| 項目 | 条件 |
| --- | --- |
| モデル | 固定した COCO-only D-FINE Nano。checkpoint SHA-256 `19e06bdc873da819920a8d373b879721a5b9759d822f8213220bb09abbdab58b` |
| 学習対象 | 最終 pretrained bbox / class ヘッド、8テンソル・60,372パラメータ。80クラス維持、prefix / LQE 固定 |
| データ | 公開実画像10枚、全58注釈。train 4枚27注釈、tuning 2枚3注釈、evaluation 4枚28注釈 |
| 同一入力 | Pillow 11.1.0 RGB bilinear stretch、640×640、NCHW FP32 / 255。各 runtime は同じ input bytes を読む |
| 更新 | 固定順で3 epoch、batch 1、12 SGD step、learning rate 1e-4、momentum / weight decay 0。augmentation・shuffle なし |
| 損失 | 既存の限定レシピ `VFL + 5 * L1 + 2 * GIoU`。補助・FGL・DDF・denoising loss なし |
| 対照 | step 0/6/12 で最初の学習画像。実際のブラウザ特徴・同じ現在重みを独立 PyTorch ヘッドへ渡す。仮の更新は本学習へ適用しない |
| 検出指標 | score ≥0.05、画像ごとに全 query/class の上位100件、NMS なし、IoU 0.5、101点補間 AP。正解のあるクラスで平均 |
| 環境 | Linux x86_64、Node 24.22.1、Chromium 156.0.8078.4、jax-js WASM、ORT Web 1.30.0、PyTorch 2.6.0+cpu、ORT Python 1.20.1 CPU |

画像・前処理・annotation・分割の manifest hash は `464570f5138ddfdd159892a9e3a9a4723d45ee335de1bc93965bc93fc6b289a6`、計画 hash は `1b8ef180b93e5b6a224fb396a1a4b54f24a8e0b74cd109b13f15c567b85888d3`。測定前の保存と、実行時の hash を照合した。

評価画像は**この追加学習・調整に未使用**という意味で、元の COCO checkpoint の選定や上流開発から独立だったとは保証しない。画像単位の分割であり、tuning と evaluation に同じ著作者の別写真がある。独自クラスや一般的な品質の認定には使わない。画像の browser decode / resize は今回の比較範囲外である。

## 学習と対照の結果

| 確認項目 | 結果 |
| --- | --- |
| 同一特徴・同一重みの対照 | 0/6/12 の3か所すべて PASS。出力、4損失、matching、全8勾配、全更新値を既存許容差で比較 |
| 各 runtime 自身の特徴での matching | 全12 step で一致。参照側の matching の強制や query 再整列は行わない |
| 同じ12 step の最終 logits / boxes・損失・勾配・更新 | 全12 step、全8学習テンソルの比較が既存の部品許容差内 |
| `corners` の同じ比較 | 9/12 step で不一致。785/885/1268 の各画像で11/5/13要素が不一致となり、3 epoch にわたって残る |
| 学習中の損失成分の最大相対差 | VFL、step 2、`9.018692577128833e-5`（約0.009019%）。事前条件の1%以内 |
| 固定対象 | PyTorch source state / LQE 不変。完全 ONNX の固定 initializer 576個も byte 単位で不変 |
| 実更新 | 位置・分類ヘッドとも非ゼロ更新。ブラウザの変化の L2 は位置 `0.0001160486653`、分類 `0.0001182715337` |

「同一特徴の対照が通る」「own-path の損失・勾配・更新が通る」「中間を含む全出力が通る」を混同しない。各 step の総合部品判定は `corners` のため9/12で FAIL であり、旧 G1 は解消しない。

| 学習の結果 | ブラウザ | PyTorch |
| --- | ---: | ---: |
| 学習画像の平均 total loss、更新前 | 0.9054103792 | 0.9054430127 |
| 同、12 step 後 | 0.9053735286 | 0.9054074436 |
| 改善率 | **0.00407004%** | **0.00392836%** |
| 事前条件「1%以上改善」 | **FAIL** | **FAIL** |
| 評価4画像の AP50、更新前 | 0.8463458012 | 0.8463458012 |
| 同、更新後 | 0.8463458012 | 0.8463458012 |

AP50 の差と悪化幅は事前条件内だったが、学習効果の条件が通ったことにはならない。この AP50 は4画像の6正解クラスだけの集計で、COCO 全体の AP や製品の実用精度ではない。固定した低い score threshold / 上位100件の条件では、両実装・更新前後とも **25 TP / 375 FP / 3 FN** だった。これらの指標は own-prefix の両経路についての結果であり、Python で実行した export の品質一致を証明するものではない。

own-path の学習前後の raw logits/boxes 比較では、tuning の **6012 と6818** がどちらの時点でも条件外だった。この2画像と、次節の export で失敗した2画像は一致しない。

## 完全 ONNX: ファイルの成立と runtime 差

生成物は **16,161,676 bytes** の完全な FP32 ONNX。SHA-256 は `07be5b63e3e6d81306733ee01b0211805da75426af650094e926a4ba5bd5a209`。

- 学習した8 initializer は最終ブラウザ重みと byte 単位で一致。
- 固定 initializer 576個、全 graph node、initializer の集合は不変。
- training context を閉じ、別 context が生成済みモデルと入力だけを読んだ。学習特徴・重み・cache を引き継がない。
- ブラウザ最終 head と fresh browser の logits/boxes は **10/10画像 PASS**。
- Python CPU とブラウザの同じ比較は **8/10画像 PASS、2/10画像 FAIL**。Python と fresh browser の比較でも同じ2画像が失敗。

| 画像 / 分割 | 条件外の値 | Python と最終 browser の最大絶対差 |
| --- | --- | ---: |
| 6012 / tuning | logits 160/24,000、boxes 8/1,200 | logits **1.7801446915**、boxes **0.3136631250** |
| 12667 / evaluation | logits 4/24,000、boxes は条件内 | logits **0.0006556511** |

元の条件は logits `atol=2e-4, rtol=1e-4`、boxes `atol=2e-5, rtol=1e-4` のまま。6012には大きい差もあり、不合格要素の少なさだけを根拠に軽微な丸め差とは扱わない。原因はこの実験から特定していない。query の順序変更や許容差の緩和で PASS にする処理は加えていない。

## 実行履歴・通信・検証

最初の試行は browser の checkpoint 0 proposal を測定した後、Python control を起動する前に Node の整数 timeout 要件で停止した。受け入れた更新は0回、完了した対照も0回だった。[失敗記録](evidence/2026-10-25-impact-infrastructure.json)を保持した。回帰テストの失敗を確認し、残り時間を整数にする修正だけを `34e1f2b` に保存した。計画・データ・学習設定・判定値は変更していない。

その後の一回の完全実行は **165.35秒**、その内の独立 PyTorch 生成は26.93秒。これは当該環境での開発用 controller 全体の経過時間であり、一般 PC の学習速度やピークメモリの保証ではない。前の失敗資料を上書きせず `artifacts/p1d/run-after-timeout-fix/` に保存した。

通信観測はローカル GET 35件、想定外通信0件、page error 0件。native 特徴や学習済みモデルをブラウザから API へ送る経路はない。大きな raw tensor / trace / モデルは ignored artifacts に保持し、Git には固定した再現コード・入力来歴・各 artifact hash・全比較集計・評価画像の検出結果を保存する。raw artifact は恒久保管サービスに保存済みという扱いにはしない。

```bash
artifacts/p1b/.venv-detector/bin/python scripts/prepare_detector_impact_data.py
node scripts/run_detector_impact.mjs --check
node scripts/run_detector_impact.mjs --output artifacts/p1d/run-after-timeout-fix
```

既存の出力先は拒否する。ローカルでは Playwright CDN が ZIP ではなく HTML を返したため、同じ版の Chrome を公式 Google 配布 URL から取得し、`--chromium /tmp/ovb-chrome-156/chrome-linux64/chrome` を指定した。比較条件や runtime の版を変更する回避はしていない。

Python **83件**、JavaScript **101件**、typecheck と build が成功。独立したコードレビューと、保存 JSON に基づく結果確認を実施した。CI は新しいデータ再現と推論をしない browser preflight を追加し、既存の29件の browser 検証・分類と限定検出の完全 ONNX を確認する。新しい real-image 比較の成功を CI の成功から推定しない。

## 次の作業への判断

1. **この実験は終了する。** 元の条件、未達結果、旧 G1 FAIL を維持する。PR #6 は固定データ・比較コード・正負の証拠を保存する単位で区切り、検出器の製品採用まで同じ PR を延長しない。
2. **分類の実画像品質・同条件 PyTorch 比較を、独立した次の P1 作業として優先する。** 検出の数値調査だけで、既に動く分類の確認を止めない。
3. 検出の学習については、PyTorch も1%基準に届かなかったため、この限定レシピ・学習量・データ適合性を見直す。今回の結果だけで runtime 不良やモデル変更を決めない。追加の学習試験は別の事前登録を必要とし、ここで step数・learning rate・画像を試行錯誤しない。
4. Python export の不一致は、判明した入力と比較を起点とする別の限定課題として残す。次に調べる場合は、問い・具体的な修正仮説・結果別判断・上限を先に定める。decoder 全体の自動分解や、観測最大差に合わせる数値契約の変更には戻らない。

アプリの検出画面、独自クラス、全層 fine-tuning、WebGPU、一般端末、edge の認定、保存再開は今回完成していない。新しい Pages 公開も行わない。初期版から完全 ONNX・ブラウザ学習・商用利用条件を守るという製品要件は変更しない。
