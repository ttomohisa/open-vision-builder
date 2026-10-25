# 分類の実画像比較: 固定データと出典

- 日付: 2026-10-25 JST。
- 対象: [分類の事前設計](../superpowers/specs/2026-10-25-classification-real-images-design.md)に使う公開写真。画像の取得・選択・目視点検・PNG 化だけを行い、このデータ準備ではモデル推論・学習を行わない。
- 固定 manifest: [`tests/fixtures/classification-real-data.json`](../../tests/fixtures/classification-real-data.json)。SHA-256: `7d6885edda14c5e08b9a138d5f56ad9ab15b4d9ce3214b1d2befd9ceb5f80805`。

## 元データと固定した配布物

TensorFlow の [画像分類 tutorial](https://www.tensorflow.org/tutorials/images/classification) が示す公式公開オブジェクト [`flower_photos.tgz`](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz) を利用する。[TensorFlow Datasets の catalog](https://www.tensorflow.org/datasets/catalog/tf_flowers) は3,670画像・5クラスを記載し、[公式 builder](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/tf_flowers/tf_flowers_dataset_builder.py) は事前定義の train/validation/test 分割がないことを明示する。

| 配布物 | 確認値 |
| --- | --- |
| Archive URL | `https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz` |
| Archive bytes | 228,813,984 |
| Archive SHA-256 | `4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c` |
| Archive 内の写真ライセンス | `flower_photos/LICENSE.txt`、418,049 bytes |
| LICENSE.txt SHA-256 | `701e91b9e9f46d1f9623fc793a00fd444fe2821f66b01f3a0ff4a73d82f49507` |

取得した全 JPEG を数え、全3,670画像に写真家と Flickr URL の attribution 行があることを確認した。9行だけは `CC-BY <URL> - by <creator>` という逆順の書式で、欠落ではない。

| class index | class name | Archive 内 | train | tuning reserved | evaluation |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | daisy | 633 | 6 | 2 | 4 |
| 1 | dandelion | 898 | 6 | 2 | 4 |
| 2 | roses | 641 | 6 | 2 | 4 |
| 3 | sunflowers | 699 | 6 | 2 | 4 |
| 4 | tulips | 799 | 6 | 2 | 4 |
| 合計 |  | 3,670 | 30 | 10 | 20 |

クラス番号は上表の順に固定する。TFDS builder の番号順をそのまま引き継いだものではない。既存アプリの合計64枚の上限に対し、選択は60枚。今回の測定に読み込むのは train と evaluation の50枚だけで、tuning の10枚を推論・学習・設定選択に使用しない。

## 写真の利用条件と表示

アーカイブ内 `LICENSE.txt` の冒頭は次の全文である。

```text
All images in this archive are licensed under the Creative Commons By-Attribution License, available at:
https://creativecommons.org/licenses/by/2.0/
The photographers are listed below, thanks to all of them for making their work available, and please be sure to credit them for any use as per the license.
```

写真の利用条件は **CC BY 2.0**。TensorFlow の説明ページの CC BY 4.0 やコードの Apache-2.0 と混同しない。[Creative Commons の deed](https://creativecommons.org/licenses/by/2.0/) と [legal code](https://creativecommons.org/licenses/by/2.0/legalcode) に従い、写真の複製・加工・商用利用には著作者表示等の条件を保つ。

固定 manifest の各画像に `creator`、`sourceUrl`、`licenseUrl` とアーカイブからの `attributionLine` を保存した。元 attribution inventory は写真タイトルを提供していない。写真や contact sheet を再配布するときは、対応する著作者・出典・ライセンスへのリンク・この PNG 変換の説明を付ける。元の作者による推薦を示唆しない。モデル自体の利用条件は既存の [MODELS](../MODELS.md) に従い、写真のライセンスと区別する。

元 JPEG、canonical PNG、contact sheet、アーカイブを Git・`public/`・`dist/` に含めない。今回の公開データ準備は開発者のローカル処理であり、アプリからユーザーデータを送信する経路を追加しない。

## 推論前に固定した選択規則

1. クラスを `daisy, dandelion, roses, sunflowers, tulips` の順に処理する。
2. `flower_photos/` を除いた相対パスに対して `SHA256(UTF8("ovb-classification-flowers-v1" + U+0000 + relativePath))` を計算する。NUL は1文字の区切りであり、文字列 `\\0` ではない。各クラスで digest の昇順、同値ならパスの昇順に並べる。
3. 写真家名は Unicode NFKC → casefold → Unicode 空白で分割し単一 ASCII 空白で結合する。正規化した名前の完全一致だけを用い、別名を推測して統合しない。
4. 下表の内容による除外、既に採用した写真家名、JPEG の同一 SHA-256、同じ Flickr photo ID を除く。写真家・内容・photo ID の一意性は全クラスにまたがって確認する。
5. 各クラスの最初の12枚を採用する。採用後の0〜5番が train、6〜7番が tuning reserved、8〜11番が evaluation。除外後に別の乱数 seed を探さない。

画像・分割・設定・除外はモデル出力を見て選んだものではない。最終60枚に対して著作者名、Flickr owner URL、Flickr photo ID、JPEG bytes、canonical RGBA の一意性を確認した。写真家名の別名や転載元の違いまで完全に同定するものではない。

### 目視点検で除外した7枚

初期候補と置き換え候補を目視し、以下を測定前に除外した。元ファイルの SHA-256 と理由は再現可能な source archive と固定 manifest に結び付く。写真加工、背景物体、複雑な構図、モデルにとって難しそうという理由では除外していない。

| 元画像の相対パス | 内容による理由 |
| --- | --- |
| `roses/2408236801_f43c6bcff2.jpg` | カップをつるした装飾で、自然のバラを対象にした写真ではない |
| `tulips/4522764992_e9d70b82c1_m.jpg` | 青白いアヤメ類で、チューリップというラベルと一致しない |
| `roses/17953368844_be3d18cf30_m.jpg` | 大聖堂の天井で、花がない |
| `sunflowers/13354458753_7b586f7c95_n.jpg` | 自然のヒマワリではなく、ヒマワリ形の人工の庭飾り |
| `roses/16961613890_695b36aab2_m.jpg` | バラの図版を印刷した種苗 catalog の scan |
| `roses/16149016979_23ef42b642_m.jpg` | バラの図版を印刷した種苗 catalog の scan |
| `tulips/15147473067_7c5498eb0e_m.jpg` | 花の図版を印刷した種苗 catalog の scan |

最終60枚を contact sheet で目視点検し、追加の明白な対象不一致は見つけなかった。花畑のトラクター、花と人物、写真の色・texture 加工、バラと猫の collage などはそのまま残した。これは元の分類ラベルの粗い妥当性点検であり、花の植物学的な種名を再注釈したものではない。

元アーカイブ全体には、JPEG bytes と decoded RGB の完全重複が3組あり、1組は roses と tulips の両方に入っていた。最終選択にこれらの組は含まれない。初期の候補で見えた同一写真家の類似撮影は、最終の全60枚で写真家名を一意にする規則によって分割をまたがらない。目視点検と画像 hash は、近い構図・撮影系列・元事前学習データとの重複が一切ないことの証明ではない。

## Canonical PNG と独立 decode

JPEG codec の差を学習系比較へ持ち込まないため、元 JPEG から lossless PNG を作る。両実装はこの同じ PNG bytes を別々に decode / preprocess する。

- Pillow **11.1.0**。作成環境の JPEG codec 表示は6.2、libjpeg-turbo 3.1.0、zlib 1.3。
- JPEG を decode し、`ImageOps.exif_transpose` を適用して RGBA に変換する。
- 変換後の RGBA bytes を新しい画像へコピーし、EXIF 等の付加 metadata を除く。
- PNG の `compress_level=9, optimize=False`。リサイズ・crop・augmentation はしない。ICC profile に基づく色変換は追加しない。
- 最終選択の元画像の EXIF orientation はすべて1。向きの適用処理自体は別の小さい単体テストで検証する。

元 JPEG の合計は4,199,875 bytes、canonical PNG の合計は13,165,890 bytes。各 JPEG と PNG の bytes・SHA-256、幅・高さ、decoded RGBA SHA-256 を manifest に記録する。Pillow 等が違って canonical bytes が変わった場合は処理を止め、hash を自動更新しない。

この設計は実アプリの画像取り込みと PNG decode を通すためのもの。異なる JPEG decoder 間の一般的な互換性や色管理の同等性を認定する結果には使わない。

## 再現と検証

```bash
.venv/bin/python scripts/prepare_classification_data.py
.venv/bin/python -m unittest discover -s tests -p classification_data_test.py -v
```

取得済みアーカイブを明示する場合:

```bash
.venv/bin/python scripts/prepare_classification_data.py \
  --archive artifacts/p1e/data-source/flower_photos.tgz
```

`--manifest` は同じ固定 fixture bytes を別の場所から読むために使える。画像を差し替えるための緩和オプションではない。`--output` の既定値は `artifacts/p1e/data`。出力は `images/{class-flickrId}.png` の60ファイルと、元 fixture と同一 bytes の `data-manifest.json`。

準備処理は manifest・archive・LICENSE・JPEG・PNG の固定 hash、クラス順、分割、写真家、photo ID、寸法を確認する。tar の `extractall` は使わず、宣言した通常ファイルだけを読み取る。出力は全件の検証後に公開し、別用途のファイル・symlink・不完全な出力を上書きしない。既存の正常な出力に再実行した場合は、全 PNG と manifest を検証して再利用する。

2026-10-25 に、未実装の状態でデータ境界テスト9件の RED を確認した。その後9/9件が GREEN。固定 manifest、class/split/order/creator/hash/path の拒否、EXIF 適用、正常生成、冪等性、archive/license/JPEG/PNG の破損、宣言ファイルの symlink、無関係な出力、既存出力の改変を検証した。公式アーカイブからの60 PNG の生成と検証済み再実行も成功した。

この資料の時点ではデータ準備が完了しただけで、分類の学習改善・品質・PyTorch 一致・完全 ONNX の実画像結果は未測定。全体の測定状態は [STATUS](../STATUS.md) と当該比較記録を参照する。

## 解釈の範囲

公開 Flickr 写真5クラスの小規模な品質点検であり、ユーザーの独自データ全般に対する品質保証ではない。evaluation は今回の追加学習に未使用という意味で、元 MobileNetV4 の ImageNet-1k 学習・モデル選定との画像重複やドメイン独立を確認したものではない。事前学習については [固定 checkpoint の model card](https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k/blob/331fb803779522b685cf942e15f914fb6741c1eb/README.md) を参照する。
