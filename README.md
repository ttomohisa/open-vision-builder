# Open Vision Builder

A browser-based vision model builder for image classification and object detection, with local training and ONNX export.

**「Custom Vision が好きだった人向けの、もっとオープンな Vision Model Builder」**

画像の取り込み、アノテーション、追加学習、評価、モデル出力をブラウザで行うアプリを目指します。ユーザーの計算資源で学習し、生成したモデルは Python、ユーザーの API サーバー、対応するエッジ端末などへ持ち出せる構成にします。

## 現在の状態

**設計・技術検証前の段階です。現在のリポジトリには設計文書があり、実行できるアプリや学習済みの成果物はまだありません。**

ブラウザでの追加学習、ONNX 出力の一致、学習時間、メモリ、検出精度は未検証です。GitHub Pages の動作確認用アプリもまだ公開していません。実装状況は [STATUS](docs/STATUS.md) を参照してください。

## 製品の方針

- Cloudflare の静的ホスティングで配信できるアプリにする。
- 運営側のアカウントサービス、API サーバー、DB、学習サーバーを設けない。
- ユーザーの画像、ラベル、学習処理、学習後の重みをブラウザ内で扱う。
- 画像分類と物体検出の両方を対象にする。
- 転移学習・Fine-tuning を中心にし、物体検出は位置推定まで事前学習済みのモデルを使う。
- **初期版から、学習後の重みを含む ONNX 出力を必須にする。**
- 商用利用・必要な再配布条件を確認したコードとモデルの重みを採用する。
- モデルと実行例・配布用キットを提供し、生成モデルのホスティングはユーザーが行う。

Browser Kitty とは独立したプロジェクトです。

## モデルと学習エンジンの候補

| 用途 | 品質を重視した候補 | 軽量・検証用の候補 |
| --- | --- | --- |
| 画像分類 | MobileNetV4 Conv Medium | MobileNetV4 Conv Small |
| 物体検出 | D-FINE Small の COCO 学習済み重み | D-FINE Nano の COCO 学習済み重み |

これらは採用候補です。対応モデル・端末の確定は、[技術検証の合格条件](docs/VERIFICATION.md) を満たしてから行います。正確な重みの配布元とライセンスの確認事項は [MODELS](docs/MODELS.md) に記録します。

学習エンジンは未決定です。jax-js を最初の検証候補とし、TensorFlow.js でモデルを明示的に実装する経路も評価します。ONNX 推論が動いたことだけで、Fine-tuning も動いたと扱いません。

## 開発に参加する前に

まず [AGENTS.md](AGENTS.md) を読み、次の文書を確認してください。会話の記憶だけに依存せず、作業開始時にリポジトリ内の記録を参照します。

| 文書 | 役割 |
| --- | --- |
| [PRODUCT](docs/PRODUCT.md) | 合意済みの製品要件と初期版の範囲 |
| [ARCHITECTURE](docs/ARCHITECTURE.md) | 静的アプリ、学習、保存、ONNX 出力、配信の構成 |
| [MODELS](docs/MODELS.md) | モデル候補、重みの出典、ライセンス、採用条件 |
| [VERIFICATION](docs/VERIFICATION.md) | 学習・精度・ONNX 互換性を検証する合格条件 |
| [ROADMAP](docs/ROADMAP.md) | PR ごとの導入順序 |
| [STATUS](docs/STATUS.md) | 実装済み・未検証・次の作業 |
| [設計判断](docs/decisions/README.md) | 方針を選んだ理由と変更履歴 |

変更は専用ブランチから PR として提出し、マージはリポジトリ所有者が行います。設計や実装状況が変わる PR では、関連文書も更新します。

## ライセンス

アプリ本体のライセンスは未選定です。公開リポジトリであることを、特定の利用許諾が付与済みであるという意味に扱わないでください。モデル・依存コード・データの条件はそれぞれ確認し、採用した配布物に必要なライセンス表示を付けます。
