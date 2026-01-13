---
sidebar_position: 1
---

# イントロダクション

Capybara は以下の要素で構成されています（現行のコード構造に基づく）：

- **Vision**（`capybara.vision`）：画像／動画の I/O と処理。
- **Structures**（`capybara.structures`）：`Box/Boxes`、`Polygon/Polygons`、`Keypoints` などの幾何構造。
- **Runtime**（`capybara.runtime`）：runtime/backend のレジストリと選択ロジック。
- **推論エンジン（optional）**：
  - `capybara.onnxengine`（ONNXRuntime）
  - `capybara.openvinoengine`（OpenVINO）
  - `capybara.torchengine`（TorchScript）
- **Utils**（`capybara.utils`）：ユーティリティ（パス、ダウンロード、時間など）。
- **Extras（optional）**：
  - `visualization`：描画ユーティリティ（`capybara.vision.visualization`）
  - `ipcam`：簡易 Web demo（`capybara.vision.ipcam`）
  - `system`：システム情報ユーティリティ（`capybara.utils.system_info`）

## Vision

Vision モジュールは画像／動画の処理と I/O を扱います。

ディレクトリ構成：

```
vision
├── functionals.py       # 基本的な画像処理（フィルタ、変換など）
├── geometric.py         # 幾何処理（回転、リサイズなど）
├── improc.py            # 画像 I/O と補助処理
├── morphology.py        # 形態学処理（膨張、収縮など）
├── videotools           # 動画ツール
├── ipcam                # （extra: ipcam）IPCam demo
└── visualization        # （extra: visualization）描画／可視化
```

主な機能：

- 画像／動画の読み取り、処理、可視化。
- 複数の入力ソース（ローカルファイル、動画抽出、IPCam demo など）に対応。

## Structures

Structures モジュールは構造化データ（幾何情報）を扱います。

ディレクトリ構成：

```
structures
├── functionals.py       # 関連機能
├── boxes.py             # Box / Boxes
├── keypoints.py         # Keypoints
└── polygons.py          # Polygon / Polygons
```

主な機能：

- Boxes/Keypoints/Polygons などの構造化データ処理。
- intersection、IoU、scale などの操作。

## Runtime / 推論エンジン（optional）

推論関連機能は独立モジュールとして提供され、`capybara.runtime` は runtime/backend の統一的な表現と選択ロジックを提供します。

注意：推論 backend は optional dependency です。必要な extras を先にインストールしてください（例：`capybara-docsaid[onnxruntime]`）。

### capybara.runtime

- `Runtime` / `Backend` を定義し、`auto_backend_name()` などの選択ヘルパーを提供します。

### capybara.onnxengine

ディレクトリ構成：

```
onnxengine
├── engine.py            # コア推論ロジック
├── __init__.py          # 初期化
├── metadata.py          # モデル metadata
└── utils.py             # ONNX helper
```

主な機能：

- ONNX モデルのロードと推論。

### capybara.openvinoengine

- OpenVINO 推論 wrapper（同期推論 + optional async queue）。

### capybara.torchengine

- TorchScript 推論 wrapper（簡易 dtype/device 正規化）。

## Utils

Utils モジュールは補助ユーティリティを提供します。

ディレクトリ構成：

```
utils
├── custom_path.py       # パス操作
├── custom_tqdm.py       # 進捗バー
├── files_utils.py       # ファイルユーティリティ
├── powerdict.py         # 拡張 dict
├── system_info.py       # システム情報（extra: system）
├── time.py              # 時間ユーティリティ
└── utils.py             # 汎用ユーティリティ
```

主な機能：

- ファイル操作／システム情報取得。
- 進捗バーや拡張 dict などの補助ツール。

## Tests

Tests は機能検証のためのテスト群です。

---

API の使用方法は各ドキュメントを参照してください。import エラーが出る場合は、対応する extra がインストールされているかを確認してください。

