---
sidebar_position: 2
---

# 基本インストール

Capybara の PyPI パッケージ名は `capybara-docsaid` です。Python 3.10+ が必要です。

## PyPI からインストール（core）

```bash
pip install capybara-docsaid
python -c "import capybara; print(capybara.__version__)"
```

## optional 機能（extras）

推論 backend と一部機能は extras として分離されています。必要に応じて追加インストールしてください：

```bash
# ONNXRuntime（CPU）
pip install "capybara-docsaid[onnxruntime]"

# ONNXRuntime（GPU）
pip install "capybara-docsaid[onnxruntime-gpu]"

# OpenVINO
pip install "capybara-docsaid[openvino]"

# TorchScript
pip install "capybara-docsaid[torchscript]"

# 可視化描画（matplotlib/pillow）
pip install "capybara-docsaid[visualization]"

# IPCam demo（flask）
pip install "capybara-docsaid[ipcam]"

# システム情報（psutil）
pip install "capybara-docsaid[system]"

# 全部
pip install "capybara-docsaid[all]"
```

## システム依存パッケージ（機能に応じて）

一部機能は OS 依存の codec / 変換ツールに依存します：

- JPEG の高速 I/O（`PyTurboJPEG`）：TurboJPEG library
- HEIC/HEIF（`pillow-heif`）：libheif
- PDF → 画像（`pdf2image`）：Poppler（`pdftoppm` / `pdftocairo`）
- 動画抽出：`ffmpeg`（OpenCV の動画読み取りが安定しやすい）

### Ubuntu

```bash
sudo apt install ffmpeg libturbojpeg libheif-dev poppler-utils
```

### macOS

```bash
brew install ffmpeg jpeg-turbo libheif poppler
```

## Git からインストール

```bash
pip install git+https://github.com/DocsaidLab/Capybara.git
```

ローカル開発の場合：

```bash
pip install -e .
```

## Windows

Windows での動作は十分にテストしていません。WSL2 または Docker を推奨します： [**進階インストール**](./advance.md)。

## GPU 注意事項（ONNXRuntime CUDA）

`onnxruntime-gpu` を使用する場合、ORT のバージョンに対応する CUDA/cuDNN をインストールしてください：

- https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements

