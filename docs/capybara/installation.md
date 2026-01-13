---
sidebar_position: 2
---

# 基本安裝

Capybara 的 Python 套件名稱為 `capybara-docsaid`，需要 Python 3.10+。

## 透過 PyPI 安裝（核心）

```bash
pip install capybara-docsaid
python -c "import capybara; print(capybara.__version__)"
```

## 啟用可選功能（extras）

Capybara 將推論後端與部分功能拆成 extras，需要時再安裝：

```bash
# ONNXRuntime（CPU）
pip install "capybara-docsaid[onnxruntime]"

# ONNXRuntime（GPU）
pip install "capybara-docsaid[onnxruntime-gpu]"

# OpenVINO
pip install "capybara-docsaid[openvino]"

# TorchScript
pip install "capybara-docsaid[torchscript]"

# 視覺化繪圖（matplotlib/pillow）
pip install "capybara-docsaid[visualization]"

# IPCam demo（flask）
pip install "capybara-docsaid[ipcam]"

# 系統資訊（psutil）
pip install "capybara-docsaid[system]"

# 全部一起裝
pip install "capybara-docsaid[all]"
```

## 系統相依套件（依功能需求安裝）

部分功能會依賴 OS 層級的 codec / 轉檔工具：

- JPEG 讀寫加速（`PyTurboJPEG`）：需要 TurboJPEG library
- HEIC/HEIF（`pillow-heif`）：需要 libheif
- PDF 轉圖（`pdf2image`）：需要 Poppler（`pdftoppm` / `pdftocairo`）
- 影片抽幀：建議安裝 `ffmpeg`（讓 OpenCV 影片讀取更穩定）

### Ubuntu

```bash
sudo apt install ffmpeg libturbojpeg libheif-dev poppler-utils
```

### macOS

```bash
brew install ffmpeg jpeg-turbo libheif poppler
```

## 從 Git 安裝

```bash
pip install git+https://github.com/DocsaidLab/Capybara.git
```

若是本地開發：

```bash
pip install -e .
```

## Windows

目前沒有針對 Windows 做完整測試；建議使用 WSL2 或 Docker。可參考：[**進階安裝**](./advance.md)。

## GPU 注意事項（ONNXRuntime CUDA）

若使用 `onnxruntime-gpu`，請依 ORT 的版本安裝相容的 CUDA/cuDNN：

- https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements
