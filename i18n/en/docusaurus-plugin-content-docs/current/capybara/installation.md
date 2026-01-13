---
sidebar_position: 2
---

# Installation

The Python package name of Capybara is `capybara-docsaid`. Python 3.10+ is required.

## Install via PyPI (core)

```bash
pip install capybara-docsaid
python -c "import capybara; print(capybara.__version__)"
```

## Enable optional features (extras)

Capybara splits inference backends and some features into extras:

```bash
# ONNXRuntime (CPU)
pip install "capybara-docsaid[onnxruntime]"

# ONNXRuntime (GPU)
pip install "capybara-docsaid[onnxruntime-gpu]"

# OpenVINO
pip install "capybara-docsaid[openvino]"

# TorchScript
pip install "capybara-docsaid[torchscript]"

# Visualization (matplotlib/pillow)
pip install "capybara-docsaid[visualization]"

# IPCam demo (flask)
pip install "capybara-docsaid[ipcam]"

# System info (psutil)
pip install "capybara-docsaid[system]"

# Install everything
pip install "capybara-docsaid[all]"
```

## System dependencies (install as needed)

Some features rely on OS-level codecs / conversion tools:

- JPEG acceleration (`PyTurboJPEG`): TurboJPEG library
- HEIC/HEIF (`pillow-heif`): libheif
- PDF to images (`pdf2image`): Poppler (`pdftoppm` / `pdftocairo`)
- Video frame extraction: recommended to install `ffmpeg` (more stable OpenCV video I/O)

### Ubuntu

```bash
sudo apt install ffmpeg libturbojpeg libheif-dev poppler-utils
```

### macOS

```bash
brew install ffmpeg jpeg-turbo libheif poppler
```

## Install from Git

```bash
pip install git+https://github.com/DocsaidLab/Capybara.git
```

For local development:

```bash
pip install -e .
```

## Windows

Windows is not fully tested at the moment. Use WSL2 or Docker instead. See: [**Advanced Installation**](./advance.md).

## GPU note (ONNXRuntime CUDA)

If you use `onnxruntime-gpu`, install a compatible CUDA/cuDNN version based on the ORT version:

- https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements
