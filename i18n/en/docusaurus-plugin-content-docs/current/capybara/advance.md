---
sidebar_position: 3
---

# Advanced Installation

This section is based on the built-in `docker/Dockerfile` and describes how to build a reproducible runtime environment with Docker.

## Prerequisites

- If you need GPU inside Docker, install NVIDIA Driver and NVIDIA Container Toolkit first:
  - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- If you only use CPU, omit `--gpus all`.

## What the built-in Dockerfile does

`docker/Dockerfile` will:

- Use `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` as the base image
- Install OS dependencies required by Capybara (e.g. `ffmpeg`, `libturbojpeg`, `poppler-utils`, `libheif-dev`)
- `pip install` this project (core dependencies only; no extras)

## Build the image

```bash
cd Capybara
bash docker/build.bash
```

Default image tag: `capybara_docsaid`.

## Run the container

Interactive shell:

```bash
docker run --gpus all -v ${PWD}:/code -it --rm capybara_docsaid
```

Run an external script (example):

```bash
docker run --gpus all -v ${PWD}:/code -it --rm capybara_docsaid python your_script.py
```

## Install extras inside the container (optional)

The built-in image installs core dependencies only. Install extras in the container as needed:

```bash
pip install "capybara-docsaid[onnxruntime-gpu]"
pip install "capybara-docsaid[openvino]"
pip install "capybara-docsaid[torchscript]"
```

If you use `onnxruntime-gpu`, also check ORT/CUDA/cuDNN compatibility:

- https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements

## File permissions (common)

When running as root, generated files under the mounted directory may be owned by root.

The simplest fix is to use Docker `--user`:

```bash
docker run --user $(id -u):$(id -g) -v ${PWD}:/code -it --rm capybara_docsaid python your_script.py
```

If your workflow needs a full UID/GID mapping (create users, HOME handling, pre-chown, etc.), extend the Dockerfile and implement an entrypoint with `gosu`.
