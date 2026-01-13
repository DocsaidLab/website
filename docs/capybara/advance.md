---
sidebar_position: 3
---

# 進階安裝

本章節以專案內建的 `docker/Dockerfile` 為準，說明如何用 Docker 建立可重現的執行環境。

## 前置條件

- 若需要在 Docker 內使用 GPU，請先完成 NVIDIA Driver 與 NVIDIA Container Toolkit 的安裝：
  - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- 若只使用 CPU，可不加 `--gpus all`。

## 內建 Dockerfile 的行為

`docker/Dockerfile` 會：

- 以 `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` 作為基底映像
- 安裝 Capybara 需要的 OS 依賴（例如 `ffmpeg`、`libturbojpeg`、`poppler-utils`、`libheif-dev`）
- `pip install` 本專案（僅核心依賴；不包含任何 extras）

## 建置 image

```bash
cd Capybara
bash docker/build.bash
```

預設 image tag：`capybara_docsaid`。

## 執行容器

互動式進入：

```bash
docker run --gpus all -v ${PWD}:/code -it --rm capybara_docsaid
```

直接執行外部腳本（範例）：

```bash
docker run --gpus all -v ${PWD}:/code -it --rm capybara_docsaid python your_script.py
```

## 在容器內安裝 extras（可選）

內建 image 只安裝核心依賴；若需要推論後端或額外功能，請在容器內自行安裝：

```bash
pip install "capybara-docsaid[onnxruntime-gpu]"
pip install "capybara-docsaid[openvino]"
pip install "capybara-docsaid[torchscript]"
```

若要使用 `onnxruntime-gpu`，請同時確認 ORT 與 CUDA/cuDNN 的相容性：

- https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements

## 檔案權限（常見）

預設以 root 身份執行容器時，掛載目錄內產生的檔案可能會是 root 擁有者。

最直接的處理方式是使用 Docker 的 `--user`：

```bash
docker run --user $(id -u):$(id -g) -v ${PWD}:/code -it --rm capybara_docsaid python your_script.py
```

若你的 workflow 需要更完整的 UID/GID 映射（例如在容器內建立使用者、處理 HOME、預先調整特定目錄權限），可自行擴充 Dockerfile 並以 `gosu` 實作 entrypoint。
