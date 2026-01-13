---
sidebar_position: 3
---

# 進階インストール

この章では、プロジェクトに含まれる `docker/Dockerfile` を基準に、Docker で再現可能な実行環境を作る手順をまとめます。

## 前提条件

- Docker 内で GPU を使う場合は、NVIDIA Driver と NVIDIA Container Toolkit を先にインストールしてください：
  - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- CPU のみの場合は `--gpus all` は不要です。

## 内蔵 Dockerfile の挙動

`docker/Dockerfile` は以下を行います：

- ベースイメージ：`nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`
- OS 依存（例：`ffmpeg`、`libturbojpeg`、`poppler-utils`、`libheif-dev`）をインストール
- 本プロジェクトを `pip install`（core のみ。extras は含まれません）

## image のビルド

```bash
cd Capybara
bash docker/build.bash
```

デフォルトの image tag：`capybara_docsaid`。

## コンテナの実行

インタラクティブに入る：

```bash
docker run --gpus all -v ${PWD}:/code -it --rm capybara_docsaid
```

外部スクリプトを実行（例）：

```bash
docker run --gpus all -v ${PWD}:/code -it --rm capybara_docsaid python your_script.py
```

## コンテナ内で extras を追加インストール（optional）

内蔵 image は core のみです。推論 backend などが必要な場合は、コンテナ内で追加インストールしてください：

```bash
pip install "capybara-docsaid[onnxruntime-gpu]"
pip install "capybara-docsaid[openvino]"
pip install "capybara-docsaid[torchscript]"
```

`onnxruntime-gpu` を使用する場合、ORT と CUDA/cuDNN の互換性も確認してください：

- https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements

## ファイル権限（よくある問題）

root でコンテナを実行すると、マウント先に作られるファイルの owner が root になることがあります。

最も簡単な対処は Docker の `--user` を使うことです：

```bash
docker run --user $(id -u):$(id -g) -v ${PWD}:/code -it --rm capybara_docsaid python your_script.py
```

より厳密な UID/GID マッピング（コンテナ内ユーザー作成、HOME の扱い、特定ディレクトリの権限調整など）が必要な場合は、Dockerfile を拡張して `gosu` で entrypoint を実装してください。

