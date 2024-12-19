---
sidebar_position: 3
---

# 高度なインストール

## よく使われる参考資料

- NVIDIA によって構築された PyTorch イメージの各バージョンの詳細については、以下を参照してください：[**PyTorch リリースノート**](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html)

- NVIDIA ランタイムの準備については、以下を参照してください：[**インストール（ネイティブ GPU サポート）**](<https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(Native-GPU-Support)#usage>)

- NVIDIA Toolkit のインストール方法については、以下を参照してください：[**NVIDIA コンテナツールキットのインストール**](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

- ONNXRuntime に関する内容については、以下を参照してください：[**ONNX Runtime リリースノート**](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)

## 環境のインストール

私たちの作業環境はそれほど複雑ではありませんが、いくつかのパッケージ互換性の問題に遭遇することもあります。

簡単に言うと、通常は次のように分けられます：

- **トレーニング環境**：PyTorch、OpenCV、CUDA、cuDNN が互いに連携する必要があります。
- **デプロイ環境**：ONNXRuntime、OpenCV、CUDA が互いに連携する必要があります。

その中で最もよく発生する衝突は、PyTorch-CUDA と ONNXRuntime-CUDA のバージョンの問題です。

:::tip
なぜこれらはいつも一致しないのでしょうか？ 💢 💢 💢
:::

## Docker を使いましょう！

私たちはすべて Docker を使用してインストールを行い、環境の一貫性を確保しています、例外はありません。

Docker を使用することで、環境を調整する時間を大幅に節約でき、不要な問題を回避できます。

関連する環境は開発中にも継続的にテストしており、以下のコマンドを使用するだけです：

### 推論環境のインストール

```bash
cd Capybara
bash docker/build.bash
```

「推論環境」では、`nvcr.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`をベースイメージとして使用しています。

このイメージはモデルをデプロイするために特化しているため、トレーニング環境のパッケージは含まれておらず、PyTorch のようなパッケージは含まれていません。

ユーザーは自身のニーズに応じて変更できます。関連するバージョンは ONNXRuntime の更新に伴い変更されることがあります。

推論シリーズ用のイメージについては、以下を参照してください：[**NVIDIA NGC**](https://ngc.nvidia.com/catalog/containers/nvidia:cuda)

## 使用方法

一般的に、私たちはこのモジュールを`DocAligner`のようなプロジェクトと組み合わせて使用します。

### 日常的な使用

以下に例を示します。仮に`your_scripts.py`というファイルがあり、このファイルを Python で実行する必要があるとします。

推論環境のインストールが完了していると仮定し、次に`Dockerfile`を作成します：

```Dockerfile title="your_Dockerfile"
# syntax=docker/dockerfile:experimental
FROM capybara_infer_image:latest

# 作業ディレクトリを設定します。ユーザーは自分のニーズに応じて変更できます。
WORKDIR /code

# 例：DocAlignerをインストール
RUN git clone https://github.com/DocsaidLab/DocAligner.git && \
    cd DocAligner && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && \
    cd .. && rm -rf DocAligner

ENTRYPOINT ["python"]
```

次にこのイメージを作成します：

```bash
docker build -f your_Dockerfile -t your_image_name .
```

作成後、使用する際は以下のようにコマンドを Docker 内で実行します：

```bash
#!/bin/bash
docker run \
    --gpus all \
    -v ${PWD}:/code \
    -it --rm your_image_name your_scripts.py
```

これで、ラップされた Python 環境を直接呼び出すことになり、環境の一貫性も保証されます。

:::tip
もし Python を起動せずにコンテナに入ることを望む場合は、エントリーポイントを`/bin/bash`に変更してください。

```Dockerfile
ENTRYPOINT ["/bin/bash"]
```

:::

### gosu 設定の導入

もし Docker を実行している際に権限の問題が発生した場合：

- **例えば、コンテナ内でファイルや画像を出力すると、その権限が root:root になっており、変更や削除が面倒になることがあります！**

その場合、`gosu`というツールを使うことをお勧めします。

`gosu`を使用する方法に基づいて、元の Dockerfile を以下のように変更します：

```Dockerfile title="your_Dockerfile"
# syntax=docker/dockerfile:experimental
FROM capybara_infer_image:latest

# 作業ディレクトリを設定（ユーザーはニーズに応じて変更可能）
WORKDIR /code

# 例：DocAlignerをインストール
RUN git clone https://github.com/DocsaidLab/DocAligner.git && \
    cd DocAligner && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && \
    cd .. && rm -rf DocAligner

# エントリーポイントスクリプトのパスを設定
ENV ENTRYPOINT_SCRIPT=/entrypoint.sh

# gosuをインストール
RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

# エントリーポイントスクリプトを作成
RUN printf '#!/bin/bash\n\
    if [ ! -z "$USER_ID" ] && [ ! -z "$GROUP_ID" ]; then\n\
    groupadd -g "$GROUP_ID" -o usergroup\n\
    useradd --shell /bin/bash -u "$USER_ID" -g "$GROUP_ID" -o -c "" -m user\n\
    export HOME=/home/user\n\
    chown -R "$USER_ID":"$GROUP_ID" /home/user\n\
    chown -R "$USER_ID":"$GROUP_ID" /code\n\
    fi\n\
    \n\
    # 引数があるかどうか確認\n\
    if [ $# -gt 0 ]; then\n\
    exec gosu ${USER_ID:-0}:${GROUP_ID:-0} python "$@"\n\
    else\n\
    exec gosu ${USER_ID:-0}:${GROUP_ID:-0} bash\n\
    fi' > "$ENTRYPOINT_SCRIPT"

# 権限を付与
RUN chmod +x "$ENTRYPOINT_SCRIPT"

# エントリーポイント
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
```

次にこのイメージを作成します：

```bash
docker build -f your_Dockerfile -t your_image_name .
```

作成後、使用する際には以下のようにコマンドを Docker 内で実行します：

```bash
#!/bin/bash
docker run \
    -e USER_ID=$(id -u) \
    -e GROUP_ID=$(id -g) \
    --gpus all \
    -v ${PWD}:/code \
    -it --rm your_image_name your_scripts.py
```

### 内部パッケージのインストール

もしイメージのビルド時に内部パッケージをインストールする必要がある場合、環境変数を追加で渡す必要があります。

```Dockerfile title="your_Dockerfile"
# syntax=docker/dockerfile:experimental
FROM capybara_infer_image:latest

# 作業ディレクトリを設定（ユーザーはニーズに応じて変更可能）
WORKDIR /code

# 例：DocAlignerのインストール（内部パッケージの仮定）

# 環境変数を追加
ARG PYPI_ACCOUNT
ARG PYPI_PASSWORD

# 内部パッケージソースを設定
ENV SERVER_IP=192.168.100.100:28080/simple/

# docalignerのインストール
# パッケージサーバーのアドレスを変更する必要があります
RUN python -m pip install \
    --trusted-host 192.168.100.100 \
    --index-url http://${PYPI_ACCOUNT}:${PYPI_PASSWORD}@192.168.100.100:16000/simple docaligner

ENTRYPOINT ["python"]
```

次にこのイメージを作成します：

```bash
docker build \
    -f your_Dockerfile \
    --build-arg PYPI_ACCOUNT=your_account \
    --build-arg PYPI_PASSWORD=your_password \
    -t your_image_name .
```

もしアカウントとパスワードが別の場所に保存されている場合（例えば`pip.conf`ファイルに）、文字列解析を用いて環境変数を渡すこともできます：

```bash
docker build \
    -f your_Dockerfile \
    --build-arg PYPI_PASSWORD=$(awk -F '://|@' '/index-url/{print $2}' your/config/path/pip.conf | cut -d: -f2) \
    -t your_image_name .
```

作成後、使用する際は同じ方法でコマンドを Docker 内で実行できます。

## よくある質問

### Permission denied

`gosu`を使用してユーザーを切り替えた後、権限が制限されている範囲内で、コンテナ内のファイルを読み書きする際に権限の問題が発生することがあります。

例えば、`DocAligner`パッケージをインストールした場合、このパッケージはモデルを起動する際に自動的にモデルファイルをダウンロードし、Python 関連のフォルダに格納します。

上記の例では、モデルファイルのデフォルト保存場所は次の通りです：

- `/usr/local/lib/python3.10/dist-packages/docaligner/heatmap_reg/ckpt`

このパスは明らかにユーザーの権限範囲外です！

したがって、コンテナを起動する際に、このパスをユーザーに付与する必要があります。以下のように Dockerfile を変更してください：

```Dockerfile title="your_Dockerfile" {28}
# syntax=docker/dockerfile:experimental
FROM capybara_infer_image:latest

# 作業ディレクトリを設定（ユーザーはニーズに応じて変更可能）
WORKDIR /code

# 例：DocAlignerのインストール
RUN git clone https://github.com/DocsaidLab/DocAligner.git && \
    cd DocAligner && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && \
    cd .. && rm -rf DocAligner

# エントリーポイントスクリプトのパスを設定
ENV ENTRYPOINT_SCRIPT=/entrypoint.sh

# gosuをインストール
RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

# エントリーポイントスクリプトを作成
RUN printf '#!/bin/bash\n\
    if [ ! -z "$USER_ID" ] && [ ! -z "$GROUP_ID" ]; then\n\
    groupadd -g "$GROUP_ID" -o usergroup\n\
    useradd --shell /bin/bash -u "$USER_ID" -g "$GROUP_ID" -o -c "" -m user\n\
    export HOME=/home/user\n\
    chown -R "$USER_ID":"$GROUP_ID" /home/user\n\
    chown -R "$USER_ID":"$GROUP_ID" /code\n\
    chmod -R 777 /usr/local/lib/python3.10/dist-packages\n\
    fi\n\
    \n\
    # 引数があるかどうか確認\n\
    if [ $# -gt 0 ]; then\n\
    exec gosu ${USER_ID:-0}:${GROUP_ID:-0} python "$@"\n\
    else\n\
    exec gosu ${USER_ID:-0}:${GROUP_ID:-0} bash\n\
    fi' > "$ENTRYPOINT_SCRIPT"

# 権限を付与
RUN chmod +x "$ENTRYPOINT_SCRIPT"

# エントリーポイント
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
```

この方法で、他の似たような問題も解決できます。
