---
sidebar_position: 3
---

# 進階インストール

## よく参照される資料

環境構築を始める前に、以下の公式ドキュメントを参照することをお勧めします：

- **PyTorch リリースノート**

  NVIDIA が提供する [PyTorch イメージのリリースノート](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html) は、特定のイメージに内蔵されている PyTorch、CUDA、cuDNN のバージョンを確認するのに役立ち、依存関係の競合を減らすことができます。

---

- **NVIDIA ランタイムの前提作業**：

  Docker 内で GPU を使用したい場合は、まず [Installation (Native GPU Support)](<https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(Native-GPU-Support)#usage>) を参照し、ローカルマシンに NVIDIA ドライバとコンテナツールが正しくインストールされていることを確認してください。

---

- **NVIDIA コンテナツールキットのインストール**

  [NVIDIA コンテナツールキットのインストールガイド](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) をしっかりと読むことをお勧めします。これは Docker の GPU アクセラレーションには不可欠です。

---

- **ONNXRuntime リリースノート**

  ONNXRuntime を使用して推論を行う場合、GPU アクセラレーションが必要な場合は、公式の [CUDA Execution Provider の説明](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) を参照し、バージョンの互換性を確認してください。

## 環境のインストール

深層学習に関するプロジェクトでは、依存パッケージに関する問題がよく発生します。一般的な分け方としては：

- **トレーニング環境**：PyTorch、OpenCV、CUDA、cuDNN バージョンを互換させないと、よく「ライブラリが正しく読み込まれない」という問題が発生します。
- **デプロイ環境**：ONNXRuntime、OpenCV、CUDA も適切なバージョンに合わせる必要があります。特に GPU アクセラレーションを使用する場合、ONNXRuntime-CUDA のバージョン要件には注意が必要です。

最も注意すべき点は、**PyTorch-CUDA** と **ONNXRuntime-CUDA** のバージョンが一致しない場合です。こうした場合は、公式でテストされた組み合わせを使うか、依存関係を詳しく調べて CUDA、cuDNN の関係性を理解することが推奨されます。

:::tip
どうしてうまくいかないのか？ 💢 💢 💢
:::

## Docker を使おう！

一貫性と移植性を確保するために、**Docker の使用を強く推奨**します。もしローカルで環境をすでに構築している場合も問題ありませんが、協力開発やデプロイの段階で、余計な衝突を処理するために時間がかかることになります。

### 環境のインストール

```bash
cd Capybara
bash docker/build.bash
```

ビルド用の `Dockerfile` はプロジェクト内にもありますので、興味があれば参考にしてください：[**Capybara Dockerfile**](https://github.com/DocsaidLab/Capybara/blob/main/docker/Dockerfile)

「推論環境」では、`nvcr.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` を基本イメージとして使用しています。

このイメージはモデルのデプロイ専用であり、トレーニング環境のパッケージは含まれていません。PyTorch のようなライブラリは含まれていません。

ユーザーは必要に応じてこのイメージを変更できます。バージョンは ONNXRuntime の更新に合わせて変更されることがあります。

推論に使用されるイメージに関しては、[**NVIDIA NGC**](https://ngc.nvidia.com/catalog/containers/nvidia:cuda) を参照してください。

## 使用方法

以下は一般的な使用シナリオの一例です：Docker を使用して外部スクリプトを実行し、現在のディレクトリをコンテナ内にマウントします。

### 日常的な使用

例えば、`your_scripts.py` というスクリプトを推論コンテナ内で Python で実行したい場合、手順は以下の通りです：

1. 新しい `Dockerfile` を作成（名前は `your_Dockerfile` とします）：

   ```Dockerfile title="your_Dockerfile"
   # syntax=docker/dockerfile:experimental
   FROM capybara_infer_image:latest

   # 作業ディレクトリを設定（必要に応じて変更可能）
   WORKDIR /code

   # 例：DocAligner をインストール
   RUN git clone https://github.com/DocsaidLab/DocAligner.git && \
       cd DocAligner && \
       python setup.py bdist_wheel && \
       pip install dist/*.whl && \
       cd .. && rm -rf DocAligner

   ENTRYPOINT ["python"]
   ```

2. イメージをビルド：

   ```bash
   docker build -f your_Dockerfile -t your_image_name .
   ```

3. 実行スクリプト（例えば `run_in_docker.sh`）を作成：

   ```bash
   #!/bin/bash
   docker run \
       --gpus all \
       -v ${PWD}:/code \
       -it --rm your_image_name your_scripts.py
   ```

4. `run_in_docker.sh` を実行し、推論を開始します。

:::tip
コンテナに入って bash を起動したい場合は、`ENTRYPOINT ["python"]` を `ENTRYPOINT ["/bin/bash"]` に変更してください。
:::

### gosu の設定の導入

実際の作業では「コンテナ内で生成されたファイルの属性が root になってしまう」という問題によく直面します。

複数のエンジニアが同じディレクトリを共有している場合、その後の権限変更が面倒になることがあります。

この問題は `gosu` を使って解決できます。以下は Dockerfile の例です：

```Dockerfile title="your_Dockerfile"
# syntax=docker/dockerfile:experimental
FROM capybara_infer_image:latest

WORKDIR /code

# 例：DocAligner のインストール
RUN git clone https://github.com/DocsaidLab/DocAligner.git && \
    cd DocAligner && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && \
    cd .. && rm -rf DocAligner

ENV ENTRYPOINT_SCRIPT=/entrypoint.sh

# gosu のインストール
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
    # 引数があるかチェック\n\
    if [ $# -gt 0 ]; then\n\
        exec gosu ${USER_ID:-0}:${GROUP_ID:-0} python "$@"\n\
    else\n\
        exec gosu ${USER_ID:-0}:${GROUP_ID:-0} bash\n\
    fi' > "$ENTRYPOINT_SCRIPT"

RUN chmod +x "$ENTRYPOINT_SCRIPT"

ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
```

#### イメージのビルドと実行

1. イメージをビルド：

   ```bash
   docker build -f your_Dockerfile -t your_image_name .
   ```

2. イメージを実行（GPU 加速の例）：

   ```bash
   #!/bin/bash
   docker run \
       -e USER_ID=$(id -u) \
       -e GROUP_ID=$(id -g) \
       --gpus all \
       -v ${PWD}:/code \
       -it --rm your_image_name your_scripts.py
   ```

これで、出力されたファイルは自動的に現在のユーザー権限で保存され、後での読み書きが簡単になります。

### 内部パッケージのインストール

もし**プライベートパッケージ**や**内部ツール**（例えば、プライベートな PyPI サーバーから）をインストールする必要がある場合は、ビルド時に認証情報を渡すことができます：

```Dockerfile title="your_Dockerfile"
# syntax=docker/dockerfile:experimental
FROM capybara_infer_image:latest

WORKDIR /code

ARG PYPI_ACCOUNT
ARG PYPI_PASSWORD

# 内部パッケージのソースを指定
ENV SERVER_IP=192.168.100.100:28080/simple/

RUN python -m pip install \
    --trusted-host 192.168.100.100 \
    --index-url http://${PYPI_ACCOUNT}:${PYPI_PASSWORD}@192.168.100.100:16000/simple docaligner

ENTRYPOINT ["python"]
```

その後、ビルドコマンドでアカウント情報を渡します：

```bash
docker build \
    -f your_Dockerfile \
    --build-arg PYPI_ACCOUNT=your_account \
    --build-arg PYPI_PASSWORD=your_password \
    -t your_image_name .
```

`pip.conf` にアカウント情報が保存されている場合、文字列解析で引き渡すこともできます：

```bash
docker build \
    -f your_Dockerfile \
    --build-arg PYPI_PASSWORD=$(awk -F '://|@' '/index-url/{print $2}' your/config/path/pip.conf | cut -d: -f2) \
    -t your_image_name .
```

ビルド後、上記のようにコマンドを Docker 内で実行するだけです。

## よくある問題：権限が足りない

コマンドラインで「Permission denied」と表示される場合、これは非常に悩ましい問題です。

`gosu` を使ってユーザーを切り替えた後、権限が制限されるため、コンテナ内のファイルに対する読み書き権限が不足している場合があります。

例えば、`DocAligner` パッケージをインストールした場合、このパッケージはモデルを起動する際に自動的にモデルファイルをダウンロードし、Python 関連のディレクトリに保存します。

上記の例では、モデルファイルのデフォルト保存場所は以下のようになります：

- `/usr/local/lib/python3.10/dist-packages/docaligner/heatmap_reg/ckpt`

このパスは明らかにユーザー権限の範囲外です！

そのため、コンテナを起動する際に、このパスの権限をユーザーに付与する必要があります。以下のように Dockerfile を変更してください：

```Dockerfile title="your_Dockerfile" {23}
# syntax=docker/dockerfile:experimental
FROM capybara_infer_image:latest

WORKDIR /code

RUN git clone https://github.com/DocsaidLab/DocAligner.git && \
    cd DocAligner && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && \
    cd .. && rm -rf DocAligner

ENV ENTRYPOINT_SCRIPT=/entrypoint.sh

RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

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
    if [ $# -gt 0 ]; then\n\
        exec gosu ${USER_ID:-0}:${GROUP_ID:-0} python "$@"\n\
    else\n\
        exec gosu ${USER_ID:-0}:${GROUP_ID:-0} bash\n\
    fi' > "$ENTRYPOINT_SCRIPT"

RUN chmod +x "$ENTRYPOINT_SCRIPT"

ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
```

特定のディレクトリだけに権限を付与する場合は、そのパスを指定して権限を変更し、過度に権限を開放しないようにしてください。

## まとめ

Docker を使うことには学習コストがかかりますが、環境の一貫性を保ち、デプロイや協力開発の段階で不要なトラブルを大幅に減らすことができます。

この投資は確実に価値があるものです。ぜひその便利さを享受してください！
