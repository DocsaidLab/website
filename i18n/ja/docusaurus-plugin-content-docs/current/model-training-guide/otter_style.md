---
slug: otter-style
title: Otter Style
authors: Zephyr
---

## Pytorch-Lightning に基づく

本章では、Zephyr が担当するプロジェクトの構築方法について説明します。主に Pytorch-Lightning 訓練フレームワークを基盤としています。

:::info
詳細な実装内容は、以下を参照してください：[**DocsaidLab/Otter**](https://github.com/DocsaidLab/Otter)。

「Otter」という名前の由来について特別な意味はなく、区別するための名前にすぎません。😅
:::

## 環境の構築

以下のセクションでは、`DocClassifier` プロジェクトを例にとり、モデル訓練環境を構築する方法を説明します。この内容は、`DocAligner` や `MRZScanner` など、Zephyr が担当する他のプロジェクトにも応用可能です。

:::info
深層学習プロジェクトの多くは推論モジュールのみが公開されています。現在のところ、`DocClassifier` のみが訓練モジュールを公開しています。他のプロジェクトの訓練モジュールが必要な場合、本章の訓練方法を参考にして独自に実装することが可能です。

`DocClassifier` プロジェクトの詳細はこちら：[**DocClassifier github**](https://github.com/DocsaidLab/DocClassifier)
:::

まず、git を使用して [**Otter**](https://github.com/DocsaidLab/Otter) モジュールをダウンロードし、Docker イメージを構築します：

```bash
git clone https://github.com/DocsaidLab/Otter.git
cd Otter
bash docker/build.bash
```

ビルドスクリプトの内容は以下の通りです：

```bash title="Otter/docker/build.bash"
docker build \
    -f docker/Dockerfile \
    -t otter_base_image .
```

ファイル内で `otter_base_image` を任意の名前に置き換えることができます。この名前は後続の訓練で使用されます。

:::info
PyTorch Lightning は、PyTorch に基づく軽量な深層学習フレームワークであり、モデル訓練プロセスを簡素化することを目的としています。研究コード（モデル定義、前向き/後ろ向き伝播、オプティマイザ設定など）とエンジニアリングコード（訓練ループ、ログ記録、チェックポイント保存など）を分離することで、研究者が煩雑なエンジニアリングの詳細を処理することなく、モデルそのものに集中できるようにします。

興味のある方は以下のリソースをご覧ください：

- [**PyTorch Lightning 公式サイト**](https://lightning.ai/)
- [**PyTorch Lightning GitHub**](https://github.com/Lightning-AI/pytorch-lightning)
  :::

:::tip
`Otter` モジュールについて簡単に説明します：

以下のような基本的なモジュールが含まれています：

1. `BaseMixin`：基本的な訓練モデルで、訓練の基本設定を含みます。
2. `BorderValueMixin` と `FillValueMixin`：画像の拡張に使用されるパディングモード。
3. `build_callback`：コールバック関数を構築するためのモジュール。
4. `build_dataset`：データセットを構築するためのモジュール。
5. `build_logger`：ログ記録を構築するためのモジュール。
6. `build_trainer`：トレーナーを構築するためのモジュール。
7. `load_model_from_config`：設定ファイルからモデルを読み込むためのモジュール。

一部にはシステム情報を記録する機能が組み込まれており、特定の設定ファイル形式を使用しなければ正しく動作しません。

特に学ぶ必要はありません。経験上、各エンジニアが独自のモデル訓練方法を進化させており、これは無数にある可能性の一つにすぎません。参考程度にご覧ください。
:::

以下は、モデル訓練専用に設計された [**Dockerfile**](https://github.com/DocsaidLab/Otter/blob/main/docker/Dockerfile) です：

```dockerfile title="Otter/docker/Dockerfile"
# syntax=docker/dockerfile:experimental
FROM nvcr.io/nvidia/pytorch:24.12-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONWARNINGS="ignore" \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Taipei

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    tzdata wget git libturbojpeg exiftool ffmpeg poppler-utils libpng-dev \
    libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev gcc \
    libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk \
    python3-pip libharfbuzz-dev libfribidi-dev libxcb1-dev libfftw3-dev \
    libpq-dev python3-dev gosu && \
    ln -sf /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir -U pip setuptools wheel

COPY . /usr/local/otter
RUN cd /usr/local/otter && \
    python setup.py bdist_wheel && \
    python -m pip install dist/*.whl && \
    cd ~ && rm -rf /usr/local/otter

RUN python -m pip install --no-cache-dir -U \
    tqdm colored ipython tabulate tensorboard scikit-learn fire \
    albumentations "Pillow>=10.0.0" fitsne opencv-fixer prettytable

RUN python -c "from opencv_fixer import AutoFix; AutoFix()"
RUN python -c "import capybara; import chameleon"

WORKDIR /code

ENV ENTRYPOINT_SCRIPT=/entrypoint.sh

RUN printf '#!/bin/bash\n\
    if [ ! -z "$USER_ID" ] && [ ! -z "$GROUP_ID" ]; then\n\
    groupadd -g "$GROUP_ID" -o usergroup\n\
    useradd --shell /bin/bash -u "$USER_ID" -g "$GROUP_ID" -o -c "" -m user\n\
    export HOME=/home/user\n\
    chown -R "$USER_ID":"$GROUP_ID" /home/user\n\
    chown -R "$USER_ID":"$GROUP_ID" /code\n\
    exec gosu "$USER_ID":"$GROUP_ID" "$@"\n\
    else\n\
    exec "$@"\n\
    fi' > "$ENTRYPOINT_SCRIPT" && \
    chmod +x "$ENTRYPOINT_SCRIPT"

ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]

CMD ["bash"]
```

上記の Dockerfile を基に、画像処理や機械学習関連の作業に適した、多様なツールとライブラリを含む深層学習コンテナを構築できます。

以下に重要な部分を説明します：

---

```dockerfile
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONWARNINGS="ignore" \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Taipei
```

環境変数の設定：

- **`PYTHONDONTWRITEBYTECODE=1`**: `.pyc` のコンパイルファイル生成を防ぎ、不必要なファイルの作成を抑制します。
- **`PYTHONWARNINGS="ignore"`**: Python の警告を無視します。
- **`DEBIAN_FRONTEND=noninteractive`**: 対話型プロンプトを無効化し、自動化されたデプロイを可能にします。
- **`TZ=Asia/Taipei`**: タイムゾーンを台北に設定します。

:::tip
お好みのタイムゾーンに変更したり、他の環境変数を追加することも可能です。
:::

---

```dockerfile
COPY . /usr/local/otter
RUN cd /usr/local/otter && \
    python setup.py bdist_wheel && \
    python -m pip install dist/*.whl && \
    cd ~ && rm -rf /usr/local/otter
```

1. 現在のディレクトリの全内容をコンテナ内の `/usr/local/otter` パスにコピーします。
2. 該当ディレクトリに移動し、`setup.py` を使用して wheel 形式のインストールパッケージを生成します。
3. 生成された wheel パッケージをインストールし、その後、ビルドディレクトリを削除して環境をクリーンアップします。

---

```dockerfile
RUN python -m pip install --no-cache-dir -U \
    tqdm colored ipython tabulate tensorboard scikit-learn fire \
    albumentations "Pillow>=10.0.0" fitsne opencv-fixer prettytable
```

必要な Python のサードパーティライブラリをインストールします：

- **`tqdm`**: プログレスバーのツール。
- **`colored`**: ターミナル出力の着色。
- **`ipython`**: 対話型 Python インターフェース。
- **`tabulate`**: 表形式のデータフォーマットツール。
- **`tensorboard`**: 深層学習の可視化ツール。
- **`scikit-learn`**: 機械学習ライブラリ。
- **`fire`**: コマンドラインインターフェース生成ツール。
- **`albumentations`**: 画像データ拡張ライブラリ。
- **`Pillow`**: 画像処理ライブラリ（バージョン 10.0 以上）。
- **`fitsne`**: t-SNE の高効率実装。
- **`opencv-fixer`**: OpenCV 修正ツール。
- **`prettytable`**: 表形式のデータ出力ツール。

:::tip
他のツールが必要な場合、ここに対応するライブラリを追加することができます。
:::

---

```dockerfile
RUN python -c "from opencv_fixer import AutoFix; AutoFix()"
RUN python -c "import capybara; import chameleon"
```

以上の 2 行は、簡単な Python コマンドを実行してインストールが正常に行われたかをテストします：

1. OpenCV の設定問題を自動修正します。
2. `capybara` と `chameleon` モジュールが正常に利用可能であることを確認します。

:::tip
OpenCV はバージョンによる不具合がよく発生するため、ここでは `opencv-fixer` を用いて自動修正を行います。

さらに、`capybara` モジュールでは描画用のフォントファイルを事前にダウンロードする必要があるため、これをコンテナ内にあらかじめ取得しておきます。
:::

---

```dockerfile
RUN printf '#!/bin/bash\n\
    if [ ! -z "$USER_ID" ] && [ ! -z "$GROUP_ID" ]; then\n\
    groupadd -g "$GROUP_ID" -o usergroup\n\
    useradd --shell /bin/bash -u "$USER_ID" -g "$GROUP_ID" -o -c "" -m user\n\
    export HOME=/home/user\n\
    chown -R "$USER_ID":"$GROUP_ID" /home/user\n\
    chown -R "$USER_ID":"$GROUP_ID" /code\n\
    exec gosu "$USER_ID":"$GROUP_ID" "$@"\n\
    else\n\
    exec "$@"\n\
    fi' > "$ENTRYPOINT_SCRIPT" && \
    chmod +x "$ENTRYPOINT_SCRIPT"
```

以下のコードは Bash スクリプトを生成し、次の機能を実現します：

1. 環境変数 `USER_ID` と `GROUP_ID` が設定されている場合、それに基づいてユーザーとグループを動的に作成し、適切な権限を設定します。
2. `gosu` を使用して、そのユーザーに切り替えた状態でコマンドを実行し、コンテナ内で正しい権限を持つ操作を保証します。
3. これらの変数が設定されていない場合は、引数として渡されたコマンドをそのまま実行します。

:::tip
`gosu` はコンテナ内でのユーザー権限を切り替えるためのツールで、`sudo` を使用する際に生じる可能性のある権限の問題を回避できます。
:::

## トレーニングの実行

Docker イメージの構築が完了したので、このイメージを使用してモデルのトレーニングを実行します。

次に、`DocClassifier` プロジェクトに移動し、最初に `train.bash` ファイルの内容を確認します：

- [**DocClassifier/docker/train.bash**](https://github.com/DocsaidLab/DocClassifier/blob/main/docker/train.bash)

```bash title="DocClassifier/docker/train.bash"
#!/bin/bash

docker run \
    -e USER_ID=$(id -u) \
    -e GROUP_ID=$(id -g) \
    --gpus all \
    --shm-size=64g \
    --ipc=host --net=host \
    --cpuset-cpus="0-31" \
    -v $PWD/DocClassifier:/code/DocClassifier \
    -v /data/Dataset:/data/Dataset \ # ここでは、あなたのデータセットディレクトリに置き換えてください。
    -it --rm otter_base_image bash -c "
echo '
from fire import Fire
from DocClassifier.model import main_classifier_train

if __name__ == \"__main__\":
    Fire(main_classifier_train)
' > /code/trainer.py && python /code/trainer.py --cfg_name $1
"
```

上記のファイルについての説明は以下の通りです。変更を加えたい場合は、関連情報を参考にしてください：

1. **`-e USER_ID=$(id -u)` と `-e GROUP_ID=$(id -g)`**：現在のユーザーの UID と GID をコンテナに渡し、コンテナ内のファイル操作の権限がホストと一致するようにします。
2. **`--gpus all`**：GPU サポートを有効にし、すべての利用可能な GPU リソースをコンテナに割り当てます。
3. **`--shm-size=64g`**：共有メモリのサイズを 64GB に設定し、大量のメモリを必要とする深層学習のタスクに適しています。
4. **`--ipc=host` と `--net=host`**：ホストのプロセス間通信とネットワークリソースを共有して、パフォーマンスと互換性を向上させます。
5. **`--cpuset-cpus="0-31"`**：コンテナが CPU 0-31 コアのみを使用するように制限し、他のプロセスに影響を与えないようにします。
6. **`-v $PWD/DocClassifier:/code/DocClassifier`**：ホストの現在のディレクトリ内の `DocClassifier` フォルダをコンテナ内の `/code/DocClassifier` にマウントします。
7. **`-v /data/Dataset:/data/Dataset`**：ホストのデータセットディレクトリをコンテナ内の `/data/Dataset` にマウントします。実際の状況に応じて変更してください。
8. **`-it`**：インタラクティブモードでコンテナを実行します。
9. **`--rm`**：コンテナ終了時に自動的にコンテナを削除し、一時的なコンテナが溜まらないようにします。
10. **`otter_base_image`**：以前に構築した Docker イメージの名前を使用します。変更がある場合は、自分の名前に置き換えてください。

:::tip
ここではいくつかの一般的な問題があります：

1. `--gpus` が動作しない場合：docker が正しくインストールされているか確認してください。参考：[**進階インストール**](../capybara/advance.md)。
2. `--cpuset-cpus`：CPU コアの数を超えないようにしてください。
3. Dockerfile 内で作業ディレクトリが設定されています：`WORKDIR /code`。もし気に入らなければ、自分で変更してください。
4. `-v`：マウントしている作業ディレクトリを必ず確認してください。間違えるとファイルが見つかりません。
5. `DocClassifier` プロジェクト内では、外部から ImageNet データセットをマウントする必要があります。必要ない場合は、この部分を削除してください。
   :::

---

コンテナ起動後、私たちはトレーニングコマンドを実行します。ここでは直接 Python スクリプトを記述します：

```bash
bash -c "
echo '
from fire import Fire
from DocClassifier.model import main_classifier_train

if __name__ == \"__main__\":
    Fire(main_classifier_train)
' > /code/trainer.py && python /code/trainer.py --cfg_name $1
"
```

1. **`echo`**：Python のコードを `/code/trainer.py` ファイルに書き込みます。このコードの機能は次の通りです：

   - **`from fire import Fire`**：`fire` ライブラリをインポートし、コマンドラインインターフェースを生成します。
   - **`from DocClassifier.model import main_classifier_train`**：`DocClassifier.model` モジュールからトレーニングのメイン関数をインポートします。
   - **`if __name__ == "__main__":`**：このスクリプトが実行されると、`Fire(main_classifier_train)` を起動し、コマンドライン引数を関数にバインドします。

2. **`python /code/trainer.py --cfg_name $1`**：生成した Python スクリプトを実行し、`$1` で渡された引数を `--cfg_name` の値として使用します。この引数は通常、設定ファイルを指定するために使用されます。

### パラメータ設定

モデルのトレーニングディレクトリ内には、設定ファイルを置くための専用ディレクトリがあり、通常は `config` と名前が付けられています。

このディレクトリ内で、さまざまな設定ファイルを定義でき、異なるモデルのトレーニングに使用します。例えば：

```yaml title="config/lcnet050_cosface_f256_r128_squeeze_lbn_imagenet.yaml"
common:
  batch_size: 1024
  image_size: [128, 128]
  is_restore: False
  restore_ind: ""
  restore_ckpt: ""
  preview_batch: 1000
  use_imagenet: True
  use_clip: False

global_settings:
  image_size: [128, 128]

trainer:
  max_epochs: 40
  precision: 32
  val_check_interval: 1.0
  gradient_clip_val: 5
  accumulate_grad_batches: 1
  accelerator: gpu
  devices: [0]

model:
  name: ClassifierModel
  backbone:
    name: Backbone
    options:
      name: timm_lcnet_050
      pretrained: True
      features_only: True
  head:
    name: FeatureLearningSqueezeLBNHead
    options:
      in_dim: 256
      embed_dim: 256
      feature_map_size: 4
  loss:
    name: CosFace
    options:
      s: 64
      m: 0.4
    num_classes: -1
    embed_dim: 256

onnx:
  name: WarpFeatureLearning
  input_shape:
    img:
      shape: [1, 3, 128, 128]
      dtype: float32
  input_names: ["img"]
  output_names:
    - feats
  dynamic_axes:
    img:
      "0": batch_size
    output:
      "0": batch_size
  options:
    opset_version: 16
    verbose: False
    do_constant_folding: True

dataset:
  train_options:
    name: SynthDataset
    options:
      aug_ratio: 1
      length_of_dataset: 2560000
      use_imagenet: True
      use_clip: False
  valid_options:
    name: RealDataset
    options:
      return_tensor: True

dataloader:
  train_options:
    batch_size: -1
    num_workers: 24
    shuffle: False
    drop_last: False
  valid_options:
    batch_size: -1
    num_workers: 16
    shuffle: False
    drop_last: False

optimizer:
  name: AdamW
  options:
    lr: 0.001
    betas: [0.9, 0.999]
    weight_decay: 0.001
    amsgrad: False

lr_scheduler:
  name: PolynomialLRWarmup
  options:
    warmup_iters: -1
    total_iters: -1
  pl_options:
    monitor: loss
    interval: step

callbacks:
  - name: ModelCheckpoint
    options:
      monitor: valid_fpr@4
      mode: max
      verbose: True
      save_last: True
      save_top_k: 5
  - name: LearningRateMonitor
    options:
      logging_interval: step
  - name: RichModelSummary
    options:
      max_depth: 3
  - name: CustomTQDMProgressBar
    options:
      unit_scale: -1

logger:
  name: TensorBoardLogger
  options:
    save_dir: logger
```

各フィールドのキー値はすでに `Otter` モジュール内で定義されており、この命名方法に従うだけで正常に動作します。

ここまで読んだあなたは、なぜ最初に「`Otter` モジュールを特別に学ぶ必要はない」と言ったのか理解できるでしょう！

ここにはすでに一定の抽象化とラッピングが施されていますが、それでも非常にカスタマイズされたアーキテクチャです。

最終的には自分に最適な方法を見つけることが必要ですので、この形式にあまりこだわる必要はありません。

### 訓練開始

最後に、`DocClassifier` の上位ディレクトリに移動し、以下のコマンドを実行して訓練を開始します：

```bash
# 後で自分の設定ファイル名に置き換えてください
bash DocClassifier/docker/train.bash lcnet050_cosface_f256_r128_squeeze_lbn_imagenet
```

これらの手順を通じて、Docker コンテナ内で安全にモデル訓練のタスクを実行でき、Docker の隔離環境を利用して一貫性と再現性を確保できます。この方法により、プロジェクトのデプロイメントと拡張がより便利で柔軟になります。

## ONNX への変換

このセクションでは、モデルを ONNX フォーマットに変換する方法を説明します。

まず、`to_onnx.bash` ファイルの内容を確認してください：

```bash title="DocClassifier/docker/to_onnx.bash"
#!/bin/bash

docker run \
    -e USER_ID=$(id -u) \
    -e GROUP_ID=$(id -g) \
    --gpus all \
    --shm-size=64g \
    --ipc=host --net=host \
    --cpuset-cpus="0-31" \
    -v $PWD/DocClassifier:/code/DocClassifier \
    -it --rm otter_base_image bash -c "
echo '
from fire import Fire
from DocClassifier.model import main_classifier_torch2onnx

if __name__ == \"__main__\":
    Fire(main_classifier_torch2onnx)
' > /code/torch2onnx.py && python /code/torch2onnx.py --cfg_name $1
"
```

このファイルから確認を始めますが、変更は必要ありません。変更が必要なのは対応するファイル `model/to_onnx.py` です。

訓練過程では、モデルの訓練を監視するために多くの分岐を使用するかもしれませんが、推論段階ではそのうちの 1 つの分岐のみが必要になる場合があります。したがって、モデルを ONNX フォーマットに変換し、推論段階で必要な分岐のみを残す必要があります。

例えば：

```python
class WarpFeatureLearning(nn.Module):

    def __init__(self, model: L.LightningModule):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head

    def forward(self, img: torch.Tensor):
        xs = self.backbone(img)
        features = self.head(xs)
        return features
```

上記の例では、推論用の分岐だけを取り出し、それを新しいモデル `WarpFeatureLearning` としてラップしました。次に、yaml 設定ファイルで対応するパラメータ設定を行います：

```yaml
onnx:
  name: WarpFeatureLearning
  input_shape:
    img:
      shape: [1, 3, 128, 128]
      dtype: float32
  input_names: ["img"]
  output_names:
    - feats
  dynamic_axes:
    img:
      "0": batch_size
    output:
      "0": batch_size
  options:
    opset_version: 16
    verbose: False
    do_constant_folding: True
```

モデルの入力サイズ、入力名、出力名、および ONNX のバージョン番号について説明します。

変換部分はすでにこちらで作成済みです。上記の変更を完了した後、`model/to_onnx.py` ファイルがあなたのモデルを指していることを確認し、`DocClassifier` の上位ディレクトリに移動して、以下のコマンドを実行して変換を開始します：

```bash
# 後で自分の設定ファイル名に置き換えてください
bash DocClassifier/docker/to_onnx.bash lcnet050_cosface_f256_r128_squeeze_lbn_imagenet
```

## 最後に

再度強調しますが、私たちはすべての作業を Docker 内で行うことを推奨しています。これにより、環境が一貫性を保ち、不要な問題を避けることができます。

以上の説明を通じて、モデル訓練のプロセスをおおよそ把握できたと思います。実際のアプリケーションでは、データセットの準備、モデルのパラメータ調整、訓練過程の監視など、さらに多くの問題に直面するかもしれません。ただし、これらの問題は細かすぎて、すべてを列挙することはできません。この文章では基本的な指針を提供しています。

とにかく、素晴らしいモデルを手に入れることを祈っています！
