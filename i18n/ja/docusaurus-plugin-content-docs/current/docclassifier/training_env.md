---
sidebar_position: 8
---

# モデル訓練ガイド

:::warning
現在、私たちは `DocsaidKit` の移行作業を進めており、このページの内容を改訂し、関連するトレーニングモジュールを `Chameleon` プロジェクトに移行しています。

完了後、このページは再度書き直されますので、ご安心ください。
:::

まず初めに、`DocsaidKit` から基本の Docker イメージ `docsaid_training_base_image` を構築していることを確認してください。

まだこの手順を行っていない場合は、`DocsaidKit` のドキュメントを参照してください。

```bash
# 最初に docsaidkit から基本のイメージをビルド
git clone https://github.com/DocsaidLab/DocsaidKit.git
cd DocsaidKit
bash docker/build.bash
```

その後、以下のコマンドを使用して、`DocClassifier` が動作する Docker イメージを構築します：

```bash
# そして DocClassifier イメージをビルド
git clone https://github.com/DocsaidLab/DocClassifier.git
cd DocClassifier
bash docker/build.bash
```

## 環境構築

以下は、私たちがデフォルトで採用している [Dockerfile](https://github.com/DocsaidLab/DocClassifier/blob/main/docker/Dockerfile) で、モデル訓練のために設計されています。こちらに簡単な説明を付けており、必要に応じて変更できます：

1. **基本イメージ**

   - `FROM docsaid_training_base_image:latest`
   - この行はコンテナの基本イメージを指定します。つまり、`docsaid_training_base_image` の最新バージョンです。この基本イメージは、Docker コンテナを構築するための出発点となります。既に設定されたオペレーティングシステムやいくつかの基本ツールが含まれています。

2. **作業ディレクトリの設定**

   - `WORKDIR /code`
   - コンテナ内で作業ディレクトリを `/code` に設定します。作業ディレクトリは Docker コンテナ内のディレクトリで、アプリケーションやコマンドがそこで実行されます。

3. **環境変数の設定**

   - `ENV ENTRYPOINT_SCRIPT=/entrypoint.sh`
   - 環境変数 `ENTRYPOINT_SCRIPT` の値を `/entrypoint.sh` に設定しています。環境変数は、設定を保存するために使われ、コンテナ内のどこからでもアクセスできます。

4. **gosu のインストール**

   - `RUN` コマンドを使って `gosu` をインストールしています。`gosu` は軽量なツールで、ユーザーとしてコマンドを実行できるようにするもので、`sudo` と似ていますが、Docker コンテナ内で使うのに適しています。
   - `apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*` この行はまずパッケージリストを更新し、その後 `gosu` をインストール、最後に不要なファイルを削除してイメージサイズを小さくしています。

5. **エントリーポイントスクリプトの作成**

   - `RUN` コマンドを使ってエントリーポイントスクリプト `/entrypoint.sh` を作成しています。
   - このスクリプトは最初に環境変数 `USER_ID` と `GROUP_ID` が設定されているかを確認します。もし設定されていれば、新しいユーザーとグループを作成し、そのユーザーでコマンドを実行します。
   - これは、特にコンテナがホストのファイルにアクセスする場合に非常に役立ちます。

6. **権限の付与**

   - `RUN chmod +x "$ENTRYPOINT_SCRIPT"` この行は、エントリーポイントスクリプトに実行権限を付与します。

7. **コンテナのエントリーポイントとデフォルトコマンドの設定**
   - `ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]` と `CMD ["bash"]`
   - これらのコマンドはコンテナ起動時に実行されるデフォルトのコマンドを指定します。コンテナが起動すると、`/entrypoint.sh` スクリプトが実行されます。

## 訓練の実行

これで、構築した Docker イメージを使用してモデル訓練を行う準備が整いました。

最初に `train.bash` ファイルの内容を確認してください：

```bash
#!/bin/bash

cat > trainer.py <<EOF
from fire import Fire
from DocClassifier.model import main_docclassifier_train

if __name__ == '__main__':
    Fire(main_docclassifier_train)
EOF

docker run \
    -e USER_ID=$(id -u) \
    -e GROUP_ID=$(id -g) \
    --gpus all \
    --shm-size=64g \
    --ipc=host --net=host \
    --cpuset-cpus="0-31" \
    -v $PWD/DocClassifier:/code/DocClassifier \
    -v $PWD/trainer.py:/code/trainer.py \
    -v /data/Dataset:/data/Dataset \ # ここはデータセットのディレクトリを置き換えてください
    -it --rm doc_classifier_train python trainer.py --cfg_name $1
```

上記のファイルの説明は以下の通りで、必要に応じて変更できます：

1. **訓練スクリプトの作成**

   - `cat > trainer.py <<EOF ... EOF`
   - このコマンドは Python スクリプト `trainer.py` を作成します。このスクリプトでは必要なモジュールと関数をインポートし、スクリプトのメイン部分で `main_docalign_train` 関数を呼び出します。Google の Python Fire ライブラリを使用してコマンドライン引数を解析し、コマンドラインインターフェースを簡単に生成します。

2. **Docker コンテナの実行**

   - `docker run ... doc_classifier_train python trainer.py --cfg_name $1`
   - この行は Docker コンテナを起動し、その中で `trainer.py` スクリプトを実行します。
   - `-e USER_ID=$(id -u) -e GROUP_ID=$(id -g)`：これらのパラメータは、現在のユーザーのユーザー ID とグループ ID をコンテナに渡し、適切な権限でユーザーを作成します。
   - `--gpus all`：コンテナがすべての GPU を使用できるように指定します。
   - `--shm-size=64g`：共有メモリのサイズを設定し、大規模データ処理に役立ちます。
   - `--ipc=host --net=host`：これらの設定は、コンテナがホストの IPC 名前空間とネットワークスタックを使用できるようにし、パフォーマンスを向上させます。
   - `--cpuset-cpus="0-31"`：どの CPU コアを使用するかを指定します。
   - `-v $PWD/DocClassifier:/code/DocClassifier` など：これらはマウントパラメータで、ホストのディレクトリをコンテナの内部ディレクトリにマッピングして、訓練データとスクリプトにアクセスできるようにします。
   - `--cfg_name $1`：これは `trainer.py` に渡されるパラメータで、設定ファイルの名前を指定します。

3. **データセットのパス**
   - 特に注意すべきは、`/data/Dataset` は訓練データが格納されているディレクトリのパスです。`-v /data/Dataset:/data/Dataset` を、自分のデータセットのディレクトリに置き換える必要があります。

その後、`DocClassifier` の上層ディレクトリに戻り、以下のコマンドを実行して訓練を開始します：

```bash
bash DocClassifier/docker/train.bash lcnet050_cosface_96 # ここは自分の設定ファイル名に置き換えてください
```

これらの手順を経て、Docker コンテナ内で安全にモデル訓練を行うことができます。また、Docker の隔離環境を利用することで、一貫性と再現性が保たれます。この方法により、プロジェクトのデプロイと拡張がより便利で柔軟になります。

## ONNX への変換

次に、モデルを ONNX 形式に変換する方法を説明します。

最初に `to_onnx.bash` ファイルの内容を確認してください：

```bash
#!/bin/bash

cat > torch2onnx.py <<EOF
from fire import Fire
from DocClassifier.model import main_docclassifier_torch2onnx

if __name__ == '__main__':
    Fire(main_docclassifier_torch2onnx)
EOF

docker run \
    -e USER_ID=$(id -u) \
    -e GROUP_ID=$(id -g) \
    --shm-size=64g \
    --ipc=host --net=host \
    -v $PWD/DocClassifier:/code/DocClassifier \
    -v $PWD/torch2onnx.py:/code/torch2onnx.py \
    -it --rm doc_classifier_train python torch2onnx.py --cfg_name $1
```

このファイルから始めますが、変更を加える必要はありません。代わりに、`model/to_onnx.py` を変更する必要があります。

訓練中に複数の分岐を使用してモデルの訓練を監視している場合でも、推論段階ではその一部だけを使用することが多いため、モデルを ONNX 形式に変換し、推論段階で必要な分岐のみを残します。

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

上記の例では、推論用の分岐だけを取り出し、新しいモデル `WarpFeatureLearning` にラップしています。その後、yaml 設定ファイルで対応するパラメータを設定します：

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

これで、モデルの入力サイズ、入力名、出力名、および ONNX バージョン番号が指定されます。

変換部分はすでに書かれており、上記の変更を行った後、`model/to_onnx.py` が正しくあなたのモデルを指していることを確認し、`DocClassifier` の上層ディレクトリに戻って、次のコマンドを実行して変換を開始します：

```bash
bash DocClassifier/docker/to_onnx.bash lcnet050_cosface_96 # ここは自分の設定ファイル名に置き換えてください
```

## 最後に

`DocClassifier/model` ディレクトリ内に新しい ONNX モデルが作成されたはずです。

このモデルを `docclassifier/xxx` の対応するディレクトリに移動し、モデルのパスパラメータを変更すれば、推論を行うことができます。
