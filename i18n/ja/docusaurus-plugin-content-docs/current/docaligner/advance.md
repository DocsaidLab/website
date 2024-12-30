---
sidebar_position: 4
---

# 高度な設定

`DocAligner` モデルを呼び出す際、パラメータを渡すことで高度な設定ができます。

## 初期化

以下は初期化段階での高度な設定オプションです：

### 1. Backend

Backend は列挙型で、`DocAligner` の計算バックエンドを指定するために使用します。

次のオプションがあります：

- **cpu**：CPU を使用して計算を行います。
- **cuda**：GPU を使用して計算を行います（適切なハードウェアサポートが必要です）。

```python
from capybara import Backend

model = DocAligner(backend=Backend.cuda) # CUDA バックエンドを使用
#
# または
#
model = DocAligner(backend=Backend.cpu) # CPU バックエンドを使用
```

私たちは ONNXRuntime をモデルの推論エンジンとして使用しています。ONNXRuntime は多くのバックエンドエンジン（CPU、CUDA、OpenCL、DirectX、TensorRT など）をサポートしていますが、通常の使用環境に基づいて少しラッピングを施し、現在は CPU と CUDA のみのバックエンドエンジンを提供しています。さらに、CUDA を使用するには適切なハードウェアサポートの他に、対応する CUDA ドライバと CUDA ツールキットをインストールする必要があります。

システムに CUDA がインストールされていない、またはインストールされているバージョンが正しくない場合、CUDA バックエンドを使用することはできません。

:::tip

1. 他のニーズがある場合は、[**ONNXRuntime 公式ドキュメント**](https://onnxruntime.ai/docs/execution-providers/index.html) を参照してカスタマイズしてください。
2. 依存関係に関するインストール問題については、[**ONNXRuntime リリースノート**](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) を参照してください。
   :::

### 2. ModelType

ModelType は列挙型で、`DocAligner` が使用するモデルタイプを指定します。

次のオプションがあります：

- **heatmap**：ヒートマップモデルを使用。
- **point**：ポイント回帰モデルを使用。

私たちは「ヒートマップモデル」と「ポイント回帰モデル」という 2 種類の異なるモデルを提供しています。

`model_type` パラメータを使用して、使用するモデルを指定できます。

```python
from docaligner import ModelType

model = DocAligner(model_type=ModelType.heatmap) # ヒートマップモデルを使用
#
# または
#
model = DocAligner(model_type=ModelType.point) # ポイント回帰モデルを使用
```

:::tip
こうは言うものの、「ポイント回帰」モデルはあまり効果が良くないので、これは研究用に提供されたものです。
:::

### 3. ModelCfg

私たちは多くのモデルをトレーニングし、それらに名前を付けました。

`list_models` を使用して、使用可能なすべてのモデルを確認できます。

```python
from docaligner import DocAligner

print(DocAligner().list_models())
# >>> [
#     'lcnet100',
#     'fastvit_t8',
#     'fastvit_sa24',       <-- デフォルト
#     ...
# ]
```

`model_cfg` パラメータを使用して、モデルの設定を指定できます。

```python
model = DocAligner(model_cfg='fastvit_t8') # 'fastvit_t8' 設定を使用
```

## 推論

以下は推論段階での高度な設定オプションです：

### CenterCrop

推論段階では、適切な高度なオプション設定がモデルのパフォーマンスと効果に大きな影響を与える可能性があります。

その中で、`do_center_crop` は重要なパラメータで、推論時にセンタークロップを行うかどうかを決定します。

この設定は特に重要で、実際のアプリケーションでは、処理する画像が標準の正方形サイズでないことが多いためです。

実際には、画像のサイズや比率は多種多様で、例えば次のようなケースがあります：

- 携帯電話で撮影された写真は一般的に 9:16 のアスペクト比を採用；
- スキャンした文書は通常 A4 の用紙比率；
- ウェブページのスクリーンショットはほとんど 16:9 のアスペクト比；
- Web カメラで撮影した画像は通常 4:3 の比率。

これらの非正方形の画像は、適切に処理せずに直接推論を行うと、多くの無関係な領域や空白が含まれ、モデルの推論効果に悪影響を与えることがあります。センタークロップを行うことで、これらの無関係な領域を減らし、画像の中心領域に焦点を当てることで、推論の精度と効率を高めることができます。

使用方法は以下の通りです：

```python
from capybara import imread
from docaligner import DocAligner

model = DocAligner()

img = imread('path/to/image.jpg')
result = model(img, do_center_crop=True) # センタークロップを使用
```

:::tip
**使用時期**：『画像が切り取られない』且つ画像のアスペクト比が正方形でない場合に、センタークロップを使用できます。
:::

:::warning
センタークロップは計算処理の一部であり、元の画像を変更することはありません。最終的な結果は元の画像のサイズにマッピングされるため、画像の変形や歪みを心配する必要はありません。
:::
