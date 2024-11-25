---
sidebar_position: 4
---

# 高度な設定

`MRZScanner` モデルを呼び出す際、パラメータを渡すことで高度な設定を行うことができます。

## 初期化

以下は、初期化段階での高度な設定オプションです：

### 1. Backend

Backend は、`MRZScanner` の計算バックエンドを指定するための列挙型です。

以下のオプションがあります：

- **cpu**：CPU を使用して計算します。
- **cuda**：GPU を使用して計算します（適切なハードウェアサポートが必要です）。

```python
from docsaidkit import Backend

model = MRZScanner(backend=Backend.cuda) # CUDA バックエンドを使用
#
# または
#
model = MRZScanner(backend=Backend.cpu) # CPU バックエンドを使用
```

私たちは ONNXRuntime をモデル推論エンジンとして使用しており、ONNXRuntime は複数のバックエンジン（CPU、CUDA、OpenCL、DirectX、TensorRT など）をサポートしていますが、一般的に使用される環境を考慮して、若干のラッピングを行い、現在は CPU と CUDA の 2 つのバックエンジンのみ提供しています。さらに、CUDA を使用する場合、適切なハードウェアサポートが必要であり、CUDA ドライバとツールキットをインストールする必要があります。

システムに CUDA がインストールされていない場合、またはバージョンが正しくない場合は、CUDA バックエンドを使用できません。

:::tip

1. その他の要件がある場合は、[**ONNXRuntime の公式ドキュメント**](https://onnxruntime.ai/docs/execution-providers/index.html) を参照してカスタマイズしてください。
2. 依存関係のインストールについては、[**ONNXRuntime リリースノート**](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) を参照してください。
   :::

### 2. ModelType

ModelType は、`MRZScanner` が使用するモデルの種類を指定するための列挙型です。

以下のオプションがあります：

- **spotting**：エンドツーエンドのモデルアーキテクチャを使用します。

`model_type` パラメータを使用して、使用するモデルを指定できます。

```python
from mrzscanner import MRZScanner

model = MRZScanner(model_type=MRZScanner.spotting)
```

### 3. ModelCfg

`list_models` を使用して、使用可能なすべてのモデルを表示できます。

```python
from mrzscanner import MRZScanner

print(MRZScanner().list_models())
# >>> ['20240919']
```

`model_cfg` パラメータを使用して、モデルの設定を指定できます。

```python
model = MRZScanner(model_cfg='20240919') # '20240919' 設定を使用
```

## 推論

以下は、推論段階での高度な設定オプションです：

### 中心裁剪

推論段階では、適切な高度なオプションを設定することで、モデルのパフォーマンスと結果に大きな影響を与えることができます。

その中で、`do_center_crop` は推論時に中央裁剪を行うかどうかを決定する重要なパラメータです。

この設定は特に重要です。なぜなら、実際のアプリケーションでは、処理する画像が標準的な正方形のサイズでないことが多いためです。

実際、画像のサイズとアスペクト比はさまざまで、例えば：

- スマホで撮影した写真は通常 9:16 のアスペクト比；
- スキャンした書類は A4 のアスペクト比；
- ウェブサイトのスクリーンショットは通常 16:9 のアスペクト比；
- ウェブカメラで撮影した画像は通常 4:3 のアスペクト比。

これらの非正方形の画像は、適切に処理せずに直接推論を行うと、無関係な領域や空白が大量に含まれ、推論結果に悪影響を与えることが多いです。中央裁剪を行うことで、これらの無関係な領域を減少させ、画像の中心領域に集中することができ、推論の精度と効率を向上させます。

使用方法は以下の通りです：

```python
import docsaidkit as D
from mrzscanner import MRZScanner

model = MRZScanner()

img = D.imread('path/to/image.jpg')
result = model(img, do_center_crop=True) # 中心裁剪を使用
```

:::tip
**使用時の注意**：「MRZ 領域が切り取られない」かつ画像比率が正方形でない場合に、中央裁剪を使用することをお勧めします。
:::

### 後処理

中央裁剪に加えて、モデルの精度をさらに向上させるために後処理オプションも提供しています。デフォルトでは、後処理パラメータは `do_postprocess=True` です。

MRZ 区画にはいくつかの規則があり、例えば国コードは大文字アルファベットのみ、性別は `M` または `F` のみであるなど、これらを規範として MRZ 区画を調整できます。

そのため、規範に従った修正を行うために人工的に補正を行っています。以下のコードスニペットのように、数値が表示されないフィールドに誤って判定された数値を正しい文字に置き換えることができます：

```python
import re

def replace_digits(text: str):
    text = re.sub('0', 'O', text)
    text = re.sub('1', 'I', text)
    text = re.sub('2', 'Z', text)
    text = re.sub('4', 'A', text)
    text = re.sub('5', 'S', text)
    text = re.sub('8', 'B', text)
    return text

if doc_type == 3:  # TD1
    if len(results[0]) != 30 or len(results[1]) != 30 or len(results[2]) != 30:
        return [''], ErrorCodes.POSTPROCESS_FAILED_TD1_LENGTH
    # Line1
    doc = results[0][0:2]
    country = replace_digits(results[0][2:5])
```

この後処理の文字置換は私たちの場合、精度向上には寄与しませんでしたが、特定の状況では誤った認識結果を修正できることがあります。

推論時に `do_postprocess` を `False` に設定することで、元の認識結果をそのまま得ることができます。

```python
result, msg = model(img, do_postprocess=False)
```
