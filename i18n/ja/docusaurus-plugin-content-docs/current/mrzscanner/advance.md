---
sidebar_position: 4
---

# 高度な設定

`MRZScanner` モデルを使用する際、パラメータを渡すことで高度な設定を行うことができます。

## 初期化

以下は初期化時に利用可能な高度な設定オプションです：

### 1. Backend

Backend は列挙型で、`MRZScanner` の計算バックエンドを指定します。

利用可能なオプション：

- **cpu**：CPU を使用して計算。
- **cuda**：GPU を使用して計算（適切なハードウェアサポートが必要）。

```python
from capybara import Backend

model = MRZScanner(backend=Backend.cuda) # CUDA バックエンドを使用
#
# または
#
model = MRZScanner(backend=Backend.cpu) # CPU バックエンドを使用
```

ONNXRuntime を推論エンジンとして使用しています。ONNXRuntime は CPU、CUDA、OpenCL、DirectX、TensorRT など複数のバックエンドエンジンをサポートしていますが、一般的な使用環境を考慮して、現在は CPU と CUDA のみをサポートしています。CUDA 計算を使用するには適切なハードウェアサポートに加え、対応する CUDA ドライバーおよび CUDA ツールキットが必要です。

CUDA がインストールされていない場合、またはバージョンが適切でない場合は、CUDA 計算バックエンドを使用できません。

:::tip

1. 他のバックエンドが必要な場合は、[**ONNXRuntime 公式ドキュメント**](https://onnxruntime.ai/docs/execution-providers/index.html)を参照してカスタマイズしてください。
2. 依存関係のインストールに関する問題については、[**ONNXRuntime Release Notes**](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)をご確認ください。
   :::

### 2. ModelType

ModelType は列挙型で、`MRZScanner` が使用するモデルの種類を指定します。

利用可能なオプション：

- **spotting**：エンドツーエンドモデルアーキテクチャを使用。

`model_type` パラメータでモデルの種類を指定できます。

```python
from mrzscanner import MRZScanner

model = MRZScanner(model_type=MRZScanner.spotting)
```

### 3. ModelCfg

`list_models` を使用して利用可能なすべてのモデルを確認できます。

```python
from mrzscanner import MRZScanner

print(MRZScanner().list_models())
# >>> ['20240919']
```

`model_cfg` パラメータを使用してモデル設定を指定できます。

```python
model = MRZScanner(model_cfg='20240919') # '20240919' 設定を使用
```

## 推論

以下は推論時に利用可能な高度な設定オプションです：

### 中央クロップ

推論段階で適切な高度設定を行うことで、モデルのパフォーマンスや精度に大きな影響を与えることができます。

`do_center_crop` は中心クロップを行うかどうかを決定する重要なパラメータです。

特に実用的なシナリオでは、画像が標準的な正方形サイズでない場合が多いため、この設定が重要です。

例えば：

- スマートフォンで撮影した写真は通常 9:16 のアスペクト比。
- スキャンした書類は A4 用紙の比率。
- ウェブページのスクリーンショットは 16:9 のアスペクト比。
- ウェブカメラで撮影した画像は一般的に 4:3 の比率。

これらの非正方形画像は、適切な処理を行わずに推論を実行すると、多くの不要な領域や空白が含まれ、モデルの推論性能に悪影響を及ぼします。中心クロップを行うことで、これらの不要な領域を削減し、画像の中心領域に焦点を当てることで、推論の精度と効率を向上させることができます。

使用方法は以下の通りです：

```python
import capybara as cb
from mrzscanner import MRZScanner

model = MRZScanner()

img = cb.imread('path/to/image.jpg')
result = model(img, do_center_crop=True) # 中心クロップを使用
```

:::tip
**使用するべき場合**：MRZ エリアを切り取る心配がない場合や、画像のアスペクト比が正方形でない場合に中心クロップを使用できます。
:::

### 後処理

中心クロップ以外にも、モデルの精度をさらに向上させる後処理オプションを提供しています。後処理用のパラメータ `do_postprocess=True` がデフォルトで有効になっています。

これは、MRZ エリアにはいくつかの規則（例：国コードは大文字のアルファベットのみ、性別は `M` または `F` のみなど）があるためです。

このような規則を使用して MRZ エリアを修正することができます。以下のコード例のように、数字が出現しないはずのフィールドで誤認識された数字を正しい文字に置き換える後処理が可能です：

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

この後処理による精度向上は限られた状況で発生するものの、誤認識結果を修正する際に有効です。

推論時に `do_postprocess` を `False` に設定することで、後処理を行わない生の認識結果を取得できます。

```python
result, msg = model(img, do_postprocess=False)
```
