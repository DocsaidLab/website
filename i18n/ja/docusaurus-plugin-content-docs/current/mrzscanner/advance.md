---
sidebar_position: 4
---

# 高度な設定

`MRZScanner` モデルを呼び出す際、パラメータを渡すことで高度な設定を行うことができます。

## 初期化

以下は、初期化時の高度な設定オプションです：

### 1. Backend

Backend は列挙型で、`MRZScanner` の計算バックエンドを指定するために使用されます。

以下のオプションが含まれています：

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

私たちは ONNXRuntime をモデルの推論エンジンとして使用しており、ONNXRuntime は複数のバックエンドエンジン（CPU、CUDA、OpenCL、DirectX、TensorRT など）をサポートしていますが、日常的に使用する環境に合わせて少しラップを行い、現在は CPU と CUDA の 2 つのバックエンドエンジンのみを提供しています。さらに、cuda 計算を使用するには、適切なハードウェアサポートに加えて、対応する CUDA ドライバーと CUDA ツールキットのインストールが必要です。

もしシステムに CUDA がインストールされていない、またはインストールされているバージョンが正しくない場合、CUDA 計算バックエンドは使用できません。

:::tip

1. 他に必要な場合は、[**ONNXRuntime 公式ドキュメント**](https://onnxruntime.ai/docs/execution-providers/index.html) を参照してカスタマイズしてください。
2. 依存関係のインストールに関する問題については、[**ONNXRuntime リリースノート**](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) を参照してください。
   :::

### 2. ModelType

ModelType は列挙型で、`MRZScanner` が使用するモデルタイプを指定するために使用されます。

現在、以下のオプションがあります：

- **spotting**：エンドツーエンドのモデルアーキテクチャを使用、1 つのモデルのみを読み込みます。
- **two_stage**：2 段階のモデルアーキテクチャを使用、2 つのモデルを読み込みます。
- **detection**：MRZ の検出モデルのみを読み込みます。
- **recognition**：MRZ の認識モデルのみを読み込みます。

`model_type` パラメータを使用して、使用するモデルを指定できます。

```python
from mrzscanner import MRZScanner

model = MRZScanner(model_type=MRZScanner.spotting)
```

### 3. ModelCfg

`list_models` を使用して、利用可能なすべてのモデルを確認できます。

```python
from mrzscanner import MRZScanner

print(MRZScanner().list_models())
# {
#    'spotting': ['20240919'],
#    'detection': ['20250222'],
#    'recognition': ['20250221']
# }
```

使用するバージョンを選択し、`spotting_cfg`、`detection_cfg`、`recognition_cfg` などのパラメータと一緒に `ModelType` を使用して、使用するモデルを指定します。

1. **spotting**：

   ```python
   model = MRZScanner(
      model_type=ModelType.spotting,
      spotting_cfg='20240919'
   )
   ```

2. **two_stage**：

   ```python
   model = MRZScanner(
      model_type=ModelType.two_stage,
      detection_cfg='20250222',
      recognition_cfg='20250221'
   )
   ```

3. **detection**：

   ```python
   model = MRZScanner(
      model_type=ModelType.detection,
      detection_cfg='20250222'
   )
   ```

4. **recognition**：

   ```python
   model = MRZScanner(
      model_type=ModelType.recognition,
      recognition_cfg='20250221'
   )
   ```

指定しないこともできます。各モデルのデフォルトバージョンはすでに設定されています。

## ModelType.spotting

このモデルはエンドツーエンドのモデルで、MRZ の位置を直接検出して認識します。欠点は精度が低く、MRZ の座標が返されないことです。

使用例は以下の通りです：

```python
import cv2
from skimage import io
from mrzscanner import MRZScanner, ModelType

# モデルの作成
model = MRZScanner(
   model_type=ModelType.spotting,
   spotting_cfg='20240919'
)

# オンライン画像の読み込み
img = io.imread('https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg?raw=true')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# モデル推論
result = model(img, do_center_crop=True, do_postprocess=False)

# 結果の出力
print(result)
# {
#    'mrz_polygon': None,
#    'mrz_texts': [
#        'PCAZEQAOARIN<<FIDAN<<<<<<<<<<<<<<<<<<<<<<<<<',
#        'C946302620AZE6707297F23031072W12IMJ<<<<<<<40'
#    ],
#    'msg': <ErrorCodes.NO_ERROR: 'No error.'>
# }
```

## ModelType.two_stage

このモデルは二段階モデルで、まず MRZ の位置を検出し、次に認識を行います。利点は精度が高く、MRZ の座標も返されることです。

使用例は以下の通りで、最後に MRZ の位置を描画できます：

```python
import cv2
from skimage import io
from mrzscanner import MRZScanner, ModelType

# モデルの作成
model = MRZScanner(
   model_type=ModelType.two_stage,
   detection_cfg='20250222',
   recognition_cfg='20250221'
)

# オンライン画像の読み込み
img = io.imread('https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg?raw=true')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# モデル推論
result = model(img, do_center_crop=True, do_postprocess=False)

# 結果の出力
print(result)
# {
#     'mrz_polygon':
#         array(
#             [
#                 [ 158.536 , 1916.3734],
#                 [1682.7792, 1976.1683],
#                 [1677.1018, 2120.8926],
#                 [ 152.8586, 2061.0977]
#             ],
#             dtype=float32
#         ),
#     'mrz_texts': [
#         'PCAZEQAQARIN<<FIDAN<<<<<<<<<<<<<<<<<<<<<<<<<',
#         'C946302620AZE6707297F23031072W12IMJ<<<<<<<40'
#     ],
#     'msg': <ErrorCodes.NO_ERROR: 'No error.'>
# }

# MRZの位置を描画
from capybara import draw_polygon, imwrite, centercrop

poly_img = draw_polygon(img, result['mrz_polygon'], color=(0, 0, 255), thickness=5)
imwrite(centercrop(poly_img))
```

<div align="center" >
<figure style={{width: "70%"}}>
![demo_two_stage](./resources/demo_two_stage.jpg)
</figure>
</div>

## ModelType.detection

このモデルは MRZ の位置のみを検出し、認識は行いません。

使用例は以下の通りです：

```python
import cv2
from skimage import io
from mrzscanner import MRZScanner, ModelType

# モデルの作成
model = MRZScanner(
   model_type=ModelType.detection,
   detection_cfg='20250222',
)

# オンライン画像の読み込み
img = io.imread('https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg?raw=true')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# モデル推論
result = model(img, do_center_crop=True)

# 結果の出力
print(result)
# {
#     'mrz_polygon':
#         array(
#             [
#                 [ 158.536 , 1916.3734],
#                 [1682.7792, 1976.1683],
#                 [1677.1018, 2120.8926],
#                 [ 152.8586, 2061.0977]
#             ],
#             dtype=float32
#         ),
#     'mrz_texts': None,
#     'msg': <ErrorCodes.NO_ERROR: 'No error.'>
# }
```

ここでの MRZ の位置は先ほどと同じ結果となるため、再度描画することはありません。

## ModelType.recognition

このモデルは MRZ の認識のみを行い、位置の検出は行いません。

このモデルを実行するには、まず MRZ を切り出した画像を準備し、その画像をモデルに入力する必要があります。

まず、先ほど検出した座標を使って MRZ を切り出した画像を準備します：

```python
import numpy as np
from skimage import io
from capybara import imwarp_quadrangle, imwrite

polygon = np.array([
    [ 158.536 , 1916.3734],
    [1682.7792, 1976.1683],
    [1677.1018, 2120.8926],
    [ 152.8586, 2061.0977]
], dtype=np.float32)

img = io.imread('https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg?raw=true')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

mrz_img = imwarp_quadrangle(img, polygon)
imwrite(mrz_img)
```

上記のコードを実行すると、MRZ を切り出した画像が得られます：

<div align="center" >
<figure style={{width: "90%"}}>
![demo_recognition_warp](./resources/demo_recognition_warp.jpg)
</figure>
</div>

画像が準備できたら、認識モデルを単独で実行できます：

```python
from mrzscanner import MRZScanner, ModelType

# モデルの作成
model = MRZScanner(
   model_type=ModelType.recognition,
   recognition_cfg='20250221'
)

# MRZ切り出し後の画像を入力
result = model(mrz_img, do_center_crop=False)

# 結果の出力
print(result)
# {
#     'mrz_polygon':None,
#     'mrz_texts': [
#         'PCAZEQAQARIN<<FIDAN<<<<<<<<<<<<<<<<<<<<<<<<<',
#         'C946302620AZE6707297F23031072W12IMJ<<<<<<<40'
#     ],
#     'msg': <ErrorCodes.NO_ERROR: 'No error.'>
# }
```

:::warning
ここでは `do_center_crop=False` と設定しています。すでに切り出した画像を使用しているためです。
:::
