---
sidebar_position: 4
---

# 高度な設定

`DocClassifier` モデルを呼び出す際に、パラメータを渡して高度な設定を行うことができます。

## 初期化

以下は初期化段階での高度な設定オプションです：

### 1. Backend

Backend は列挙型で、`DocClassifier` の計算バックエンドを指定するために使用されます。

以下のオプションがあります：

- **cpu**：CPU を使用して計算を行います。
- **cuda**：GPU を使用して計算を行います（適切なハードウェアサポートが必要です）。

```python
from capybara import Backend

model = DocClassifier(backend=Backend.cuda) # CUDA バックエンドを使用
#
# または
#
model = DocClassifier(backend=Backend.cpu) # CPU バックエンドを使用
```

私たちは ONNXRuntime をモデルの推論エンジンとして使用しています。ONNXRuntime はさまざまなバックエンドエンジン（CPU、CUDA、OpenCL、DirectX、TensorRT など）をサポートしていますが、普段使用している環境に合わせて少しラップをかけており、現在提供されているのは CPU と CUDA の 2 つのバックエンドエンジンのみです。また、CUDA 計算を使用するには、適切なハードウェアサポートだけでなく、対応する CUDA ドライバとツールキットのインストールが必要です。

システムに CUDA がインストールされていない場合や、インストールされているバージョンが正しくない場合は、CUDA 計算バックエンドを使用できません。

:::tip

1. 他のニーズがある場合は、[**ONNXRuntime の公式ドキュメント**](https://onnxruntime.ai/docs/execution-providers/index.html) を参照してカスタマイズしてください。
2. 依存関係のインストールに関する問題については、[**ONNXRuntime のリリースノート**](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) を参照してください。
   :::

### 2. ModelType

ModelType は列挙型で、`DocClassifier` が使用するモデルの種類を指定します。

以下のオプションがあります：

- **margin_based**：マージンベースの方法を使用したモデルアーキテクチャ。

`model_type` パラメータを使用して、使用するモデルを指定できます。

```python
from docclassifier import ModelType

model = DocClassifier(model_type=ModelType.margin_based)
```

### 3. ModelCfg

`list_models` を使用して、利用可能なすべてのモデルを確認できます。

```python
from docclassifier import DocClassifier

print(DocClassifier().list_models())
# >>> ['20240326']
```

`model_cfg` パラメータを使用して、モデルの設定を指定できます。

```python
model = DocClassifier(model_cfg='20240326') # '20240326' 設定を使用
```

## 推論

このモジュールには推論段階での高度な設定オプションはありません。今後のバージョンでさらに機能が追加される可能性があります。

## 特徴抽出

ドキュメントの分類よりも、ドキュメントの特徴に興味がある場合は、`extract_feature` メソッドを提供しています。

```python
from docclassifier import DocClassifier
import capybara as cb

model = DocClassifier()
img = cb.imread('path/to/image.jpg')

# 特徴を抽出： 256 次元の特徴ベクトルを返す
features = model.extract_feature(img)
```
