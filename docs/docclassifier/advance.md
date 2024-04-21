---
sidebar_position: 4
---

# 進階設定

調用 `DocClassifier` 模型時，你可以透過傳遞參數來進行進階設定。

## Initialization

以下是在初始化階段的進階設定選項：

### 1. Backend

Backend 是一個列舉類型，用於指定 `DocClassifier` 的運算後端。

它包含以下選項：
- **cpu**：使用 CPU 進行運算。
- **cuda**：使用 GPU 進行運算（需要適當的硬體支援）。


```python
from docsaidkit import Backend

model = DocClassifier(backend=Backend.cuda) # 使用 CUDA 後端
#
# 或是
#
model = DocClassifier(backend=Backend.cpu) # 使用 CPU 後端
```

我們是使用 ONNXRuntime 作為模型的推論引擎，雖然 ONNXRuntime 支援了多種後端引擎（包括 CPU、CUDA、OpenCL、DirectX、TensorRT 等等），但限於平常使用的環境，我們稍微做了一點封裝，目前只提供了 CPU 和 CUDA 兩種後端引擎。此外，使用 cuda 運算除了需要適當的硬體支援外，還需要安裝相應的 CUDA 驅動程式和 CUDA 工具包。

如果你的系統中沒有安裝 CUDA，或安裝的版本不正確，則無法使用 CUDA 運算後端。

:::tip
1. 如果你有其他需求，請參考 [**ONNXRuntime 官方文件**](https://onnxruntime.ai/docs/execution-providers/index.html) 進行自定義。
2. 關於安裝依賴相關的問題，請參考 [**ONNXRuntime Release Notes**](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)
:::

### 2. ModelType

ModelType 是一個列舉類型，用於指定 `DocClassifier` 使用的模型類型。

它包含以下選項：

- **margin_based**：使用基於 margin 方法的模型架構。

你可以透過 `model_type` 參數來指定使用的模型。

```python
from docclassifier import ModelType

model = DocClassifier(model_type=ModelType.margin_based)
```

### 3. ModelCfg

你可以透過 `list_models` 來查看所有可用的模型。

```python
from docclassifier import DocClassifier

print(DocClassifier().list_models())
# >>> ['20240326']
```

你可以透過 `model_cfg` 參數來指定模型的配置。

```python
model = DocClassifier(model_cfg='20240326') # 使用 '20240326' 配置
```

## Inference

本模組在推論階段沒有進階設定選項，未來版本可能會加入更多功能。

## Feature Extraction

比起文件的分類，你可能對文件的特徵更感興趣，為此我們提供了 `extract_feature` 方法。

```python
from docclassifier import DocClassifier
import docsaidkit as D

model = DocClassifier()
img = D.imread('path/to/image.jpg')

# 提取特徵： 返回 256 維特徵向量
features = model.extract_feature(img)
```
