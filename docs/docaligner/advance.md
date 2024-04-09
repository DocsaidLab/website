---
sidebar_position: 4
---

# 進階設定

調用 `DocAligner` 模型時，你可以透過傳遞參數來進行進階設定。

## Initialization

以下是在初始化階段的進階設定選項：

### 1. Backend

Backend 是一個列舉類型，用於指定 `DocAligner` 的運算後端。

它包含以下選項：
- **cpu**：使用 CPU 進行運算。
- **cuda**：使用 GPU 進行運算（需要適當的硬體支援）。

我們是使用 ONNXRuntime 作為模型的推論引擎，雖然 ONNXRuntime 支援了多種後端引擎（包括 CPU、CUDA、OpenCL、DirectX、TensorRT 等等），但限於平常使用的環境，我們稍微做了一點封裝，目前只提供了 CPU 和 CUDA 兩種後端引擎。此外，使用 cuda 運算除了需要適當的硬體支援外，還需要安裝相應的 CUDA 驅動程式和 CUDA 工具包。

如果你的系統中沒有安裝 CUDA，或安裝的版本不正確，則無法使用 CUDA 運算後端。

:::tip
如果你有其他需求，請參考 [**ONNXRuntime 官方文件**](https://onnxruntime.ai/docs/execution-providers/index.html) 進行自定義。
:::

:::tip


關於安裝依賴相關的問題，請參考 [**ONNXRuntime Release Notes**](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)
:::

```python
from docsaidkit import Backend

model = DocAligner(backend=Backend.cuda) # 使用 CUDA 後端
#
# 或是
#
model = DocAligner(backend=Backend.cpu) # 使用 CPU 後端
```

### 2. ModelType

ModelType 是一個列舉類型，用於指定 `DocAligner` 使用的模型類型。

它包含以下選項：
- **heatmap**：使用熱圖模型。
- **point**：使用點回歸模型。

我們提供了兩種不同的模型：「熱圖模型」和「點回歸模型」。

你可以透過 `model_type` 參數來指定使用的模型。

```python
from docaligner import ModelType

model = DocAligner(model_type=ModelType.heatmap) # 使用熱圖模型
#
# 或是
#
model = DocAligner(model_type=ModelType.point) # 使用點回歸模型
```

:::tip
話是這樣講，但不建議用『點回歸』模型，因為它的效果不太好，這個純粹為了研究用的。
:::

### 3. ModelCfg

我們訓練了很多模型，並且幫它們取了名字，

你可以透過 `list_models` 來查看所有可用的模型。

```python
from docaligner import DocAligner

print(DocAligner().list_models())
# >>> [
#     'lcnet050',
#     'lcnet050_fpn',
#     'lcnet100',
#     'lcnet100_fpn',
#     'mobilenetv2_140',
#     'fastvit_t8',
#     'fastvit_sa24',       <-- Default
#     ...
# ]
```

你可以透過 `model_cfg` 參數來指定模型的配置。

```python
model = DocAligner(model_cfg='mobilenetv2_140') # 使用 'mobilenetv2_140' 配置
```

## Inference

以下是在推論階段的進階設定選項：

### CenterCrop

這個功能透過 `do_center_crop` 參數進行設定。
