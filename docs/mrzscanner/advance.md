---
sidebar_position: 4
---

# 進階設定

調用 `MRZScanner` 模型時，你可以透過傳遞參數來進行進階設定。

## Initialization

以下是在初始化階段的進階設定選項：

### 1. Backend

Backend 是一個列舉類型，用於指定 `MRZScanner` 的運算後端。

它包含以下選項：

- **cpu**：使用 CPU 進行運算。
- **cuda**：使用 GPU 進行運算（需要適當的硬體支援）。

```python
from docsaidkit import Backend

model = MRZScanner(backend=Backend.cuda) # 使用 CUDA 後端
#
# 或是
#
model = MRZScanner(backend=Backend.cpu) # 使用 CPU 後端
```

我們是使用 ONNXRuntime 作為模型的推論引擎，雖然 ONNXRuntime 支援了多種後端引擎（包括 CPU、CUDA、OpenCL、DirectX、TensorRT 等等），但限於平常使用的環境，我們稍微做了一點封裝，目前只提供了 CPU 和 CUDA 兩種後端引擎。此外，使用 cuda 運算除了需要適當的硬體支援外，還需要安裝相應的 CUDA 驅動程式和 CUDA 工具包。

如果你的系統中沒有安裝 CUDA，或安裝的版本不正確，則無法使用 CUDA 運算後端。

:::tip

1. 如果你有其他需求，請參考 [**ONNXRuntime 官方文件**](https://onnxruntime.ai/docs/execution-providers/index.html) 進行自定義。
2. 關於安裝依賴相關的問題，請參考 [**ONNXRuntime Release Notes**](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)
   :::

### 2. ModelType

ModelType 是一個列舉類型，用於指定 `MRZScanner` 使用的模型類型。

它包含以下選項：

- **spotting**：使用端到端的模型架構。

你可以透過 `model_type` 參數來指定使用的模型。

```python
from mrzscanner import MRZScanner

model = MRZScanner(model_type=MRZScanner.spotting)
```

### 3. ModelCfg

你可以透過 `list_models` 來查看所有可用的模型。

```python
from mrzscanner import MRZScanner

print(MRZScanner().list_models())
# >>> ['20240919']
```

你可以透過 `model_cfg` 參數來指定模型的配置。

```python
model = MRZScanner(model_cfg='20240919') # 使用 '20240919' 配置
```

## Inference

以下是在推論階段的進階設定選項：

### 中心裁剪

在進行推論階段，設定適當的進階選項能夠顯著影響模型的表現和效果。

其中，`do_center_crop`是一個關鍵的參數，它決定是否在推論時進行中心裁剪。

這項設定尤其重要，因為在實際應用中，我們遇到的圖像往往並非標準的正方形尺寸。

實際上，圖像的尺寸和比例多種多樣，例如：

- 手機拍攝的照片普遍採用 9:16 的寬高比；
- 掃描的文件常見於 A4 的紙張比例；
- 網頁截圖大多是 16:9 的寬高比；
- 透過 webcam 拍攝的圖片，則通常是 4:3 的比例。

這些非正方形的圖像，在不經過適當處理直接進行推論時，往往會包含大量的無關區域或空白，從而對模型的推論效果產生不利影響。進行中心裁剪能夠有效減少這些無關區域，專注於圖像的中心區域，從而提高推論的準確性和效率。

使用方式如下：

```python
import docsaidkit as D
from mrzscanner import MRZScanner

model = MRZScanner()

img = D.imread('path/to/image.jpg')
result = model(img, do_center_crop=True) # 使用中心裁剪
```

:::tip
**使用時機**：『不會切到 MRZ 區域』且圖片比例不是正方形時，可以使用中心裁切。
:::

### 後處理

除了中心裁剪外，我們還提供了一個後處理的選項，用於進一步提高模型的準確性。我們有提供一個後處理參數，預設為 `do_postprocess=True`。

那是因為 MRZ 區塊中存在一些規則，例如國家代碼只能為大寫英文字母；性別只有 `M` 和 `F` 等等。這些是可以用來規範 MRZ 區塊的。

因此我們針對可以規範的區塊進行人工校正，例如以下程式碼片段，在不可能出現數字的欄位中，把可能誤判的數字替換成正確的字元：

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

雖然在我們的時候中，這個替換字元的後處理沒有幫我們提高更多的準確度，但保留這個功能還是可以在某些情況下把錯誤的辨識結果修正回來。

你可以在推論的時候把 `do_postprocess` 設為 `False`，這樣就可以得到原始的辨識結果。

```python
result, msg = model(img, do_postprocess=False)
```
