---
sidebar_position: 3
---

# 快速開始

我們提供了一個簡單的模型推論介面，其中包含了前後處理的邏輯。

首先，你需要導入所需的相關依賴並創建 `DocAligner` 類別。

## 模型推論

以下是一個簡單的範例，展示如何使用 `DocAligner` 進行模型推論：

```python
from docaligner import DocAligner

model = DocAligner()
```

啟動模型之後，接著要準備一張圖片進行推論：

:::tip
你可以使用 `DocAligner` 提供的測試圖片：

下載連結：[**run_test_card.jpg**](https://github.com/DocsaidLab/DocAligner/blob/main/docs/run_test_card.jpg)
:::

```python
import docsaidkit as D

img = D.imread('path/to/run_test_card.jpg')
```

或是你可以直接透過 URL 進行讀取：

```python
import cv2
from skimage import io

img = io.imread('https://github.com/DocsaidLab/DocAligner/blob/main/docs/run_test_card.jpg?raw=true')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
```

![test_card](./resources/run_test_card.jpg)

接著，我們可以使用 `model` 進行推論：

```python
result = model(img)
```

你得到的推論結果是經過我們包裝的 [**Document**](../docsaidkit/funcs/objects/document) 類型，它包含了文件的多邊形、OCR 文字資訊等等。在這個模組中，我們不會用到 OCR 相關的功能，因此我們只會使用 `image` 和 `doc_polygon` 屬性。獲取到推論結果後，你可以進行多種後處理操作。

:::tip
`DocAligner` 已經用 `__call__` 進行了封裝，因此你可以直接呼叫實例進行推論。
:::

:::tip
**模型下載**：我們有設計了自動下載模型的功能，當你第一次使用 `DocAligner` 時，會自動下載模型。
:::

## 輸出結果

### 1. 繪製多邊形

繪製並保存帶有文件多邊形的圖像。

```python
# draw
result.draw_doc(
    folder='path/to/save/folder',
    name='output_image.jpg'
)
```

或是不指定路徑，直接輸出：

```python
# 預設的輸出路徑為當前目錄
# 預設的輸出檔名會調用當前時刻，為 f"output_{D.now()}.jpg"。
result.draw_doc()
```

![output_image](./resources/flat_result.jpg)

###  2. 取得 NumPy 圖像

如果你有其他需求，可以使用 `gen_doc_info_image` 方法，之後再自行處理。

```python
img = result.gen_doc_info_image()
```

### 3. 提取攤平後的圖像

如果你知道文件的原始大小，即可以使用 `gen_doc_flat_img` 方法，將文件圖像根據其多邊形邊界轉換為矩形圖像。

```python
H, W = 1080, 1920
flat_img = result.gen_doc_flat_img(image_size=(H, W))
```

如果是一個未知的影像類別，也可以不用給定 `image_size` 參數，此時將會根據文件多邊形的邊界自動計算出『**最小的矩形**』圖像，並將最小矩形的長寬設為 `H` 和 `W`。

:::tip
當你的文件在圖像中呈現大幅度傾斜時，可能會算出較扁平的最小矩形，此時進行攤平會有一定的形變。因此，建議在這種情況下，使用 `image_size` 參數進行手動設定。
:::

```python
flat_img = result.gen_doc_flat_img()
```
