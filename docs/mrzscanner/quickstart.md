---
sidebar_position: 3
---

# 快速開始

我們提供了一個簡單的模型推論介面，其中包含了前後處理的邏輯。

首先，你需要導入所需的相關依賴並創建 `MRZScanner` 類別。

## 模型推論

以下是一個簡單的範例，展示如何使用 `MRZScanner` 進行模型推論：

```python
from mrzscanner import MRZScanner

model = MRZScanner()
```

啟動模型之後，接著要準備一張圖片進行推論：

:::tip
你可以使用 `MRZScanner` 提供的測試圖片：

下載連結：[**midv2020_test_mrz.jpg**](https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg)

<div align="center" >
<figure style={{width: "50%"}}>
![test_mrz](./resources/test_mrz.jpg)
</figure>
</div>
:::

```python
import docsaidkit as D

img = D.imread('path/to/run_test_card.jpg')
```

或是你可以直接透過 URL 進行讀取：

```python
import cv2
from skimage import io

img = io.imread('https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg?raw=true')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
```

這張影像太長了，如果直接推論的話，會造成過多的文字形變，因此我們調用模型的時候，同時開啟 `do_center_crop` 參數：

接著，我們可以使用 `model` 進行推論：

```python
from mrzscanner import MRZScanner

model = MRZScanner()

result, msg = model(img, do_center_crop=True)
print(result)
# >>> ('PCAZEQAOARIN<<FIDAN<<<<<<<<<<<<<<<<<<<<<<<<<',
#      'C946302620AZE6707297F23031072W12IMJ<<<<<<<40')
print(msg)
# >>> <ErrorCodes.NO_ERROR: 'No error.'>
```

:::tip
`MRZScanner` 已經用 `__call__` 進行了封裝，因此你可以直接呼叫實例進行推論。
:::

:::info
我們有設計了自動下載模型的功能，當你第一次使用 `MRZScanner` 時，會自動下載模型。
:::

## 搭配 `DocAligner` 使用

仔細看看上面的輸出結果，發現儘管做了 `do_center_crop` ，但有幾個錯字。

因為我們剛才使用全圖掃描，模型對於圖片中的文字可能會有一些誤判。

為了提高準確度，我們加入 `DocAligner` 來幫助我們對齊 MRZ 區塊：

```python
import cv2
from docaligner import DocAligner # 導入 DocAligner
from mrzscanner import MRZScanner
from skimage import io

model = MRZScanner()

doc_aligner = DocAligner()

img = io.imread(
    'https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg?raw=true')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

flat_img = doc_aligner(img).doc_flat_img # 對齊 MRZ 區塊
print(model(flat_img))
# >>> ('PCAZEQAQARIN<<FIDAN<<<<<<<<<<<<<<<<<<<<<<<<<',
#      'C946302620AZE6707297F23031072W12IMJ<<<<<<<40')
```

使用 `DocAligner` 之後，就不需要再使用 `do_center_crop` 參數了。

現在，你可以看到輸出結果更加準確了，這張圖片的 MRZ 區塊已經成功辨識。

## 錯誤訊息

為了讓使用者可以解釋錯誤的原因是什麼，我們有回傳錯誤訊息的欄位，內容涵蓋範圍為：

```python
class ErrorCodes(Enum):
    NO_ERROR = 'No error.'
    INVALID_INPUT_FORMAT = 'Invalid input format.'
    POSTPROCESS_FAILED_LINE_COUNT = 'Postprocess failed, number of lines not 2 or 3.'
    POSTPROCESS_FAILED_TD1_LENGTH = 'Postprocess failed, length of lines not 30 when `doc_type` is TD1.'
    POSTPROCESS_FAILED_TD2_TD3_LENGTH = 'Postprocess failed, length of lines not 36 or 44 when `doc_type` is TD2 or TD3.'
```

主要為了對輸出結果進行初步的過濾，一些肉眼可見的錯誤，例如字串長度不對，或是行數不對等，都可以在這裡被檢查出來。

## 檢查碼

檢查碼 (Check Digit) 是 MRZ 中用來確保資料正確性的關鍵部分，它用來檢查數字的正確性，防止資料輸入錯誤。

- 詳細操作流程我們寫在 [**參考文獻：檢查碼**](./reference#檢查碼) 中。

---

這裡我們要說的是：我們「沒有提供檢查碼」的計算功能。

除了正規的檢查碼計算方法，不同區域的 MRZ 可以自己重新定義檢查碼的計算方法，因此給定一個檢查碼計算方法可能會限制使用者的彈性。

此外，我們的目標是訓練一個專注於 MRZ 辨識的模型，每個輸出由模型自動判定格式，如果要套用檢查碼的話，必須預先知道辨識目標，若不然，就要依序把每個格式的檢查碼計算方法都算過一次，這顯然不是我們的目標。

檢查碼的計算功能有很多其他的開源專案可以使用，就如我們之前引用的 [**Arg0s1080/mrz**](https://github.com/Arg0s1080/mrz) 中就有提供檢查碼的計算方法，我們建議使用者可以直接使用這個專案。
