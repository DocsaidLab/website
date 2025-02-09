---
sidebar_position: 3
---

# 快速開始

我們提供了一個簡單的模型推論介面，其中包含了前後處理的邏輯。

首先，你需要導入所需的相關依賴並創建 `MRZScanner` 類別。

## 模型推論

:::info
我們有設計了自動下載模型的功能，當程式檢查你缺少模型時，會自動連接到我們的伺服器進行下載。
:::

以下是一個簡單的範例：

```python
import cv2
from skimage import io
from mrzscanner import MRZScanner

# build model
model = MRZScanner()

# read image
img = io.imread('https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg?raw=true')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# inference
result_mrz, error_msg = model(img)

# 輸出為 MRZ 區塊的兩行文字及錯誤訊息提示
print(result_mrz)
# >>> ('PCAZEQAQARIN<<FIDAN<<<<<<<<<<<<<<<<<<<<<<<<<',
#     'C946302620AZE6707297F23031072W12IMJ<<<<<<<40')
print(error_msg)
# >>> <ErrorCodes.NO_ERROR: 'No error.'>
```

:::tip
在上面範例中，圖片下載連結請參考：[**midv2020_test_mrz.jpg**](https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg)

<div align="center" >
<figure style={{width: "30%"}}>
![test_mrz](./resources/test_mrz.jpg)
</figure>
</div>
:::

## 使用 `do_center_crop` 參數

這張影像應該是與行動裝置拍攝的，形狀偏狹長，如果直接給模型推論的話，會造成過多的文字形變，因此我們調用模型的時候，同時開啟 `do_center_crop` 參數，方式如下：

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

## 搭配 `DocAligner` 使用

仔細看看上面的輸出結果，發現儘管做了 `do_center_crop` ，但有幾個錯字。

因為我們剛才使用全圖掃描，模型對於圖片中的文字可能會有一些誤判。

為了提高準確度，我們加入 `DocAligner` 來幫助我們對齊 MRZ 區塊：

```python
import cv2
from docaligner import DocAligner # 導入 DocAligner
from mrzscanner import MRZScanner
from capybara import imwarp_quadrangle
from skimage import io

model = MRZScanner()

doc_aligner = DocAligner()

img = io.imread('https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg?raw=true')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

polygon = doc_aligner(img)
flat_img = imwarp_quadrangle(img, polygon, dst_size=(800, 480))

print(model(flat_img))
# >>> ('PCAZEQAQARIN<<FIDAN<<<<<<<<<<<<<<<<<<<<<<<<<',
#      'C946302620AZE6707297F23031072W12IMJ<<<<<<<40')
```

使用 `DocAligner` 之後，就不需要再使用 `do_center_crop` 參數了。

現在，你可以看到輸出結果更加準確，這張圖片的 MRZ 區塊已經成功辨識。

## 錯誤訊息

為了讓使用者可以解釋錯誤的原因是什麼，我們設計了 `ErrorCodes` 類。

當模型推論出錯時，會得到錯誤訊息，內容涵蓋範圍為：

```python
class ErrorCodes(Enum):
    NO_ERROR = 'No error.'
    INVALID_INPUT_FORMAT = 'Invalid input format.'
    POSTPROCESS_FAILED_LINE_COUNT = 'Postprocess failed, number of lines not 2 or 3.'
    POSTPROCESS_FAILED_TD1_LENGTH = 'Postprocess failed, length of lines not 30 when `doc_type` is TD1.'
    POSTPROCESS_FAILED_TD2_TD3_LENGTH = 'Postprocess failed, length of lines not 36 or 44 when `doc_type` is TD2 or TD3.'
```

這裡會過濾一些基本的錯誤，例如輸入格式不正確、行數不正確等。

## 檢查碼

檢查碼 (Check Digit) 是 MRZ 中用來確保資料正確性的關鍵部分，它用來檢查數字的正確性，防止資料輸入錯誤。

- 詳細操作流程我們寫在 [**參考文獻：檢查碼**](./reference#檢查碼) 中。

---

在這一段中，我們要說的是：

- **我們沒有提供檢查碼的計算功能！**

因為 MRZ 的檢查碼計算方法並不是唯一的，除了正規的檢查碼計算方法，不同區域的 MRZ 可以自己重新定義檢查碼的計算方法，因此給定一個檢查碼計算方法可能會限制使用者的彈性。

:::info
順便分享一個冷知識：

臺灣的外國人居留證後面的 MRZ 的檢查碼跟世界的標準不一樣，如果沒有和政府合作開發的話，無法得知這個檢查碼的計算方法。
:::

我們的目標是訓練一個專注於 MRZ 辨識的模型，每個輸出由模型自動判定格式，檢查碼的計算功能有很多其他的開源專案可以使用，就如我們之前引用的 [**Arg0s1080/mrz**](https://github.com/Arg0s1080/mrz) 中就有提供檢查碼的計算方法，我們建議使用者可以直接使用這個專案。
