---
sidebar_position: 3
---

# 快速開始

我們提供了一個簡單的模型推論介面，其中包含了前後處理的邏輯。

## 模型推論

首先，什麼都別管，跑跑看以下程式碼，看一下能不能完整執行：

```python
import cv2
from skimage import io
from mrzscanner import MRZScanner

# 建立模型
model = MRZScanner()

# 讀取線上影像
img = io.imread('https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg?raw=true')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# 模型推論
result = model(img, do_center_crop=True, do_postprocess=False)

# 輸出結果
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
```

成功執行後，我們往下看一下程式碼的細節。

:::info
我們有設計了自動下載模型的功能，當程式檢查你缺少模型時，會自動連接到我們的伺服器進行下載。
:::

:::tip
在上面範例中，圖片下載連結請參考：[**midv2020_test_mrz.jpg**](https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg)

<div align="center" >
<figure style={{width: "30%"}}>
![test_mrz](./resources/test_mrz.jpg)
</figure>
</div>
:::

## 使用 `do_center_crop` 參數

這張影像應該是與行動裝置拍攝的，形狀偏狹長，如果直接給模型推論的話，會造成過多的文字形變。所以我們在推論的時候加入了 `do_center_crop` 參數，這個參數是用來對圖片進行中心裁剪。

這個參數預設為 `False`，因為我們認為在未經過使用者的認知下，不應該對圖片進行任何的修改。但是在實際應用中，我們遇到的圖像往往並非標準的正方形尺寸。

實際上，圖像的尺寸和比例多種多樣，例如：

- 手機拍攝的照片普遍採用 9:16 的寬高比；
- 掃描的文件常見於 A4 的紙張比例；
- 網頁截圖大多是 16:9 的寬高比；
- 透過 webcam 拍攝的圖片，則通常是 4:3 的比例。

這些非正方形的圖像，在不經過適當處理直接進行推論時，往往會包含大量的無關區域或空白，從而對模型的推論效果產生不利影響。進行中心裁剪能夠有效減少這些無關區域，專注於圖像的中心區域，從而提高推論的準確性和效率。

使用方式如下：

```python
from mrzscanner import MRZScanner

model = MRZScanner()

result = model(img, do_center_crop=True) # 使用中心裁剪
```

:::tip
**使用時機**：『不會切到 MRZ 區域』且圖片比例不是正方形時，可以使用中心裁切。
:::

:::info
`MRZScanner` 已經用 `__call__` 進行了封裝，因此你可以直接呼叫實例進行推論。
:::

## 使用 `do_postprocess` 參數

除了中心裁剪外，我們還提供了一個後處理的選項 `do_postprocess`，用於進一步提高模型的準確性。

這個參數預設同樣是 `False`，原因和剛才一樣，我們認為在未經過使用者的認知下，不應該對辨識結果進行任何的修改。

在實際應用中，MRZ 區塊中存在一些規則，例如：國家代碼只能為大寫英文字母、性別只有 `M` 和 `F` 以及跟日期有關的欄位只能是數字等。這些規則都可以用來規範 MRZ 區塊。

因此我們針對可以規範的區塊進行人工校正，以下實作校正概念的程式碼片段，在不可能出現數字的欄位中，把可能誤判的數字替換成正確的字元：

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

雖然在我們的專案中，這個後處理沒有幫我們提高更多的準確度，但保留這個功能還是可以在某些情況下把錯誤的辨識結果修正回來。

你可以考慮推論的時候把 `do_postprocess` 設為 `True`，通常結果會更好：

```python
result = model(img, do_postprocess=True)
```

又或是你更喜歡看到原始的模型輸出結果，那就用預設值即可。

## 搭配 `DocAligner` 使用

有時候就算使用了 `do_center_crop` 參數，也有偵測失敗的可能，這時候我們可以使用 `DocAligner` 來幫助我們先找出證件的位置，然後再進行 MRZ 辨識。

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
# {
#     'mrz_polygon':
#         array(
#         [
#             [ 34.0408 , 378.497  ],
#             [756.4258 , 385.0492 ],
#             [755.8944 , 443.63843],
#             [ 33.5094 , 437.08618]
#         ], dtype=float32
#     ),
#     'mrz_texts': [
#         'PCAZEQAQARIN<<FIDAN<<<<<<<<<<<<<<<<<<<<<<<<<',
#         'C946302620AZE6707297F23031072W12IMJ<<<<<<<40'
#     ],
#     'msg': <ErrorCodes.NO_ERROR: 'No error.'>
# }
```

:::warning
如果使用 `DocAligner` 做前處理，表示 MRZ 可能已經佔據了一定的版面，這時候就不需要再使用 `do_center_crop` 參數了，因為中心裁切可能會導致 MRZ 部分被切掉。
:::

:::tip
`DocAligner` 的使用方式請參考 [**DocAligner 技術文件**](https://docsaid.org/docs/docaligner/)。
:::

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
