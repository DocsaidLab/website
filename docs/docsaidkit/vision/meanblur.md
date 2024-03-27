---
sidebar_position: 1
---

---

source: [docsaidkit/vision/functionals#meanblur](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/functionals.py#L33)

---

`meanblur` 函數提供了一種對影像應用均值模糊（Mean Blur）的處理方式。均值模糊是一種簡單的影像模糊技術，透過對影像中每個像素及其周圍像素的值取平均，達到模糊影像的效果。 此方法常用於影像預處理，以減少影像雜訊或降低影像細節的複雜度。

## 參數

- `img` (`np.ndarray`): 需要模糊處理的輸入影像，應為 NumPy 陣列格式。
- `ksize` (`Union[int, Tuple[int, int]]`): 模糊核的大小。 如果提供整數值，則使用指定大小的正方形核。如果提供元組 `(k_height, k_width)`，則使用指定大小的矩形核。 預設值為 3。

## 傳回值

- `np.ndarray`: 經過均值模糊處理的影像，以 NumPy 陣列格式傳回。

## 使用範例

```python
import cv2
from docsaidkit import meanblur

# 載入圖片
img = cv2.imread('your_image_path.jpg')

# 應用均值模糊，使用 5x5 的核
blurred_img = meanblur(img, ksize=5)

# 顯示原始影像和模糊影像
cv2.imshow('Original Image', img)
cv2.imshow('Blurred Image', blurred_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

在這個範例中，我們首先定義了 `meanblur` 函數，並在函數內部根據 `ksize` 參數的類型調整模糊核的大小。之後，我們載入了一張圖像，應用了均值模糊處理，並展示了處理前後的圖像。 請將 `'your_image_path.jpg'` 替換為您的圖片檔案路徑。

透過調整 `ksize` 參數，您可以控制模糊的程度，較大的 `ksize` 值會產生更模糊的效果。
