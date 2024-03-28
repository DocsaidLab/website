---
sidebar_position: 6
---

>[imbinarize(img: np.ndarray, threth: int = cv2.THRESH_BINARY, color_base: str = 'BGR') -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/functionals.py#L336)

- **說明**：對輸入影像進行二值化處理。

- **參數**

    - **img** (`np.ndarray`)：要進行二值化處理的輸入影像。如果輸入影像是 3 通道，則函數會自動應用 `bgr2gray` 函數。
    - **threth** (`int`)：閾值類型。有兩種閾值類型：
        1. `cv2.THRESH_BINARY`：`cv2.THRESH_OTSU + cv2.THRESH_BINARY`
        2. `cv2.THRESH_BINARY_INV`：`cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV`
    - **color_base** (`str`)：輸入影像的顏色空間。預設為 `'BGR'`。

- **傳回值**

    - **np.ndarray**：二值化後的影像。

- **範例**

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    bin_img = D.imbinarize(img)
    ```

    ![imbinarize](./resource/test_imbinarize.jpg)
