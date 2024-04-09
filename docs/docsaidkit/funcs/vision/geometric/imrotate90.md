---
sidebar_position: 2
---

# imrotate90

>[imrotate90(img: np.ndarray, rotate_code: ROTATE) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/geometric.py#L66C1-L77C47)


- **說明**：對輸入影像進行 90 度旋轉處理。

- **參數**

    - **img** (`np.ndarray`)：要進行旋轉處理的輸入影像。
    - **rotate_code** (`RotateCode`)：旋轉程式碼。可用選項有：
        - ROTATE.ROTATE_90： 90 度。
        - ROTATE.ROTATE_180：旋轉 180 度。
        - ROTATE.ROTATE_270：逆時針旋轉 90 度。

- **傳回值**

    - **np.ndarray**：旋轉後的影像。

- **範例**

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    rotate_img = D.imrotate90(img, D.ROTATE.ROTATE_270)
    ```

    ![imrotate90](./resource/test_imrotate90.jpg)
