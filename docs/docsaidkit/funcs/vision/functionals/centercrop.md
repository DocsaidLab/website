---
sidebar_position: 8
---

# centercrop

>[centercrop(img: np.ndarray) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/functionals.py#L374)

- **說明**：對輸入影像進行中心裁剪處理。

- **參數**

    - **img** (`np.ndarray`)：要進行中心裁剪處理的輸入影像。

- **傳回值**

    - **np.ndarray**：裁剪後的影像。

- **範例**

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    img = D.imresize(img, [128, 256])
    crop_img = D.centercrop(img)
    ```

    綠色框表示中心裁剪的區域。

    ![centercrop](./resource/test_centercrop.jpg)
