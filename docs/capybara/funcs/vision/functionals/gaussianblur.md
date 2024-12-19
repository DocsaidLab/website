---
sidebar_position: 2
---

# gaussianblur

>[gaussianblur(img: np.ndarray, ksize: _Ksize = 3, sigmaX: int = 0, **kwargs) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/functionals.py#L54)

- **說明**：對輸入影像套用高斯模糊處理。

- 參數

    - **img** (`np.ndarray`)：要進行模糊處理的輸入影像。
    - **ksize** (`Union[int, Tuple[int, int]]`)：用於模糊處理的核心大小。如果提供了整數值，則使用指定大小的正方形核。如果提供了元組`(k_height, k_width)`，則使用指定大小的矩形核。預設為 3。
    - **sigmaX** (`int`)：高斯核心的 X 方向標準差。預設為 0。

- **傳回值**

    - **np.ndarray**：模糊處理後的影像。

- **範例**

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    blur_img = D.gaussianblur(img, ksize=5)
    ```

    ![gaussianblur](./resource/test_gaussianblur.jpg)

