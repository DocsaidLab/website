# gaussianblur

> [gaussianblur(img: np.ndarray, ksize: int | tuple[int, int] | np.ndarray = 3, sigma_x: int = 0, **kwargs) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/functionals.py)

- **說明**：對輸入影像套用高斯模糊處理。

- 參數

  - **img** (`np.ndarray`)：要進行模糊處理的輸入影像。
  - **ksize** (`Union[int, Tuple[int, int]]`)：用於模糊處理的核心大小。如果提供了整數值，則使用指定大小的正方形核。如果提供了元組`(k_height, k_width)`，則使用指定大小的矩形核。預設為 3。
  - **sigma_x** (`int`)：高斯核心的 X 方向標準差。預設為 0。

- **備註**

  - 為了兼容舊版呼叫方式，亦接受 `sigmaX=...`。

- **傳回值**

  - **np.ndarray**：模糊處理後的影像。

- **範例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  blur_img = cb.gaussianblur(img, ksize=5)
  ```

  ![gaussianblur](./resource/test_gaussianblur.jpg)
