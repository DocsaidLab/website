# medianblur

> [medianblur(img: np.ndarray, ksize: int = 3, **kwargs) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/functionals.py)

- **說明**：對輸入影像套用中值模糊處理。

- **參數**

  - **img** (`np.ndarray`)：要進行模糊處理的輸入影像。
  - **ksize** (`int`)：中值濾波 kernel size，需為正奇數。預設為 3。

- **傳回值**

  - **np.ndarray**：模糊處理後的影像。

- **範例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  blur_img = cb.medianblur(img, ksize=5)
  ```

  ![medianblur](./resource/test_medianblur.jpg)
