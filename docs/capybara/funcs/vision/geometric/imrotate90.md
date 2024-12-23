# imrotate90

> [imrotate90(img: np.ndarray, rotate_code: ROTATE) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/geometric.py#L66)

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
  import capybara as cb

  img = cb.imread('lena.png')
  rotate_img = cb.imrotate90(img, cb.ROTATE.ROTATE_270)
  ```

  ![imrotate90](./resource/test_imrotate90.jpg)
