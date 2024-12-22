# imcvtcolor

> [imcvtcolor(img: np.ndarray, cvt_mode: Union[int, str]) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/functionals.py#L96)

- **說明**：對輸入影像進行顏色空間轉換。

- **參數**

  - **img** (`np.ndarray`)：要進行轉換的輸入影像。
  - **cvt_mode** (`Union[int, str]`)：顏色轉換模式。可以是表示轉換代碼的整數常數，也可以是表示 OpenCV 顏色轉換名稱的字串。例如：`BGR2GRAY` 用於轉換 BGR 影像為灰階。可用參數請直接參考 [**OpenCV COLOR**](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html)。

- **傳回值**

  - **np.ndarray**：具有所需顏色空間的影像。

- **範例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  ycrcb_img = cb.imcvtcolor(img, 'BGR2YCrCb')
  ```

  ![imcvtcolor_ycrcb](./resource/test_imcvtcolor_ycrcb.jpg)

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  ycrcb_img = cb.imcvtcolor(img, 'BGR2YCrCb')
  ```

  ![imcvtcolor_gray](./resource/test_imcvtcolor_gray.jpg)
