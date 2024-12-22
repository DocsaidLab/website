# centercrop

> [centercrop(img: np.ndarray) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/functionals.py#L374)

- **說明**：對輸入影像進行中心裁剪處理。

- **參數**

  - **img** (`np.ndarray`)：要進行中心裁剪處理的輸入影像。

- **傳回值**

  - **np.ndarray**：裁剪後的影像。

- **範例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  img = cb.imresize(img, [128, 256])
  crop_img = cb.centercrop(img)
  ```

  綠色框表示中心裁剪的區域。

  ![centercrop](./resource/test_centercrop.jpg)
