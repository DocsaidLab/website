# meanblur

> [meanblur(img: np.ndarray, ksize: \_Ksize = 3, \*\*kwargs) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/functionals.py#L33)

- **說明**：對輸入影像套用均值模糊處理。

- **參數**

  - **img** (`np.ndarray`)：要進行模糊處理的輸入影像。
  - **ksize** (`Union[int, Tuple[int, int]]`)：用於模糊處理的核心大小。如果提供了整數值，則使用指定大小的正方形核。如果提供了元組`(k_height, k_width)`，則使用指定大小的矩形核。預設為 3。

- **傳回值**

  - **np.ndarray**：模糊處理後的影像。

- **範例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  blur_img = cb.meanblur(img, ksize=5)
  ```

  ![meanblur](./resource/test_meanblur.jpg)
