# centercrop

> [centercrop(img: np.ndarray) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/functionals.py#L374)

- **説明**：入力画像に対してセンタークロップ処理を行います。

- **引数**

  - **img** (`np.ndarray`)：センタークロップ処理を行う入力画像。

- **戻り値**

  - **np.ndarray**：クロップされた画像。

- **例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  img = cb.imresize(img, [128, 256])
  crop_img = cb.centercrop(img)
  ```

  緑色の枠はセンタークロップされた領域を示します。

  ![centercrop](./resource/test_centercrop.jpg)
