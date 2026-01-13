# centercrop

> [centercrop(img: np.ndarray) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/functionals.py)

- **説明**：入力画像に対してセンタークロップ処理を行います。

- **引数**

  - **img** (`np.ndarray`)：センタークロップ処理を行う入力画像。

- **戻り値**

  - **np.ndarray**：クロップされた画像。

- **例**

  ```python
  import capybara as cb
  from capybara.vision.functionals import centercrop

  img = cb.imread('lena.png')
  img = cb.imresize(img, [128, 256])
  crop_img = centercrop(img)
  ```

  緑色の枠はセンタークロップされた領域を示します。

  ![centercrop](./resource/test_centercrop.jpg)
