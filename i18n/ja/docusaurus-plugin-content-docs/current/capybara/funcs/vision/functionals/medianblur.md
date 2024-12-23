# medianblur

> [medianblur(img: np.ndarray, ksize: \_Ksize = 3, \*\*kwargs) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/functionals.py#L78)

- **説明**：入力画像に中値ぼかし処理を適用します。

- **引数**

  - **img** (`np.ndarray`)：ぼかし処理を行う入力画像。
  - **ksize** (`Union[int, Tuple[int, int]]`)：ぼかし処理に使用するカーネルのサイズ。整数値を指定すると、指定されたサイズの正方形カーネルを使用します。タプル `(k_height, k_width)` を指定すると、指定されたサイズの矩形カーネルを使用します。デフォルトは 3。

- **戻り値**

  - **np.ndarray**：ぼかし処理後の画像。

- **例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  blur_img = cb.medianblur(img, ksize=5)
  ```

  ![medianblur](./resource/test_medianblur.jpg)
