# imbinarize

> [imbinarize(img: np.ndarray, threth: int = cv2.THRESH_BINARY, color_base: str = 'BGR') -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/functionals.py#L336)

- **説明**：入力画像に対して二値化処理を行います。

- **引数**

  - **img** (`np.ndarray`)：二値化処理を行う入力画像。もし入力画像が 3 チャネルであれば、関数は自動的に `bgr2gray` 関数を適用します。
  - **threth** (`int`)：閾値の種類。次の 2 つの閾値タイプがあります：
    1. `cv2.THRESH_BINARY`：`cv2.THRESH_OTSU + cv2.THRESH_BINARY`
    2. `cv2.THRESH_BINARY_INV`：`cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV`
  - **color_base** (`str`)：入力画像の色空間。デフォルトは `'BGR'`。

- **戻り値**

  - **np.ndarray**：二値化後の画像。

- **例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  bin_img = cb.imbinarize(img)
  ```

  ![imbinarize](./resource/test_imbinarize.jpg)
