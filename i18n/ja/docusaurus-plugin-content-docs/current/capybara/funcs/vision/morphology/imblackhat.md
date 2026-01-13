# imblackhat

> [imblackhat(img: np.ndarray, ksize: int | tuple[int, int] = (3, 3), kstruct: str | int | MORPH = MORPH.RECT)](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/morphology.py)

- **説明**：ブラックハット演算：閉演算の結果から元の画像を引いたものです。マルチチャネル画像の場合、各チャネルは個別に処理されます。この手法は、元の画像よりも暗い領域を抽出するために使用され、暗点や細かな構造を強調することができます。また、大面積の暗い領域を除去または弱めることができます。

- **引数**

  - **img** (`np.ndarray`)：入力画像。
  - **ksize** (`Union[int, Tuple[int, int]]`)：構造要素のサイズ。デフォルトは (3, 3)。
  - **kstruct** (`str | int | MORPH`)：要素の形状。`MORPH.CROSS/RECT/ELLIPSE`、文字列 `"CROSS"/"RECT"/"ELLIPSE"`、または OpenCV の整数値を指定できます。デフォルトは `MORPH.RECT`。

- **例**

  ```python
  import numpy as np
  from capybara.vision.morphology import imblackhat

  img = np.array([[1, 1, 1, 0, 0],
                  [1, 1, 1, 0, 0],
                  [1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0], # <- Look at this row
                  [0, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1]], dtype=np.uint8)

  blackhat_img = imblackhat(img, ksize=3, kstruct='CROSS')

  # Kernel will be like this:
  # >>> np.array([[0, 1, 0],
  #               [1, 1, 1],
  #               [0, 1, 0]], dtype=np.uint8)

  # After blackhat, the image will be like this:
  # >>> np.array([[0, 0, 0, 0, 0],
  #               [0, 0, 0, 0, 0],
  #               [0, 0, 0, 0, 0],
  #               [0, 0, 1, 1, 0], # <- 1's are extracted
  #               [0, 0, 0, 0, 0],
  #               [0, 0, 0, 0, 0]], dtype=np.uint8)
  ```
