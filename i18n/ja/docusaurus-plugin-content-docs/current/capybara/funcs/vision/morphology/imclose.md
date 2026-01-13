# imclose

> [imclose(img: np.ndarray, ksize: int | tuple[int, int] = (3, 3), kstruct: str | int | MORPH = MORPH.RECT) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/morphology.py)

- **説明**：クロージング演算：膨張後に侵食を行う過程で、物体内部の小さな穴を埋めたり、物体のエッジを滑らかにしたり、二つの物体を接続するために使用されます。マルチチャネル画像の場合、各チャネルは個別に処理されます。

- **引数**

  - **img** (`np.ndarray`)：入力画像。
  - **ksize** (`Union[int, Tuple[int, int]]`)：構造要素のサイズ。デフォルトは (3, 3)。
  - **kstruct** (`str | int | MORPH`)：要素の形状。`MORPH.CROSS/RECT/ELLIPSE`、文字列 `"CROSS"/"RECT"/"ELLIPSE"`、または OpenCV の整数値を指定できます。デフォルトは `MORPH.RECT`。

- **例**

  ```python
  import numpy as np
  from capybara.vision.morphology import imclose

  img = np.array([[1, 1, 1, 0, 0],
                  [1, 1, 1, 0, 0],
                  [1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0], # <- Look at this row
                  [0, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1]], dtype=np.uint8)

  closed_img = imclose(img, ksize=3, kstruct='CROSS')

  # Kernel will be like this:
  # >>> np.array([[0, 1, 0],
  #               [1, 1, 1],
  #               [0, 1, 0]], dtype=np.uint8)

  # After closing, the image will be like this:
  # >>> np.array([[1, 1, 1, 0, 0],
  #               [1, 1, 1, 0, 0],
  #               [1, 1, 1, 0, 0],
  #               [0, 0, 1, 1, 0], # <- 1's are connected
  #               [0, 0, 0, 1, 1],
  #               [0, 0, 0, 1, 1]], dtype=np.uint8)
  ```
