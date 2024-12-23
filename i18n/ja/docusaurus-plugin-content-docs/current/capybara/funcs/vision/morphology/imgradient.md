# imgradient

> [imgradient(img: np.ndarray, ksize: Union[int, Tuple[int, int]] = (3, 3), kstruct: Union[str, int, "MORPH"] = "MORPH.RECT") -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/morphology.py#L135)

- **説明**：勾配演算：膨張画像から侵食画像を引いた結果です。マルチチャネル画像の場合、各チャネルは個別に処理されます。この手法は物体のエッジを抽出するために使用されます。

- **引数**

  - **img** (`np.ndarray`)：入力画像。
  - **ksize** (`Union[int, Tuple[int, int]]`)：構造要素のサイズ。デフォルトは (3, 3)。
  - **kstruct** (`MORPH`)：要素の形状。`"MORPH.CROSS"`, `"MORPH.RECT"`, `"MORPH.ELLIPSE"` のいずれか。デフォルトは `"MORPH.RECT"`。

- **例**

  ```python
  import numpy as np
  import capybara as cb

  img = np.array([[0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0]], dtype=np.uint8)

  gradient_img = cb.imgradient(img, ksize=3, kstruct='RECT')

  # Kernel will be like this:
  # >>> np.array([[1, 1, 1],
  #               [1, 1, 1],
  #               [1, 1, 1]], dtype=np.uint8)

  # After gradient, the image will be like this:
  # >>> np.array([[1, 1, 1, 1, 1],
  #               [1, 1, 1, 1, 1],
  #               [1, 1, 0, 1, 1],
  #               [1, 1, 1, 1, 1],
  #               [1, 1, 1, 1, 1]], dtype=np.uint8)
  ```
