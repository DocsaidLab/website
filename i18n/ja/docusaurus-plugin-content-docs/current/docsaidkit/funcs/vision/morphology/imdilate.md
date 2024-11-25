---
sidebar_position: 2
---

# imdilate

> [imdilate(img: np.ndarray, ksize: Union[int, Tuple[int, int]] = (3, 3), kstruct: Union[str, int, "MORPH"] = "MORPH.RECT") -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/morphology.py#L45)

- **説明**：膨張操作：指定された構造要素を使用して元の画像を膨張させ、その構造要素が最大値を取るピクセルの近隣形状を決定します。複数チャンネル画像の場合、各チャンネルは独立して処理されます。

- 引数

  - **img** (`np.ndarray`)：入力画像。
  - **ksize** (`Union[int, Tuple[int, int]]`)：構造要素のサイズ。デフォルトは (3, 3)。
  - **kstruct** (`MORPH`)：要素の形状。`"MORPH.CROSS"`, `"MORPH.RECT"`, `"MORPH.ELLIPSE"` のいずれか。デフォルトは `"MORPH.RECT"`。

- **例**

  ```python
  import numpy as np
  import docsaidkit as D

  img = np.array([[0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0]], dtype=np.uint8)

  dilated_img = D.imdilate(img, ksize=3, kstruct='RECT')

  # Kernel will be like this:
  # >>> np.array([[1, 1, 1],
  #               [1, 1, 1],
  #               [1, 1, 1]], dtype=np.uint8)

  # After dilation, the image will be like this:
  # >>> np.array([[1, 1, 1, 1, 1],
  #               [1, 1, 1, 1, 1],
  #               [1, 1, 1, 1, 1],
  #               [1, 1, 1, 1, 1],
  #               [1, 1, 1, 1, 1]], dtype=np.uint8)
  ```
