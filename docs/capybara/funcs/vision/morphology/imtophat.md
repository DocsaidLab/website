# imtophat

> [imtophat(img: np.ndarray, ksize: int | tuple[int, int] = (3, 3), kstruct: str | int | MORPH = MORPH.RECT) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/morphology.py)

- **說明**：頂帽運算：原圖像減去開運算的結果。對於多通道圖像，每個通道都將獨立處理。意義是可以用來提取比原圖像亮的區域，例如亮點或細小結構，同時去除或減弱大面積的亮區域。

- **參數**

  - **img** (`np.ndarray`)：輸入圖像。
  - **ksize** (`Union[int, Tuple[int, int]]`)：結構元素的大小。預設為 (3, 3)。
  - **kstruct** (`str | int | MORPH`)：元素形狀。可用 `MORPH.CROSS/RECT/ELLIPSE`、字串 `"CROSS"/"RECT"/"ELLIPSE"` 或 OpenCV 的整數值。預設為 `MORPH.RECT`。

- **範例**

  ```python
  import numpy as np
  from capybara.vision.morphology import imtophat

  img = np.array([[1, 1, 1, 0, 0],
                  [1, 1, 1, 0, 0],
                  [1, 1, 1, 0, 0],
                  [0, 0, 1, 1, 0],
                  [0, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1]], dtype=np.uint8)

  tophat_img = imtophat(img, ksize=3, kstruct='RECT')

  # Kernel will be like this:
  # >>> np.array([[1, 1, 1],
  #               [1, 1, 1],
  #               [1, 1, 1]], dtype=np.uint8)

  # After tophat, the image will be like this:
  # >>> np.array([[0, 0, 0, 0, 0],
  #               [0, 0, 0, 0, 0],
  #               [0, 0, 0, 0, 0],
  #               [0, 0, 1, 1, 0],
  #               [0, 0, 0, 0, 0],
  #               [0, 0, 0, 0, 0]], dtype=np.uint8)
  ```
