# imclose

> [imclose(img: np.ndarray, ksize: int | tuple[int, int] = (3, 3), kstruct: str | int | MORPH = MORPH.RECT) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/morphology.py)

- **說明**：閉運算：先膨脹再侵蝕的過程，可以用來填充物體內部的小洞、平滑物體的邊緣、連接兩個物體等。對於多通道圖像，每個通道都將獨立處理。

- **參數**

  - **img** (`np.ndarray`)：輸入圖像。
  - **ksize** (`Union[int, Tuple[int, int]]`)：結構元素的大小。預設為 (3, 3)。
  - **kstruct** (`str | int | MORPH`)：元素形狀。可用 `MORPH.CROSS/RECT/ELLIPSE`、字串 `"CROSS"/"RECT"/"ELLIPSE"` 或 OpenCV 的整數值。預設為 `MORPH.RECT`。

- **範例**

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
