# imdilate

> [imdilate(img: np.ndarray, ksize: int | tuple[int, int] = (3, 3), kstruct: str | int | MORPH = MORPH.RECT) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/morphology.py)

- **說明**：膨脹操作：使用指定的結構元素膨脹源圖像，該結構元素確定了取最大值的像素鄰域的形狀。對於多通道圖像，每個通道都將獨立處理。

- **參數**

  - **img** (`np.ndarray`)：輸入圖像。
  - **ksize** (`Union[int, Tuple[int, int]]`)：結構元素的大小。預設為 (3, 3)。
  - **kstruct** (`str | int | MORPH`)：元素形狀。可用 `MORPH.CROSS/RECT/ELLIPSE`、字串 `"CROSS"/"RECT"/"ELLIPSE"` 或 OpenCV 的整數值。預設為 `MORPH.RECT`。

- **範例**

  ```python
  import numpy as np
  from capybara.vision.morphology import imdilate

  img = np.array([[0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0]], dtype=np.uint8)

  dilated_img = imdilate(img, ksize=3, kstruct='RECT')

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
