# imblackhat

> [imblackhat(img: np.ndarray, ksize: Union[int, Tuple[int, int]] = (3, 3), kstruct: Union[str, int, "MORPH"] = "MORPH.RECT") -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/morphology.py#L191)

- **說明**：黑帽運算：閉運算的結果減去原圖像。對於多通道圖像，每個通道都將獨立處理。意義是可以用來提取比原圖像暗的區域，例如暗點或細小結構，同時去除或減弱大面積的暗區域。

- **參數**

  - **img** (`np.ndarray`)：輸入圖像。
  - **ksize** (`Union[int, Tuple[int, int]]`)：結構元素的大小。預設為 (3, 3)。
  - **kstruct** (`MORPH`)：元素形狀，可以是 "MORPH.CROSS", "MORPH.RECT", "MORPH.ELLIPSE" 之一。預設為 "MORPH.RECT"。

- **範例**

  ```python
  import numpy as np
  import capybara as cb

  img = np.array([[1, 1, 1, 0, 0],
                  [1, 1, 1, 0, 0],
                  [1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0], # <- Look at this row
                  [0, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1]], dtype=np.uint8)

  blackhat_img = cb.imblackhat(img, ksize=3, kstruct='CROSS')

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
