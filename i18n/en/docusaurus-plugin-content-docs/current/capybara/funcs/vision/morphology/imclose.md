# imclose

> [imclose(img: np.ndarray, ksize: Union[int, Tuple[int, int]] = (3, 3), kstruct: Union[str, int, "MORPH"] = "MORPH.RECT") -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/morphology.py#L105)

- **Description**: Closing operation: a process of dilation followed by erosion, which can be used to fill small holes inside objects, smooth object edges, or connect two objects. For multi-channel images, each channel is processed independently.

- **Parameters**

  - **img** (`np.ndarray`): The input image.
  - **ksize** (`Union[int, Tuple[int, int]]`): The size of the structuring element. Default is (3, 3).
  - **kstruct** (`MORPH`): The shape of the structuring element, which can be one of "MORPH.CROSS", "MORPH.RECT", or "MORPH.ELLIPSE". Default is "MORPH.RECT".

- **Example**

  ```python
  import numpy as np
  import capybara as cb

  img = np.array([[1, 1, 1, 0, 0],
                  [1, 1, 1, 0, 0],
                  [1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0], # <- Look at this row
                  [0, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1]], dtype=np.uint8)

  closed_img = cb.imclose(img, ksize=3, kstruct='CROSS')

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
