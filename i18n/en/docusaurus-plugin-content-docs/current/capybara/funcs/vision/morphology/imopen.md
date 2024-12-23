# imopen

> [imopen(img: np.ndarray, ksize: Union[int, Tuple[int, int]] = (3, 3), kstruct: Union[str, int, "MORPH"] = "MORPH.RECT") -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/morphology.py#L76)

- **Description**: Opening operation: A process of erosion followed by dilation, which can be used to remove small objects, disconnect objects, smooth object edges, and eliminate small holes. For multi-channel images, each channel is processed independently.

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
                  [0, 0, 1, 1, 0], # <- Look at this row
                  [0, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1]], dtype=np.uint8)

  opened_img = cb.imopen(img, ksize=3, kstruct='RECT')

  # Kernel will be like this:
  # >>> np.array([[1, 1, 1],
  #               [1, 1, 1],
  #               [1, 1, 1]], dtype=np.uint8)

  # After opening, the image will be like this:
  # >>> np.array([[1, 1, 1, 0, 0],
  #               [1, 1, 1, 0, 0],
  #               [1, 1, 1, 0, 0],
  #               [0, 0, 0, 0, 0], # <- 1's are removed
  #               [0, 0, 0, 1, 1],
  #               [0, 0, 0, 1, 1]], dtype=np.uint8)
  ```
