# imopen

> [imopen(img: np.ndarray, ksize: int | tuple[int, int] = (3, 3), kstruct: str | int | MORPH = MORPH.RECT) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/morphology.py)

- **Description**: Opening operation: A process of erosion followed by dilation, which can be used to remove small objects, disconnect objects, smooth object edges, and eliminate small holes. For multi-channel images, each channel is processed independently.

- **Parameters**

  - **img** (`np.ndarray`): The input image.
  - **ksize** (`int | tuple[int, int]`): The size of the structuring element. Default is (3, 3).
  - **kstruct** (`str | int | MORPH`): Structuring element shape. Accepts `MORPH.CROSS/RECT/ELLIPSE`, string `"CROSS"/"RECT"/"ELLIPSE"`, or an OpenCV integer. Default is `MORPH.RECT`.

- **Example**

  ```python
  import numpy as np
  from capybara.vision.morphology import imopen

  img = np.array([[1, 1, 1, 0, 0],
                  [1, 1, 1, 0, 0],
                  [1, 1, 1, 0, 0],
                  [0, 0, 1, 1, 0], # <- Look at this row
                  [0, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1]], dtype=np.uint8)

  opened_img = imopen(img, ksize=3, kstruct='RECT')

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
