# imerode

> [imerode(img: np.ndarray, ksize: int | tuple[int, int] = (3, 3), kstruct: str | int | MORPH = MORPH.RECT) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/morphology.py)

- **Description**: Erosion operation: Erodes the source image using the specified structuring element, which determines the shape of the pixel neighborhood where the minimum value is taken. For multi-channel images, each channel is processed independently.

- **Parameters**

  - **img** (`np.ndarray`): The input image.
  - **ksize** (`int | tuple[int, int]`): The size of the structuring element. Default is (3, 3).
  - **kstruct** (`str | int | MORPH`): Structuring element shape. Accepts `MORPH.CROSS/RECT/ELLIPSE`, string `"CROSS"/"RECT"/"ELLIPSE"`, or an OpenCV integer. Default is `MORPH.RECT`.

- **Example**

  ```python
  import numpy as np
  from capybara.vision.morphology import imerode

  img = np.array([[0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0]], dtype=np.uint8)

  eroded_img = imerode(img, ksize=3, kstruct='RECT')

  # Kernel will be like this:
  # >>> np.array([[1, 1, 1],
  #               [1, 1, 1],
  #               [1, 1, 1]], dtype=np.uint8)

  # After erosion, the image will be like this:
  # >>> np.array([[0, 0, 0, 0, 0],
  #               [0, 0, 0, 0, 0],
  #               [0, 0, 1, 0, 0],
  #               [0, 0, 0, 0, 0],
  #               [0, 0, 0, 0, 0]], dtype=np.uint8)
  ```
