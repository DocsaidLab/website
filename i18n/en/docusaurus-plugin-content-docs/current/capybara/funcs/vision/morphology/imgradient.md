# imgradient

> [imgradient(img: np.ndarray, ksize: int | tuple[int, int] = (3, 3), kstruct: str | int | MORPH = MORPH.RECT) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/morphology.py)

- **Description**: Gradient operation: The result of dilating the image minus the eroded image. For multi-channel images, each channel is processed independently. This operation is useful for extracting the edges of objects.

- **Parameters**

  - **img** (`np.ndarray`): The input image.
  - **ksize** (`int | tuple[int, int]`): The size of the structuring element. Default is (3, 3).
  - **kstruct** (`str | int | MORPH`): Structuring element shape. Accepts `MORPH.CROSS/RECT/ELLIPSE`, string `"CROSS"/"RECT"/"ELLIPSE"`, or an OpenCV integer. Default is `MORPH.RECT`.

- **Example**

  ```python
  import numpy as np
  from capybara.vision.morphology import imgradient

  img = np.array([[0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0]], dtype=np.uint8)

  gradient_img = imgradient(img, ksize=3, kstruct='RECT')

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
