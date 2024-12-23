# imgradient

> [imgradient(img: np.ndarray, ksize: Union[int, Tuple[int, int]] = (3, 3), kstruct: Union[str, int, "MORPH"] = "MORPH.RECT") -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/morphology.py#L135)

- **Description**: Gradient operation: The result of dilating the image minus the eroded image. For multi-channel images, each channel is processed independently. This operation is useful for extracting the edges of objects.

- **Parameters**

  - **img** (`np.ndarray`): The input image.
  - **ksize** (`Union[int, Tuple[int, int]]`): The size of the structuring element. Default is (3, 3).
  - **kstruct** (`MORPH`): The shape of the structuring element, which can be one of "MORPH.CROSS", "MORPH.RECT", or "MORPH.ELLIPSE". Default is "MORPH.RECT".

- **Example**

  ```python
  import numpy as np
  import capybara as cb

  img = np.array([[0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0]], dtype=np.uint8)

  gradient_img = cb.imgradient(img, ksize=3, kstruct='RECT')

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
