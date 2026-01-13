# imblackhat

> [imblackhat(img: np.ndarray, ksize: int | tuple[int, int] = (3, 3), kstruct: str | int | MORPH = MORPH.RECT)](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/morphology.py)

- **Description**: Black-hat operation: the result of closing operation subtracted from the original image. For multi-channel images, each channel is processed independently. This operation is useful for extracting darker areas than the original image, such as dark spots or small structures, while removing or reducing large dark regions.

- **Parameters**

  - **img** (`np.ndarray`): The input image.
  - **ksize** (`int | tuple[int, int]`): The size of the structuring element. Default is (3, 3).
  - **kstruct** (`str | int | MORPH`): Structuring element shape. Accepts `MORPH.CROSS/RECT/ELLIPSE`, string `"CROSS"/"RECT"/"ELLIPSE"`, or an OpenCV integer. Default is `MORPH.RECT`.

- **Example**

  ```python
  import numpy as np
  from capybara.vision.morphology import imblackhat

  img = np.array([[1, 1, 1, 0, 0],
                  [1, 1, 1, 0, 0],
                  [1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0], # <- Look at this row
                  [0, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1]], dtype=np.uint8)

  blackhat_img = imblackhat(img, ksize=3, kstruct='CROSS')

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
