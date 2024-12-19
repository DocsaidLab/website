---
sidebar_position: 5
---

# imgradient

>[imgradient(img: np.ndarray, ksize: Union[int, Tuple[int, int]] = (3, 3), kstruct: Union[str, int, "MORPH"] = "MORPH.RECT") -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/morphology.py#L135)

- **Description**: Gradient operation: The result of dilating the image subtracted by the result of eroding the image. For multi-channel images, each channel will be processed independently. This operation is useful for extracting the edges of objects.

- **Parameters**

    - **img** (`np.ndarray`): Input image.
    - **ksize** (`Union[int, Tuple[int, int]]`): Size of the structuring element. Default is (3, 3).
    - **kstruct** (`MORPH`): Shape of the element, which can be one of "MORPH.CROSS", "MORPH.RECT", or "MORPH.ELLIPSE". Default is "MORPH.RECT".

- **Example**

    ```python
    import numpy as np
    import docsaidkit as D

    img = np.array([[0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0]], dtype=np.uint8)

    gradient_img = D.imgradient(img, ksize=3, kstruct='RECT')

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
