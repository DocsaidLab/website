---
sidebar_position: 1
---

# imerode

>[imerode(img: np.ndarray, ksize: Union[int, Tuple[int, int]] = (3, 3), kstruct: Union[str, int, "MORPH"] = "MORPH.RECT") -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/morphology.py#L14C1-L42C69)

- **Description**: Erosion operation: Erodes the source image using a specified structuring element, which determines the shape of the pixel neighborhood from which the minimum value is taken. For multi-channel images, each channel will be processed independently.

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

    eroded_img = D.imerode(img, ksize=3, kstruct='RECT')

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
