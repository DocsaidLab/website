---
sidebar_position: 3
---

# imopen

>[imopen(img: np.ndarray, ksize: Union[int, Tuple[int, int]] = (3, 3), kstruct: Union[str, int, "MORPH"] = "MORPH.RECT") -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/morphology.py#L76C5-L76C11)


- **Description**: Opening operation: The process of erosion followed by dilation, which can be used to remove small objects, break apart objects, smooth the edges of objects, eliminate small holes, etc. For multi-channel images, each channel will be processed independently.

- **Parameters**

    - **img** (`np.ndarray`): Input image.
    - **ksize** (`Union[int, Tuple[int, int]]`): Size of the structuring element. Default is (3, 3).
    - **kstruct** (`MORPH`): Shape of the element, which can be one of "MORPH.CROSS", "MORPH.RECT", or "MORPH.ELLIPSE". Default is "MORPH.RECT".

- **Example**

    ```python
    import numpy as np
    import docsaidkit as D

    img = np.array([[1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 0], # <- Look at this row
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1]], dtype=np.uint8)

    opened_img = D.imopen(img, ksize=3, kstruct='RECT')

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
