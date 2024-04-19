---
sidebar_position: 4
---

# imclose

>[imclose(img: np.ndarray, ksize: Union[int, Tuple[int, int]] = (3, 3), kstruct: Union[str, int, "MORPH"] = "MORPH.RECT") -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/morphology.py#L105)

- **Description**: Closing operation: The process of dilation followed by erosion, which can be used to fill small holes inside objects, smooth the edges of objects, connect two objects, etc. For multi-channel images, each channel will be processed independently.

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
                    [0, 0, 0, 0, 0], # <- Look at this row
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1]], dtype=np.uint8)

    closed_img = D.imclose(img, ksize=3, kstruct='CROSS')

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
