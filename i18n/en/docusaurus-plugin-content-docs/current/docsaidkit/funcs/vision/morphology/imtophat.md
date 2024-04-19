---
sidebar_position: 6
---

# imtophat

>[imtophat(img: np.ndarray, ksize: Union[int, Tuple[int, int]] = (3, 3), kstruct: Union[str, int, "MORPH"] = "MORPH.RECT") -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/morphology.py#L163)

- **Description**: Top Hat operation: The original image subtracted by the result of the opening operation. For multi-channel images, each channel will be processed independently. This operation is useful for extracting regions brighter than the original image, such as bright spots or fine structures, while removing or weakening large bright areas.

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
                    [0, 0, 1, 1, 0],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1]], dtype=np.uint8)

    tophat_img = D.imtophat(img, ksize=3, kstruct='RECT')

    # Kernel will be like this:
    # >>> np.array([[1, 1, 1],
    #               [1, 1, 1],
    #               [1, 1, 1]], dtype=np.uint8)

    # After tophat, the image will be like this:
    # >>> np.array([[0, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 0],
    #               [0, 0, 1, 1, 0],
    #               [0, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 0]], dtype=np.uint8)
    ```
