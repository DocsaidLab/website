---
sidebar_position: 7
---
# imblackhat

>[imblackhat(img: np.ndarray, ksize: Union[int, Tuple[int, int]] = (3, 3), kstruct: Union[str, int, "MORPH"] = "MORPH.RECT") -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/morphology.py#L191)

- **Description**: Black Hat operation: Subtract the result of a closing operation from the original image. For multi-channel images, each channel will be processed independently. This operation is useful for extracting regions darker than the original image, such as dark spots or fine structures, while removing or weakening large dark areas.

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

    blackhat_img = D.imblackhat(img, ksize=3, kstruct='CROSS')

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
