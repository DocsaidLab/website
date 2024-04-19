---
sidebar_position: 6
---

>[imbinarize(img: np.ndarray, threth: int = cv2.THRESH_BINARY, color_base: str = 'BGR') -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/functionals.py#L336)

- **Description**: Perform binarization on the input image.

- **Parameters**:

    - **img** (`np.ndarray`): Input image to be binarized. If the input image is 3-channel, the function will automatically apply the `bgr2gray` function.
    - **threshold** (`int`): Threshold type. There are two types of thresholds:
        1. `cv2.THRESH_BINARY`: `cv2.THRESH_OTSU + cv2.THRESH_BINARY`
        2. `cv2.THRESH_BINARY_INV`: `cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV`
    - **color_base** (`str`): Color space of the input image. Default is `'BGR'`.

- **Returns**:

    - **np.ndarray**: Binarized image.

- **Example**:

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    bin_img = D.imbinarize(img)
    ```

    ![imbinarize](./resource/test_imbinarize.jpg)
