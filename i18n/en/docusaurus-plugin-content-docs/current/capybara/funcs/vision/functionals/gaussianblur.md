---
sidebar_position: 2
---

# gaussianblur

>[gaussianblur(img: np.ndarray, ksize: _Ksize = 3, sigmaX: int = 0, **kwargs) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/functionals.py#L54)

- **Description**: Apply Gaussian blur to the input image.

- **Parameters**:

    - **img** (`np.ndarray`): Input image to be blurred.
    - **ksize** (`Union[int, Tuple[int, int]]`): Size of the kernel used for blurring. If an integer value is provided, a square kernel of the specified size is used. If a tuple `(k_height, k_width)` is provided, a rectangular kernel of the specified size is used. Default is 3.
    - **sigmaX** (`int`): Standard deviation of the Gaussian kernel in the X direction. Default is 0.

- **Returns**:

    - **np.ndarray**: Blurred image.

- **Example**:

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    blur_img = D.gaussianblur(img, ksize=5)
    ```

    ![gaussianblur](./resource/test_gaussianblur.jpg)
