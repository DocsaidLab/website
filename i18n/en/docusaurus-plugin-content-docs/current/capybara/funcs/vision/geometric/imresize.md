---
sidebar_position: 1
---

# imresize

>[imresize(img: np.ndarray, size: Tuple[int, int], interpolation: Union[str, int, INTER] = INTER.BILINEAR, return_scale: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, float, float]]](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/geometric.py#L15)

- **Description**: Resize the input image.

- **Parameters**:

    - **img** (`np.ndarray`): Input image to be resized.
    - **size** (`Tuple[int, int]`): Size of the resized image. If only one dimension is given, the other dimension is calculated while maintaining the aspect ratio of the original image.
    - **interpolation** (`Union[str, int, INTER]`): Interpolation method. Available options are: INTER.NEAREST, INTER.LINEAR, INTER.CUBIC, INTER.LANCZOS4. Default is INTER.LINEAR.
    - **return_scale** (`bool`): Whether to return the scaling factor. Default is False.

- **Returns**:

    - **np.ndarray**: Resized image.
    - **Tuple[np.ndarray, float, float]**: Resized image and scaling factors for width and height.

- **Example**:

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')

    # Resize the image to H=256, W=256
    resized_img = D.imresize(img, [256, 256])

    # Resize the image to H=256, keep the aspect ratio
    resized_img = D.imresize(img, [256, None])

    # Resize the image to W=256, keep the aspect ratio
    resized_img = D.imresize(img, [None, 256])
    ```
