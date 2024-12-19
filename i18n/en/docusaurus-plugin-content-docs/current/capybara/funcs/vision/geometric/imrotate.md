---
sidebar_position: 3
---

# imrotate

>[imrotate(img: np.ndarray, angle: float, scale: float = 1, interpolation: Union[str, int, INTER] = INTER.BILINEAR, bordertype: Union[str, int, BORDER] = BORDER.CONSTANT, bordervalue: Union[int, Tuple[int, int, int]] = None, expand: bool = True, center: Tuple[int, int] = None) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/geometric.py#L80C1-L153C1)

- **Description**: Rotate the input image.

- **Parameters**:

    - **img** (`np.ndarray`): Input image to be rotated.
    - **angle** (`float`): Rotation angle in degrees, counterclockwise.
    - **scale** (`float`): Scale factor. Default is 1.
    - **interpolation** (`Union[str, int, INTER]`): Interpolation method. Available options are: INTER.NEAREST, INTER.LINEAR, INTER.CUBIC, INTER.LANCZOS4. Default is INTER.LINEAR.
    - **border_type** (`Union[str, int, BORDER]`): Border type. Available options are: BORDER.CONSTANT, BORDER.REPLICATE, BORDER.REFLECT, BORDER.REFLECT_101. Default is BORDER.CONSTANT.
    - **border_value** (`Union[int, Tuple[int, int, int]]`): Value used for padding borders. Only valid when border_type is BORDER.CONSTANT. Default is None.
    - **expand** (`bool`): Whether to expand the output image to accommodate the entire rotated image. If True, the output image is expanded to be large enough to accommodate the entire rotated image. If False or omitted, the output image is the same size as the input image. Note that the expand flag assumes rotation around the center and no translation. Default is False.
    - **center** (`Tuple[int, int]`): Rotation center. Default is the center of the image.

- **Returns**:

    - **np.ndarray**: Rotated image.

- **Example**:

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    rotate_img = D.imrotate(img, 45, bordertype=D.BORDER.CONSTANT, expand=True)

    # Resize the rotated image to the original size for visualization
    rotate_img = D.imresize(rotate_img, [img.shape[0], img.shape[1]])
    ```

    ![imrotate](./resource/test_imrotate.jpg)
