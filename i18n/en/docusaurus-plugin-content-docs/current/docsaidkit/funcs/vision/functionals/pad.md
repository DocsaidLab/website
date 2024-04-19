---
sidebar_position: 5
---

# pad

>[pad(img: np.ndarray, pad_size: Union[int, Tuple[int, int], Tuple[int, int, int, int]], fill_value: Optional[Union[int, Tuple[int, int, int]]] = 0, pad_mode: Union[str, int, BORDER] = BORDER.CONSTANT) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/functionals.py#L194)

- **Description**: Perform padding on the input image.

- **Parameters**:

    - **img** (`np.ndarray`): Input image to be padded.
    - **pad_size** (`Union[int, Tuple[int, int], Tuple[int, int, int, int]]`): Padding size. It can be an integer to specify the same padding amount for all sides, a tuple `(pad_top, pad_bottom, pad_left, pad_right)` to specify different padding amounts for each side, or a tuple `(pad_height, pad_width)` to specify the same padding amount for height and width.
    - **fill_value** (`Optional[Union[int, Tuple[int, int, int]]]`): Value used for padding. If the input image is a color image (3 channels), fill_value can be an integer or a tuple `(R, G, B)` to specify the filling color. If the input image is a grayscale image (1 channel), fill_value should be an integer. Default is 0.
    - **pad_mode** (`Union[str, int, BORDER]`): Padding mode. Available options are:
        - BORDER.CONSTANT: Pad with constant values (fill_value).
        - BORDER.REPLICATE: Pad by replicating edge pixels.
        - BORDER.REFLECT: Pad by reflecting the image around the edges.
        - BORDER.REFLECT101: Pad by reflecting the image around the edges, with a slight adjustment to avoid artificial seams. Default is BORDER.CONSTANT.

- **Returns**:

    - **np.ndarray**: Padded image.

- **Example**:

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    pad_img = D.pad(img, pad_size=20, fill_value=(255, 0, 0))

    # Resize the padded image to the original size for visualization
    pad_img = D.imresize(pad_img, [img.shape[0], img.shape[1]])
    ```

    ![pad](./resource/test_pad.jpg)
