---
sidebar_position: 2
---

# imrotate90

>[imrotate90(img: np.ndarray, rotate_code: ROTATE) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/geometric.py#L66C1-L77C47)


- **Description**: Rotate the input image by 90 degrees.

- **Parameters**:

    - **img** (`np.ndarray`): Input image to be rotated.
    - **rotate_code** (`RotateCode`): Rotation code. Available options are:
        - ROTATE.ROTATE_90: 90 degrees.
        - ROTATE.ROTATE_180: 180 degrees.
        - ROTATE.ROTATE_270: 90 degrees counterclockwise.

- **Returns**:

    - **np.ndarray**: Rotated image.

- **Example**:

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    rotate_img = D.imrotate90(img, D.ROTATE.ROTATE_270)
    ```

    ![imrotate90](./resource/test_imrotate90.jpg)
