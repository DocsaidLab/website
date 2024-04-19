---
sidebar_position: 4
---

# imcvtcolor

>[imcvtcolor(img: np.ndarray, cvt_mode: Union[int, str]) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/functionals.py#L96)

- **Description**: Perform color space conversion on the input image.

- **Parameters**:

    - **img** (`np.ndarray`): Input image to be converted.
    - **cvt_mode** (`Union[int, str]`): Color conversion mode. It can be an integer constant representing the conversion code, or a string representing the OpenCV color conversion name. For example, `BGR2GRAY` is used to convert a BGR image to grayscale. For available parameters, please refer directly to [**OpenCV COLOR**](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html).

- **Returns**:

    - **np.ndarray**: Image with the desired color space.

- **Example**:

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    ycrcb_img = D.imcvtcolor(img, 'BGR2YCrCb')
    ```

    ![imcvtcolor_ycrcb](./resource/test_imcvtcolor_ycrcb.jpg)

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    ycrcb_img = D.imcvtcolor(img, 'BGR2YCrCb')
    ```

    ![imcvtcolor_gray](./resource/test_imcvtcolor_gray.jpg)
