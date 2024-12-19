---
sidebar_position: 5
---

# imadjust

>[imadjust(img: np.ndarray, rng_out: Tuple[int, int] = (0, 255), gamma: float = 1.0, color_base: str = 'BGR') -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/functionals.py#L122)

- **Description**: Adjust the intensity of an image.

- **Parameters**:

    - **img** (`np.ndarray`): Input image to adjust the intensity of. Should be 2-D or 3-D.
    - **rng_out** (`Tuple[int, int]`): Target intensity range for the output image. Default is (0, 255).
    - **gamma** (`float`): Value used for gamma correction. If gamma is less than 1, the mapping will bias towards higher (brighter) output values. If gamma is greater than 1, the mapping will bias towards lower (darker) output values. Default is 1.0 (linear mapping).
    - **color_base** (`str`): Color base of the input image. Should be 'BGR' or 'RGB'. Default is 'BGR'.

- **Returns**:

    - **np.ndarray**: Adjusted image.

- **Example**:

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    adj_img = D.imadjust(img, gamma=2)
    ```

    ![imadjust](./resource/test_imadjust.jpg)
