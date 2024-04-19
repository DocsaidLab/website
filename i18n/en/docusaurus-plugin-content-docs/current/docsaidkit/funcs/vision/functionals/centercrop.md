---
sidebar_position: 8
---

# centercrop

>[centercrop(img: np.ndarray) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/functionals.py#L374)

- **Description**: Performs center cropping on the input image.

- **Parameters**:
    - **img** (`np.ndarray`): The input image to be center cropped.

- **Returns**:
    - **np.ndarray**: The cropped image.

- **Example**:

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    img = D.imresize(img, [128, 256])
    crop_img = D.centercrop(img)
    ```

    The green box indicates the center cropped area.

    ![centercrop](./resource/test_centercrop.jpg)
