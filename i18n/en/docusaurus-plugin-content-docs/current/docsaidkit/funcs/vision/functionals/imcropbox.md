---
sidebar_position: 7
---

# imcropbox

>[imcropbox(img: np.ndarray, box: Union[Box, np.ndarray], use_pad: bool = False) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/functionals.py#L257)

- **Description**: Crop the input image using the provided bounding box.

- **Parameters**:

    - **img** (`np.ndarray`): Input image to be cropped.
    - **box** (`Union[Box, np.ndarray]`): Cropping box. Input can be a Box object customized by DocsaidKit, defined by coordinates (x1, y1, x2, y2), or a NumPy array with the same format.
    - **use_pad** (`bool`): Whether to use padding to handle areas outside the boundaries. If set to True, the outer regions will be padded with zeros. Default is False.

- **Returns**:

    - **np.ndarray**: Cropped image.

- **Example**:

    ```python
    import docsaidkit as D

    # 使用自定義 Box 物件
    img = D.imread('lena.png')
    box = D.Box([50, 50, 200, 200], box_mode='xyxy')
    cropped_img = D.imcropbox(img, box, use_pad=True)

    # Resize the cropped image to the original size for visualization
    cropped_img = D.imresize(cropped_img, [img.shape[0], img.shape[1]])
    ```

    ![imcropbox_box](./resource/test_imcropbox.jpg)
