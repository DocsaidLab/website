---
sidebar_position: 7
---

# imcropboxes

>[imcropboxes(img: np.ndarray, boxes: Union[Box, np.ndarray], use_pad: bool = False) -> List[np.ndarray]](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/functionals.py#L325)

- **Description**: Crop the input image using multiple provided bounding boxes.

- **Parameters**:

    - **img** (`np.ndarray`): Input image to be cropped.
    - **boxes** (`Union[Boxes, np.ndarray]`): Cropping boxes. Input can be a Boxes object customized by DocsaidKit, defined as List[Box], or a NumPy array with the same format.
    - **use_pad** (`bool`): Whether to use padding to handle areas outside the boundaries. If set to True, the outer regions will be padded with zeros. Default is False.

- **Returns**:

    - **List[np.ndarray]**: List of cropped images.

- **Example**:

    ```python
    import docsaidkit as D

    # 使用自定義 Box 物件
    img = D.imread('lena.png')
    box1 = D.Box([50, 50, 200, 200], box_mode='xyxy')
    box2 = D.Box([50, 50, 100, 100], box_mode='xyxy')
    boxes = D.Boxes([box1, box2])
    cropped_imgs = D.imcropboxes(img, boxes, use_pad=True)

    # Resize the cropped image to the original size for visualization
    cropped_img = D.imresize(cropped_img, [img.shape[0], img.shape[1]])
    ```

    ![imcropbox_boxes](./resource/test_imcropboxes.jpg)
