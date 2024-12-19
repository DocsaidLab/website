---
sidebar_position: 5
---

# imwarp_quadrangles

>[imwarp_quadrangles(img: np.ndarray, polygons: Union[Polygons, np.ndarray]) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/geometric.py#L206)

- **Description**: Apply perspective transformation to the input image using the 4 points defined by the given "multiple" polygons. The function automatically sorts the four points in the order: the first point is the top-left corner, the second point is the top-right corner, the third point is the bottom-right corner, and the fourth point is the bottom-left corner. The target size of the image transformation is determined by the width and height of the minimum bounding rectangle of the polygons.

- **Parameters**:

    - **img** (`np.ndarray`): Input image to be transformed.
    - **polygons** (`Union[Polygons, np.ndarray]`): Polygon objects containing the four points defining the transformation for "multiple" polygons.

- **Returns**:

    - **List[np.ndarray]**: List of transformed images.

- **Example**:

    ```python
    import docsaidkit as D

    img = D.imread('./resource/test_warp.jpg')
    polygons = D.Polygons([[[602, 404], [1832, 530], [1588, 985], [356, 860]]])
    warp_imgs = D.imwarp_quadrangles(img, polygons)
    ```

    Please refer to the picture [**imwarp_quadrangle**](./imwarp_quadrangle.md)。
