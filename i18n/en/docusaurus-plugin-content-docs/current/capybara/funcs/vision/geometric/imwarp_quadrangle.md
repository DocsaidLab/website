# imwarp_quadrangle

> [imwarp_quadrangle(img: np.ndarray, polygon: Union[Polygon, np.ndarray]) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/geometric.py#L155)

- **Description**: Applies a perspective transformation to the input image based on a quadrilateral defined by four points. The function automatically sorts the points in the following order: the first point is the top-left, the second is the top-right, the third is the bottom-right, and the fourth is the bottom-left. The target size of the transformed image is determined by the width and height of the minimum rotated bounding box of the polygon.

- **Parameters**

  - **img** (`np.ndarray`): The input image to be transformed.
  - **polygon** (`Union[Polygon, np.ndarray]`): A polygon object containing the four points that define the transformation.

- **Returns**

  - **np.ndarray**: The transformed image.

- **Example**

  ```python
  import capybara as cb

  img = cb.imread('./resource/test_warp.jpg')
  polygon = cb.Polygon([[602, 404], [1832, 530], [1588, 985], [356, 860]])
  warp_img = cb.imwarp_quadrangle(img, polygon)
  ```

  ![imwarp_quadrangle](./resource/test_imwarp_quadrangle.jpg)

  The green box in the image above represents the original polygon area.
