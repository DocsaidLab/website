# imwarp_quadrangles

> [imwarp_quadrangles(img: np.ndarray, polygons: Union[Polygons, np.ndarray]) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/geometric.py#L206)

- **Description**: Applies perspective transformations to the input image based on multiple quadrilaterals, each defined by four points. The function automatically sorts the points in the following order: the first point is the top-left, the second is the top-right, the third is the bottom-right, and the fourth is the bottom-left. The target size of the transformed image is determined by the width and height of the minimum rotated bounding box of each polygon.

- **Parameters**

  - **img** (`np.ndarray`): The input image to be transformed.
  - **polygons** (`Union[Polygons, np.ndarray]`): A collection of polygons, each containing four points that define the transformation.

- **Returns**

  - **List[np.ndarray]**: A list of transformed images.

- **Example**

  ```python
  import capybara as cb

  img = cb.imread('./resource/test_warp.jpg')
  polygons = cb.Polygons([[[602, 404], [1832, 530], [1588, 985], [356, 860]]])
  warp_imgs = cb.imwarp_quadrangles(img, polygons)
  ```

  For visual reference, please refer to [**imwarp_quadrangle**](./imwarp_quadrangle.md).
