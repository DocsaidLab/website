# imwarp_quadrangles

> [imwarp_quadrangles(img: np.ndarray, polygons: Polygons, dst_size: tuple[int, int] | None = None, do_order_points: bool = True) -> list[np.ndarray]](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/geometric.py)

- **Description**: Applies perspective transformations to the input image based on multiple quadrilaterals, each defined by four points. The function automatically sorts the points in the following order: the first point is the top-left, the second is the top-right, the third is the bottom-right, and the fourth is the bottom-left. The target size of the transformed image is determined by the width and height of the minimum rotated bounding box of each polygon.

- **Parameters**

  - **img** (`np.ndarray`): The input image to be transformed.
  - **polygons** (`Polygons`): A `Polygons` object containing 4-point polygons.
  - **dst_size** (`tuple[int, int] | None`): Output image size `(width, height)`. If `None`, it is inferred from `polygon.min_box_wh`.
  - **do_order_points** (`bool`): Whether to order the 4 points clockwise (TL, TR, BR, BL). Default is `True`.

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
