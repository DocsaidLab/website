---
sidebar_position: 10
---

# jaccard_index

> [jaccard_index(pred_poly: np.ndarray, gt_poly: np.ndarray, image_size: Tuple[int, int]) -> float](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/structures/functionals.py#L95)

- **Description**:

  `jaccard_index` is a function used to calculate the Jaccard index between two polygons. It computes the ratio of the intersection area to the union area of the two polygons. The input polygons must be of type `np.ndarray`.

- **Parameters**

  - **pred_poly** (`np.ndarray`): The predicted polygon, represented by a 4-point polygon.
  - **gt_poly** (`np.ndarray`): The ground truth polygon, represented by a 4-point polygon.
  - **image_size** (`Tuple[int, int]`): The size of the image, in the format (height, width).

- **Example**

  ```python
  import capybara as cb

  pred_poly = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
  gt_poly = np.array([[0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5]])
  image_size = (2, 2)
  jaccard_index = cb.jaccard_index(pred_poly, gt_poly, image_size)
  print(jaccard_index)
  # >>> 0.14285714285714285
  ```
