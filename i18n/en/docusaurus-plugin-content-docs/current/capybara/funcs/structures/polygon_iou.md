---
sidebar_position: 9
---

# polygon_iou

> [polygon_iou(poly1: Polygon, poly2: Polygon) -> float](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/structures/functionals.py#L169)

- **Description**:

  `polygon_iou` is a function used to calculate the Intersection over Union (IoU) between two polygons. It computes the ratio of the intersection area to the union area of the two polygons. The input polygons must be of type `Polygon`.

- **Parameters**

  - **poly1** (`Polygon`): The predicted polygon.
  - **poly2** (`Polygon`): The ground truth polygon.

- **Example**

  ```python
  import capybara as cb

  poly1 = cb.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
  poly2 = cb.Polygon([(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)])
  iou = cb.polygon_iou(poly1, poly2)
  print(iou)
  # >>> 0.14285714285714285
  ```
