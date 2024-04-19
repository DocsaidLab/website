---
sidebar_position: 9
---

# polygon_iou

> [polygon_iou(poly1: Polygon, poly2: Polygon) -> float](https://github.com/DocsaidLab/DocsaidKit/blob/6db820b92e709b61f1848d7583a3fa856b02716f/docsaidkit/structures/functionals.py#L166)

- **Description**

    `polygon_iou` is a function used to compute the Intersection over Union (IoU) between two polygons. This function calculates the ratio of intersection area to union area between the two polygons. The input polygon types must be `Polygon`.

- **Parameters**

    - **poly1** (`Polygon`): The predicted polygon.
    - **poly2** (`Polygon`): The ground truth polygon.

- **Example**

    ```python
    import docsaidkit as D

    poly1 = D.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    poly2 = D.Polygon([(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)])
    iou = D.polygon_iou(poly1, poly2)
    print(iou)
    # >>> 0.14285714285714285
    ```
