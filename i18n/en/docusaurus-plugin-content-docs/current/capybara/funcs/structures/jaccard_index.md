---
sidebar_position: 10
---

# jaccard_index

> [jaccard_index(pred_poly: np.ndarray, gt_poly: np.ndarray, image_size: Tuple[int, int]) -> float](https://github.com/DocsaidLab/DocsaidKit/blob/6db820b92e709b61f1848d7583a3fa856b02716f/docsaidkit/structures/functionals.py#L93C5-L93C18)

- **Description**

    `jaccard_index` is a function used to calculate the Jaccard index between two polygons. This function calculates the ratio of the intersection area to the union area between two polygons. The input polygons must be of type `np.ndarray`.

- **Parameters**

    - **pred_poly** (`np.ndarray`): The predicted polygon, a polygon with 4 points.
    - **gt_poly** (`np.ndarray`): The ground truth polygon, a polygon with 4 points.
    - **image_size** (`Tuple[int, int]`): The size of the image, (height, width).

- **Example**

    ```python
    import docsaidkit as D

    pred_poly = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    gt_poly = np.array([[0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5]])
    image_size = (2, 2)
    jaccard_index = D.jaccard_index(pred_poly, gt_poly, image_size)
    print(jaccard_index)
    # >>> 0.14285714285714285
    ```
