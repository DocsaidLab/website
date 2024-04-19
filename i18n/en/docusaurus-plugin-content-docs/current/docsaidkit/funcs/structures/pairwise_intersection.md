---
sidebar_position: 6
---

# pairwise_intersection

>[pairwise_intersection(boxes1: Boxes, boxes2: Boxes) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/structures/functionals.py#L17)

- **Description**

    `pairwise_intersection` is a function used to calculate the intersection area between two lists of bounding boxes. This function calculates the intersection area for all N x M pairs of bounding boxes. The input bounding box type must be `Boxes`.

- **Parameters**

    - **boxes1** (`Boxes`): The first list of bounding boxes, containing N bounding boxes.
    - **boxes2** (`Boxes`): The second list of bounding boxes, containing M bounding boxes.

- **Example**

    ```python
    import docsaidkit as D

    boxes1 = D.Boxes([[10, 20, 50, 80], [20, 30, 60, 90]])
    boxes2 = D.Boxes([[20, 30, 60, 90], [30, 40, 70, 100]])
    intersection = D.pairwise_intersection(boxes1, boxes2)
    print(intersection)
    # >>> [[1500. 800.]
    #      [2400. 1500.]]
    ```
