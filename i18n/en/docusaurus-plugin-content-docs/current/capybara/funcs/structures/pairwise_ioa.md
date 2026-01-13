---
sidebar_position: 8
---

# pairwise_ioa

> [pairwise_ioa(boxes1: Boxes, boxes2: Boxes) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/structures/functionals.py)

- **Description**:

  `pairwise_ioa` is a function used to calculate the Intersection over Area (IoA) between two lists of bounding boxes. This function computes the IoA for all N x M pairs of bounding boxes. The input bounding boxes must be of type `Boxes`.

- **Parameters**

  - **boxes1** (`Boxes`): The first list of bounding boxes, containing N bounding boxes.
  - **boxes2** (`Boxes`): The second list of bounding boxes, containing M bounding boxes.

- **Returns**

  - **np.ndarray**: IoA matrix with shape `[N, M]`.

- **Notes**

  - IoA is defined as `intersection(boxes1, boxes2) / area(boxes2)`.

- **Exceptions**

  - **TypeError**: `boxes1` or `boxes2` is not `Boxes`.
  - **ValueError**: Empty boxes exist (width/height <= 0).

- **Example**

  ```python
  import capybara as cb

  boxes1 = cb.Boxes([[10, 20, 50, 80], [20, 30, 60, 90]])
  boxes2 = cb.Boxes([[20, 30, 60, 90], [30, 40, 70, 100]])
  ioa = cb.pairwise_ioa(boxes1, boxes2)
  print(ioa)
  # >>> [[0.625 0.33333334]
  #      [1.0 0.625]]
  ```
