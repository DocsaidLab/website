---
sidebar_position: 8
---

# pairwise_ioa

> [pairwise_ioa(boxes1: Boxes, boxes2: Boxes) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/structures/functionals.py#L70)

- **Description**:

  `pairwise_ioa` is a function used to calculate the Intersection over Area (IoA) between two lists of bounding boxes. This function computes the IoA for all N x M pairs of bounding boxes. The input bounding boxes must be of type `Boxes`.

- **Parameters**

  - **boxes1** (`Boxes`): The first list of bounding boxes, containing N bounding boxes.
  - **boxes2** (`Boxes`): The second list of bounding boxes, containing M bounding boxes.

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

## Additional Information

### Introduction to IoA

IoA (Intersection over Area) is a metric used to evaluate the overlap between bounding boxes. It measures the ratio of the intersection area to the area of the ground truth bounding box.

### Definition

The IoA formula is the intersection area of the predicted and ground truth bounding boxes divided by the area of the ground truth bounding box. The IoA value ranges from 0 to 1, with higher values indicating better overlap between the predicted and ground truth bounding boxes.

### Calculation Steps

1. **Determine Bounding Box Coordinates**: The first step is to determine the position of both the predicted and ground truth bounding boxes in the image. These are usually represented by four coordinates: (x0, y0, x1, y1), where (x0, y0) is the top-left corner and (x1, y1) is the bottom-right corner.

2. **Calculate Intersection Area**: Compute the intersection area between the predicted and ground truth bounding boxes.

3. **Calculate IoA**: Divide the intersection area by the area of the ground truth bounding box to get the IoA value.

### Applications

- **Object Detection**: In object detection tasks, IoA is used to evaluate how well the predicted bounding boxes overlap with the ground truth, helping assess the accuracy of the detection model.

- **Model Evaluation**: IoA is commonly used to assess and compare the performance of different object detection models. Higher IoA values indicate better detection accuracy.

- **Non-Maximum Suppression (NMS)**: In post-processing of object detection, IoA is used in non-maximum suppression to eliminate overlapping detection boxes and retain the best results.

### Advantages and Limitations

- **Advantages**: IoA quantifies the overlap between predicted and ground truth bounding boxes, providing a clear metric for model accuracy.

- **Limitations**: IoA only considers the overlap between the predicted and ground truth bounding boxes and does not account for other factors, such as the shape or orientation of the bounding boxes. Therefore, it may not be sufficiently accurate in certain scenarios.
