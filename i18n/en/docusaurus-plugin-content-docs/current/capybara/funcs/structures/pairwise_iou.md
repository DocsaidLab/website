---
sidebar_position: 7
---

# pairwise_iou

> [pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/structures/functionals.py#L42)

- **Description**:

  `pairwise_iou` is a function used to calculate the Intersection over Union (IoU) between two lists of bounding boxes. This function computes the IoU for all N x M pairs of bounding boxes. The input bounding boxes must be of type `Boxes`.

- **Parameters**

  - **boxes1** (`Boxes`): The first list of bounding boxes, containing N bounding boxes.
  - **boxes2** (`Boxes`): The second list of bounding boxes, containing M bounding boxes.

- **Example**

  ```python
  import capybara as cb

  boxes1 = cb.Boxes([[10, 20, 50, 80], [20, 30, 60, 90]])
  boxes2 = cb.Boxes([[20, 30, 60, 90], [30, 40, 70, 100]])
  iou = cb.pairwise_iou(boxes1, boxes2)
  print(iou)
  # >>> [[0.45454547 0.2]
  #      [1.0 0.45454547]]
  ```

## Additional Information

### Introduction to IoU

IoU (Intersection over Union) is an important metric in computer vision for evaluating the performance of object detection algorithms, especially in tasks like object detection and segmentation. It measures the overlap between the predicted bounding box and the ground truth bounding box.

### Definition

The IoU formula is the intersection area of the predicted and ground truth bounding boxes divided by their union area. The IoU value ranges from 0 to 1, with higher values indicating better overlap and more accurate predictions.

### Calculation Steps

1. **Determine Bounding Box Coordinates**: First, determine the positions of the predicted and ground truth bounding boxes in the image. These are typically represented by four coordinates: (x0, y0, x1, y1), where (x0, y0) is the top-left corner and (x1, y1) is the bottom-right corner.

2. **Calculate Intersection Area**: Compute the area of the overlap between the predicted and ground truth bounding boxes. If the two bounding boxes do not overlap, the intersection area will be 0.

3. **Calculate Union Area**: The union area is the total area covered by both bounding boxes, which is the sum of their individual areas minus the intersection area.

4. **Calculate IoU**: Divide the intersection area by the union area to obtain the IoU value.

### Applications

- **Object Detection**: In object detection tasks, IoU is used to evaluate how accurately the predicted bounding box overlaps with the ground truth. A threshold (e.g., 0.5) is often set, and if the IoU is greater than this threshold, the detection is considered successful.

- **Model Evaluation**: IoU is commonly used to evaluate and compare the performance of different object detection models. A higher IoU value indicates better detection accuracy.

- **Non-Maximum Suppression (NMS)**: In post-processing of object detection, IoU is used in non-maximum suppression to remove overlapping detection boxes and retain the best results.

### Advantages and Limitations

- **Advantages**

  - **Intuitive**: IoU provides an intuitive way to quantify the similarity between the predicted and ground truth bounding boxes.
  - **Standardized**: As a scalar value in the range [0, 1], IoU is easy to compare and evaluate.

- **Limitations**

  - **Insensitivity**: When the predicted and ground truth bounding boxes have small deviations (i.e., they almost overlap), IoU may not be sensitive enough to detect minor differences.
  - **Threshold Selection**: The choice of IoU threshold can influence the evaluation results. Different thresholds may lead to different evaluation standards.
