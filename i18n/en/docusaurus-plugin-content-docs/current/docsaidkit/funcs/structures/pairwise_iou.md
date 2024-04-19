---
sidebar_position: 7
---

# pairwise_iou

>[paiwise_iou(boxes1: Boxes, boxes2: Boxes) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/structures/functionals.py#L41)

- **Description**

    `pairwise_iou` is a function used to compute the Intersection over Union (IoU) between two lists of bounding boxes. This function computes IoU for all N x M pairs of bounding boxes. The input bounding boxes must be of type `Boxes`.

- **Parameters**

    - **boxes1** (`Boxes`): The first list of bounding boxes, containing N bounding boxes.
    - **boxes2** (`Boxes`): The second list of bounding boxes, containing M bounding boxes.

- **Example**

    ```python
    import docsaidkit as D

    boxes1 = D.Boxes([[10, 20, 50, 80], [20, 30, 60, 90]])
    boxes2 = D.Boxes([[20, 30, 60, 90], [30, 40, 70, 100]])
    iou = D.pairwise_iou(boxes1, boxes2)
    print(iou)
    # >>> [[0.45454547 0.2]
    #      [1.0 0.45454547]]
    ```

## Additional information

### Introduction to IoU

IoU (Intersection over Union) is a crucial metric in computer vision for evaluating the performance of object detection algorithms, particularly in tasks such as object detection and segmentation. It measures the overlap between predicted bounding boxes and ground truth bounding boxes.

### Definition

The IoU calculation formula computes the ratio of the area of intersection to the area of union between the predicted bounding box and the ground truth bounding box. IoU values range from 0 to 1, where higher values indicate greater overlap and hence more accurate predictions.

### Computation Steps

1. **Determine Bounding Box Coordinates**: Firstly, it's necessary to establish the positions of the predicted and ground truth bounding boxes in the image. Typically, bounding boxes are represented using four coordinates: (x0, y0, x1, y1), where (x0, y0) denotes the coordinates of the top-left corner, and (x1, y1) denotes the coordinates of the bottom-right corner.

2. **Compute Intersection Area**: Calculate the area of overlap between the two bounding boxes. If the two bounding boxes do not overlap at all, the intersection area is 0.

3. **Compute Union Area**: The union area is equal to the sum of the areas of the two bounding boxes minus the intersection area.

4. **Compute IoU**: Divide the intersection area by the union area to obtain the IoU value.

### Applications

- **Object Detection**: In object detection tasks, IoU is used to assess whether the detection box accurately covers the object. A threshold (e.g., 0.5) is often set, and detections with IoU above this threshold are considered successful.

- **Model Evaluation**: IoU is commonly used to evaluate and compare the performance of different object detection models, with higher IoU values indicating better detection accuracy.

- **Non-Maximum Suppression (NMS)**: In post-processing of object detection, IoU is utilized for non-maximum suppression to eliminate overlapping detection boxes and retain the best detection results.

### Advantages and Limitations

- **Advantages**

    - **Intuitive**: IoU provides an intuitive way to quantify the similarity between predicted bounding boxes and ground truth bounding boxes.
    - **Standardized**: Being a scalar value ranging from 0 to 1, IoU facilitates comparison and evaluation.

- **Limitations**

    - **Insensitivity**: IoU may not be sensitive enough to small deviations between predicted and ground truth bounding boxes, particularly when the overlap is high.
    - **Threshold Selection**: The choice of IoU threshold may influence the final evaluation results, with different thresholds leading to different evaluation criteria.