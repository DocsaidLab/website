---
sidebar_position: 8
---

# pairwise_ioa

> [pairwise_ioa(boxes1: Boxes, boxes2: Boxes) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/structures/functionals.py#L69C46-L69C47)

- **Description**

    `pairwise_ioa` is a function used to calculate the Intersection over Area (IoA) between two lists of bounding boxes. This function computes the IoA for all N x M pairs of bounding boxes. The input bounding box type must be `Boxes`.

- **Parameters**

    - **boxes1** (`Boxes`): The first list of bounding boxes, containing N bounding boxes.
    - **boxes2** (`Boxes`): The second list of bounding boxes, containing M bounding boxes.

- **Example**

    ```python
    import docsaidkit as D

    boxes1 = D.Boxes([[10, 20, 50, 80], [20, 30, 60, 90]])
    boxes2 = D.Boxes([[20, 30, 60, 90], [30, 40, 70, 100]])
    ioa = D.pairwise_ioa(boxes1, boxes2)
    print(ioa)
    # >>> [[0.625 0.33333334]
    #      [1.0 0.625]]
    ```

## Additional information

### Introduction to IoA

IoA (Intersection over Area) is a metric used to evaluate the overlap of bounding boxes. It measures the ratio of the intersection area between the predicted and the ground truth bounding boxes to the area of the ground truth bounding box.

### Definition

The IoA is calculated as the area of intersection between the predicted and the ground truth bounding boxes divided by the area of the ground truth bounding box. The value of IoA ranges from 0 to 1, with higher values indicating greater coverage of the ground truth by the predicted bounding box.

### Calculation Steps

1. **Determine Bounding Box Coordinates**: First, the positions of the predicted and ground truth bounding boxes in the image must be established. Typically, four coordinates represent a bounding box: (x0, y0, x1, y1), where (x0, y0) are the coordinates of the top-left corner, and (x1, y1) are the coordinates of the bottom-right corner.

2. **Calculate Intersection Area**: Compute the area of intersection between the predicted and ground truth bounding boxes.

3. **Calculate IoA**: Divide the intersection area by the ground truth bounding box area to get the IoA value.

### Application Scenarios

- **Object Detection**: In object detection tasks, IoA is used to assess the overlap between the predicted and ground truth bounding boxes, thus evaluating the accuracy of the detection model.

- **Model Evaluation**: IoA is commonly used to evaluate and compare the performance of different object detection models, with higher IoA values indicating better detection accuracy.

- **Non-Maximum Suppression (NMS)**: In the post-processing of object detection, IoA is used for non-maximum suppression to eliminate overlapping detection boxes and retain the best detection results.

### Advantages and Limitations

- **Advantages**: IoA quantifies the degree of overlap between the predicted and ground truth bounding boxes, helping to assess the accuracy of models.

- **Limitations**: IoA only considers the degree of overlap between the predicted and ground truth bounding boxes and does not account for other factors such as the shape and orientation of the bounding boxes, which may lead to inaccuracies under certain circumstances.