---
sidebar_position: 6
---

# pairwise_intersection

>[pairwise_intersection(boxes1: Boxes, boxes2: Boxes) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/structures/functionals.py#L17)

- **說明**：

    `pairwise_intersection` 是一個用來計算兩個邊界框列表之間的交集面積的函數。這個函數會計算所有 N x M 對的邊界框之間的交集面積。輸入的邊界框類型必須為 `Boxes`。

- **參數**

    - **boxes1** (`Boxes`)：第一個邊界框列表。包含 N 個邊界框。
    - **boxes2** (`Boxes`)：第二個邊界框列表。包含 M 個邊界框。

- **範例**

    ```python
    import docsaidkit as D

    boxes1 = D.Boxes([[10, 20, 50, 80], [20, 30, 60, 90]])
    boxes2 = D.Boxes([[20, 30, 60, 90], [30, 40, 70, 100]])
    intersection = D.pairwise_intersection(boxes1, boxes2)
    print(intersection)
    # >>> [[1500. 800.]
    #      [2400. 1500.]]
    ```
