---
sidebar_position: 9
---

# polygon_iou

> [polygon_iou(poly1: Polygon, poly2: Polygon) -> float](https://github.com/DocsaidLab/DocsaidKit/blob/6db820b92e709b61f1848d7583a3fa856b02716f/docsaidkit/structures/functionals.py#L166)

- **說明**：

    `polygon_iou` 是一個用來計算兩個多邊形之間的 IoU (交集比例) 的函數。這個函數會計算兩個多邊形之間的交集面積與聯集面積之比。輸入的多邊形類型必須為 `Polygon`。

- **參數**

    - **poly1** (`Polygon`)：預測的多邊形。
    - **poly2** (`Polygon`)：真實的多邊形。

- **範例**

    ```python
    import docsaidkit as D

    poly1 = D.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    poly2 = D.Polygon([(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)])
    iou = D.polygon_iou(poly1, poly2)
    print(iou)
    # >>> 0.14285714285714285
    ```
