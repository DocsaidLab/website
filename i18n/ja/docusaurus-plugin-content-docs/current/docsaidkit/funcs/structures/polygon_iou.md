---
sidebar_position: 9
---

# polygon_iou

> [polygon_iou(poly1: Polygon, poly2: Polygon) -> float](https://github.com/DocsaidLab/DocsaidKit/blob/6db820b92e709b61f1848d7583a3fa856b02716f/docsaidkit/structures/functionals.py#L166)

- **説明**：

  `polygon_iou` は、2 つの多角形間の IoU（交差比率）を計算する関数です。この関数は、2 つの多角形の交差面積と和集合面積の比率を計算します。入力される多角形は `Polygon` 型である必要があります。

- **パラメータ**

  - **poly1** (`Polygon`)：予測された多角形。
  - **poly2** (`Polygon`)：実際の多角形。

- **例**

  ```python
  import docsaidkit as D

  poly1 = D.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
  poly2 = D.Polygon([(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)])
  iou = D.polygon_iou(poly1, poly2)
  print(iou)
  # >>> 0.14285714285714285
  ```
