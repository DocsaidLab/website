---
sidebar_position: 9
---

# polygon_iou

> [polygon_iou(poly1: Polygon, poly2: Polygon) -> float](https://github.com/DocsaidLab/Capybara/blob/main/capybara/structures/functionals.py)

- **説明**：

  `polygon_iou` は、2 つの多角形間で IoU（交差領域比）を計算する関数です。この関数は、2 つの多角形の交差面積と和集合面積の比率を計算します。入力される多角形は `Polygon` 型である必要があります。

- **パラメータ**

  - **poly1** (`Polygon`)：予測された多角形。
  - **poly2** (`Polygon`)：実際の多角形。

- **例**

  ```python
  import capybara as cb

  poly1 = cb.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
  poly2 = cb.Polygon([(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)])
  iou = cb.polygon_iou(poly1, poly2)
  print(iou)
  # >>> 0.14285714285714285
  ```
