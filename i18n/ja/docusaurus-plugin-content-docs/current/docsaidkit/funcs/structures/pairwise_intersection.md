---
sidebar_position: 6
---

# pairwise_intersection

> [pairwise_intersection(boxes1: Boxes, boxes2: Boxes) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/structures/functionals.py#L17)

- **説明**：

  `pairwise_intersection` は、2 つの境界ボックスリスト間の交差面積を計算する関数です。この関数は、すべての N x M のペアにおける境界ボックス間の交差面積を計算します。入力される境界ボックスは `Boxes` 型である必要があります。

- **パラメータ**

  - **boxes1** (`Boxes`)：最初の境界ボックスリスト。N 個の境界ボックスを含みます。
  - **boxes2** (`Boxes`)：2 番目の境界ボックスリスト。M 個の境界ボックスを含みます。

- **例**

  ```python
  import docsaidkit as D

  boxes1 = D.Boxes([[10, 20, 50, 80], [20, 30, 60, 90]])
  boxes2 = D.Boxes([[20, 30, 60, 90], [30, 40, 70, 100]])
  intersection = D.pairwise_intersection(boxes1, boxes2)
  print(intersection)
  # >>> [[1500. 800.]
  #      [2400. 1500.]]
  ```
