---
sidebar_position: 6
---

# pairwise_intersection

> [pairwise_intersection(boxes1: Boxes, boxes2: Boxes) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/structures/functionals.py#L18)

- **説明**：

  `pairwise_intersection` は、2 つの境界ボックスリスト間で交差面積を計算する関数です。この関数は、すべての N x M ペアの境界ボックス間で交差面積を計算します。入力される境界ボックスは `Boxes` 型である必要があります。

- **パラメータ**

  - **boxes1** (`Boxes`)：最初の境界ボックスリスト。N 個の境界ボックスを含みます。
  - **boxes2** (`Boxes`)：2 番目の境界ボックスリスト。M 個の境界ボックスを含みます。

- **例**

  ```python
  import capybara as cb

  boxes1 = cb.Boxes([[10, 20, 50, 80], [20, 30, 60, 90]])
  boxes2 = cb.Boxes([[20, 30, 60, 90], [30, 40, 70, 100]])
  intersection = cb.pairwise_intersection(boxes1, boxes2)
  print(intersection)
  # >>> [[1500. 800.]
  #      [2400. 1500.]]
  ```
