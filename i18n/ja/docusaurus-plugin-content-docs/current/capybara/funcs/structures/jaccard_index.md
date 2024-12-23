---
sidebar_position: 10
---

# jaccard_index

> [jaccard_index(pred_poly: np.ndarray, gt_poly: np.ndarray, image_size: Tuple[int, int]) -> float](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/structures/functionals.py#L95)

- **説明**：

  `jaccard_index` は、2 つの多角形間の Jaccard 指数を計算する関数です。この関数は、2 つの多角形の交差面積と、結合面積との比率を計算します。入力される多角形は `np.ndarray` 型である必要があります。

- **パラメータ**

  - **pred_poly** (`np.ndarray`)：予測された多角形、4 つの点からなる多角形。
  - **gt_poly** (`np.ndarray`)：実際の多角形、4 つの点からなる多角形。
  - **image_size** (`Tuple[int, int]`)：画像のサイズ、(高さ, 幅)。

- **例**

  ```python
  import capybara as cb

  pred_poly = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
  gt_poly = np.array([[0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5]])
  image_size = (2, 2)
  jaccard_index = cb.jaccard_index(pred_poly, gt_poly, image_size)
  print(jaccard_index)
  # >>> 0.14285714285714285
  ```
