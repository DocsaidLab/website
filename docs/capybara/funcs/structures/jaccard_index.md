---
sidebar_position: 10
---

# jaccard_index

> [jaccard_index(pred_poly: np.ndarray, gt_poly: np.ndarray, image_size: Tuple[int, int]) -> float](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/structures/functionals.py#L95)

- **說明**：

  `jaccard_index` 是一個用來計算兩個多邊形之間的 Jaccard index 的函數。這個函數會計算兩個多邊形之間的交集面積與聯集面積之比。輸入的多邊形類型必須為 `np.ndarray`。

- **參數**

  - **pred_poly** (`np.ndarray`)：預測的多邊形，一個 4 個點的多邊形。
  - **gt_poly** (`np.ndarray`)：真實的多邊形，一個 4 個點的多邊形。
  - **image_size** (`Tuple[int, int]`)：影像大小，(高度, 寬度)。

- **範例**

  ```python
  import capybara as cb

  pred_poly = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
  gt_poly = np.array([[0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5]])
  image_size = (2, 2)
  jaccard_index = cb.jaccard_index(pred_poly, gt_poly, image_size)
  print(jaccard_index)
  # >>> 0.14285714285714285
  ```
