---
sidebar_position: 8
---

# pairwise_ioa

> [pairwise_ioa(boxes1: Boxes, boxes2: Boxes) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/structures/functionals.py)

- **說明**：

  `pairwise_ioa` 是一個用來計算兩個邊界框列表之間的 IoA (交集比例) 的函數。這個函數會計算所有 N x M 對的邊界框之間的 IoA。輸入的邊界框類型必須為 `Boxes`。

- **參數**

  - **boxes1** (`Boxes`)：第一個邊界框列表。包含 N 個邊界框。
  - **boxes2** (`Boxes`)：第二個邊界框列表。包含 M 個邊界框。

- **傳回值**

  - **np.ndarray**：IoA 矩陣，shape 為 `[N, M]`。

- **備註**

  - IoA 定義為 `intersection(boxes1, boxes2) / area(boxes2)`。

- **例外**

  - **TypeError**：`boxes1` 或 `boxes2` 不是 `Boxes`。
  - **ValueError**：存在空框（寬或高 <= 0）時。

- **範例**

  ```python
  import capybara as cb

  boxes1 = cb.Boxes([[10, 20, 50, 80], [20, 30, 60, 90]])
  boxes2 = cb.Boxes([[20, 30, 60, 90], [30, 40, 70, 100]])
  ioa = cb.pairwise_ioa(boxes1, boxes2)
  print(ioa)
  # >>> [[0.625 0.33333334]
  #      [1.0 0.625]]
  ```
