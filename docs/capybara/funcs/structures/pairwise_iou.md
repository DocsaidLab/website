---
sidebar_position: 7
---

# pairwise_iou

> [paiwise_iou(boxes1: Boxes, boxes2: Boxes) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/structures/functionals.py#L42)

- **說明**：

  `pairwise_iou` 是一個用來計算兩個邊界框列表之間的 IoU (交集比例) 的函數。這個函數會計算所有 N x M 對的邊界框之間的 IoU。輸入的邊界框類型必須為 `Boxes`。

- **參數**

  - **boxes1** (`Boxes`)：第一個邊界框列表。包含 N 個邊界框。
  - **boxes2** (`Boxes`)：第二個邊界框列表。包含 M 個邊界框。

- **範例**

  ```python
  import capybara as cb

  boxes1 = cb.Boxes([[10, 20, 50, 80], [20, 30, 60, 90]])
  boxes2 = cb.Boxes([[20, 30, 60, 90], [30, 40, 70, 100]])
  iou = cb.pairwise_iou(boxes1, boxes2)
  print(iou)
  # >>> [[0.45454547 0.2]
  #      [1.0 0.45454547]]
  ```

## 補充說明

### IoU 簡介

IoU（Intersection over Union）是電腦視覺中評估物件偵測演算法效能的重要指標，特別是在目標偵測和分割任務中。 它衡量的是預測邊界框和真實邊界框之間的重疊程度。

### 定義

IoU 計算公式為預測邊界框和真實邊界框交集的面積除以它們並集的面積。 IoU 的值範圍從 0 到 1，數值越大表示重疊程度越高，預測結果越準確。

### 計算步驟

1. **確定邊界框座標**：首先，需要確定預測邊界框和真實邊界框在影像中的位置，通常使用四個座標來表示一個邊界框：(x0, y0, x1, y1)，其中 (x0, y0) 是邊界框左上角的座標，(x1, y1) 是右下角的座標。

2. **計算交集面積**：計算兩個邊界框重疊部分的面積。 如果兩個邊界框完全不重疊，則交集面積為 0。

3. **計算並集面積**：並集面積等於兩個邊界框各自的面積總和減去交集面積。

4. **計算 IoU**：交集面積除以並集面積，得到 IoU 值。

### 應用場景

- **目標偵測**：在目標偵測任務中，IoU 用於評估偵測框是否準確覆寫到目標物件。 通常會設定一個閾值（如 0.5），當 IoU 大於這個閾值時，可以認為偵測是成功的。

- **模型評估**：IoU 常用於評估和比較不同物件偵測模型的效能，較高的 IoU 值表示模型具有較好的偵測精度。

- **非極大值抑制（NMS）**：在目標檢測後處理中，IoU 用於非極大值抑制，以消除重疊的檢測框，保留最佳的檢測結果。

### 優點與限制

- **優點**

  - **直覺式**：IoU 提供了一個直覺的方式來量化預測邊界框與真實邊界框之間的相似度。
  - **標準化**：作為一個範圍在 0 到 1 之間的標量值，IoU 便於比較和評估。

- **局限**

  - **不敏感性**：當預測框與真實框之間的偏差較小，即兩者幾乎重疊時，IoU 的變化可能不夠敏感。
  - **閾值選擇**：IoU 的閾值選擇可能會影響到最終的評估結果，不同的閾值可能導致不同的評價標準。
