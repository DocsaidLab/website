---
sidebar_position: 1
---

# BoxMode

> [BoxMode](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/structures/boxes.py#L26)

- **說明**：

  `BoxMode` 是一個列舉類別，用來表示不同的邊界框表示方式。

  一般來說，常見的邊界框的表現方式有三種：

  - **XYXY**：表示為 `(x0, y0, x1, y1)`，使用絕對浮點座標。 座標範圍 `[0, w]` 及 `[0, h]`。
  - **XYWH**：表示為 `(x0, y0, w, h)`，使用絕對浮點座標。 `(x0, y0)` 是邊界框的左上角點，`(w, h)` 是邊界框的寬度和高度。
  - **CXCYWH**：表示為 `(xc, yc, w, h)`，使用絕對浮點座標。 `(xc, yc)` 是邊界框的中心點，`(w, h)` 是邊界框的寬度和高度。

  我們認為一個好的設計，必須可以愉快地在這幾種類型間自由轉換，所以我們在 `BoxMode` 底下實作了一個 `convert` 方法。相關用法可以參考以下範例說明。此外，在這個類別中我們也實作了一個 `align_code` 的方法，可以接受大小寫不定的字串，並將其轉換為大寫的表示方式。

- **範例**

  ```python
  import capybara as cb
  import numpy as np

  box = np.array([10, 20, 50, 80]).astype(np.float32)
  box = cb.BoxMode.convert(box, from_mode=cb.BoxMode.XYXY, to_mode=cb.BoxMode.XYWH)
  # >>> array([10, 20, 40, 60])

  # Using string to represent the mode
  box = cb.BoxMode.convert(box, from_mode='XYWH', to_mode='CXCYWH')
  # >>> array([30, 50, 40, 60])
  ```
