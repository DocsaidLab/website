---
sidebar_position: 1
---

# BoxMode

> [BoxMode](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/structures/boxes.py#L26)

- **説明**：

  `BoxMode` は、異なる境界ボックス表現方法を表す列挙クラスです。

  一般的に使用される境界ボックスの表現方法は以下の 3 つです：

  - **XYXY**：`(x0, y0, x1, y1)` として表現され、絶対的な浮動小数点座標を使用します。座標範囲は `[0, w]` と `[0, h]` です。
  - **XYWH**：`(x0, y0, w, h)` として表現され、絶対的な浮動小数点座標を使用します。`(x0, y0)` は境界ボックスの左上角、`(w, h)` は境界ボックスの幅と高さです。
  - **CXCYWH**：`(xc, yc, w, h)` として表現され、絶対的な浮動小数点座標を使用します。`(xc, yc)` は境界ボックスの中心、`(w, h)` は境界ボックスの幅と高さです。

  良い設計とは、これらの種類の間で自由に変換できることが求められるため、`BoxMode` では `convert` メソッドを実装しています。以下のサンプルコードで、関連する使用法を確認できます。また、このクラスには `align_code` メソッドも実装されており、大文字小文字を区別しない文字列を受け入れ、大文字で表現を返すことができます。

- **例**

  ```python
  import capybara as cb
  import numpy as np

  box = np.array([10, 20, 50, 80]).astype(np.float32)
  box = cb.BoxMode.convert(box, from_mode=cb.BoxMode.XYXY, to_mode=cb.BoxMode.XYWH)
  # >>> array([10, 20, 40, 60])

  # モードを表す文字列を使用
  box = cb.BoxMode.convert(box, from_mode='XYWH', to_mode='CXCYWH')
  # >>> array([30, 50, 40, 60])
  ```
