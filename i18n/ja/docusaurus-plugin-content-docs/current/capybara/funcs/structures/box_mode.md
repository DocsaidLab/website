---
sidebar_position: 1
---

# BoxMode

> [BoxMode](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/structures/boxes.py#L31)

- **説明**：

  `BoxMode` は、異なる境界ボックス表現方法を表す列挙クラスです。

  通常、境界ボックスの表現方法には 3 つの形式があります：

  - **XYXY**：`(x0, y0, x1, y1)` の形式で、絶対的な浮動小数点座標を使用します。座標範囲は `[0, w]` と `[0, h]` です。
  - **XYWH**：`(x0, y0, w, h)` の形式で、絶対的な浮動小数点座標を使用します。`(x0, y0)` はボックスの左上の点で、`(w, h)` はボックスの幅と高さです。
  - **CXCYWH**：`(xc, yc, w, h)` の形式で、絶対的な浮動小数点座標を使用します。`(xc, yc)` はボックスの中心点で、`(w, h)` はボックスの幅と高さです。

  このクラスでは、これらの形式間で自由に変換できるように `BoxMode` 下に `convert` メソッドを実装しました。また、`align_code` メソッドも実装されており、大文字・小文字を区別せずに文字列を受け付け、対応する大文字形式に変換します。

- **例**

  ```python
  import docsaidkit as D
  import numpy as np

  box = np.array([10, 20, 50, 80]).astype(np.float32)
  box = D.BoxMode.convert(box, from_mode=D.BoxMode.XYXY, to_mode=D.BoxMode.XYWH)
  # >>> array([10, 20, 40, 60])

  # モードを文字列で指定
  box = D.BoxMode.convert(box, from_mode='XYWH', to_mode='CXCYWH')
  # >>> array([30, 50, 40, 60])
  ```
