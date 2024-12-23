---
sidebar_position: 1
---

# BoxMode

> [BoxMode](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/structures/boxes.py#L26)

- **Description**:

  `BoxMode` is an enumeration class used to represent different bounding box formats.

  Generally, there are three common bounding box representations:

  - **XYXY**: Represented as `(x0, y0, x1, y1)`, using absolute floating-point coordinates. The coordinate range is `[0, w]` and `[0, h]`.
  - **XYWH**: Represented as `(x0, y0, w, h)`, using absolute floating-point coordinates. `(x0, y0)` is the top-left corner of the bounding box, and `(w, h)` is the width and height of the bounding box.
  - **CXCYWH**: Represented as `(xc, yc, w, h)`, using absolute floating-point coordinates. `(xc, yc)` is the center of the bounding box, and `(w, h)` is the width and height of the bounding box.

  We believe a good design should allow smooth conversion between these formats. Therefore, the `BoxMode` class implements a `convert` method for this purpose. You can refer to the following example for usage. Additionally, this class also implements an `align_code` method, which accepts case-insensitive strings and converts them to uppercase representation.

- **Example**

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
