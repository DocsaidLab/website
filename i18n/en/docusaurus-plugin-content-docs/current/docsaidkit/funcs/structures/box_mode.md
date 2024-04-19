---
sidebar_position: 1
---

# BoxMode

>[BoxMode](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/structures/boxes.py#L31)

- **Description**

    `BoxMode` is an enumeration class used to represent different bounding box formats. Typically, bounding boxes are expressed in one of three formats:

    - **XYXY**: Defined as `(x0, y0, x1, y1)` using absolute float coordinates. Coordinates range between `[0, w]` and `[0, h]`.
    - **XYWH**: Defined as `(x0, y0, w, h)` using absolute float coordinates. `(x0, y0)` is the top-left corner of the bounding box, and `(w, h)` are its width and height.
    - **CXCYWH**: Defined as `(xc, yc, w, h)` using absolute float coordinates. `(xc, yc)` is the center of the bounding box, and `(w, h)` are its width and height.

    A good design should allow easy conversion between these types, so we implemented a `convert` method under `BoxMode`. You can refer to the example below for usage. Additionally, this class also features an `align_code` method that accepts a string of the mode, regardless of case, and converts it to an uppercase representation.

- **Example**

    ```python
    import docsaidkit as D
    import numpy as np

    box = np.array([10, 20, 50, 80]).astype(np.float32)
    box = D.BoxMode.convert(box, from_mode=D.BoxMode.XYXY, to_mode=D.BoxMode.XYWH)
    # >>> array([10, 20, 40, 60])

    # Using string to represent the mode
    box = D.BoxMode.convert(box, from_mode='XYWH', to_mode='CXCYWH')
    # >>> array([30, 50, 40, 60])
    ```