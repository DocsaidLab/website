---
sidebar_position: 1
---

# EnumCheckMixin

> [EnumCheckMixin](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/mixins.py#L57)

- **說明**：提供 Enum 物件 `obj_to_enum` 方法，可以用來承接不同型態的列舉查詢。

- **範例**

    ```python
    from enum import IntEnum
    from docsaidkit import EnumCheckMixin

    class Color(EnumCheckMixin, IntEnum):
        RED = 1
        GREEN = 2
        BLUE = 3

    color = Color.obj_to_enum('GREEN')
    print(color)  # Color.GREEN

    color = Color.obj_to_enum(Color.RED)
    print(color)  # Color.RED

    color = Color.obj_to_enum(3)
    print(color)  # Color.BLUE
    ```

