---
sidebar_position: 1
---

# EnumCheckMixin

> [EnumCheckMixin](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/mixins.py#L57)

- **說明**：提供 Enum 物件 `obj_to_enum` 方法，可以用來承接不同型態的列舉查詢。這個方法的設計理念是希望可以透過列舉的型態限制，來避免使用者在程式中使用錯誤的列舉值。但同時又不希望使用者會因為找不到對應的列舉值而感到生氣。因此這裡提供了一個 `obj_to_enum` 方法，可以用來將不同型態的列舉值轉換為列舉型態。在這裏，使用者可以使用字串、列舉值或整數來查詢列舉值。

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

