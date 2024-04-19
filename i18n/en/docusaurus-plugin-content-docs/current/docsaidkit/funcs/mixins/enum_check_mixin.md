---
sidebar_position: 1
---

# EnumCheckMixin

> [EnumCheckMixin](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/mixins.py#L57)

- **Description**

    Provides an `obj_to_enum` method for Enum objects, allowing for the retrieval of different types of enum values.

    The design philosophy behind this method is to prevent users from using incorrect enum values in the program by enforcing enum type constraints. However, it also aims to prevent users from feeling frustrated if they cannot find the corresponding enum value.

    Therefore, this function also provides an `obj_to_enum` method, which can be used to convert different types of enum values to the enum type.

    Users can query enum values using strings, enum values, or integers.

- **Example**

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
