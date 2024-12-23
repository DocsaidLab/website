# EnumCheckMixin

> [EnumCheckMixin](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/mixins.py#L57)

- **Description**: Provides the `obj_to_enum` method for Enum objects, allowing for the conversion of various types into the corresponding enum values.

The design concept of this method is to prevent users from using incorrect enum values by enforcing type constraints while also avoiding frustration when a matching enum value cannot be found.

Thus, the `obj_to_enum` method is available to convert different types of enum values into the correct enum type.

Users can query enum values using strings, enum values, or integers.

- **Example**

  ```python
  from enum import IntEnum
  from capybara import EnumCheckMixin

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
