# EnumCheckMixin

> [EnumCheckMixin](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/mixins.py#L57)

- **説明**：Enum オブジェクトの `obj_to_enum` メソッドを提供し、異なる型の列挙値を取得する際に使用できます。

このメソッドの設計理念は、列挙型による型制限を活用して、ユーザーが間違った列挙値を使用するのを防ぎつつ、対応する列挙値が見つからないことでユーザーが不快に思うことを避けることです。

そのため、`obj_to_enum` メソッドを提供し、異なる型の列挙値を列挙型に変換できます。

ユーザーは文字列、列挙値、または整数を使って列挙値を検索できます。

- **例**

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
