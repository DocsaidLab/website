---
sidebar_position: 1
---

# EnumCheckMixin

> [EnumCheckMixin](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/mixins.py#L57)

- **説明**：Enum オブジェクトに`obj_to_enum`メソッドを提供し、異なる型の列挙値を変換することができます。

このメソッドの設計理念は、列挙型の制約を通じて、プログラム内で誤った列挙値の使用を避けることです。しかし、同時に、ユーザーが対応する列挙値を見つけられないことに対して不満を持たないようにしたいという考えから、この関数は異なる型の列挙値を列挙型に変換する`obj_to_enum`メソッドも提供しています。

ユーザーは文字列、列挙値、整数を使用して列挙値を検索できます。

- **例**

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
