---
sidebar_position: 3
---

# DataClassToJsonMixin

> [DataClassToJsonMixin](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/mixins.py#L90)

- **説明**：DataClass オブジェクトの JSON シリアライズ方法を提供します。これを使用すると、DataClass オブジェクトを JSON 形式に変換できます。

- **dict_to_jsonable** のサポートする形式：

  - **Box**、**Boxes**：`List[float]`形式に変換。
  - **Polygon**、**Polygons**：`List[List[float]]`形式に変換。
  - **np.ndarray**、**np.generic**：`List`形式に変換。
  - **list**、**tuple**：再帰的に`List`形式に変換。
  - **Enum**：`str`形式に変換。
  - **Mapping**：再帰的に`Dict`形式に変換。

- **例**

  ```python
  from dataclasses import dataclass
  from docsaidkit import DataclassToJsonMixin

  @dataclass
  class Person(DataclassToJsonMixin):
      name: str
      age: int

  person = Person('Alice', 20)
  print(person.be_jsonable())
  # >>> OrderedDict([('name', 'Alice'), ('age', 20)])
  ```
