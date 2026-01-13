# DataclassToJsonMixin

> [DataclassToJsonMixin](https://github.com/DocsaidLab/Capybara/blob/main/capybara/mixins.py)

- **説明**：dataclass オブジェクトを JSON 化可能な dict に変換するための mixin を提供します。

- **dict_to_jsonable** のサポート形式：

  - **Box**、**Boxes**：`List[float]` 形式に変換。
  - **Polygon**、**Polygons**：`List[List[float]]` 形式に変換。
  - **np.ndarray**、**np.generic**：`List` 形式に変換。
  - **list**、**tuple**：再帰的に `List` 形式に変換。
  - **Enum**：`str` 形式に変換。
  - **Mapping**：再帰的に `Dict` 形式に変換。

- **例**

  ```python
  from dataclasses import dataclass
  from capybara import DataclassToJsonMixin

  @dataclass
  class Person(DataclassToJsonMixin):
      name: str
      age: int

  person = Person('Alice', 20)
  print(person.be_jsonable())
  # >>> OrderedDict([('name', 'Alice'), ('age', 20)])
  ```
