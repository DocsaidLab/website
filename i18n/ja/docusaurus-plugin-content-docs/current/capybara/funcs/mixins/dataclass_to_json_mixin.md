# DataClassToJsonMixin

> [DataClassToJsonMixin](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/mixins.py#L90)

- **説明**：DataClass オブジェクトの JSON シリアライズ方法を提供し、DataClass オブジェクトを JSON 形式に変換する際に使用できます。

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
