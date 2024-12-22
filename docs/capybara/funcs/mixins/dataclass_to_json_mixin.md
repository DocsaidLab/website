# DataClassToJsonMixin

> [DataClassToJsonMixin](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/mixins.py#L90)

- **說明**：提供 DataClass 物件的 JSON 序列化方法，可以用來將 DataClass 物件轉換成 JSON 格式。

- **dict_to_jsonable** 支援格式說明：

  - **Box**、**Boxes**：轉換成 `List[float]` 格式。
  - **Polygon**、**Polygons**：轉換成 `List[List[float]]` 格式。
  - **np.ndarray**、**np.generic**：轉換成 `List` 格式。
  - **list**、**tuple**：遞迴轉換成 `List` 格式。
  - **Enum**：轉換成 `str` 格式。
  - **Mapping**：遞迴轉換成 `Dict` 格式。

- **範例**

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
