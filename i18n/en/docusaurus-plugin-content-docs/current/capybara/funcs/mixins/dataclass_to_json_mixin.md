# DataClassToJsonMixin

> [DataClassToJsonMixin](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/mixins.py#L90)

- **Description**: Provides JSON serialization methods for DataClass objects, enabling conversion of DataClass instances to JSON format.

- **dict_to_jsonable** format support:

  - **Box**, **Boxes**: Converts to `List[float]` format.
  - **Polygon**, **Polygons**: Converts to `List[List[float]]` format.
  - **np.ndarray**, **np.generic**: Converts to `List` format.
  - **list**, **tuple**: Recursively converts to `List` format.
  - **Enum**: Converts to `str` format.
  - **Mapping**: Recursively converts to `Dict` format.

- **Example**

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
