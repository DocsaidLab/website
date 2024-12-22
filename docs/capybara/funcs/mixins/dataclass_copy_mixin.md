# DataClassCopyMixin

> [DataClassCopyMixin](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/mixins.py#L77)

- **說明**：提供 DataClass 物件的複製方法，可以用來複製 DataClass 物件。

- **範例**

  ```python
  from dataclasses import dataclass
  from capybara import DataclassCopyMixin

  @dataclass
  class Person(DataclassCopyMixin):
      name: str
      age: int

  person = Person('Alice', 20)
  person_copy = person.__copy__()
  person_deepcopy = person.__deepcopy__()
  ```
