# DataclassCopyMixin

> [DataclassCopyMixin](https://github.com/DocsaidLab/Capybara/blob/main/capybara/mixins.py)

- **說明**：提供 DataClass 物件的複製方法，可以用來複製 DataClass 物件。

- **範例**

  ```python
  import copy
  from dataclasses import dataclass
  from capybara import DataclassCopyMixin

  @dataclass
  class Person(DataclassCopyMixin):
      name: str
      age: int

  person = Person('Alice', 20)
  person_copy = copy.copy(person)
  person_deepcopy = copy.deepcopy(person)
  ```
