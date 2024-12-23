# DataClassCopyMixin

> [DataClassCopyMixin](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/mixins.py#L77)

- **Description**: Provides methods for copying DataClass objects, allowing you to clone DataClass instances.

- **Example**

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
