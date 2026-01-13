# DataclassCopyMixin

> [DataclassCopyMixin](https://github.com/DocsaidLab/Capybara/blob/main/capybara/mixins.py)

- **Description**: Provides methods for copying DataClass objects, allowing you to clone DataClass instances.

- **Example**

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
