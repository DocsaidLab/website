---
sidebar_position: 2
---

# DataClassCopyMixin

> [DataClassCopyMixin](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/mixins.py#L77)

- **Description**

    Provides a cloning method for DataClass objects, allowing the duplication of DataClass objects.

- **Example**

    ```python
    from dataclasses import dataclass
    from docsaidkit import DataclassCopyMixin

    @dataclass
    class Person(DataclassCopyMixin):
        name: str
        age: int

    person = Person('Alice', 20)
    person_copy = person.__copy__()
    person_deepcopy = person.__deepcopy__()
    ```
