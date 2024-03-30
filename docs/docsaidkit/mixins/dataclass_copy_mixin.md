---
sidebar_position: 2
---

# DataClassCopyMixin

> [DataClassCopyMixin](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/mixins.py#L77)

- **說明**：提供 DataClass 物件的複製方法，可以用來複製 DataClass 物件。

- **範例**

    ```python
    from dataclasses import dataclass
    from docsaidkit import DataclassCopyMixin

    @dataclass
    class Person(DataclassCopyMixin):
        name: str
        age: int
    ```
