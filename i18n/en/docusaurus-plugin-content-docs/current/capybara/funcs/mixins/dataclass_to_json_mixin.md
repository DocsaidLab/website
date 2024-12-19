---
sidebar_position: 3
---

# DataClassToJsonMixin

> [DataClassToJsonMixin](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/mixins.py#L90)


- **Description**

    Provides a JSON serialization method for DataClass objects, enabling the conversion of DataClass objects into JSON format.

- **Supported Formats for dict_to_jsonable**
    - **Box**, **Boxes**: Converted to `List[float]` format.
    - **Polygon**, **Polygons**: Converted to `List[List[float]]` format.
    - **np.ndarray**, **np.generic**: Converted to `List` format.
    - **list**, **tuple**: Recursively converted to `List` format.
    - **Enum**: Converted to `str` format.
    - **Mapping**: Recursively converted to `Dict` format.

- **Example**

    ```python
    from dataclasses import dataclass
    from docsaidkit import DataclassToJsonMixin

    @dataclass
    class Person(DataclassToJsonMixin):
        name: str
        age: int

    person = Person('Alice', 20)
    print(person.be_jsonable())
    # >>> OrderedDict([('name', 'Alice'), ('age', 20)])
    ```
