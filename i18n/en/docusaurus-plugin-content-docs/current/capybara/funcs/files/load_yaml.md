---
sidebar_position: 7
---

# load_yaml

> [load_yaml(path: Union[Path, str]) -> dict](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/files_utils.py#L185)

- **Description**

    Read a YAML file.

- **Parameters**:
    - **path** (`Union[Path, str]`): The path to the YAML file.

- **Returns**:
    - **dict**: The content of the YAML file.

- **Example**:

    ```python
    import docsaidkit as D

    path = '/path/to/your/yaml'
    data = D.load_yaml(path)
    print(data)
    # >>> {'key': 'value'}
    ```
