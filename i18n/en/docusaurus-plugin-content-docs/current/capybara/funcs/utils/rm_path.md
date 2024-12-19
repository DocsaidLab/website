---
sidebar_position: 4
---

# rm_path

>[rm_path(path: Union[str, Path]) -> str](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/custom_path.py#L26)

- **Description**: Remove a path/file.

- **Parameters**:
    - **path** (`path: Union[str, Path]`): The path/file to remove.

- **Example**:

    ```python
    import docsaidkit as D

    path = '/path/to/your/directory'
    new_path = D.rm_path(path)
    ```
