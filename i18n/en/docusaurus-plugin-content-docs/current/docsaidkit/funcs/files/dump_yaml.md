---
sidebar_position: 8
---

# dump_yaml

> [dump_yaml(obj, path: Union[str, Path] = None, **kwargs) -> None](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/files_utils.py#L200)

- **Description**

    Write an object to a YAML file.

- **Parameters**

    - **obj** (`Any`): The object to write.
    - **path** (`Union[str, Path]`): The path to the YAML file. Defaults to None, indicating writing to `tmp.yaml` in the current directory.
    - `**kwargs`: Additional parameters for `yaml.dump`.

- **Example**

    ```python
    import docsaidkit as D

    data = {'key': 'value'}
    D.dump_yaml(data)
    ```
