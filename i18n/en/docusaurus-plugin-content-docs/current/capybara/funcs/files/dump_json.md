---
sidebar_position: 4
---

# dump_json

> [dump_json(obj: Any, path: Union[str, Path] = None, **kwargs) -> None](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/files_utils.py#L65)

- **Description**

    Write an object to JSON. Here, `ujson` is used for writing because it is much faster than `json`.

- **Parameters**

    - **obj** (`Any`): The object to write.
    - **path** (`Union[str, Path]`): The path to the JSON file. Defaults to None, indicating writing to `tmp.json` in the current directory.
    - `**kwargs`: Additional parameters for `ujson.dump`.

- **Example**

    ```python
    import docsaidkit as D

    data = {'key': 'value'}
    D.dump_json(data)
    ```
