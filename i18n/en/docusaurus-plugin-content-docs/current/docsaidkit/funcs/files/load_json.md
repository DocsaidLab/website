---
sidebar_position: 3
---

# load_json

> [load_json(path: Union[Path, str], **kwargs) -> dict](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/files_utils.py#L50)

- **Description**

    Read a JSON file. Here, `ujson` is used for reading because it is much faster than `json`.

- **Parameters**
    - **path** (`Union[Path, str]`): The path to the JSON file.
    - `**kwargs`: Additional parameters for `ujson.load`.

- **Returns**
    - **dict**: The content of the JSON file.

- **Example**

    ```python
    import docsaidkit as D

    path = '/path/to/your/json'
    data = D.load_json(path)
    print(data)
    # >>> {'key': 'value'}
    ```
