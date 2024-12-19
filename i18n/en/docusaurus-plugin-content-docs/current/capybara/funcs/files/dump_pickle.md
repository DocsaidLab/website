---
sidebar_position: 6
---

# dump_pickle

> [dump_pickle(obj, path: Union[str, Path]) -> None](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/files_utils.py#L173)

- **Description**

    Write an object to a pickle file.

- **Parameters**

    - **obj** (`Any`): The object to write.
    - **path** (`Union[str, Path]`): The path to the pickle file.

- **Example**

    ```python
    import docsaidkit as D

    data = {'key': 'value'}
    D.dump_pickle(data, '/path/to/your/pickle')
    ```
