---
sidebar_position: 5
---

# load_pickle

> [load_pickle(path: Union[str, Path]) -> dict](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/files_utils.py#L159)

- **Description**

    Read a pickle file.

- **Parameters**
    - **path** (`Union[str, Path]`): The path to the pickle file.

- **Returns**
    - **dict**: The content of the pickle file.

- **Example**

    ```python
    import docsaidkit as D

    path = '/path/to/your/pickle'
    data = D.load_pickle(path)
    print(data)
    # >>> {'key': 'value'}
    ```
