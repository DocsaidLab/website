---
sidebar_position: 2
---

# get_curdir

>[get_curdir() -> str](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/custom_path.py#L8)

- **Description**

    Get the path of the current working directory. Here, the working directory refers to the directory where the Python file calling this function is located. Typically, this is used as a reference for relative paths.

- **Returns**

    - **str**: The path of the current working directory.

- **Example**

    ```python
    import docsaidkit as D

    DIR = D.get_curdir()
    print(DIR)
    # >>> '/path/to/your/current/directory'
    ```
