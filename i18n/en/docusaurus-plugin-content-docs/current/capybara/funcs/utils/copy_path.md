# copy_path

> [copy_path(path_src: str | Path, path_dst: str | Path) -> None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/custom_path.py)

- **Description**: Copies a file.

- **Parameters**

  - **path_src** (`str | Path`): Source file path (must be a file).
  - **path_dst** (`str | Path`): Destination path.

- **Exceptions**

  - **ValueError**: Raised when `path_src` is not a file.

- **Example**

  ```python
  from capybara.utils import copy_path

  path_src = '/path/to/your/source'
  path_dst = '/path/to/your/destination'
  copy_path(path_src, path_dst)
  ```
