# get_curdir

> [get_curdir(path: str | Path, absolute: bool = True) -> Path](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/custom_path.py)

- **Description**: Returns the parent directory of the given `path` (i.e. `path.parent`).

  A typical usage is to pass `__file__` and use it as the base for relative paths.

  Note: this function does **not** return `Path.cwd()` (the shell working directory).

- **Parameters**

  - **path** (`str | Path`): Any file path (typically `__file__`).
  - **absolute** (`bool`): Whether to resolve the input to an absolute path before taking `parent`. Default is `True`.

- **Returns**

  - **Path**: The parent directory of `path`.

- **Example**:

  ```python
  from capybara import get_curdir

  DIR = get_curdir(__file__)
  print(DIR)
  # >>> '/path/to/your/current/directory'
  ```
