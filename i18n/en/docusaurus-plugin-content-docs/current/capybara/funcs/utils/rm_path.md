# rm_path

> [rm_path(path: str | Path) -> None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/custom_path.py)

- **Description**: Removes a path/file.

- **Parameters**

  - **path** (`Union[str, Path]`): The path/file to be removed.

- **Behavior**

  - If `path` is a directory and not a symlink: removed recursively via `shutil.rmtree`.
  - Otherwise (regular file/symlink): removed via `Path.unlink()`.

- **Example**

  ```python
  from capybara.utils import rm_path

  path = '/path/to/your/directory'
  rm_path(path)
  ```
