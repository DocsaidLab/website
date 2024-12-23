# get_curdir

> [get_curdir() -> str](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/custom_path.py#L8)

- **Description**: Retrieves the current working directory path. The working directory here refers to the directory of the Python file that calls this function. This is typically used as a reference for relative paths.

- **Return Value**:

  - **str**: The path of the current working directory.

- **Example**:

  ```python
  import capybara as cb

  DIR = cb.get_curdir()
  print(DIR)
  # >>> '/path/to/your/current/directory'
  ```
