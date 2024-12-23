# load_yaml

> [load_yaml(path: Union[Path, str]) -> dict](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/files_utils.py#L185)

- **Description**: Reads a YAML file.

- **Parameters**:

  - **path** (`Union[Path, str]`): The path to the YAML file.

- **Return Value**:

  - **dict**: The content of the YAML file.

- **Example**:

  ```python
  import capybara as cb

  path = '/path/to/your/yaml'
  data = cb.load_yaml(path)
  print(data)
  # >>> {'key': 'value'}
  ```
