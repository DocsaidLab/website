# load_yaml

> [load_yaml(path: Path | str) -> dict](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/files_utils.py)

- **Description**: Reads a YAML file.

- **Parameters**:

  - **path** (`Union[Path, str]`): The path to the YAML file.

- **Return Value**:

  - **dict**: The content of the YAML file.

- **Example**:

  ```python
  from capybara.utils.files_utils import load_yaml

  path = '/path/to/your/yaml'
  data = load_yaml(path)
  print(data)
  # >>> {'key': 'value'}
  ```
