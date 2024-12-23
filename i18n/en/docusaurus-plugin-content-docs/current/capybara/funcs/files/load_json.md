# load_json

> [load_json(path: Union[Path, str], \*\*kwargs) -> dict](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/files_utils.py#L50)

- **Description**: Reads a JSON file. It uses `ujson` for faster reading compared to the standard `json` module.

- **Parameters**:

  - **path** (`Union[Path, str]`): The path to the JSON file.
  - **kwargs**: Additional parameters for `ujson.load`.

- **Return Value**:

  - **dict**: The content of the JSON file.

- **Example**:

  ```python
  import capybara as cb

  path = '/path/to/your/json'
  data = cb.load_json(path)
  print(data)
  # >>> {'key': 'value'}
  ```
