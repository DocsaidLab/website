# load_json

> [load_json(path: Path | str, **kwargs) -> dict](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/files_utils.py)

- **Description**: Reads a JSON file. It uses `ujson` for faster reading compared to the standard `json` module.

- **Parameters**:

  - **path** (`Union[Path, str]`): The path to the JSON file.
  - **kwargs**: Additional parameters for `ujson.load`.

- **Return Value**:

  - **dict**: The content of the JSON file.

- **Example**:

  ```python
  from capybara.utils.files_utils import load_json

  path = '/path/to/your/json'
  data = load_json(path)
  print(data)
  # >>> {'key': 'value'}
  ```
