# dump_json

> [dump_json(obj: Any, path: str | Path | None = None, **kwargs) -> None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/files_utils.py)

- **Description**: Writes an object to a JSON file. It uses `ujson` for faster writing compared to the standard `json` module.

- **Parameters**:

  - **obj** (`Any`): The object to write.
  - **path** (`Union[str, Path]`): The path for the JSON file. Defaults to None, which writes to `tmp.json` in the current directory.
  - **kwargs**: Additional parameters for `ujson.dump`.

- **Example**:

  ```python
  from capybara.utils.files_utils import dump_json

  data = {'key': 'value'}
  dump_json(data)
  ```
