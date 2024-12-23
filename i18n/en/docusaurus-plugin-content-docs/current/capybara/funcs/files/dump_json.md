# dump_json

> [dump_json(obj: Any, path: Union[str, Path] = None, \*\*kwargs) -> None](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/files_utils.py#L65)

- **Description**: Writes an object to a JSON file. It uses `ujson` for faster writing compared to the standard `json` module.

- **Parameters**:

  - **obj** (`Any`): The object to write.
  - **path** (`Union[str, Path]`): The path for the JSON file. Defaults to None, which writes to `tmp.json` in the current directory.
  - **kwargs**: Additional parameters for `ujson.dump`.

- **Example**:

  ```python
  import capybara as cb

  data = {'key': 'value'}
  cb.dump_json(data)
  ```
