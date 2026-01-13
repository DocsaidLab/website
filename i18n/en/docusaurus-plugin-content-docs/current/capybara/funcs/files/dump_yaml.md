# dump_yaml

> [dump_yaml(obj: Any, path: str | Path | None = None, **kwargs) -> None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/files_utils.py)

- **Description**: Writes an object to a YAML file.

- **Parameters**:

  - **obj** (`Any`): The object to write.
  - **path** (`Union[str, Path]`): The path for the YAML file. Defaults to None, which writes to `tmp.yaml` in the current directory.
  - **kwargs**: Additional parameters for `yaml.dump`.

- **Example**:

  ```python
  from capybara.utils.files_utils import dump_yaml

  data = {'key': 'value'}
  dump_yaml(data)
  ```
