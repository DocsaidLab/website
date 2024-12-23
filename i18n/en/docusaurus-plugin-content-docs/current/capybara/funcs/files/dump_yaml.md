# dump_yaml

> [dump_yaml(obj, path: Union[str, Path] = None, \*\*kwargs) -> None](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/files_utils.py#L200)

- **Description**: Writes an object to a YAML file.

- **Parameters**:

  - **obj** (`Any`): The object to write.
  - **path** (`Union[str, Path]`): The path for the YAML file. Defaults to None, which writes to `tmp.yaml` in the current directory.
  - **kwargs**: Additional parameters for `yaml.dump`.

- **Example**:

  ```python
  import capybara as cb

  data = {'key': 'value'}
  cb.dump_yaml(data)
  ```
