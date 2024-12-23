# dump_pickle

> [dump_pickle(obj, path: Union[str, Path]) -> None](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/files_utils.py#L173)

- **Description**: Writes an object to a pickle file.

- **Parameters**:

  - **obj** (`Any`): The object to write.
  - **path** (`Union[str, Path]`): The path for the pickle file.

- **Example**:

  ```python
  import capybara as cb

  data = {'key': 'value'}
  cb.dump_pickle(data, '/path/to/your/pickle')
  ```
