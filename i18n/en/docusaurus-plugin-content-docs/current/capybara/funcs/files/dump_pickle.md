# dump_pickle

> [dump_pickle(obj: Any, path: str | Path) -> None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/files_utils.py)

- **Description**: Writes an object to a pickle file.

- **Parameters**:

  - **obj** (`Any`): The object to write.
  - **path** (`Union[str, Path]`): The path for the pickle file.

- **Example**:

  ```python
  from capybara.utils.files_utils import dump_pickle

  data = {'key': 'value'}
  dump_pickle(data, '/path/to/your/pickle')
  ```
