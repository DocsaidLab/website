# load_pickle

> [load_pickle(path: Union[str, Path]) -> dict](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/files_utils.py#L159)

- **Description**: Reads a pickle file.

- **Parameters**:

  - **path** (`Union[str, Path]`): The path to the pickle file.

- **Return Value**:

  - **dict**: The content of the pickle file.

- **Example**:

  ```python
  import capybara as cb

  path = '/path/to/your/pickle'
  data = cb.load_pickle(path)
  print(data)
  # >>> {'key': 'value'}
  ```
