# load_pickle

> [load_pickle(path: str | Path)](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/files_utils.py)

- **Description**: Reads a pickle file.

- **Parameters**:

  - **path** (`Union[str, Path]`): The path to the pickle file.

- **Return Value**:

  - **Any**: The deserialized content.

- **Example**:

  ```python
  from capybara.utils.files_utils import load_pickle

  path = '/path/to/your/pickle'
  data = load_pickle(path)
  print(data)
  # >>> {'key': 'value'}
  ```
