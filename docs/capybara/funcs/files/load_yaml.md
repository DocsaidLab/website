# load_yaml

> [load_yaml(path: Path | str) -> dict](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/files_utils.py)

- **說明**：讀取 yaml 檔案。

- **參數**

  - **path** (`Union[Path, str]`)：yaml 檔案的路徑。

- **傳回值**

  - **dict**：yaml 檔案的內容。

- **範例**

  ```python
  from capybara.utils.files_utils import load_yaml

  path = '/path/to/your/yaml'
  data = load_yaml(path)
  print(data)
  # >>> {'key': 'value'}
  ```
