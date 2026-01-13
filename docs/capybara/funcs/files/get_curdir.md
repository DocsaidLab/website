# get_curdir

> [get_curdir(path: str | Path, absolute: bool = True) -> Path](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/custom_path.py)

- **說明**：取得指定 `path` 所在的資料夾路徑（也就是 `path.parent`）。

  典型用法是用 `__file__` 當作輸入，取得當前 Python 檔案所在目錄，做為相對路徑的基準。

  注意：這個函式**不會**回傳 `Path.cwd()`（也就是 shell 的當前工作目錄）。

- **參數**

  - **path** (`str | Path`)：任意檔案路徑（通常是 `__file__`）。
  - **absolute** (`bool`)：是否先將輸入轉為絕對路徑再取 parent。預設為 `True`。

- **傳回值**

  - **Path**：`path` 的 parent 目錄。

- **範例**

  ```python
  from capybara import get_curdir

  DIR = get_curdir(__file__)
  print(DIR)
  # >>> '/path/to/your/current/directory'
  ```
