# get_curdir

> [get_curdir() -> str](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/custom_path.py#L8)

- **說明**：取得當前工作目錄的路徑。這裡的工作目錄是指「調用本函數」的當下，那個 Python 檔案所在的目錄。一般來說，我們會透過這種方式來作為相對路徑的基準。

- **傳回值**

  - **str**：當前工作目錄的路徑。

- **範例**

  ```python
  import capybara as cb

  DIR = cb.get_curdir()
  print(DIR)
  # >>> '/path/to/your/current/directory'
  ```
