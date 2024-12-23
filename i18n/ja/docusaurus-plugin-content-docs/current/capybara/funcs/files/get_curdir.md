# get_curdir

> [get_curdir() -> str](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/custom_path.py#L8)

- **説明**：現在の作業ディレクトリのパスを取得する。この作業ディレクトリは「この関数が呼ばれた」瞬間、つまりその Python ファイルが存在するディレクトリを指します。一般的に、この方法は相対パスの基準として使用されます。

- **返り値**

  - **str**：現在の作業ディレクトリのパス。

- **例**

  ```python
  import capybara as cb

  DIR = cb.get_curdir()
  print(DIR)
  # >>> '/path/to/your/current/directory'
  ```
