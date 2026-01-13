# download_from_google

> [download_from_google(file_id: str, file_name: str, target: str | Path = ".") -> Path](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/utils.py)

- **説明**：Google Drive からファイルをダウンロードし、大きなファイルの確認トークン（confirmation token）も処理します。

- **パラメータ**

  - **file_id** (`str`)：Google Drive の file id。
  - **file_name** (`str`)：保存するファイル名。
  - **target** (`str | Path`, optional)：保存先ディレクトリ。デフォルトは `"."`。

- **戻り値**

  - **Path**：ダウンロードしたファイルのパス。

- **例外**

  - **Exception**：レスポンスからダウンロードリンク／確認パラメータをパースできない場合。
  - **RuntimeError**：ファイル書き込み処理に失敗した場合。

- **備考**

  - 小さなファイルと大きなファイルの両方に対応します。大きなファイルの場合、Google の confirmation token を自動的に処理します。

- **例**

  ```python
  from capybara.utils import download_from_google

  path = download_from_google(
      file_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      file_name="example_file.txt",
  )

  path = download_from_google(
      file_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      file_name="example_file.txt",
      target="./downloads",
  )
  ```
