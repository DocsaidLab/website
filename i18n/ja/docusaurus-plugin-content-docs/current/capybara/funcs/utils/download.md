# download

## download_from_google

> [download_from_google(file_id: str, file_name: str, target: str = ".") -> None](https://github.com/DocsaidLab/Capybara/blob/c83d96363a3de3686a98dd7b558a168f08b9bc97/capybara/utils/utils.py#L70)

- **説明**：Google ドライブからファイルをダウンロードし、大きなファイルの確認問題を処理します。

- **パラメーター**

  - **file_id** (`str`)：Google ドライブからダウンロードするファイルの ID。
  - **file_name** (`str`)：ダウンロード後に保存するファイル名。
  - **target** (`str`, オプション)：ファイルを保存するターゲットディレクトリ。デフォルトは現在のディレクトリ `"."`。

- **例外**

  - **Exception**：ダウンロードに失敗した場合やファイルが作成できない場合、例外が発生します。

- **注意事項**

  - この関数は、小さなファイルと大きなファイルの両方を処理します。大きなファイルの場合、Google の確認手続きを自動的に処理し、ウイルススキャンやファイルサイズ制限の警告を回避します。
  - ターゲットディレクトリが存在することを確認してください。存在しない場合は自動的に作成されます。

- **例**

  ```python
  # 例 1：現在のディレクトリにファイルをダウンロード
  download_from_google(
      file_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      file_name="example_file.txt"
  )

  # 例 2：指定したディレクトリにファイルをダウンロード
  download_from_google(
      file_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      file_name="example_file.txt",
      target="./downloads"
  )
  ```
