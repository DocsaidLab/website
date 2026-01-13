# pdf2imgs

> [pdf2imgs(stream: str | Path | bytes) -> list[np.ndarray] | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **説明**：PDF ファイルを NumPy 形式の画像リストに変換します。

- **依存関係**

  - `poppler`（例：Ubuntu の `poppler-utils`）が必要です。未インストールの場合、`pdf2image` が正しく変換できないことがあります。

- **パラメータ**

  - **stream** (`Union[str, Path, bytes]`)：PDF ファイルのパスまたはバイナリデータ。

- **戻り値**

  - **List[np.ndarray]**：PDF ファイルの各ページを NumPy 画像としてリストで返します。失敗した場合は None を返します。

- **使用例**

  ```python
  from capybara.vision.improc import pdf2imgs

  imgs = pdf2imgs('sample.pdf')
  ```
