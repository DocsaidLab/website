# pdf2imgs

> [pdf2imgs(stream: Union[str, Path, bytes]) -> Union[List[np.ndarray], None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L275)

- **説明**：PDF ファイルを NumPy 形式の画像リストに変換します。

- **パラメータ**

  - **stream** (`Union[str, Path, bytes]`)：PDF ファイルのパスまたはバイナリデータ。

- **戻り値**

  - **List[np.ndarray]**：PDF ファイルの各ページを NumPy 画像としてリストで返します。失敗した場合は None を返します。

- **使用例**

  ```python
  import capybara as cb

  imgs = cb.pdf2imgs('sample.pdf')
  ```
