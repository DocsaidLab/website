---
sidebar_position: 3
---

# pdf2imgs

> [pdf2imgs(stream: Union[str, Path, bytes]) -> Union[List[np.ndarray], None]](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/improc.py#L275C1-L292C15)

- **説明**：PDF ファイルを NumPy 形式の画像リストに変換します。

- 引数

  - **stream** (`Union[str, Path, bytes]`)：PDF ファイルのパスまたはバイナリデータ。

- **返り値**

  - **List[np.ndarray]**：成功した場合、PDF ファイルの各ページを NumPy 画像として格納したリストを返します。失敗した場合は`None`を返します。

- **例**

  ```python
  import docsaidkit as D

  imgs = D.pdf2imgs('sample.pdf')
  ```
