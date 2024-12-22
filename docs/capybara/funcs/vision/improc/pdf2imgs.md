# pdf2imgs

> [pdf2imgs(stream: Union[str, Path, bytes]) -> Union[List[np.ndarray], None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L275)

- **說明**：將 PDF 文件轉換為 numpy 格式的圖片列表。

- **參數**

  - **stream** (`Union[str, Path, bytes]`)：PDF 文件的路徑或二進制數據。

- **傳回值**

  - **List[np.ndarray]**：成功時返回 PDF 文件的每一頁的 numpy 圖片列表，否則返回 None。

- **範例**

  ```python
  import capybara as cb

  imgs = cb.pdf2imgs('sample.pdf')
  ```
