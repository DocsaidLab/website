# pdf2imgs

> [pdf2imgs(stream: str | Path | bytes) -> list[np.ndarray] | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **說明**：將 PDF 文件轉換為 numpy 格式的圖片列表。

- **依賴**

  - 需要系統安裝 `poppler`（例如 Ubuntu 的 `poppler-utils`），否則 `pdf2image` 可能無法正確轉換。

- **參數**

  - **stream** (`Union[str, Path, bytes]`)：PDF 文件的路徑或二進制數據。

- **傳回值**

  - **list[np.ndarray] | None**：成功時回傳 PDF 每一頁的 BGR numpy 影像；失敗時回傳 `None`。

- **範例**

  ```python
  from capybara.vision.improc import pdf2imgs

  imgs = pdf2imgs('sample.pdf')
  ```
