# imwrite

> [imwrite(img: np.ndarray, path: Union[str, Path] = None, color_base: str = 'BGR', suffix: str = '.jpg') -> bool](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L245)

- **說明**：將影像寫入檔案，並可選擇轉換色彩空間。不給定路徑時，將寫入臨時檔案。

- **參數**

  - **img** (`np.ndarray`)：要寫入的影像，以 numpy ndarray 表示。
  - **path** (`Union[str, Path]`)：要寫入影像檔案的路徑。如果為 None，則寫入臨時檔案。預設為 None。
  - **color_base** (`str`)：影像的當前色彩空間。如果不是 `BGR`，函數將嘗試將其轉換為 `BGR`。預設為 `BGR`。
  - **suffix** (`str`)：如果 path 為 None，則臨時檔案的後綴。預設為 `.jpg`。

- **傳回值**

  - **bool**：如果寫入操作成功，則返回 True，否則返回 False。

- **範例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  cb.imwrite(img, 'lena.jpg')
  ```
