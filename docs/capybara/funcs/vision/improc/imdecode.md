# imdecode

> [imdecode(byte\_: bytes) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L107)

- **說明**：將圖像字節串解碼為 NumPy 圖像數組。

- **參數**

  - **byte\_** (`bytes`)：要解碼的圖像字節串。

- **傳回值**

  - **np.ndarray**：解碼後的圖像數組。

- **範例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  encoded_bytes = cb.imencode(img, IMGTYP=cb.IMGTYP.PNG)
  decoded_img = cb.imdecode(encoded_bytes)
  ```
