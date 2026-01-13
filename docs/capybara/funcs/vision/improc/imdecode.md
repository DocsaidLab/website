# imdecode

> [imdecode(byte\_: bytes) -> np.ndarray | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **說明**：將圖像字節串解碼為 NumPy 圖像數組。

- **參數**

  - **byte\_** (`bytes`)：要解碼的圖像字節串。

- **傳回值**

  - **np.ndarray | None**：解碼後的圖像數組；解碼失敗時回傳 `None`。

- **範例**

  ```python
  from capybara.vision.improc import IMGTYP, imdecode, imencode, imread

  img = imread('lena.png')
  encoded_bytes = imencode(img, imgtyp=IMGTYP.PNG)
  decoded_img = imdecode(encoded_bytes)
  ```
