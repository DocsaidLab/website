# imencode

> [imencode(img: np.ndarray, imgtyp: str | int | IMGTYP = IMGTYP.JPEG, **kwargs: object) -> bytes | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **說明**：將 NumPy 圖像數組編碼為指定格式的字節串。

- **參數**

  - **img** (`np.ndarray`)：要編碼的圖像數組。
  - **imgtyp** (`str | int | IMGTYP`)：圖像類型。支援 `IMGTYP.JPEG` / `IMGTYP.PNG`。預設為 `IMGTYP.JPEG`。

- **傳回值**

  - **bytes | None**：編碼後的圖像字節串；編碼失敗時回傳 `None`。

- **備註**

  - 為了兼容舊版呼叫方式，亦接受 `IMGTYP=...`（與 `imgtyp` 同時提供會拋出 `TypeError`）。

- **範例**

  ```python
  from capybara.vision.improc import IMGTYP, imencode, imread

  img = imread('lena.png')
  encoded_bytes = imencode(img, imgtyp=IMGTYP.PNG)
  ```
