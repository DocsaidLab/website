# PNG Process

## pngencode

> [pngencode(img: np.ndarray, compression: int = 1) -> bytes | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **説明**：NumPy の画像配列を PNG bytes にエンコードします。

- **パラメータ**

  - **img** (`np.ndarray`)：エンコードする画像配列。
  - **compression** (`int`)：圧縮レベル（0〜9）。0 は無圧縮、9 は最大圧縮。デフォルトは 1。

- **戻り値**

  - **bytes | None**：PNG bytes。失敗時は `None`。

- **例**

  ```python
  from capybara.vision.improc import imread, pngencode

  img = imread('lena.png')
  encoded_bytes = pngencode(img, compression=9)
  ```

## pngdecode

> [pngdecode(byte\_: bytes) -> np.ndarray | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **説明**：PNG bytes を NumPy 画像配列にデコードします。

- **パラメータ**

  - **byte\_** (`bytes`)：デコードする PNG bytes。

- **戻り値**

  - **np.ndarray | None**：NumPy 画像配列。失敗時は `None`。

- **例**

  ```python
  from capybara.vision.improc import pngdecode

  decoded_img = pngdecode(encoded_bytes)
  ```

