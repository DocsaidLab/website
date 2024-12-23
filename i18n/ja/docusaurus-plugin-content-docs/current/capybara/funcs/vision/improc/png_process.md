# PNG Process

## pngencode

> [pngencode(img: np.ndarray, compression: int = 1) -> Union[bytes, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L80)

- **説明**：NumPy の画像配列を PNG 形式のバイト列にエンコードします。

- **パラメータ**：

  - **img** (`np.ndarray`)：エンコードする画像配列。
  - **compression** (`int`)：圧縮レベル、範囲は 0 から 9。0 は圧縮なし、9 は最大圧縮。デフォルトは 1。

- **戻り値**

  - **bytes**：エンコードされた PNG 形式のバイト列。

- **使用例**

  ```python
  import numpy as np
  import capybara as cb

  img_array = np.random.rand(100, 100, 3) * 255
  encoded_bytes = cb.pngencode(img_array, compression=9)
  ```

## pngdecode

> [pngdecode(byte\_: bytes) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L91)

- **説明**：PNG 形式のバイト列を NumPy の画像配列にデコードします。

- **パラメータ**：

  - **byte\_** (`bytes`)：デコードする PNG 形式のバイト列。

- **戻り値**

  - **np.ndarray**：デコードされた画像配列。

- **使用例**

  ```python
  decoded_img = cb.pngdecode(encoded_bytes)
  ```
