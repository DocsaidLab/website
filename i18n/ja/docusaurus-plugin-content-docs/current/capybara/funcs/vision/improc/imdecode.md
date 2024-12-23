# imdecode

> [imdecode(byte\_: bytes) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L107)

- **説明**：画像のバイト列を NumPy 画像配列にデコードします。

- **パラメータ**

  - **byte\_** (`bytes`)：デコードする画像のバイト列。

- **戻り値**

  - **np.ndarray**：デコードされた画像配列。

- **使用例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  encoded_bytes = cb.imencode(img, IMGTYP=cb.IMGTYP.PNG)
  decoded_img = cb.imdecode(encoded_bytes)
  ```
