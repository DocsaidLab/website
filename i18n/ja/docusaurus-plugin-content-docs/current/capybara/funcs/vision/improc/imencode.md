# imencode

> [imencode(img: np.ndarray, IMGTYP: Union[str, int, IMGTYP] = IMGTYP.JPEG) -> Union[bytes, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L100)

- **説明**：NumPy 画像配列を指定された形式のバイト列にエンコードします。

- **パラメータ**

  - **img** (`np.ndarray`)：エンコードする画像配列。
  - **IMGTYP** (`Union[str, int, IMGTYP]`)：画像タイプ。サポートされているタイプは`IMGTYP.JPEG`と`IMGTYP.PNG`です。デフォルトは`IMGTYP.JPEG`。

- **戻り値**

  - **bytes**：エンコードされた画像のバイト列。

- **使用例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  encoded_bytes = cb.imencode(img, IMGTYP=cb.IMGTYP.PNG)
  ```
