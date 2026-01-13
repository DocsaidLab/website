# imdecode

> [imdecode(byte\_: bytes) -> np.ndarray | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **説明**：画像のバイト列を NumPy 画像配列にデコードします。

- **パラメータ**

  - **byte\_** (`bytes`)：デコードする画像のバイト列。

- **戻り値**

  - **np.ndarray | None**：デコードされた画像配列。失敗時は `None`。

- **使用例**

  ```python
  from capybara.vision.improc import IMGTYP, imdecode, imencode, imread

  img = imread('lena.png')
  encoded_bytes = imencode(img, imgtyp=IMGTYP.PNG)
  decoded_img = imdecode(encoded_bytes)
  ```
