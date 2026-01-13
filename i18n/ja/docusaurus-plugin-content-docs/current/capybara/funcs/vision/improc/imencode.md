# imencode

> [imencode(img: np.ndarray, imgtyp: str | int | IMGTYP = IMGTYP.JPEG, **kwargs: object) -> bytes | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **説明**：NumPy 画像配列を指定された形式のバイト列にエンコードします。

- **パラメータ**

  - **img** (`np.ndarray`)：エンコードする画像配列。
  - **imgtyp** (`str | int | IMGTYP`)：画像タイプ。`IMGTYP.JPEG` / `IMGTYP.PNG` を指定できます。デフォルトは `IMGTYP.JPEG`。

- **戻り値**

  - **bytes | None**：エンコードされた画像のバイト列。失敗時は `None`。

- **備考**

  - 互換性のため `IMGTYP=...` も受け付けます（`imgtyp` と同時に渡すと `TypeError`）。

- **使用例**

  ```python
  from capybara.vision.improc import IMGTYP, imencode, imread

  img = imread('lena.png')
  encoded_bytes = imencode(img, imgtyp=IMGTYP.PNG)
  ```
