---
sidebar_position: 8
---

# imdecode

> [imdecode(byte\_: bytes) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L107)

- **説明**：画像のバイト文字列を解凍して NumPy 画像配列に変換します。

- 引数

  - **byte\_** (`bytes`)：解凍する画像のバイト文字列。

- **返り値**

  - **np.ndarray**：解凍後の画像配列。

- **例**

  ```python
  import docsaidkit as D

  img = D.imread('lena.png')
  encoded_bytes = D.imencode(img, IMGTYP=D.IMGTYP.PNG)
  decoded_img = D.imdecode(encoded_bytes)
  ```
