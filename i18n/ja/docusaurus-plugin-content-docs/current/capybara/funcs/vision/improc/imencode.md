---
sidebar_position: 7
---

# imencode

> [imencode(img: np.ndarray, IMGTYP: Union[str, int, IMGTYP] = IMGTYP.JPEG) -> Union[bytes, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L100)

- **説明**：NumPy 画像配列を指定された形式のバイト文字列にエンコードします。

- 引数

  - **img** (`np.ndarray`)：エンコードする画像配列。
  - **IMGTYP** (`Union[str, int, IMGTYP]`)：画像タイプ。サポートされているタイプは `IMGTYP.JPEG` と `IMGTYP.PNG` です。デフォルトは `IMGTYP.JPEG`。

- **返り値**

  - **bytes**：エンコード後の画像バイト文字列。

- **例**

  ```python
  import docsaidkit as D

  img = D.imread('lena.png')
  encoded_bytes = D.imencode(img, IMGTYP=D.IMGTYP.PNG)
  ```
