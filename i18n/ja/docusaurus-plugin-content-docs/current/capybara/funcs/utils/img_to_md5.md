---
sidebar_position: 8
---

# img_to_md5

> [img_to_md5(img: np.ndarray) -> str](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/files_utils.py#L42)

- **説明**：指定された画像から MD5 ハッシュを生成します。`gen_md5`と同じ理由で存在しており、異なる点はこの関数が画像データを入力として使用することです。

- **パラメータ**

  - **img** (`np.ndarray`)：画像。

- **戻り値**

  - **str**：MD5 ハッシュ。

- **例**

  ```python
  import docsaidkit as D

  img = D.imread('lena.png')
  md5 = D.img_to_md5(img)
  print(md5)
  # >>> 'd41d8cd98f00b204e9800998ecf8427e'
  ```
