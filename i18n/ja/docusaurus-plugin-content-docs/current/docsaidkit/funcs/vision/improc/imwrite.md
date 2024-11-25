---
sidebar_position: 2
---

# imwrite

> [imwrite(img: np.ndarray, path: Union[str, Path] = None, color_base: str = 'BGR', suffix: str = '.jpg') -> bool](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/improc.py#L245C1-L272C67)

- **説明**：画像をファイルに書き込むことができ、色空間の変換を選択できます。パスを指定しない場合、画像は一時ファイルに書き込まれます。

- 引数

  - **img** (`np.ndarray`)：書き込む画像、NumPy ndarray として表現されます。
  - **path** (`Union[str, Path]`)：画像ファイルの保存先パス。`None`の場合、一時ファイルに書き込まれます。デフォルトは`None`。
  - **color_base** (`str`)：画像の現在の色空間。`BGR`でない場合、関数はそれを`BGR`に変換しようとします。デフォルトは`BGR`です。
  - **suffix** (`str`)：`path`が`None`の場合、一時ファイルの拡張子。デフォルトは`.jpg`です。

- **返り値**

  - **bool**：書き込み操作が成功した場合は`True`、失敗した場合は`False`を返します。

- **例**

  ```python
  import docsaidkit as D

  img = D.imread('lena.png')
  D.imwrite(img, 'lena.jpg')
  ```
