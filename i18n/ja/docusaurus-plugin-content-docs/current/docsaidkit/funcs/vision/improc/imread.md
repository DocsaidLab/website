---
sidebar_position: 1
---

# imread

> [imread(path: Union[str, Path], color_base: str = 'BGR', verbose: bool = False) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/improc.py#L197C1-L242C15)

- **説明**：画像を読み込み、画像形式に応じて異なる読み込み方法を使用します。サポートされる形式は以下の通りです：

  - `.heic`：`read_heic_to_numpy`を使用して読み込み、`BGR`形式に変換します。
  - `.jpg`：`jpgread`を使用して読み込み、`BGR`形式に変換します。
  - その他の形式：`cv2.imread`を使用して読み込み、`BGR`形式に変換します。
  - `jpgread`で`None`が返された場合、`cv2.imread`を使用して読み込みます。

- 引数

  - **path** (`Union[str, Path]`)：読み込む画像のパス。
  - **color_base** (`str`)：画像の色空間。`BGR`でない場合、`imcvtcolor`関数を使って変換します。デフォルトは`BGR`です。
  - **verbose** (`bool`)：`True`に設定すると、画像が`None`の場合に警告を表示します。デフォルトは`False`です。

- **返り値**

  - **np.ndarray**：成功した場合、画像の NumPy ndarray を返します。失敗した場合は`None`を返します。

- **例**

  ```python
  import docsaidkit as D

  img = D.imread('lena.png')
  ```
