# imread

> [imread(path: Union[str, Path], color_base: str = 'BGR', verbose: bool = False) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L197)

- **説明**：画像を読み込み、異なる画像形式に基づいて異なる読み込み方法を使用します。サポートされている形式は以下の通りです：

  - `.heic`：`read_heic_to_numpy`を使用して読み込み、`BGR`形式に変換します。
  - `.jpg`：`jpgread`を使用して読み込み、`BGR`形式に変換します。
  - その他の形式：`cv2.imread`を使用して読み込み、`BGR`形式に変換します。
  - `jpgread`で`None`が返される場合は、`cv2.imread`を使用して読み込みます。

- **パラメータ**

  - **path** (`Union[str, Path]`)：読み込む画像のパス。
  - **color_base** (`str`)：画像の色空間。`BGR`でない場合、`imcvtcolor`関数を使用して変換します。デフォルトは`BGR`。
  - **verbose** (`bool`)：True に設定すると、読み込んだ画像が None の場合に警告を出します。デフォルトは False。

- **戻り値**

  - **np.ndarray**：成功時に画像の NumPy ndarray を返し、失敗時には`None`を返します。

- **使用例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  ```
