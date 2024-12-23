# imwrite

> [imwrite(img: np.ndarray, path: Union[str, Path] = None, color_base: str = 'BGR', suffix: str = '.jpg') -> bool](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L245)

- **説明**：画像をファイルに書き込むとともに、必要に応じて色空間の変換を行います。パスを指定しない場合は、一時ファイルに書き込まれます。

- **パラメータ**

  - **img** (`np.ndarray`)：書き込む画像、NumPy ndarray として表現されます。
  - **path** (`Union[str, Path]`)：画像ファイルを保存するパス。None の場合、一時ファイルに保存されます。デフォルトは None。
  - **color_base** (`str`)：画像の現在の色空間。`BGR`でない場合、関数はこれを`BGR`に変換しようとします。デフォルトは`BGR`。
  - **suffix** (`str`)：path が None の場合、一時ファイルの拡張子。デフォルトは`.jpg`。

- **戻り値**

  - **bool**：書き込み操作が成功した場合は True、それ以外は False。

- **使用例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  cb.imwrite(img, 'lena.jpg')
  ```
