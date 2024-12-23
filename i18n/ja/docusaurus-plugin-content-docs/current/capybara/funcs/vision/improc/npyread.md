# npyread

> [npyread(path: Union[str, Path]) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L174)

- **説明**：NumPy の`.npy`ファイルから画像配列を読み込みます。

- **パラメータ**

  - **path** (`Union[str, Path]`)：`.npy`ファイルのパス。

- **戻り値**

  - **np.ndarray**：読み込んだ画像配列。読み込みに失敗した場合は`None`を返します。

- **使用例**

  ```python
  import capybara as cb

  img = cb.npyread('lena.npy')
  ```
