# npyread

> [npyread(path: str | Path) -> np.ndarray | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **説明**：NumPy の`.npy`ファイルから画像配列を読み込みます。

- **パラメータ**

  - **path** (`Union[str, Path]`)：`.npy`ファイルのパス。

- **戻り値**

  - **np.ndarray**：読み込んだ画像配列。読み込みに失敗した場合は`None`を返します。

- **使用例**

  ```python
  from capybara.vision.improc import npyread

  img = npyread('lena.npy')
  ```
