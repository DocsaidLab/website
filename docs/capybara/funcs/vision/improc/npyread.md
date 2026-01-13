# npyread

> [npyread(path: str | Path) -> np.ndarray | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **說明**：從 NumPy `.npy` 文件中讀取圖像數組。

- **參數**

  - **path** (`Union[str, Path]`)：`.npy` 文件的路徑。

- **傳回值**

  - **np.ndarray**：讀取的圖像數組。如果讀取失敗，則返回 `None`。

- **範例**

  ```python
  from capybara.vision.improc import npyread

  img = npyread('lena.npy')
  ```
