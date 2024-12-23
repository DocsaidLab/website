# imrotate90

> [imrotate90(img: np.ndarray, rotate_code: ROTATE) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/geometric.py#L66)

- **説明**：入力画像を 90 度回転させる処理を行います。

- **パラメータ**

  - **img** (`np.ndarray`)：回転させる入力画像。
  - **rotate_code** (`RotateCode`)：回転コード。選べるオプションは次の通り：
    - ROTATE.ROTATE_90： 90 度。
    - ROTATE.ROTATE_180： 180 度回転。
    - ROTATE.ROTATE_270：反時計回りに 90 度回転。

- **戻り値**

  - **np.ndarray**：回転後の画像。

- **使用例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  rotate_img = cb.imrotate90(img, cb.ROTATE.ROTATE_270)
  ```

  ![imrotate90](./resource/test_imrotate90.jpg)
