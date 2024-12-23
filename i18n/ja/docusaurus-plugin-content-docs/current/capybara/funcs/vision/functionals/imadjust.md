# imadjust

> [imadjust(img: np.ndarray, rng_out: Tuple[int, int] = (0, 255), gamma: float = 1.0, color_base: str = 'BGR') -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/functionals.py#L122)

- **説明**：画像の強度を調整します。

- **引数**

  - **img** (`np.ndarray`)：強度調整を行う入力画像。2D または 3D である必要があります。
  - **rng_out** (`Tuple[int, int]`)：出力画像の強度範囲。デフォルトは (0, 255)。
  - **gamma** (`float`)：ガンマ補正に使用する値。ガンマが 1 より小さい場合、マッピングは明るい値に偏り、1 より大きい場合、マッピングは暗い値に偏ります。デフォルトは 1.0（線形マッピング）。
  - **color_base** (`str`)：入力画像の色空間。'BGR' または 'RGB' のいずれか。デフォルトは 'BGR'。

- **戻り値**

  - **np.ndarray**：調整後の画像。

- **例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  adj_img = cb.imadjust(img, gamma=2)
  ```

  ![imadjust](./resource/test_imadjust.jpg)
