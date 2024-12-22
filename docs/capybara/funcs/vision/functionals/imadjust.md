# imadjust

> [imadjust(img: np.ndarray, rng_out: Tuple[int, int] = (0, 255), gamma: float = 1.0, color_base: str = 'BGR') -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/functionals.py#L122)

- **說明**：調整影像的強度。

- **參數**

  - **img** (`np.ndarray`)：要進行強度調整的輸入影像。應為 2-D 或 3-D。
  - **rng_out** (`Tuple[int, int]`)：輸出影像的強度目標範圍。預設為 (0, 255)。
  - **gamma** (`float`)：用於伽瑪校正的值。如果 gamma 小於 1，則映射將偏向於較高（較亮）的輸出值。如果 gamma 大於 1，則映射將偏向於較低（較暗）的輸出值。預設為 1.0（線性映射）。
  - **color_base** (`str`)：輸入影像的顏色基礎。應為 'BGR' 或 'RGB'。預設為 'BGR'。

- **傳回值**

  - **np.ndarray**：調整後的影像。

- **範例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  adj_img = cb.imadjust(img, gamma=2)
  ```

  ![imadjust](./resource/test_imadjust.jpg)
