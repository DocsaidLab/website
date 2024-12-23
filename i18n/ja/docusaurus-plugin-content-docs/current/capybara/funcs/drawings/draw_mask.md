# draw_mask

> [draw_mask(img: np.ndarray, mask: np.ndarray, colormap: int = cv2.COLORMAP_JET, weight: Tuple[float, float] = (0.5, 0.5), gamma: float = 0, min_max_normalize: bool = False) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/visualization/draw.py#L507)

- **説明**：画像にマスクを描画します。

  - **img** (`np.ndarray`)：描画する画像。
  - **mask** (`np.ndarray`)：描画するマスク。
  - **colormap** (`int`)：マスクに使用するカラーマップ。デフォルトは `cv2.COLORMAP_JET`。
  - **weight** (`Tuple[float, float]`)：画像とマスクの重み。デフォルトは(0.5, 0.5)。
  - **gamma** (`float`)：マスクのガンマ値。デフォルトは 0。
  - **min_max_normalize** (`bool`)：マスクを範囲[0, 1]に正規化するかどうか。デフォルトは False。

- **戻り値**

  - **np.ndarray**：マスクを描画した画像。

- **例**

  ```python
  import capybara as cb
  import numpy as np

  img = cb.imread('lena.png')
  polygon = cb.Polygon([(20, 20), (100, 20), (80, 80), (20, 40)])
  mask = cb.draw_polygon(np.zeros_like(img), polygon, fillup=True, color=255)
  mask_img = cb.draw_mask(img, mask, colormap=cv2.COLORMAP_JET, weight=(0.5, 0.5), gamma=0, min_max_normalize=False)
  ```

  ![draw_mask](./resource/test_draw_mask.jpg)
