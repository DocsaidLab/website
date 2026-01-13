# draw_mask

> [draw_mask(img: np.ndarray, mask: np.ndarray, colormap: int = cv2.COLORMAP_JET, weight: tuple[float, float] = (0.5, 0.5), gamma: float = 0, min_max_normalize: bool = False) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/visualization/draw.py)

- **依存関係**

  - `capybara-docsaid[visualization]` を先にインストールしてください。

- **説明**：画像にマスクを描画します。

- **パラメータ**

  - **img** (`np.ndarray`)：描画する画像。
  - **mask** (`np.ndarray`)：描画するマスク。
  - **colormap** (`int`)：マスクに使用するカラーマップ。デフォルトは `cv2.COLORMAP_JET`。
  - **weight** (`tuple[float, float]`)：画像とマスクの重み。デフォルトは(0.5, 0.5)。
  - **gamma** (`float`)：マスクのガンマ値。デフォルトは 0。
  - **min_max_normalize** (`bool`)：マスクを範囲[0, 1]に正規化するかどうか。デフォルトは False。

- **戻り値**

  - **np.ndarray**：マスクを描画した画像。

- **例**

  ```python
  import cv2
  import numpy as np
  from capybara import Polygon, imread
  from capybara.vision.visualization.draw import draw_mask, draw_polygon

  img = imread('lena.png')
  polygon = Polygon([(20, 20), (100, 20), (80, 80), (20, 40)])
  mask = draw_polygon(np.zeros_like(img), polygon, fillup=True, color=255)
  mask_img = draw_mask(
      img,
      mask,
      colormap=cv2.COLORMAP_JET,
      weight=(0.5, 0.5),
      gamma=0,
      min_max_normalize=False,
  )
  ```

  ![draw_mask](./resource/test_draw_mask.jpg)
