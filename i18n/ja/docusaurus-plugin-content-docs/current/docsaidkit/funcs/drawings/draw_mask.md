---
sidebar_position: 6
---

# draw_mask

> [draw_mask(img: np.ndarray, mask: np.ndarray, colormap: int = cv2.COLORMAP_JET, weight: Tuple[float, float] = (0.5, 0.5), gamma: float = 0, min_max_normalize: bool = False) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/visualization/draw.py#L366)

- **説明**：画像上にマスクを描画します。

  - **img** (`np.ndarray`)：描画する画像。
  - **mask** (`np.ndarray`)：描画するマスク。
  - **colormap** (`int`)：マスクに使用するカラーマップ。デフォルトは `cv2.COLORMAP_JET`。
  - **weight** (`Tuple[float, float]`)：画像とマスクの重み。デフォルトは (0.5, 0.5)。
  - **gamma** (`float`)：マスクの Gamma 値。デフォルトは 0。
  - **min_max_normalize** (`bool`)：マスクを範囲 [0, 1] に正規化するかどうか。デフォルトは False。

- **戻り値**

  - **np.ndarray**：マスクが描画された画像。

- **例**

  ```python
  import docsaidkit as D
  import numpy as np

  img = D.imread('lena.png')
  polygon = D.Polygon([(20, 20), (100, 20), (80, 80), (20, 40)])
  mask = D.draw_polygon(np.zeros_like(img), polygon, fillup=True, color=255)
  mask_img = D.draw_mask(img, mask, colormap=cv2.COLORMAP_JET, weight=(0.5, 0.5), gamma=0, min_max_normalize=False)
  ```

  ![draw_mask](./resource/test_draw_mask.jpg)
