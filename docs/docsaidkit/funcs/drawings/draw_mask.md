---
sidebar_position: 6
---

# draw_mask

> [draw_mask(img: np.ndarray, mask: np.ndarray, colormap: int = cv2.COLORMAP_JET, weight: Tuple[float, float] = (0.5, 0.5), gamma: float = 0, min_max_normalize: bool = False) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/visualization/draw.py#L366)

- **說明**：在影像上繪製遮罩。

    - **img** (`np.ndarray`)：要繪製的影像。
    - **mask** (`np.ndarray`)：要繪製的遮罩。
    - **colormap** (`int`)：用於遮罩的色彩地圖。預設為 `cv2.COLORMAP_JET`。
    - **weight** (`Tuple[float, float]`)：影像和遮罩的權重。預設為 (0.5, 0.5)。
    - **gamma** (`float`)：遮罩的 Gamma 值。預設為 0。
    - **min_max_normalize** (`bool`)：是否將遮罩正規化為範圍 [0, 1]。預設為 False。

- **傳回值**
    - **np.ndarray**：繪製了遮罩的影像。

- **範例**

    ```python
    import docsaidkit as D
    import numpy as np

    img = D.imread('lena.png')
    polygon = D.Polygon([(20, 20), (100, 20), (80, 80), (20, 40)])
    mask = D.draw_polygon(np.zeros_like(img), polygon, fillup=True, color=255)
    mask_img = D.draw_mask(img, mask, colormap=cv2.COLORMAP_JET, weight=(0.5, 0.5), gamma=0, min_max_normalize=False)
    ```

    ![draw_mask](./resource/test_draw_mask.jpg)
