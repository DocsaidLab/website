---
sidebar_position: 3
---

# draw_polygon

> [draw_polygon(img: np.ndarray, polygon: Union[Polygon, np.ndarray], color: _Color = (0, 255, 0), thickness: _Thickness = 2, fillup=False, **kwargs) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/visualization/draw.py#L106)

- **說明**：在影像上繪製多邊形。

- **參數**
    - **img** (`np.ndarray`)：要繪製的影像，為 NumPy 陣列。
    - **polygon** (`Union[Polygon, np.ndarray]`)：要繪製的多邊形，可以是多邊形物件或 NumPy 陣列形式的 [[x1, y1], [x2, y2], ...]。
    - **color** (`_Color`)：要繪製的多邊形的顏色。預設為 (0, 255, 0)。
    - **thickness** (`_Thickness`)：要繪製的多邊形邊線的粗細。預設為 2。
    - **fillup** (`bool`)：是否填滿多邊形。預設為 False。
    - **kwargs**：其他參數。

- **傳回值**
    - **np.ndarray**：繪製了多邊形的影像。

- **範例**

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    polygon = D.Polygon([(20, 20), (100, 20), (80, 80), (20, 40)])
    polygon_img = D.draw_polygon(img, polygon, color=(0, 255, 0), thickness=2)
    ```

    ![draw_polygon](./resource/test_draw_polygon.jpg)
