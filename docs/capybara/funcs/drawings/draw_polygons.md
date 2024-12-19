---
sidebar_position: 4
---

# draw_polygons

> [draw_polygons(img: np.ndarray, polygons: Polygons, color: _Colors = (0, 255, 0), thickness: _Thicknesses = 2, fillup=False, **kwargs) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/visualization/draw.py#L156)

- **說明**：在影像上繪製多個多邊形。

- **參數**
    - **img** (`np.ndarray`)：要繪製的影像，為 NumPy 陣列。
    - **polygons** (`List[Union[Polygon, np.ndarray]]`)：要繪製的多邊形，可以是多邊形物件的列表或 NumPy 陣列形式的 [[x1, y1], [x2, y2], ...]。
    - **color** (`_Colors`)：要繪製的多邊形的顏色。可以是單一顏色或顏色列表。預設為 (0, 255, 0)。
    - **thickness** (`_Thicknesses`)：要繪製的多邊形邊線的粗細。可以是單一粗細或粗細列表。預設為 2。
    - **fillup** (`bool`)：是否填滿多邊形。預設為 False。
    - **kwargs**：其他參數。

- **傳回值**
    - **np.ndarray**：繪製了多邊形的影像。

- **範例**

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    polygons = [
        D.Polygon([(20, 20), (100, 20), (80, 80), (20, 40)]),
        D.Polygon([(100, 100), (20, 100), (40, 40), (100, 80)])
    ]
    polygons_img = D.draw_polygons(img, polygons, color=[(0, 255, 0), (255, 0, 0)], thickness=2)
    ```

    ![draw_polygons](./resource/test_draw_polygons.jpg)
