---
sidebar_position: 4
---

# Polygon

>[Polygon(array: _Polygon, normalized: bool = False)](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/structures/polygons.py#L64)

- **說明**：

    `Polygon` 是一個用來表示多邊形的類別。這個類別提供了許多方法，用來操作多邊形的座標，例如：正規化座標、反正規化座標、裁剪多邊形、移動多邊形、縮放多邊形、轉換多邊形為凸多邊形、轉換多邊形為最小外接矩形、轉換多邊形為邊界框等等。

- **參數**

    - **array** (`_Polygon`)：多邊形的座標。
    - **normalized** (`bool`)：是否為正規化多邊形的座標，是一個屬性標記。預設為 `False`。

- **屬性**

    - **normalized**：取得多邊形的正規化狀態。
    - **moments**：取得多邊形的矩。
    - **area**：取得多邊形的面積。
    - **arclength**：取得多邊形的周長。
    - **centroid**：取得多邊形的質心。
    - **boundingbox**：取得多邊形的邊界框。
    - **min_circle**：取得多邊形的最小外接圓。
    - **min_box**：取得多邊形的最小外接矩形。
    - **orientation**：取得多邊形的方向。
    - **min_box_wh**：取得多邊形的最小外接矩形的寬高。
    - **extent**：取得多邊形的占比。
    - **solidity**：取得多邊形的實心度。

- **方法**

    - **copy**()：複製多邊形物件。
    - **numpy**()：將多邊形物件轉換為 numpy 陣列。
    - **normalize**(`w: float, h: float`)：正規化多邊形的座標。
    - **denormalize**(`w: float, h: float`)：反正規化多邊形的座標。
    - **clip**(`xmin: int, ymin: int, xmax: int, ymax: int`)：裁剪多邊形。
    - **shift**(`shift_x: float, shift_y: float`)：移動多邊形。
    - **scale**(`distance: int, join_style: JOIN_STYLE = JOIN_STYLE.mitre`)：縮放多邊形。
    - **to_convexhull**()：將多邊形轉換為凸多邊形。
    - **to_min_boxpoints**()：將多邊形轉換為最小外接矩形的座標。
    - **to_box**(`box_mode: str = 'xyxy'`)：將多邊形轉換為邊界框。
    - **to_list**(`flatten: bool = False`)：將多邊形轉換為列表。
    - **is_empty**(`threshold: int = 3`)：判斷多邊形是否為空。

- **範例**

    ```python
    import docsaidkit as D

    polygon = D.Polygon([[10., 20.], [50, 20.], [50, 80.], [10., 80.]])
    print(polygon)
    # >>> Polygon([[10. 20.], [50. 20.], [50. 80.], [10. 80.]])

    polygon1 = polygon.normalize(100, 100)
    print(polygon1)
    # >>> Polygon([[0.1 0.2], [0.5 0.2], [0.5 0.8], [0.1 0.8]])

    polygon2 = polygon.denormalize(100, 100)
    print(polygon2)
    # >>> Polygon([[1000. 2000.], [5000. 2000.], [5000. 8000.], [1000. 8000.]])

    polygon3 = polygon.clip(20, 20, 60, 60)
    print(polygon3)
    # >>> Polygon([[20. 20.], [50. 20.], [50. 60.], [20. 60.]])

    polygon4 = polygon.shift(10, 10)
    print(polygon4)
    # >>> Polygon([[20. 30.], [60. 30.], [60. 90.], [20. 90.]])

    polygon5 = polygon.scale(10)
    print(polygon5)
    # >>> Polygon([[0. 10.], [60. 10.], [60. 90.], [0. 90.]])

    polygon6 = polygon.to_convexhull()
    print(polygon6)
    # >>> Polygon([[50. 80.], [10. 80.], [10. 20.], [50. 20.]])

    polygon7 = polygon.to_min_boxpoints()
    print(polygon7)
    # >>> Polygon([[10. 20.], [50. 20.], [50. 80.], [10. 80.]])

    polygon8 = polygon.to_box('xywh')
    print(polygon8)
    # >>> Box([10. 20. 40. 60.]), BoxMode.XYWH

    polygon9 = polygon.to_list()
    print(polygon9)
    # >>> [[10.0, 20.0], [50.0, 20.0], [50.0, 80.0], [10.0, 80.0]]

    polygon10 = polygon.is_empty()
    print(polygon10)
    # >>> False
    ```
