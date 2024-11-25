---
sidebar_position: 4
---

# Polygon

> [Polygon(array: \_Polygon, normalized: bool = False)](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/structures/polygons.py#L64)

- **説明**：

  `Polygon` は多角形を表すクラスです。このクラスは、多角形の座標を操作するためのさまざまなメソッドを提供します。これには、座標の正規化、反正規化、裁切り、移動、スケーリング、凸包への変換、最小外接矩形への変換、境界ボックスへの変換などが含まれます。

- **パラメータ**

  - **array** (`_Polygon`)：多角形の座標。
  - **normalized** (`bool`)：座標が正規化されているかどうかを示すフラグ。デフォルトは `False`。

- **属性**

  - **normalized**：多角形が正規化されているかどうか。
  - **moments**：多角形のモーメント。
  - **area**：多角形の面積。
  - **arclength**：多角形の周長。
  - **centroid**：多角形の重心。
  - **boundingbox**：多角形の境界ボックス。
  - **min_circle**：多角形の最小外接円。
  - **min_box**：多角形の最小外接矩形。
  - **orientation**：多角形の向き。
  - **min_box_wh**：最小外接矩形の幅と高さ。
  - **extent**：多角形の占有率。
  - **solidity**：多角形の実質的な面積。

- **メソッド**

  - **copy**()：多角形オブジェクトをコピーします。
  - **numpy**()：多角形を numpy 配列に変換します。
  - **normalize**(`w: float, h: float`)：多角形の座標を正規化します。
  - **denormalize**(`w: float, h: float`)：多角形の座標を反正規化します。
  - **clip**(`xmin: int, ymin: int, xmax: int, ymax: int`)：多角形を裁切ります。
  - **shift**(`shift_x: float, shift_y: float`)：多角形を移動させます。
  - **scale**(`distance: int, join_style: JOIN_STYLE = JOIN_STYLE.mitre`)：多角形をスケールします。
  - **to_convexhull**()：多角形を凸包に変換します。
  - **to_min_boxpoints**()：多角形を最小外接矩形の座標に変換します。
  - **to_box**(`box_mode: str = 'xyxy'`)：多角形を境界ボックスに変換します。
  - **to_list**(`flatten: bool = False`)：多角形をリストに変換します。
  - **is_empty**(`threshold: int = 3`)：多角形が空であるかどうかを判断します。

- **例**

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
