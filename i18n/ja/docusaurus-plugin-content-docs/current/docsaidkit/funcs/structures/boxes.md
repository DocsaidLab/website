---
sidebar_position: 3
---

# Boxes

> [Boxes(array: \_Boxes, box_mode: \_BoxMode = BoxMode.XYXY, normalized: bool = False)](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/structures/boxes.py#L361)

- **説明**：

  `Boxes` は複数の境界ボックスを表すクラスです。このクラスは、複数の境界ボックスの座標を操作するためのさまざまなメソッドを提供します。これには、座標系の変換、正規化、反正規化、裁切り、移動、拡大縮小などが含まれます。

- **パラメータ**

  - **array** (`_Boxes`)：複数の境界ボックス。
  - **box_mode** (`_BoxMode`)：境界ボックスを表現する方法を示す列挙クラス。デフォルトは `XYXY`。
  - **normalized** (`bool`)：境界ボックスの座標が正規化されているかどうか。デフォルトは `False`。

- **属性**

  - **box_mode**：境界ボックスの表現方法。
  - **normalized**：境界ボックスの正規化状態。
  - **width**：境界ボックスの幅。
  - **height**：境界ボックスの高さ。
  - **left_top**：境界ボックスの左上角の点。
  - **right_bottom**：境界ボックスの右下角の点。
  - **area**：境界ボックスの面積。
  - **aspect_ratio**：境界ボックスのアスペクト比（幅／高さ）。
  - **center**：境界ボックスの中心点。

- **メソッド**

  - **convert**(`to_mode: _BoxMode`)：境界ボックスのフォーマットを変換します。
  - **copy**()：境界ボックスをコピーします。
  - **numpy**()：境界ボックスを numpy 配列に変換します。
  - **square**()：境界ボックスを正方形に変換します。
  - **normalize**(`w: int, h: int`)：境界ボックスの座標を正規化します。
  - **denormalize**(`w: int, h: int`)：境界ボックスの座標を反正規化します。
  - **clip**(`xmin: int, ymin: int, xmax: int, ymax: int`)：境界ボックスを裁切ります。
  - **shift**(`shift_x: float, shift_y: float`)：境界ボックスを移動します。
  - **scale**(`dsize: Tuple[int, int] = None, fx: float = None, fy: float = None`)：境界ボックスをスケールします。
  - **to_list**()：境界ボックスをリストに変換します。
  - **to_polygons**()：境界ボックスを多角形（`docsaidkit.Polygons`）に変換します。

- **例**

  ```python
  import docsaidkit as D

  boxes = D.Boxes([[10, 20, 50, 80], [20, 30, 60, 90]])
  print(boxes)
  # >>> Boxes([[10. 20. 50. 80.], [20. 30. 60. 90.]]), BoxMode.XYXY

  boxes1 = boxes.convert(D.BoxMode.XYWH)
  print(boxes1)
  # >>> Boxes([[10. 20. 40. 60.], [20. 30. 40. 60.]]), BoxMode.XYWH

  boxes2 = boxes.normalize(100, 100)
  print(boxes2)
  # >>> Boxes([[0.1 0.2 0.5 0.8], [0.2 0.3 0.6 0.9]]), BoxMode.XYXY

  boxes3 = boxes.denormalize(100, 100)
  print(boxes3)
  # >>> Boxes([[1000. 2000. 5000. 8000.], [2000. 3000. 6000. 9000.]]), BoxMode.XYXY

  boxes4 = boxes.clip(0, 0, 50, 50)
  print(boxes4)
  # >>> Boxes([[10. 20. 50. 50.], [20. 30. 50. 50.]]), BoxMode.XYXY

  boxes5 = boxes.shift(10, 10)
  print(boxes5)
  # >>> Boxes([[20. 30. 60. 90.], [30. 40. 70. 100.]]), BoxMode.XYXY

  boxes6 = boxes.scale(dsize=(10, 10))
  print(boxes6)
  # >>> Boxes([[5. 15. 55. 85.], [15. 25. 65. 95.]]), BoxMode.XYXY

  boxes7 = boxes.square()
  print(boxes7)
  # >>> Boxes([[0. 20. 60. 80.], [10. 30. 70. 90.]]), BoxMode.XYXY

  boxes8 = boxes.to_list()
  print(boxes8)
  # >>> [[10.0, 20.0, 50.0, 80.0], [20.0, 30.0, 60.0, 90.0]]

  polys = boxes.to_polygons() # Box.to_polygon()とは異なります
  print(polys)
  # >>> Polygons([
  #       Polygon([
  #           [10. 20.]
  #           [50. 20.]
  #           [50. 80.]
  #           [10. 80.]
  #       ]),
  #       Polygon([
  #           [20. 30.]
  #           [60. 30.]
  #           [60. 90.]
  #           [20. 90.]
  #       ])
  #    ])
  ```
