---
sidebar_position: 3
---

# Boxes

> [Boxes(array: \_Boxes, box_mode: \_BoxMode = BoxMode.XYXY, normalized: bool = False)](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/structures/boxes.py#L362)

- **説明**：

  `Boxes` は、複数の境界ボックスを表すクラスです。このクラスは、複数の境界ボックスの座標を操作するための多くのメソッドを提供します。例えば、座標系の変換、座標の正規化、非正規化、境界ボックスのクリッピング、移動、スケーリングなどです。

- **パラメータ**

  - **array** (`_Boxes`)：複数の境界ボックス。
  - **box_mode** (`_BoxMode`)：境界ボックスの異なる表現方法を示す列挙型。デフォルトは `XYXY`。
  - **normalized** (`bool`)：境界ボックスの座標が正規化されているかどうかを示すフラグ。デフォルトは `False`。

- **属性**

  - **box_mode**：境界ボックスの表現方法を取得。
  - **normalized**：境界ボックスの正規化状態を取得。
  - **width**：境界ボックスの幅を取得。
  - **height**：境界ボックスの高さを取得。
  - **left_top**：境界ボックスの左上角の座標を取得。
  - **right_bottom**：境界ボックスの右下角の座標を取得。
  - **area**：境界ボックスの面積を取得。
  - **aspect_ratio**：境界ボックスのアスペクト比を計算。
  - **center**：境界ボックスの中心座標を計算。

- **メソッド**

  - **convert**(`to_mode: _BoxMode`)：境界ボックスの形式を変換。
  - **copy**()：境界ボックスオブジェクトのコピーを作成。
  - **numpy**()：境界ボックスオブジェクトを numpy 配列に変換。
  - **square**()：境界ボックスを正方形の境界ボックスに変換。
  - **normalize**(`w: int, h: int`)：境界ボックスの座標を正規化。
  - **denormalize**(`w: int, h: int`)：境界ボックスの座標を非正規化。
  - **clip**(`xmin: int, ymin: int, xmax: int, ymax: int`)：境界ボックスをクリップ。
  - **shift**(`shift_x: float, shift_y: float`)：境界ボックスを移動。
  - **scale**(`dsize: Tuple[int, int] = None, fx: float = None, fy: float = None`)：境界ボックスをスケーリング。
  - **to_list**()：境界ボックスをリストに変換。
  - **to_polygons**()：境界ボックスを多角形（capybara.Polygons）に変換。

- **例**

  ```python
  import capybara as cb

  boxes = cb.Boxes([[10, 20, 50, 80], [20, 30, 60, 90]])
  print(boxes)
  # >>> Boxes([[10. 20. 50. 80.], [20. 30. 60. 90.]]), BoxMode.XYXY

  boxes1 = boxes.convert(cb.BoxMode.XYWH)
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

  polys = boxes.to_polygons() # Notice: It's different from Box.to_polygon()
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
