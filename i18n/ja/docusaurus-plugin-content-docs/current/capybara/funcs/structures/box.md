---
sidebar_position: 2
---

# Box

> [Box(array: \_Box, box_mode: \_BoxMode = BoxMode.XYXY, normalized: bool = False) -> None](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/structures/boxes.py#L101)

- **説明**：

  `Box` は、境界ボックスを表すクラスです。このクラスは、座標系の変換、座標の正規化、非正規化、境界ボックスのクリッピング、移動、スケーリングなど、境界ボックスの座標を操作するための多くのメソッドを提供します。

- **パラメータ**

  - **array** (`_Box`)：境界ボックス。
  - **box_mode** (`_BoxMode`)：境界ボックスの異なる表現方法を示す列挙型。デフォルトは `XYXY`。
  - **normalized** (`bool`)：境界ボックスの座標が正規化されているかどうかを示すフラグ。デフォルトは `False`。

- **属性**

  - **box_mode**：境界ボックスの表現方法を取得。
  - **normalized**：境界ボックスの正規化状態を取得。
  - **width**：境界ボックスの幅を取得。
  - **height**：境界ボックスの高さを取得。
  - **left_top**：境界ボックスの左上角の座標を取得。
  - **right_bottom**：境界ボックスの右下角の座標を取得。
  - **left_bottom**：境界ボックスの左下角の座標を取得。
  - **right_top**：境界ボックスの右上角の座標を取得。
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
  - **to_polygon**()：境界ボックスを多角形（capybara.Polygon）に変換。

- **例**

  ```python
  import capybara as cb

  box = cb.Box([10, 20, 50, 80])
  print(box)
  # >>> Box([10. 20. 50. 80.]), BoxMode.XYXY

  box1 = box.convert(cb.BoxMode.XYWH)
  print(box1)
  # >>> Box([10. 20. 40. 60.]), BoxMode.XYWH

  box2 = box.normalize(100, 100)
  print(box2)
  # >>> Box([0.1 0.2 0.5 0.8]), BoxMode.XYXY

  box3 = box.denormalize(100, 100)
  print(box3)
  # >>> Box([1000. 2000. 5000. 8000.]), BoxMode.XYXY

  box4 = box.clip(0, 0, 50, 50)
  print(box4)
  # >>> Box([10. 20. 50. 50.]), BoxMode.XYXY

  box5 = box.shift(10, 10)
  print(box5)
  # >>> Box([20. 30. 60. 90.]), BoxMode.XYXY

  box6 = box.scale(dsize=(10, 10))
  print(box6)
  # >>> Box([5. 15. 55. 85.]), BoxMode.XYXY

  box7 = box.scale(fx=1.1, fy=1.1)
  print(box7)
  # >>> Box([8. 17. 52. 83.]), BoxMode.XYXY

  box8 = box.square()
  print(box8)
  # >>> Box([10. 30. 50. 70.]), BoxMode.XYXY

  box9 = box.to_list()
  print(box9)
  # >>> [10.0, 20.0, 50.0, 80.0]

  poly = box.to_polygon()
  print(poly)
  # >>> Polygon([[10. 20.], [50. 20.], [50. 80.], [10. 80.]])
  ```
