---
sidebar_position: 2
---

# Box

> [Box(array: \_Box, box_mode: \_BoxMode = BoxMode.XYXY, normalized: bool = False) -> None](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/structures/boxes.py#L106)

- **説明**：

  `Box` は境界ボックスを表すクラスです。このクラスは、座標の変換、正規化、反正規化、裁切り、移動、拡大縮小など、さまざまな方法を提供して、境界ボックスを操作できます。

- **パラメータ**

  - **array** (`_Box`)：境界ボックス。
  - **box_mode** (`_BoxMode`)：境界ボックスを表現する方法を示す列挙クラス。デフォルトは `XYXY`。
  - **normalized** (`bool`)：境界ボックスの座標が正規化されているかどうかを示すフラグ。デフォルトは `False`。

- **属性**

  - **box_mode**：境界ボックスの表現方法。
  - **normalized**：境界ボックスの正規化状態。
  - **width**：境界ボックスの幅。
  - **height**：境界ボックスの高さ。
  - **left_top**：境界ボックスの左上角の点。
  - **right_bottom**：境界ボックスの右下角の点。
  - **left_bottom**：境界ボックスの左下角の点。
  - **right_top**：境界ボックスの右上角の点。
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
  - **to_polygon**()：境界ボックスを多角形（`docsaidkit.Polygon`）に変換します。

- **例**

  ```python
  import docsaidkit as D

  box = D.Box([10, 20, 50, 80])
  print(box)
  # >>> Box([10. 20. 50. 80.]), BoxMode.XYXY

  box1 = box.convert(D.BoxMode.XYWH)
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
