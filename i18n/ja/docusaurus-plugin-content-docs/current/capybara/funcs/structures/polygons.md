---
sidebar_position: 5
---

# Polygons

> [Polygons(array: \_Polygons, normalized: bool = False)](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/structures/polygons.py#L336)

- **説明**：

  `Polygons` は、複数の多角形を表すクラスです。このクラスは、複数の多角形の座標を操作するための多くのメソッドを提供します。例えば、座標の正規化、非正規化、多角形のクリッピング、移動、スケーリング、凸多角形への変換、最小外接矩形への変換、境界ボックスへの変換などです。

- **パラメータ**

  - **array** (`_Polygons`)：複数の多角形の座標。
  - **normalized** (`bool`)：多角形の座標が正規化されているかどうかを示すフラグ。デフォルトは `False`。

- **属性**

  - **normalized**：多角形の正規化状態を取得。
  - **moments**：多角形のモーメントを取得。
  - **area**：多角形の面積を取得。
  - **arclength**：多角形の周囲長を取得。
  - **centroid**：多角形の質量中心を取得。
  - **boundingbox**：多角形の境界ボックスを取得。
  - **min_circle**：多角形の最小外接円を取得。
  - **min_box**：多角形の最小外接矩形を取得。
  - **orientation**：多角形の向きを取得。
  - **min_box_wh**：多角形の最小外接矩形の幅と高さを取得。
  - **extent**：多角形の占有率を取得。
  - **solidity**：多角形の充実度（solidity）を取得。

- **メソッド**

  - **copy**()：多角形オブジェクトをコピー。
  - **numpy**()：多角形オブジェクトを numpy 配列に変換。
  - **normalize**(`w: float, h: float`)：多角形の座標を正規化。
  - **denormalize**(`w: float, h: float`)：多角形の座標を非正規化。
  - **clip**(`xmin: int, ymin: int, xmax: int, ymax: int`)：多角形をクリッピング。
  - **shift**(`shift_x: float, shift_y: float`)：多角形を移動。
  - **scale**(`distance: int, join_style: JOIN_STYLE = JOIN_STYLE.mitre`)：多角形をスケーリング。
  - **to_convexhull**()：多角形を凸多角形に変換。
  - **to_min_boxpoints**()：多角形を最小外接矩形の座標に変換。
  - **to_box**(`box_mode: str = 'xyxy'`)：多角形を境界ボックスに変換。
  - **to_list**(`flatten: bool = False`)：多角形をリストに変換。
  - **is_empty**(`threshold: int = 3`)：多角形が空かどうかを判定。

- **クラスメソッド**

  - **from_image**(`image: np.ndarray, mode: int = cv2.RETR_EXTERNAL, method: int = cv2.CHAIN_APPROX_SIMPLE`)：画像から多角形の座標を取得。
  - **cat**(`polygons_list: List["Polygons"]`)：複数の多角形リストを連結して 1 つの多角形リストにする。

- **例**

  ```python
  import capybara as cb

  polygons = cb.Polygons([
      [[10., 20.], [50, 20.], [50, 80.], [10., 80.]],
      [[60., 20.], [100, 20.], [100, 80.], [60., 80.]]
  ])
  print(polygons)
  # >>> Polygons(
  #   [
  #       Polygon([
  #           [10. 20.]
  #           [50. 20.]
  #           [50. 80.]
  #           [10. 80.]
  #       ]),
  #       Polygon([
  #           [60. 20.]
  #           [100. 20.]
  #           [100. 80.]
  #           [60. 80.]
  #       ])
  #   ]
  # )

  polygon1 = polygons.normalize(100, 100)
  print(polygon1)
  # >>> Polygons(
  #   [
  #       Polygon([
  #           [0.1 0.2]
  #           [0.5 0.2]
  #           [0.5 0.8]
  #           [0.1 0.8]
  #       ]),
  #       Polygon([
  #           [0.6 0.2]
  #           [1.  0.2]
  #           [1.  0.8]
  #           [0.6 0.8]
  #       ])
  #   ]
  # )

  polygon2 = polygons.denormalize(100, 100)
  print(polygon2)
  # >>> Polygons(
  #   [
  #       Polygon([
  #           [1000. 2000.]
  #           [5000. 2000.]
  #           [5000. 8000.]
  #           [1000. 8000.]
  #       ]),
  #       Polygon([
  #           [6000. 2000.]
  #           [10000. 2000.]
  #           [10000. 8000.]
  #           [6000. 8000.]
  #       ])
  #   ]
  # )

  polygon3 = polygons.clip(20, 20, 60, 60)
  print(polygon3)
  # >>> Polygons(
  #   [
  #       Polygon([
  #           [20. 20.]
  #           [50. 20.]
  #           [50. 60.]
  #           [20. 60.]
  #       ]),
  #       Polygon([
  #           [60. 20.]
  #           [60. 20.]
  #           [60. 60.]
  #           [60. 60.]
  #       ])
  #   ]
  # )

  polygon4 = polygons.shift(10, 10)
  print(polygon4)
  # >>> Polygons(
  #   [
  #       Polygon([
  #           [20. 30.]
  #           [60. 30.]
  #           [60. 90.]
  #           [20. 90.]
  #       ]),
  #       Polygon([
  #           [ 70.  30.]
  #           [110.  30.]
  #           [110.  90.]
  #           [ 70.  90.]
  #       ])
  #   ]
  # )


  polygon5 = polygons.scale(10)
  print(polygon5)
  # >>> Polygons(
  #   [
  #       Polygon([
  #           [ 0. 10.]
  #           [60. 10.]
  #           [60. 90.]
  #           [ 0. 90.]
  #       ]),
  #       Polygon([
  #           [ 50.  10.]
  #           [110.  10.]
  #           [110.  90.]
  #           [ 50.  90.]
  #       ])
  #   ]
  # )

  polygon6 = polygons.to_convexhull()
  print(polygon6)
  # >>> Polygons(
  #   [
  #       Polygon([
  #           [50. 80.],
  #           [10. 80.],
  #           [10. 20.],
  #           [50. 20.]
  #       ]),
  #       Polygon([
  #           [100.  80.],
  #           [ 60.  80.],
  #           [ 60.  20.],
  #           [100.  20.]
  #       ])
  #   ]
  # )

  polygon7 = polygons.to_min_boxpoints()
  print(polygon7)
  # >>> Polygons(
  #   [
  #       Polygon([
  #           [10. 20.]
  #           [50. 20.]
  #           [50. 80.]
  #           [10. 80.]
  #       ]),
  #       Polygon([
  #           [ 60.  20.]
  #           [100.  20.]
  #           [100.  80.]
  #           [ 60.  80.]
  #       ])
  #   ]
  # )

  polygon8 = polygons.to_boxes() # notice that the method name is different from Polygon
  print(polygon8)
  # >>> Boxes([[ 10.  20.  50.  80.], [ 60.  20. 100.  80.]]), BoxMode.XYXY

  polygon9 = polygons.to_list()
  print(polygon9)
  # >>> [
  #   [
  #       [10.0, 20.0],
  #       [50.0, 20.0],
  #       [50.0, 80.0],
  #       [10.0, 80.0]
  #   ],
  #   [
  #       [60.0, 20.0],
  #       [100.0, 20.0],
  #       [100.0, 80.0],
  #       [60.0, 80.0]
  #   ]
  # ]

  polygon10 = polygons.is_empty()
  print(polygon10)
  # >>> [False, False]
  ```
