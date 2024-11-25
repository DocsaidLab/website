---
sidebar_position: 7
---

# imcropboxes

> [imcropboxes(img: np.ndarray, boxes: Union[Box, np.ndarray], use_pad: bool = False) -> List[np.ndarray]](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/functionals.py#L325)

- **説明**：提供された複数のボックスを使用して入力画像をクロップします。

- 引数

  - **img** (`np.ndarray`)：クロップする入力画像。
  - **boxes** (`Union[Boxes, np.ndarray]`)：クロップボックス。入力は DocsaidKit のカスタム Boxes オブジェクトで、`List[Box]`として定義されます。また、同じ形式の NumPy 配列も使用可能です。
  - **use_pad** (`bool`)：境界外の領域をパディングで処理するかどうか。True に設定すると、外部領域はゼロでパディングされます。デフォルトは False。

- **返り値**

  - **List[np.ndarray]**：クロップ後の画像リスト。

- **例**

  ```python
  import docsaidkit as D

  # カスタムBoxオブジェクトを使用
  img = D.imread('lena.png')
  box1 = D.Box([50, 50, 200, 200], box_mode='xyxy')
  box2 = D.Box([50, 50, 100, 100], box_mode='xyxy')
  boxes = D.Boxes([box1, box2])
  cropped_imgs = D.imcropboxes(img, boxes, use_pad=True)

  # クロップした画像を元のサイズにリサイズして可視化
  cropped_img = D.imresize(cropped_img, [img.shape[0], img.shape[1]])
  ```

  ![imcropbox_boxes](./resource/test_imcropboxes.jpg)
