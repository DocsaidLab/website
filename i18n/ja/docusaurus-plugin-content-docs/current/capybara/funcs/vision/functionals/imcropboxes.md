# imcropboxes

> [imcropboxes(img: np.ndarray, boxes: Union[Box, np.ndarray], use_pad: bool = False) -> List[np.ndarray]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/functionals.py#L325)

- **説明**：提供された複数のボックスを使用して入力画像をクロップします。

- **引数**

  - **img** (`np.ndarray`)：クロップする入力画像。
  - **boxes** (`Union[Boxes, np.ndarray]`)：クロップボックス。入力は Capybara 独自の `Boxes` オブジェクト（`List[Box]` で定義）または同じ形式の NumPy 配列が可能です。
  - **use_pad** (`bool`)：境界外の領域をゼロパディングで処理するかどうか。`True` に設定すると、外部領域はゼロで埋められます。デフォルトは `False`。

- **戻り値**

  - **List[np.ndarray]**：クロップ後の画像リスト。

- **例**

  ```python
  import capybara as cb

  # 自定義 Box オブジェクトを使用
  img = cb.imread('lena.png')
  box1 = cb.Box([50, 50, 200, 200], box_mode='xyxy')
  box2 = cb.Box([50, 50, 100, 100], box_mode='xyxy')
  boxes = cb.Boxes([box1, box2])
  cropped_imgs = cb.imcropboxes(img, boxes, use_pad=True)

  # クロップされた画像を元のサイズにリサイズして視覚化
  cropped_img = cb.imresize(cropped_img, [img.shape[0], img.shape[1]])
  ```

  ![imcropbox_boxes](./resource/test_imcropboxes.jpg)
