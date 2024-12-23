# imcropbox

> [imcropbox(img: np.ndarray, box: Union[Box, np.ndarray], use_pad: bool = False) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/functionals.py#L257)

- **説明**：提供されたボックスを使用して入力画像をクロップします。

- **引数**

  - **img** (`np.ndarray`)：クロップする入力画像。
  - **box** (`Union[Box, np.ndarray]`)：クロップボックス。入力は Capybara 独自の `Box` オブジェクト（`(x1, y1, x2, y2)` 座標で定義）または同じ形式の NumPy 配列が可能です。
  - **use_pad** (`bool`)：境界外の領域をゼロパディングで処理するかどうか。`True` に設定すると、外部領域はゼロで埋められます。デフォルトは `False`。

- **戻り値**

  - **np.ndarray**：クロップ後の画像。

- **例**

  ```python
  import capybara as cb

  # 自定義 Box オブジェクトを使用
  img = cb.imread('lena.png')
  box = cb.Box([50, 50, 200, 200], box_mode='xyxy')
  cropped_img = cb.imcropbox(img, box, use_pad=True)

  # クロップされた画像を元のサイズにリサイズして視覚化
  cropped_img = cb.imresize(cropped_img, [img.shape[0], img.shape[1]])
  ```

  ![imcropbox_box](./resource/test_imcropbox.jpg)
