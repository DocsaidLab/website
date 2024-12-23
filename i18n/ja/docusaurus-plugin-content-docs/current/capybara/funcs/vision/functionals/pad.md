# pad

> [pad(img: np.ndarray, pad_size: Union[int, Tuple[int, int], Tuple[int, int, int, int]], fill_value: Optional[Union[int, Tuple[int, int, int]]] = 0, pad_mode: Union[str, int, BORDER] = BORDER.CONSTANT) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/functionals.py#L194)

- **説明**：入力画像に対してパディング処理を行います。

- **引数**

  - **img** (`np.ndarray`)：パディング処理を行う入力画像。
  - **pad_size** (`Union[int, Tuple[int, int], Tuple[int, int, int, int]]`)：パディングのサイズ。整数を指定するとすべての辺に同じパディング量が適用され、タプル `(pad_top, pad_bottom, pad_left, pad_right)` で各辺に異なるパディング量を指定できます。もしくは `(pad_height, pad_width)` として高さと幅に同じパディング量を指定することもできます。
  - **fill_value** (`Optional[Union[int, Tuple[int, int, int]]]`)：パディングに使用する値。カラー画像（3 チャネル）の場合、`fill_value` は整数または `(R, G, B)` のタプルで指定できます。グレースケール画像（1 チャネル）の場合、`fill_value` は整数です。デフォルトは 0。
  - **pad_mode** (`Union[str, int, BORDER]`)：パディングモード。選択肢は以下の通りです：
    - `BORDER.CONSTANT`：定数値（`fill_value`）を使用してパディング。
    - `BORDER.REPLICATE`：端のピクセルをコピーしてパディング。
    - `BORDER.REFLECT`：端を反射させてパディング。
    - `BORDER.REFLECT101`：端を反射させ、人工的な痕跡を避けるために微調整してパディング。
      デフォルトは `BORDER.CONSTANT`。

- **戻り値**

  - **np.ndarray**：パディング処理後の画像。

- **例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  pad_img = cb.pad(img, pad_size=20, fill_value=(255, 0, 0))

  # パディング後の画像を元のサイズにリサイズして視覚化
  pad_img = cb.imresize(pad_img, [img.shape[0], img.shape[1]])
  ```

  ![pad](./resource/test_pad.jpg)
