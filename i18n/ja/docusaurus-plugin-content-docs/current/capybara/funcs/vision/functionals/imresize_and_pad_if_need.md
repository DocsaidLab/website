# imresize_and_pad_if_need

> [imresize_and_pad_if_need(img: np.ndarray, max_h: int, max_w: int, interpolation: str | int | INTER = INTER.BILINEAR, pad_value: int | tuple[int, int, int] | None = 0, pad_mode: str | int | BORDER = BORDER.CONSTANT, return_scale: bool = False) -> np.ndarray | tuple[np.ndarray, float]](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/functionals.py)

- **説明**：画像を `(max_h, max_w)` の枠内に収まるようにリサイズし、必要に応じて固定サイズにパディングします。

- **パラメータ**

  - **img** (`np.ndarray`)：入力画像。
  - **max_h** (`int`)：出力の最大高さ（パディング後の固定高さ）。
  - **max_w** (`int`)：出力の最大幅（パディング後の固定幅）。
  - **interpolation** (`str | int | INTER`)：リサイズ時の補間方法。デフォルトは `INTER.BILINEAR`。
  - **pad_value** (`int | tuple[int, int, int] | None`)：パディング値。3-channel の場合は int または tuple（OpenCV 慣例：BGR）が使用できます。デフォルトは 0。
  - **pad_mode** (`str | int | BORDER`)：パディングモード。デフォルトは `BORDER.CONSTANT`。
  - **return_scale** (`bool`)：リサイズ倍率を返すかどうか。デフォルトは `False`。

- **戻り値**

  - `return_scale=False` のとき：`np.ndarray`。
  - `return_scale=True` のとき：`(np.ndarray, float)`。float は `scale = min(max_h/raw_h, max_w/raw_w)`。

- **備考**

  - パディングは bottom/right のみに適用されます（top=0, left=0）。
  - `max_h/max_w` が元画像より小さい場合は縮小し、大きい場合は拡大します。

- **例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')

  out, scale = cb.imresize_and_pad_if_need(
      img,
      max_h=640,
      max_w=640,
      pad_value=0,
      return_scale=True,
  )
  print(out.shape, scale)
  ```

