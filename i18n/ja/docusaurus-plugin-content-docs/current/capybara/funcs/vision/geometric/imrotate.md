# imrotate

> [imrotate(img: np.ndarray, angle: float, scale: float = 1, interpolation: Union[str, int, INTER] = INTER.BILINEAR, bordertype: Union[str, int, BORDER] = BORDER.CONSTANT, bordervalue: Union[int, Tuple[int, int, int]] = None, expand: bool = True, center: Tuple[int, int] = None) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/geometric.py#L80)

- **説明**：入力画像を回転させる処理を行います。

- **パラメータ**

  - **img** (`np.ndarray`)：回転させる入力画像。
  - **angle** (`float`)：回転角度。度単位で、反時計回りの方向。
  - **scale** (`float`)：スケール比。デフォルトは 1。
  - **interpolation** (`Union[str, int, INTER]`)：補間方法。選べるオプションは、INTER.NEAREST、INTER.LINEAR、INTER.CUBIC、INTER.LANCZOS4 です。デフォルトは INTER.LINEAR です。
  - **bordertype** (`Union[str, int, BORDER]`)：境界タイプ。選べるオプションは、BORDER.CONSTANT、BORDER.REPLICATE、BORDER.REFLECT、BORDER.REFLECT_101 です。デフォルトは BORDER.CONSTANT です。
  - **bordervalue** (`Union[int, Tuple[int, int, int]]`)：境界の填充値。bordertype が BORDER.CONSTANT の場合にのみ有効です。デフォルトは None です。
  - **expand** (`bool`)：回転後の画像全体を収容できるように出力画像を拡張するかどうか。True の場合、回転後の画像を収容するのに十分な大きさになるように拡張されます。False または省略した場合、入力画像と同じサイズの出力画像になります。なお、expand フラグは中心を基準に回転し、平行移動がないことを前提としています。デフォルトは False です。
  - **center** (`Tuple[int, int]`)：回転の中心。デフォルトは画像の中心です。

- **戻り値**

  - **np.ndarray**：回転後の画像。

- **使用例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  rotate_img = cb.imrotate(img, 45, bordertype=cb.BORDER.CONSTANT, expand=True)

  # 回転した画像を元のサイズにリサイズして可視化
  rotate_img = cb.imresize(rotate_img, [img.shape[0], img.shape[1]])
  ```

  ![imrotate](./resource/test_imrotate.jpg)
