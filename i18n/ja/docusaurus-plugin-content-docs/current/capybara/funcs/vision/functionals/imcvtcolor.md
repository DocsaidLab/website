# imcvtcolor

> [imcvtcolor(img: np.ndarray, cvt_mode: Union[int, str]) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/functionals.py#L96)

- **説明**：入力画像に対して色空間変換を行います。

- **引数**

  - **img** (`np.ndarray`)：変換する入力画像。
  - **cvt_mode** (`Union[int, str]`)：色変換モード。変換コードを表す整数定数、または OpenCV 色変換名を表す文字列で指定できます。例えば、`BGR2GRAY` は BGR 画像をグレースケールに変換するために使用されます。利用可能なパラメータは [**OpenCV COLOR**](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html) を参照してください。

- **戻り値**

  - **np.ndarray**：変換後の画像。

- **例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  ycrcb_img = cb.imcvtcolor(img, 'BGR2YCrCb')
  ```

  ![imcvtcolor_ycrcb](./resource/test_imcvtcolor_ycrcb.jpg)

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  gray_img = cb.imcvtcolor(img, 'BGR2GRAY')
  ```

  ![imcvtcolor_gray](./resource/test_imcvtcolor_gray.jpg)
