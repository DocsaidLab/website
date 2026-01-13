# centercrop

> [centercrop(img: np.ndarray) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/functionals.py)

- **說明**：對輸入影像進行中心裁剪處理。

- **參數**

  - **img** (`np.ndarray`)：要進行中心裁剪處理的輸入影像。

- **傳回值**

  - **np.ndarray**：裁剪後的影像。

- **範例**

  ```python
  from capybara import imread, imresize
  from capybara.vision.functionals import centercrop

  img = imread('lena.png')
  img = imresize(img, [128, 256])
  crop_img = centercrop(img)
  ```

  綠色框表示中心裁剪的區域。

  ![centercrop](./resource/test_centercrop.jpg)
