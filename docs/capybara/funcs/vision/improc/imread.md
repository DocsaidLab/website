# imread

> [imread(path: str | Path, color_base: str = 'BGR', verbose: bool = False) -> np.ndarray | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **說明**：讀取圖片並回傳 BGR numpy 影像（必要時進行色彩空間轉換）。

  - 若副檔名為 `.heic`：使用 `pillow-heif` 讀取（直接輸出 BGR）。
  - 其他格式：先嘗試 `jpgread`（可處理 JPEG 與 EXIF 方向），失敗則退回 `cv2.imread`。

- **參數**

  - **path** (`Union[str, Path]`)：要讀取的圖片路徑。
  - **color_base** (`str`)：圖片的色彩空間。如果不是 `BGR`，將使用 `imcvtcolor` 函數進行轉換。預設為 `BGR`。
  - **verbose** (`bool`)：如果設置為 True，當讀取的圖片為 None 時，將發出警告。預設為 False。

- **傳回值**

  - **np.ndarray**：成功時返回圖片的 numpy ndarray，否則返回 None。

- **例外**

  - **FileExistsError**：`path` 不存在時。

- **範例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  ```
