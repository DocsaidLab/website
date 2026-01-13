# JPG Process

> [get_orientation_code(stream: str | Path | bytes) -> ROTATE | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

> [jpgencode(img: np.ndarray, quality: int = 90) -> bytes | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

> [jpgdecode(byte\_: bytes) -> np.ndarray | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

> [jpgread(img_file: str | Path) -> np.ndarray | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

這一系列函數提供了對 JPG 圖像的編碼、解碼和讀取支持，以及從 EXIF 數據自動調整圖像方向的功能。

## 說明

- **get_orientation_code**：從圖像的 EXIF 數據中提取方向信息，並將其轉換為適合於圖像旋轉的代碼。這一步是在 `jpgdecode` 和 `jpgread` 函數中自動完成的，以確保讀取的圖像顯示方向正確。
- **jpgencode**：將 NumPy 圖像數組編碼為 JPG 格式的字節串；失敗時回傳 `None`。
- **jpgdecode**：將 JPG 格式的字節串解碼為 NumPy 圖像數組，並根據 EXIF 數據調整圖像方向；失敗時回傳 `None`。
- **jpgread**：從文件中讀取 JPG 圖像，解碼為 NumPy 圖像數組，並根據 EXIF 數據調整圖像方向。

## 依賴

- 此模組使用 `turbojpeg`（PyTurboJPEG）進行 JPEG 編解碼。

## 參數

### jpgencode

- **img** (`np.ndarray`)：要編碼的圖像數組。
- **quality** (`int`)：編碼品質，範圍為 1 至 100。預設為 90。

### jpgdecode

- **byte\_** (`bytes`)：要解碼的 JPG 格式的字節串。

### jpgread

- **img_file** (`Union[str, Path]`)：要讀取的 JPG 圖像文件路徑。

## 範例

### jpgencode

```python
from capybara.vision.improc import imread, jpgencode

img = imread('lena.jpg')
encoded_bytes = jpgencode(img, quality=95)
```

### jpgdecode

```python
from capybara.vision.improc import jpgdecode

decoded_img = jpgdecode(encoded_bytes)
```

### jpgread

```python
from capybara.vision.improc import jpgread

img_array = jpgread('path/to/image.jpg')
```
