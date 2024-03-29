---
sidebar_position: 4
---

# JPG Process

>[get_orientation_code(stream: Union[str, Path, bytes]) -> Union[ROTATE, None]](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/improc.py#L34C5-L34C25)

>[jpgencode(img: np.ndarray, quality: int = 90) -> Union[bytes, None]](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/improc.py#L50)

>[jpgdecode(byte_: bytes) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/improc.py#L60)

>[jpgread(img_file: Union[str, Path]) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/improc.py#L72)

這一系列函數提供了對 JPG 圖像的編碼、解碼和讀取支持，以及從 EXIF 數據自動調整圖像方向的功能。

## 說明

- **get_orientation_code**：從圖像的 EXIF 數據中提取方向信息，並將其轉換為適合於圖像旋轉的代碼。這一步是在 `jpgdecode` 和 `jpgread` 函數中自動完成的，以確保讀取的圖像顯示方向正確。
- **jpgencode**：將 NumPy 圖像數組編碼為 JPG 格式的字節串，使用 `jpgencode` 函數時，可以通過調整 `quality` 參數來平衡圖像質量與文件大小。
- **jpgdecode**：將 JPG 格式的字節串解碼為 NumPy 圖像數組，並根據 EXIF 數據調整圖像方向。
- **jpgread**：從文件中讀取 JPG 圖像，解碼為 NumPy 圖像數組，並根據 EXIF 數據調整圖像方向。

## 參數

### jpgencode

- **img** (`np.ndarray`)：要編碼的圖像數組。
- **quality** (`int`)：編碼質量，範圍為 1 至 100。預設為 90。

### jpgdecode

- **byte_** (`bytes`)：要解碼的 JPG 格式的字節串。

### jpgread

- **img_file** (`Union[str, Path]`)：要讀取的 JPG 圖像文件路徑。

## 範例

### jpgencode

```python
import numpy as np
import docsaidkit as D

img_array = np.random.rand(100, 100, 3) * 255
encoded_bytes = D.jpgencode(img_array, quality=95)
```

### jpgdecode

```python
decoded_img = D.jpgdecode(encoded_bytes)
```

### jpgread

```python
img_array = D.jpgread('path/to/image.jpg')
```

## 附加說明：TurboJPEG

[**TurboJPEG**](https://github.com/libjpeg-turbo/libjpeg-turbo) 是一種高效的 JPEG 圖像處理庫，提供了快速的圖像編碼、解碼、壓縮和解壓縮功能。在 `jpgencode` 和 `jpgdecode` 函數中，我們使用了 `TurboJPEG` 進行 JPG 圖像的編碼和解碼。`TurboJPEG` 是 `libjpeg-turbo` 的 Python 封裝，它提供了更快的圖像編碼和解碼速度，並支持多種圖像格式。

- **特點**

    - **高效率**：利用了 libjpeg-turbo 庫的高性能特性，相較於傳統的 JPEG 處理方法，TurboJPEG 可以大幅提高圖像處理的速度。
    - **易用性**：提供了簡潔明了的 API，使得開發者可以輕鬆地在應用程序中實現高效的 JPEG 圖像處理。
    - **靈活性**：支持多種圖像質量和壓縮等級的設置，滿足不同場景下對圖像質量和文件大小的需求。
    - **跨平台**：支持 Windows、macOS 和 Linux 等多個操作系統，方便在不同開發環境中使用。

安裝完成後，可以通過以下方式在 Python 中使用 TurboJPEG 進行編解碼的功能：

```python
from turbojpeg import TurboJPEG

# 初始化 TurboJPEG 實例
jpeg = TurboJPEG()

# 解碼
bgr_array = jpeg.decode(byte_)

# 編碼
byte_ = jpeg.encode(img, quality=quality)
```
