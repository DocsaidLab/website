---
slug: opencv-imread
title: 用 Python 讀取 HEIC 圖檔與加速載入
authors: Z. Yuan
tags: [HEIC, TurboJPEG]
image: /img/2024/0213.webp
description: 來優化一下 OpenCV imread 吧！
---

當你想要讀取一張影像時，你可能會使用 OpenCV 的 `imread` 函數。

可惜這個函數並不是萬能的，有時候可能會遇到一些問題。

<!-- truncate -->

## 基礎用法

`imread` 函數的基礎用法非常簡單，只要傳入影像的路徑即可：

```python
import cv2

image = cv2.imread('path/to/image.jpg')
```

其中，可以使用的影像格式包括：BMP, JPG, PNG, TIF 等常見影像格式。

## 限制一：HEIC 格式

在 iOS 裝置上拍攝的照片通常是 HEIC 格式，這種格式在 OpenCV 中是不支援的。如果你嘗試使用 `imread` 函數讀取 HEIC 格式的影像，會得到一個 `None` 的返回值。

為了解決這個問題，我們得用 pyheif 這個套件來讀取 HEIC 格式的影像，然後再將其轉換成 `numpy.ndarray` 類型的變數。

首先，安裝必要套件：

```bash
sudo apt install libheif-dev
pip install pyheif
```

接著寫個簡單的函數：

```python
import cv2
import pyheif
import numpy as np

def read_heic_to_numpy(file_path: str):
    heif_file = pyheif.read(file_path)
    data = heif_file.data
    if heif_file.mode == "RGB":
        numpy_array = np.frombuffer(data, dtype=np.uint8).reshape(
            heif_file.size[1], heif_file.size[0], 3)
    elif heif_file.mode == "RGBA":
        numpy_array = np.frombuffer(data, dtype=np.uint8).reshape(
            heif_file.size[1], heif_file.size[0], 4)
    else:
        raise ValueError("Unsupported HEIC color mode")
    return numpy_array


img = read_heic_to_numpy('path/to/image.heic')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
```

## 限制二：JPG 讀取慢

在某些情況下，`imread` 函數讀取 JPG 格式的影像速度會非常慢。這是因為 OpenCV 在讀取 JPG 格式的影像時，會使用 libjpeg 這個庫，而 libjpeg 本身的效能就不是很好。

在這邊，我們引入 TurboJPEG 這個套件，它是 libjpeg 的一個替代品，效能更好。

跟之前一樣，先安裝必要套件：

```bash
sudo apt install libturbojpeg exiftool
pip install PyTurboJPEG
```

再來寫一點程式，幫他加速一下：

一般來說，可以加速大約 2-3 倍。

```python
import cv2
import piexif
from enum import IntEnum
from pathlib import Path
from turbojpeg import TurboJPEG


jpeg = TurboJPEG()


class ROTATE(IntEnum):
    ROTATE_90 = cv2.ROTATE_90_CLOCKWISE
    ROTATE_180 = cv2.ROTATE_180
    ROTATE_270 = cv2.ROTATE_90_COUNTERCLOCKWISE


def imrotate90(img, rotate_code: ROTATE) -> np.ndarray:
    return cv2.rotate(img.copy(), rotate_code)


def get_orientation_code(stream: Union[str, Path, bytes]):
    code = None
    try:
        exif_dict = piexif.load(stream)
        if piexif.ImageIFD.Orientation in exif_dict["0th"]:
            orientation = exif_dict["0th"][piexif.ImageIFD.Orientation]
            if orientation == 3:
                code = ROTATE.ROTATE_180
            elif orientation == 6:
                code = ROTATE.ROTATE_90
            elif orientation == 8:
                code = ROTATE.ROTATE_270
    finally:
        return code


def jpgdecode(byte_: bytes) -> Union[np.ndarray, None]:
    try:
        bgr_array = jpeg.decode(byte_)
        code = get_orientation_code(byte_)
        bgr_array = imrotate90(
            bgr_array, code) if code is not None else bgr_array
    except:
        bgr_array = None

    return bgr_array


def jpgread(img_file: Union[str, Path]) -> Union[np.ndarray, None]:
    with open(str(img_file), 'rb') as f:
        binary_img = f.read()
        bgr_array = jpgdecode(binary_img)

    return bgr_array

img = jpgread('path/to/image.jpg')
```

這樣就可以加速讀取 JPG 格式的影像了。

## 結語

那如果我們希望這個程式可以聰明一點，自己選一個適合載入的方式呢？

這裡可以彙整一下上面的程式，根據影像的格式來選擇合適的載入方式：

```python title="custom_imread.py"
def imread(
    path: Union[str, Path],
    color_base: str = 'BGR',
    verbose: bool = False
) -> Union[np.ndarray, None]:

    if not Path(path).exists():
        raise FileExistsError(f'{path} can not found.')

    if Path(path).suffix.lower() == '.heic':
        img = read_heic_to_numpy(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = jpgread(path)
        img = cv2.imread(str(path)) if img is None else img

    if img is None:
        if verbose:
            warnings.warn("Got a None type image.")
        return

    if color_base != 'BGR':
        img = imcvtcolor(img, cvt_mode=f'BGR2{color_base}')

    return img
```
