---
slug: opencv-imread
title: Python で HEIC 画像を読み込む
authors: Z. Yuan
tags: [HEIC, TurboJPEG]
image: /ja/img/2024/0213.webp
description: OpenCV imread を最適化しましょう！
---

画像を読み込む際、OpenCV の `imread` 関数を使用することがよくあります。

しかし、この関数は万能ではなく、場合によっては問題が発生することがあります。

<!-- truncate -->

## 基本的な使い方

`imread` 関数の基本的な使い方は非常に簡単で、画像のパスを渡すだけです：

```python
import cv2

image = cv2.imread('path/to/image.jpg')
```

この関数で使用できる画像フォーマットには、BMP、JPG、PNG、TIF などの一般的な形式が含まれます。

## 制限 1：HEIC フォーマット

iOS デバイスで撮影した写真は通常 HEIC フォーマットです。この形式は OpenCV ではサポートされていません。もし `imread` 関数を使って HEIC 画像を読み込もうとすると、`None` が返されます。

この問題を解決するには、HEIC フォーマットの画像を読み込むために `pyheif` ライブラリを使用し、その後 `numpy.ndarray` 型の変数に変換する必要があります。

まず、必要なライブラリをインストールします：

```bash
sudo apt install libheif-dev
pip install pyheif
```

次に、簡単な関数を作成します：

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

## 制限 2：JPG の読み込みが遅い

特定の状況では、`imread` 関数を使用して JPG フォーマットの画像を読み込む速度が非常に遅い場合があります。これは、OpenCV が JPG フォーマットの画像を読み込む際に使用する `libjpeg` ライブラリのパフォーマンスがあまり良くないためです。

そこで、`libjpeg` の代替として、より高性能な TurboJPEG ライブラリを導入します。TurboJPEG を使用することで、JPG フォーマットの画像の読み込み速度を向上させることができます。

まず、必要なライブラリをインストールします：

```bash
sudo apt install libturbojpeg exiftool
pip install PyTurboJPEG
```

次に、以下のコードで読み込みを高速化します：

一般的に、読み込み速度は約 2 ～ 3 倍向上します。

```python
import cv2
import piexif
from enum import IntEnum
from pathlib import Path
from turbojpeg import TurboJPEG
import numpy as np
from typing import Union


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

これで、JPG フォーマットの画像の読み込みを高速化することができます。

## 結論

プログラムをもう少し賢くして、適切な読み込み方法を自動的に選択させたい場合はどうすればよいでしょうか？

画像のフォーマットに応じて適切な読み込み方法を選択する関数を作成できます：

```python
def imread(
    path: Union[str, Path],
    color_base: str = 'BGR',
    verbose: bool = False
) -> Union[np.ndarray, None]:

    if not Path(path).exists():
        raise FileExistsError(f'{path} が見つかりません。')

    if Path(path).suffix.lower() == '.heic':
        img = read_heic_to_numpy(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = jpgread(path)
        img = cv2.imread(str(path)) if img is None else img

    if img is None:
        if verbose:
            warnings.warn("画像が None 型です。")
        return

    if color_base != 'BGR':
        img = imcvtcolor(img, cvt_mode=f'BGR2{color_base}')

    return img
```
