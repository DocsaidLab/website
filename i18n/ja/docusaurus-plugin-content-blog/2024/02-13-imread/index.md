---
slug: opencv-imread
title: PythonでHEIC画像を読み込む方法と読み込みの高速化
authors: Z. Yuan
tags: [HEIC, TurboJPEG]
image: /ja/img/2024/0213.webp
description: OpenCVのimreadを最適化してみよう！
---

画像を読み込むとき、通常はOpenCVの`imread`関数を使用します。

しかし、この関数は万能ではなく、時々いくつかの問題に直面することがあります。

<!-- truncate -->

## 基本的な使い方

`imread`関数の基本的な使い方は非常に簡単で、画像のパスを渡すだけです：

```python
import cv2

image = cv2.imread('path/to/image.jpg')
```

使用可能な画像フォーマットには、BMP、JPG、PNG、TIFなどの一般的な画像フォーマットが含まれます。

## 限界1: HEICフォーマット

iOSデバイスで撮影した写真は通常HEICフォーマットで保存されますが、このフォーマットはOpenCVではサポートされていません。もし`imread`関数を使ってHEICフォーマットの画像を読み込もうとすると、`None`が返されます。

この問題を解決するために、`pyheif`パッケージを使用してHEICフォーマットの画像を読み込み、その後`numpy.ndarray`型の変数に変換する方法を取ります。

まず、必要なパッケージをインストールします：

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

## 限界2: JPGの読み込み速度が遅い

場合によっては、`imread`関数でJPGフォーマットの画像を読み込む速度が非常に遅くなることがあります。これは、OpenCVがJPGフォーマットの画像を読み込む際に`libjpeg`ライブラリを使用しているためですが、`libjpeg`自体のパフォーマンスがあまり良くありません。

ここで、`libjpeg`の代替品である`TurboJPEG`パッケージを導入し、パフォーマンスを向上させる方法を取ります。

先ほどと同様に、必要なパッケージをインストールします：

```bash
sudo apt install libturbojpeg exiftool
pip install PyTurboJPEG
```

次に、少しコードを書いて画像読み込みを高速化します：

通常、約2〜3倍の速度向上が期待できます。

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

これでJPGフォーマットの画像読み込みが高速化されます。

## 結論

このプログラムを賢くして、画像のフォーマットに応じて最適な読み込み方法を選択できるようにしたい場合は、先ほどのコードを整理して適切な方法を選べるようにします。

```python title="custom_imread.py"
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
