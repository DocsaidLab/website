---
slug: opencv-imread
title: Reading HEIC Images in Python
authors: Zephyr
tags: [HEIC, TurboJPEG]
image: /en/img/2024/0213.webp
description: Optimizing imread for reading HEIC images!
---

When you want to read an image, you might use the `imread` function from OpenCV.

Unfortunately, this function is not universal, and you may encounter some problems.

<!-- truncate -->

## Basic Usage

The basic usage of the `imread` function is straightforward; you just need to pass the path to the image:

```python
import cv2

image = cv2.imread('path/to/image.jpg')
```

You can use common image formats such as BMP, JPG, PNG, TIF, and others.

## Limitation 1: HEIC Format

Images captured on iOS devices are typically in HEIC format, which is not supported in OpenCV. If you try to use the `imread` function to read HEIC format images, you will get a `None` return value.

To address this issue, we need to use the `pyheif` package to read HEIC format images and then convert them into `numpy.ndarray` variables.

First, install the necessary packages:

```bash
sudo apt install libheif-dev
pip install pyheif
```

Then, write a simple function:

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

## Limitation 2: Slow JPG Reading

In some cases, the `imread` function's performance in reading JPG format images can be slow. This is because OpenCV uses the `libjpeg` library to read JPG format images, and `libjpeg` itself is not very efficient.

Here, we introduce the `TurboJPEG` package, an alternative to `libjpeg` with better performance. We can use `TurboJPEG` to accelerate the reading of JPG format images.

Similarly, install the necessary packages:

```bash
sudo apt install libturbojpeg exiftool
pip install PyTurboJPEG
```

Then, let's optimize it a bit:

Generally, this can speed up the process by about 2-3 times.

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

This way, reading JPG format images can be accelerated.

## Conclusion

What if we want this program to be more intelligent, choosing a suitable way to load images on its own?

We can write a function to select the appropriate loading method based on the image's format:

```python
def imread(
    path: Union[str, Path],
    color_base: str = 'BGR',
    verbose: bool = False
) -> Union[np.ndarray, None]:

    if not Path(path).exists():
        raise FileExistsError(f'{path} can not be found.')

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

For detailed code, you can refer to: [imread.py](https://github.com/DocsaidLab/DocsaidKit/blob/eb8ac0a56779a75dcc951c683001e6129052cc5a/docsaidkit/vision/improc.py#L197)
