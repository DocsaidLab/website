---
sidebar_position: 4
---

# JPG Process

>[get_orientation_code(stream: Union[str, Path, bytes]) -> Union[ROTATE, None]](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/improc.py#L34C5-L34C25)

>[jpgencode(img: np.ndarray, quality: int = 90) -> Union[bytes, None]](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/improc.py#L50)

>[jpgdecode(byte_: bytes) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/improc.py#L60)

>[jpgread(img_file: Union[str, Path]) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/improc.py#L72)

This set of functions provides support for encoding, decoding, and reading JPG images, as well as automatically adjusting the image orientation from EXIF data.

## Description

- **get_orientation_code**: Extracts orientation information from the image's EXIF data and converts it into a code suitable for image rotation. This step is automatically performed in the `jpgdecode` and `jpgread` functions to ensure that the orientation of the read image is correct.
- **jpgencode**: Encodes a NumPy image array into a byte string in JPG format. When using the `jpgencode` function, you can adjust the `quality` parameter to balance image quality and file size.
- **jpgdecode**: Decodes a byte string in JPG format into a NumPy image array and adjusts the image orientation based on EXIF data.
- **jpgread**: Reads a JPG image from a file, decodes it into a NumPy image array, and adjusts the image orientation based on EXIF data.

## Parameters

### jpgencode

- **img** (`np.ndarray`): The image array to be encoded.
- **quality** (`int`): Encoding quality, ranging from 1 to 100. Default is 90.

### jpgdecode

- **byte_** (`bytes`): The byte string in JPG format to be decoded.

### jpgread

- **img_file** (`Union[str, Path]`): The path to the JPG image file to be read.

## Example

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

## Additional Note: TurboJPEG

[TurboJPEG](https://github.com/libjpeg-turbo/libjpeg-turbo) is an efficient JPEG image processing library that provides fast encoding, decoding, compression, and decompression of images. In the `jpgencode` and `jpgdecode` functions, we utilize `TurboJPEG` for encoding and decoding JPG images. TurboJPEG is a Python wrapper for `libjpeg-turbo`, which offers faster image encoding and decoding speeds and supports various image formats.

- **Features**

    - **Efficiency**: Leveraging the high-performance features of the libjpeg-turbo library, TurboJPEG significantly improves image processing speed compared to traditional JPEG processing methods.
    - **Ease of Use**: Provides a concise and clear API, allowing developers to easily implement efficient JPEG image processing in their applications.
    - **Flexibility**: Supports various image quality and compression level settings to meet the requirements for image quality and file size in different scenarios.
    - **Cross-Platform**: Supports multiple operating systems including Windows, macOS, and Linux, making it convenient to use in different development environments.

Once installed, TurboJPEG can be used in Python for encoding and decoding functionalities as follows:

```python
from turbojpeg import TurboJPEG

# Initialize a TurboJPEG instance
jpeg = TurboJPEG()

# Decode
bgr_array = jpeg.decode(byte_)

# Encode
byte_ = jpeg.encode(img, quality=quality)
```