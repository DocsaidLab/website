# JPG Process

> [get_orientation_code(stream: Union[str, Path, bytes]) -> Union[ROTATE, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L34)

> [jpgencode(img: np.ndarray, quality: int = 90) -> Union[bytes, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L50)

> [jpgdecode(byte\_: bytes) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L60)

> [jpgread(img_file: Union[str, Path]) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L72)

These functions provide support for encoding, decoding, and reading JPG images, as well as automatically adjusting image orientation from EXIF data.

## Description

- **get_orientation_code**: Extracts orientation information from the image's EXIF data and converts it into a code suitable for image rotation. This step is automatically performed in the `jpgdecode` and `jpgread` functions to ensure that the image is correctly oriented when read.
- **jpgencode**: Encodes a NumPy image array into a JPG format byte string. The `quality` parameter allows for balancing image quality and file size.
- **jpgdecode**: Decodes a JPG format byte string into a NumPy image array, adjusting the image orientation based on EXIF data.
- **jpgread**: Reads a JPG image file, decodes it into a NumPy image array, and adjusts the image orientation based on EXIF data.

## Parameters

### jpgencode

- **img** (`np.ndarray`): The image array to encode.
- **quality** (`int`): The encoding quality, ranging from 1 to 100. Default is 90.

### jpgdecode

- **byte\_** (`bytes`): The JPG format byte string to decode.

### jpgread

- **img_file** (`Union[str, Path]`): The path to the JPG image file to read.

## Example

### jpgencode

```python
import numpy as np
import capybara as cb

img_array = np.random.rand(100, 100, 3) * 255
encoded_bytes = cb.jpgencode(img_array, quality=95)
```

### jpgdecode

```python
decoded_img = cb.jpgdecode(encoded_bytes)
```

### jpgread

```python
img_array = cb.jpgread('path/to/image.jpg')
```

## Additional Notes: TurboJPEG

[**TurboJPEG**](https://github.com/libjpeg-turbo/libjpeg-turbo) is an efficient JPEG image processing library that offers fast encoding, decoding, compression, and decompression functionalities. The `jpgencode` and `jpgdecode` functions utilize `TurboJPEG` for encoding and decoding JPG images. `TurboJPEG` is a Python wrapper for `libjpeg-turbo`, providing faster image processing speeds and supporting multiple image formats.

- **Features**:

  - **High efficiency**: Leverages the high-performance features of the libjpeg-turbo library, significantly improving image processing speed compared to traditional JPEG methods.
  - **Ease of use**: Provides a clear and simple API, making it easy for developers to implement efficient JPEG image processing in applications.
  - **Flexibility**: Supports various image quality and compression settings to meet different requirements for image quality and file size.
  - **Cross-platform**: Supports multiple operating systems, including Windows, macOS, and Linux, making it convenient for use in different development environments.

After installation, you can use TurboJPEG for encoding and decoding in Python as follows:

```python
from turbojpeg import TurboJPEG

# Initialize TurboJPEG instance
jpeg = TurboJPEG()

# Decode
bgr_array = jpeg.decode(byte_)

# Encode
byte_ = jpeg.encode(img, quality=quality)
```
