# JPG Process

> [get_orientation_code(stream: str | Path | bytes) -> ROTATE | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

> [jpgencode(img: np.ndarray, quality: int = 90) -> bytes | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

> [jpgdecode(byte\_: bytes) -> np.ndarray | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

> [jpgread(img_file: str | Path) -> np.ndarray | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

These helpers provide JPEG encode/decode/read support and apply EXIF orientation during decode.

## Description

- **get_orientation_code**: Reads EXIF orientation and returns the corresponding rotate code (used by `jpgdecode` / `jpgread`).
- **jpgencode**: Encodes a numpy image into JPEG bytes; returns `None` on failure.
- **jpgdecode**: Decodes JPEG bytes into a numpy image and applies EXIF orientation; returns `None` on failure.
- **jpgread**: Reads a JPEG file and returns a numpy image (via `jpgdecode`).

## Dependencies

- This module uses `turbojpeg` (PyTurboJPEG) for JPEG encode/decode.

## Parameters

### jpgencode

- **img** (`np.ndarray`): Image array.
- **quality** (`int`): JPEG quality (1 to 100). Default is 90.

### jpgdecode

- **byte_** (`bytes`): JPEG bytes.

### jpgread

- **img_file** (`str | Path`): JPEG file path.

## Examples

### jpgencode

```python
from capybara.vision.improc import imread, jpgencode

img = imread('lena.png')
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
