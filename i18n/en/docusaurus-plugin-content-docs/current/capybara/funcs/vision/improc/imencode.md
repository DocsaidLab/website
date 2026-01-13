# imencode

> [imencode(img: np.ndarray, imgtyp: str | int | IMGTYP = IMGTYP.JPEG, **kwargs: object) -> bytes | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **Description**: Encodes a numpy image into bytes with the requested format.

- **Parameters**

  - **img** (`np.ndarray`): Image array.
  - **imgtyp** (`str | int | IMGTYP`): Image type. Supports `IMGTYP.JPEG` / `IMGTYP.PNG`. Default is `IMGTYP.JPEG`.

- **Returns**

  - **bytes | None**: Encoded bytes; returns `None` on failure.

- **Notes**

  - For backward compatibility, `IMGTYP=...` is also accepted (providing both `imgtyp` and `IMGTYP` raises `TypeError`).

- **Example**

  ```python
  from capybara.vision.improc import IMGTYP, imencode, imread

  img = imread('lena.png')
  encoded_bytes = imencode(img, imgtyp=IMGTYP.PNG)
  ```
