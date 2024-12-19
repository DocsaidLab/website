---
sidebar_position: 6
---

# Base64 Process

`pybase64` is a Python library that provides functionality for Base64 encoding and decoding. It supports various encoding formats, including standard Base64, Base64 URL, and Base64 URL filename-safe encoding. `pybase64` is an enhanced version based on the `base64` module, offering additional features and options.

In image processing, we often need to convert image data into a Base64 encoded string for use in network transmission. `pybase64` provides a convenient interface for performing Base64 encoding and decoding operations quickly, while also supporting multiple encoding formats to meet different requirements.

- **Common Question: String vs. Byte String?**

    In Python, a string is a Unicode character sequence, while a byte string is a sequence of "bytes." In Base64 encoding, we typically use byte strings for encoding and decoding operations because Base64 encoding operates on "byte" data.

## img_to_b64

> [img_to_b64(img: np.ndarray, IMGTYP: Union[str, int, IMGTYP] = IMGTYP.JPEG) -> Union[bytes, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L116)

- **Description**: Convert a NumPy image array to a Base64 byte string.

- **Parameters**
    - **img** (`np.ndarray`): The image array to be converted.
    - **IMGTYP** (`Union[str, int, IMGTYP]`): The type of the image. Supported types are `IMGTYP.JPEG` and `IMGTYP.PNG`. Default is `IMGTYP.JPEG`.

- **Returns**
    - **bytes**: The converted Base64 byte string.

- **Example**

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    b64 = D.img_to_b64(img, IMGTYP=D.IMGTYP.PNG)
    ```

## npy_to_b64

> [npy_to_b64(x: np.ndarray, dtype='float32') -> bytes](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L126)

- **Description**: Convert a NumPy array to a Base64 byte string.

- **Parameters**
    - **x** (`np.ndarray`): The NumPy array to be converted.
    - **dtype** (`str`): Data type. Default is `'float32'`.

- **Returns**
    - **bytes**: The converted Base64 byte string.

- **Example**

    ```python
    import docsaidkit as D
    import numpy as np

    x = np.random.rand(100, 100, 3)
    b64 = D.npy_to_b64(x)
    ```

## npy_to_b64str

> [npy_to_b64str(x: np.ndarray, dtype='float32', string_encode: str = 'utf-8') -> str](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L130)

- **Description**: Convert a NumPy array to a Base64 string.

- **Parameters**
    - **x** (`np.ndarray`): The NumPy array to be converted.
    - **dtype** (`str`): Data type. Default is `'float32'`.
    - **string_encode** (`str`): String encoding. Default is `'utf-8'`.

- **Returns**
    - **str**: The converted Base64 string.

- **Example**

    ```python
    import docsaidkit as D
    import numpy as np

    x = np.random.rand(100, 100, 3)

    b64str = D.npy_to_b64str(x)
    ```

## img_to_b64str

> [img_to_b64str(img: np.ndarray, IMGTYP: Union[str, int, IMGTYP] = IMGTYP.JPEG, string_encode: str = 'utf-8') -> Union[str, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L134)

- **Description**: Convert a NumPy image array to a Base64 string.

- **Parameters**
    - **img** (`np.ndarray`): The image array to be converted.
    - **IMGTYP** (`Union[str, int, IMGTYP]`): The type of the image. Supported types are `IMGTYP.JPEG` and `IMGTYP.PNG`. Default is `IMGTYP.JPEG`.
    - **string_encode** (`str`): String encoding. Default is `'utf-8'`.

- **Returns**
    - **str**: The converted Base64 string.

- **Example**

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    b64str = D.img_to_b64str(img, IMGTYP=D.IMGTYP.PNG)
    ```

## b64_to_img

> [b64_to_img(b64: bytes) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L143)

- **Description**: Convert a Base64 byte string to a NumPy image array.

- **Parameters**
    - **b64** (`bytes`): The Base64 byte string to be converted.

- **Returns**
    - **np.ndarray**: The converted NumPy image array.

- **Example**

    ```python
    import docsaidkit as D

    b64 = D.img_to_b64(D.imread('lena.png'))
    img = D.b64_to_img(b64)
    ```

## b64str_to_img

> [b64str_to_img(b64str: Union[str, None], string_encode: str = 'utf-8') -> Union[np.ndarray, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L151)

- **Description**: Convert a Base64 string to a NumPy image array.

- **Parameters**
    - **b64str** (`Union[str, None]`): The Base64 string to be converted.
    - **string_encode** (`str`): String encoding. Default is `'utf-8'`.

- **Returns**
    - **np.ndarray**: The converted NumPy image array.

- **Example**

    ```python
    import docsaidkit as D

    b64 = D.img_to_b64(D.imread('lena.png'))
    b64str = b64.decode('utf-8')
    img = D.b64str_to_img(b64str)
    ```

## b64_to_npy

> [b64_to_npy(x: bytes, dtype='float32') -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L166)

- **Description**: Convert a Base64 byte string to a NumPy array.

- **Parameters**
    - **x** (`bytes`): The Base64 byte string to be converted.
    - **dtype** (`str`): Data type. Default is `'float32'`.

- **Returns**
    - **np.ndarray**: The converted NumPy array.

- **Example**

    ```python
    import docsaidkit as D

    b64 = D.npy_to_b64(np.random.rand(100, 100, 3))
    x = D.b64_to_npy(b64)
    ```

## b64str_to_npy

> [b64str_to_npy(x: bytes, dtype='float32', string_encode: str = 'utf-8') -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L170)

- **Description**: Convert a Base64 string to a NumPy array.

- **Parameters**
    - **x** (`bytes`): The Base64 string to be converted.
    - **dtype** (`str`): Data type. Default is `'float32'`.
    - **string_encode** (`str`): String encoding. Default is `'utf-8'`.

- **Returns**
    - **np.ndarray**: The converted NumPy array.

- **Example**

    ```python
    import docsaidkit as D

    b64 = D.npy_to_b64(np.random.rand(100, 100, 3))
    x = D.b64str_to_npy(b64.decode('utf-8'))
    ```
