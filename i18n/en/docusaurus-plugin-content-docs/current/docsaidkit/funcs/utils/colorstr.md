---
sidebar_position: 20
---

# colorstr

> [colorstr(obj: Any, color: Union[COLORSTR, int, str] = COLORSTR.BLUE, fmt: Union[FORMATSTR, int, str] = FORMATSTR.BOLD) -> str](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/utils/utils.py#L37)

- **Description**: This function is used to convert Python objects into colored strings.

- **Parameters**:
    - **obj** (`Any`): Python object to be converted into a colored string.
    - **color** (`Union[COLORSTR, int, str]`): Color of the object. Default is `COLORSTR.BLUE`.
    - **fmt** (`Union[FORMATSTR, int, str]`): Format of the object. Default is `FORMATSTR.BOLD`.

- **Returns**:
    - **str**: Colored string.

- **Example**:

    ```python
    import docsaidkit as D

    # Print blue color string
    blue_str = D.colorstr('This is blue color string.', color='blue')
    print(blue_str)

    # Print red color string with bold format
    red_str = D.colorstr('This is red color string with bold format.', color='red', fmt='bold')
    print(red_str)
    ```
