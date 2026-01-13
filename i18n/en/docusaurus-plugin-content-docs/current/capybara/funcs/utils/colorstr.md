# colorstr

> [colorstr(obj: Any, color: COLORSTR | int | str = COLORSTR.BLUE, fmt: FORMATSTR | int | str = FORMATSTR.BOLD) -> str](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/utils.py)

- **Description**: This function is used to convert a Python object into a string with color.

- **Parameters**

  - **obj** (`Any`): The Python object to be converted into a colored string.
  - **color** (`Union[COLORSTR, int, str]`): The color of the object. Default is `COLORSTR.BLUE`.
  - **fmt** (`Union[FORMATSTR, int, str]`): The format of the object. Default is `FORMATSTR.BOLD`.
    Supported: `BOLD` / `ITALIC` / `UNDERLINE`.

- **Returns**

  - **str**: A string with color.

- **Example**

  ```python
  import capybara as cb

  # Print blue color string
  blue_str = cb.colorstr('This is blue color string.', color='blue')
  print(blue_str)

  # Print red color string with bold format
  red_str = cb.colorstr('This is red color string with bold format.', color='red', fmt='bold')
  print(red_str)
  ```
