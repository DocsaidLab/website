# colorstr

> [colorstr(obj: Any, color: Union[COLORSTR, int, str] = COLORSTR.BLUE, fmt: Union[FORMATSTR, int, str] = FORMATSTR.BOLD) -> str](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/utils.py#L37)

- **Description**: This function is used to convert a Python object into a string with color.

- **Parameters**

  - **obj** (`Any`): The Python object to be converted into a colored string.
  - **color** (`Union[COLORSTR, int, str]`): The color of the object. Default is `COLORSTR.BLUE`.
  - **fmt** (`Union[FORMATSTR, int, str]`): The format of the object. Default is `FORMATSTR.BOLD`.

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
