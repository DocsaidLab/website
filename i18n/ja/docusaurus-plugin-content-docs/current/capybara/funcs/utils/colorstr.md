# colorstr

> [colorstr(obj: Any, color: Union[COLORSTR, int, str] = COLORSTR.BLUE, fmt: Union[FORMATSTR, int, str] = FORMATSTR.BOLD) -> str](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/utils.py#L37)

- **説明**：この関数は、Python オブジェクトを色付きの文字列に変換するために使用されます。

- **引数**

  - **obj** (`Any`)：色付きの文字列に変換する Python オブジェクト。
  - **color** (`Union[COLORSTR, int, str]`)：オブジェクトの色。デフォルトは `COLORSTR.BLUE`。
  - **fmt** (`Union[FORMATSTR, int, str]`)：オブジェクトの形式。デフォルトは `FORMATSTR.BOLD`。

- **戻り値**

  - **str**：色付きの文字列。

- **例**

  ```python
  import capybara as cb

  # 青色の文字列を表示
  blue_str = cb.colorstr('This is blue color string.', color='blue')
  print(blue_str)

  # 赤色の文字列を太字形式で表示
  red_str = cb.colorstr('This is red color string with bold format.', color='red', fmt='bold')
  print(red_str)
  ```
