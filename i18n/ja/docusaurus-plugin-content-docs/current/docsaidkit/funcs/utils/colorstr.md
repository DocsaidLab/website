---
sidebar_position: 20
---

# colorstr

> [colorstr(obj: Any, color: Union[COLORSTR, int, str] = COLORSTR.BLUE, fmt: Union[FORMATSTR, int, str] = FORMATSTR.BOLD) -> str](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/utils/utils.py#L37)

- **説明**：この関数は、Python オブジェクトを色付きの文字列に変換するために使用されます。

- **パラメータ**

  - **obj** (`Any`)：色付きの文字列に変換する Python オブジェクト。
  - **color** (`Union[COLORSTR, int, str]`)：オブジェクトの色。デフォルトは `COLORSTR.BLUE`。
  - **fmt** (`Union[FORMATSTR, int, str]`)：オブジェクトのフォーマット。デフォルトは `FORMATSTR.BOLD`。

- **戻り値**

  - **str**：色付きの文字列。

- **例**

  ```python
  import docsaidkit as D

  # 青色の文字列を表示
  blue_str = D.colorstr('This is blue color string.', color='blue')
  print(blue_str)

  # 赤色で太字の文字列を表示
  red_str = D.colorstr('This is red color string with bold format.', color='red', fmt='bold')
  print(red_str)
  ```
