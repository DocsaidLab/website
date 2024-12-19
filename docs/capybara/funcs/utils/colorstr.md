---
sidebar_position: 20
---

# colorstr

> [colorstr(obj: Any, color: Union[COLORSTR, int, str] = COLORSTR.BLUE, fmt: Union[FORMATSTR, int, str] = FORMATSTR.BOLD) -> str](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/utils/utils.py#L37)

- **說明**：這個函數用於將 Python 物件轉換為帶有顏色的字串。

- **參數**
    - **obj** (`Any`)：要轉換為帶有顏色的字串的 Python 物件。
    - **color** (`Union[COLORSTR, int, str]`)：物件的顏色。預設為 `COLORSTR.BLUE`。
    - **fmt** (`Union[FORMATSTR, int, str]`)：物件的格式。預設為 `FORMATSTR.BOLD`。

- **傳回值**

    - **str**：帶有顏色的字串。

- **範例**

    ```python
    import docsaidkit as D

    # Print blue color string
    blue_str = D.colorstr('This is blue color string.', color='blue')
    print(blue_str)

    # Print red color string with bold format
    red_str = D.colorstr('This is red color string with bold format.', color='red', fmt='bold')
    print(red_str)
    ```
