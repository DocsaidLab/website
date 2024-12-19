---
sidebar_position: 1
---

# opencc

## convert_simplified_to_traditional

> [convert_simplified_to_traditional(text: str) -> str](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/custom_opencc.py#L10)

- **說明**：將簡體中文轉換為繁體中文。

- **參數**
    - **text** (`str`)：要轉換的簡體中文文本。

- **傳回值**
    - **str**：轉換後的繁體中文文本。

- **範例**

    ```python
    import docsaidkit as D

    text = '这是一个简体中文文本。'
    traditional_text = D.convert_simplified_to_traditional(text)
    print(traditional_text)
    # >>> '這是一個簡體中文文本。'
    ```

## convert_traditional_to_simplified

> [convert_traditional_to_simplified(text: str) -> str](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/custom_opencc.py#L15)

- **說明**：將繁體中文轉換為簡體中文。

- **參數**
    - **text** (`str`)：要轉換的繁體中文文本。

- **傳回值**
    - **str**：轉換後的簡體中文文本。

- **範例**

    ```python
    import docsaidkit as D

    text = '這是一個繁體中文文本。'
    simplified_text = D.convert_traditional_to_simplified(text)
    print(simplified_text)
    # >>> '这是一个繁体中文文本。'
    ```
