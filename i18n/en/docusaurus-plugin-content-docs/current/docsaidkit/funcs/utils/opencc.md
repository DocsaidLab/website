---
sidebar_position: 1
---

# opencc

## convert_simplified_to_traditional

> [convert_simplified_to_traditional(text: str) -> str](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/custom_opencc.py#L10)

- **Description**: Convert Simplified Chinese text to Traditional Chinese.

- **Parameters**:
    - **text** (`str`): The Simplified Chinese text to convert.

- **Returns**:
    - **str**: The converted Traditional Chinese text.

- **Example**:

    ```python
    import docsaidkit as D

    text = '这是一个简体中文文本。'
    traditional_text = D.convert_simplified_to_traditional(text)
    print(traditional_text)
    # >>> '這是一個簡體中文文本。'
    ```

## convert_traditional_to_simplified

> [convert_traditional_to_simplified(text: str) -> str](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/custom_opencc.py#L15)

- **Description**: Convert Traditional Chinese to Simplified Chinese text.

- **Parameters**:
    - **text** (`str`): The Traditional Chinese text to convert.

- **Returns**:
    - **str**: The converted Simplified Chinese text.

- **Example**:

    ```python
    import docsaidkit as D

    text = '這是一個簡體中文文本。'
    simplified_text = D.convert_traditional_to_simplified(text)
    print(simplified_text)
    # >>> '这是一个简体中文文本。'
    ```
