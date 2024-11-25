---
sidebar_position: 1
---

# opencc

## convert_simplified_to_traditional

> [convert_simplified_to_traditional(text: str) -> str](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/custom_opencc.py#L10)

- **説明**：簡体字を繁体字に変換します。

- **パラメータ**

  - **text** (`str`)：変換する簡体字のテキスト。

- **戻り値**

  - **str**：変換された繁体字のテキスト。

- **例**

  ```python
  import docsaidkit as D

  text = '这是一个简体中文文本。'
  traditional_text = D.convert_simplified_to_traditional(text)
  print(traditional_text)
  # >>> '這是一個簡體中文文本。'
  ```

## convert_traditional_to_simplified

> [convert_traditional_to_simplified(text: str) -> str](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/custom_opencc.py#L15)

- **説明**：繁体字を簡体字に変換します。

- **パラメータ**

  - **text** (`str`)：変換する繁体字のテキスト。

- **戻り値**

  - **str**：変換された簡体字のテキスト。

- **例**

  ```python
  import docsaidkit as D

  text = '這是一個繁體中文文本。'
  simplified_text = D.convert_traditional_to_simplified(text)
  print(simplified_text)
  # >>> '这是一个繁体中文文本。'
  ```
