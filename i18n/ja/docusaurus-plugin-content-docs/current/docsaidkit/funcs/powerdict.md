---
sidebar_position: 5
---

# PowerDict

> [PowerDict(d=None, \*\*kwargs)](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/powerdict.py#L10)

- **説明**：このクラスは、凍結と解凍機能を持つ辞書を作成するために使用され、` .` を使って内部の属性にアクセスすることができます。

- **属性**

  - **is_frozen** (`bool`)：辞書が凍結されているかどうかを判断します。

- **メソッド**

  - **freeze()**：辞書を凍結します。
  - **melt()**：辞書を解凍します。
  - **to_dict()**：辞書を標準の辞書に変換します。
  - **to_json(path: Union[str, Path]) -> None**：辞書を json ファイルに書き込みます。
  - **to_yaml(path: Union[str, Path]) -> None**：辞書を yaml ファイルに書き込みます。
  - **to_txt(path: Union[str, Path]) -> None**：辞書を txt ファイルに書き込みます。
  - **to_pickle(path: Union[str, Path]) -> None**：辞書を pickle ファイルに書き込みます。

- **クラスメソッド**

  - **load_json(path: Union[str, Path]) -> PowerDict**：json ファイルから辞書を読み込みます。
  - **load_pickle(path: Union[str, Path]) -> PowerDict**：pickle ファイルから辞書を読み込みます。
  - **load_yaml(path: Union[str, Path]) -> PowerDict**：yaml ファイルから辞書を読み込みます。

- **パラメータ**

  - **d** (`dict`)：変換する辞書。デフォルトは None です。

- **例**

  ```python
  from docsaidkit import PowerDict

  d = {'key': 'value'}
  cfg = PowerDict(d)
  print(cfg.key)
  # >>> 'value'
  ```
