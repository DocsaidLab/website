# PowerDict

> [PowerDict(d=None, \*\*kwargs)](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/powerdict.py#L10)

- **説明**：このクラスは、凍結および解凍機能を持つ辞書を作成するために使用され、` .` を使って内部の属性にアクセスできます。

- **属性**

  - **is_frozen** (`bool`)：辞書が凍結されているかどうかを判定します。

- **メソッド**

  - **freeze()**：辞書を凍結します。
  - **melt()**：辞書を解凍します。
  - **to_dict()**：辞書を標準的な辞書に変換します。
  - **to_json(path: Union[str, Path]) -> None**：辞書を JSON ファイルに書き込みます。
  - **to_yaml(path: Union[str, Path]) -> None**：辞書を YAML ファイルに書き込みます。
  - **to_txt(path: Union[str, Path]) -> None**：辞書を TXT ファイルに書き込みます。
  - **to_pickle(path: Union[str, Path]) -> None**：辞書を Pickle ファイルに書き込みます。

－ **クラスメソッド**

    - **load_json(path: Union[str, Path]) -> PowerDict**：JSON ファイルから辞書を読み込みます。
    - **load_pickle(path: Union[str, Path]) -> PowerDict**：Pickle ファイルから辞書を読み込みます。
    - **load_yaml(path: Union[str, Path]) -> PowerDict**：YAML ファイルから辞書を読み込みます。

- **パラメータ**

  - **d** (`dict`)：変換する辞書。デフォルトは None。

- **例**

  ```python
  from capybara import PowerDict

  d = {'key': 'value'}
  cfg = PowerDict(d)
  print(cfg.key)
  # >>> 'value'
  ```
