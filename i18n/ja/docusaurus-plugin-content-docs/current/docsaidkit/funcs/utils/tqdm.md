---
sidebar_position: 6
---

# Tqdm

> [Tqdm(iterable=None, desc=None, smoothing=0, \*\*kwargs)](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/custom_tqdm.py#L8)

- **説明**：`tqdm`を継承したカスタム進捗バーで、イテレータの繰り返しの進行状況を表示します。このカスタマイズでは、`total`パラメータに変更を加えています。ユーザーが`total`を指定しない場合、自動的に`iterable`の長さを計算して`total`に設定します。これにより、ユーザーが`total`を指定しなくても正しく進捗バーが表示されるようになります。

- **パラメータ**

  - **iterable** (`Iterable`)：イテレートするオブジェクト。
  - **desc** (`str`)：進捗バーの説明。
  - **smoothing** (`int`)：平滑化パラメータ。
  - **kwargs** (`Any`)：その他のパラメータ。

- **例**

  ```python
  import docsaidkit as D

  for i in D.Tqdm(range(100), desc='Processing'):
      pass
  ```
