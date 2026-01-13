# Tqdm

> [Tqdm(iterable=None, desc=None, smoothing=0, **kwargs)](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/custom_tqdm.py)

- **説明**：これは `tqdm` を継承したカスタム進捗バーで、イテレータの繰り返し処理中に進捗バーを表示するために使用されます。この進捗バーは、元々の `tqdm` に対して `total` パラメータを自動で計算し、`iterable` の長さを `total` に設定する変更を加えています。これにより、ユーザーが `total` を指定しなくても進捗バーが正しく表示されるようになります。

- **引数**

  - **iterable** (`Iterable`)：反復可能なオブジェクト。
  - **desc** (`str`)：進捗バーの説明。
  - **smoothing** (`int`)：平滑化パラメータ。
  - **kwargs** (`Any`)：その他のパラメータ。

- **例**

  ```python
  from capybara.utils.custom_tqdm import Tqdm

  for _i in Tqdm(range(100), desc='Processing'):
      pass
  ```
