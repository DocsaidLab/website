---
sidebar_position: 2
---

# インストール

PyPI からインストールする方法、または GitHub から本プロジェクトをクローンしてインストールする方法を提供しています。

## PyPI からインストール

1. `docclassifier-docsaid` をインストールします：

   ```bash
   pip install docclassifier-docsaid
   ```

2. インストールが成功したか確認するため、以下のコマンドを実行します：

   ```bash
   python -c "import docclassifier; print(docclassifier.__version__)"
   ```

3. バージョン番号が表示されれば、インストールが成功したことを意味します。

## GitHub からインストール

:::tip
GitHub からインストールするには、まず `Capybara` がインストールされていることを確認してください。

インストールされていない場合は、[**Capybara インストールガイド**](../capybara/installation.md) を参照してください。
:::

1. **プロジェクトをクローンします：**

   ```bash
   git clone https://github.com/DocsaidLab/DocClassifier.git
   ```

2. **プロジェクトディレクトリに移動します：**

   ```bash
   cd DocClassifier
   ```

3. **依存パッケージをインストールします：**

   ```bash
   pip install setuptools wheel
   ```

4. **パッケージファイルを作成します：**

   ```bash
   python setup.py bdist_wheel
   ```

5. **パッケージファイルをインストールします：**

   ```bash
   pip install dist/docclassifier_docsaid-*-py3-none-any.whl
   ```

これらの手順に従うことで、`DocClassifier` のインストールが正常に完了するはずです。

インストール後、以下のコマンドでインストールが成功したかどうかを確認できます：

```bash
python -c "import docclassifier; print(docclassifier.__version__)"
# >>> 0.10.0
```

`0.10.0` のようなバージョン番号が表示されれば、インストールが成功したことを意味します。
