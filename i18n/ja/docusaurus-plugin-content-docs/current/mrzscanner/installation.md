---
sidebar_position: 2
---

# インストール

`MRZScanner` をインストールするには、PyPI 経由または GitHub リポジトリからクローンする方法があります。

## PyPI を使用したインストール

1. `mrzscanner-docsaid` をインストール：

   ```bash
   pip install mrzscanner-docsaid
   ```

2. インストールの確認：

   ```bash
   python -c "import mrzscanner; print(mrzscanner.__version__)"
   ```

3. バージョン番号が表示されれば、インストールは成功です。

## GitHub を使用したインストール

:::tip
GitHub を使用してインストールする場合、事前に `Capybara` がインストールされていることを確認してください。

インストールされていない場合は、[**Capybara インストールガイド**](../capybara/installation.md) を参照してください。
:::

1. **リポジトリをクローン：**

   ```bash
   git clone https://github.com/DocsaidLab/MRZScanner.git
   ```

2. **プロジェクトディレクトリに移動：**

   ```bash
   cd MRZScanner
   ```

3. **依存パッケージをインストール：**

   ```bash
   pip install wheel
   ```

4. **パッケージをビルド：**

   ```bash
   python setup.py bdist_wheel
   ```

5. **ビルド済みパッケージをインストール：**

   ```bash
   pip install dist/mrzscanner_docsaid-*-py3-none-any.whl
   ```

これらの手順を実行すれば、`MRZScanner` を正しくインストールできます。

インストールが完了したら、以下のコマンドでテストしてください：

```bash
python -c "import mrzscanner; print(mrzscanner.__version__)"
# >>> 0.3.2
```

類似のバージョン番号（例：`0.3.2`）が表示されれば、インストールは成功です。
