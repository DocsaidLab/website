---
sidebar_position: 2
---

# インストール

私たちは PyPI 経由でのインストール、または GitHub からプロジェクトをクローンしてインストールする方法を提供しています。

## PyPI からのインストール

1. `wordcanvas-docsaid` をインストールします：

   ```bash
   pip install wordcanvas-docsaid
   ```

2. インストールを検証します：

   ```bash
   python -c "import wordcanvas; print(wordcanvas.__version__)"
   ```

3. バージョン番号が表示されれば、インストールは成功です。

## GitHub からのインストール

:::tip
GitHub からインストールする場合は、事前に `Capybara` がインストールされていることを確認してください。

インストールされていない場合は、[**Capybara インストールガイド**](../capybara/installation.md) を参照してください。
:::

1. **プロジェクトをクローンします：**

   ```bash
   git clone https://github.com/DocsaidLab/WordCanvas.git
   ```

2. **プロジェクトディレクトリに移動します：**

   ```bash
   cd WordCanvas
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
   pip install dist/wordcanvas_docsaid-*-py3-none-any.whl
   ```

以上の手順に従えば、`WordCanvas` を正常にインストールできるはずです。

インストール後、以下のコマンドでインストールの成功をテストできます：

```bash
python -c "import wordcanvas; print(wordcanvas.__version__)"
# >>> 2.0.0
```

`2.0.0` のようなバージョン番号が表示されれば、インストールは成功です。
