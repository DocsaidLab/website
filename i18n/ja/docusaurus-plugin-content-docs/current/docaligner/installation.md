---
sidebar_position: 2
---

# インストール

PyPI 上でのインストールまたは GitHub から本プロジェクトをクローンしてインストールする方法を提供しています。

## PyPI を使用したインストール

1. `docaligner-docsaid` をインストール：

   ```bash
   pip install docaligner-docsaid
   ```

2. インストールを確認：

   ```bash
   python -c "import docaligner; print(docaligner.__version__)"
   ```

3. バージョン番号が表示されれば、インストールは成功です。

## GitHub を使用したインストール

:::tip
GitHub からインストールするには、`Capybara` がインストールされていることを確認してください。

もしインストールされていない場合は、[**Capybara インストールガイド**](../capybara/installation.md) を参照してください。
:::

1. **プロジェクトをクローン：**

   ```bash
   git clone https://github.com/DocsaidLab/DocAligner.git
   ```

2. **プロジェクトディレクトリに移動：**

   ```bash
   cd DocAligner
   ```

3. **依存パッケージをインストール：**

   ```bash
   pip install setuptools wheel
   ```

4. **パッケージファイルを作成：**

   ```bash
   python setup.py bdist_wheel
   ```

5. **パッケージファイルをインストール：**

   ```bash
   pip install dist/docaligner_docsaid-*-py3-none-any.whl
   ```

これらの手順に従うことで、`DocAligner` のインストールが成功するはずです。

インストールが成功したかどうかを次のコマンドで確認できます：

```bash
python -c "import docaligner; print(docaligner.__version__)"
# >>> 1.1.0
```

`1.1.0` のようなバージョン番号が表示されれば、インストールは成功です。
