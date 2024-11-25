---
sidebar_position: 2
---

# インストール

現在、Pypi にインストールパッケージは提供されておらず、短期間内にその予定もありません。

このプロジェクトを使用するには、GitHub から直接クローンして、依存パッケージをインストールする必要があります。

:::tip
インストール前に、`DocsaidKit` がインストールされていることを確認してください。

まだ `DocsaidKit` をインストールしていない場合は、[**DocsaidKit インストールガイド**](../docsaidkit/installation)を参照してください。
:::

## インストール手順

1. **プロジェクトをクローンする：**

   ```bash
   git clone https://github.com/DocsaidLab/MRZScanner.git
   ```

2. **プロジェクトディレクトリに移動する：**

   ```bash
   cd MRZScanner
   ```

3. **依存パッケージをインストールする：**

   ```bash
   pip install wheel
   ```

4. **パッケージファイルを作成する：**

   ```bash
   python setup.py bdist_wheel
   ```

5. **パッケージファイルをインストールする：**

   ```bash
   pip install dist/mrzscanner-*-py3-none-any.whl
   ```

これらの手順に従えば、`MRZScanner` のインストールが完了するはずです。

インストールが完了した後、プロジェクトを使用することができます。

## インストールテスト

インストールが成功したかどうかを確認するには、以下のコマンドを使用できます：

```bash
python -c "import mrzscanner; print(mrzscanner.__version__)"
# >>> 0.1.0
```

`0.1.0` のようなバージョン番号が表示されれば、インストールは成功しています。
