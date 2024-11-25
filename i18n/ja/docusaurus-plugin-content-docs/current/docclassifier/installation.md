---
sidebar_position: 2
---

# インストール

現在、Pypi 上でインストールパッケージは提供されておらず、短期間での予定もありません。

本プロジェクトを使用するには、GitHub から直接プロジェクトをクローンし、依存パッケージをインストールする必要があります。

:::tip
インストール前に `DocsaidKit` がインストールされていることを確認してください。

まだ `DocsaidKit` をインストールしていない場合は、[**DocsaidKit インストールガイド**](../docsaidkit/installation) を参照してください。
:::

## インストール手順

1. **プロジェクトをクローンする：**

   ```bash
   git clone https://github.com/DocsaidLab/DocClassifier.git
   ```

2. **プロジェクトディレクトリに移動する：**

   ```bash
   cd DocClassifier
   ```

3. **依存パッケージをインストールする：**

   ```bash
   pip install setuptools wheel
   ```

4. **パッケージファイルを作成する：**

   ```bash
   python setup.py bdist_wheel
   ```

5. **パッケージファイルをインストールする：**

   ```bash
   pip install dist/docclassifier-*-py3-none-any.whl
   ```

これらの手順に従うことで、`DocClassifier` のインストールが正常に完了するはずです。

インストールが完了したら、本プロジェクトを使用できます。

## インストールの確認

インストールが成功したかどうかを確認するために、次のコマンドを使用できます：

```bash
python -c "import docclassifier; print(docclassifier.__version__)"
# >>> 0.8.0
```

`0.8.0` のようなバージョン番号が表示されれば、インストールは成功です。
