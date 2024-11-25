---
sidebar_position: 2
---

# インストール

現在、Pypi でのインストールパッケージは提供されておらず、短期間での予定はありません。

このプロジェクトを使用するには、GitHub から直接プロジェクトをクローンし、依存パッケージをインストールする必要があります。

:::tip
インストール前に `DocsaidKit` がインストールされていることを確認してください。

もしまだ `DocsaidKit` をインストールしていない場合は、[**DocsaidKit インストールガイド**](../docsaidkit/installation) を参照してください。
:::

## インストール手順

1. **プロジェクトをクローン：**

   ```bash
   git clone https://github.com/DocsaidLab/WordCanvas.git
   ```

2. **プロジェクトディレクトリに移動：**

   ```bash
   cd WordCanvas
   ```

3. **依存パッケージをインストール：**

   ```bash
   pip install wheel
   ```

4. **パッケージファイルを作成：**

   ```bash
   python setup.py bdist_wheel
   ```

5. **パッケージファイルをインストール：**

   ```bash
   pip install dist/wordcanvas-*-py3-none-any.whl
   ```

これらの手順に従うことで、`WordCanvas` を正常にインストールできるはずです。

インストールが完了したら、プロジェクトを使用する準備が整います。

## インストールのテスト

以下のコマンドでインストールが成功したかどうかをテストできます：

```bash
python -c "import wordcanvas; print(wordcanvas.__version__)"
# >>> 0.4.2
```

もし `0.4.2` のようなバージョン番号が表示されれば、インストールは成功しています。
