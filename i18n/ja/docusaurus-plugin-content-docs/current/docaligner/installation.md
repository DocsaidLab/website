---
sidebar_position: 2
---

# インストール

現在、Pypi 上でのインストールパッケージは提供されておらず、短期間での提供予定もありません。

このプロジェクトを使用するには、Github から直接プロジェクトをクローンし、依存パッケージをインストールする必要があります。

:::tip
インストール前に `DocsaidKit` がインストールされていることを確認してください。

もしまだ `DocsaidKit` をインストールしていない場合は、[**DocsaidKit インストールガイド**](../docsaidkit/installation) を参照してください。
:::

## インストール手順

1. **プロジェクトをクローンする：**

   ```bash
   git clone https://github.com/DocsaidLab/DocAligner.git
   ```

2. **プロジェクトディレクトリに移動する：**

   ```bash
   cd DocAligner
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
   pip install dist/docaligner-*-py3-none-any.whl
   ```

これらの手順に従うことで、`DocAligner` のインストールが正常に完了するはずです。

インストールが完了したら、本プロジェクトを使用できます。

## インストールのテスト

以下のコマンドを使用して、インストールが成功したかどうかをテストできます：

```bash
python -c "import docaligner; print(docaligner.__version__)"
# >>> 0.5.0
```

もし `0.5.0` のようなバージョン番号が表示されれば、インストールは成功です。
