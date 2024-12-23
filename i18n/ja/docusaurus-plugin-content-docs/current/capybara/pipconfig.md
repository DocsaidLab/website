---
sidebar_position: 4
---

# PIP パラメータ設定

この章では、pip の設定メカニズムについて詳しく説明し、複数の Python 環境でパッケージの競合や権限の問題を避ける方法を紹介します。

## 使用方法

Linux/macOS システムでは、次のコマンドを使用してローカルとグローバル設定を管理できます：

```bash
python -m pip config [<file-option>] list
python -m pip config [<file-option>] [--editor <editor-path>] edit
```

ここで、`<file-option>` は次のオプションを指定できます：

- `--global`：システム全体の設定ファイルを指定します。
- `--user`：ユーザー単位の設定ファイルを指定します。
- `--site`：現在の仮想環境内の設定ファイルを指定します。

`--editor` パラメータを使用すると、外部エディタのパスを指定できます。このパラメータを指定しない場合、`VISUAL` または `EDITOR` 環境変数に基づいてデフォルトのテキストエディタが使用されます。

例えば、Vim エディタを使用してグローバル設定ファイルを編集したい場合、次のコマンドを使用します：

```bash
python -m pip config --global --editor vim edit
```

:::tip
Windows システムを使用している場合、設定ファイルは `%APPDATA%\pip\pip.ini` にあるか、`%HOMEPATH%\.pip\pip.ini` のようなパスで確認できます。公式ドキュメントを参照するか、`pip config list` コマンドを使用して実際の場所を確認してください。
:::

## 優先順位

設定ファイルの優先順位は非常に重要です。次は、あなたのマシンに存在する可能性のある設定ファイルを優先順位順にリストしたものです：

1. **サイトレベルのファイル**：
   - `/home/user/.pyenv/versions/3.x.x/envs/envs_name/pip.conf`
2. **ユーザーレベルのファイル**：
   - `/home/user/.config/pip/pip.conf`
   - `/home/user/.pip/pip.conf`
3. **グローバルレベルのファイル**：
   - `/etc/pip.conf`
   - `/etc/xdg/pip/pip.conf`

Python 環境では、pip はこの順序で設定ファイルを探し、適用します。

どの設定ファイルを編集しているのかを確認することは、追跡が難しいエラーを避けるために重要です。

## 設定ファイルの例

以下は設定ファイルの例です：

```ini
[global]
index-url = https://pypi.org/simple
trusted-host = pypi.org
               pypi.python.org
               files.pythonhosted.org
extra-index-url = https://pypi.anaconda.org/simple
```

この設定ファイルでは、各パラメータの意味は次の通りです：

- `index-url`：pip がパッケージをインストールする際に使用するデフォルトのリポジトリを設定します。
- `trusted-host`：HTTPS を使用して安全確認を行う必要のないホストをリストし、SSL エラーを防ぎます。
- `extra-index-url`：依存関係パッケージを検索およびインストールするための追加のソース URL を提供します。`index-url` と異なり、`index-url` で見つからないパッケージがある場合、pip は `extra-index-url` を参照して探します。

:::warning
複数のソースを使用する場合、すべてのソースが信頼できるものであるべきです。なぜなら、インストール過程でこれらのソースから最適なバージョンが選択されるからです。信頼されていないソースはセキュリティリスクを伴う可能性があります。
:::

:::tip
プライベートなパッケージサーバーを使用している場合や、認証のためにユーザー名とパスワードを指定する必要がある場合は、`pip.conf` にその情報を記載して自動化できますが、ファイルの権限を適切に管理して安全を確保してください。
:::
