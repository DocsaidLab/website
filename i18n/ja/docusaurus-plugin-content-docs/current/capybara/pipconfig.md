---
sidebar_position: 4
---

# PIP パラメータ設定

## 使用方法

Linux/macOS システムでは、以下のコマンドを使用してローカルおよびグローバル設定を管理できます：

```bash
python -m pip config [<file-option>] list
python -m pip config [<file-option>] [--editor <editor-path>] edit
```

ここで、`<file-option>` は以下のオプションが指定できます：

- `--global`：オペレーティングシステム全体の設定ファイルを指定します。
- `--user`：操作ユーザーの設定ファイルを指定します。
- `--site`：現在の仮想環境内の設定ファイルを指定します。

`--editor`パラメータは外部エディタのパスを指定することを可能にします。指定しない場合は、`VISUAL`または`EDITOR`環境変数に基づいてデフォルトのテキストエディタが使用されます。

例えば、Vim エディタを使ってグローバル設定ファイルを編集したい場合は、以下のコマンドを使用します：

```bash
python -m pip config --global --editor vim edit
```

## 優先順位

設定ファイルの優先順位は非常に重要です。以下は、マシンに存在する可能性のある設定ファイルのリストで、優先順位順に並べられています：

1. **サイトレベルのファイル**：
   - `/home/user/.pyenv/versions/3.x.x/envs/envs_name/pip.conf`
2. **ユーザーレベルのファイル**：
   - `/home/user/.config/pip/pip.conf`
   - `/home/user/.pip/pip.conf`
3. **グローバルレベルのファイル**：
   - `/etc/pip.conf`
   - `/etc/xdg/pip/pip.conf`

Python 環境内で、pip は上記の順序で設定ファイルを探して適用します。どの設定ファイルを変更しているのかを確認することが、追跡が難しいエラーを防ぐために役立ちます。

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

この設定ファイル内で、各パラメータの意味は以下の通りです：

- `index-url`：pip がパッケージをインストールする際に使用するデフォルトのソースを設定します。
- `trusted-host`：HTTPS を使用したセキュアな検証なしで信頼できるソースをリストします。これにより、SSL エラーを防ぐことができます。
- `extra-index-url`：依存パッケージを検索してインストールするための追加のソースアドレスを提供します。`index-url`とは異なり、必要なパッケージが`index-url`で指定されたソースに見つからない場合、pip は`extra-index-url`を検索します。

:::warning
複数のソースを使用する場合、すべてのソースは信頼できるものでなければなりません。なぜなら、インストールプロセス中に最適なバージョンがこれらのソースから選択されるためです。信頼されていないソースはセキュリティリスクを引き起こす可能性があります。
:::
