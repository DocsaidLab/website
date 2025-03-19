---
slug: setting-up-pypiserver-on-ubuntu-with-docker
title: PyPiServerのセットアップ記録
authors: Z. Yuan
tags: [docker, pypiserver]
image: /ja/img/2023/0913.webp
description: Ubuntu上でPyPiServerをセットアップする過程の記録。
---

最近、私のパッケージインストールファイルを保管するためにPyPi Serverを立てました。UbuntuシステムとDockerを使用して設定を行い、その過程をここに記録します。

:::tip
読者がすでにUbuntuにDockerをインストールしており、Dockerの基本操作に慣れていることを前提としています。もしそうでない場合は、関連する知識を補ってから進めてください。
:::

<!-- truncate -->

## イメージの取得

```bash
docker pull pypiserver/pypiserver:latest
```

## ディレクトリの作成

無駄にしないために、ホームディレクトリにPythonパッケージを保存するためのディレクトリを作成します。

```bash
mkdir ~/packages
```

好きな名前に変更しても構いませんが、その後の設定ファイルも変更する必要があります。

## htpasswdの設定

:::tip
もしパスワードを設定したくない場合、この手順はスキップできます。
:::

htpasswdはユーザー名とパスワードを保存するファイル形式で、pypiserverはこのファイルを使ってユーザー認証を行います。これは簡単で有効な方法であり、pypiserverのセキュリティを強化する手段となります。

まず、apache2-utilsをインストールします：

```bash
sudo apt install apache2-utils
```

次に、以下のコマンドを使って新しい`.htpasswd`ファイルを作成します：

```bash
htpasswd -c ~/.htpasswd [username]
```

このコマンドを実行すると、`username`のパスワードを入力するよう求められます。パスワードを入力後、`.htpasswd`ファイルがホームディレクトリに作成されます。ファイルが作成されたら、前述の`docker run`コマンドを使って`pypiserver`を起動し、`.htpasswd`ファイルで認証を行います。

## バックグラウンドサービスとしての設定

Dockerコンテナをバックグラウンドサービスとして実行するために、Docker ComposeとSystemdを解決策として使用します。

まだDocker Composeをインストールしていない場合は、インストールが必要です。以下を参照してください：

- [**Docker Composeのインストール公式ドキュメント**](https://docs.docker.com/compose/install/)

最近、Docker Composeは大規模なアップデートを行い、多くの使い方が以前と異なっています。最も顕著な変更は、従来の`docker-compose`コマンドが`docker compose`に変更されたことです。

最新のDocker Composeをインストールするには：

```bash
sudo apt update
sudo apt install docker-compose-plugin
```

Docker Composeが正しくインストールされているか確認します：

```bash
docker compose version
```

### 設定ファイルの作成

任意の場所に`docker-compose.yml`を作成し、以下の内容を入力します：

```yaml {6-7} title="docker-compose.yml"
version: "3.3"
services:
  pypiserver:
    image: pypiserver/pypiserver:latest
    volumes:
      - /home/[ユーザー名]/auth:/data/auth
      - /home/[ユーザー名]/packages:/data/packages
    command: run -P /data/auth/.htpasswd -a update,download,list /data/packages
    ports:
      - "8080:8080"
```

- 上記の`[ユーザー名]`を実際のユーザー名に置き換えてください。
- 外部のポートマッピングを変更することもできます。例えば、`"18080:8080"`に変更できます。

:::tip
`pypiserver`が提供するテンプレートを参考にできます：[**docker-compose.yml**](https://github.com/pypiserver/pypiserver/blob/master/docker-compose.yml)
:::

パスワードを設定したくない場合、上記の`command`を以下のように変更します：

```yaml
command: run -a . -P . /data/packages --server wsgiref
```

### Systemdサービスの作成

設定ファイルを作成します：

```bash
sudo vim /etc/systemd/system/pypiserver.service
```

以下の内容を記入します：

```bash {7} title="/etc/systemd/system/pypiserver.service"
[Unit]
Description=PypiServer Docker Compose
Requires=docker.service
After=docker.service

[Service]
WorkingDirectory=/path/to/your/docker-compose/directory
ExecStart=/usr/bin/docker compose up --remove-orphans
ExecStop=/usr/bin/docker compose down
Restart=always

[Install]
WantedBy=multi-user.target
```

- `/path/to/your/docker-compose/directory`を実際の`docker-compose.yml`のディレクトリに置き換えてください。ファイル名は不要です。
- Dockerのパスが正しいことを確認してください。`which docker`で確認できます。

### サービスの起動

systemdに新しいサービス設定を読み込ませます：

```bash
sudo systemctl daemon-reload
```

サービスを起動します：

```bash
sudo systemctl enable pypiserver.service
sudo systemctl start pypiserver.service
```

## サービスの状態確認

サービスの現在の状態を確認するには、以下のコマンドを使用します：

```bash
sudo systemctl status pypiserver.service
```

これにより、`pypiserver`サービスの現在の状態や、実行中かどうか、最新のログ出力が表示されます。

<div align="center">
<figure style={{"width": "80%"}}>
![pypiserver status](./img/pypiserver.jpg)
</figure>
</div>

## 使用開始

これで、`pip`を使ってパッケージのインストールとアップロードができるようになりました。

### パッケージのアップロード

例えば、`example_package-0.1-py3-none-any.whl`というパッケージがあると仮定します。次に`twine`ツールを使ってパッケージをアップロードします：

```bash
pip install twine
twine upload --repository-url http://localhost:8080/ example_package-0.1-py3-none-any.whl
```

- `localhost:8080`はあなたのpypiserverサービスのアドレスとポートであることを確認してください。

### パッケージのインストール

インストール時には、`pypiserver`サービスのアドレスとポートを指定する必要があります：

```bash
pip install --index-url http://localhost:8080/ example_package
```

### 基本認証の使用

もしpypiserverが基本認証を設定している場合、アップロードまたはダウンロード時に認証情報を提供する必要があります：

- パッケージをアップロード：

  ```bash
  twine upload \
    --repository-url http://localhost:8080/ \
    --username [username] \
    --password [password] \
    example_package-0.1-py3-none-any.whl
  ```

- パッケージをインストール：

  ```bash
  pip install \
    --index-url http://[username]:[password]@localhost:8080/ \
    example_package
  ```

## `pip.conf`の設定

もしこのサーバーから頻繁にパッケージをインストールする場合、毎回`pip install`時に`--index-url`を指定するのが面倒なら、`pip.conf`に設定情報を記入することを検討できます。

### 設定ファイル

`pip.conf`ファイルは複数の場所に配置できます。優先順位順に探してください：

- 優先順位 1: サイトレベルの設定ファイル：

  - `/home/[ユーザー名]/.pyenv/versions/3.x.x/envs/main/pip.conf`

- 優先順位 2: ユーザー単位の設定ファイル：

  - `/home/[ユーザー名]/.pip/pip.conf`
  - `/home/[ユーザー名]/.config/pip/pip.conf`

- 優先順位 3: グローバル設定ファイル：

  - `/etc/pip.conf`
  - `/etc/xdg/pip/pip.conf`

現在のPython環境がどのファイルを使用しているかを確認し、そのファイルに以下の内容を追加します：

```bash
[global]
index-url = http://[サービスのIP]:8080/
trusted-host = [サービスのIP]
```

再度、`[サービスのIP]:8080`を正しいpypiserverのアドレスとポートに置き換えてください。

設定後、`pip install [package_name]`を実行すると、システムは自動的に`pip.conf`に設定されたサーバーアドレスをパッケージのソースとして使用します。

## 結語

これで、自分のPyPIサーバーを立て、パッケージのアップロードとインストールができるようになりました。

この記事が役に立ち、問題解決に繋がることを願っています。