---
slug: setting-up-pypiserver-on-ubuntu-with-docker
title: PyPiServerの設定記録
authors: Z. Yuan
tags: [docker, pypiserver]
image: /ja/img/2023/0913.webp
description: UbuntuでPyPiServerを設定する過程の記録。
---

今日は Docker を使って PyPi Server を立て、Ubuntu 上で実行する方法を記録します。

Ubuntu に Docker がインストールされており、基本的な Docker 操作に慣れている前提です。

<!-- truncate -->

## イメージをプルする

```bash
docker pull pypiserver/pypiserver:latest
```

## ディレクトリを作成する

手間を省くために、直接ホームディレクトリに Python パッケージを格納するためのディレクトリを作成します。

```bash
mkdir ~/packages
```

好きな名前に変更できますが、後の設定ファイルでも名前を変更する必要があります。

## htpasswd の設定

:::tip
パスワードを設定したくない場合、この手順をスキップできます。
:::

htpasswd は、ユーザー名とパスワードを格納するファイル形式で、pypiserver はこのファイルを使用してユーザーを認証します。

これは簡単で効果的な方法で、pypiserver のセキュリティを強化することができます。

まず、apache2-utils をインストールします：

```bash
sudo apt install apache2-utils
```

次に、以下のコマンドで新しい`.htpasswd`ファイルを作成します：

```bash
htpasswd -c ~/.htpasswd [username]
```

`username`のパスワードを入力するように求められます。パスワードを入力すると、`.htpasswd`ファイルがホームディレクトリに作成されます。

ファイルが作成された後は、上記の`docker run`コマンドを使って`pypiserver`を実行し、`.htpasswd`ファイルで認証を行います。

## バックグラウンドサービスとして実行する

Docker コンテナをバックグラウンドサービスとして実行するために、ここでは Docker Compose と Systemd を使用します。

### Docker Compose のインストール

まだ Docker Compose をインストールしていない場合、まずインストールします。インストール方法については以下のリンクを参考にしてください：

- [**Docker Compose の公式インストールガイド**](https://docs.docker.com/compose/install/)

注意点として、最近 Docker Compose に大きな更新があり、多くのコマンドが変更されました。最も顕著なのは、以前の`docker-compose`コマンドが、現在はすべて`docker compose`に変更されたことです。

公式ドキュメントに従って、インストール手順を以下にまとめました：

最新の Docker Compose をインストールします：

```bash
sudo apt update
sudo apt install docker-compose-plugin
```

Docker Compose が正しくインストールされたか確認するには：

```bash
docker compose version
```

### 設定ファイルの作成

適当な場所に`docker-compose.yml`を作成し、以下の内容を入力します：

```yaml {6-7}
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

- 上記で強調されている[ユーザー名]を実際のユーザー名に置き換えてください。
- 外部ポートのマッピング値を変更したい場合は、例えば`"18080:8080"`のように変更できます。

:::tip
`pypiserver`が提供しているサンプルを参考にすることもできます：[**docker-compose.yml**](https://github.com/pypiserver/pypiserver/blob/master/docker-compose.yml)
:::

パスワード設定を省略したい場合は、上記の`command`内のコマンドを以下のように変更します：

```yaml
command: run -a . -P . /data/packages --server wsgiref
```

### Systemd サービスの作成

設定ファイルを作成します：

```bash
sudo vim /etc/systemd/system/pypiserver.service
```

以下の内容を入力します：

```bash {7}
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

- `/path/to/your/docker-compose/directory`を`docker-compose.yml`の実際のパスに置き換えてください。ファイル名は不要です。
- Docker のパスが正しいことを確認してください。`which docker`コマンドで確認できます。
- 新しい`docker compose`コマンドを使用していることに注意してください。

### サービスの起動

Systemd に新しいサービス設定を再読み込みさせます：

```bash
sudo systemctl daemon-reload
```

サービスを起動します：

```bash
sudo systemctl enable pypiserver.service
sudo systemctl start pypiserver.service
```

## 状態の確認

サービスの現在の状態を確認したい場合は、以下のコマンドを実行します：

```bash
sudo systemctl status pypiserver.service
```

これにより、`pypiserver`サービスの現在の状態や実行中かどうか、最新のログ出力が表示されます。

<div align="center">
<figure style={{"width": "80%"}}>
![pypiserver status](./img/pypiserver.jpg)
</figure>
</div>

## 使い始める

これで、`pip`を使ってパッケージをインストール・アップロードできるようになりました。

### パッケージのアップロード

まず、`example_package-0.1-py3-none-any.whl`という名前のパッケージがあると仮定します。

次に、`twine`ツールを使ってパッケージをアップロードします：

```bash
pip install twine
twine upload --repository-url http://localhost:8080/ example_package-0.1-py3-none-any.whl
```

- `localhost:8080`はあなたの pypiserver サービスのアドレスとポートである必要があります。

### パッケージのダウンロードとインストール

`pip`を使ってパッケージをインストールする際、`pypiserver`サービスのアドレスとポートを指定する必要があります：

```bash
pip install --index-url http://localhost:8080/ example_package
```

### 基本認証の使用

pypiserver に基本認証を設定した場合、アップロードやダウンロード時に認証情報を提供する必要があります：

- パッケージをアップロードする場合：

  ```bash
  twine upload \
    --repository-url http://localhost:8080/ \
    --username [username] \
    --password [password] \
    example_package-0.1-py3-none-any.whl
  ```

- パッケージをインストールする場合：

  ```bash
  pip install \
    --index-url http://[username]:[password]@localhost:8080/ \
    example_package
  ```

## `pip.conf`の設定

このサーバーから頻繁にパッケージをインストールする場合、毎回`pip install`で`--index-url`を指定したくありません。

そのため、関連する設定情報を`pip.conf`に書き込むことができます。

### 設定ファイル

`pip.conf`ファイルは以下の場所に存在する場合があります。優先順位に従って検索します：

- 優先順位 1: サイトレベルの設定ファイル：

  - `/home/[ユーザー名]/.pyenv/versions/3.8.18/envs/main/pip.conf`

- 優先順位 2: ユーザーレベルの設定ファイル：

  - `/home/[ユーザー名]/.pip/pip.conf`
  - `/home/[ユーザー名]/.config/pip/pip.conf`

- 優先順位 3: グローバルレベルの設定ファイル：

  - `/etc/pip.conf`
  - `/etc/xdg/pip/pip.conf`

現在の Python 環境がどのファイルを使用するかを確認し、そのファイルに以下の内容を追加します：

```bash
[global]
index-url = http://[サービス提供者のIPアドレス]:8080/
trusted-host = [サービス提供者のIPアドレス]
```

再度、`[サービス提供者のIPアドレス]:8080`を正しい`pypiserver`のアドレスとポートに置き換えてください。

設定が完了した後、`pip install [package_name]`を実行すると、`pip.conf`で設定したサーバーアドレスが自動的に使用されます。

## 結語

これで、PyPiServer を構築し、パッケージのアップロードとダウンロードを行う方法を学びました。

この記事があなたの問題解決に役立ったことを願っています。
