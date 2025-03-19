---
slug: setting-up-nextcloud
title: Nextcloud の設定記録
authors: Z. Yuan
tags: [Nextcloud, Docker]
image: /ja/img/2024/0304.webp
description: Ubuntu 22.04 上で Nextcloud をセットアップする過程を記録します。
---

以前はファイルを Google Drive に保存していて、ファイルをダウンロードする際は wget コマンドを使っていました。

しかし、ある日、Google が少し更新を行い、元々のダウンロードコマンドが突然使えなくなったのです……

本当に困りました。

そこで、Nextcloud を試してみることにしました。以下は Ubuntu 22.04 に基づいて行った設定手順です。

<!-- truncate -->

:::tip
開始する前に、ドメイン名を準備し、そのドメインを自分のサーバーに向けてください。

やり方がわからない場合は、ChatGPT に聞いてみてください。教えてくれますよ。
:::

## Nextcloud のインストール

**第一の質問：なぜ Nextcloud を使うのか？**

- プライベートクラウドが欲しくて、他人のサーバーにファイルを置きたくないからです。

**第二の質問：Nextcloud と Owncloud の違いは何か？**

- Nextcloud は Owncloud の開発者が分派して作ったもので、機能的にはほぼ同じですが、Nextcloud の方が開発速度が速いです。

**第三の質問：Nextcloud はどうやってインストールするのか？**

- これは少し複雑な問題です。というのも、Nextcloud のインストール方法は多数あり、それぞれに異なる利点と欠点があります。
- この記事で私が唯一お勧めするインストール方法は、Docker を使うことです。

## Nextcloud All-in-One の設定

- 公式ドキュメントを参照：[**Nextcloud All-in-One**](https://github.com/nextcloud/all-in-one)

**まず、Docker と Docker Compose をインストールしていることを確認してください。**

次に、NextCloud 用のフォルダを作成し、Docker Compose の設定ファイル `docker-compose.yml` を書きます：

```bash
mkdir nextcloud
vim nextcloud/docker-compose.yml
```

以下の内容を `docker-compose.yml` に貼り付けてください：

```yaml
services:
  nextcloud-aio-mastercontainer:
    image: nextcloud/all-in-one:latest  # 使用する Docker イメージを指定
    init: true  # ゾンビプロセスを防止、必ずこのオプションは有効にしてください
    restart: always  # コンテナ再起動ポリシーを Docker デーモンで自動的に再起動する設定
    container_name: nextcloud-aio-mastercontainer  # コンテナ名を設定、変更しないでください。更新の問題を避けるため
    volumes:
      - nextcloud_aio_mastercontainer:/mnt/docker-aio-config  # mastercontainer のファイル保存場所、この設定は変更不可
      - /var/run/docker.sock:/var/run/docker.sock:ro  # Docker ソケットをマウントし、他のコンテナや機能を管理。Windows/macOS や rootless モードの場合は調整が必要
    ports:
      - 80:80    # AIO インターフェースを介して有効な証明書を取得するために使用、オプション
      - 8080:8080  # デフォルトで AIO インターフェース（自己署名証明書）を提供、ホスト機の 8080 番ポートが使用中の場合、他のポートに変更可能（例：8081:8080）
      - 8443:8443  # 公開する場合、このポートで AIO インターフェースにアクセスして有効な証明書を取得、オプション

volumes:
  nextcloud_aio_mastercontainer:
    name: nextcloud_aio_mastercontainer  # Docker ボリュームの名前、この設定は変更不可
```

より詳細な設定情報については公式ドキュメントを参照してください：[**compose.yaml**](https://github.com/nextcloud/all-in-one/blob/main/compose.yaml)

## システムサービスの設定

上記の設定が完了したら、次はシステムサービスの設定です。

```bash
sudo vim /etc/systemd/system/nextcloud.service
```

以下の内容を貼り付けてください：

```bash {7}
[Unit]
Description=NextCloud Docker Compose
Requires=docker.service
After=docker.service

[Service]
WorkingDirectory=/home/[YourName]/nextcloud
ExecStart=/usr/bin/docker compose up --remove-orphans
ExecStop=/usr/bin/docker compose down
Restart=always

[Install]
WantedBy=multi-user.target
```

`[YourName]` を自分のユーザー名に置き換えることを忘れないでください。

## Nextcloud の起動

```bash
sudo systemctl enable nextcloud
sudo systemctl start nextcloud
```

## Nextcloud の設定

1. **Nextcloud AIO インターフェースへのアクセス**：

   初回起動が完了したら、`https://ip.address.of.this.server:8080` を使用して Nextcloud AIO インターフェースにアクセスしてください。`ip.address.of.this.server` は Nextcloud サービスを展開したサーバーの IP アドレスに置き換えてください。Docker と Nextcloud AIO が正しくインストールされ、起動していることを確認してください。初回起動には数分かかる場合があります。

   8080 ポートにアクセスするには、ドメイン名ではなく IP アドレスを使用することをお勧めします。HTTP Strict Transport Security（HSTS）がドメイン名でのアクセスを制限する可能性があるためです。HSTS は、ブラウザーに対して HTTPS 経由でのみウェブサイトにアクセスすることを要求するセキュリティ機能です。

2. **自己署名証明書の使用**：

   8080 ポートを介してアクセスすると、システムは自己署名証明書を使用して通信の安全を確保します。

   この証明書は信頼された証明機関（CA）によって発行されたものではないため、ブラウザーは信頼できないという警告を表示する可能性があります。その場合は、ブラウザーの指示に従い手動で承認してください。自己署名証明書は、テスト環境でのみ使用することをお勧めします。正式な運用環境では適していません。

3. **有効な証明書を取得するための自動化方法**：

   ファイアウォールやルーターが 80 番および 8443 番ポートを開放している、または正しく転送している場合、そしてドメイン名をサーバーに向けている場合、`https://your-domain-that-points-to-this-server.tld:8443` を使用して、信頼された証明機関（例：Let's Encrypt）によって発行された有効な証明書を自動的に取得できます。これにより、セキュリティと利便性が向上します。

   `your-domain-that-points-to-this-server.tld` は正しいドメイン名に置き換え、DNS 設定が有効になっていることを確認してください。また、ファイアウォール設定をチェックし、接続がブロックされていないことを確認してください。

4. **Nextcloud Talk のポート開放**：

   Nextcloud Talk（ビデオ通話やメッセージ機能）が正常に動作するためには、ファイアウォールやルーターで Talk コンテナの 3478/TCP および 3478/UDP ポートを開放する必要があります。

   NAT 環境にいる場合は、対応するポート転送が正しく設定されていることを確認し、ISP が関連する UDP ポートをブロックしていないかをチェックしてください。

## よくある質問

1. **家庭用ネットワークは動的 IP で、どうやってドメインに指すか？**

    No-IP などの動的 DNS ソリューションを試すほか、最も速くて安定した方法として、直接中華電信に固定 IP を申し込むことをお勧めします。

    :::tip
    台湾以外の国に住んでいる場合は、あなたの国の通信事業者に同様のサービスが提供されているかどうかを問い合わせてみてください。
    :::

2. **Docker を使いたくない、他に方法はあるか？**

    あります。Nextcloud を直接インストールすることができますが、その場合、すべての依存関係や環境設定を手動で行う必要があり、途中で多くの問題に直面することがあります。

    何度もトラブルに見舞われた後、最終的に Docker の方法に戻ってきたので、最初から Docker を使えばよかったと感じています。

3. **セットアップしたのに接続できないのはなぜか？**

    まず、ファイアウォールの設定が必要な接続を許可しているか確認してください。ファイアウォール設定に問題がない場合、ルーターのポート転送設定に問題がある可能性があります。Docker のログを確認して、詳細なエラー情報を取得してください。

## 最後に

設定 URL にアクセスすると、バックエンドのさらに奥にある設定インターフェースに入ります。

<div align="center">
<figure style={{"width": "70%"}}>
![login_1](./img/login_1.jpg)
</figure>
</div>

ここまで進むと、驚くことに気づくかもしれません：

- **パスワードがない！**

初回ログイン時にシステムがパスワードを生成しますが、通常は見逃しがちです。

もし忘れてしまっても心配無用です。以下のコマンドでパスワードを確認できます：

```bash
sudo grep password /var/lib/docker/volumes/nextcloud_aio_mastercontainer/_data/data/configuration.json
```

ログイン後、次のような設定画面が表示されます：

<div align="center">
<figure style={{"width": "70%"}}>
![login_2](./img/login_2.jpg)
</figure>
</div>

この画面が表示されたら、設定が正常に完了したことを意味します。

初回ログインの場合、準備していたドメイン名を入力してください。その後、システムが必要な Docker イメージをダウンロードし、再起動を自動的に行います。起動が完了したら、Nextcloud を使い始めることができます。パスワードの変更とその他のセキュリティ設定の確認を早急に行うことをお勧めします。

## 終わりに

上記の手順が完了したら、ブラウザのアドレスバーにあなたのドメインを入力してください。そうすると、美しいインターフェースが表示され、これがあなたのプライベートクラウドです。

<div align="center">
<figure style={{"width": "70%"}}>
![login_3](./img/login_3.jpg)
</figure>
</div>

このインターフェースには多くの機能があり、ファイルの管理やファイルの共有をこのインターフェースを通じて行うことができます。

さらに、スマートフォンに Nextcloud のアプリをダウンロードすれば、スマートフォンから直接ファイルを管理することも可能です。

Nextcloud があれば、もう Google Drive の容量制限を心配する必要はありません。

:::tip
もしあなたのサーバーで他のサービスが稼働している場合、Nginx のリバースプロキシを使って、Nextcloud のドメインを転送することができます。

この部分は本章の内容を超えているので、また機会があればお話ししましょう。
:::