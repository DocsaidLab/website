---
slug: ubuntu-install-ssh
title: Ubuntu で SSH サーバーを設定する
authors: Z. Yuan
tags: [ubuntu, ssh]
image: /ja/img/2023/0912.webp
description: サーバーの設定とパスワードなしログインの手順。
---

SSH はネットワークプロトコルの一つで、ユーザーがリモートサーバーに安全にアクセスし、管理できるようにします。

今回はパスワードなしログインの設定を行います。

<!-- truncate -->

## OpenSSH サーバーのインストール

ターミナルを開きます。

以下のコマンドを入力して、OpenSSH サーバーをインストールします：

```bash
sudo apt update
sudo apt install openssh-server
```

## SSH サーバーの状態を確認

以下のコマンドで SSH サーバーの状態を確認します：

```bash
sudo systemctl status ssh
```

「Active: active (running)」と表示されていれば、SSH サーバーが正常に起動しています。

## SSH パスワードなしログインの設定

### クライアントで SSH 鍵ペアを生成

ターミナルを開きます。

以下のコマンドを入力して鍵ペアを生成します：

```bash
ssh-keygen
```

画面の指示に従って操作します。通常はデフォルト設定で十分です。パスフレーズの入力を求められた場合、Enter を押すことでパスフレーズなしの鍵ペアを作成できます。

### 公開鍵をサーバーにコピー

`ssh-copy-id` コマンドを使用して公開鍵をサーバーにコピーします。[username] と [server-ip] を自分のサーバー情報に置き換えてください。

```bash
ssh-copy-id [username]@[server-ip]
```

例：

```bash
ssh-copy-id john@192.168.0.100
```

サーバーがデフォルトの SSH ポートを変更している場合（例：2222）、`-p` オプションを使います：

```bash
ssh-copy-id -p 2222 john@192.168.0.100
```

このコマンドでは、サーバーのパスワードを入力するよう求められます。

認証に成功すると、公開鍵がサーバーの `~/.ssh/authorized_keys` ファイルに追加されます。

### パスワードなしログインをテスト

サーバーに SSH 接続してみます：

```bash
ssh [username]@[server-ip]
```

設定が正しく行われていれば、パスワードなしでサーバーにログインできるはずです。

## パスワード認証を無効化

SSH 鍵を設定した後、セキュリティ向上のため、パスワード認証を無効化することを検討してください。

サーバーの `/etc/ssh/sshd_config` ファイルを編集します：

```bash
sudo vim /etc/ssh/sshd_config
```

ファイル内の `PasswordAuthentication` オプションを探し、それを `no` に設定します。

これで設定完了です！SSH を便利かつ安全にお楽しみください。
