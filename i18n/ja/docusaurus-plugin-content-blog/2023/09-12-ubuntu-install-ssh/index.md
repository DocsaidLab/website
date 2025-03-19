---
slug: ubuntu-install-ssh
title: UbuntuにSSHサーバーを設定する
authors: Z. Yuan
tags: [ubuntu, ssh]
image: /ja/img/2023/0912.webp
description: サーバー設定とパスワードなしログインの方法。
---

SSHは、ユーザーがリモートサーバーに安全にアクセスし、管理できるネットワークプロトコルです。

今回は、パスワードなしでログインするための詳細な手順を記録します。

<!-- truncate -->

## OpenSSHサーバーのインストール

ターミナルを開きます。

以下のコマンドを入力してOpenSSHサーバーをインストールします：

```bash
sudo apt update
sudo apt install openssh-server
```

## SSHサーバーの状態を確認

以下のコマンドを使用してSSHサーバーの状態を確認します：

```bash
sudo systemctl status ssh
```

「Active: active (running)」と表示されれば、SSHサーバーは正常に起動しています。

## SSHパスワードなしログインの設定：

### クライアントでSSHキー対を生成

ターミナルを開きます。

以下のコマンドを入力してキー対を生成します：

```bash
ssh-keygen
```

指示に従って操作します。通常、デフォルト設定で問題ありません。パスフレーズを尋ねられたら、Enterキーを押してパスワードなしでキー対を作成します。

### 公開鍵をサーバーにコピー

`ssh-copy-id`コマンドを使用して公開鍵をサーバーにコピーします。[username] と [server-ip] をあなたのサーバー情報に置き換えてください。

```bash
ssh-copy-id [username]@[server-ip]
```

例えば：

```bash
ssh-copy-id john@192.168.0.100
```

もしサーバーがデフォルトのSSHポート（22番）を変更している場合（例：2222番など）、-pオプションを使って指定します：

```bash
ssh-copy-id -p 2222 john@192.168.0.100
```

このコマンドを実行すると、サーバーのパスワードを入力するように求められます。

認証が成功すると、公開鍵がサーバーの`~/.ssh/authorized_keys`ファイルに追加されます。

### パスワードなしログインのテスト

SSHでサーバーに接続してみます：

```bash
ssh [username]@[server-ip]
```

すべてが正しく設定されていれば、パスワードなしでサーバーにログインできるはずです。

## パスワード認証の無効化

SSHキーが設定できたら、セキュリティを強化するために、パスワード認証を無効にすることができます。

これは、サーバーの`/etc/ssh/sshd_config`ファイルで設定できます：

```bash
sudo vim /etc/ssh/sshd_config
```

ファイル内の`PasswordAuthentication`オプションを探し、`no`に設定します。

これで設定が完了しました！SSHが安全にパスワードなしで使用できるようになりました。