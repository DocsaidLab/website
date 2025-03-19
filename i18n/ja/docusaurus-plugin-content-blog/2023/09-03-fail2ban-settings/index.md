---
slug: fail2ban-settings
title: Fail2ban：SSHサービスの保護
authors: Z. Yuan
tags: [ubuntu, fail2ban]
image: /ja/img/2023/0903.webp
description: 悪意のある攻撃を外部からブロックする方法。
---

外部のSSH接続を開放した後、すぐに悪意のある接続がたくさん現れ、あなたのホストにログインしようと試みることに気づくでしょう。

<!-- truncate -->

<div align="center">
<figure style={{"width": "40%"}}>
![attack from ssh](./img/ban_1.jpg)
</figure>
<figcaption>悪意のある攻撃の例</figcaption>
</div>

---

一般的な対策として、Fail2banを使用してホストを保護します。Fail2banは、サーバーがブルートフォース攻撃を受けるのを防ぐためのソフトウェアです。

システムが疑わしい動作（例：繰り返しのログイン失敗）を検出すると、Fail2banは自動的にファイアウォールのルールを変更して攻撃者のIPアドレスをブロックします。

## 1. Fail2banのインストール

ほとんどのLinuxディストリビューションでは、パッケージ管理ツールを使用してFail2banをインストールできます。

ここでは、私のホストがUbuntuであるため、`apt`を使用してインストールします：

```bash
sudo apt install fail2ban
```

## 2. 設定

設定ファイルは`/etc/fail2ban/jail.conf`にあります。

でもちょっと待ってください！

このファイルを直接変更するのではなく、`jail.local`にコピーしてから変更しましょう：

```bash
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
```

`jail.local`を編集します：

```bash
sudo vim /etc/fail2ban/jail.local
```

このファイルにはいくつかの重要な設定パラメータがあります。対応する機能は以下の通りです：

- **ignoreip:** 無視するIPアドレスまたはネットワーク範囲（例：127.0.0.1/8）
- **bantime:** ブロック時間（秒単位）（デフォルトは600秒）
- **findtime:** この時間内で失敗した試行回数（デフォルトは600秒）
- **maxretry:** `findtime`で指定された時間内に許可される最大失敗試行回数

## 3. 起動と監視

Fail2banを起動します：

```bash
sudo service fail2ban start
```

Fail2banの状態を確認します：

```bash
sudo fail2ban-client status
```

## 4. カスタムルールの追加

特定のサービスに対して特別なルールを設定したい場合は、`jail.local`に対応するセクションを追加または変更できます。例えば、SSHの設定は次のようになります：

```bash
[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
```

## 5. テスト

設定を変更した後、Fail2banを再起動して変更を適用します：

```bash
sudo service fail2ban restart
```

次に、別のマシンまたは異なるIPアドレスを使用してテストを行い、何度も失敗したログインを試して、ブロックされるかどうか確認します。

## 6. チェック

ログファイルを定期的に確認し、ルールを更新して、最適な保護を維持するようにしましょう。

```bash
sudo fail2ban-client status sshd
```

## 7. アンブロック

テスト中に自分のIPがブロックされた場合は、次のコマンドでテストしたIPをアンブロックします：

```bash
sudo fail2ban-client set sshd unbanip <IPアドレス>
```

## 結語

このプロセスは手順が多いですが、複雑ではありません。

この記事が設定の完了に役立つことを願っています。