---
sidebar_position: 5
---

# Nginx ログと監視

API を外部に公開した後、私たちはすぐに四方八方からの悪意のあるリクエストを受け取ることになります。

ログ記録を確認したことがあれば、さまざまな IP が異なるユーザー名とパスワードでサーバーへのログインを試みているのを簡単に見つけることができるでしょう。これは一般的なブルートフォース攻撃であり、攻撃者は自動化ツールを使ってログインを繰り返し、正しいパスワードを見つけるまで試行を続けます。

パスワードの複雑さを限界まで高めることはできますが、常にそれを気にするのも一つの手ではありません。

そこで、Fail2ban というツールを使い、これらの IP をすべてブロックします。

## Fail2ban とは？

Fail2ban は、サーバーログを監視し、規則に基づいて悪意のある IP をブロックするオープンソースの侵入防止ツールです。SSH、HTTP、その他のネットワークサービスに対するブルートフォース攻撃を防ぐのに効果的です。

Fail2ban の動作の基本は、「フィルター（filters）」と「ジャイル（jails）」の二つの概念に基づいています：

- **フィルター**：ログ内で疑わしい行動パターンを検出するために定義されたもの
- **ジャイル**：フィルターとブロックメカニズムを組み合わせ、異常な動作を検出すると自動的に対応を実行します。

Nginx の適用シーンでは、Fail2ban は Nginx のアクセスログ（access log）やエラーログ（error log）を監視し、疑わしいリクエストを識別します。例えば：

- **頻繁な 404 エラー**（潜在的なスキャン攻撃）
- **連続するログイン失敗**（ブルートフォース攻撃）
- **短時間での高頻度リクエスト**（DDoS 攻撃や悪意のあるクローラー）

異常が検出されると、Fail2ban は設定に基づいて、iptables や nftables を通じて自動的に送信元 IP をブロックし、Nginx サーバーへの攻撃を防ぎます。

Ubuntu 環境では、Nginx のログはデフォルトで以下のパスに保存されます：

- **アクセスログ**：`/var/log/nginx/access.log`
- **エラーログ**：`/var/log/nginx/error.log`

Fail2ban はこれらのログを監視し、デフォルトまたはカスタムのフィルター規則テンプレートに基づいてログ内容を照合します。ログが悪意のある行動パターンに一致すると、Fail2ban は関連する IP をブロックリストに追加します。

:::info
規則テンプレートは通常、`/etc/fail2ban/filter.d/`にあります。
:::

:::tip
Nginx を保護するだけでなく、Fail2ban は SSH、FTP、メールサーバーなど、さまざまなサービスにも適用でき、広範囲なサーバーセキュリティ保護機能を提供します。このガイドは主に Nginx 関連の防御設定に焦点を当てています。
:::

# Fail2ban 基本設定

Ubuntu では、Fail2ban は APT パッケージマネージャを使って直接インストールできます。これは公式のソフトウェアリポジトリに含まれているため、以下の手順でインストールと起動を行います：

1. **システムパッケージリストを更新**：
   ```bash
   sudo apt update
   ```
2. **Fail2ban のインストール**：

   ```bash
   sudo apt install -y fail2ban
   ```

   これにより、Fail2ban とその依存パッケージが自動的にダウンロードされ、インストールされます。インストールが完了すると、Fail2ban は自動的に起動し、システム起動時に自動実行されるように設定されます。

3. **Fail2ban サービスの状態確認**：

   ```bash
   sudo systemctl status fail2ban
   ```

   インストールが成功すれば、`active (running)` と表示され、Fail2ban が正常に動作していることが確認できます。

4. **Fail2ban のバージョンと状態の確認**：

   - 現在インストールされている Fail2ban のバージョンを表示：

     ```bash
     fail2ban-client --version
     ```

   - 現在有効になっているジャイル（監獄）の確認：

     ```bash
     sudo fail2ban-client status
     ```

     デフォルトでは、Fail2ban は SSH の保護機能のみが有効になっており、Nginx 関連の防御には追加設定が必要です。

---

Fail2ban の主な設定ファイルは `/etc/fail2ban/jail.conf` ですが、公式では「このファイルを直接編集しないように」と推奨されています。これはソフトウェアの更新時に設定が上書きされるのを防ぐためです。

そのため、ローカル設定ファイル `jail.local` を作成し、デフォルトの値を上書きします。

1. **`jail.conf` を `jail.local` にコピー**：

   ```bash
   sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
   ```

2. **`jail.local` の編集**：

   ```bash
   sudo vim /etc/fail2ban/jail.local
   ```

`[DEFAULT]` セクションを探し、いくつかの重要なパラメータを設定します。これには、無視する IP、ブロック時間、観察時間、最大再試行回数などの設定が含まれます：

```ini
[DEFAULT]
ignoreip = 127.0.0.1/8 ::1    # 信頼するIPアドレスを設定。これらのIPはブロックされません
bantime  = 1h                 # デフォルトのブロック時間。秒数または時間単位（例：1h、10m）で設定できます
findtime = 10m                # この時間内に…
maxretry = 5                  # 最大5回の失敗でIPがブロックされます
```

これらの設定の意味は次の通りです：

- **ignoreip**：ブロックしない IP を指定します。例えば、ローカルホスト（`127.0.0.1`）や管理用 IP など。
- **bantime**：ブロックする時間の長さ。デフォルトでは 1 時間（`1h`）に設定されています。この時間内にその IP はアクセスできなくなります。
- **findtime**：失敗行動を観察する時間範囲。例えば、`10m`を設定すると、10 分間の間に発生した失敗行為がカウントされます。
- **maxretry**：最大再試行回数。例えば、`5`を設定すると、同一の IP が`findtime`内に 5 回失敗した場合、その IP がブロックされます。

これらのパラメータはグローバル設定であり、すべてのジャイルに適用されます。後ほど、Nginx 専用のブロックポリシーを設定し、悪意のある攻撃をより効果的にブロックする必要があります。

設定を完了したら、Fail2ban を再起動して新しい設定を適用します：

```bash
sudo systemctl restart fail2ban
```

これで、Fail2ban の基本的なインストールとグローバル設定が完了しました。

次に、Fail2ban をさらに設定して、Nginx のログを監視し、悪意のあるリクエストを防止する方法を紹介します。

## 一般的な攻撃タイプの監視

ここでは、いくつかの一般的な Nginx 攻撃タイプに対して、Fail2ban を使用して監視と自動ブロックを設定します。各攻撃行動は通常、Nginx のログに明確な特徴がありますので、まずは「フィルタ規則」を定義して検出し、対応するブロック戦略を適用します。

### 悪意のあるクローラーの防止

多くの悪意のあるクローラーや攻撃ツールは、特定の User-Agent を使用します。例えば：

- `Java`
- `Python-urllib`
- `Curl`
- `sqlmap`
- `Wget`
- `360Spider`
- `MJ12bot`

これらの User-Agent はほとんどが自動化ツールによるもので、通常のユーザーには属しません。そのため、これらの User-Agent に対してリアルタイムでブロックを行うことができます。

まず、`/etc/fail2ban/filter.d/`ディレクトリ内に新しいフィルタファイルを作成します：

```bash
sudo vim /etc/fail2ban/filter.d/nginx-badbots.conf
```

以下の内容を入力します：

```ini
[Definition]
failregex = <HOST> -.* "(GET|POST).*HTTP.*" .* "(?:Java|Python-urllib|Curl|sqlmap|Wget|360Spider|MJ12bot)"
ignoreregex =

# failregex は悪意のあるリクエストを検出するための正規表現です
# - <HOST> はIPアドレスを表し、Fail2Banがこのプレースホルダーを解釈します
# - "(GET|POST).*HTTP.*" はHTTP GETまたはPOSTリクエストに限定
# - .* はリクエスト内容をマッチさせるために使用
# - "(?:Java|Python-urllib|Curl|sqlmap|Wget|360Spider|MJ12bot)"
#   - これらは一般的なクローラー、スキャナー、攻撃ツール：
#     - Java: Javaベースのクローラーから来ることが多い
#     - Python-urllib: Pythonの標準URLリクエストライブラリ
#     - Curl: コマンドラインHTTPリクエストツール
#     - sqlmap: 自動SQLインジェクションツール
#     - Wget: ダウンロードツール、サイトクローリングにも使用
#     - 360Spider: 360検索エンジンのクローラー
#     - MJ12bot: Majestic SEOクローラー

# ignoreregex は特定のリクエストを除外するための正規表現
# - ここでは何も除外しないため空白にしています
```

次に、`jail.local`ファイルで Jail を設定します：

```ini
[nginx-badbots]
enabled  = true
port     = http,https
filter   = nginx-badbots
logpath  = /var/log/nginx/access.log
maxretry = 1
bantime  = 86400
```

これで、これらの User-Agent を使用したリクエストは、24 時間の間に即座にブロックされます。

### 404 スキャン攻撃の防止

この攻撃手法は存在しないページをブルートフォースで探し、短期間で大量のリクエストを行います。

攻撃者はスクリプトを使って存在しないページを繰り返しアクセスし、脆弱性や敏感なファイルを発見しようとすることがあり、その結果大量の HTTP 404 エラーが発生します。

新しいフィルターファイルを作成します：

```bash
sudo vim /etc/fail2ban/filter.d/nginx-404.conf
```

以下の内容を入力します：

```ini
[Definition]
failregex = <HOST> -.* "(GET|POST).*HTTP.*" 404
ignoreregex =

# failregex は特定のログパターンにマッチさせるための正規表現です
# - <HOST> はIPアドレス（Fail2Banはこのプレースホルダーを実際の送信元IPに置き換えます）
# - "(GET|POST).*HTTP.*"：
#   - GETまたはPOSTリクエストに限定
#   - .* はURLやHTTPプロトコルバージョン（例：HTTP/1.1）をマッチさせます
# - "404" はHTTPステータスコード404（リソースが見つからない）にマッチ

# このルールは、404エラーを頻繁にトリガーするIPをブロックします。
# 例えば、存在しないページをスキャンしている悪意のあるクローラーや攻撃者に対して効果的です
```

次に、`jail.local`ファイルに Jail を設定します：

```ini
[nginx-404]
enabled  = true
port     = http,https
filter   = nginx-404
logpath  = /var/log/nginx/access.log
findtime = 10m
maxretry = 5
bantime  = 86400
```

これにより、10 分以内に 5 回の 404 エラーをトリガーした IP が 24 時間ブロックされます。

### DDoS 攻撃の防止

短期間で大量のリクエストを送信する IP は、DDoS 攻撃や悪意のあるクローリングの可能性があります。

新しいフィルターファイルを作成します：

```bash
sudo vim /etc/fail2ban/filter.d/nginx-limitreq.conf
```

以下の内容を入力します：

```ini
[Definition]
failregex = limiting requests, excess: .* by zone .* client: <HOST>
ignoreregex =

# failregex はNginxのログでリクエスト制限（レートリミット）のイベントを検出するための正規表現です
# - "limiting requests, excess: .* by zone .* client: <HOST>"
#   - "limiting requests, excess:" は設定された制限（例：Nginxのlimit_req_zone制限）を超えたリクエストを示します
#   - .* は任意のデータ（リクエスト数や超過の詳細）を許容します
#   - "by zone .*" はNginx設定で指定されたリミットゾーン（zone）を表します
#   - "client: <HOST>" は実際のクライアントIPアドレスに置き換わります
```

この `failregex` は、Nginx で設定されたリクエスト制限（`limit_req_zone`）にトリガーされるログを検出するために使用されます。例えば、次のようなログが記録されます：

```log
2024/02/15 12:34:56 [error] 1234#5678: *90123 limiting requests, excess: 20.000 by zone "api_limit" client: 192.168.1.100, server: example.com, request: "GET /api/v1/data HTTP/1.1"
```

対応する Nginx 設定は次のようになります：

```nginx
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
server {
    location /api/ {
        limit_req zone=api_limit burst=20 nodelay;
    }
}
```

このように、IP が短時間で多くのリクエストを送信した場合、Nginx はログに「`limiting requests, excess: ... client: <IP>`」と記録し、Fail2Ban はこの `failregex` に基づいてその IP をブロックします。

次に、`jail.local`ファイルに Jail を設定します：

```ini
[nginx-limitreq]
enabled  = true
port     = http,https
filter   = nginx-limitreq
logpath  = /var/log/nginx/error.log
findtime = 10m
maxretry = 10
bantime  = 86400
```

これにより、10 分以内に 10 回以上 Nginx のレート制限をトリガーした IP が 24 時間ブロックされます。

### ブルートフォースログインの防止

ウェブサイトにログイン機能がある場合、攻撃者はブルートフォース攻撃を試みる可能性があります。

新しいフィルターファイルを作成します：

```bash
sudo vim /etc/fail2ban/filter.d/nginx-login.conf
```

以下の内容を入力します：

```ini
[Definition]
failregex = <HOST> -.* "(POST|GET) /(admin|wp-login.php) HTTP.*"
ignoreregex =

# failregex は悪意のあるリクエストをマッチさせ、管理者ページのログイン試行をターゲットにします
# - <HOST> はIPアドレス（Fail2Banがこれを実際の送信元IPに置き換えます）
# - "(POST|GET) /(admin|wp-login.php) HTTP.*"
#   - "(POST|GET)" はHTTPメソッド（攻撃者がGETまたはPOSTでログインを試みる可能性があります）
#   - "/admin" は多くのウェブサイトで使用される管理ページの一般的なパスです
#   - "/wp-login.php" はWordPressのログインページ
#   - "HTTP.*" は異なるHTTPバージョン（例：HTTP/1.1、HTTP/2）にマッチします
#
# ignoreregex は特定のリクエストを除外するためのパターンです
# - ここでは何も除外しないため、空白にしています
```

これにより、WordPress や一般的なウェブサイトの管理者ページへのログイン試行を検出できます。

次に、`jail.local`ファイルに Jail を設定します：

```ini
[nginx-login]
enabled  = true
port     = http,https
filter   = nginx-login
logpath  = /var/log/nginx/access.log
findtime = 5m
maxretry = 5
bantime  = 86400
```

これにより、5 分間に 5 回以上ログイン失敗をした IP が 24 時間ブロックされます。

### 敏感情報へのアクセス試行の防止

悪意のある攻撃者は、`/etc/passwd`、`/.git`、`/.env`などの敏感なパスにアクセスしようとする可能性があります。

新しいフィルターファイルを作成します：

```bash
sudo vim /etc/fail2ban/filter.d/nginx-sensitive.conf
```

以下の内容を入力します：

```ini
[Definition]
failregex = <HOST> -.* "(GET|POST) /(etc/passwd|\.git|\.env) HTTP.*"
ignoreregex =

# failregex は敏感なファイルにアクセスしようとする悪意のあるリクエストをマッチさせます
# - <HOST> はIPアドレス（Fail2Banがこれを実際の送信元IPに置き換えます）
# - "(GET|POST) /(etc/passwd|\.git|\.env) HTTP.*"
#   - "(GET|POST)" はHTTPメソッド（攻撃者はGETまたはPOSTでファイルを探す可能性があります）
#   - "/etc/passwd" はLinuxシステムのパスワードファイルで、攻撃者が読み取ろうとする可能性があります
#   - "/.git" はGitのリポジトリディレクトリで、攻撃者がソースコードや機密情報を探してダウンロードする可能性があります
#   - "/.env" は環境変数ファイルで、データベースのパスワードやAPIキーなどを含む場合があります
#   - "HTTP.*" は異なるHTTPバージョン（例：HTTP/1.1、HTTP/2）にマッチします
#
# ignoreregex は特定のリクエストを除外するためのパターンです
# - ここでは何も除外しないため、空白にしています
```

このルールは、ディレクトリトラバーサル攻撃、Git ディレクトリの保護、`.env`ファイルの探索防止、そして Web スキャナーによるスキャン防止に役立ちます。

次に、`jail.local`ファイルに Jail を設定します：

```ini
[nginx-sensitive]
enabled  = true
port     = http,https
filter   = nginx-sensitive
logpath  = /var/log/nginx/access.log
maxretry = 1
bantime  = 86400
```

これにより、これらの敏感なファイルにアクセスしようとした IP は、即座に 24 時間ブロックされます。

### API 攻撃への対応

:::tip
API のエンドポイントに合わせてフィルタを調整することを忘れないでください。
:::

API エンドポイント（例：`/api/login`、`/api/register`）に対するブルートフォース攻撃には、リクエスト頻度を制限する必要があります。

新しいフィルターファイルを作成します：

```bash
sudo vim /etc/fail2ban/filter.d/nginx-api.conf
```

以下の内容を入力します：

```ini
[Definition]
failregex = <HOST> -.* "(POST) /api/(login|register) HTTP.*"
ignoreregex =

# failregex はAPIのログインや登録に対する悪意のあるリクエストをマッチさせます
# - <HOST> はIPアドレス（Fail2Banがこれを実際の送信元IPに置き換えます）
# - "(POST) /api/(login|register) HTTP.*"
#   - "(POST)" はHTTP POSTメソッドに限定（ブルートフォース攻撃を防ぐ）
#   - "/api/login" はログインAPIで、ブルートフォース攻撃の対象になる可能性があります
#   - "/api/register" は登録APIで、攻撃者が自動化されたアカウント登録を試みる場合があります
#   - "HTTP.*" はHTTPバージョンにマッチ（例：HTTP/1.1、HTTP/2）
#
# ignoreregex は特定のリクエストを除外するためのパターンです
# - ここでは何も除外しないため、空白にしています
```

このルールは、ブルートフォースログイン、自動化されたアカウント登録攻撃を防ぎ、API のセキュリティを向上させます。

次に、`jail.local`ファイルに Jail を設定します：

```ini
[nginx-api]
enabled  = true
port     = http,https
filter   = nginx-api
logpath  = /var/log/nginx/access.log
findtime = 1m
maxretry = 10
bantime  = 86400
```

これにより、1 分以内に 10 回以上ログイン試行を行った IP が 24 時間ブロックされます。

### 新規ルールの適用

設定が完了したら、Fail2ban を再起動します：

```bash
sudo systemctl restart fail2ban
```

そして、Jail の状態を確認します：

```bash
sudo fail2ban-client status
```

特定の Jail の状態を確認したい場合は、次のコマンドを使用します：

```bash
sudo fail2ban-client status nginx-404  # 自分のJail名に置き換えてください
```

これで、Nginx に対する一般的な攻撃タイプに対して Fail2ban を設定し、サイトのセキュリティを強化しました。

## 防御効果のテスト

Fail2ban による Nginx の防御設定が完了した後、効果を簡単にテストできます。

### `curl` を使った攻撃のシミュレーション

`curl`を使用して悪意のあるリクエストを送信し、Fail2ban のブロック機能をトリガーできます。

テスト時には、`localhost`からリクエストを送らないでください。`127.0.0.1`は`ignoreip`に設定されている場合があり、Fail2ban がブロックしないからです。他のネットワーク環境のコンピュータやサーバーを使用するか、一時的に`ignoreip`設定を空にしてください。

- **1. 404 スキャンのブロックテスト**

  存在しないページをランダムにスキャンして攻撃者をシミュレートします：

  ```bash
  for i in $(seq 1 6); do
      curl -I http://<あなたのサーバーIPまたはドメイン>/nonexistentpage_$i ;
  done
  ```

  これにより、6 つの存在しないページに対して連続リクエストが送信され、HTTP 404 エラーが発生します。もし設定が「5 回の 404 エラーでブロック」となっている場合、Fail2ban は 5 回目または 6 回目のリクエスト時に IP をブロックします。
  ブロック後、再度`curl`でリクエストを送ると、接続失敗（タイムアウトや接続拒否）を確認できるはずです。

---

- **2. 敏感な URL のブロックテスト**

  攻撃者がシステムの敏感なファイル、例えば`/etc/passwd`にアクセスしようとすることがあります：

  ```bash
  curl -I http://<あなたのサーバー>/etc/passwd
  ```

  `nginx-sensitive`フィルターが正しく設定されていれば、このリクエストは Fail2ban によって検出され、IP は即座にブロックされるはずです（`maxretry = 1`の場合）。

---

- **3. User-Agent のブロックテスト**

  `sqlmap`のような悪意のあるクローラーをシミュレートします：

  ```bash
  curl -A "sqlmap/1.5.2#stable" http://<あなたのサーバー>/
  ```

  もし`sqlmap`が User-Agent に含まれている場合、Fail2ban は即座にその IP をブロックするはずです。

---

- **4. ブルートフォースログイン攻撃のテスト**

  WordPress やその他のログインページに対して複数回リクエストを送信し、ブルートフォース攻撃をシミュレートします：

  ```bash
  for i in $(seq 1 6); do
      curl -X POST -d "username=admin&password=wrongpassword" http://<あなたのサーバー>/wp-login.php ;
  done
  ```

  `nginx-sensitive`フィルターが「5 回の失敗でブロック」と設定されている場合、Fail2ban は 5 回目または 6 回目の失敗時に IP をブロックするはずです。

### ステータスとログの確認

以下のコマンドを使って、Jail のステータスを確認できます。例えば、`nginx-sensitive` Jail のステータスを確認する場合：

```bash
sudo fail2ban-client status nginx-sensitive
```

期待される出力：

```
Status for the jail: nginx-sensitive
|- Filter
|  |- Currently failed: 0
|  |- Total failed: 9
|  `- File list: /var/log/nginx/access.log /var/log/nginx/error.log
`- Actions
   |- Currently banned: 1
   |- Total banned: 2
   `- Banned IP list: 203.0.113.45
```

- **Currently banned**：現在ブロックされている IP の数。
- **Total failed**：累積した悪意のあるリクエストの回数。
- **Banned IP list**：現在ブロックされている IP のリスト（例：`203.0.113.45`）。

---

次に、Fail2ban のログを確認して、ブロックの記録を確認できます：

```bash
sudo tail -n 20 /var/log/fail2ban.log
```

期待される出力例：

```
fail2ban.actions [INFO] Ban 203.0.113.45 on nginx-sensitive
```

これは、`203.0.113.45`が`nginx-sensitive`ルールによってブロックされたことを示します。

### IP の解除

テストや日常的な管理中に、特定の IP を手動でブロックまたは解除することが必要な場合があります。

特定の IP を手動でブロックするには、次のコマンドを実行します：

```bash
sudo fail2ban-client set nginx-sensitive banip 203.0.113.45
```

これにより、`203.0.113.45`が直ちにブロックされ、`nginx-sensitive` Jail のブロックリストに追加されます。

誤ってブロックされた IP を解除するには、次のコマンドを実行します：

```bash
sudo fail2ban-client set nginx-sensitive unbanip 203.0.113.45
```

このコマンドを実行後、`fail2ban-client status nginx-sensitive`を使って、IP がブロックリストから削除されているかを確認できます。

---

最後に、ファイアウォールのルール（iptables / nftables）が正しく設定されているかを確認できます。

Fail2ban は主に`iptables`（または`nftables`）を通じて IP をブロックするため、直接ファイアウォールルールを確認することもできます：

```bash
sudo iptables -L -n --line-numbers
```

ブロック規則が適用されていれば、次のような表示が確認できます：

```
Chain f2b-nginx-sensitive (1 references)
num  target     prot opt source               destination
1    REJECT     all  --  203.0.113.45          0.0.0.0/0   reject-with icmp-port-unreachable
```

これは、`203.0.113.45`がブロックされ、すべてのトラフィックが拒否されることを示しています。

## 結論

これで、Nginx 用の防御規則を設定し、基本的な Fail2ban 設定が完了しました。

この部分では、実際のニーズに応じてパラメータを調整してください。例えば、異なる攻撃タイプごとに異なる Jail を作成し、`maxretry`や`findtime`を設定すること、重要な IP を誤ってブロックしないように`ignoreip`のホワイトリストを慎重に設定することが必要です。

デプロイ後も、Fail2ban のログを定期的に監視し、正常に動作していることを確認して、悪意のある攻撃に対する効果的な防御を維持してください。
