---
sidebar_position: 3
---

# Nginx HTTPS の設定

現代のインターネット環境において、ウェブサイトのセキュリティは無視できない問題となっており、SSL/TLS 証明書はウェブサイトとユーザー間でのデータ送信を保護するための重要な技術です。

Let's Encrypt は、無料で自動化されたオープンな証明書発行機関（CA）であり、世界中の数百万のウェブサイトに暗号化保護を提供し、HTTPS の普及を容易にしました。

無料で提供されているので、ぜひ利用しない手はありませんよね？

そこで、Nginx サーバーに Let's Encrypt を使用して HTTPS を設定します。

この章で達成する目標は以下の通りです：

- **SSL 証明書の取得**：Let's Encrypt の無料証明書を使用してウェブサイトのトラフィックを暗号化する。
- **HTTPS 強制**：すべての HTTP リクエストを自動的に HTTPS にリダイレクト（301 永続的リダイレクト）する。
- **Nginx リバースプロキシ**：Nginx をリバースプロキシとして使用する環境で、フロントエンドの SSL が正常に動作することを確認する。
- **自動更新**：証明書の自動更新メカニズムを設定し、90 日間の有効期限を迎える証明書の期限切れを防ぐ。

:::tip
前の章で、Let's Encrypt のツールの利点と欠点について簡単に紹介しました。

もしよく知らない場合は、補足説明ファイルを参照してください：[**Let's Encrypt**](./supplementary/about-lets-encrypt.md)
:::

## Let's Encrypt のインストール

私たちは Let's Encrypt の ACME プロトコルを使用して SSL 証明書を取得します。

まず、Let's Encrypt が推奨する ACME クライアントである **Certbot** をインストールする必要があります。Certbot は Let's Encrypt と自動的にやり取りし、証明書の申請や更新を行うことができます。

Debian/Ubuntu の場合、パッケージリポジトリから直接インストールできます：

```bash
sudo apt update
sudo apt install -y certbot python3-certbot-nginx
```

上記のコマンドで、Certbot とその Nginx プラグインがインストールされ、Certbot が自動的に Nginx 設定を変更して証明書の展開を行えるようになります。

次のステップに進む前に、以下の準備を完了していることを確認してください：

- **ドメイン名を所有しており、その DNS 記録がサーバーの IP アドレスを指していること**。
- **ファイアウォールの 80 番と 443 番ポートが開放されており、HTTP/HTTPS トラフィックが検証とその後のアクセスのために通過できること**。

## Let's Encrypt 証明書の申請

Certbot があれば、Let's Encrypt に証明書を申請できます。

Let's Encrypt は自動化された ACME 検証メカニズムを採用しており、ドメインの所有権を証明する必要があります。一般的な検証方法は、HTTP 検証（Certbot が一時的なファイルをウェブサイトに配置して Let's Encrypt が検証する）または DNS 検証（複雑で、ワイルドカード証明書申請に使用）です。

ここでは、簡単な HTTP 検証を使用し、Nginx プラグインを使って Certbot が自動的に設定を行う方法を示します：

```bash
# Nginx を停止（webroot モードを使用する場合は必要なことがありますが、--nginx プラグインを使用する場合は通常手動で停止する必要はありません）
# sudo systemctl stop nginx

# Certbot を実行し、Nginx プラグインを使用して証明書を申請
sudo certbot --nginx -d your-domain.com -d www.your-domain.com
```

`your-domain.com` と `www.your-domain.com` を実際のドメイン名に置き換えてください。

実行後、Certbot は Let's Encrypt サーバーと通信します：

- **ドメインの検証**：Certbot は一時的に Nginx 設定を変更したり、一時的なサービスを開始したりして Let's Encrypt の検証リクエストに応答します。例えば、Let's Encrypt サーバーは `http://your-domain.com/.well-known/acme-challenge/` パスにある検証ファイルにアクセスしようとします。検証が成功すれば、そのドメインに対する管理権限が確認されます。
- **証明書の取得**：検証が通過すると、Let's Encrypt は SSL 証明書を発行します（完全な証明書チェーン fullchain.pem と秘密鍵 privkey.pem が含まれます）。Certbot はこれらのファイルをデフォルトのパス（通常は `/etc/letsencrypt/live/your-domain.com/`）に保存します。

実行中、Certbot は以下のことを尋ねる場合があります：

- サービス利用規約への同意と電子メールの提供（更新通知用）。
- HTTP トラフィックを自動的に HTTPS にリダイレクトするかどうか。**リダイレクトを選択することをお勧めします**。Certbot は自動的に 301 リダイレクトルールを設定し、ユーザーが HTTPS 経由でアクセスすることを保証します。

:::tip
`--nginx` プラグインを使用しない場合は、`certbot certonly --webroot` モードを使用して手動で証明書を取得し、Nginx 設定を自分で編集することもできます。しかし、`--nginx` を使用すれば手動で設定を行う必要がなくなります。
:::

## Nginx で HTTPS を有効にする設定

証明書の発行が完了したら、Nginx で HTTPS（SSL）を有効にし、先ほど取得した証明書を読み込む必要があります。

以下は設定例です（Certbot のデフォルトインストールパスを使用していると仮定）：

```nginx
# Nginx 設定ファイル（例: /etc/nginx/sites-available/your-domain.conf）に追加：

# 80 番ポートのサービス、すべてのリクエストを HTTPS にリダイレクト
server {
    listen 80;
    listen [::]:80;
    server_name your-domain.com www.your-domain.com;
    # HTTP リクエストを永続的に HTTPS にリダイレクト
    return 301 https://$host$request_uri;
}

# 443 番ポートのサービス、HTTPS を提供
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name your-domain.com www.your-domain.com;

    # SSL 証明書ファイルのパス（Certbot によって発行された）
    ssl_certificate      /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key  /etc/letsencrypt/live/your-domain.com/privkey.pem;

    # (以下はリバースプロキシ設定の例)
    location / {
        proxy_pass http://127.0.0.1:8000;  # リクエストをバックエンドアプリケーションに転送（例: ローカルポート 8000 で実行されている FastAPI）
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_intercept_errors on;
    }
}
```

各セクションの設定についての説明は以下の通りです：

- **80 番ポートの HTTP サービス（強制 HTTPS へのリダイレクト）**

  ```nginx
  server {
      listen 80;
      listen [::]:80;
      server_name your-domain.com www.your-domain.com;
      # HTTP リクエストを永続的に HTTPS にリダイレクト
      return 301 https://$host$request_uri;
  }
  ```

  **このブロックの役割：**

  1. **HTTP（80 番ポート）のリスニング**

     - `listen 80;`：Nginx が IPv4 で 80 番ポート（標準的な HTTP 接続）をリスンします。
     - `listen [::]:80;`：IPv6 接続も許可します。

  2. **`server_name` の設定**

     - `server_name your-domain.com www.your-domain.com;`
     - これにより、Nginx はこの設定が `your-domain.com` と `www.your-domain.com` に適用されることを認識します。
     - この設定は、www 付きのドメインと、www なしのドメインの両方のトラフィックを正しく処理します。

  3. **HTTP を HTTPS に強制的にリダイレクト**

     - `return 301 https://$host$request_uri;`
     - これにより、すべての HTTP リクエストが 301 永続的リダイレクトを使用して HTTPS に自動的にリダイレクトされます。
     - `$host` はリクエストされたホスト名（`your-domain.com` または `www.your-domain.com`）を示し、`$request_uri` はリクエストされた完全な URI（例：`/about`）を示します。

---

- **443 番ポートの HTTPS サービス**

  ```nginx
  server {
      listen 443 ssl http2;
      listen [::]:443 ssl http2;
      server_name your-domain.com www.your-domain.com;
  ```

  **このブロックの役割：**

  1. **HTTPS（443 番ポート）のリスニング**

     - `listen 443 ssl http2;`
     - `ssl`：SSL 暗号化を有効にします。
     - `http2`：HTTP/2 を有効にし、パフォーマンスを向上させ（接続遅延の減少）、より高速な通信を実現します。
     - `listen [::]:443 ssl http2;`：IPv6 での HTTPS を許可します。

  2. **`server_name` の設定**

     - HTTP 設定と同様に、`your-domain.com` と `www.your-domain.com` に適用されます。

---

- **SSL 証明書の設定**

  ```nginx
      # SSL 証明書ファイルのパス（Certbot によって発行された）
      ssl_certificate      /etc/letsencrypt/live/your-domain.com/fullchain.pem;
      ssl_certificate_key  /etc/letsencrypt/live/your-domain.com/privkey.pem;
  ```

  **このブロックの役割：**

  - これらは **Let's Encrypt** によって自動的に生成された証明書のパスです：
    - `fullchain.pem`：完全な証明書（中間 CA 証明書を含む）
    - `privkey.pem`：秘密鍵（SSL 暗号化に使用）

---

- **FastAPI へのリバースプロキシ**

  ```nginx
      location / {
          proxy_pass http://127.0.0.1:8000;  # リクエストをバックエンドアプリケーション（FastAPI）に転送
          proxy_http_version 1.1;
          proxy_set_header Host $host;
          proxy_set_header X-Real-IP $remote_addr;
          proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
          proxy_set_header X-Forwarded-Proto $scheme;

          # Nginx が FastAPI のエラーを処理する
          proxy_intercept_errors on;
      }
  ```

  **このブロックの役割：**

  - `proxy_pass http://127.0.0.1:8000;`

    - FastAPI が `127.0.0.1:8000` で実行されていると仮定しています（`uvicorn --host 127.0.0.1 --port 8000` で起動）。
    - `proxy_pass` を使用して、API リクエストをローカルの FastAPI アプリケーションに転送します。

  - **リクエストヘッダーのプロキシ**

    - `proxy_set_header Host $host;`：元のリクエストの Host ヘッダーを保持します。
    - `proxy_set_header X-Real-IP $remote_addr;`：実際の IP アドレスをバックエンドに送信します。
    - `proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;`：プロキシチェーン全体を記録します。
    - `proxy_set_header X-Forwarded-Proto $scheme;`：バックエンドが元のリクエストのプロトコル（HTTP/HTTPS）を認識できるようにします。

  - **エラーハンドリング**
    - `proxy_intercept_errors on;`
      - Nginx がバックエンドのエラー（例：502、503）をインターセプトします。
      - `error_page` を使ってカスタムエラーページを指定できます。

---

設定が完了したら、Nginx を再読み込みして新しい設定を適用します：

```bash
sudo nginx -t  # Nginx 設定の構文が正しいか確認
sudo systemctl reload nginx  # 設定変更を適用してサービスを再起動
```

## Nginx のリバースプロキシとバックエンドサービス

上記のアーキテクチャでは、Nginx は HTTPS エンドポイントとリバースプロキシ（Reverse Proxy）の役割を同時に担っています。

Nginx は TLS ハンドシェイクと暗号化・復号化を処理し、SSL の終了が Nginx 層で行われます。バックエンドアプリケーションは Nginx からの HTTP リクエストを処理するだけで、HTTPS をサポートする必要はありません。内部通信は軽量化され、追加の暗号化負担を避けることができます。

このアプローチにより、バックエンドアプリケーションの設定が簡素化され、HTTPS に関するメンテナンス作業が削減されます。

さらに、Nginx をリバースプロキシとして使用する場合、バックエンドアプリケーションはデフォルトでクライアントの実際の IP アドレスを取得できません。代わりに、Nginx サーバーの IP アドレスしか表示されません。

そのため、重要な HTTP ヘッダーをバックエンドアプリケーションに渡すために `proxy_set_header` を使用し、バックエンドがクライアント情報を正しく識別できるようにする必要があります。

重要なヘッダーには次のものがあります：

- **`X-Forwarded-For`** → クライアントの実際の IP アドレスを伝える
- **`Host`** → 元のリクエストのホスト名を保持
- **`X-Forwarded-Proto`** → 元のリクエストで使用されたプロトコル（`http` または `https`）を示す

Nginx の設定：

```nginx
proxy_set_header Host $host;
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
proxy_set_header X-Forwarded-Proto $scheme;
```

仮にバックエンドが Python で開発された FastAPI アプリケーションであれば、FastAPI はデフォルトではプロキシで渡されたヘッダーを解析しません。そのため、`proxy-headers` モードを有効にして、`X-Forwarded-For` などの情報を正しく識別する必要があります：

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --proxy-headers
```

または、FastAPI 内で `StarletteMiddleware.ProxyHeaders` を使用することもできます：

```python
from fastapi import FastAPI
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.proxy_headers import ProxyHeadersMiddleware

app = FastAPI()

# プロキシヘッダー解析を有効にする
app.add_middleware(ProxyHeadersMiddleware)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
```

これにより、バックエンドアプリケーションはクライアントの実際の IP アドレスやリクエストプロトコルなどの情報を取得でき、誤判定を防ぐことができます。

最後に、セキュリティを確保するために、FastAPI はローカルインターフェースのみをリッスンし、HTTP サービスを外部に公開しないようにするべきです。これにより、未承認のアクセスのリスクを減らすことができます。

ベストプラクティスは次の通りです：

- **FastAPI がローカルポートをリッスン**
  ```bash
  uvicorn app:app --host 127.0.0.1 --port 8000
  ```
- **Nginx の `proxy_pass` はローカルポートを指す**
  ```nginx
  location / {
      proxy_pass http://127.0.0.1:8000;
  }
  ```

これにより、外部ユーザーはバックエンドの FastAPI アプリケーションに直接アクセスできず、Nginx を介してアクセスすることになります。セキュリティが向上し、内部トラフィックは暗号化されない HTTP で維持され、HTTPS による追加の負担が減り、パフォーマンスが向上します。

## 証明書の自動更新設定

Let's Encrypt の証明書の有効期限は 90 日間であり、この設計はセキュリティを強化するとともに、ウェブサイト管理者に自動更新メカニズムの実装を促すためです。手動での更新は証明書が期限切れになる原因となり、ウェブサイトの HTTPS 接続に影響を及ぼす可能性があるため、Certbot の自動更新を使用して証明書が常に有効であることを確認することをお勧めします。

### Certbot の自動更新メカニズム

Certbot はシステムの初期化システム（init system）に応じて、異なる自動更新方法を使用します：

systemd システム（ほとんどの現代的な Linux バージョン）では、Certbot は systemd タイマーを使用して更新を管理し、cron タスクは実行しません。

systemd タイマーが有効になっているか確認するには、次のコマンドを実行します：

```bash
systemctl list-timers | grep certbot
```

有効になっている場合、次のような出力が表示されるはずです：

```
certbot.timer               2025-02-13 02:00:00 UTC   12h left
```

また、systemd タイマーの状態を手動で確認することもできます：

```bash
systemctl status certbot.timer
```

systemd は `certbot.timer` の設定に基づき、毎日 2 回証明書更新のチェックを実行します。証明書が有効期限の **30 日以内** に達する場合にのみ、実際の更新が行われます。

---

systemd を使用しないシステムでは、Certbot は cron タスクを使用します。デフォルトでは、`/etc/cron.d/certbot` に保存されています。

システム内の cron 設定は次のようになっている場合があります：

```
0 */12 * * * root test -x /usr/bin/certbot -a \! -d /run/systemd/system && perl -e 'sleep int(rand(43200))' && certbot -q renew --no-random-sleep-on-renew
```

これは 12 時間ごとに実行される設定です。システムが systemd を使用している場合、この cron タスクは実行されません。

また、サーバーが同時に複数の更新を実行しないように、最大 12 時間のランダム遅延を設けて、静かな更新が行われます。

### 自動更新が正常に機能しているか確認

**systemd タイマー** または **cron** を使用している場合でも、自動更新メカニズムを手動でテストできます：

```bash
sudo certbot renew --dry-run
```

エラーが表示されなければ、自動更新メカニズムは正常に動作しています。

インストールされている証明書を確認することもできます：

```bash
sudo certbot certificates
```

これにより、すべての証明書の有効期限と保存パスが表示され、証明書が期限切れでないことを確認できます。

### 証明書の更新後に自動的に反映させる

Let's Encrypt で証明書が更新された後、Nginx/Apache は新しい証明書を読み込むまで、古い証明書を使用し続けます。

新しい証明書が反映されるようにするには、次のコマンドを使用します：

```bash
sudo systemctl reload nginx  # または systemctl reload apache2
```

また、`certbot renew` コマンド後に `--post-hook "systemctl reload nginx"` を追加すると、Certbot が証明書を更新した後に自動的にサーバーを再読み込みすることができます。

### 証明書が正常に更新されたか確認

現在の証明書の有効期限を確認するには、次のコマンドを使用します：

```bash
sudo certbot certificates
```

これにより、Certbot が管理しているすべての証明書が表示され、各証明書の有効期限と保存場所を確認できます。証明書が更新された後、有効期限が更新されていることを確認してください。

## HTTPS 設定のテストと検証

HTTPS 設定が完了した後、次のテストを実行して、すべてが正常であり、セキュリティ標準に準拠していることを確認します：

### ブラウザテスト

ブラウザで `https://your-domain.com` を入力し、セキュリティロック 🔒 が表示されることを確認します。証明書情報をクリックし、発行者が **Let's Encrypt** であり、証明書が有効で、ドメインと一致していることを確認します。

`http://your-domain.com` を入力して、HTTPS に自動的にリダイレクトされることを確認します。

### HTTP から HTTPS へのリダイレクトの確認

`curl` を使用してテストします：

```bash
curl -I http://your-domain.com
```

`301 Moved Permanently` が返され、`Location` ヘッダーが `https://your-domain.com/...` を指していることを確認します。

### ウェブサイトの機能が正常か確認

バックエンド API / ウェブサイトが正常に動作していることを確認します。例えば：

- FastAPI の API は正しく応答していますか？
- サイト内の HTTPS リンクは有効ですか？
- `X-Forwarded-Proto` を考慮して HTTPS のリダイレクトが適切に処理されていますか？

### Nginx ログの確認

エラーログとアクセスログを確認して、異常がないか確認します：

```bash
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

TLS 接続エラーが発生した場合、古いデバイスが **TLS 1.2** をサポートしていない可能性があります。必要に応じて設定を調整します。

## 結論

かなりの作業を行ったようですね。

この章では、Nginx で HTTPS を有効にし、すべてのトラフィックが Let's Encrypt によって発行された SSL 証明書で暗号化されるようにし、HTTP から HTTPS への強制リダイレクトを設定して、より安全な接続環境を提供しました。

基本的な HTTPS 配置に加えて、Nginx をリバースプロキシとして使用し、FastAPI へのリクエストを安全に転送できるようにし、元のリクエスト情報（クライアント IP やリクエストプロトコル）を保持することで、バックエンドアプリケーションの整合性と追跡性を確保しました。

最後に、証明書の自動更新メカニズムを設定し、Let's Encrypt の証明書が自動的に更新され、更新後に自動的に反映されることを確保し、ウェブサイトの HTTPS 接続が常に有効であることを確認しました。

これでウェブサイトのセキュリティ基準は完了したでしょうか？

実はまだです。次の章では、HSTS、CSP などのセキュリティヘッダーの設定や、一般的なウェブ攻撃を防ぐ方法を学びます。
