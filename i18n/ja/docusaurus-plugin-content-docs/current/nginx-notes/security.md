---
sidebar_position: 4
---

# Nginx のセキュリティ強化

Nginx はその柔軟性と高いパフォーマンスにより、現代のネットワーク環境で広く使用されています。しかし、適切なセキュリティ設定がないと、せっかく構築したウェブサイトが攻撃者のターゲットになる可能性があります。

主なセキュリティリスクには次のようなものがあります：

- **強化されていない TLS 設定**：中間者攻撃や弱い暗号化の脆弱性を引き起こす可能性があります。
- **HSTS 機能がない**：ユーザーが HTTP 経由でサイトにアクセスしてしまう可能性があり、HTTPS のセキュリティが低下します。
- **適切な HTTP セキュリティヘッダーが設定されていない**：XSS やクリックジャッキングなどの攻撃に対して脆弱になります。
- **トラフィック制限や DDoS 防止機能がない**：サーバーリソースが乱用され、サービスが中断される可能性があります。

Nginx サーバーのセキュリティを確保するためには、設定にいくつかの強化措置を講じる必要があります。

## TLS セキュリティ強化

:::tip
この設定は `http` ブロック内に記述する必要があります。TLS は HTTP プロトコルのセキュリティ拡張であるため、以下の設定は `/etc/nginx/nginx.conf` のメイン設定ファイルに記述する必要があります。
:::

TLS（Transport Layer Security）は、ネットワーク通信の安全性を保護するための暗号化プロトコルです。

現代のネットワーク環境では、HTTPS による暗号化はウェブサイトの基本的な標準となっており、TLS は HTTPS のコア技術です。HTTPS はサイトと訪問者間のトラフィックを暗号化しますが、強化されていない TLS 設定は、次のような脆弱性を引き起こす可能性があります：

- **古い TLS プロトコル（TLS 1.0/1.1）の使用** → BEAST や POODLE などの攻撃に対して脆弱
- **弱い暗号化アルゴリズム（RC4、3DES など）の許可** → 解読される可能性がある
- **OCSP Stapling の欠如** → 証明書チェック時のパフォーマンス低下
- **セッションチケットの有効化** → 攻撃者が古い暗号化キーを再利用することでリプレイ攻撃が可能になる

サイトのセキュリティを確保するためには、Nginx を適切に設定して TLS 設定を強化し、MITM（中間者攻撃）、証明書の偽造、弱い暗号化攻撃などの潜在的なセキュリティリスクを防ぐ必要があります。

以下は考慮すべき設定のいくつかです：

1. **古い TLS を無効化し、TLS 1.2 と TLS 1.3 のみを許可**

   ```nginx
   ssl_protocols TLSv1.2 TLSv1.3;
   ```

   - **TLS 1.0 / 1.1 は安全ではない**。多くのブラウザ（Chrome、Firefox など）はすでにサポートを終了しています。
   - **TLS 1.2 はまだ安全で信頼性が高い**。広くサポートされています。
   - **TLS 1.3 はより高速で、セキュリティも向上**。ゼロ遅延ハンドシェイクをサポートし、現代のアプリケーションに適しています。

---

2. **安全な暗号化アルゴリズムの使用**

   ```nginx
   ssl_ciphers 'ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384';
   ssl_prefer_server_ciphers on;
   ```

   これらの暗号化スイートは、**RC4、3DES、MD5 などの弱い暗号化アルゴリズムを排除**し、サーバーが安全な暗号化セットを優先的に使用するように設定します。

---

3. **OCSP Stapling を有効にして、証明書検証の速度を向上**

   ```nginx
   ssl_stapling on;
   ssl_stapling_verify on;
   resolver 8.8.8.8 1.1.1.1 valid=300s;
   resolver_timeout 5s;
   ```

   OCSP Stapling により、サーバーは証明書の状態を事前にキャッシュし、CA サーバーへの照会の遅延を減らしてパフォーマンスを向上させます。

---

4. **セッションチケットを無効化し、キーの再利用を防ぐ**

   ```nginx
   ssl_session_tickets off;
   ```

   セッションチケットは、攻撃者がチケットを盗んで暗号化セッションを再利用することを防ぐため、無効化する必要があります。

---

5. **安全な DH パラメーターを設定**

   ```nginx
   ssl_dhparam /etc/nginx/ssl/dhparam.pem;
   ```

   Diffie-Hellman（DH）鍵交換方式は、追加のセキュリティを提供し、いくつかの攻撃を防ぎます。

   DH パラメーターファイルがない場合、OpenSSL を使って DH パラメーターを生成できます：

   ```bash
   sudo mkdir -p /etc/nginx/ssl
   sudo openssl dhparam -out /etc/nginx/ssl/dhparam.pem 2048
   ```

---

6. **セッションキャッシュを有効にして、TLS 接続のパフォーマンスを向上**

   ```nginx
   ssl_session_cache shared:SSL:10m;
   ssl_session_timeout 1h;
   ```

   - **TLS セッションの再利用を許可してパフォーマンスを向上**
   - **CPU の負担を減らし、ユーザー体験を改善**

## HSTS の高度な設定

:::tip
この設定は `server` ブロック内で行う必要があります。HSTS は特定のウェブサイトに対するセキュリティ機能であるため、以下の設定は `/etc/nginx/sites-available/example.com` のようなサイトの設定ファイルに記述する必要があります。
:::

HSTS（HTTP Strict Transport Security）は、ブラウザにウェブサイトへのアクセスを HTTPS のみで行うよう強制するセキュリティ機能です。

HTTPS はすでにウェブサイトの標準設定ですが、単純な HTTPS だけではダウングレード攻撃（Downgrade Attack）や中間者攻撃（MITM Attack）を防ぐことはできません。このような場合に、HSTS が重要な役割を果たします。HSTS は、ユーザーがサイトにアクセスするたびに自動的に HTTPS にリダイレクトさせ、不安全な HTTP 接続の攻撃を防ぎます。

HSTS は **RFC 6797** で定義されており、主な機能は以下の通りです：

- **ウェブサイトへのアクセスを HTTPS のみで強制**
- **ダウングレード攻撃を防止**
- **SSL ストリッピング攻撃を防止**
- **ウェブサイトの信頼性と SEO ランキングを向上させる**

例えば、あるウェブサイトが HTTPS をサポートしているが HSTS が有効でない場合、攻撃者は中間者攻撃を使って最初の HTTP 接続を改ざんし、ユーザーを強制的に HTTP にダウングレードさせ、すべてのデータ通信を傍受することができます。この時、すべてのログインパスワードやクレジットカード情報などの機密データが攻撃者に盗まれ、ユーザーのプライバシーが危険にさらされます。

HSTS を有効にすると：

1. 最初の接続が成功した後、ブラウザはそのサイトを HTTPS 接続専用として記録します。
2. それ以降のすべての接続では、ブラウザは自動的に HTTPS に転送され、ユーザーが `http://example.com` と入力しても自動的に `https://example.com` に変更されます。
3. 攻撃者が初期の HTTP 接続を傍受しても、サイトを HTTP にダウングレードすることはできません。

Nginx で HSTS を設定するには、HTTPS サーバーにのみ適用する必要があるため、`server` ブロック内に記述します。

```nginx
server {
    listen 443 ssl http2;
    server_name example.com www.example.com;

    # HSTS を有効にし、すべてのサブドメインにも適用
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
}
```

- 各パラメータの意味は次の通りです：

  - `max-age=31536000`：HSTS の有効期限（秒単位）。ここでは 1 年（31536000 秒）に設定しています。
  - `includeSubDomains`：すべてのサブドメインに対して HSTS を強制的に適用し、攻撃者によるサブドメインの悪用を防止します。
  - `preload`：ウェブサイトをブラウザの HSTS プリロードリストに追加することを許可します。
  - `always`：すべての応答に HSTS ヘッダーを追加することを保証します（エラーページも含む）。

---

設定時には、HSTS は HTTPS にのみ適用する必要があります。HTTP 接続が 301 リダイレクトで HTTPS に転送されるべきであり、HSTS ヘッダーは含まれないようにします。そのため、HTTP サーバーに次の設定を追加します：

```nginx
server {
    listen 80;
    server_name example.com www.example.com;

    # HTTPS に転送するが、HSTS は加えない
    return 301 https://$host$request_uri;
}
```

これにより、攻撃者が最初の HTTP 接続を利用して HSTS を回避することが防止され、ブラウザが正しく記録できるようになります。

## その他のセキュリティヘッダー

:::tip
この設定もヘッダーの部分に関するものですので、以下の設定は `/etc/nginx/sites-available/example.com` などのサイトの設定ファイルに記述する必要があります。
:::

ウェブサイトのセキュリティについて考えるとき、ほとんどの人は最初に HTTPS 暗号化やファイアウォールを思い浮かべますが、これらはトランスポート層のセキュリティの問題を解決するだけです。

実際、XSS（クロスサイトスクリプティング）、クリックジャッキング、MIME 偽装などの多くのウェブ攻撃はブラウザの脆弱性を利用して行われます。そのため、HTTP セキュリティヘッダー（Security Headers）を正しく設定することは、これらの攻撃を防ぐための最良の方法です。

HTTP セキュリティヘッダーは、以下の攻撃に対して有効です：

- **XSS（クロスサイトスクリプティング攻撃）**
- **クリックジャッキング（Clickjacking）**
- **MIME 偽装攻撃**
- **Cookie の盗難やハイジャック**

たとえウェブサイトが HTTPS を有効にしていても、これらのヘッダーを正しく設定しなければ、セキュリティが全面的に強化されることはありません。

---

Nginx の `server` ブロック内で、`add_header` ディレクティブを使用してこれらのヘッダーを追加できます。

```nginx
server {
    listen 443 ssl http2;
    server_name example.com www.example.com;

    # セキュリティ強化
    add_header X-Frame-Options "DENY" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Permissions-Policy "geolocation=(), microphone=()" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' https://trusted-cdn.com;" always;

    root /var/www/html;
    index index.html index.htm;
}
```

これらのセキュリティヘッダーの用途を順に見ていきましょう：

1. **X-Frame-Options：クリックジャッキング攻撃の防止**

   ```nginx
   add_header X-Frame-Options "DENY" always;
   ```

   サイトが `iframe` 内で埋め込まれるのを防ぎ、攻撃者が隠れたフレームを使ってユーザーを騙して悪意のあるボタンをクリックさせるのを防ぎます。

   オプションには `DENY`（サイトを埋め込むことを完全に禁止）または `SAMEORIGIN`（同一オリジンの `iframe` のみ許可）があります。サイトがフレーム内で動作する必要がない限り、`DENY` の使用を推奨します。

---

2. **X-XSS-Protection：XSS 攻撃の防止**

   ```nginx
   add_header X-XSS-Protection "1; mode=block" always;
   ```

   ブラウザ内蔵の XSS フィルターを有効にし、攻撃者がサイトに悪意のあるスクリプトを挿入するのを防ぎます。`1; mode=block` は XSS 攻撃が検出された場合、ブラウザがページの読み込みをブロックすることを意味します。

   CSP（Content Security Policy）はより強力な XSS 防御メカニズムですが、`X-XSS-Protection` は追加のセキュリティ対策として機能します。

---

3. **X-Content-Type-Options：MIME 型偽装攻撃の防止**

   ```nginx
   add_header X-Content-Type-Options "nosniff" always;
   ```

   ブラウザが不明な MIME タイプを推測するのを防ぎ、攻撃者が悪意のあるファイル（例えば、`.js` を `.jpg` と偽装する）をアップロードするのを防ぎます。すべてのダウンロードコンテンツは、サーバーが設定した MIME タイプで解析されることが保証され、脆弱性のリスクが減少します。

   これは OWASP 標準で推奨されている必須のセキュリティヘッダーです。

---

4. **Referrer-Policy：ユーザーのプライバシー保護**

   ```nginx
   add_header Referrer-Policy "strict-origin-when-cross-origin" always;
   ```

   `Referer` ヘッダーの送信方法を制御し、外部サイトに完全な URL 情報が漏れるのを防ぎます。`strict-origin-when-cross-origin` 設定では、同一オリジンのリクエストは完全な `Referer` を送信しますが、クロスオリジンのリクエストではオリジン（`origin`）のみが送信され、HTTP から HTTPS にダウングレードされるときには `Referer` を送信しません。

   これにより、サイトの分析データを保持しながら、プライバシーの漏洩リスクを低減できます。

---

5. **Permissions-Policy（Feature Policy）：ブラウザ機能の制限**

   ```nginx
   add_header Permissions-Policy "geolocation=(), microphone=()" always;
   ```

   サイトで使用可能なブラウザ機能を制限し、ユーザーの権限（例えば、カメラ、マイク、位置情報）を濫用されるのを防ぎます。上記の設定は、位置情報（geolocation）とマイク（microphone）のアクセスを禁止します。

   現代のプライバシー保護において非常に重要で、特に GDPR やプライバシーに関する法律が求めるウェブサイトに役立ちます。

---

6. **Content-Security-Policy（CSP）：最強の XSS 防御**

   ```nginx
   add_header Content-Security-Policy "default-src 'self'; script-src 'self' https://trusted-cdn.com;" always;
   ```

   CSP は XSS 攻撃に対する最強の防御機構であり、ウェブサイトで許可されたリソースのソースを制限できます。`default-src 'self'` は、デフォルトでサイト内のリソースのみを許可し、`script-src 'self' https://trusted-cdn.com;` は、指定された CDN とサイト内の JavaScript のみを許可します。

   CSP は XSS 攻撃を効果的に防ぎ、サイトのセキュリティを向上させます。

## メイン設定ファイルの設定

前の章では、メイン設定ファイルを見てきましたが、その時は飛ばしていました。今回は、再度その設定を詳しく見ていきます。通常、TLS 設定、HTTP セキュリティヘッダー、リクエストレート制限などのグローバル設定は `nginx.conf` に配置し、サイト固有の HSTS や HTTPS へのリダイレクトなどの設定は `sites-available/default` などのサイト設定ファイルに記述します。

前述のディスカッションに基づいて、デフォルトの `nginx.conf` 設定ファイルを私たちのニーズに合わせて少し変更できます。

```nginx title="/etc/nginx/nginx.conf"
user www-data;
worker_processes auto;
pid /run/nginx.pid;

# エラーログの設定
error_log /var/log/nginx/error.log warn;
include /etc/nginx/modules-enabled/*.conf;

events {
    worker_connections 1024;  # 最大同時接続数を増加（元は 768）
    multi_accept on;          # worker プロセスが複数の新しい接続を同時に受け入れることを許可
}

http {
    ##
    # 基本設定
    ##
    sendfile on;
    tcp_nopush on;
    types_hash_max_size 2048;
    server_tokens off;  # Nginx のバージョン情報を隠す、攻撃情報の漏洩防止

    ##
    # SSL 設定
    ##
    ssl_protocols TLSv1.2 TLSv1.3;  # TLS 1.0 / 1.1 を無効化してセキュリティを強化
    ssl_ciphers 'ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384';
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;
    ssl_session_tickets off;
    ssl_stapling on;
    ssl_stapling_verify on;
    resolver 8.8.8.8 1.1.1.1 valid=300s;
    resolver_timeout 5s;
    ssl_dhparam /etc/nginx/ssl/dhparam.pem;  # DH パラメーターを強化、手動で生成が必要

    ##
    # ログ設定
    ##
    log_format main '$remote_addr - $remote_user [$time_local] '
                    '"$request" $status $body_bytes_sent '
                    '"$http_referer" "$http_user_agent"';
    access_log /var/log/nginx/access.log main;

    ##
    # Gzip 圧縮
    ##
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_buffers 16 8k;
    gzip_http_version 1.1;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    ##
    # HTTP セキュリティヘッダー（全サイト適用、個別の server 設定で上書き可能）
    ##
    add_header X-Frame-Options "DENY" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Permissions-Policy "geolocation=(), microphone=()" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' https://trusted-cdn.com;" always;

    ##
    # バーチャルホスト設定
    ##
    include /etc/nginx/conf.d/*.conf;
    include /etc/nginx/sites-enabled/*;

    ##
    # グローバル DDoS 防御変数（ただし、具体的な設定は `server` 内で行う）
    ##
    map $http_user_agent $bad_bot {
        default 0;
        "~*(nikto|curl|wget|python)" 1;  # 悪質なクローラーやスキャンツールをブロック
    }

    limit_req_zone $binary_remote_addr zone=general:10m rate=5r/s;  # IP ごとに 1 秒あたり最大 5 リクエスト
    limit_conn_zone $binary_remote_addr zone=connlimit:10m;         # 同時接続数を制限
}
```

## サイト設定ファイル

サイト設定ファイルでは、特定のウェブサイトに対してセキュリティ設定をさらに行うことができます。

以下に簡単な例を示しますが、実際にはウェブサイトの特性に応じて、さらに多くの設定が必要になる場合があります。

```nginx title="/etc/nginx/sites-available/example.com"
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name example.com www.example.com;

    # すべての HTTP トラフィックを HTTPS にリダイレクト
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;

    server_name example.com www.example.com;

    root /var/www/html;
    index index.html index.htm;

    ##
    # SSL 証明書（Let's Encrypt）
    ##
    ssl_certificate /etc/letsencrypt/live/example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;

    ##
    # HSTS（すべてのサブドメインが HTTPS を使用している場合のみ includeSubDomains を加える）
    ##
    add_header Strict-Transport-Security "max-age=31536000; preload" always;

    ##
    # サイト固有の DDoS 防止
    ##
    limit_req zone=general burst=10 nodelay;
    limit_conn connlimit 20;

    ##
    # メインルートの処理
    ##
    location / {
        try_files $uri $uri/ =404;
    }

    ##
    # PHP 処理（PHP が必要なサイトのみ有効化し、セキュリティ問題を避ける）
    ##
    location ~ \.php$ {
        try_files $uri =404;
        include snippets/fastcgi-php.conf;
        fastcgi_pass unix:/run/php/php7.4-fpm.sock;
        fastcgi_param SCRIPT_FILENAME $document_root$fastcgi_script_name;
        include fastcgi_params;
    }

    ##
    # 隠しファイルや敏感なファイルへのアクセスを禁止（Let's Encrypt を除外）
    ##
    location ~ /\.(?!well-known).* {
        deny all;
    }

    location ^~ /.well-known/acme-challenge/ {
        allow all;
    }
}
```

---

もう一つの例として、前に議論した API を外部公開するシナリオを考えます。例えば、API エンドポイントが以下のような場合：

- `https://temp_api.example.com/test`

この API エンドポイントに対して Nginx を使ってセキュリティ設定を行います：

```nginx title="/etc/nginx/sites-available/temp_api.example.com"
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name temp_api.example.com;

    # すべての HTTP リクエストを HTTPS にリダイレクト
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name temp_api.example.com;

    ##
    # SSL 証明書のパス（証明書の位置に依存）
    ##
    ssl_certificate /etc/letsencrypt/live/temp_api.example.com/fullchain.pem; # Certbot によって管理
    ssl_certificate_key /etc/letsencrypt/live/temp_api.example.com/privkey.pem; # Certbot によって管理

    ##
    # HTTP セキュリティヘッダー
    ##
    add_header Strict-Transport-Security "max-age=63072000; includeSubdomains; preload" always;

    ##
    # リクエストボディのサイズ制限（DDoS 防止）
    ##
    client_max_body_size 10M;

    ##
    # サイト専用のリクエスト制限
    ##
    limit_req zone=general burst=10 nodelay;
    limit_conn connlimit 20;

    ##
    # FastAPI アプリケーションへのリバースプロキシ
    ##
    location /test {
      proxy_pass http://127.0.0.1:8000;
      proxy_http_version 1.1;
      proxy_set_header Host $host;
      proxy_set_header X-Real-IP $remote_addr;
      proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
      proxy_set_header X-Forwarded-Proto $scheme;
      proxy_cache_bypass $http_cache_control;
      proxy_no_cache $http_cache_control;

      # Nginx が FastAPI のエラー応答を処理する
      proxy_intercept_errors on;

      # GET、POST、HEAD のみを許可
      limit_except GET POST HEAD {
          deny all;
      }
    }

    ##
    # 悪質な User-Agent をブロック
    ##
    if ($bad_bot) {
        return 403;  # よく知られたスキャナーの User-Agent をブロック
    }
}
```

## 結論

ネットワーク環境において、セキュリティは常に永遠のテーマです。

これらのセキュリティ対策が絶対的な安全を保証するわけではありませんが、ウェブサイトが攻撃を受けるリスクを効果的に減らすことができます：

> **悪意のある攻撃者は、セキュリティ対策が施されているウェブサイトを避け、攻撃のターゲットとしてより簡単なサイトを探す可能性が高いです。（もちろん、あなたのサイトが非常に高い価値を持っている場合は、話は別ですが。）**

一般のユーザーにとって、これらのセキュリティ対策を有効にすることは、ウェブサイトを保護するだけでなく、SEO ランキングの向上やユーザーの信頼度を高めることにも繋がります。まさに投資する価値のある方向です。

確かに、このプロセスを一通り終わらせるのは非常に手間がかかり、煩わしく感じ、疲れてしまうこともあるかもしれません。しかし、決して諦めてはいけません。ネットワーク環境は悪意に満ちており、少しの不注意で重大な損失を招く可能性があります。

少し休憩した後、次の章では Fail2Ban を使って Nginx のセキュリティ監視を行う方法について学びましょう。
