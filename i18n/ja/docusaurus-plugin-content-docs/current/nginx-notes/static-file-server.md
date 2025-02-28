---
sidebar_position: 6
---

# Nginx による静的リソースの提供

この章では、Docusaurus を使用して構築した静的サイトを例に、Nginx を使ってサイトを提供する方法を説明します。

以下のように、すでにドメイン名を準備し、サーバーにポイントしていると仮定します：

- `your.domain`

:::tip
以下のコマンドおよび設定例の`your.domain`は、必ず実際に使用するドメイン名に置き換えてください。
:::

## 静的サイトの構築

構築前に、`docusaurus.config.js`ファイルの`url`が正しくあなたのドメインに設定されていることを確認してください：

```javascript
module.exports = {
  url: "https://your.domain",
  // ...他の設定
};
```

確認が完了したら、以下のコマンドを実行して静的ファイルを生成します：

```bash
DOCUSAURUS_IGNORE_SSG_WARNINGS=true yarn build
```

:::tip
`DOCUSAURUS_IGNORE_SSG_WARNINGS`環境変数を使用しない場合、奇妙な警告メッセージが大量に表示される可能性がありますが、ビルド結果には影響しません。
:::

このコマンドは、`build/`フォルダ内に静的な HTML、CSS、JS ファイルを生成します。

次に、ビルドしたファイルをサーバーの指定ディレクトリにアップロードし、ファイル権限を設定します：

```bash
sudo mkdir -p /var/www/your.domain
sudo rsync -av build/ /var/www/your.domain/
sudo chown -R www-data:www-data /var/www/your.domain
```

## SSL 証明書の取得

Let's Encrypt を使用して SSL 証明書を発行し、HTTPS 経由で安全にサービスを提供することをお勧めします：

```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your.domain
```

## Nginx の設定

専用の Nginx 設定ファイルを作成します：

```bash
sudo vim /etc/nginx/sites-available/your.domain
```

設定内容の例：

```nginx
server {
    listen 80;
    server_name your.domain;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your.domain;

    # SSL証明書（Certbotで発行されたものを使用）
    ssl_certificate /etc/letsencrypt/live/your.domain/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your.domain/privkey.pem;

    # 静的ファイル提供ディレクトリの設定
    root /var/www/your.domain;
    index index.html;

    # MIMEタイプの設定
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

設定が完了したら、設定ファイルを有効にします：

```bash
sudo ln -s /etc/nginx/sites-available/your.domain /etc/nginx/sites-enabled/
```

Nginx 設定をテストして再読み込みします：

```bash
sudo nginx -t
sudo systemctl reload nginx
```

## 高度な設定

本番環境では、より多くのセキュリティとパフォーマンス設定を追加することをお勧めします。以下は、完全な高度な設定例です：

```nginx
server {
    listen 80;
    listen [::]:80;
    server_name your.domain;

    # 🔒 HTTPをHTTPSに自動でリダイレクト
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name your.domain;

    ssl_certificate /etc/letsencrypt/live/your.domain/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your.domain/privkey.pem;

    # 🔒 セキュリティヘッダー
    add_header Strict-Transport-Security "max-age=63072000; includeSubdomains; preload" always;

    # 🔧 DDoS対策としてファイルアップロードサイズの制限
    client_max_body_size 10M;

    # 📁 buildディレクトリ
    root /var/www/your.domain;
    index index.html;

    # 🗃️ 静的リソースのキャッシュ
    location ~* \.(jpg|jpeg|png|gif|ico|svg|woff2?|ttf|css|js)$ {
        expires 7d;
        add_header Cache-Control "public, must-revalidate";
    }

    # 🔧 メインのルーティング規則
    location / {
        try_files $uri $uri/ /index.html;
    }

    # MIMEタイプの設定
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # 🔍 カスタムログファイルの位置
    access_log /var/log/nginx/your.domain.access.log main;
    error_log /var/log/nginx/your.domain.error.log warn;
}
```

## 結論

上記の手順と高度な設定を使用することで、Nginx を用いて静的リソースを安全かつ効率的に提供できます。設定が完了したら、`https://your.domain`でウェブサイトにアクセスできます。

最後に、SSL 証明書の有効期限を定期的に確認し、Nginx のバージョンやセキュリティルールを更新することを忘れないでください。これらの措置は、潜在的なセキュリティリスクから保護し、サイトの安定した運用を長期にわたって確保するのに役立ちます。
