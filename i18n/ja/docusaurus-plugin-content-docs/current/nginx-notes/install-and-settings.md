---
sidebar_position: 2
---

# Nginx のインストールと設定

ここでは Ubuntu を例に、Nginx をインストールし、基本的な設定を行う方法を説明します。

他のオペレーティングシステムについては、公式ドキュメントや他のリソースを参照してインストールしてください。

## Nginx のインストール

まず、ソフトウェアリポジトリを更新し、Nginx をインストールします：

```bash
sudo apt update
sudo apt install -y nginx
```

インストールが完了すると、Nginx サービスは自動的に起動します。

`systemctl` を使用して Nginx が正常に動作しているか確認します：

```bash
sudo systemctl status nginx
```

`active (running)` と表示されれば、Nginx は正常に起動しています。

<div align="center">
<figure style={{"width": "90%"}}>
![active-running](./resources/img1.jpg)
</figure>
</div>

:::warning
サーバーでファイアウォール（例：UFW）が有効になっている場合、HTTP と HTTPS のトラフィックを許可するようにしてください：

```bash
sudo ufw allow 'Nginx Full'
```

:::

## Nginx の設定構造について

### メイン設定ファイル

Nginx のメイン設定ファイルは **`/etc/nginx/nginx.conf`** です。

詳細な使用方法については、補足説明ドキュメント：[**nginx.conf**](./supplementary/nginx-conf-intro.md) を参照してください。

:::tip
ページの長さを考慮して、詳細な説明は別途記載していますので、ぜひそちらもご覧ください。⬆ ⬆ ⬆
:::

### サイト設定ファイル

各ウェブサイトやサービスに対して、通常は次の 2 つのフォルダで設定を行います：

- **`/etc/nginx/sites-available/`**：独立した設定ファイルを作成します。
- **`/etc/nginx/sites-enabled/`**：シンボリックリンクを使ってサイトを有効にします。

Nginx は `sites-available` と `sites-enabled` フォルダを使用して、複数のサイトの設定を管理します。

詳細な使用方法については、補足説明ドキュメント：[**sites-available/default**](./supplementary/sites-available-intro.md) を参照してください。

:::tip
同様に、説明ファイルも忘れずにご確認ください。⬆ ⬆ ⬆
:::

## 基本設定の例

以下は基本的な Nginx の設定例で、すべてのリクエストをバックエンド API に転送する方法を示しています。

まず、`sites-available` ディレクトリに設定ファイルを作成します：

```bash
sudo vim /etc/nginx/sites-available/temp_api.example.com
```

ファイルに以下の内容を追加します：

```nginx
server {
    listen 80;
    server_name temp_api.example.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

この設定では、`temp_api.example.com` の HTTP リクエストを、ローカルの 8000 番ポートで実行されているバックエンド API に転送します（実際の API パスに変更してください）。また、クライアントの情報も保持しており、バックエンドでのログ記録や処理に役立ちます。

次に、この設定ファイルを `sites-enabled` ディレクトリにリンクします：

```bash
sudo ln -s /etc/nginx/sites-available/temp_api.example.com /etc/nginx/sites-enabled/
```

## テストと検証

設定が完了したら、設定が正しいかどうかを確認する必要があります：

1. **Nginx 設定の構文を確認**

   ```bash
   sudo nginx -t
   ```

   `syntax is ok` と `test is successful` が表示されれば、設定に問題はありません。

2. **Nginx をリロードして新しい設定を適用**

   ```bash
   sudo systemctl reload nginx
   ```

3. **サービスを検証**

   ブラウザまたは `curl` を使用して `http://temp_api.example.com/test` にアクセスし、期待する応答が得られるか確認します。

:::tip
ここでの `http://temp_api.example.com/test` は、前の章で仮定した API エンドポイントに基づいています。

実際の状況に応じて、設定ファイルの `server_name` と `proxy_pass` を変更してください。
:::

:::info
**よく使うコマンドメモ**

```bash
# Nginx を起動
sudo systemctl start nginx

# Nginx を停止
sudo systemctl stop nginx

# Nginx を再起動（通常は重大な設定変更後に使用）
sudo systemctl restart nginx

# Nginx をリロード（軽微な設定変更後に推奨）
sudo systemctl reload nginx
```

:::

## 複数サイトの設定

Nginx は `server` ブロックを使用して仮想ホストを定義し、リクエストの **`Host`** と **`listen` 監視ポート** に基づいてどのサイトを処理するかを決定します。ユーザーが HTTP リクエストを送信すると、Nginx は最初にリクエストの「ホスト名（Host）」と「ポート番号（Port）」を照合し、それに対応する `server` ブロックを選んで処理します。

通常、各サイトは `listen` ディレクティブで監視するポート（例えば 80 や 443）を指定し、`server_name` ディレクティブを使用して特定のドメイン名に対応させます。例えば、`example.com` と `api.example.com` はそれぞれ異なる `server_name` を設定でき、Nginx は Host ヘッダーに基づいて適切な設定を選んでリクエストを処理します。

次に、典型的な複数サイトの設定例を見てみましょう：

```nginx title="/etc/nginx/sites-available/example.com"
server {
    listen 80;
    server_name example.com;
    root /var/www/example.com;
    index index.html;
}

server {
    listen 80;
    server_name api.example.com;
    location / {
        proxy_pass http://localhost:5000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

- 最初の `server` ブロックは `example.com` の静的リソースを処理します。
- 2 番目の `server` ブロックは `api.example.com` を処理し、リクエストをローカルの `5000` ポートの API サービスに転送します。

例えば、ユーザーが `http://example.com` にアクセスすると、Nginx はリクエストを `/var/www/example.com` ディレクトリ内の `index.html` ファイルに転送します。一方、`http://api.example.com` にアクセスすると、リクエストはローカルの `5000` ポートに転送され、API サービスが応答します。

## デフォルトホストの設定

設定ファイルで `default_server` を指定することができます。この設定により、リクエストの `server_name` に一致しない場合、Nginx はデフォルトのホストを使用して処理します：

```nginx
server {
    listen 80 default_server;
    server_name _;
    return 404;
}
```

この設定は、指定されていないリクエストが誤ったサイトに進入しないようにします。

## よくあるエラーとトラブルシューティング

最後に、問題が発生した場合のトラブルシューティング方法について説明します。

1.  **Nginx 設定が正しいか確認**

    ```bash
    sudo nginx -t
    ```

    エラーメッセージが表示された場合は、指示に従って設定を修正します。

2.  **Nginx サービスの状態を確認**

    ```bash
    sudo systemctl status nginx
    ```

    `active (running)` と表示されていれば Nginx は正常に動作しています。`failed` と表示される場合は、エラーログを確認します。

3.  **エラーログを確認**

    ```bash
    sudo journalctl -u nginx --no-pager --lines=30
    ```

    または、`error.log` を直接確認します：

    ```bash
    sudo tail -f /var/log/nginx/error.log
    ```

    これらのログは、404、502、403 などのエラーの原因を特定するのに役立ちます。

## 結論

本章では、Ubuntu に Nginx をインストールし、基本的なサイト設定、リバースプロキシ、および複数サイト管理を設定する方法を紹介しました。

さらに、設定をテストおよび検証し、サイトが正常に動作していることを確認する方法を学びました。これらの基本設定により、Nginx は静的サイトや API プロキシのニーズを満たすことができます。

次の章では、HTTPS の設定方法と Let's Encrypt の使用方法について学びます。
