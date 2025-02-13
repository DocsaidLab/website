# nginx.conf

## Nginx の主設定ファイル

Nginx のすべての設定はここから始まります。ファイルのパスは **`/etc/nginx/nginx.conf`** です。

その階層構造は以下の通りです：

- **全体レベル**：サービス全体の設定を定義します。例えば、実行ユーザーやプロセスの数など。
- **イベントレベル**：接続管理を担当します。例えば、最大並列接続数など。
- **HTTP レベル**：HTTP 関連の設定を定義します。ログのフォーマット、Gzip 圧縮、仮想ホストの設定などが含まれます。
- **Server レベル**：特定のウェブサイトのドメイン名、リスニングポート、SSL 設定などを定義します。
- **Location レベル**：特定の URL パスにマッチし、リクエストの処理方法を指定します。

ですが、このままだと感覚がつかみにくいので、ファイルの内容を詳しく見ていきましょう。

```nginx title="/etc/nginx/nginx.conf"
user www-data;
worker_processes auto;
pid /run/nginx.pid;
include /etc/nginx/modules-enabled/*.conf;

events {
	worker_connections 768;
	# multi_accept on;
}

http {

	##
	# Basic Settings
	##

	sendfile on;
	tcp_nopush on;
	types_hash_max_size 2048;
	# server_tokens off;

	# server_names_hash_bucket_size 64;
	# server_name_in_redirect off;

	include /etc/nginx/mime.types;
	default_type application/octet-stream;

	##
	# SSL Settings
	##

	ssl_protocols TLSv1 TLSv1.1 TLSv1.2 TLSv1.3; # Dropping SSLv3, ref: POODLE
	ssl_prefer_server_ciphers on;

	##
	# Logging Settings
	##

	access_log /var/log/nginx/access.log;
	error_log /var/log/nginx/error.log;

	##
	# Gzip Settings
	##

	gzip on;

	# gzip_vary on;
	# gzip_proxied any;
	# gzip_comp_level 6;
	# gzip_buffers 16 8k;
	# gzip_http_version 1.1;
	# gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

	##
	# Virtual Host Configs
	##

	include /etc/nginx/conf.d/*.conf;
	include /etc/nginx/sites-enabled/*;
}


#mail {
#	# See sample authentication script at:
#	# http://wiki.nginx.org/ImapAuthenticateWithApachePhpScript
#
#	# auth_http localhost/auth.php;
#	# pop3_capabilities "TOP" "USER";
#	# imap_capabilities "IMAP4rev1" "UIDPLUS";
#
#	server {
#		listen     localhost:110;
#		protocol   pop3;
#		proxy      on;
#	}
#
#	server {
#		listen     localhost:143;
#		protocol   imap;
#		proxy      on;
#	}
#}
```

## 主設定ファイル：全体設定

この部分では、Nginx サーバーの基本的な運用パラメータが定義されています。

```nginx
user www-data;
worker_processes auto;
pid /run/nginx.pid;
include /etc/nginx/modules-enabled/*.conf;
```

- **`user www-data;`**：
  - Nginx を実行するシステムユーザーを設定します。通常は `www-data` です。
- **`worker_processes auto;`**：
  - ワーカープロセスの数を設定します。`auto` は自動的に調整されることを意味します。
  - 通常は CPU コアの数に基づいて決定され、最適なパフォーマンスを確保します。
- **`pid /run/nginx.pid;`**：
  - Nginx プロセスの PID を保存するファイルの場所を指定します。システム管理者が Nginx プロセスを確認・制御する際に便利です。
- **`include /etc/nginx/modules-enabled/*.conf;`**：
  - 追加の Nginx モジュールを読み込みます。これにより、`modules-enabled/` ディレクトリ内でさまざまな機能のモジュール（例えば、HTTP/2 や stream モジュールなど）を有効にすることができます。

## 主設定ファイル：イベント処理設定

この部分では、Nginx が接続やリクエストをどのように処理するかが決定されます。

```nginx
events {
    worker_connections 768;
    # multi_accept on;
}
```

- **`worker_connections 768;`**：
  - 各 `worker_process` が同時に処理できる最大接続数を設定します。例えば、`worker_processes` が 4 の場合、最大並列接続数は `4 × 768 = 3072` となります。
- **`# multi_accept on;`** （コメントアウト）：
  - 有効にすると、Nginx は新しい接続を受けた際に複数のリクエストを同時に受け入れ、個別に処理するのではなく、効率的に高トラフィックなシナリオでパフォーマンスを向上させます。

## 主設定ファイル：HTTP サービス設定

この部分では、HTTP に関連する設定が含まれており、すべての HTTP サービスに適用されます。

```nginx
http {...}
```

ここは少し冗長なので、各部分をまとめておきます。興味がある方はさらに詳細を見てください。

- **HTTP 設定**

  ```nginx
  sendfile on;
  tcp_nopush on;
  types_hash_max_size 2048;
  # server_tokens off;
  ```

  これらの設定は、Nginx が HTTP 接続を処理する方法に影響を与えます。

  - **`sendfile on;`**：
    - `sendfile()` システムコールを有効にし、静的ファイルの転送を加速し、パフォーマンスを向上させます。
  - **`tcp_nopush on;`**：
    - Nginx ができるだけ一度に完全な HTTP 応答を送信し、ネットワーク効率を高めます。
  - **`types_hash_max_size 2048;`**：
    - MIME タイプのハッシュテーブルの最大サイズを設定し、`mime.types` 設定の解析効率に影響を与えます。
  - **`# server_tokens off;`** （コメントアウト）：
    - 有効にすると、Nginx はエラーページや HTTP 応答ヘッダーにバージョン情報を表示しなくなり、セキュリティを強化します。

- **MIME タイプ設定**

  ```nginx
  include /etc/nginx/mime.types;
  default_type application/octet-stream;
  ```

  - **`include /etc/nginx/mime.types;`**：
    - `mime.types` ファイルを読み込み、異なる種類のファイルの `Content-Type` 応答ヘッダーを設定します。
  - **`default_type application/octet-stream;`**：
    - ファイルタイプが識別できない場合のデフォルトの `Content-Type` を設定し、`application/octet-stream` を応答します。

- **SSL 設定**

  ```nginx
  ssl_protocols TLSv1 TLSv1.1 TLSv1.2 TLSv1.3; # Dropping SSLv3, ref: POODLE
  ssl_prefer_server_ciphers on;
  ```

  - **`ssl_protocols TLSv1 TLSv1.1 TLSv1.2 TLSv1.3;`**：
    - サポートされる SSL/TLS バージョンを定義し、POODLE 脆弱性を避けるために古い SSLv3 を無効にします。
  - **`ssl_prefer_server_ciphers on;`**：
    - サーバーが自分の暗号スイートを優先的に選択し、セキュリティを強化します。

- **ログ設定**

  ```nginx
  access_log /var/log/nginx/access.log;
  error_log /var/log/nginx/error.log;
  ```

  - **`access_log /var/log/nginx/access.log;`**：
    - すべての HTTP アクセスリクエストを記録し、ウェブサイトのトラフィックを監視します。
  - **`error_log /var/log/nginx/error.log;`**：
    - Nginx のエラーログを記録し、デバッグと問題診断に役立てます。

- **Gzip 圧縮設定**

  ```nginx
  gzip on;
  # gzip_vary on;
  # gzip_proxied any;
  # gzip_comp_level 6;
  # gzip_buffers 16 8k;
  # gzip_http_version 1.1;
  # gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
  ```

  - **`gzip on;`**：
    - Gzip 圧縮を有効にし、ウェブページの転送効率を向上させ、トラフィックを削減します。
  - **（コメントアウト部分）**：
    - `gzip_comp_level 6;` で圧縮比率を調整できます。数値が高いほど圧縮効果が高くなりますが、CPU 負荷も増えます。
    - **`gzip_types ...`**：
      - Gzip 圧縮される MIME タイプ（CSS、JS、JSON など）を定義します。

- **仮想ホスト設定**

  ```nginx
  include /etc/nginx/conf.d/*.conf;
  include /etc/nginx/sites-enabled/*;
  ```

  - **`include /etc/nginx/conf.d/*.conf;`**：
    - `conf.d` ディレクトリ内のすべての `.conf` ファイルを読み込み、通常はグローバルな設定を格納します。
  - **`include /etc/nginx/sites-enabled/*;`**：
    - `sites-enabled` 内のウェブサイト設定ファイルを読み込みます。各サイトの設定ファイルは通常 `sites-available` に格納され、シンボリックリンクを使って有効化されます。

## 主設定ファイル：メールプロキシ

```nginx
#mail {
#	server {
#		listen     localhost:110;
#		protocol   pop3;
#		proxy      on;
#	}
#	server {
#		listen     localhost:143;
#		protocol   imap;
#		proxy      on;
#	}
#}
```

この部分では、Nginx のメールプロキシ機能（POP3/IMAP）が定義されていますが、デフォルトではコメントアウトされており、有効になっていません。

---

デフォルトのファイル内容はこのようになっています。通常、最初はこのファイルを直接変更することはありません。これは Nginx のグローバル設定であり、`sites-available` および `sites-enabled` ディレクトリ内で個別のサイト設定ファイルを作成して、複数のサイトを管理します。
