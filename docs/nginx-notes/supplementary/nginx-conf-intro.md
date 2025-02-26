# nginx.conf

## Nginx 主配置文件

Nginx 的所有設定起點都在這裡，檔案路徑是： **`/etc/nginx/nginx.conf`**。

其中的層級結構如下：

- **全域層級**：定義服務全域設定，如執行用戶、進程數量。
- **事件層級**：負責連線管理，如最大併發連線數。
- **HTTP 層級**：定義 HTTP 相關配置，包括日誌格式、Gzip 壓縮、虛擬主機設定等。
- **Server 層級**：定義特定網站的域名、監聽埠與 SSL 設定。
- **Location 層級**：匹配特定 URL 路徑並指定請求處理方式。

但是這樣看實在沒有感覺，所以我們仔細看一下檔案內容是什麼：

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

## 主配置檔案：全域設定

這部分定義了整體 Nginx 伺服器的基本運行參數。

```nginx
user www-data;
worker_processes auto;
pid /run/nginx.pid;
include /etc/nginx/modules-enabled/*.conf;
```

- **`user www-data;`**：
  - 設定 Nginx 運行的系統用戶，通常是 `www-data`。
- **`worker_processes auto;`**：
  - 設定工作進程數量，`auto` 表示自動調整
  - 通常會根據 CPU 核心數決定，確保最佳效能。
- **`pid /run/nginx.pid;`**：
  - 指定 Nginx 進程 PID 儲存的檔案位置，方便系統管理員檢查與控制 Nginx 進程。
- **`include /etc/nginx/modules-enabled/*.conf;`**：
  - 載入額外的 Nginx 模組，允許在 `modules-enabled/` 目錄內啟用不同功能的模組（例如 HTTP/2、stream 模組等）。

## 主配置檔案：事件處理設定

這部分決定 Nginx 如何處理連線與請求。

```nginx
events {
    worker_connections 768;
    # multi_accept on;
}
```

- **`worker_connections 768;`**：
  - 每個 `worker_process` 能夠同時處理的最大連線數。若 `worker_processes` 為 4，則最大併發連線數為 `4 × 768 = 3072`。
- **`# multi_accept on;`** （註解）：
  - 如果啟用，Nginx 在收到新連線時會同時接受多個請求，而非逐個處理，可提升高流量場景的效能。

## 主配置檔案：HTTP 服務設定

這部分包含多個 HTTP 相關設定，適用於所有 HTTP 服務的站點。

```nginx
http {...}
```

這裡有點繁瑣，我們把每個部分包起來，有興趣的讀者再點開來看就好。

- **HTTP 設定**

  ```nginx
  sendfile on;
  tcp_nopush on;
  types_hash_max_size 2048;
  # server_tokens off;
  ```

  這些設定影響 Nginx 處理 HTTP 連線的方式。

  - **`sendfile on;`**：
  - 啟用 `sendfile()` 系統調用，加速靜態文件傳輸，提高效能。
  - **`tcp_nopush on;`**：
  - 讓 Nginx 盡可能一次性傳輸完整的 HTTP 響應，提高網路效能。
  - **`types_hash_max_size 2048;`**：
  - 設定 MIME 類型哈希表的最大大小，影響 `mime.types` 配置的解析效率。
  - **`# server_tokens off;`** （註解）：
  - 若啟用，Nginx 不會在錯誤頁面與 HTTP 回應標頭中顯示版本資訊，提升安全性。

- **MIME 類型設定**

  ```nginx
  include /etc/nginx/mime.types;
  default_type application/octet-stream;
  ```

  - **`include /etc/nginx/mime.types;`**：
  - 載入 `mime.types` 檔案，設定不同類型文件的 `Content-Type` 回應標頭。
  - **`default_type application/octet-stream;`**：
  - 設定默認的 `Content-Type`，如果無法辨識檔案類型則回應 `application/octet-stream`。

- **SSL 設定**

  ```nginx
  ssl_protocols TLSv1 TLSv1.1 TLSv1.2 TLSv1.3; # Dropping SSLv3, ref: POODLE
  ssl_prefer_server_ciphers on;
  ```

  - **`ssl_protocols TLSv1 TLSv1.1 TLSv1.2 TLSv1.3;`**：
  - 定義支援的 SSL/TLS 版本，停用已過時的 SSLv3 以避免 POODLE 漏洞。
  - **`ssl_prefer_server_ciphers on;`**：
  - 讓伺服器優先選擇自己的加密套件，增強安全性。

- **記錄檔設定**

  ```nginx
  access_log /var/log/nginx/access.log;
  error_log /var/log/nginx/error.log;
  ```

  - **`access_log /var/log/nginx/access.log;`**：
  - 記錄所有 HTTP 存取請求，便於監控網站流量。
  - **`error_log /var/log/nginx/error.log;`**：
  - 記錄 Nginx 錯誤日誌，有助於偵錯與診斷問題。

- **Gzip 壓縮設定**

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
  - 啟用 Gzip 壓縮，提高網頁傳輸效率，減少流量消耗。
  - **（註解部分）**：
  - `gzip_comp_level 6;` 可調整壓縮比率，數值越高壓縮效果越好但 CPU 開銷增加。
  - **`gzip_types ...`**：
  - 定義哪些 MIME 類型的回應會被 Gzip 壓縮（如 CSS、JS、JSON 等）。

- **虛擬主機設定**

  ```nginx
  include /etc/nginx/conf.d/*.conf;
  include /etc/nginx/sites-enabled/*;
  ```

  - **`include /etc/nginx/conf.d/*.conf;`**：
  - 載入 `conf.d` 目錄內的所有 `.conf` 檔案，通常用來存放全域性設定。
  - **`include /etc/nginx/sites-enabled/*;`**：
  - 載入 `sites-enabled` 內的網站設定檔，每個站點的設定檔案通常位於 `sites-available`，透過符號連結啟用。

## 主配置檔案：郵件代理

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

這部分定義了 Nginx 的郵件代理功能（POP3/IMAP），預設為註解狀態，未啟用。

---

預設的檔案內容就是這樣，通常一開始我們不太會去動這個檔案，因為這是 Nginx 的全域設定，我們會在 `sites-available` 與 `sites-enabled` 目錄中建立獨立的站點配置檔案，以便管理多個站點。
