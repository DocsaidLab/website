---
sidebar_position: 4
---

# Nginx 安全強化

由於 Nginx 靈活和高效，在現代網路環境中被廣泛使用。然而，如果沒有適當的安全設定，你辛苦架設的網站可能成為攻擊者的目標。

主要的安全風險包括：

- **未加強的 TLS 配置**：可能導致中間人攻擊和弱加密漏洞。
- **缺乏 HSTS 機制**：可能導致使用者意外透過 HTTP 訪問網站，降低 HTTPS 的安全保障。
- **未設置適當的 HTTP 安全標頭**：可能讓網站容易受到 XSS、點擊劫持等攻擊。
- **缺乏流量限制和 DDoS 防禦機制**：可能導致伺服器資源被濫用，甚至造成服務中斷。

為了確保 Nginx 服務器的安全，我們必須在配置中採取一系列強化措施。

## TLS 安全強化

:::tip
這個部分的配置層級是在 `http` 區塊內，因為 TLS 是基於 HTTP 協議的安全擴展，也因此以下設定需要寫在 `/etc/nginx/nginx.conf` 主配置檔案中。
:::

TLS，完整名稱為 Transport Layer Security，是一種加密協議，用於保護網路通信的安全性。

在現代網路環境中，HTTPS 加密已成為網站的基本標準，而 TLS 則是 HTTPS 的核心技術。雖然 HTTPS 能加密網站與訪客之間的流量，但許多未經強化的 TLS 設定仍然充滿漏洞，像是：

- **使用舊版 TLS 協議（如 TLS 1.0/1.1）** → 容易遭受攻擊，如 BEAST、POODLE
- **允許弱加密演算法（如 RC4、3DES）** → 可能被破解
- **缺乏 OCSP Stapling** → 會導致憑證檢查時效能降低
- **啟用了 Session Tickets** → 可能讓攻擊者重用舊加密密鑰，導致重放攻擊

為了確保網站的安全性，我們必須正確配置 Nginx，強化 TLS 設定，以防範潛在的安全風險，如中間人攻擊（MITM）、憑證偽造、弱加密攻擊等。

以下有幾個可以考慮配置的優化方式：

1. **禁用舊版 TLS，僅允許 TLS 1.2 / 1.3**

   ```nginx
   ssl_protocols TLSv1.2 TLSv1.3;
   ```

   - **TLS 1.0 / 1.1 已不安全**，許多瀏覽器（如 Chrome、Firefox）已經停止支援。
   - **TLS 1.2 仍然安全可靠**，廣泛支援。
   - **TLS 1.3 速度更快，安全性更高**，支援「零延遲握手」，適合現代應用。

---

2. **使用安全的加密演算法**

   ```nginx
   ssl_ciphers 'ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384';
   ssl_prefer_server_ciphers on;
   ```

   這些加密套件 **排除了 RC4、3DES、MD5 等弱加密算法**，並確保伺服器優先使用自己的安全加密組合。

---

3. **啟用 OCSP Stapling，提升憑證驗證速度**

   ```nginx
   ssl_stapling on;
   ssl_stapling_verify on;
   resolver 8.8.8.8 1.1.1.1 valid=300s;
   resolver_timeout 5s;
   ```

   OCSP Stapling 可讓伺服器預先快取憑證狀態，減少每次連線時向 CA 伺服器查詢的延遲，提高效能。

---

4. **禁用 Session Tickets，防止密鑰重用**

   ```nginx
   ssl_session_tickets off;
   ```

   Session Tickets 可能導致密鑰重複使用，讓攻擊者透過竊取票據來復用加密會話。

---

5. **設定安全的 DH 參數**

   ```nginx
   ssl_dhparam /etc/nginx/ssl/dhparam.pem;
   ```

   Diffie-Hellman（DH）金鑰交換機制提供額外的安全性，防止某些攻擊。

   如果沒有 DH 參數檔案，可以使用 OpenSSL 生成 DH 參數檔案：

   ```bash
   sudo mkdir -p /etc/nginx/ssl
   sudo openssl dhparam -out /etc/nginx/ssl/dhparam.pem 2048
   ```

---

6. **啟用會話快取，提高 TLS 連線效能**

   ```nginx
   ssl_session_cache shared:SSL:10m;
   ssl_session_timeout 1h;
   ```

   - **允許重用 TLS 會話，提高性能**
   - **減少 CPU 負擔，提升用戶體驗**

## HSTS 進階配置

:::tip
這個部分的配置層級是在 `server` 區塊內，因為 HSTS 是針對特定網站的安全機制，所以以下設定需要寫在 `/etc/nginx/sites-available/example.com` 等網站配置檔案中。
:::

HSTS，原文為 HTTP Strict Transport Security，是一種安全機制，用於強制瀏覽器僅透過 HTTPS 訪問網站。

雖然 HTTPS 已經成為網站的標準配置，但單純的 HTTPS 仍然無法防止降級攻擊（Downgrade Attack）和中間人攻擊（MITM Attack）。這時，HSTS 就成為一個關鍵機制，它能確保使用者每次連線時都自動轉向 HTTPS，防止網站遭受不安全的 HTTP 連線攻擊。

HSTS 由 **RFC 6797** 標準定義，其核心功能是：

- **強制網站僅能透過 HTTPS 存取**
- **防止降級攻擊（Downgrade Attack）**
- **阻擋 SSL 剝離攻擊（SSL Stripping Attack）**
- **提升網站的信任度與 SEO 排名**

舉個例子：如果一個網站支援 HTTPS，但沒有啟用 HSTS，攻擊者可以透過中間人攻擊竄改初始 HTTP 連線，強迫用戶降級到 HTTP，並攔截所有數據流量。這時所有登入密碼、信用卡資訊等敏感資料就可以被攻擊者竊取，用戶的隱私就暴露於風險之中。

當 HSTS 啟用後：

1. 瀏覽器在第一次連線成功後，會記住該網站只允許 HTTPS 連線。
2. 往後的所有連線，瀏覽器會直接轉為 HTTPS，即使用戶輸入 `http://example.com` 也會自動轉成 `https://example.com`。
3. 即使攻擊者攔截初始 HTTP 連線，也無法降級網站至 HTTP。

在 Nginx 中，HSTS 必須僅適用於 HTTPS 伺服器，因此應該放在 `server` 區塊內。

```nginx
server {
    listen 443 ssl http2;
    server_name example.com www.example.com;

    # 啟用 HSTS，並強制應用於所有子網域
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
}
```

- 各項參數意義如下：

  - `max-age=31536000`：HSTS 設定的有效期（以秒為單位），這裡設定為 1 年。
  - `includeSubDomains`：對所有子網域強制啟用 HSTS，防止子網域被攻擊者利用。
  - `preload`：允許網站加入瀏覽器 HSTS 預載入列表。
  - `always`：確保 Nginx 在所有回應中都加上 HSTS 標頭（包含各種錯誤頁面）。

---

配置時，必須只對 HTTPS 啟用 HSTS，確保 HTTP 連線被 301 轉向 HTTPS，而不包含 HSTS 標頭，因此應該在 HTTP 伺服器中添加以下配置：

```nginx
server {
    listen 80;
    server_name example.com www.example.com;

    # 轉向 HTTPS，但不加 HSTS
    return 301 https://$host$request_uri;
}
```

這樣可以防止攻擊者利用初始 HTTP 連線來繞過 HSTS，讓瀏覽器正確記錄。

## 其他安全標頭

:::tip
同樣還是標頭的部分，所以以下設定仍然是需要寫在 `/etc/nginx/sites-available/example.com` 等網站配置檔案中。
:::

當我們談論網站安全，大多數人首先想到的是 HTTPS 加密與防火牆，但這些只能解決傳輸層的安全問題。

實際上，許多網頁攻擊（如 XSS、點擊劫持、MIME 偽裝）都是透過瀏覽器漏洞發動的，而正確配置 HTTP 安全標頭（Security Headers）則是防止此類攻擊的最佳方式。

HTTP 安全標頭可有效抵禦：

- **XSS（跨站腳本攻擊）**
- **Clickjacking（點擊劫持）**
- **MIME 偽裝攻擊**
- **Cookie 竊取與劫持**

即使你的網站已啟用 HTTPS，仍然需要正確配置這些標頭，才能全面提升安全性。

---

在 Nginx 的 `server` 區塊內，我們可以透過 `add_header` 指令來加入這些標頭。

```nginx
server {
    listen 443 ssl http2;
    server_name example.com www.example.com;

    # 強化安全性
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

這些安全標頭的用途，我們依序來看看：

1. **X-Frame-Options：防止 Clickjacking 攻擊**

   ```nginx
   add_header X-Frame-Options "DENY" always;
   ```

   阻止網站被嵌入到 `iframe` 中，防止攻擊者透過隱藏框架誘騙用戶點擊惡意按鈕。

   選項可以是 `DENY`（完全禁止網站被嵌入）或 `SAMEORIGIN`（僅允許相同來源的 `iframe` 內嵌）。推薦使用 `DENY`，除非你的網站需要在內嵌框架內運行。

---

2. **X-XSS-Protection：防止 XSS 攻擊**

   ```nginx
   add_header X-XSS-Protection "1; mode=block" always;
   ```

   啟用瀏覽器內建的 XSS 過濾器，防止攻擊者在網站中插入惡意腳本。`1; mode=block` 代表偵測到 XSS 攻擊時，瀏覽器將阻止頁面載入。

   雖然 CSP（Content Security Policy）是更好的 XSS 防禦機制，但 `X-XSS-Protection` 可作為額外保障。

---

3. **X-Content-Type-Options：防止 MIME 類型偽裝攻擊**

   ```nginx
   add_header X-Content-Type-Options "nosniff" always;
   ```

   阻止瀏覽器對未知 MIME 類型進行猜測，防止攻擊者上傳惡意文件（如 `.js` 變 `.jpg`）。確保所有下載的內容都按照伺服器設定的 MIME 類型解析，減少漏洞風險。

   這是 OWASP 標準建議的強制性安全標頭之一。

---

4. **Referrer-Policy：保護用戶隱私**

   ```nginx
   add_header Referrer-Policy "strict-origin-when-cross-origin" always;
   ```

   控制 `Referer` 標頭的傳送方式，防止外部網站獲取完整的網址資訊。`strict-origin-when-cross-origin` 設定：相同來源的請求傳送完整 `Referer`，跨網站請求僅傳送來源 (`origin`)，降級 HTTP 時（HTTPS → HTTP）不傳送 `Referer`。

   這樣可以在保持網站分析數據的同時，降低隱私洩露風險。

---

5. **Permissions-Policy（Feature Policy）：限制瀏覽器功能**

   ```nginx
   add_header Permissions-Policy "geolocation=(), microphone=()" always;
   ```

   限制網站可用的瀏覽器功能，防止濫用用戶權限（如攝影機、麥克風、地理位置）。上述設定禁止地理位置存取（geolocation）和麥克風存取（microphone）。

   這在現代隱私保護中相當重要，特別是對 GDPR 或隱私法規要求的網站。

---

6. **Content-Security-Policy（CSP）：最強的 XSS 防禦**

   ```nginx
   add_header Content-Security-Policy "default-src 'self'; script-src 'self' https://trusted-cdn.com;" always;
   ```

   CSP 是防禦 XSS 的最強機制，它可以限制網站上允許載入的資源來源。`default-src 'self'` 表示預設只允許本站資源，`script-src 'self' https://trusted-cdn.com;` 表示僅允許本站和指定 CDN 的 JavaScript 腳本。

   CSP 可以有效防止 XSS 攻擊，並提高網站的安全性。

## 主設定檔配置

在之前的章節中，我們先看過了主設定檔，但是先跳過，沒有對它進行修改。

現在我們回來仔細看看。一般來說，我們可以把 TLS 設定、HTTP 安全標頭、請求速率限制等全域性設定放在 `nginx.conf` 中，而將站點特定的 HSTS、轉向 HTTPS 等配置放在 `sites-available/default` 等站點配置檔案中。

基於剛才的討論，我們可以把預設的 `nginx.conf` 主配置檔進行一些修改，以符合我們的需求。

```nginx title="/etc/nginx/nginx.conf"
user www-data;
worker_processes auto;
pid /run/nginx.pid;

# 設定錯誤日誌
error_log /var/log/nginx/error.log warn;
include /etc/nginx/modules-enabled/*.conf;

events {
    worker_connections 1024;  # 提高最大並發連線數 (原 768)
    multi_accept on;          # 允許 worker 進程同時接受多個新連線
}

http {
    ##
    # 基本設定
    ##
    sendfile on;
    tcp_nopush on;
    types_hash_max_size 2048;
    server_tokens off;  # 隱藏 Nginx 版本資訊，防止洩漏攻擊資訊

    ##
    # SSL 設定
    ##
    ssl_protocols TLSv1.2 TLSv1.3;  # 停用 TLS 1.0 / 1.1，提高安全性
    ssl_ciphers 'ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384';
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;
    ssl_session_tickets off;
    ssl_stapling on;
    ssl_stapling_verify on;
    resolver 8.8.8.8 1.1.1.1 valid=300s;
    resolver_timeout 5s;
    ssl_dhparam /etc/nginx/ssl/dhparam.pem;  # 強化 DH 參數，需手動生成

    ##
    # 記錄設定
    ##
    log_format main '$remote_addr - $remote_user [$time_local] '
                    '"$request" $status $body_bytes_sent '
                    '"$http_referer" "$http_user_agent"';
    access_log /var/log/nginx/access.log main;

    ##
    # Gzip 壓縮
    ##
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_buffers 16 8k;
    gzip_http_version 1.1;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    ##
    # HTTP 安全標頭 (全站適用，可被個別 server 設定覆蓋)
    ##
    add_header X-Frame-Options "DENY" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Permissions-Policy "geolocation=(), microphone=()" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' https://trusted-cdn.com;" always;

    ##
    # 虛擬主機配置
    ##
    include /etc/nginx/conf.d/*.conf;
    include /etc/nginx/sites-enabled/*;

    ##
    # 全域 DDoS 防禦變數（但具體設定在 `server` 內）
    ##
    map $http_user_agent $bad_bot {
        default 0;
        "~*(nikto|curl|wget|python)" 1;  # 阻擋惡意爬蟲與掃描工具
    }

    limit_req_zone $binary_remote_addr zone=general:10m rate=5r/s;  # 限制每 IP 每秒最多 5 個請求
    limit_conn_zone $binary_remote_addr zone=connlimit:10m;         # 限制同時連線數
}
```

## 站點配置檔案

在站點配置檔案中，我們可以進一步針對特定網站進行安全設定。

以下我們寫個簡單的範例，實際上，你可能需要根據網站的特性進行更多的設定。

```nginx title="/etc/nginx/sites-available/example.com"
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name example.com www.example.com;

    # 強制將所有 HTTP 流量重定向到 HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;

    server_name example.com www.example.com;

    root /var/www/html;
    index index.html index.htm;

    ##
    # SSL 憑證（Let's Encrypt）
    ##
    ssl_certificate /etc/letsencrypt/live/example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;

    ##
    # HSTS（如果所有子網域都啟用 HTTPS，才加 includeSubDomains）
    ##
    add_header Strict-Transport-Security "max-age=31536000; preload" always;

    ##
    # 站點專屬的 DDoS 防禦
    ##
    limit_req zone=general burst=10 nodelay;
    limit_conn connlimit 20;

    ##
    # 主要路由處理
    ##
    location / {
        try_files $uri $uri/ =404;
    }

    ##
    # PHP 解析（僅在需要 PHP 的站點啟用，避免安全問題）
    ##
    location ~ \.php$ {
        try_files $uri =404;
        include snippets/fastcgi-php.conf;
        fastcgi_pass unix:/run/php/php7.4-fpm.sock;
        fastcgi_param SCRIPT_FILENAME $document_root$fastcgi_script_name;
        include fastcgi_params;
    }

    ##
    # 禁止存取隱藏文件與敏感檔案（保留 Let's Encrypt）
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

另外一種範例就是基於我們之前所討論的部署對外 API 的情境，假設我們的 API 端點在：

- `https://temp_api.example.com/test`

我們可以透過 Nginx 來對這個 API 端點進行安全設定：

```nginx title="/etc/nginx/sites-available/temp_api.example.com"
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name temp_api.example.com;

    # Redirect all HTTP requests to HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name temp_api.example.com;

    ##
    # SSL 具體路徑取決於你的憑證位置
    ##
    ssl_certificate /etc/letsencrypt/live/temp_api.example.com/fullchain.pem; # managed by Certbot
    ssl_certificate_key /etc/letsencrypt/live/temp_api.example.com/privkey.pem; # managed by Certbot

    ##
    # HTTP 安全頭
    ##
    add_header Strict-Transport-Security "max-age=63072000; includeSubdomains; preload" always;

    ##
    # 限制請求體積（防止 DDoS）
    ##
    client_max_body_size 10M;

    ##
    # 站點專屬的請求限制
    ##
    limit_req zone=general burst=10 nodelay;
    limit_conn connlimit 20;

    ##
    # Proxy requests to FastAPI app
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

      # 讓 Nginx 處理 FastAPI 回應的錯誤
      proxy_intercept_errors on;

      # 限制只允許 GET、POST、HEAD
      limit_except GET POST HEAD {
          deny all;
      }
    }

    ##
    # 屏蔽惡意的 User-Agent
    ##
    if ($bad_bot) {
        return 403;  # 阻止常見掃描器的 User-Agent
    }
}
```

## 結論

所謂道高一尺，魔高一丈。在網路環境中，安全性永遠是一個永恆的話題。

雖然這些安全措施無法保證絕對的安全，但它們可以有效降低網站遭受攻擊的風險：

> **柿子挑軟的捏，惡意攻擊者看到你的網站有安全措施，可能不會浪費時間攻擊，轉而尋找更容易的目標。(如果你的網站有極高的價值，那另當別論。）**

對於一般使用者來說，啟用這些安全措施除了保護網站之外，還能提升 SEO 排名與使用者信任度，真的是個值得投資的方向。

雖然整個流程做下來確實很繁瑣，讓人不禁感到煩躁和疲憊。但請不要因此而放棄，畢竟網路環境充滿惡意，稍有不慎就可能導致重大損失。

先休息一下，下一章我們來學習一下如何使用 Fail2Ban 進行 Nginx 的安全監控。
