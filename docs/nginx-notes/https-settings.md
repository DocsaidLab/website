---
sidebar_position: 3
---

# Nginx HTTPS 設定

在現代網際網路環境中，網站安全性已成為不可忽視的議題，而 SSL/TLS 憑證則是保護網站與用戶之間傳輸數據的關鍵技術。

Let's Encrypt 是一個免費、自動化且開放的憑證頒發機構（Certificate Authority, CA），為全球數百萬網站提供了加密保護，使 HTTPS 普及化變得更加容易。

既然它免費，那我們肯定得捧個場，對吧？

因此我們採用 Let's Encrypt 在 Nginx 伺服器上配置 HTTPS。

本章節要完成的目標包括：

- **取得 SSL 憑證**：使用 Let's Encrypt 的免費憑證來加密網站流量。
- **強制 HTTPS 傳輸**：將所有使用者的 HTTP 請求自動重定向至 HTTPS（301 永久轉址）。
- **Nginx 反向代理**：在 Nginx 作為反向代理的架構下，確保前端 SSL 正常運作。
- **自動續約**：設定憑證自動續期機制，避免 90 天有效期的憑證過期失效。

:::tip
我們在上一章節中簡單介紹了 Let's Encrypt 這個工具的優缺點。

如果你對它不太熟，可以先去看看我們的補充說明文件：[**Let's Encrypt**](./supplementary/about-lets-encrypt.md)
:::

## Let's Encrypt 安裝

我們採用 Let's Encrypt 的 ACME 協議來獲取 SSL 憑證。

首先，我們需要安裝 Let’s Encrypt 推薦的 ACME 用戶端 **Certbot**。Certbot 可以自動與 Let's Encrypt 交互，申請和續約憑證。

以 Debian/Ubuntu 為例，可直接透過套件庫安裝：

```bash
sudo apt update
sudo apt install -y certbot python3-certbot-nginx
```

上述命令將安裝 Certbot 以及其 Nginx 外掛模組，使 Certbot 可以自動修改 Nginx 設定檔完成憑證部署。

在進行下一步之前，請確保你已經完成以下準備工作：

- **擁有一個網域名稱**，並將其 DNS 記錄指向你的伺服器 IP。
- **開放防火牆的 80 和 443 埠**，讓 HTTP/HTTPS 流量可以進入以進行驗證和後續訪問。

## 申請 Let's Encrypt 憑證

有了 Certbot，我們可以向 Let's Encrypt 申請憑證。

Let's Encrypt 採用自動化的 ACME 驗證機制，會要求你證明對網域的控制權。常用的驗證方式是透過 HTTP 驗證（Certbot 會在你網站上放置一個臨時檔案供 Let's Encrypt 驗證）或 DNS 驗證（較複雜，用於通配符憑證申請）。

這裡我們使用簡單的 HTTP 驗證並透過 Nginx 外掛讓 Certbot 自動配置：

```bash
# 停止 Nginx（若使用 webroot 模式可能需要，但使用 --nginx 外掛通常不需要手動停止）
# sudo systemctl stop nginx

# 執行 Certbot，使用 Nginx 外掛申請憑證
sudo certbot --nginx -d your-domain.com -d www.your-domain.com
```

將其中的 `your-domain.com` 和 `www.your-domain.com` 替換為實際的網域名稱。

執行後，Certbot 會與 Let’s Encrypt 伺服器通訊：

- **驗證網域**：Certbot 可能會臨時修改 Nginx 設定或啟動一個臨時的服務，以回答 Let's Encrypt 的驗證請求。例如，Let's Encrypt 伺服器將嘗試透過 `http://your-domain.com/.well-known/acme-challenge/` 路徑存取驗證檔案。如果驗證成功，表示你對該網域有控制權。
- **取得憑證**：驗證通過後，Let's Encrypt 會簽發 SSL 憑證（包含完整憑證鏈 fullchain.pem 和私鑰檔案 privkey.pem）。Certbot 會將這些檔案儲存至預設路徑（通常在 `/etc/letsencrypt/live/your-domain.com/`）。

在執行過程中，Certbot 可能會詢問：

- 是否同意服務條款、提供電子郵件（用於續約通知）。
- 是否要將 HTTP 流量自動轉向 HTTPS。**建議選擇轉向 (redirect)**，Certbot 將自動為你設定 301 重定向規則，確保使用者以 HTTPS 存取。

:::tip
如果沒有使用 `--nginx` 外掛，亦可使用 `certbot certonly --webroot` 模式來手動取得憑證，然後自行編輯 Nginx 設定。但使用 `--nginx` 可省去手動設定的步驟。
:::

## 配置 Nginx 以啟用 HTTPS

憑證簽發完成後，需要在 Nginx 中開啟 HTTPS（SSL）並載入剛才取得的憑證。

以下是範例設定（假設使用預設安裝路徑的 Certbot）：

```nginx
# 在 Nginx 的設定檔（例如 /etc/nginx/sites-available/your-domain.conf）中加入：

# 80 埠的服務，將所有請求轉址到 HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name your-domain.com www.your-domain.com;
    # 將 HTTP 請求永久轉換為 HTTPS
    return 301 https://$host$request_uri;
}

# 443 埠的服務，提供 HTTPS
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name your-domain.com www.your-domain.com;

    # SSL 憑證檔案路徑（由 Certbot 簽發）
    ssl_certificate      /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key  /etc/letsencrypt/live/your-domain.com/privkey.pem;

    # (以下為反向代理設定範例)
    location / {
        proxy_pass http://127.0.0.1:8000;  # 將請求轉發給後端應用（例如在本機埠 8000 執行的 FastAPI）
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_intercept_errors on;
    }
}
```

每個部分的設定說明如下：

- **80 埠的 HTTP 服務（強制轉向 HTTPS）**

  ```nginx
  server {
      listen 80;
      listen [::]:80;
      server_name your-domain.com www.your-domain.com;
      # 將 HTTP 請求永久轉換為 HTTPS
      return 301 https://$host$request_uri;
  }
  ```

  **這個區塊的作用：**

  1. **監聽 HTTP（80 埠）**

     - `listen 80;`：讓 Nginx 在 IPv4 監聽 80 埠（標準 HTTP 連線）。
     - `listen [::]:80;`：允許 IPv6 連線。

  2. **設定 `server_name`**

     - `server_name your-domain.com www.your-domain.com;`
     - 這告訴 Nginx 此配置適用於 `your-domain.com` 和 `www.your-domain.com`。
     - 這個設定可確保 www 版與非 www 版的流量都能被正確處理。

  3. **強制將 HTTP 轉向 HTTPS**

     - `return 301 https://$host$request_uri;`
     - 這會讓所有 HTTP 請求透過 301 永久重定向，自動轉向 HTTPS。
     - `$host` 代表請求的主機（`your-domain.com` 或 `www.your-domain.com`）。
     - `$request_uri` 代表請求的完整 URI（例如 `/about`）。

---

- **443 埠的 HTTPS 服務**

  ```nginx
  server {
      listen 443 ssl http2;
      listen [::]:443 ssl http2;
      server_name your-domain.com www.your-domain.com;
  ```

  **這個區塊的作用：**

  1. **監聽 HTTPS（443 埠）**

     - `listen 443 ssl http2;`
     - `ssl`：啟用 SSL 加密
     - `http2`：啟用 HTTP/2，提升效能（減少連線延遲）
     - `listen [::]:443 ssl http2;`：允許 IPv6 使用 HTTPS

  2. **設定 `server_name`**

     - 與 HTTP 配置一致，適用於 `your-domain.com` 和 `www.your-domain.com`。

---

- **SSL 憑證設定**

  ```nginx
      # SSL 憑證檔案路徑（由 Certbot 簽發）
      ssl_certificate      /etc/letsencrypt/live/your-domain.com/fullchain.pem;
      ssl_certificate_key  /etc/letsencrypt/live/your-domain.com/privkey.pem;
  ```

  **這個區塊的作用：**

  - 這些是 **Let's Encrypt** 自動生成的憑證路徑：
    - `fullchain.pem`：完整憑證（包含中繼 CA 憑證）
    - `privkey.pem`：私鑰（用於 SSL 加密）

---

- **反向代理 FastAPI**

  ```nginx
      location / {
          proxy_pass http://127.0.0.1:8000;  # 將請求轉發給後端應用（FastAPI）
          proxy_http_version 1.1;
          proxy_set_header Host $host;
          proxy_set_header X-Real-IP $remote_addr;
          proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
          proxy_set_header X-Forwarded-Proto $scheme;

          # 讓 Nginx 處理 FastAPI 回應的錯誤
          proxy_intercept_errors on;
      }
  ```

  **這個區塊的作用：**

  - `proxy_pass http://127.0.0.1:8000;`

    - 假設 FastAPI 在 `127.0.0.1:8000` 執行（啟動指令為 `uvicorn --host 127.0.0.1 --port 8000`）
    - 透過 `proxy_pass` 將 API 請求轉發到本機的 FastAPI 應用

  - **代理請求標頭**

    - `proxy_set_header Host $host;`：保留原請求的 Host 標頭
    - `proxy_set_header X-Real-IP $remote_addr;`：將真實 IP 傳給後端
    - `proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;`：記錄整個代理鏈
    - `proxy_set_header X-Forwarded-Proto $scheme;`：確保後端知道原始請求的協議（HTTP/HTTPS）

  - **錯誤處理**
    - `proxy_intercept_errors on;`
      - 讓 Nginx 攔截後端的錯誤（如 502、503）
      - 可用 `error_page` 指定自訂錯誤頁面

---

設定完成後，重新載入 Nginx 使新配置生效：

```bash
sudo nginx -t  # 測試 Nginx 設定語法是否正確
sudo systemctl reload nginx  # 套用配置變更，重新載入服務
```

## Nginx 反向代理與後端服務

在上述架構中，Nginx 同時擔任 HTTPS 端點和反向代理（Reverse Proxy）的角色。

Nginx 負責處理 TLS 握手與加解密，即 SSL 終止發生在 Nginx 層，後端應用只需處理來自 Nginx 的 HTTP 請求，而無需支援 HTTPS。本機內部通訊可保持輕量化，不需處理額外的加密負擔。

這種做法可簡化後端應用的設定，減少 HTTPS 相關的維護工作。

此外，在使用 Nginx 作為反向代理時，後端應用預設無法直接取得用戶端的真實 IP，而是只會看到 Nginx 伺服器的 IP。

也正因如此，需要透過 `proxy_set_header` 將重要的 HTTP 標頭傳遞給後端應用，以確保後端能正確識別用戶端資訊。

關鍵標頭包括：

- **`X-Forwarded-For`** → 傳遞用戶端的原始 IP 地址
- **`Host`** → 保留原始請求的主機名稱
- **`X-Forwarded-Proto`** → 指示原始請求使用的協議（`http` 或 `https`）

Nginx 配置：

```nginx
proxy_set_header Host $host;
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
proxy_set_header X-Forwarded-Proto $scheme;
```

假設我們的後端是使用 Python 開發的 FastAPI 應用，那麼 FastAPI 預設不會解析代理傳遞的標頭，因此需要啟用 `proxy-headers` 模式，以正確辨識 `X-Forwarded-For` 等資訊：

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --proxy-headers
```

或者，在 FastAPI 內部使用 `StarletteMiddleware.ProxyHeaders`：

```python
from fastapi import FastAPI
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.proxy_headers import ProxyHeadersMiddleware

app = FastAPI()

# 啟用代理標頭解析
app.add_middleware(ProxyHeadersMiddleware)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
```

這樣後端應用即可獲取用戶端的真實 IP、請求協議等資訊，避免誤判。

最後，為了確保安全性，FastAPI 應僅監聽本機介面，避免直接對外開放 HTTP 服務，減少未經授權存取的風險。

最佳做法是：

- **FastAPI 監聽本機埠**
  ```bash
  uvicorn app:app --host 127.0.0.1 --port 8000
  ```
- **Nginx `proxy_pass` 應指向本機埠**
  ```nginx
  location / {
      proxy_pass http://127.0.0.1:8000;
  }
  ```

這樣外部用戶無法直接存取後端 FastAPI 應用，只能透過 Nginx 代理存取，提升安全性；內部流量維持非加密 HTTP，減少 HTTPS 帶來的額外負擔，提升效能。

## 設定憑證自動續約

Let's Encrypt 的憑證有效期僅 90 天，這樣的設計是為了提高安全性，同時促使網站管理員實施自動化的續約機制。手動續約可能導致憑證過期，影響網站的 HTTPS 連線，因此推薦使用 Certbot 自動續約來確保憑證始終有效。

### Certbot 的自動續約機制

Certbot 會根據系統的初始化系統（init system）使用不同的自動續約方式：

對於 systemd 系統（大多數現代 Linux 版本）Certbot 會使用 systemd timer 來管理續約，而不執行 cron 任務。

你可以檢查 systemd timer 是否已啟用：

```bash
systemctl list-timers | grep certbot
```

如果啟用，應該會看到類似：

```
certbot.timer               2025-02-13 02:00:00 UTC   12h left
```

你也可以手動檢查 systemd timer 的狀態：

```bash
systemctl status certbot.timer
```

systemd 會根據 `certbot.timer` 設定，每天執行 2 次續約檢查。只有當憑證到期前 **30 天內** 才會執行真正的續約。

---

對於非 systemd 系統 Certbot 會使用 cron 任務，預設存放於 `/etc/cron.d/certbot`。

系統內 cron 設定可能類似：

```
0 */12 * * * root test -x /usr/bin/certbot -a \! -d /run/systemd/system && perl -e 'sleep int(rand(43200))' && certbot -q renew --no-random-sleep-on-renew
```

這代表每 12 小時執行一次。若系統使用 systemd，則此 cron 任務不會執行。

此外，為了避免伺服器在同一時間進行續約，會隨機延遲最多 12 小時進行靜默續約。

### 確認自動續約是否正常

不論是 **systemd timer** 或 **cron**，你都可以手動測試續約機制：

```bash
sudo certbot renew --dry-run
```

如果輸出沒有錯誤，表示自動續約機制正常。

你也可以檢查已安裝的憑證：

```bash
sudo certbot certificates
```

這會列出所有憑證的有效期限與存放路徑，確保憑證未過期。

### 確保憑證續約後自動生效

Let's Encrypt 續約憑證後，Nginx/Apache 仍然會載入舊憑證，直到它們被重新載入。

可以使用以下方式確保新憑證生效：

```bash
sudo systemctl reload nginx  # 或 systemctl reload apache2
```

你也可以在 `certbot renew` 指令後加入 `--post-hook "systemctl reload nginx"`，讓 Certbot 續約成功後自動重新載入伺服器。

### 確認憑證是否成功續約

你可以使用以下指令，檢查目前的憑證有效期限：

```bash
sudo certbot certificates
```

這會列出所有由 Certbot 管理的憑證，包括每張憑證的到期日和存放位置。確保你的憑證續約後，其到期日已更新。

## 測試與驗證 HTTPS 配置

完成 HTTPS 設定後，請執行以下測試，確保一切正常且符合安全標準：

### 瀏覽器測試

在瀏覽器輸入 `https://your-domain.com`，確認顯示安全鎖 🔒，點擊查看憑證資訊，確保發行者為 **Let's Encrypt**，憑證有效且匹配網域。

嘗試輸入 `http://your-domain.com`，應自動跳轉至 HTTPS。

### 檢查 HTTP 轉 HTTPS 重定向

使用 `curl` 測試：

```bash
curl -I http://your-domain.com
```

確認返回 `301 Moved Permanently`，`Location` 標頭應指向 `https://your-domain.com/...`。

### 確保網站功能正常

確認後端 API / 網站可正常運作，例如：

- FastAPI 的 API 是否能正確回應？
- 網站內部的 HTTPS 連結是否有效？
- 是否考慮 `X-Forwarded-Proto` 來處理 HTTPS 轉向？

### 檢查 Nginx 日誌

查看錯誤與存取日誌，確保沒有異常：

```bash
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

若發現 TLS 連線錯誤，可能是舊設備不支援 **TLS 1.2**，可根據需求調整設定。

## 結論

我們好像做了不少事情。

在本章節中，我們成功地為 Nginx 啟用 HTTPS，確保所有流量都透過 Let's Encrypt 簽發的 SSL 憑證進行加密，並強制將 HTTP 轉向 HTTPS，提供更安全的連線環境。

除了基本的 HTTPS 部署，我們還透過 Nginx 反向代理 FastAPI，確保前端 Nginx 能夠安全地將請求轉發至後端服務，並保留原始請求資訊，如 客戶端 IP、請求協議（HTTP/HTTPS）等，以確保後端應用的完整性與可追溯性。

最後，我們還確保了憑證的自動續約機制，讓 Let's Encrypt 的憑證能夠自動更新，並在續約後自動生效，確保網站的 HTTPS 連線始終有效。

到這裡，網站的安全標準已經完成了嗎？

其實還沒，我們在下個章節來學習一下安全性的進階設定，包括 HSTS、CSP 等安全標頭的配置，以及如何防止常見的網站攻擊。
