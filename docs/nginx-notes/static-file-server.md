---
sidebar_position: 6
---

# Nginx 提供靜態資源

本章以使用 Docusaurus 建立的靜態網站為範例，說明如何透過 Nginx 提供網站服務。

假設你已經準備好一個網域名稱，並將其指向了你的伺服器，例如：

- `your.domain`

:::tip
下方所有指令與設定範例中的 `your.domain`，請務必替換成你實際使用的網域名稱。
:::

## 建構靜態網站

建構前，請確認你的 `docusaurus.config.js` 檔案內的 `url` 已正確設置為你的網域：

```javascript
module.exports = {
  url: "https://your.domain",
  // ...其他設定
};
```

確認完成後，執行以下指令產生靜態檔案：

```bash
DOCUSAURUS_IGNORE_SSG_WARNINGS=true yarn build
```

:::tip
如果不使用 `DOCUSAURUS_IGNORE_SSG_WARNINGS` 環境變數，可能會看到一大堆奇怪的警告訊息，但不影響建構結果。
:::

此指令會在 `build/` 資料夾內生成靜態 HTML、CSS 與 JS 檔案。

接下來，將建構好的檔案上傳到伺服器的指定目錄，並設定檔案權限：

```bash
sudo mkdir -p /var/www/your.domain
sudo rsync -av build/ /var/www/your.domain/
sudo chown -R www-data:www-data /var/www/your.domain
```

## 取得 SSL 憑證

建議使用 Let's Encrypt 來簽發 SSL 憑證，確保網站以 HTTPS 安全提供服務：

```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your.domain
```

## 配置 Nginx

建立一個專用的 Nginx 設定檔：

```bash
sudo vim /etc/nginx/sites-available/your.domain
```

設定內容範例：

```nginx
server {
    listen 80;
    server_name your.domain;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your.domain;

    # SSL 憑證（需要使用 Certbot 簽發）
    ssl_certificate /etc/letsencrypt/live/your.domain/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your.domain/privkey.pem;

    # 設定靜態文件伺服目錄
    root /var/www/your.domain;
    index index.html;

    # 設定 MIME 類型
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

完成後啟用設定檔：

```bash
sudo ln -s /etc/nginx/sites-available/your.domain /etc/nginx/sites-enabled/
```

測試並重新載入 Nginx 設定：

```bash
sudo nginx -t
sudo systemctl reload nginx
```

## 進階配置

在正式環境中建議加入更多安全與效能設定，以下是一個完整的進階設定範例：

```nginx
server {
    listen 80;
    listen [::]:80;
    server_name your.domain;

    # 🔒 自動將 HTTP 重導至 HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name your.domain;

    ssl_certificate /etc/letsencrypt/live/your.domain/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your.domain/privkey.pem;

    # 🔒 安全性標頭
    add_header Strict-Transport-Security "max-age=63072000; includeSubdomains; preload" always;

    # 🔧 限制上傳檔案大小，避免 DDoS
    client_max_body_size 10M;

    # 📁 build 目錄
    root /var/www/your.domain;
    index index.html;

    # 🗃️ 靜態資源緩存
    location ~* \.(jpg|jpeg|png|gif|ico|svg|woff2?|ttf|css|js)$ {
        expires 7d;
        add_header Cache-Control "public, must-revalidate";
    }

    # 🔧 主要路由規則
    location / {
        try_files $uri $uri/ /index.html;
    }

    # 設定 MIME 類型
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # 🔍 自訂日誌檔案位置
    access_log /var/log/nginx/your.domain.access.log main;
    error_log /var/log/nginx/your.domain.error.log warn;
}
```

## 結論

透過以上步驟與進階配置，即可使用 Nginx 提供安全、高效的靜態資源服務。配置完成後，即可透過 `https://your.domain` 來訪問你的網站。

最後，要記得定期檢查 SSL 憑證有效期、更新 Nginx 版本與安全規則。這些都能有效防範潛在的安全風險，確保網站長期穩定運行。
