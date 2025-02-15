---
sidebar_position: 5
---

# Nginx 日誌與監控

將 API 對外開放之後，我們立刻就會收到來自四面八方的惡意請求。

如果你有看去過日誌記錄，應該不難發現有各種 IP 一直嘗試使用不同的用戶名和密碼登入你的伺服器。這是一種常見的暴力破解攻擊，攻擊者會使用自動化工具不斷嘗試登入，直到找到正確的密碼。

雖然我們可以讓密碼的複雜度突破天際，但是總是這樣讓人惦記著也不是辦法。

所以我們用 Fail2ban 這個工具，把那些 IP 通通封鎖掉。

## 什麼是 Fail2ban？

Fail2ban 是一款開源的入侵防禦工具，主要用於監控伺服器日誌並根據規則封鎖惡意 IP。它能有效防禦暴力破解攻擊，例如針對 SSH、HTTP 和其他網路服務的惡意行為。

Fail2ban 的運作基礎是「過濾器」（filters）和「監獄」（jails）兩大概念：

- **過濾器**：用於定義應在日誌中尋找的可疑行為模式
- **監獄**： 則負責將過濾器與封鎖機制結合，當偵測到異常行為時自動執行對應動作。

在 Nginx 的應用場景中，Fail2ban 可透過監視 Nginx 的訪問日誌（access log）與錯誤日誌（error log），辨識可疑請求，例如：

- **頻繁的 404 錯誤**（潛在的掃描攻擊）
- **連續的登入失敗**（暴力破解攻擊）
- **短時間內的高頻請求**（DDoS 或惡意爬蟲）

一旦偵測到異常，Fail2ban 會根據設定，透過 iptables 或 nftables 自動封鎖來源 IP，使其無法繼續攻擊 Nginx 伺服器。

在 Ubuntu 環境中，Nginx 的日誌預設存放於：

- **訪問日誌**：`/var/log/nginx/access.log`
- **錯誤日誌**：`/var/log/nginx/error.log`

Fail2ban 可透過設定監控這些日誌，並根據預設或自訂的過濾規則範本來比對日誌內容。當日誌記錄與惡意行為模式匹配時，Fail2ban 會將相關 IP 加入封鎖清單。

:::info
規則範本通常位於 `/etc/fail2ban/filter.d/`
:::

:::tip
除了保護 Nginx 之外，Fail2ban 也能應用於 SSH、FTP、郵件伺服器等多種服務，提供廣泛的伺服器安全防護功能。本指南主要聚焦於 Nginx 相關的防禦配置。
:::

## Fail2ban 基本設定

在 Ubuntu 上，Fail2ban 可以直接透過 APT 套件管理器安裝，因為它已包含在官方的軟體庫中。請依照以下步驟進行安裝與啟動：

1. **更新系統套件庫**：
   ```bash
   sudo apt update
   ```
2. **安裝 Fail2ban**：

   ```bash
   sudo apt install -y fail2ban
   ```

   這將自動下載並安裝 Fail2ban 及其相依套件。安裝完成後，Fail2ban 會自動啟動並設定為開機自動執行。

3. **確認 Fail2ban 服務狀態**：

   ```bash
   sudo systemctl status fail2ban
   ```

   若安裝成功，應會顯示 `active (running)`，代表 Fail2ban 已正常運行。

4. **檢查 Fail2ban 版本與狀態**：

   - 顯示目前安裝的 Fail2ban 版本：

     ```bash
     fail2ban-client --version
     ```

   - 檢查目前已啟用的監獄們
     ：

     ```bash
     sudo fail2ban-client status
     ```

     預設情況下，Fail2ban 可能僅啟用 SSH 的保護機制，Nginx 相關的防禦則需要額外設定。

---

Fail2ban 的主要設定檔案為 `/etc/fail2ban/jail.conf`，但官方建議：「不要直接修改這個檔案」，以避免日後軟體更新時覆寫你的設定。

因此，我們建立本地設定檔 `jail.local` 來覆蓋預設值。

1. **複製 `jail.conf` 為 `jail.local`**：

   ```bash
   sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
   ```

2. **編輯 `jail.local`**：

   ```bash
   sudo vim /etc/fail2ban/jail.local
   ```

找到 `[DEFAULT]` 區段，設定一些關鍵參數，例如：忽略的 IP、封鎖時間、觀察時間及最大重試次數等相關設定：

```ini
[DEFAULT]
ignoreip = 127.0.0.1/8 ::1    # 設定信任的 IP 地址，這些 IP 不會被封鎖
bantime  = 1h                 # 預設封鎖時間，可設為秒數或時間單位 (如 1h, 10m)
findtime = 10m                # 在此時間內…
maxretry = 5                  # 超過 maxretry 次失敗則封鎖該 IP
```

這些設定的含義如下：

- **ignoreip**：指定哪些 IP 不會被封鎖，例如本機 (`127.0.0.1`) 或其他管理用 IP。
- **bantime**：封鎖時長，預設為 1 小時（`1h`）。此時間內該 IP 會被禁止存取。
- **findtime**：觀察時間範圍，例如設定 `10m` 代表 10 分鐘內累積的失敗行為將被計算。
- **maxretry**：最大重試次數，例如設定 `5` 代表同一個 IP 在 `findtime` 內觸發 5 次規則後會被封鎖。

這些參數為全域設定，適用於所有監獄。我們稍後必須針對 Nginx 設定單獨的封鎖策略，確保惡意攻擊能被更有效地攔截。

完成設定後，重新啟動 Fail2ban 以套用新設定：

```bash
sudo systemctl restart fail2ban
```

至此，Fail2ban 的基本安裝與全域設定已完成。

接下來，我們將進一步配置 Fail2ban 來監控 Nginx 日誌，以防止惡意請求。

## 監控常見攻擊類型

這裡我們將針對幾種常見的 Nginx 攻擊類型，設定 Fail2ban 進行監控與自動封鎖。每種攻擊行為在 Nginx 日誌中通常會有明確的特徵，因此我們先制定「過濾規則」來偵測，並套用對應的封鎖策略。

### 防止惡意爬蟲

許多惡意爬蟲或攻擊工具會使用特定的 User-Agent，例如：

- `Java`
- `Python-urllib`
- `Curl`
- `sqlmap`
- `Wget`
- `360Spider`
- `MJ12bot`

這些 User-Agent 多數來自自動化掃描工具，並不屬於正常用戶。因此，我們可以針對這類 User-Agent 進行即時封鎖。

首先，在 `/etc/fail2ban/filter.d/` 目錄下建立一個新的過濾器檔案：

```bash
sudo vim /etc/fail2ban/filter.d/nginx-badbots.conf
```

填入以下內容：

```ini
[Definition]
failregex = <HOST> -.* "(GET|POST).*HTTP.*" .* "(?:Java|Python-urllib|Curl|sqlmap|Wget|360Spider|MJ12bot)"
ignoreregex =

# failregex 用於匹配惡意請求的正則表達式
# - <HOST> 代表 IP 地址，占位符會被 Fail2Ban 解析
# - "(GET|POST).*HTTP.*" 限定為 HTTP GET 或 POST 請求
# - .* 用於匹配請求內容
# - "(?:Java|Python-urllib|Curl|sqlmap|Wget|360Spider|MJ12bot)"
#   - 這些是常見的爬蟲、掃描器或攻擊工具：
#     - Java: 一般來自 Java-based 爬蟲
#     - Python-urllib: Python 內建的 URL 請求庫
#     - Curl: 命令行 HTTP 請求工具
#     - sqlmap: 自動化 SQL 注入工具
#     - Wget: 下載工具，也可用於爬取網站
#     - 360Spider: 360 搜索引擎的爬蟲
#     - MJ12bot: Majestic SEO 爬蟲

# ignoreregex 是用來排除某些請求的正則表達式
# - 這裡留空，表示不排除任何請求
```

接著設定 Jail，在 `jail.local` 檔案中加入：

```ini
[nginx-badbots]
enabled  = true
port     = http,https
filter   = nginx-badbots
logpath  = /var/log/nginx/access.log
maxretry = 1
bantime  = 86400
```

這樣一來，任何使用這些 User-Agent 的請求都會被立即封鎖 24 小時。

### 防止 404 掃描攻擊

這種攻擊方式會暴力探測不存在頁面，通常會在短時間內對網站進行大量請求。

攻擊者可能會透過腳本不斷存取網站不存在的頁面，以嘗試發現漏洞或敏感檔案，並產生大量 HTTP 404 錯誤。

建立一個新的過濾器檔案：

```bash
sudo vim /etc/fail2ban/filter.d/nginx-404.conf
```

填入以下內容：

```ini
[Definition]
failregex = <HOST> -.* "(GET|POST).*HTTP.*" 404
ignoreregex =

# failregex 用於匹配特定的日誌模式
# - <HOST> 代表 IP 地址（Fail2Ban 會自動替換為實際的來源 IP）
# - "(GET|POST).*HTTP.*"：
#   - 限定請求方法為 GET 或 POST
#   - .* 用於匹配 URL 及 HTTP 協議版本（如 HTTP/1.1）
# - "404" 指定匹配 HTTP 404 狀態碼（表示請求的資源未找到）

# 這條規則用來封鎖頻繁觸發 404 錯誤的 IP，例如掃描不存在頁面的惡意爬蟲或攻擊者
# 適用於 Web 伺服器（如 Nginx、Apache）的日誌分析

# ignoreregex 是用來排除某些請求的匹配規則
# - 這裡留空，表示不排除任何請求
```

接著設定 Jail，在 `jail.local` 檔案中加入：

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

這將在 10 分鐘內觸發 5 次 404 錯誤 的 IP 封鎖 24 小時。

### 防止 DDoS 攻擊

當某個 IP 在短時間內發送大量請求，可能是 DDoS 攻擊或惡意爬取。

建立一個新的過濾器檔案：

```bash
sudo vim /etc/fail2ban/filter.d/nginx-limitreq.conf
```

填入以下內容：

```ini
[Definition]
failregex = limiting requests, excess: .* by zone .* client: <HOST>
ignoreregex =

# failregex 用於匹配 Nginx 日誌中的請求限流（Rate Limiting）事件
# - "limiting requests, excess: .* by zone .* client: <HOST>"
#   - "limiting requests, excess:" 表示請求超過了設置的限制（如 Nginx 的 limit_req_zone 限制）
#   - .* 允許匹配任意數據（如請求數量或超出的詳細資訊）
#   - "by zone .*" 代表 Nginx 配置的限流區域（zone）
#   - "client: <HOST>" 這部分 Fail2Ban 會替換為實際的客戶端 IP 地址
#
# ignoreregex 用於排除特定的匹配模式
# - 這裡留空，表示不排除任何請求
```

這條 failregex 主要是用來匹配 Nginx 限流（limit_req_zone） 觸發的日誌，如：

```log
2024/02/15 12:34:56 [error] 1234#5678: *90123 limiting requests, excess: 20.000 by zone "api_limit" client: 192.168.1.100, server: example.com, request: "GET /api/v1/data HTTP/1.1"
```

對應的 Nginx 配置可能是：

```nginx
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
server {
    location /api/ {
        limit_req zone=api_limit burst=20 nodelay;
    }
}
```

這樣，當某個 IP 短時間內發送過多請求，Nginx 會在日誌中記錄 `"limiting requests, excess: ... client: <IP>"`，Fail2Ban 就能根據 failregex 來匹配並封鎖該 IP。

接著設定 Jail，在 `jail.local` 檔案中加入：

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

這將在 10 分鐘內 超過 10 次觸發 Nginx 速率限制的 IP 封鎖 24 小時。

### 防止暴力登入

如果網站提供登入功能，攻擊者可能會透過暴力嘗試帳號密碼。

建立一個新的過濾器檔案：

```bash
sudo vim /etc/fail2ban/filter.d/nginx-login.conf
```

填入以下內容：

```ini
[Definition]
failregex = <HOST> -.* "(POST|GET) /(admin|wp-login.php) HTTP.*"
ignoreregex =

# failregex 用於匹配惡意請求，目標是常見的管理後台登錄頁面
# - <HOST> 代表 IP 地址（Fail2Ban 會自動替換為實際的來源 IP）
# - "(POST|GET) /(admin|wp-login.php) HTTP.*"
#   - "(POST|GET)" 代表 HTTP 方法（攻擊者可能會用 GET 或 POST 嘗試登入）
#   - "/admin" 是一些網站後台的常見路徑
#   - "/wp-login.php" 是 WordPress 登錄頁面
#   - "HTTP.*" 用於匹配不同的 HTTP 版本，如 HTTP/1.1、HTTP/2
#
# ignoreregex 用於排除某些特定請求
# - 這裡留空，表示不排除任何請求
```

這會匹配 WordPress 或一般網站的管理後台登入嘗試。

接著設定 Jail，在 `jail.local` 檔案中加入：

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

這會在 5 分鐘內超過 5 次登入失敗的 IP 封鎖 24 小時。

### 嘗試存取敏感資料

惡意攻擊者可能會試圖存取 `/etc/passwd`、`/.git`、`/.env` 等敏感路徑。

建立一個新的過濾器檔案：

```bash
sudo vim /etc/fail2ban/filter.d/nginx-sensitive.conf
```

填入以下內容：

```ini
[Definition]
failregex = <HOST> -.* "(GET|POST) /(etc/passwd|\.git|\.env) HTTP.*"
ignoreregex =

# failregex 用於匹配試圖訪問敏感文件的惡意請求
# - <HOST> 代表 IP 地址（Fail2Ban 會自動替換為實際的來源 IP）
# - "(GET|POST) /(etc/passwd|\.git|\.env) HTTP.*"
#   - "(GET|POST)" 代表 HTTP 方法，攻擊者可能會用 GET 或 POST 來探測文件
#   - "/etc/passwd" 是 Linux 系統的密碼文件，攻擊者可能會試圖讀取它
#   - "/.git" 目錄可能包含源代碼與敏感信息，攻擊者可能會試圖下載 `.git` 文件夾
#   - "/.env" 環境變數文件，可能包含 **資料庫密碼、API Key、密鑰**
#   - "HTTP.*" 匹配不同的 HTTP 版本（如 HTTP/1.1、HTTP/2）
#
# ignoreregex 用於排除特定請求的匹配
# - 這裡留空，表示不排除任何請求

```

這條規則用於防止目錄遍歷攻擊、保護 Git 目錄、阻擋對 `.env` 文件的探測，以及防範 Web Scanner 掃描。

接著設定 Jail，在 `jail.local` 檔案中加入：

```ini
[nginx-sensitive]
enabled  = true
port     = http,https
filter   = nginx-sensitive
logpath  = /var/log/nginx/access.log
maxretry = 1
bantime  = 86400
```

這會對任何嘗試存取這些敏感檔案的 IP 立即封鎖 24 小時。

### 針對 API 的攻擊

:::tip
要記得根據你的 api 來調整過濾器。
:::

針對 API 端點（如 `/api/login`、`/api/register`）進行暴力攻擊時，應限制請求頻率。

建立一個新的過濾器檔案：

```bash
sudo vim /etc/fail2ban/filter.d/nginx-api.conf
```

填入以下內容：

```ini
[Definition]
failregex = <HOST> -.* "(POST) /api/(login|register) HTTP.*"
ignoreregex =

# failregex 用於匹配針對 API 登入和註冊的惡意請求
# - <HOST> 代表 IP 地址（Fail2Ban 會自動替換為實際的來源 IP）
# - "(POST) /api/(login|register) HTTP.*"
#   - "(POST)" 限定為 HTTP POST 方法（防止暴力破解登入或濫用註冊）
#   - "/api/login" 是登入 API，可能成為暴力破解攻擊目標
#   - "/api/register" 是註冊 API，攻擊者可能會用來濫發帳號（如機器人註冊）
#   - "HTTP.*" 匹配 HTTP 版本（如 HTTP/1.1、HTTP/2）
#
# ignoreregex 用於排除某些特定請求的匹配
# - 這裡留空，表示不排除任何請求

```

這條規則用於防止暴力破解登入、自動化註冊攻擊，以及提升 API 安全性。

接著設定 Jail，在 `jail.local` 檔案中加入：

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

這會在 1 分鐘內超過 10 次登入嘗試的 IP 封鎖 24 小時。

### 套用新規則

完成設定後，重新啟動 Fail2ban：

```bash
sudo systemctl restart fail2ban
```

並確認 Jail 狀態：

```bash
sudo fail2ban-client status
```

如果想查看某個監獄的狀態，可以使用：

```bash
sudo fail2ban-client status nginx-404 # 替換為你的 jail 名稱
```

到這裡，我們已針對 Nginx 的常見攻擊類型設定 Fail2ban 進行防禦，確保網站安全。

## 測試防禦效果

完成 Fail2ban 的 Nginx 防禦設定後，我們簡單測試一下。

### 使用 `curl` 模擬攻擊

可以在本機或另一台電腦上使用 `curl` 發送惡意請求來觸發 Fail2ban 的封鎖機制。

請注意測試時不要從本機 `localhost` 發送，因為 `127.0.0.1` 可能被 `ignoreip` 設定排除，Fail2ban 不會封鎖它。可以使用其他網路環境的電腦或伺服器來測試，或是暫時將 `ignoreip` 清空。

- **1. 測試 404 掃描攔截**

  模擬攻擊者隨機掃描不存在的頁面：

  ```bash
  for i in $(seq 1 6); do
      curl -I http://<你的伺服器IP或域名>/nonexistentpage_$i ;
  done
  ```

  這將連續請求 6 個不存在的頁面，產生 HTTP 404 錯誤。如果你的規則設定為 5 次 404 即封鎖，Fail2ban 應該會在第 5 或第 6 次時將該 IP 封鎖。
  封鎖後，你再執行 `curl` 請求，應該會發現連線失敗（如超時或連接被拒）。

---

- **2. 測試敏感 URL 攔截**

  攻擊者通常會嘗試存取系統敏感檔案，例如 `/etc/passwd`，來確認伺服器是否存在漏洞：

  ```bash
  curl -I http://<你的伺服器>/etc/passwd
  ```

  如果 Fail2ban 的 `nginx-sensitive` 過濾器正確設定，這次請求應該會被 Fail2ban 偵測到並立即封鎖該 IP（如果 `maxretry = 1`）。

---

- **3. 測試 User-Agent 攔截**

  模擬惡意爬蟲工具，例如 `sqlmap`：

  ```bash
  curl -A "sqlmap/1.5.2#stable" http://<你的伺服器>/
  ```

  如果你的規則中設定了 `sqlmap` 這個 User-Agent，Fail2ban 應該會立即封鎖該 IP。

---

- **4. 測試暴力登入攻擊**

  對 WordPress 或其他登入頁發送多次請求，模擬攻擊者嘗試暴力破解：

  ```bash
  for i in $(seq 1 6); do
      curl -X POST -d "username=admin&password=wrongpassword" http://<你的伺服器>/wp-login.php ;
  done
  ```

  如果你的 `nginx-sensitive` 過濾器設定為 5 次失敗即封鎖，則 Fail2ban 應該會在第 5 或 6 次時將該 IP 封鎖。

### 驗證狀態與日誌

你可以使用以下指令查看監獄內的狀態，假設我們要查看 `nginx-sensitive`：

```bash
sudo fail2ban-client status nginx-sensitive
```

預期輸出：

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

- **Currently banned**：當前被封鎖的 IP 數量。
- **Total failed**：累計的惡意請求次數。
- **Banned IP list**：顯示當前被封鎖的 IP（例如 `203.0.113.45`）。

---

接著，你可以查看 Fail2ban 的日誌，確認封鎖紀錄：

```bash
sudo tail -n 20 /var/log/fail2ban.log
```

預期輸出範例：

```
fail2ban.actions [INFO] Ban 203.0.113.45 on nginx-sensitive
```

表示 `203.0.113.45` 這個 IP 因觸發 `nginx-sensitive` 規則而被封鎖。

### 解除封鎖 IP

在測試或日常管理時，你可能需要手動封鎖或解除封鎖某個 IP。

你可以手動封鎖某個 IP：

```bash
sudo fail2ban-client set nginx-sensitive banip 203.0.113.45
```

這將立即封鎖 `203.0.113.45`，並且該 IP 會被加入 `nginx-sensitive` 的封鎖名單中。

如果發現某個 IP 被誤封鎖，可以解除封鎖：

```bash
sudo fail2ban-client set nginx-sensitive unbanip 203.0.113.45
```

執行該指令後，可以使用 `fail2ban-client status nginx-sensitive` 確認該 IP 是否已從封鎖名單移除。

---

最後，你可以檢查防火牆規則（iptables / nftables）是否正確設定。

Fail2ban 主要透過 `iptables`（或 `nftables`）來封鎖 IP，因此也可以直接檢查防火牆規則：

```bash
sudo iptables -L -n --line-numbers
```

如果封鎖規則生效，你應該會看到類似：

```
Chain f2b-nginx-sensitive (1 references)
num  target     prot opt source               destination
1    REJECT     all  --  203.0.113.45          0.0.0.0/0   reject-with icmp-port-unreachable
```

這表示 `203.0.113.45` 已被封鎖，所有流量都會被拒絕。

## 結論

到這邊，我們設定了針對 Nginx 的防禦規則，算是完成了基本的 Fail2ban 設定。

這個部分請務必按照你的實際需求調整參數，比如不同類型的攻擊可以拆分不同監獄、設置不同的 maxretry/findtime，以及審慎設定 ignoreip 白名單以避免誤封關鍵 IP。

部署完成後，還是要持續關心 Fail2ban 的日誌，確保它正常運行並能有效防禦惡意攻擊。
