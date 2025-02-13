# Let's Encrypt

## 什麼是 Let's Encrypt？

:::info
已經熟悉 Let's Encrypt 的讀者可以直接跳過這一章節。
:::

Let's Encrypt 是由 Internet Security Research Group（ISRG） 推出的 免費、開放、自動化 的 SSL/TLS 憑證頒發機構（Certificate Authority, CA），於 2015 年 12 月 正式進入公測，並迅速成長為全球最大的憑證簽發機構之一。

其核心目標是提升網站安全性，讓 HTTPS 加密技術變得普及，降低憑證管理的技術與成本門檻。Let's Encrypt 僅提供 Domain Validation（DV） 類型的 SSL/TLS 憑證，透過 ACME（Automatic Certificate Management Environment）協議 自動化發放與續期，讓網站管理員無需手動處理繁瑣的憑證簽署與續期流程。

Let's Encrypt 獲得來自 Mozilla、Google、Cisco、Akamai、電子前哨基金會（EFF） 及眾多科技公司的支持，並已成為 HTTPS 普及化的重要推動者。

:::tip
**Domain Validation（DV）** 類型的 SSL/TLS 憑證是一種基本的加密憑證，僅驗證網站對其網域的控制權，而不包含企業或個人身份資訊。它的主要目的在於 確保網站與訪客之間的通訊加密，防止流量被攔截或竄改。

DV 憑證適合 個人網站、部落格、小型企業，或任何需要基本 HTTPS 加密的應用。如果網站不需要顯示企業認證（如 OV 或 EV 憑證提供的組織資訊），DV 憑證是一個快速且簡單的選擇。
:::

## Let's Encrypt 的特點

- **1. 免費**

  傳統 SSL/TLS 憑證通常需要支付費用，且價格隨憑證類型（DV、OV、EV）與信任等級而有所不同。Let's Encrypt 完全免費，使企業、個人網站、開發者都能 零成本獲取受信任的 SSL/TLS 憑證，大幅降低網站部署 HTTPS 的門檻。

---

- **2. 自動化**

  Let's Encrypt 採用 ACME（Automatic Certificate Management Environment，自動化憑證管理環境）協議，讓憑證的 申請、驗證、簽發、安裝與續期 都能自動完成。使用 ACME 客戶端（如 Certbot、acme.sh），網站管理員無需手動提交 CSR（憑證簽署請求） 或與憑證頒發機構（CA）來回交涉，大幅簡化憑證管理流程。

---

- **3. 開放與透明**

  Let's Encrypt 的運作機制 完全開放與透明：

  - **開源技術**：Let's Encrypt 的核心軟體、ACME 協議、驗證機制等皆為開源，讓任何人都可審查與改進。
  - **公開記錄**：所有發出的憑證都會記錄於 Certificate Transparency（CT） 公開日誌，可供查詢與審核，防止惡意濫用或未授權憑證。

---

- **4. 廣泛支援與兼容性**

  Let's Encrypt 憑證已獲得 所有主要瀏覽器與作業系統 的信任，包括：

  - **桌面端**：Chrome、Firefox、Safari、Microsoft Edge
  - **行動端**：Android（7.1.1 以上）、iOS
  - **伺服器/作業系統**：Linux（Ubuntu、Debian、CentOS、RHEL 等）、Windows Server、macOS

  :::warning
  極舊版本的 Android（7.1.1 以下）可能無法信任 Let's Encrypt 預設的根憑證 ISRG Root X1，但可透過開啟「信任舊根 DST Root CA X3」的方式維持相容性。
  :::

---

- **5. 有效期 90 天**

  Let's Encrypt 憑證的 有效期限為 90 天（傳統商業 CA 憑證通常為 1 至 2 年）。此設計的優勢包括：

  - **提高安全性**：即使憑證私鑰洩漏，短期有效期可降低風險。
  - **鼓勵自動續期**：官方建議使用自動續期機制，如 certbot renew 或 ACME 客戶端排程，避免手動更新的風險。
  - **符合現代最佳實踐**：許多瀏覽器和安全機構已推動縮短憑證有效期，以增強安全性和靈活性。

## Let's Encrypt 運作方式

Let's Encrypt 採用 ACME（Automatic Certificate Management Environment，自動化憑證管理環境）協議 來自動發放與續期 SSL/TLS 憑證，整個過程無需人工介入，讓網站管理員能夠輕鬆部署 HTTPS。

ACME 流程如下：

1. **用戶安裝 ACME 客戶端**（如 `Certbot`、`acme.sh` 或其他 ACME 相容工具）。
2. **ACME 客戶端向 Let's Encrypt 發送憑證申請**，請求加密特定的網域名稱（如 `example.com`）。
3. **Let's Encrypt 進行網域驗證**，確認用戶對該網域的控制權，驗證方式包括：
   - **HTTP 驗證（HTTP-01）**：Let's Encrypt 提供一個隨機驗證檔案，ACME 客戶端需將其放置在網站伺服器的 `.well-known/acme-challenge/` 目錄下。Let's Encrypt 伺服器會嘗試存取該文件，以確認該網域的控制權。
   - **DNS 驗證（DNS-01）**：適用於通配符憑證（Wildcard SSL）或無法使用 HTTP 驗證的情況。用戶需在 DNS 記錄中新增特定的 TXT 記錄，Let's Encrypt 伺服器會查詢該 DNS 記錄以完成驗證。
   - **TLS-ALPN 驗證（TLS-ALPN-01）**：適用於不使用 HTTP 伺服器的環境，例如僅提供 TLS 服務的應用程式。這種方式會在伺服器的 443 埠 上建立一個臨時的 TLS 憑證來完成驗證。
4. **驗證成功後，Let's Encrypt 簽發 SSL/TLS 憑證**，憑證將存放於 `/etc/letsencrypt/live/yourdomain/` 目錄（視 ACME 客戶端而定）。
5. **ACME 客戶端自動續期**：一般來說，客戶端會每 60 天自動檢查是否需要續期，確保憑證不會過期（憑證有效期為 90 天）。

## 如何安裝與使用？

Let's Encrypt 支援多種 ACME 客戶端，其中最受歡迎的是 **Certbot**，由 Electronic Frontier Foundation（EFF）開發。

- **1. 安裝 Certbot**

  在 Ubuntu/Debian：

  ```bash
  sudo apt update
  sudo apt install certbot python3-certbot-nginx
  ```

  在 CentOS/RHEL：

  ```bash
  sudo yum install certbot python3-certbot-nginx
  ```

- **2. 申請 SSL 憑證**

  假設網站是透過 Nginx 伺服器運行，使用以下指令申請憑證：

  ```bash
  sudo certbot --nginx -d example.com -d www.example.com
  ```

  這將自動修改 Nginx 設定並啟用 HTTPS。

- **3. 設定自動續期**

  Certbot 預設會自動續期憑證，建議手動測試：

  ```bash
  sudo certbot renew --dry-run
  ```

  若無誤，系統會定期自動執行續期。

## 缺點其實也不少

Let's Encrypt 雖然提供免費且自動化的 SSL 憑證，但也有一些限制：

- **1. 只能提供 Domain Validation（DV）憑證**

  Let's Encrypt 只提供 DV 憑證，也就是說，它無法提供 OV（企業驗證）或 EV（擴展驗證）憑證，這意味著無法顯示企業名稱（部分銀行、電商或大型企業網站可能需要 EV 憑證來提升信任度）。

  此外，也不適合需要高信任度的業務網站，如金融機構、政府網站等，因為這些機構通常需要 OV 或 EV 憑證來證明其合法性。

---

- **2. 憑證有效期僅 90 天**

  相較於傳統 CA（憑證頒發機構） 提供的 1-2 年憑證，Let's Encrypt 憑證的有效期只有 90 天，若自動續期機制出錯，網站可能無法正常運行，導致使用者無法訪問。

---

- **3. 不支援 IP 位址 SSL 憑證**

  Let's Encrypt 不提供針對 IP 位址的 SSL 憑證，它僅支援基於網域名稱（FQDN，Fully Qualified Domain Name）的憑證。

---

- **4. 依賴 ACME 客戶端與伺服器配置**

  雖然 Let's Encrypt 支援自動化，但用戶需要安裝並配置 ACME 客戶端（如 `Certbot`），對於 Windows 伺服器或自訂 Web 服務架構，這可能會增加技術門檻。

---

- **5. 信任度在某些情境下較低**

  由於 Let's Encrypt 憑證可以快速、自動化取得，部分惡意網站也可能利用它來獲得 HTTPS 鎖頭標誌，使用戶誤以為該網站是安全的。但其實釣魚網站、惡意網站也可能使用 Let's Encrypt，使訪問者降低警戒心。

---

- **6. 不提供技術支援**

  最後，Let's Encrypt 不提供不提供專屬客服支援，只能透過官方文件與社群論壇尋求解決方案，這對於一些技術不熟練的用戶來說可能會有困難。

---

雖然 Let's Encrypt 降低了 SSL/TLS 加密的門檻，讓 HTTPS 普及，但它仍然不是所有網站的最佳解，使用者應根據自身需求來選擇適合的 SSL 憑證。

## 結論

Let's Encrypt 透過免費、自動化和開放的方式，讓 HTTPS 普及化，推動網際網路安全的發展。對於個人網站、部落格和中小企業來說，這是一個理想的選擇，能夠輕鬆獲取 SSL/TLS 加密保護。

在我們大致理解 Let's Encrypt 的特點和運作方式後，接下來我們將進入 Nginx 的 HTTPS 配置，並使用 Let's Encrypt 憑證來保護網站數據的傳輸。
