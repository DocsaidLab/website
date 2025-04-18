---
slug: fail2ban-settings
title: Fail2ban：保護 SSH 服務
authors: Z. Yuan
tags: [ubuntu, fail2ban]
image: /img/2023/0903.webp
description: 把那些惡意都封鎖在外面。
---

當你成功打開的外部的 SSH 通道之後，你會發現立刻出現一堆惡意連線，想嘗試登入你的主機。

<!-- truncate -->

<div align="center">
<figure style={{"width": "40%"}}>
![attack from ssh](./img/ban_1.jpg)
</figure>
<figcaption>惡意攻擊示意圖</figcaption>
</div>

---

常見的做法是用 Fail2ban 來保護我們的主機，這是一個防止伺服器受到暴力攻擊的軟體。

當系統偵測到可疑的行為（例如：重複的登錄失敗）出現時，Fail2ban 會自動修改防火牆規則來封鎖攻擊者的 IP 地址。

## 1. Fail2ban 安裝

在大部分的 Linux 發行版上，你可以使用包管理工具來安裝 Fail2ban。

這裡由於我的主機是 Ubuntu，所以使用 apt 來安裝：

```bash
sudo apt install fail2ban
```

## 2. 設定

設定檔位於 `/etc/fail2ban/jail.conf`。

但是等等！

這裡不要直接修改這個文件，而是複製一份到 `jail.local` 並修改它：

```bash
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
```

編輯 `jail.local`：

```bash
sudo vim /etc/fail2ban/jail.local
```

這一份檔案內有幾個重要的設定參數，對應的功能如下：

- **ignoreip:** 忽略的 IP 地址或網段，例如： 127.0.0.1/8
- **bantime:** 封鎖時間，以秒為單位 (預設是 600 秒)
- **findtime:** 在這段時間內觀察到多少次失敗嘗試 (預設是 600 秒)
- **maxretry:** 在 findtime 所設定的時間內允許的最大失敗嘗試次數。

## 3. 啟動與監控

啟動 Fail2ban：

```bash
sudo service fail2ban start
```

查看 Fail2ban 的狀態：

```bash
sudo fail2ban-client status
```

## 4. 新增自定義規則

如果你想為特定的服務設定特別的規則，可以在 `jail.local` 裡新增或修改對應的段落，例如針對 SSH 的設定：

```bash
[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
```

## 5. 測試

在修改完設定後，重新啟動 Fail2ban 以應用更改：

```bash
sudo service fail2ban restart
```

然後從另一部機器或使用不同的 IP 進行測試，嘗試多次失敗登錄，看看是否被封鎖。

## 6. 查看

確保定期檢查日誌文件和更新規則，以獲得最佳的保護效果。

```bash
sudo fail2ban-client status sshd
```

## 7. 解除封鎖

如果你剛剛在測試的時候被封鎖了，這裡我們記得把你的測試 IP 解除封鎖：

```bash
sudo fail2ban-client set sshd unbanip <IP地址>
```

## 結語

整個過程只是步驟繁瑣，但也不算複雜。

希望這篇文章能夠幫助你順利完成相關設定。
