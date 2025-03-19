---
slug: ubuntu-install-ssh
title: 在 Ubuntu 上設定 SSH 伺服器
authors: Z. Yuan
tags: [ubuntu, ssh]
image: /img/2023/0912.webp
description: 伺服器的設定與無密碼登入教學。
---

SSH 是一種網路協議，允許用戶安全地訪問和管理遠程伺服器。

這次我們來記錄一下無密碼登入詳細步驟。

<!-- truncate -->

## 安裝 OpenSSH 伺服器

打開終端機。

輸入以下命令來安裝 OpenSSH 伺服器：

```bash
sudo apt update
sudo apt install openssh-server
```

## 檢查 SSH 伺服器狀態

使用以下命令檢查 SSH 伺服器的狀態：

```bash
sudo systemctl status ssh
```

如果你看到 「Active: active (running)」，那麼 SSH 伺服器已經成功啟動。

## SSH 無密碼登入設定：

### 在用戶端生成 SSH 金鑰對

打開終端機。

輸入以下命令以生成金鑰對：

```bash
ssh-keygen
```

按照提示操作。通常預設的設定就足夠了。在詢問密碼的部分可以直接按 Enter 以建立一個無密碼的金鑰對。

### 把公開金鑰複製到伺服器

使用 ssh-copy-id 命令將公開金鑰複製到伺服器。替換 [username] 與 [server-ip] 為你的伺服器資料。

```bash
ssh-copy-id [username]@[server-ip]
```

例如：

```bash
ssh-copy-id john@192.168.0.100
```

如果伺服器更改了預設的 SSH 埠（如：2222），則使用 -p 參數：

```bash
ssh-copy-id -p 2222 john@192.168.0.100
```

此命令將會提示你輸入伺服器的密碼。

一旦驗證成功，你的公開金鑰就會被添加到伺服器上的 `~/.ssh/authorized_keys` 檔案中。

### 測試無密碼登入

嘗試 SSH 至伺服器：

```bash
ssh [username]@[server-ip]
```

如果一切設定正確，你就可以無需密碼登入伺服器。

## 禁用密碼

有了 SSH 金鑰之後，為了增加安全性，可以考慮禁止密碼認證方式。

這可以在伺服器的 `/etc/ssh/sshd_config` 中設定：

```bash
sudo vim /etc/ssh/sshd_config
```

找到檔案中的 `PasswordAuthentication` 選項，並將其設置為 `no`。

完成上述步驟後，恭喜你可以開心愉快地使用 SSH 了！
