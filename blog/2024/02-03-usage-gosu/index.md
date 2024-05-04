---
slug: gosu-usage
title: 容器下的使用者切換工具：gosu
authors: Zephyr
tags: [docker, gosu, sudo, container]
image: /img/2024/0203.webp
description: 這麼好用的工具，肯定要學習一下。
---

<figure>
![title](/img/2024/0203.webp)
<figcaption>封面圖片：由 GPT-4 閱讀本文之後自動生成</figcaption>
</figure>

---

Docker 技術已經大量的應用在部署和管理上，這種技術允許開發者將應用及其依賴打包在一起，確保在不同的環境中一致地運行。

## 常見問題

然而，如果你很常用，你一定躲不開幾個常見的問題。

### TTY 轉換

一個比較常見的情況是，當你在容器中輸出了某個檔案。

接著離開容器，然後發現檔案的權限都是 root？

這時候你又必須使用 `chown` 來更改檔案的權限，一次又一次，有夠煩。

---

又或是在 Docker 容器中使用 sudo 啟動一個需要與終端交互的應用程序，這些應用可能無法正確地檢測到終端（TTY），因為 sudo 在創建新會話時可能不會適當地處理終端的所有權和控制。

結果，這些需要與終端交互的應用可能無法正常運行，或者在試圖使用它們時遇到輸入/輸出錯誤。

### 信號轉發

假設你有一個容器，其中運行著 Web 伺服器，例如 Apache 或 Nginx。

通常，你可能會使用命令列工具來管理這個容器，包括啟動和停止容器。在容器內部，如果你使用 sudo 來啟動 Web 伺服器，那麼 sudo 就會建立一個新的進程來運行 Web 伺服器。

問題出現在當你想要停止或重新啟動容器時。容器管理系統會向容器發送訊號（如 SIGTERM），以通知容器內的程序準備停止。然而，如果 Web 伺服器是透過 sudo 啟動的，那麼這個訊號可能只會被傳送到 sudo 進程，而不是實際運行 Web 伺服器的進程。這意味著 Web 伺服器可能不會收到停止訊號，因此無法進行適當的清理和安全關閉。

:::tip
sudo 的設計初衷是提高安全性，允許一般使用者以其他使用者（通常是 root 使用者）的身分執行指令。在這個過程中，sudo 會啟動一個新的會話來執行指令。這個行為在傳統的作業系統環境中通常沒有問題，但在容器這樣的輕量級虛擬化環境中，它可能導致訊號傳遞問題，因為 sudo 創建的新會話與容器管理系統發送訊號的方式可能不相容 。
:::

## 什麼是 gosu？

- [**gosu GitHub repository**](https://github.com/tianon/gosu)

gosu 是一個專門為了容器設計的工具，它的目的是讓在容器內部執行命令變得更簡單和安全。當你需要以不同的用戶身份（比如從管理員變成普通用戶）來運行某個程序時，gosu 就能派上用場。它核心工作原理直接借鑒了 `Docker/libcontainer` 啟動容器內應用程式的方式（實際上，它直接使用了 `libcontainer` 代碼庫中的 `/etc/passwd` 處理代碼）。

如果你對它的工作原理不感興趣，那簡單來說，gosu 就像是一個幫手，當你告訴它「請以這個用戶的身份執行這個命令」時，它就會幫你做到，而且做完之後就會退出，不會留下任何痕迹。

### 實際應用場景

在 Docker 容器的 ENTRYPOINT 腳本中使用 gosu 是一個非常典型的應用場景，尤其當我們需要從 root 用戶降級到非特權用戶來執行某些操作時。這種做法對於保護容器運行環境的安全至關重要，因為它能有效減少潛在的安全風險。

安裝 gosu 非常簡單，通常只需要在 Dockerfile 中添加幾行指令即可完成安裝和配置。下面的範例展示了如何在 Dockerfile 中安裝 gosu，並通過一個入口點腳本來動態創建用戶和用戶組，然後使用 gosu 以指定的用戶身份執行命令。

```Dockerfile title="Dockerfile"
# 基於一個已有的基礎鏡像
FROM some_base_image:latest

WORKDIR /app

# 安裝gosu
RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

# 準備入口點腳本
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["default_command"]
```

`entrypoint.sh` 腳本的範例內容如下，它根據環境變量 USER_ID 和 GROUP_ID 動態創建用戶，然後用 gosu 執行命令：

```bash title="entrypoint.sh"
#!/bin/bash
# 檢查是否設置了USER_ID和GROUP_ID環境變量
if [ ! -z "$USER_ID" ] && [ ! -z "$GROUP_ID" ]; then
    # 創建用戶組和用戶
    groupadd -g "$GROUP_ID" usergroup
    useradd -u "$USER_ID" -g usergroup -m user
    # 使用gosu執行命令
    exec gosu user "$@"
else
    exec "$@"
fi
```

如果需要實際的範例，可以參考：[**DocClassifier training docker**](https://github.com/DocsaidLab/DocClassifier/blob/main/docker/Dockerfile)

### 安全性考量

儘管 gosu 的主要用途是在容器啟動時從 `root` 用戶切換到非特權用戶，但其開發者也強調了在特定情境下使用 gosu 可能存在的安全風險。

這是因為任何允許用戶切換的工具，若被不當使用，都可能打開安全漏洞的大門。因此，開發和運維團隊需要對 gosu 的使用場景有充分的了解，並確保其僅在安全的環境中使用。
