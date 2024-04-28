---
slug: gosu-usage
title: 容器下的使用者切換工具：gosu
authors: Zephyr
tags: [docker, gosu, sudo, container]
image: /img/2024/0203.webp
description: 這麼神的工具，一定要來了解一下！
---

<figure>
![title](/img/2024/0203.webp)
<figcaption>封面圖片：由 GPT-4 閱讀本文之後自動生成</figcaption>
</figure>

---

容器技術已經大量的應用在部署和管理上，它提供了前所未有的靈活性和可靠性。這種技術允許開發者將應用及其依賴打包在一起，確保在不同的環境中一致地運行。

然而，如果你很常用，你一定躲不開幾個常見的問題。

### 可能問題 1：TTY 轉換

一個比較常見的情況是，當我們在容器中輸出了檔案，接著離開容器，發現檔案的權限都是 root，導致我們需要先進行 chown 之類的操作才能正常使用。又或是在 Docker 容器中使用 sudo 啟動一個需要與終端交互的應用程序，比如一個文本編輯器。當你通過 sudo 執行這類應用時，這些應用可能無法正確地檢測到終端（TTY），因為 sudo 在創建新會話時可能不會適當地處理終端的所有權和控制。結果，這些需要與終端交互的應用可能無法正常運行，或者在試圖使用它們時遇到輸入/輸出錯誤。

### 可能問題 2：信號轉發

假設你有一個容器，其中運行著 Web 伺服器，例如 Apache 或 Nginx。 通常，你可能會使用命令列工具（如 docker 或 kubectl）來管理這個容器，包括啟動和停止容器。 在容器內部，如果你使用 sudo 來啟動 Web 伺服器，那麼 sudo 就會建立一個新的進程來運行 Web 伺服器。

問題出現在當你想要停止或重新啟動容器時。 容器管理系統（如 Docker 或 Kubernetes）會向容器發送訊號（如 SIGTERM），以通知容器內的程序準備停止。 然而，如果 Web 伺服器是透過 sudo 啟動的，那麼這個 SIGTERM 訊號可能只會被傳送到 sudo 進程，而不是實際運行 Web 伺服器的進程。這意味著 Web 伺服器可能不會收到停止訊號，因此無法進行適當的清理和安全關閉。

- **為什麼會這樣？**

  sudo 的設計初衷是提高安全性，允許一般使用者以其他使用者（通常是 root 使用者）的身分執行指令。在這個過程中，sudo 會啟動一個新的會話來執行指令。這個行為在傳統的作業系統環境中通常沒有問題，但在容器這樣的輕量級虛擬化環境中，它可能導致訊號傳遞問題，因為 sudo 創建的新會話與容器管理系統發送訊號的方式可能不相容 。

### gosu 的設計理念和核心優勢

gosu 是一個專門為了容器設計的工具，它的目的是讓在容器內部執行命令變得更簡單和安全。當你需要以不同的用戶身份（比如從管理員變成普通用戶）來運行某個程序時，gosu 就能派上用場。這個工具的靈感來自於 Docker 啟動程序的方式，甚至直接使用了 Docker 用來識別用戶的一些技術，這就確保了 gosu 能夠很好地和 Docker 配合使用。

簡單來說，gosu 就像是一個幫手，當你告訴它「請以這個用戶的身份執行這個命令」時，它就會幫你做到，而且做完之後就會退出，不會留下任何痕迹。這種做法避免了一些技術上的麻煩，比如程序之間的信息傳遞和控制終端的問題，讓一切都變得更加順暢。

### 實際應用場景

在 Docker 容器的 ENTRYPOINT 腳本中使用 gosu 是一個非常典型的應用場景，尤其當我們需要從 root 用戶降級到非特權用戶來執行某些操作時。這種做法對於保護容器運行環境的安全至關重要，因為它能有效減少潛在的安全風險。

安裝 gosu 非常簡單，通常只需要在 Dockerfile 中添加幾行指令即可完成安裝和配置。下面的範例展示了如何在 Dockerfile 中安裝 gosu，並通過一個入口點腳本來動態創建用戶和用戶組，然後使用 gosu 以指定的用戶身份執行命令。

```Dockerfile
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

```bash
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

如果需要實際的範例，可以參考：[DocAligner training docker](https://github.com/DocsaidLab/DocAligner/blob/main/docker/Dockerfile)

### 安全性考量

值得一提的是，儘管 gosu 的主要用途是在容器啟動時從 `root` 用戶切換到非特權用戶，但其開發者也強調了在特定情境下使用 gosu 可能存在的安全風險。這是因為任何允許用戶切換的工具，若被不當使用，都可能打開安全漏洞的大門。因此，開發和運維團隊需要對 gosu 的使用場景有充分的了解，並確保其僅在安全的上下文中使用。

### 結論

gosu 以其簡潔、高效和安全的特點，為 Docker 容器中的用戶身份切換提供了一個理想的解決方案。它解決了長期困擾開發和運維團隊的一大痛點，並以最小的學習曲線實現了最大的使用效果。對於那些尋求在容器環境中實現靈活且安全用戶切換的團隊來說，gosu 無疑是一個值得探索的工具。

### 參考資料

- [gosu GitHub repository](https://github.com/tianon/gosu)
