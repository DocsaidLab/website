---
slug: gosu-usage
title: 容器下的使用者切換工具：gosu
authors: Z. Yuan
tags: [docker, gosu]
image: /img/2024/0203.webp
description: 這麼好用的工具，肯定要學習一下。
---

Docker 技術已經大量的應用在部署和管理上。

我們通常會將各種應用程式和相關依賴項目打包在一起，確保在不同的環境中一致地運行。

<!-- truncate -->

## 常見問題

然而，如果你很常用，你一定躲不開幾個常見的問題。

### TTY 轉換

一個比較常見的情況是，當你在容器中輸出了某個檔案。

接著離開容器，然後發現檔案的權限都是 root？

這時候你又必須使用 `chown` 來更改檔案的權限。

一次又一次，是不是很煩？

---

又或是在 Docker 容器中使用 sudo 啟動一個需要與終端交互的應用程序，這些應用可能無法正確地檢測到終端，因為 sudo 在創建新會話時可能不會適當地處理終端的所有權和控制。

結果，這些需要與終端交互的應用可能無法正常運行，或者在試圖使用它們時遇到輸入/輸出錯誤。

### 信號轉發

假設你有一個容器，其中運行著 Web 伺服器，例如 Apache 或 Nginx。

通常，你可能會使用命令列工具來管理這個容器，包括啟動和停止容器。在容器內部，如果你使用 sudo 來啟動 Web 伺服器，那麼 sudo 就會建立一個新的進程來運行 Web 伺服器。

問題出現在當你想要停止或重新啟動容器時。容器管理系統會向容器發送訊號（如 SIGTERM），以通知容器內的程序準備停止。然而，如果 Web 伺服器是透過 sudo 啟動的，那麼這個訊號可能只會被傳送到 sudo 進程，而不是實際運行 Web 伺服器的進程。這意味著 Web 伺服器可能不會收到停止訊號，因此無法進行適當的清理和安全關閉。

:::tip
sudo 的設計初衷是提高安全性，允許一般使用者以其他使用者（通常是 root 使用者）的身分執行指令。在這個過程中，sudo 會啟動一個新的會話來執行指令。

這個行為在傳統的作業系統環境中通常沒有問題，但在容器這樣的輕量級虛擬化環境中，它可能導致訊號傳遞問題，因為 sudo 創建的新會話與容器管理系統發送訊號的方式可能不相容 。
:::

## 什麼是 gosu？

- [**gosu GitHub repository**](https://github.com/tianon/gosu)

gosu 是一個專門為了容器設計的工具，它的目的是讓在容器內部執行命令變得更簡單和安全。

當你需要以不同的用戶身份（比如從管理員變成普通用戶）來運行某個程序時，gosu 就能派上用場。它核心工作原理直接借鑒了 `Docker/libcontainer` 啟動容器內應用程式的方式（實際上，它直接使用了 `libcontainer` 代碼庫中的 `/etc/passwd` 處理代碼）。

如果你對它的工作原理不感興趣，那簡單來說，gosu 就像是一個幫手，當你告訴它「請以這個用戶的身份執行這個命令」時，它就會幫你做到，而且做完之後就會退出，不會留下任何痕迹。

### 實際應用場景

使用 gosu 最常見的情境，就是在 Docker 的入口點（ENTRYPOINT）腳本中，把容器從 root 用戶降級到普通用戶，以避免權限問題。

下面是一個具體的例子：

首先，在 Dockerfile 中加入幾行指令來安裝 gosu：

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

然後建立一個入口點腳本 `entrypoint.sh`，它會根據環境變數來動態創建用戶，再使用 gosu 來切換身份運行命令：

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

如果需要實際的範例，可以參考：[**Example training docker**](https://github.com/DocsaidLab/Otter/blob/main/docker/Dockerfile)

## 安全性注意事項

雖然 gosu 在容器環境裡非常方便，但開發者也提醒了可能存在的安全風險。任何允許切換用戶身份的工具，都需要謹慎使用。

就像家裡的門鑰匙一樣，雖然很方便，但如果使用不當，反而可能讓安全大門敞開。因此，使用 gosu 時，團隊應該確保充分了解使用場景，避免在不安全的情況下濫用。

相關的討論串可以參考：[**Keeping the TTY across a privilege boundary might be insecure #37**](https://github.com/tianon/gosu/issues/37)

:::info
我知道你懶得看，所以這裡簡單節錄重點：

**有開發者提出在跨越權限邊界時保留 TTY 可能存在安全隱患。**

當程式在由高權限轉換至低權限執行時，如果沒有建立全新的虛擬終端，原先父程序中未關閉的文件描述符（如標準輸入、輸出等）可能會被新程序繼續使用。可以利用 TIOCSTI 這個 ioctl 呼叫，攻擊者可以向 TTY 緩衝區注入輸入字符，模擬鍵盤輸入，進而執行未經授權的命令，這種漏洞在設計上是被允許的。

舉例來說，下面這段程式碼會將 "id\n" 一個字元一個字元注入標準輸入中：

```c
#include <unistd.h>
#include <sys/ioctl.h>
#include <stdio.h>

int main()
{
    for (char *cmd = "id\n"; *cmd; cmd++) {
        if (ioctl(STDIN_FILENO, TIOCSTI, cmd)) {
            fprintf(stderr, "ioctl failed\n");
            return 1;
        }
    }
    return 0;
}
```

這段程式被視為惡意的原因在於它故意使用 TIOCSTI 這個 ioctl 呼叫來模擬鍵盤輸入，從而在沒有使用者明示操作的情況下向終端注入命令，這相當於模擬使用者在鍵盤上輸入「id」這個命令。透過這種手法可以進行惡意注入或權限提升攻擊，因此被視為具有安全風險的行為。

由於 Docker 分配了新的 TTY 並替換了父 shell，注入的命令無法影響到主機或原始終端，因此風險較低。在非預期的互動式環境下，如果直接在終端中運行 gosu，原始 TTY 未被替換，攻擊者的程式就能成功將命令注入，產生安全漏洞。

不過，這個漏洞主要影響的是非預期使用情境。如果你按照設計用途，例如在 Docker 容器內作為 entrypoint 使用，並且讓 Docker 分配新的 TTY，那麼風險就會大幅降低。

因此，只要在預期環境下正確使用，基本上不必過度擔心。
:::
