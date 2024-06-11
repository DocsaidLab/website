---
sidebar_position: 1
---

# 介紹

本專案的核心功能為「**郵件摘要（Email Summary**」。

- [**GmailSummary Github**](https://github.com/DocsaidLab/GmailSummary)

![title](./resources/title.jpg)

:::info
本專案不是開箱即用的 python 工具，而是一個串接 API 的範例。若你剛好有類似的需求，可以參考本專案的說明進行對應的修改和後續開發。
:::

## 概述

在日常生活中，我們經常會因為點擊了 GitHub 上的某個專案的 Watch 選項而開始收到該專案的活動更新郵件。這些更新包括但不限於新功能的討論、issue 回報、pull request (PR) 以及 bug 報告等。

舉個例子，如果你關注了一些 Github 項目，然後採用「所有活動」的方式：

- [**albumentations**](https://github.com/albumentations-team/albumentations): 大約每天收到 10 封郵件。
- [**onnxruntime**](https://github.com/microsoft/onnxruntime): 大約每天收到 200 封郵件。
- [**PyTorch**](https://github.com/pytorch/pytorch): 大約每天收到 1,500 封郵件。

我們可以想像，如果你還關注了更多的項目，那麼你每天將會收到 5,000 封以上的郵件。

＊

**真的會有人會「一封不漏地」閱讀這些郵件嗎？**

＊

反正我不會，通常連看都不看就直接刪除了。

因此，作為尋求效率（偷懶）的工程師，我們必須思考如何解決這個問題。

## 問題拆解

為了解決大量郵件的問題，我們可以將問題拆解為兩個部分：自動下載和自動分析。

### 自動下載

要能從 Gmail 中自動下載這些郵件，然後找出關鍵信息。

我們稍微思考一下可行方案，可能有這些：

1. **使用 Zapier 或 IFTTT 等服務**

   - [**Zapier**](https://zapier.com/)：這是個專注於增強工作效率的自動化平台，它支持連接超過 3,000 種不同的網絡應用程序，包括 Gmail、Slack、Mailchimp 等。這個平台允許用戶通過創建自動化工作流程，來實現各種應用之間的自動交互。
   - [**IFTTT**](https://ifttt.com/)：IFTTT 是一個免費的網絡服務，它允許用戶創建「如果這樣，那麼那樣」的自動化任務，這些任務稱為「Applets」。每個 Applet 都是由一個觸發器和一個動作組成的。當觸發器條件被滿足時，Applet 將自動執行動作。

2. **使用 GmailAPI**

   - [**GmailAPI**](https://developers.google.com/gmail/api)：可以讓我們使用程式來讀取郵件、寫入郵件、搜尋郵件等操作。

:::tip
既然我們都要寫程式了，那就不必再考慮第一種方案了，就用 GmailAPI 上吧。
:::

### 自動分析

取回大量的郵件之後，我們需要對這些郵件進行分析，找出關鍵信息。

這個部分在這個 ChatGPT 的時代下，已經沒有太大的難度。我們可以使用 ChatGPT 來進行自然語言處理，從而找出郵件中的關鍵信息。

## 最後

我們把整個流程分成幾個部分來說明：

1. **自動下載郵件**：使用 GmailAPI。
2. **郵件內容解析**：自行實作邏輯。
3. **郵件內容摘要**：使用 ChatGPT。
4. **輸出＆排程**：使用 Markdown 來輸出，使用 crontab 來排程。

以上就是本專案的核心功能，我們把成果展示在 **輸出示範** 頁面。

接下來，我們將逐一介紹這些部分的功能實現方式。
