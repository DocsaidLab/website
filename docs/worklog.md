---
sidebar_position: 2
---

# 工作日誌

---

:::info
本頁面僅用來記錄我們的工作內容。
:::

---

## 2024

### 8 月

- 更新 **@docusaurus/core@3.5.2**，發現居然沒有向下相容......
  - 只好花點時間去更新錯誤程式區塊。
- 下定決心來排查 OpenCV 的依賴問題，發現我們不是唯一的受害者：
  - [**修复 OpenCV 依赖错误的小工具：OpenCV Fixer**](https://soulteary.com/2024/01/07/fix-opencv-dependency-errors-opencv-fixer.html)
  - 開源專案：[**soulteary/opencv-fixer**](https://github.com/soulteary/opencv-fixer/tree/main)
  - 謝謝[**蘇洋博客**](https://soulteary.com/)的分享，讓我們省下了不少時間。

### 7 月

- 延續上個月的論文馬拉松，我們持續撰寫論文導讀，前前後後大概寫了 50 篇了吧...
  - 完成度：153 %
  - 覺得好累
- MRZScanner：開始進行開發。

### 6 月

- AutoTraderX：完成元富證券 API 串接，轉成開源專案。🎉 🎉 🎉
- 儲值給 OpenAI 的錢花完了，關閉 GmailSummary 每日新聞推送功能。
- 這個月要進行每半年一次的論文馬拉松，本月暫定目標 ~100（想累死誰？）~ 30 篇。
  - 完成度：50 %

### 5 月

- 完成 Text Recognizer 模型。
  - 最後評測的效果不錯，但我們認為這仍然是一個「假裝自己不是過擬合的過擬合模型」。(？？？)
  - 它距離我們心目中的好架構還有一段距離，因此暫時先不公開。
- 探索 Docusaurus 的 Search 功能，測試並加入 Algolia 的搜尋服務。
  - 感謝 [**WeiWei**](https://github.com/WeiYun0912) 所撰寫的文章：
    - [**[docusaurus] 在 Docusaurus 中使用 Algolia 實作搜尋功能**](https://wei-docusaurus-vercel.vercel.app/docs/Docusaurus/Algolia)
- 開發 Text Recognizer 模型，調測模型參數，並進行模型訓練。
- AutoTraderX： 開始進行開發。

### 4 月

- 學習配置 CSS 樣式，調整部落格的外觀。
  - 感謝 [**朝八晚八**](https://from8to8.com/) 所撰寫的文章：
    - [**部落格首頁**](https://from8to8.com/docs/Website/blog/blog_homepage/)
- TextRecognizer：延續 WordCanvas 的開發，繼續推進文字辨識的專案。
- GmailSummary：修改功能：推送每日新聞至網站的技術文件頁面。
- 完成目前所有專案的技術文件。
- 探索 Docusaurus 的 i18n 功能，並同步撰寫英文文件。
- 探索 Docusaurus 的技術文件功能，開始著手撰寫技術文件，並且從 github 搬移內容至此。
- WordCanvas：開發完成，轉成開源專案。🎉 🎉 🎉
- 更改 `blog` 的 Github 專案名稱為 `website`，這樣比較符合實際情況。

### 3 月

某天發現 Google Drive 的檔案下載功能壞了，原本能透過 `gen_download_cmd` 取得的資料變成「一團錯誤的 html」。👻 👻 👻

經過考慮...

後來決定用 [**NextCloud**](https://github.com/nextcloud) 開源架構建立私有雲，專門用於存放資料，並更新過去發布的下載連結。

- GmailSummary：開發完成，轉成開源專案。🎉 🎉 🎉
- DocClassifier：發現疊加多個標準化層對模型效果有顯著提升。（意外發現的...）
- TextRecognizer：前期專案規劃。
- WordCanvas：開始進行開發。
- TextDetector：遇到許多困難，先擱置一下。

### 2 月

- TextDetector：收集公開資料。
- DocClassifier：把 CLIP 帶進來，對模型進行知識蒸餾，效果甚好！
- 探索 Docusaurus 的留言功能，並加入 giscus 留言服務。
  - 感謝 [**不務正業的架構師**](https://ouch1978.github.io/) 所撰寫的文章：
    - [**在文件庫和部落格的文章下方加上 giscus 留言區**](https://ouch1978.github.io/docs/docusaurus/customization/add-giscus-to-docusaurus)

### 1 月

- TextDetector：前期專案規劃。
- DocClassifier：開發完成，轉成開源專案。🎉 🎉 🎉

## 2023

### 12 月

- DocClassifier：開始進行開發。
- DocAligner：開發完成，轉成開源專案。🎉 🎉 🎉
- blog：發現了一個有趣的 Meta 開源項目 [**docusaurus**](https://github.com/facebook/docusaurus)。它提供了一個簡單的方式來建立一個靜態網站，並且可以透過 Markdown 來撰寫內容。所以我決定用它來寫個部落格。
- 捨棄並刪除由 wordpress 所建置的網站，將內容轉移到 github 上的 `blog` 專案中。

### 11 月

- DocClassifier：前期專案規劃。
- DocsaidKit：開發完成，轉成開源專案。🎉 🎉 🎉

### 10 月

- WordCanvas：前期專案規劃。
- DocGenerator：第二階段開發完成，拆分文字合成模組至 WordCanvas 專案。

### 9 月

- DocAligner：開始進行開發。
- DocGenerator：第一階段開發完成。

### 8 月

- DocAligner：前期規劃。
- DocsaidKit：整理常用的工具，開始進行開發。
- 探索 [**Wordpress**](https://wordpress.org/) 的功能，嘗試自行架站並撰寫部落格。
  - 感謝[**諾特斯網站**](https://notesstartup.com/)的無私分享，獲益良多。
- 建立 DOCSAID 的 GitHub 帳號，並且開始規劃一些專案。
