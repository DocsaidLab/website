# 工作日誌

---

:::info
本頁面僅用來記錄我們的工作內容。
:::

---

## 2025

### 5 月

- 完成專題論文導讀：[**Face Anti-Spoofing 技術地圖**](https://docsaid.org/blog/fas-paper-roadmap)
- 撰寫論文筆記，目前累計 210 篇。

### 4 月

- 新增 [**技術服務**](https://docsaid.org/services) 頁面。
- 新增內嵌於 Blog 和 Paper 的 CTA 模組。
- 撰寫論文筆記，目前累計 185 篇。

### 3 月

- 完成建置網站第二代後台
  - [x] 會員登入／註冊系統
  - [x] 電子郵件系統
  - [x] API 管理系統
- 撰寫論文筆記，目前累計 175 篇。

### 2 月

- 為了共用 Demo 相關功能，Demo 相關程式重構。
- 網站首頁重構，重新設計排版並增加更多區塊。
- **MRZScanner**：完成 MRZ 二階段模型，並發布 1.0.6 版本。
- 完成 MRZScanner Demo 的功能：[**mrzscanner-demo**](https://docsaid.org/playground/mrzscanner-demo)
- 撰寫論文筆記，目前累計 165 篇。

### 1 月

- 關閉 GCP，更換模型檔案位置，並更新所有下載連結。
  - 更新 Capybara, DocAligner, MRZScanner 模型檔案。
- **DocsaidKit**：完成拆分，功成身退，下架該專案。

## 2024

### 12 月

- 邀請到四位作者加入，分享開發日常。
- 完成 **StockAnalysis Demo** 的功能：[**stock-demo**](https://docsaid.org/playground/stock-demo)
- 新增「作者欄位」至論文筆記，順便美化首頁、論文筆記與部落格樣式。
- 開始建構網頁後台管理系統，資料庫系統及會員註冊系統。
- 撰寫論文筆記，目前累計 150 篇。

### 11 月

- 新增多國語系：[**日本語**](https://docsaid.org/ja/)
- 更新 **@docusaurus/core@3.6.1**，又壞掉啦！
  - 只好花點時間去更新錯誤程式區塊。
- 撰寫論文筆記，目前累計 135 篇。
- **DocumentAI**：同步進行開發。
- **TextRecognizer**：延續 10 月進度，繼續開發。

### 10 月

- **TextRecognizer**：延續 5 月進度，繼續開發。
- 完成模型 Demo 的功能：[**docaligner-demo**](https://docsaid.org/playground/docaligner-demo)
- 把 NextCloud 從我們自己的主機上搬到 GCP，並更新所有下載連結。

### 9 月

- **MRZScanner**：開發完成，轉成開源專案。🎉 🎉 🎉
- **TextDetector**：延續 3 月進度，繼續開發。
- 路過一個網站，覺得好美好厲害，趕緊把它記在小本本上。
  - [**Hello 演算法**](https://www.hello-algo.com/)
- 撰寫論文筆記，目前累計 100 篇。

### 8 月

- **MRZScanner**：部署測試，回爐重造。
- 更新 **@docusaurus/core@3.5.2**，發現居然沒有向下相容......
  - 只好花點時間去更新錯誤程式區塊。
- 下定決心來排查 OpenCV 的依賴問題，發現我們不是唯一的受害者：
  - [**修复 OpenCV 依赖错误的小工具：OpenCV Fixer**](https://soulteary.com/2024/01/07/fix-opencv-dependency-errors-opencv-fixer.html)
  - 開源專案：[**soulteary/opencv-fixer**](https://github.com/soulteary/opencv-fixer/tree/main)
  - 謝謝[**蘇洋博客**](https://soulteary.com/)的分享，讓我們省下了不少時間。
- 撰寫論文筆記，目前累計 90 篇。

### 7 月

- 撰寫論文筆記，累計 80 篇。
- **MRZScanner**：開始進行開發。

### 6 月

- **AutoTraderX**：完成元富證券 API 串接，轉成開源專案。🎉 🎉 🎉
- 儲值給 OpenAI 的錢花完了，關閉 GmailSummary 每日新聞推送功能。
- 撰寫論文筆記，累計 50 篇。

### 5 月

- 完成 Text Recognizer 模型。
  - 最後評測的效果不錯，但我們認為這仍然是一個「假裝自己不是過擬合的過擬合模型」。(？？？)
  - 它距離我們心目中的好架構還有一段距離，因此暫時先不公開。
- 探索 Docusaurus 的 Search 功能，測試並加入 Algolia 的搜尋服務。
  - 感謝 [**WeiWei**](https://github.com/WeiYun0912) 所撰寫的文章：
    - [**[docusaurus] 在 Docusaurus 中使用 Algolia 實作搜尋功能**](https://wei-docusaurus-vercel.vercel.app/docs/Docusaurus/Algolia)
- 開發 Text Recognizer 模型，調測模型參數，並進行模型訓練。
- **AutoTraderX**： 開始進行開發。

### 4 月

- 學習配置 CSS 樣式，調整部落格的外觀。
  - 感謝 [**朝八晚八**](https://from8to8.com/) 所撰寫的文章：[**部落格首頁**](https://from8to8.com/docs/Website/blog/blog_homepage/)
- **TextRecognizer**：延續 WordCanvas 的開發，繼續推進文字辨識的專案。
- **GmailSummary**：修改功能：推送每日新聞至網站的技術文件頁面。
- 完成目前所有專案的技術文件。
- 探索 Docusaurus 的 i18n 功能，並同步撰寫英文文件。
- 探索 Docusaurus 的技術文件功能，開始著手撰寫技術文件，並且從 github 搬移內容至此。
- **WordCanvas**：開發完成，轉成開源專案。🎉 🎉 🎉

### 3 月

某天發現 Google Drive 的檔案下載功能壞了，原本能透過 `gen_download_cmd` 取得的資料變成「一團錯誤的 html」。👻 👻 👻

經過考慮...

後來決定用 [**NextCloud**](https://github.com/nextcloud) 開源架構建立私有雲，專門用於存放資料，並更新過去發布的下載連結。

- **GmailSummary**：開發完成，轉成開源專案。🎉 🎉 🎉
- **DocClassifier**：發現疊加多個標準化層對模型效果有顯著提升。（意外發現的...）
- **TextRecognizer**：前期專案規劃。
- **WordCanvas**：開始進行開發。
- **TextDetector**：遇到許多困難，先擱置一下。

### 2 月

- **TextDetector**：收集公開資料。
- **DocClassifier**：把 CLIP 帶進來，對模型進行知識蒸餾，效果甚好！
- 探索 Docusaurus 的留言功能，並加入 giscus 留言服務。
  - 感謝 [**不務正業的架構師**](https://ouch1978.github.io/) 所撰寫的文章：
    - [**在文件庫和部落格的文章下方加上 giscus 留言區**](https://ouch1978.github.io/docs/docusaurus/customization/add-giscus-to-docusaurus)

### 1 月

- **TextDetector**：前期專案規劃。
- **DocClassifier**：開發完成，轉成開源專案。🎉 🎉 🎉

## 2023

### 12 月

- **DocClassifier**：開始進行開發。
- **DocAligner**：開發完成，轉成開源專案。🎉 🎉 🎉
- **Website**：發現了一個有趣的 Meta 開源項目 [**docusaurus**](https://github.com/facebook/docusaurus)。它提供了一個簡單的方式來建立一個靜態網站，並且可以透過 Markdown 來撰寫內容。所以我決定用它來寫個部落格。
- 捨棄並刪除由 wordpress 所建置的網站，將內容轉移到 github 上的 `website` 專案中。

### 11 月

- **DocClassifier**：前期專案規劃。
- **DocsaidKit**：開發完成，轉成開源專案。🎉 🎉 🎉
- 撰寫論文筆記，累計 20 篇。

### 10 月

- **WordCanvas**：前期專案規劃。
- **DocGenerator**：第二階段開發完成，拆分文字合成模組至 WordCanvas 專案。

### 9 月

- **DocAligner**：開始進行開發。
- **DocGenerator**：第一階段開發完成。
- 撰寫論文筆記，累計 5 篇。

### 8 月

- **DocAligner**：前期規劃。
- **DocsaidKit**：整理常用的工具，開始進行開發。
- 探索 [**Wordpress**](https://wordpress.org/) 的功能，嘗試自行架站並撰寫部落格。
  - 感謝[**諾特斯網站**](https://notesstartup.com/)的無私分享，獲益良多。
- 建立 DOCSAID 的 GitHub 帳號，並且開始規劃一些專案。

### 在此之前

我們輾轉於各種工作之間，日復一日，年復一年。聽著不同老闆說著同樣的夢，嚼著那些食之無味的餅。

無數個日夜趕工的專案，交織了充滿熱情的理想，卻都擺盪在資本市場的愛與不愛之間。

不愛了，就散了。

在青春散盡之前，我們還是想留下一點什麼。

什麼都好，就在這裡，記下我們曾經來過。
