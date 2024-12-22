# 敬啟者

非常感謝你加入我們的作者群，以下提供了一些常用的工具與規範，希望能協助你更快上手。

## 啟動網頁

我們目前採用 Docusaurus 作為開發基礎，以下為啟動網頁的基本指令：

```bash
git clone https://github.com/DocsaidLab/website.git
cd website
nvm use node
yarn start
```

### 詳細配置流程

請依照下列步驟檢查並設定你的環境，以確保能順利執行：

1. **確認 Node.js 版本**

   - 我們建議使用 Node.js v22 或以上版本。
   - 若尚未安裝 `nvm`，請先參考 [nvm 官方文件](https://github.com/nvm-sh/nvm#installing-and-updating) 進行安裝。
   - 安裝完成後，執行以下指令檢查並設定 Node.js 版本：
     ```bash
     nvm install 22
     nvm use 22
     ```

2. **安裝 Yarn**

   - 如果尚未安裝 Yarn，請透過以下指令進行安裝：
     ```bash
     npm install -g yarn
     ```

3. **安裝相依套件**

   - 進入專案目錄後，執行以下指令安裝所需的套件：
     ```bash
     yarn install
     ```

4. **檢查網頁啟動埠**

   - 預設情況下，Docusaurus 會在 `http://localhost:3000` 執行。
   - 若該埠已被佔用，可修改 `package.json` 中的啟動指令，或透過環境變數設定新的埠號：
     ```bash
     yarn start --port 3001
     ```

5. **測試網頁啟動**

   - 執行啟動指令後，請確定能透過瀏覽器正常訪問網頁。
   - 如果遇到問題，可嘗試清除快取並重新啟動：
     ```bash
     yarn clear
     yarn start
     ```

6. **額外測試**
   - 若需要測試生產模式，可使用以下指令：
     ```bash
     yarn build
     yarn serve
     ```
   - 網頁將於 `http://localhost:3000` 提供測試版本。

Docusaurus 讓我們能直接使用 Markdown 撰寫網頁內容，並透過 React 進行客製化。
詳細功能使用方式，請參考：[**Docusaurus Markdown Features**](https://docusaurus.io/docs/markdown-features)

## 編寫技術文件

在完成專案的開發，並取得一定的成果後，你可能迫不及待地想和所有人分享你的成果。這時候，你可以透過以下步驟，將你的技術文件發佈到我們的網站上：

以下我們以建立 `DocAligner` 專案的技術文件為例：

1. 在 `docs` 資料夾下，新增一個資料夾，例如 `docs/docaligner`。
2. 在資料夾中新增一個 `index.md` 檔案，內容為：

   ````markdown
   # DocAligner（專案名稱）

   本專案的核心功能稱為「**文件定位（Document Localization）**」。（專案介紹）

   - [**DocAligner Github（專案的 Github）**](https://github.com/DocsaidLab/DocAligner)

   ---

   ![title](./resources/title.jpg)(專案的圖片，可以自己畫或是請 GPT 生成)

   ---

   （固定的程式碼，用來顯示專案的卡片）

   ```mdx-code-block
   import DocCardList from '@theme/DocCardList';

   <DocCardList />
   ```
   ````

3. 在資料夾中新增一個 `resources` 資料夾，用來存放專案的圖片。
4. 其他的技術文件，例如：

   - `docs/docaligner/quickstart.md`: 快速入門指南
   - `docs/docaligner/installation.md`: 安裝指南
   - `docs/docaligner/advanced.md`: 進階使用方法
   - `docs/docaligner/model_arch`：模型架構
   - `docs/docaligner/benchmark`：效能評估
   - ...（其他你想分享的內容）

5. 完成後，發 PR 到 `main` 分支，等待審核。

## 編寫部落格

在開發過程中，你可能會遇到各種大小問題。

你的問題就是其他人的問題，你的解決方案就是其他人的解決方案。

所以，歡迎你將你的問題和解決方案寫成部落格，分享給其他人。

以下是部落格的編寫規範：

1. 在 `blog` 資料夾下，根據日期，找到對應的年份，例如 `blog/2024`，如果沒有，請新增一個。
2. 在年份資料夾下，新增一個資料夾，包含月份日期和標題，例如 `12-17-flexible-video-conversion-by-python`。
3. 在資料夾中新增一個 `index.md` 檔案，以剛才的標題為例，內容為：

   ```markdown
   ---
   slug: flexible-video-conversion-by-python （文章的網址）
   title: 批次影片轉檔
   authors: Zephyr （必須存在於 authors.yml）
   image: /img/2024/1217.webp （請 GPT 生成，並放在 /static/img 資料夾內）
   tags: [Media-Processing, Python, ffmpeg]
   description: 使用 Python 與 ffmpeg 建立指定格式的批次轉換流程。
   ---

   收到一批 MOV 的影音檔，但是系統不支援讀取，要轉成 MP4 才行。

   只好自己動手寫點程式。

    <!-- truncate --> （摘要結束標記）

   ## 設計草稿 （正文開始）
   ```

4. 在資料夾中新增一個 `img` 資料夾，用來存放部落格的圖片。

   為了美觀，使用圖片時，可以在 markdown 文件中使用 html 的語法：

   ```html
   <div align="center"> （圖片置中）
   <figure style={{"width": "90%"}}> （圖片縮放）
   ![img description](./img/img_name.jpg)
   </figure>
   </div>
   ```

5. 完成後，發 PR 到 `main` 分支，等待審核。
6. 最後，請確認你的相關資訊有寫入 `authors.yml` 檔案中，以便我們可以正確顯示你的作者資訊。

   舉例來說，目前檔案內容如下：

   ```yaml
   Zephyr: （用來定位作者的名稱）
     name: Zephyr （網頁顯示的名稱）
     title: Dosaid maintainer, Full-Stack AI Engineer （作者的職稱）
     url: https://github.com/zephyr-sh （作者的github）
     image_url: https://github.com/zephyr-sh.png （作者的頭像）
     socials:
       github: "zephyr-sh" (作者的 github 帳號)
   ```

   更詳細的設定請參考：[**Docusaurus Blog Authors**](https://docusaurus.io/docs/blog#global-authors)

## 編寫論文筆記

讀論文是我們的宿命，寫筆記是為了提醒未來的自己，因為我們的記憶是如此脆弱，連昨天午餐吃了什麼都不記得，更別說讀過的論文。

如果你也想寫筆記，以下是筆記的編寫規範：

1. **論文挑選指南**：

   1. 選擇發表在頂會的論文，例如 CVPR、ICCV、NeurIPS 等，以保證論文的品質。
   2. 若不符合第一項，則選擇引用數高於 100 的論文，表示該論文有一定的參考價值。
   3. 需要付費才能閱讀的論文不要碰。
   4. 選擇在 ArXiv 上公開的論文，方便讓讀者取得全文。

2. **論文年份**：論文年份有幾種，分別是在 ArXiv 上的公開日期、會議的舉辦日期、論文的發表日期等，為了方便查閱，我們選用 ArXiv 上的公開日期。
3. 在 `papers` 資料夾下，根據論文的分支，找到對應的分支，例如 `papers/multimodality`，如果沒有，請新增一個。
4. 在分支資料夾下，新增一個資料夾，包含論文「發表」年月和標題，例如 `2408-xgen-mm`。
5. 在新建資料夾底下建立 `index.md` 檔案，文章需要的影像請放在同一層級的 `img` 資料夾中。
6. 論文筆記撰寫標準格式如下：

   - **標題**：格式為年份、月份、論文名稱
   - **作者**：作者名稱
   - **副標題**：自己想個有趣應景的副標題
   - **論文連結**：論文的完整標題與連結
   - **定義問題**：整理論文作者所定義的問題
   - **解決問題**：詳細說明作者解決問題的方式
   - **討論**：解決問題的有效性或爭議性或實驗結果
   - **結論**：總結論文的重點

7. 一個基本範例如下：

   ```markdown
   ---
   title: "[24.08] xGen-MM" （論文的標題，業界慣例或作者自己定義）
   authors: Zephyr (作者名稱，相關定義請寫在 `/blog/authors.json` 檔案內)
   ---

   ## 又叫做 BLIP-3 （自己想個有趣的副標題）

   [**xGen-MM (BLIP-3): A Family of Open Large Multimodal Models**](https://arxiv.org/abs/2408.08872)（論文完整標題與連結）

   ---

   隨便閒聊。

   ## 定義問題

   整理論文作者所定義的問題。

   ## 解決問題

   詳細說明作者解決問題的方式。

   ## 討論

   解決問題的有效性或爭議性或實驗結果。

   ## 結論

   總結論文的重點。
   ```

8. 撰寫指南：

   1. 如果你覺得論文有問題，請優先懷疑是自己沒看懂或是理解錯誤。
   2. 如果你還是覺得有問題，請先找其他參考資料，不要妄下評論。
   3. 每篇論文一定有所取捨，請把重點放在該論文的啟發，而非論文的缺點。
   4. 請保持客觀中立，不要過度批評或過度讚美。
   5. 如果真的找不到好的部分，請放棄撰寫該篇筆記，你選錯論文了。
   6. 請保持專業，不要使用不當言語或不當圖片。

9. 完成後，發 PR 到 `main` 分支，等待審核。

## 在文章內寫程式

我們採用的是基於 React 的 MDX 語法，所以你可以在文章內直接寫 React 程式碼。

以下是一個簡單的範例，先寫一個 `HelloWorld.js` 組件：

```jsx
import React from "react";

const HelloWorld = () => {
  return <div>Hello, World!</div>;
};

export default HelloWorld;
```

然後在 markdown 文章內引入，雖然這裡是 md 檔，但實際上都會被解析成 mdx 檔，因此可以直接在其中撰寫 React 程式碼：

```mdx
import HelloWorld from "./HelloWorld";

# 標題

<HelloWorld />

## 子標題

其他內容
```

## 多國語系支援

寫完文章後，你愕然發現，我們的網站支援多國語系，可是你不懂其他語言啊！

沒關係，一般來說，有個叫做 Zephyr 的人會幫你做完這件事（？），但你可以自己動手：

1. 把你寫的文章放到對應的 `i18n` 資料夾下，例如：

   - `docs` 的文章放到 `i18n/en/docusaurus-plugin-content-docs/current` 資料夾下，
   - `blog` 的文章放到 `i18n/en/docusaurus-plugin-content-blog/current` 資料夾下，
   - `papers` 的文章放到 `i18n/en/docusaurus-plugin-content-papers/current` 資料夾下。

   ***

   日文的部分要放到 `i18n/ja` 資料夾下，其他語言依此類推。

2. 然後把放在 `i18n` 資料夾下的文章內容改成對應的語言，建議可以直接用 GPTs 翻譯，然後移除多餘的語句和肉眼可見的錯誤。
3. 最後，發 PR 到 `main` 分支，等待審核。

## 最後

我們要提醒的是，儘管現在很多 AI 工具都可以幫助我們生成文章，但一篇真正引人入勝的好文章，必然承載著作者獨特的個人風格和情感表達，而這正是目前 AI 無法完全複製或取代的部分。

AI 模型的核心運作基於統計模型，通過最大似然估計生成內容。這意味著模型更傾向於產生**大眾化、常見的語法與句型**，從而導致其創作出的內容在風格上往往顯得平淡而類似。正因如此，過度依賴模型的結果，可能使創作者失去原本的個性與靈魂，讓文章缺乏深度與感染力。

如果你發現自己在沒有 AI 幫助的情況下無法創作，甚至頭腦一片空白，這就是個警示：表示你得先鞏固自己的寫作能力，掌握基本的創作技巧。不然，你可能只會淪為模型的傳聲筒。

就我們的經驗來看，AI 模型非常適合用在無聊乏味的重複性工作上，例如表格資料分析、數據彙整等，因為這些任務通常具有高標準化的特點，對準確性和效率的要求極高，但對創造性和靈活度的需求相對較低：

> 畢竟表格數據不會因為你富有創意而變得更有趣或改變其實驗結果。

以「論文筆記」為例，AI 能協助我們面對艱澀難懂的數學理論，也可以用在「## 討論」這個章節，梳理文獻中的實驗結果和結論。但無法取代我們對論文的理解和思考，也無法替代我們對論文的批判性思考和深入分析；又以「專案文件」為例，對於枯燥的函數輸入和輸出細節，AI 可以幫我們快速產出大量的技術文件，但對於模型設計理念還是得靠自己。至於「部落格」這種完全依賴創作者的風格和思維的文章，AI 的幫助就更少了。

因此，我們應該明確 AI 的使用範圍，並適時地調整自己的創作策略，以確保文章的品質和獨特性。AI 是一種工具，能幫助我們拓展視野、提升效率，而非限制我們的思維或風格。

請記住，在寫作的過程中，真正動人的文字來源於人心。

AI 只會取代不願意思考的人，我思，故我在。

＊

2024 © Zephyr
