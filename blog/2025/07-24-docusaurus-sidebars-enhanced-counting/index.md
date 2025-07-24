---
slug: docusaurus-sidebars-enhanced-counting
title: Docusaurus Sidebar 計數功能的進階實作：遞迴統計與穩定 slug
authors: Z. Yuan
image: /img/2025/0724.jpg
tags: [Docusaurus, Sidebar, Enhancement]
description: 子資料夾遞迴統計與 slug 穩定性優化。
---

很久以前我寫了一篇文章，介紹該怎麼修改讓 Docusaurus 的 Sidebar 自動計算文章數量。

- 前情提要：[**讓 Docusaurus 的 Sidebar 自動計算文章數量**](/blog/customized-docusaurus-sidebars-auto-count)

實際使用數月後，我發現仍有改善空間，於是這次做了 **三個改進**，一次解決痛點並補強體驗。

<!-- truncate -->

## 原版本的問題

1. **計數不夠精確**：僅統計直接子層的 Markdown，忽略更深層目錄。

   舉例：若檔案結構為

   ```bash
   papers/
   ├── classic-cnns/
   │   ├── alexnet.md
   │   ├── vgg/
   │   │   ├── vgg16.md
   │   │   └── vgg19.md
   │   └── resnet/
   │       └── resnet50.md

   ```

   舊版只會顯示 `Classic CNNs (1)`，但我們期待的結果應該是： `Classic CNNs (4)`。

2. **slug 不穩定**：Category 連結依賴 Docusaurus 的自動產生機制，偶爾造成路徑變動。

   由於 Docusaurus 會根據我們給定的分類名稱自動生成 slug，因此當我們隨著分類名稱的變更或新增分類時，slug 就會跟著改變，導致原本的連結失效。

   在舊版中，連結會長得類似這要： `papers/category/classic-cnns-11`。

   當我新增一篇文章後，這個連結就會變成 `papers/category/classic-cnns-12`。這樣的變化會讓使用者感到困惑，因為他們可能已經將這個連結分享給其他人，而現在這個連結已經失效了。

   除了使用者感到不愉快之外，Google 的爬蟲降低了對這些連結的信任度，因為它們經常變動，這會影響網站的 SEO 排名。

## 三大改進

### 1. 遞迴計算所有子資料夾

舊版 Sidebar 僅統計當前資料夾內的 .md 檔案，這版透過深度遞迴搜尋，可以正確計算包含所有子資料夾的 Markdown 數量，確保分類標籤上的數字準確無誤。

```js
/**
 * Recursively counts all Markdown (.md) files under a given directory.
 * Ignores hidden entries (starting with '.') and Docusaurus config files like '_category_.json'.
 *
 * @param {string} dirPath - Absolute path to the target directory.
 * @returns {number} Total number of Markdown files found.
 */
function countMarkdownFiles(dirPath) {
  let count = 0;

  // 讀取當前目錄底下所有項目（檔案與子目錄）
  for (const name of fs.readdirSync(dirPath)) {
    // 忽略隱藏檔與 _category_.json 設定檔
    if (name.startsWith(".") || name === "_category_.json") continue;

    const fullPath = path.join(dirPath, name); // 取得完整路徑
    const stat = fs.statSync(fullPath); // 取得檔案或目錄的狀態資訊

    if (stat.isDirectory()) {
      // 若為資料夾，遞迴統計其內部 Markdown 數量
      count += countMarkdownFiles(fullPath);
    } else if (stat.isFile() && name.endsWith(".md")) {
      // 若為 Markdown 檔案，計入總數
      count += 1;
    }
  }

  return count;
}
```

### 2. 穩定的 slug 生成

在 Docusaurus 中，每個 category 的連結會自動轉換為一組 URL slug。

但這樣做可能會產生「不可預測」的變化，例如路徑中含有空白、中文、或特殊字元時，每次 build 出來的 URL 不一定一致。為了保證連結的**穩定性與可控性**，我們實作一個 `toSlug` 函數，用來將路徑編碼為穩定、可預測的 slug。

```js
/**
 * Converts a relative POSIX path into a URL-safe slug.
 * Ensures cross-platform consistency by normalizing slashes and applying encodeURIComponent.
 *
 * @param {string} relPath - Relative path like 'classic-cnns/vgg'.
 * @returns {string} Encoded URL slug like 'classic-cnns/vgg'.
 */
function toSlug(relPath) {
  return relPath
    .replace(/\\/g, "/") // 替換 Windows 的反斜線為 POSIX 標準分隔符
    .split("/") // 拆成每一層目錄
    .map(encodeURIComponent) // 對每個 segment 做 URL 編碼
    .join("/"); // 再用 / 串起來，形成乾淨穩定的 slug
}
```

接著在 `buildCategoryItem()` 中，我們會優先讀取 `_category_.json` 中自定義的 slug，若沒設定，就 fallback 使用上述 `toSlug()` 產生的預設值：

```js
const defaultSlug = `/category/${toSlug(relativeDirPath)}`;

const link = {
  type: "generated-index",
  slug: metadata.link?.slug || defaultSlug, // 優先使用自訂 slug，否則 fallback
  title: metadata.link?.title || baseLabel,
  ...metadata.link, // 保留其他欄位（如 description）
};
```

### 3. 資料夾排序優化

在 Docusaurus 的 Sidebar 自動生成中，若未控制順序，目錄會以預設字母序排列。

但實務上，我們常希望：

- **優先呈現有 `_category_.json` 的分類資料夾**（代表明確定義的區塊）
- **其餘未定義的目錄再依字母排序**

這樣能讓 Sidebar 更有組織感，也能引導讀者優先瀏覽「設計過的主題群組」。

以下是排序邏輯的實作：

```js
/**
 * Sorts subdirectories such that:
 * - Folders with `_category_.json` come first (i.e., manually defined categories)
 * - Others follow, sorted alphabetically
 *
 * @param {string} dir - Absolute path to parent directory
 * @returns {string[]} Sorted list of subdirectory names
 */
function getSortedSubDirs(dir) {
  return fs
    .readdirSync(dir)
    .filter((name) => {
      const fullPath = path.join(dir, name);
      return !name.startsWith(".") && fs.statSync(fullPath).isDirectory();
    })
    .sort((a, b) => {
      const aHasCategory = fs.existsSync(path.join(dir, a, "_category_.json"));
      const bHasCategory = fs.existsSync(path.join(dir, b, "_category_.json"));

      // 有 _category_.json 的排前面
      if (aHasCategory && !bHasCategory) return -1;
      if (bHasCategory && !aHasCategory) return 1;

      // 否則按字母排序
      return a.localeCompare(b);
    });
}
```

## 小結

現在，Sidebar 的「數字、連結、層次」都搞定了。

應該可以再運作一段時間了吧？

祝我好運。

## 成果預覽

- 完整程式碼請見 👉 [**sidebarsPapers.js**](https://github.com/DocsaidLab/website/blob/main/sidebarsPapers.js)
- 執行結果請直接參考 👉 [**Paper Notes**](https://docsaid.org/papers/intro)
