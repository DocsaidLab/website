---
slug: docusaurus-sidebars-enhanced-counting
title: Docusaurus Sidebar のカウント機能の高度な実装：再帰的集計と安定した slug
authors: Z. Yuan
image: /ja/img/2025/0724.jpg
tags: [Docusaurus, Sidebar, Enhancement]
description: サブフォルダの再帰集計と slug の安定性の最適化。
---

かなり前に、Docusaurus の Sidebar が記事数を自動計算するように変更する方法を紹介した記事を書きました。

- 前回の記事：[**Docusaurus の Sidebar で記事数を自動計算する方法**](/ja/blog/customized-docusaurus-sidebars-auto-count)

数ヶ月実際に使ってみて、まだ改善の余地があることに気づいたため、今回は **3 つの改良** を行い、一度に問題点を解決し体験を強化しました。

<!-- truncate -->

## 旧バージョンの問題点

1. **カウントが正確でない**：直接の子階層の Markdown のみを集計し、さらに深い階層のディレクトリを無視している。

   例：ファイル構造が以下のような場合

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

   旧版では `Classic CNNs (1)` としか表示されませんが、期待する結果は `Classic CNNs (4)` です。

2. **slug が安定しない**：Category のリンクは Docusaurus の自動生成機構に依存しており、時々パスが変動する。

   Docusaurus は指定したカテゴリ名に基づいて slug を自動生成するため、カテゴリ名を変更したり新規追加すると slug も変わり、元のリンクが無効になります。

   旧版ではリンクは例えば `papers/category/classic-cnns-11` のように生成されます。

   新しい記事を追加すると、そのリンクが `papers/category/classic-cnns-12` に変わることがあります。こうした変動はユーザーに混乱をもたらし、既に共有されているリンクが無効になる恐れがあります。

   ユーザーの不快感だけでなく、Google のクローラーがこれらのリンクを頻繁に変わるものとして信頼度を下げ、SEO の順位にも悪影響が出ます。

## 3 つの改良点

### 1. 全てのサブフォルダを再帰的にカウント

旧 Sidebar は現在のフォルダ内の .md ファイルのみを集計していましたが、本バージョンでは深い再帰検索を行い、全てのサブフォルダ内の Markdown 数を正確にカウントし、カテゴリラベルの数字が正確になるようにしました。

```js
/**
 * 指定ディレクトリ以下の全ての Markdown (.md) ファイルを再帰的にカウントする。
 * 隠しファイル（先頭が '.'）や '_category_.json' は無視する。
 *
 * @param {string} dirPath - 対象ディレクトリの絶対パス。
 * @returns {number} 見つかった Markdown ファイルの合計数。
 */
function countMarkdownFiles(dirPath) {
  let count = 0;

  // 現在のディレクトリ内の全項目（ファイルとサブディレクトリ）を取得
  for (const name of fs.readdirSync(dirPath)) {
    // 隠しファイルと _category_.json 設定ファイルを除外
    if (name.startsWith(".") || name === "_category_.json") continue;

    const fullPath = path.join(dirPath, name); // フルパスを取得
    const stat = fs.statSync(fullPath); // ファイルまたはディレクトリの状態を取得

    if (stat.isDirectory()) {
      // ディレクトリなら再帰的に中の Markdown 数をカウント
      count += countMarkdownFiles(fullPath);
    } else if (stat.isFile() && name.endsWith(".md")) {
      // Markdown ファイルならカウントを増やす
      count += 1;
    }
  }

  return count;
}
```

### 2. 安定した slug 生成

Docusaurus では各 category のリンクが自動的に URL slug に変換されます。

しかし、空白や日本語、特殊文字を含むパスだとビルドのたびに URL が変わることがあり、「予測不可能」な変動を引き起こします。

連結の**安定性と制御性**を確保するため、パスを安定・予測可能な slug に変換する `toSlug` 関数を実装しました。

```js
/**
 * 相対 POSIX パスを URL 安全な slug に変換する。
 * スラッシュを正規化し、encodeURIComponent を適用してクロスプラットフォームで一貫した slug を保証。
 *
 * @param {string} relPath - 'classic-cnns/vgg' のような相対パス。
 * @returns {string} 'classic-cnns/vgg' のようにエンコードされた URL slug。
 */
function toSlug(relPath) {
  return relPath
    .replace(/\\/g, "/") // Windows のバックスラッシュを POSIX スタイルのスラッシュに置換
    .split("/") // 各ディレクトリ階層に分割
    .map(encodeURIComponent) // 各セグメントを URL エンコード
    .join("/"); // スラッシュで再結合し、安定した slug を作成
}
```

`buildCategoryItem()` 内で、まず `_category_.json` のカスタム slug を優先して読み込み、設定がなければ上記 `toSlug()` で生成したデフォルト値を使うようにします。

```js
const defaultSlug = `/category/${toSlug(relativeDirPath)}`;

const link = {
  type: "generated-index",
  slug: metadata.link?.slug || defaultSlug, // カスタム slug があれば優先、なければフォールバック
  title: metadata.link?.title || baseLabel,
  ...metadata.link, // description などその他のフィールドを保持
};
```

### 3. フォルダの並び順の最適化

Docusaurus の Sidebar 自動生成では、順序を指定しなければディレクトリはデフォルトでアルファベット順に並びます。

しかし、実務上は

- **`_category_.json` を含む分類フォルダを優先表示**（明確に定義されたカテゴリとして）
- **それ以外のフォルダはアルファベット順で並べる**

ことが望まれます。

これにより Sidebar が整理され、利用者が設計済みのテーマグループを優先して閲覧できます。

以下はその並び替えの実装例です。

```js
/**
 * サブディレクトリを並び替える：
 * - `_category_.json` を含むフォルダを先頭に
 * - その他はアルファベット順
 *
 * @param {string} dir - 親ディレクトリの絶対パス
 * @returns {string[]} 並び替えられたサブディレクトリ名の配列
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

      // _category_.json があるフォルダを先に
      if (aHasCategory && !bHasCategory) return -1;
      if (bHasCategory && !aHasCategory) return 1;

      // それ以外はアルファベット順
      return a.localeCompare(b);
    });
}
```

## まとめ

これで Sidebar の「数字」「リンク」「階層構造」がすべて正しく機能するようになりました。

しばらくは安定して運用できるはずです。

頑張ります。

## 成果プレビュー

- 完全なコードはこちら 👉 [**sidebarsPapers.js**](https://github.com/DocsaidLab/website/blob/main/sidebarsPapers.js)
- 実際の動作結果はこちら 👉 [**Paper Notes**](https://docsaid.org/ja/papers/intro)
