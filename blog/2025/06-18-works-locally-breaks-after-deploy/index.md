---
slug: works-locally-breaks-after-deploy
title: 本機正常，上線就壞掉？多半不是鬼，是路徑
authors: Z. Yuan
image: /img/2025/0618-static-site-baseurl-trap.svg
tags: [docusaurus, deployment, frontend, debugging]
description: 本機正常、部署後 404，通常不是前端在鬧脾氣，而是你的 baseUrl、資源路徑、快取和部署假設一起聯手教育你。
---

有一種 bug 很常見，而且很沒創意。

你在本機開發時看起來一切正常：

- 頁面會開
- 圖片會顯示
- CSS 沒炸
- 連結也點得動

然後一部署上去，整站開始表演：

- 圖片 404
- JS chunk 載不到
- CSS 路徑錯掉
- 子頁正常，重新整理就死
- `/docs/intro` 可以點進去，但直接開就變成伺服器一臉無辜

這種時候，很多人第一反應是：

> 奇怪，明明本機是好的。

對。

本機通常都很好。

因為本機最會包庇你。

<!-- truncate -->

真正的問題通常不是「程式昨天還可以，今天突然不行」。

而是你把一堆**只在本機成立的假設**，帶去了不打算配合你的正式環境。

其中最常見的一類，就是：

> **路徑。**

講白一點：不是鬼，是你把檔案放在 A，卻叫瀏覽器去 B 找。

## 這類問題為什麼特別常見？

因為本機開發環境通常很寬容：

- dev server 會幫你補路由
- 網站常常掛在根路徑 `/`
- 沒有 CDN、反向代理、子目錄部署
- 快取比較少，錯誤比較不持久
- 你習慣用點進去，不習慣直接打 URL 測

所以很多錯誤在本機根本不容易暴露。

一旦到正式環境，條件稍微變一下，問題就出來了。

例如：

- 站不是掛在 `/`，而是掛在 `/blog/` 或 `/docs/`
- 靜態資源有 CDN 前綴
- 伺服器不幫你做 SPA fallback
- 反向代理把前綴路徑吃掉或重寫錯
- 你以為是「絕對路徑」，其實只是「相對於網域根目錄」

這幾件事一湊在一起，整站就會用 404 跟你溝通。

很直接。

## 最典型的坑：你以為 `/img/a.png` 很安全

先看這種寫法：

```md
![cover](/img/cover.png)
```

或這種：

```tsx
<img src="/img/cover.png" alt="cover" />
```

如果你的網站部署在：

```text
https://example.com/
```

那這通常沒事。

但如果你的網站其實部署在：

```text
https://example.com/docs/
```

那瀏覽器看到 `/img/cover.png`，會去找：

```text
https://example.com/img/cover.png
```

不是：

```text
https://example.com/docs/img/cover.png
```

也就是說，你心裡想的是「站內圖片」，瀏覽器理解的是「網域根目錄圖片」。

它沒有錯。

是你想太多。

## `baseUrl` 不是裝飾品

如果你用的是 Docusaurus，這個問題通常跟 `baseUrl` 脫不了關係。

例如：

```ts
const config = {
  url: 'https://example.com',
  baseUrl: '/docs/',
};
```

這代表網站實際掛在 `/docs/` 底下。

此時如果你自己手寫字串路徑：

```tsx
<img src="/img/cover.png" alt="cover" />
```

你其實是在**繞過框架的部署設定**。

比較穩妥的方式通常是交給框架處理，例如：

```tsx
import useBaseUrl from '@docusaurus/useBaseUrl';

export default function Hero() {
  const imageUrl = useBaseUrl('/img/cover.png');
  return <img src={imageUrl} alt="cover" />;
}
```

這樣在 `/` 跟 `/docs/` 下面都比較不容易出事。

如果你明知道網站只會掛在根目錄，那硬寫也不是不行。

但很多 bug 的根源就是：

> 一開始只打算放根目錄，後來部署方式變了，字串路徑還停留在美好舊時代。

程式沒有懷舊能力，所以它會直接壞給你看。

## 不只圖片，JS、CSS、字型也會一起陪葬

很多人第一次遇到時，只注意到圖片不見。

其實更麻煩的通常是這些：

- script chunk 路徑錯
- lazy load 的 asset 指到舊位置
- font URL 沒帶對前綴
- manifest、favicon、social image 指錯

然後畫面看起來就像：

- 樣式一半有、一半沒有
- 按鈕能按，但 icons 消失
- 首頁正常，內頁像被拆過

這種現象很容易讓人誤判成：

- 打包壞了
- 快取沒清
- 某個套件版本爆了

當然，這些也可能是真的。

但先檢查路徑，成本最低。

因為它出事的機率高得很不體面。

## 另一個老坑：相對路徑在巢狀頁面很會背刺你

例如你寫：

```html
<img src="img/cover.png" alt="cover" />
```

如果目前頁面在：

```text
https://example.com/posts/hello/
```

瀏覽器可能會把它解成：

```text
https://example.com/posts/hello/img/cover.png
```

這跟你以為的：

```text
https://example.com/img/cover.png
```

完全不是同一回事。

所以你會看到一種很煩的症狀：

- 首頁正常
- 某些文章頁正常
- 深一層的頁面全部掛掉

因為相對路徑不是壞掉。

它只是非常誠實地照規則做事。

而你沒有真的記得那個規則。

## 還有一種：前端路由正常，直接刷新就死

這也是經典。

例如你有一個 SPA 或靜態站前端路由：

```text
/docs/intro
```

從首頁點進去時正常。

因為是前端接手導頁。

但你直接開這個 URL，或重新整理頁面，伺服器就回你 404。

原因通常不是前端路由壞掉。

而是伺服器根本不知道：

> 這個路徑其實應該回 `index.html`，再交給前端處理。

像 Nginx 常見會需要這種設定：

```nginx
location /docs/ {
    try_files $uri $uri/ /docs/index.html;
}
```

如果沒有類似 fallback，很多 client-side route 都會在「直接進入」時死亡。

這種 bug 也很會騙人，因為：

- 用站內連結進去時正常
- 用書籤、分享連結、重新整理時才壞

於是它可以潛伏很久。

直到有人真的照正常人方式使用網站。

## 怎麼查比較快？先看 Network，不要先拜神

碰到這種問題，第一件事不是重跑 build。

第一件事是打開瀏覽器 DevTools 的 **Network**。

看這幾件事：

1. **哪個 URL 真的 404？**
2. **它是少了前綴，還是多了前綴？**
3. **是根目錄 `/` 被誤用，還是相對路徑跑偏？**
4. **是 HTML 找不到，還是 JS chunk 找不到？**
5. **重新整理內頁時，是不是伺服器沒做 fallback？**

很多人 debug 卡很久，是因為只盯著畫面說「怎麼沒出來」。

畫面不會告訴你答案。

404 URL 會。

差很多。

## 一套比較不容易出事的檢查順序

我通常會這樣查：

### 1. 先確認實際部署位置

先搞清楚網站到底掛在哪：

- `/`
- `/docs/`
- `/product/site/`
- CDN path prefix 後面

不要用想像的。

看實際網址。

### 2. 檢查框架設定

例如 Docusaurus：

- `url`
- `baseUrl`
- `trailingSlash`

例如 Vite / Next / 其他框架，也會有自己的：

- `base`
- `assetPrefix`
- `publicPath`

名字不同，問題差不多。

### 3. 找出所有硬編碼路徑

特別是這些：

```text
/src="/..."
/href="/..."
url(/...)
fetch('/...')
```

它們不一定錯。

但很值得懷疑。

### 4. 測「直接進內頁」

不要只測首頁。

請直接開：

```text
https://example.com/docs/some/page
```

如果只有站內跳轉能用，那問題通常不在「頁面內容」，而在路由或伺服器設定。

### 5. 清掉快取再看一次

尤其是：

- service worker
- CDN cache
- hashed assets 與 HTML 不同步

有時候你修好了，但快取還在堅持它的舊世界觀。

也很常見。

## 一個實際上比較安全的習慣

原則其實不複雜：

> **不要在需要部署彈性的專案裡，到處手寫你自以為不會變的資源路徑。**

能交給框架處理，就交給框架。

能集中封裝，就不要散落全專案。

例如包一層：

```ts
import useBaseUrl from '@docusaurus/useBaseUrl';

export function useAsset(path: string) {
  return useBaseUrl(path);
}
```

之後統一用：

```tsx
const logo = useAsset('/img/logo.svg');
```

這不會讓世界和平。

但至少未來部署位置改掉時，你不需要在專案裡挖一百個 `"/img/..."`。

## 最後

「本機正常、上線壞掉」這件事，很多時候不是神祕 bug。

只是正式環境終於停止縱容你。

而在所有原因裡，**路徑問題**通常是最常見、最便宜、也最值得先懷疑的那一個。

所以如果你下次看到：

- 首頁正常，內頁怪怪的
- 部分圖片消失
- JS/CSS 在 production 失蹤
- 點進去正常，重新整理就 404

先不要急著怪框架。

也先不要把責任推給瀏覽器快取、Node 版本、月亮位置、或某個你其實沒看懂的 bundler 更新。

先去看路徑。

很多時候，問題沒有很深。

只是你剛好走錯目錄而已。
