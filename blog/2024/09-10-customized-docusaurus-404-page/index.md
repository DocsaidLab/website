---
slug: customized-docusaurus-404-page
title: 自訂 Docusaurus 的 404 頁面
authors: Zephyr
image: /img/2024/0910.webp
tags: [Docusaurus, 404NotFound]
description: 預設的 404 頁面必須改一改！
---

import Layout from '@theme/Layout';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import Translate from '@docusaurus/Translate';

<figure>
![title](/img/2024/0910.webp)
<figcaption>封面圖片：由 GPT-4 閱讀本文之後自動生成</figcaption>
</figure>

---

Docusaurus 是一個由 Meta 開發的靜態網站生成器，用於構建開源文檔網站。它提供了一個簡單的方式來創建和維護網站，並且支持自定義主題和插件。

<!-- truncate -->

如果你不認識 Docusaurus，可以到他們家逛逛：[**Docusaurus 官網**](https://docusaurus.io/)

我們的網站也是透過 Docusaurus 搭建的，但是在網站上線之後，我們發現當用戶訪問不存在的頁面時，Docusaurus 所提供一個 404 頁面似乎還能再優化一下。

因此，我們決定自訂一個 404 頁面，來提高用戶體驗。

## 參考資料

為了解決這個問題，我們首先找到了 Docusaurus 專案內的討論頁面：

- [**How can I customize the 404 page?**](https://github.com/facebook/docusaurus/discussions/6030)

我們根據這個討論頁面，解決了這個問題。

以下是我們的解決方案。

## 匯出 404 頁面設定

:::warning
從這個步驟開始，我們得去修改 Docusaurus 的原始碼。

之後 Docusaurus 若有破壞性的版本更新，該修改可能會導致網站無法正常運行，請確保你有維護網站的能力，再繼續進行。
:::

在 Docusaurus 中，若遇到 404 錯誤，會導向 `@docusaurus/theme-classic` 主題的 `NotFound` 頁面。

所以我們要匯出這個頁面的設定程式碼，執行以下指令：

```bash
npm run swizzle @docusaurus/theme-classic NotFound
```

執行後，選擇 `JavaScript`，接著選擇 `--eject`，這樣就會在 `src/theme` 目錄下生成一個 `NotFound` 目錄。

這裡彈出的原始碼，如果你有興趣，他們位置在這裡：

- [**docusaurus-theme-classic/src/theme/NotFound**](https://github.com/facebook/docusaurus/tree/e8c6787ec20adc975dd6cd292a731d01206afe92/packages/docusaurus-theme-classic/src/theme/NotFound)

目錄下有一個 `index.js` 檔案，這個檔案我們先不管他，接著看到目錄下有另外一個子目錄 `Content`，這個目錄下有一個 `index.js` 檔案，這個檔案就是我們要修改的。

原始檔案內容如下：

```jsx
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from "react";
import clsx from "clsx";
import Translate from "@docusaurus/Translate";
import type { Props } from "@theme/NotFound/Content";
import Heading from "@theme/Heading";

export default function NotFoundContent({ className }: Props): JSX.Element {
  return (
    <main className={clsx("container margin-vert--xl", className)}>
      <div className="row">
        <div className="col col--6 col--offset-3">
          <Heading as="h1" className="hero__title">
            <Translate
              id="theme.NotFound.title"
              description="The title of the 404 page"
            >
              Page Not Found
            </Translate>
          </Heading>
          <p>
            <Translate
              id="theme.NotFound.p1"
              description="The first paragraph of the 404 page"
            >
              We could not find what you were looking for.
            </Translate>
          </p>
          <p>
            <Translate
              id="theme.NotFound.p2"
              description="The 2nd paragraph of the 404 page"
            >
              Please contact the owner of the site that linked you to the
              original URL and let them know their link is broken.
            </Translate>
          </p>
        </div>
      </div>
    </main>
  );
}
```

## 開始修改

我們想要有幾個功能：

1. 想要有個可愛的圖示。
2. 想要有個倒數計時，然後自動跳轉首頁的功能。
3. 想要修改文字內容，告訴讀者更多資訊。

### 倒數計時

首先，我們要加入倒數計時的功能，這個功能我們可以使用 `useEffect` 來實現。

```jsx
import React, { useEffect, useState } from "react";

const [countdown, setCountdown] = useState(15);

useEffect(() => {
  const timer = setInterval(() => {
    setCountdown((prevCountdown) =>
      prevCountdown > 0 ? prevCountdown - 1 : 0
    );
  }, 1000);

  if (countdown === 0) {
    window.location.href = "/";
  }

  return () => clearInterval(timer);
}, [countdown]);
```

倒數計時結束就自動跳轉首頁。

### 圖示

我們去找了一個免費的圖示網站，找了一個可愛的圖示：

- [**Freepik**](https://www.freepik.com/icons/error)

然後下載下來，放在 `static/img` 目錄下，然後在 `index.js` 中引入。

```jsx
<img
  src="/img/error-icon.png"
  alt="Error icon"
  style={{
    width: "150px",
    height: "150px",
    marginBottom: "20px",
    animation: "bounce 1s infinite",
  }}
/>

<style>{`
    @keyframes bounce {
    0%, 100% {
        transform: translateY(0);
    }
    50% {
        transform: translateY(-10px);
    }
    }
`}</style>
```

這是一個圖片元素，用來顯示一個錯誤圖標：

- `src="/img/error-icon.png"`：圖片來源，這裡是本地 /img/error-icon.png。
- `alt="Error icon"`：替代文字，當圖片無法顯示時，這段文字會被呈現。
- `style` 屬性裡定義了圖片的大小和動畫效果：
  - `width: '150px'` 和 `height: '150px'`：設置圖片的寬度和高度。
  - `marginBottom: '20px'`：圖片和下一個元素之間的下邊距為 20 像素。
  - `animation: 'bounce 1s infinite'`：應用了 bounce 動畫，讓圖片垂直彈跳，每次循環持續 1 秒，無限循環。

### 文字內容

原本預設的 404 頁面長這樣：

<div style={{ textAlign: 'center' }}>
<iframe
  src="https://docusaurus.io/non-exist"
  width="80%"
  height="500px"
  center="true"
></iframe>
</div>

---

我們替換成這樣：

```jsx
<p style={{ fontSize: '1.2rem', marginBottom: '20px' }}>
很抱歉，我們無法找到您要的頁面。
</p>
<p style={{ fontSize: '1.2rem', marginBottom: '20px' }}>
網頁結構已經修改了，而您可能選到過時的連結。
</p>
<p style={{ fontSize: '1.2rem', marginBottom: '20px' }}>
請點擊上方導航欄，或許可以找到您要的資訊。
</p>
```

因為我們常在改網頁，路徑被 Google 收錄之後沒有及時更新，導致用戶訪問到錯誤的頁面。

這裡我們告訴用戶：

- 這些頁面很大的可能是存在的，只是被我們換了位置！

希望他們看到這個資訊後，能夠從上方導航欄重新找到他們要的資訊。

---

這個部分可以根據你的需求自由修改。

### 成果展示

如果你想看到會跳動的圖示效果，可以在我們的網站上隨便輸入一個不存在的路徑，就可以看到這個 404 頁面了。

畫面跑起來像是這樣：

<br /><br />

<div className="row" style={{
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    flexDirection: 'column',
    textAlign: 'center',
    animation: 'fadeIn 0.5s ease-in-out',
  }}>

<img
src="/img/error-icon.png"
alt="Error icon"
style={{
      width: '150px',
      height: '150px',
      marginBottom: '20px',
      animation: 'bounce 1s infinite',
    }}
/>

  <div>
    <p style={{ fontSize: '1.2rem', marginBottom: '20px' }}>
      很抱歉，我們無法找到您要的頁面。
    </p>
    <p style={{ fontSize: '1.2rem', marginBottom: '20px' }}>
      網頁結構已經修改了，而您可能選到過時的連結。
    </p>
    <p style={{ fontSize: '1.2rem', marginBottom: '20px' }}>
      請點擊上方導航欄，或許可以找到您要的資訊。
    </p>
  </div>

  <style>{`
    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }
    @keyframes bounce {
      0%, 100% {
        transform: translateY(0);
      }
      50% {
        transform: translateY(-10px);
      }
    }
  `}</style>

</div>

### 完整程式碼

最後，我們附上完整的程式碼：

```jsx title='src/theme/NotFound/Content/index.js'
import Translate from "@docusaurus/Translate";
import Heading from "@theme/Heading";
import clsx from "clsx";
import React, { useEffect, useState } from "react";

export default function NotFoundContent({ className }) {
  const [countdown, setCountdown] = useState(15);

  useEffect(() => {
    const timer = setInterval(() => {
      setCountdown((prevCountdown) =>
        prevCountdown > 0 ? prevCountdown - 1 : 0
      );
    }, 1000);

    if (countdown === 0) {
      window.location.href = "/";
    }

    return () => clearInterval(timer);
  }, [countdown]);

  return (
    <main className={clsx("container margin-vert--xl", className)}>
      <div
        className="row"
        style={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          flexDirection: "column",
          textAlign: "center",
          animation: "fadeIn 0.5s ease-in-out",
        }}
      >
        <img
          src="/img/error-icon.png"
          alt="Error icon"
          style={{
            width: "150px",
            height: "150px",
            marginBottom: "20px",
            animation: "bounce 1s infinite",
          }}
        />

        <div>
          <Heading as="h1" className="hero__title">
            <Translate
              id="theme.NotFound.title"
              description="The title of the 404 page"
            >
              Page Not Found
            </Translate>
          </Heading>
          <p style={{ fontSize: "1.2rem", marginBottom: "20px" }}>
            很抱歉，我們無法找到您要的頁面。
          </p>
          <p style={{ fontSize: "1.2rem", marginBottom: "20px" }}>
            網頁結構已經修改了，而您可能選到過時的連結。
          </p>
          <p style={{ fontSize: "1.2rem", marginBottom: "20px" }}>
            請點擊上方導航欄，或許可以找到您要的資訊。
          </p>
          <p aria-live="polite" style={{ fontSize: "1rem", color: "#555" }}>
            {countdown > 0
              ? `將在 ${countdown} 秒後自動返回首頁...`
              : "即將跳轉..."}
          </p>
        </div>

        <style>{`
          @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
          }
          @keyframes bounce {
            0%, 100% {
              transform: translateY(0);
            }
            50% {
              transform: translateY(-10px);
            }
          }
        `}</style>
      </div>
    </main>
  );
}
```
