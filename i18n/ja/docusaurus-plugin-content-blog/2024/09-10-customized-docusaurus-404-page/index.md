---
slug: customized-docusaurus-404-page
title: Docusaurus の 404 ページをカスタマイズする
authors: Z. Yuan
image: /ja/img/2024/0910.webp
tags: [Docusaurus, 404NotFound]
description: デフォルトの 404 ページを改善します！
---

Docusaurus は、Meta が開発した静的サイトジェネレーターで、オープンソースのドキュメントサイトを構築するために使用されます。

簡単にウェブサイトを作成・管理できるだけでなく、カスタムテーマやプラグインもサポートしています。

<!-- truncate -->

もし Docusaurus を知らないなら、こちらを訪問してみてください：[**Docusaurus 公式サイト**](https://docusaurus.io/)

私たちのウェブサイトも Docusaurus を使って構築されています。しかし、サイトを公開した後、存在しないページにアクセスしたときに表示される Docusaurus のデフォルト 404 ページが、少し物足りないと感じました。

そこで、ユーザー体験を向上させるために、404 ページをカスタマイズすることにしました。

## 参考資料

この問題を解決するため、まず Docusaurus のプロジェクト内のディスカッションページを見つけました：

- [**How can I customize the 404 page?**](https://github.com/facebook/docusaurus/discussions/6030)

このディスカッションを基に問題を解決しました。

以下はその解決策です。

## 404 ページ設定のエクスポート

:::warning
ここから Docusaurus のソースコードを変更する必要があります。

今後 Docusaurus が破壊的なアップデートを行った場合、この変更が原因でサイトが正常に動作しなくなる可能性があります。サイトを維持管理する能力があることを確認してから、作業を続行してください。
:::

Docusaurus では、404 エラーが発生すると、`@docusaurus/theme-classic`テーマの`NotFound`ページが表示されます。

そのため、このページの設定コードをエクスポートする必要があります。以下のコマンドを実行します：

```bash
npm run swizzle @docusaurus/theme-classic NotFound
```

実行後、`JavaScript`を選択し、その後`--eject`を選びます。これにより、`src/theme`ディレクトリ内に`NotFound`ディレクトリが生成されます。

生成されたソースコードは興味があれば以下で確認できます：

- [**docusaurus-theme-classic/src/theme/NotFound**](https://github.com/facebook/docusaurus/tree/e8c6787ec20adc975dd6cd292a731d01206afe92/packages/docusaurus-theme-classic/src/theme/NotFound)

ディレクトリ内には`index.js`というファイルがありますが、これは一旦置いておきます。ディレクトリ内の`Content`というサブディレクトリに`index.js`というファイルがあり、これを編集します。

元のファイルの内容は以下の通りです：

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

## 編集を開始

以下の機能を追加したいと考えました：

1. 可愛らしいアイコンを追加する。
2. カウントダウン機能を追加し、終了後に自動的にホームページへリダイレクトする。
3. テキスト内容を変更して、読者により多くの情報を提供する。

### カウントダウン

まず、カウントダウン機能を追加します。この機能は`useEffect`を使って実現できます。

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

カウントダウンが終了すると、自動的にホームページにリダイレクトされます。

### アイコン

無料のアイコンサイトで可愛いアイコンを探し、それを`static/img`ディレクトリに配置して`index.js`にインポートします。

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

### テキスト内容

元の 404 ページは以下のように表示されていました：

<div style={{ textAlign: 'center' }}>
<iframe
  src="https://docusaurus.io/non-exist"
  width="80%"
  height="500px"
  center="true"
></iframe>
</div>

---

新しいテキストは以下のようになります：

```jsx
<p style={{ fontSize: '1.2rem', marginBottom: '20px' }}>
お探しのページは見つかりませんでした。
</p>
<p style={{ fontSize: '1.2rem', marginBottom: '20px' }}>
ウェブサイトの構造が変更されている可能性があります。
</p>
<p style={{ fontSize: '1.2rem', marginBottom: '20px' }}>
ナビゲーションバーを使用して目的の情報を見つけてください。
</p>
```

私たちは頻繁にウェブサイトを更新するため、Google に登録されたパスが最新状態に追いつかず、ユーザーが誤ったページにアクセスしてしまうことがあります。

ここでユーザーに伝えたいのは次のことです：

- これらのページはおそらく存在しているが、私たちが場所を移動した可能性があります！

この情報をユーザーに伝えることで、ナビゲーションバーから目的の情報を再度探し出してもらえることを期待しています。

---

この部分はあなたのニーズに応じて自由に変更できます。

### 成果のデモ

跳ねるアイコンの効果を確認したい場合、私たちのウェブサイトで存在しないパスを適当に入力すると、この 404 ページを見ることができます。

画面は以下のように表示されます：

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
      お探しのページは見つかりませんでした。
    </p>
    <p style={{ fontSize: '1.2rem', marginBottom: '20px' }}>
      ウェブサイトの構造が変更された可能性があります。
    </p>
    <p style={{ fontSize: '1.2rem', marginBottom: '20px' }}>
      上部のナビゲーションバーをクリックして、必要な情報を探してみてください。
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

### 完全なコード

最後に、完全なコードを提供します：

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
            お探しのページは見つかりませんでした。
          </p>
          <p style={{ fontSize: "1.2rem", marginBottom: "20px" }}>
            ウェブサイトの構造が変更された可能性があります。
          </p>
          <p style={{ fontSize: "1.2rem", marginBottom: "20px" }}>
            上部のナビゲーションバーをクリックして、必要な情報を探してみてください。
          </p>
          <p aria-live="polite" style={{ fontSize: "1rem", color: "#555" }}>
            {countdown > 0
              ? `${countdown} 秒後に自動的にホームページへリダイレクトされます...`
              : "リダイレクト中..."}
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
