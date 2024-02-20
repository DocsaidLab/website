---
slug: error-record
title: 日常錯誤排除紀錄
authors: Zephyr
tags: [error, record]
---

我們總是會遇到一堆問題。有些問題是我們自己造成的，有些問題是別人造成的，有些問題是我們無法控制的。

這裡紀錄一些簡單問題和解決方法。

<!--truncate-->

1. 執行 `npx docusaurus start` 時出現以下錯誤：

    ```bash
    file:///home/shayne/workspace/blog/node_modules/@docusaurus/core/bin/docusaurus.mjs:30
    process.env.BABEL_ENV ??= 'development';
                        ^^^

    SyntaxError: Unexpected token '??='
    ```

    解決方法：`??=` 操作符需要 Node.js 15.0.0或更高版本才能支持。

    ```bash
    nvm install node
    nvm use node
    ```
