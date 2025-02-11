---
slug: update-docusaurus-to-3-6-0
title: 更新 Docusaurus 到 3.6.0
authors: Z. Yuan
image: /img/2024/1109.webp
tags: [Docusaurus, Update]
description: 排除更新過程中遇到的問題
---

Docusaurus 發布了 3.6.0 版本，這個版本主要更新了打包工具，大幅提升編譯速度。

但是我們在更新過程中又壞掉啦！

<!-- truncate -->

## 更新內容

如果你還不知道這件事情的話，可以先去看一下他們最新的部落格內容：

- [**Docusaurus 3.6**](https://docusaurus.io/blog/releases/3.6)

  <iframe
    src="https://docusaurus.io/blog/releases/3.6"
    width="80%"
    height="300px"
    center="true"
    ></iframe>

## 問題描述

更新本身沒有問題，但在這個版本中，Docusaurus 新增了一個特性，即可以在 config 中設定：

```js title="docusaurus.config.js"
const config = {
  future: {
    experimental_faster: true,
  },
};
```

當我們把這個設定加入到我們的 `docusaurus.config.js` 檔案中，就會出現以下錯誤：

```shell
yarn run v1.22.22
$ docusaurus start
[INFO] Starting the development server...
[SUCCESS] Docusaurus website is running at: http://localhost:3000/
● Client ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ (83%) sealing chunk ids
Segmentation fault (core dumped)
error Command failed with exit code 139.
info Visit https://yarnpkg.com/en/docs/cli/run for documentation about this command.
```

---

在看到這個錯誤的當下，我們其實是有點生氣的。

這裡就只給我們一個 `Segmentation fault`？是怎樣？要通靈嗎？

## 解決問題

我們在官方的 issue 中也沒有找到相關的解決方案，只好自己手動排查。

經過一番查找，我們發現這個問題在於 `_category_.json` 檔案中不能使用中文。

或者更正確的說，不能「特定」的中文字，至於具體是哪些字會導致錯誤，我們也不清楚。

例如，原本我們其中一個檔案是這樣的：

```json title="_category_.json"
{
  "label": "元富證券",
  "position": 1,
  "link": {
    "type": "generated-index"
  }
}
```

把 `元富證券` 改成 `中文` 可以正常運行！

:::tip
一樣都是中文？為什麽這個可以，那個不行？
:::

把 `中文` 改成英文，也可以正常運行了！

```json title="_category_.json"
{
  "label": "English Label",
  "position": 1,
  "link": {
    "type": "generated-index"
  }
}
```

## 除此之外

我們意外地發現另外一個錯誤，就是新設定不能支援奇怪字元的檔案名稱。

例如，我們原本有一個檔案名稱是：`Bézier`，這裡面有個重音符號，就會導致錯誤。

移除重音符號，就可以正常運行了。

## 最後

這個特性我們最後決定不採用。

畢竟我們的網站也就這麼一丁點大，編譯速度並不是我們的瓶頸，但是這個特性卻要讓我們修改很多檔案。

還是改天再來看看吧！

## 2024-11-20 更新

收到官方推送更新的消息了，這次版本更新到 v3.6.2，修復了上面文章提到的問題。

在這個版本中，我們可以順利使用`experimental_faster` 這個設定：

```js title="docusaurus.config.js"
const config = {
  future: {
    experimental_faster: true,
  },
};
```

經過測試，目前沒有再出現 `Segmentation fault` 的問題。

但是...

啟動後，如果在開發環境下修改檔案，會出現以下錯誤：

```shell
Panic occurred at runtime. Please file an issue on GitHub with the backtrace below: https://github.com/web-infra-dev/rspack/issues
Message:  Chunk(ChunkUkey(Ukey(606), PhantomData<rspack_core::chunk::Chunk>)) not found in ChunkByUkey
Location: crates/rspack_core/src/lib.rs:328

Run with COLORBT_SHOW_HIDDEN=1 environment variable to disable frame filtering.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ BACKTRACE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 1: start_thread
    at ./nptl/pthread_create.c:447
 2: clone3
    at ./misc/../sysdeps/unix/sysv/linux/x86_64/clone3.S:78
Aborted (core dumped)
error Command failed with exit code 134.
info Visit https://yarnpkg.com/en/docs/cli/run for documentation about this command.
```

看起來是 rspack 的問題，我們立刻找到了相關的 issue：

- [**web-infra-dev/rspack: [Bug]:using docusaurus edit mdx or md file, process crash. #8480**](https://github.com/web-infra-dev/rspack/issues/8480)

看來我們不孤單啊！還是再等等吧。

## 2024-11-24 更新

延續上次的問題，這次把版本更新到 v3.6.3。

rspack 的問題已經修復了，我們可以開心愉快地正常使用了！
