---
slug: update-docusaurus-to-3-6-0
title: Docusaurusを3.6.0にアップデート
authors: Z. Yuan
image: /ja/img/2024/1109.webp
tags: [Docusaurus, Update]
description: アップデート過程での問題を解決
---

Docusaurus がバージョン 3.6.0 をリリースしました。このバージョンではビルドツールが主に更新され、コンパイル速度が大幅に向上しました。

しかし、アップデートの途中でまたもやトラブルが発生しました！

<!-- truncate -->

## アップデート内容

まだこの話を知らない場合は、まず彼らの最新ブログ記事を確認してください：

- [**Docusaurus 3.6**](https://docusaurus.io/blog/releases/3.6)

  <iframe
    src="https://docusaurus.io/blog/releases/3.6"
    width="80%"
    height="300px"
    center="true"
    ></iframe>

## 問題の説明

アップデート自体は問題ありませんでしたが、このバージョンでは新しい機能が追加されました。それは、設定ファイルで以下のように設定できるというものです：

```js title="docusaurus.config.js"
const config = {
  future: {
    experimental_faster: true,
  },
};
```

この設定を`docusaurus.config.js`に追加した途端、以下のエラーが発生しました：

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

このエラーを見た瞬間、少し怒りが込み上げました。

「Segmentation fault」だけですか？これではまるで超能力者に解読を求めているようなものです！

## 問題解決

公式の issue を調べても解決策は見つかりませんでしたので、自分たちで手動で調査を行いました。

調査の結果、この問題は`_category_.json`ファイルに特定の日本語文字を使用できないことが原因であると判明しました。

具体的には、以下のような例があります：

```json title="_category_.json"
{
  "label": "元富証券",
  "position": 1,
  "link": {
    "type": "generated-index"
  }
}
```

ここで「元富証券」を「中文」に変更すると、正常に動作しました！

:::tip
同じ日本語でも、なぜこれが動作してあれが動作しないのか？
:::

「中文」を英語に変更しても、正常に動作しました：

```json title="_category_.json"
{
  "label": "English Label",
  "position": 1,
  "link": {
    "type": "generated-index"
  }
}
```

## その他の問題

さらに、別のエラーも偶然発見しました。それは、新しい設定では特殊文字を含むファイル名がサポートされないことです。

例えば、ファイル名に「Bézier」のようなアクセント記号が含まれると、エラーが発生します。

アクセント記号を削除すれば、問題なく動作します。

## 結論

この機能を最終的に採用しないことにしました。

なぜなら、私たちのサイトは小規模であり、コンパイル速度は特にボトルネックではありません。一方で、この機能を使うためには多数のファイルを修正する必要があります。

またの機会に試すことにします！

## 2024-11-20 の更新

公式から更新が通知されました。今回のバージョンアップで 3.6.2 となり、上記の問題が修正されました。

このバージョンでは、`experimental_faster`設定を問題なく使用できます：

```js title="docusaurus.config.js"
const config = {
  future: {
    experimental_faster: true,
  },
};
```

テストの結果、`Segmentation fault`の問題は発生しなくなりました。

しかし...

開発環境でファイルを編集すると、以下のエラーが発生します：

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

どうやら、rspack の問題のようです。すぐに関連する issue を見つけました：

- [**web-infra-dev/rspack: [Bug]:using docusaurus edit mdx or md file, process crash. #8480**](https://github.com/web-infra-dev/rspack/issues/8480)

仲間がいるようですね！引き続き様子を見ることにします。

## 2024-11-24 の更新

前回の問題に続き、今回はバージョンを 3.6.3 に更新しました。

rspack の問題が修正され、ようやく快適に使用できるようになりました！
