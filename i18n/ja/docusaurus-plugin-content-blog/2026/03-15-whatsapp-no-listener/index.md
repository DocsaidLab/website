---
slug: whatsapp-online-but-no-listener
title: OpenClaw × WhatsApp：bundler が引き起こした runtime state の分裂
authors: Z. Yuan
tags: [openclaw, whatsapp, debugging, javascript, bundling]
image: /img/2026/0315-whatsapp-no-listener.svg
description: OpenClaw の WhatsApp 経路で起きた state consistency 問題を分析する。
---

ようやく少し時間ができたので、OpenClaw を WhatsApp につないでみることにしました。

基本の流れ自体はかなり順調でした。
ユーザーが WhatsApp からメッセージを送ると、AI Agent は普通に返答できます。

ところが、OpenClaw から WhatsApp へ能動的にメッセージを push しようとすると、システムはずっとこう返してきました。

```
No active WhatsApp Web listener
```

不思議なのは、ほかのシグナルはどれも正常に見えたことです。

<!-- truncate -->

## 問題の現れ方

全体の症状は、かなり不整合に見えました。

- gateway log では WhatsApp inbound listener が起動済みになっている
- dashboard は普通に開く
- inbound message で agent reply も発火する
- それでも能動 send path に入ると、必ず次が返る

```
No active WhatsApp Web listener
```

言い換えると:

- monitor path からは listener が見えている
- send path からは listener が存在しないように見える

つまり問題は WhatsApp 接続そのものでも、gateway service そのものでもありません。listener state がシステム内部で一貫しない形で観測されていたのです。

## 初期の切り分け、でも実際は違った

最初に疑う方向として自然だったのは、次のようなものです。

- WhatsApp session の失効
- QR pairing flow の問題
- gateway service が正しく起動していない
- listener lifecycle の timing race condition

どれももっともらしい仮説ですが、次の重要なシグナルを説明できません。

> monitor には listener が存在すると明確に記録されているのに、send path はそれでも listener 不在を返していた。

これは listener state が消えたのではなく、別の module から別バージョンのものとして見えていたことを意味します。

## Root Cause：bundler による runtime state の分裂

OpenClaw の WhatsApp integration には、共有状態が一つあります。

```
active web listener registry
```

設計上は:

- monitor path が listener を登録する
- send path が listener を読む

理論上、この二つは同じ module state を共有しているはずです。

しかし bundling 後は事情が変わりました。

build 産物では:

- monitor code と send code が別々の bundle chunk に分かれた
- module-scoped store がそれぞれで初期化された

結果はこうです。

```
monitor chunk -> store A
send chunk    -> store B
```

それぞれの側だけを見ると、動きは筋が通っています。

- monitor は確かに listener を書き込んでいた
- send は確かに listener を見つけられなかった

ただし、操作していたのは完全に別々の runtime state でした。

だからこそ:

- log は正しく見える
- error message も正しく見える
- それでも全体の振る舞いは整合しない

## 最初の patch では足りなかった理由

最初の patch は Homebrew で入れた OpenClaw のコピーに当てていました。

診断の正しさを確かめるには役立ちましたが、保守可能な解法ではありません。

理由は単純です。

- 触っているのはインストール済み成果物
- package update で上書きされる
- 次に壊れたらまた patch を当て直すことになる

そのため、最終的な修復は次へ戻す必要がありました。

```
~/openclaw
```

source tree で修正して再 build し、runtime と source と test を同じ保守面に揃える必要がありました。

## 最終的な修復方針

目標は単純です。

> monitor path と send path が、常に同じ listener store を共有すること。

そのために module-scoped state を global runtime store へ移しました。

```javascript
const STORE_KEY = Symbol.for("openclaw.whatsapp.active-web-listener-store");
```

そして listener registry を次にぶら下げます。

```
globalThis[STORE_KEY]
```

この方法の利点は:

- 別 bundle chunk でも同じ Symbol key に到達できる
- module reload で state が再初期化されない
- 同じ JavaScript runtime の中にいる限り、必ず同じ store を共有できる

言い換えると:

```
module state   -> unreliable
global runtime -> stable
```

## Regression Test

runtime state の bug は、修正だけして test がないと、後の build 変更で簡単に戻ってきます。

そこで今回は次の境界に対する regression coverage も追加しました。

- module reload
- lazy-load boundary
- bundle chunk boundary

test が保証しているのは次の点です。

- listener registry が常に同じ store を指すこと
- send path が新しい registry を取り直さないこと

## 検証時の落とし穴

この種の問題を検証するときは、誤判定しやすいポイントがあります。

`openclaw message send` は最適な smoke test ではありません。

理由は:

- CLI command が独自の process を起動する
- send path が lazy-loaded である

つまり証明できるのは:

```
CLI process は listener を見つけられる
```

ということだけで、必ずしも:

```
常駐 gateway service が listener 共有を回復した
```

ことまでは言えません。

より正確な検証方法は、次を直接叩くことです。

```
gateway send RPC
```

今回の最終 smoke test でも、この経路を使いました。

## 工学的に広く使える教訓

システムが同時に次のような信号を出しているとき:

- monitor は見えている
- service は生きている
- 能動操作だけ失敗する
- 共有オブジェクトだけ見つからない

疑うべきなのは transport layer ではなく、runtime state ownership であることが多いです。

特に先に確認したい境界は次です。

- process boundary
- lazy-loading boundary
- bundle chunk boundary
- global state boundary

表面上は network failure や session failure に見える問題でも、実際には state が別 runtime context で複製されたり再初期化されたりしていることがよくあります。

runtime state と module boundary を早めに疑うようにすると、診断速度はかなり上がります。
