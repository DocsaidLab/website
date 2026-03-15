---
slug: whatsapp-online-but-no-listener
title: WhatsApp はオンラインに見えるのに、なぜ OpenClaw は listener がないと言うのか
authors: Z. Yuan
tags: [openclaw, whatsapp, debugging, javascript, bundling]
image: /img/2026/0315-whatsapp-no-listener.svg
description: Gateway は WhatsApp を監視中だと言っているのに、送信すると No active WhatsApp Web listener が返る。接続が死んでいたのではなく、共有状態が bundle chunk の境界で分裂していました。
---

オンラインには見えていました。

log も監視中だと言っていました。

送信すると、

こう返ってきます。

```text
No active WhatsApp Web listener
```

この手のバグが腹立たしいのは、どちらも真実を話しているように見えるからです。

monitor 側は listener を登録したと言う。

send 側はここには listener がないと言う。

どちらも嘘ではありません。

ただ、別の宇宙に住んでいました。

> **問題は WhatsApp の接続断ではなく、同じ runtime state が bundler によって二重化されていたことでした。**

今回はその症状、修正、そして最終的に `~/openclaw` 側へ戻した理由をまとめておきます。

<!-- truncate -->

## まず症状：生きているように見えるのに送れない

見えていた現象はかなり矛盾していました。

- gateway log には WhatsApp inbound を監視中と出る
- dashboard も普通に開く
- それでも送信すると `No active WhatsApp Web listener` が返る

表面だけ見ると、疑う先はいくつもあります。

- WhatsApp session が切れたのかもしれない
- QR pairing が壊れたのかもしれない
- gateway service が半分だけ死んでいるのかもしれない
- 再起動のタイミングが悪かったのかもしれない

どれも自然な推測です。

ただし今回は全部違いました。

## 本当の原因：listener が二つに割れた

OpenClaw の WhatsApp 経路には、「今生きている web listener」を持つ共有状態があります。

本来は、

- monitor 側が listener を登録し
- send 側がそれを読んで送信に使う

という流れです。

問題は bundle 後に起きました。

monitor と send が別の chunk に入りました。

どちらも `active-listener` を import しているのに、build 後は同じ module state を共有していませんでした。

結果はこうです。

- chunk A には本物の listener がいる
- chunk B の registry は空のまま
- log はその世界では正しい
- send のエラーもその世界では正しい

普通の「どこかで上書きされた」系のバグではありません。

同じオフィスに白板を二枚置いて、それぞれが真面目に更新しているのに、なぜか話が噛み合わない感じです。

## 以前 Homebrew 側で直しただけでは足りなかった理由

最初の応急処置は Homebrew で入れたシステム側に当てました。

診断を確かめるには役立ちました。

ただ、そこに留まるのは危ない。

理由は単純です。

- 直しているのはインストール済みの成果物
- 更新が入れば修正が消えるかもしれない
- 次に再発したとき、また同じ追跡をやり直す

なので本当にやるべきことは、「とりあえず送れるようにする」だけではありませんでした。

主系統を `~/openclaw` の repo に戻し、source から build し、そのコードで service を動かすことです。

そうすれば：

- 修正がパッケージ更新で消えない
- regression test を source と一緒に残せる
- gateway が本当に今見ているコードを動かす

## 修正は素直です：共有状態を `globalThis` に出す

問題が「chunk 境界では module state が共有されない」なら、bundler の気分に期待しない方が早いです。

active listener registry を `globalThis` に載せ、`Symbol.for(...)` で安定した key を使う形に変えました。

考え方は単純です。

- monitor がどの chunk から来ても
- send がどの chunk から来ても
- 同じ JavaScript runtime にいる限り
- 同じ store に着地する

核になる方向はこうです。

```ts
const STORE_KEY = Symbol.for("openclaw.whatsapp.active-web-listener-store");
```

もともと module 内に閉じていた singleton を、`globalThis` 上の共有 store に置き換えたわけです。

派手ではありません。

でも効きます。

## コードだけでなく、ちゃんと再発防止のテストも置く

この種のバグは厄介です。

今は直って見えても、次の bundling 変更や lazy load の調整で静かに戻ってきます。

だから今回は `active-listener.ts` の修正だけでは終わらせませんでした。

regression test も追加しました。

見るポイントは単なる set/get ではなく、

- module reload 境界をまたいでも store が残るか
- 共有 listener が同じ backing state を見続けるか
- send 側が後から別宇宙を増やさないか

要するに、今回落ちた穴にベルを付けました。

## 検証にも罠がある：`message send` を主系統の証拠にしない

もう一つ、見落としやすい点がありました。

`openclaw message send` は、主系統の WhatsApp push が直った証明としては最適ではありません。

使えないからではありません。

CLI process 側の lazy-loaded send path を通るからです。

つまり、うっかり次のことだけを証明してしまいます。

- CLI process 単体では送れる
- でも常駐している gateway service はまだ壊れている

本当に確認したいのが「主系統 service が直ったか」なら、見るべきは gateway の `send` RPC です。

最終 smoke test はその経路で行い、repo ベースの主系統が本当に回復したことを確認しました。

## この教訓は、実は WhatsApp に限らない

見た目は transport の障害に見えました。

でも違いました。

runtime 境界の問題です。

- process boundary
- lazy-loading boundary
- bundle chunk boundary
- global-state boundary

この手のバグには特徴があります。

- log は別に嘘ではない
- ただし各 log は自分の世界しか見ていない
- 並べて初めて、状態が共有されていなかったと分かる

なので今後、こんな症状を見たら：

- 「オンラインに見える」
- 「monitor は動いている」
- 「service も生きている」
- 「でも能動操作だけ共有物がないと言う」

最初から transport を疑わない方がいいです。

まず、本当にその境界を越えて状態が共有されているかを見る。

たいてい、幽霊はネットワークの中ではありません。

自分の runtime の中にいます。
