---
slug: whatsapp-online-but-no-listener
title: Active Monitor 存在下で OpenClaw が `No active WhatsApp Web listener` を返した理由
authors: Z. Yuan
tags: [openclaw, whatsapp, debugging, javascript, bundling]
image: /img/2026/0315-whatsapp-no-listener.svg
description: monitor が listener を正常に登録しているにもかかわらず send が `No active WhatsApp Web listener` を返した OpenClaw WhatsApp 経路の状態整合性問題を分析する。根因は bundling 後に発生した module-scoped runtime state の分裂でした。
---

## Summary

本稿では、OpenClaw の WhatsApp 経路で発生した状態整合性問題を扱います。monitor 側は listener の登録に成功しているのに、主動送信経路では `No active WhatsApp Web listener` が返るという現象です。根因は session 失効でも gateway 未起動でもなく、bundling 後に module-scoped runtime state が分裂し、monitor と send が同一の listener registry を共有していなかったことにあります。

最終的な remediation は二段構えでした。共有 registry を `globalThis` に移し、`Symbol.for(...)` で安定した key を与えること。さらに、module reload 境界で同じ失敗が再発しないよう regression coverage を追加することです。

- monitor 経路では listener 登録が完了していた
- send 経路では listener 不在が報告された
- 問題の中心は transport layer ではなく runtime state の共有境界にあった
- 恒久対応には repo source tree への回帰が必要だった

<!-- truncate -->

## Observed Symptoms

観測された症状は内部的に整合していませんでした。

- gateway log には WhatsApp inbound を監視中と出る
- dashboard も普通に開く
- それでも送信すると `No active WhatsApp Web listener` が返る

これらの信号から分かるのは、monitor path と send path が同じ listener state を見ていないということです。表面的には部分的に正常に見えますが、共有 listener が必要な主動送信で失敗しています。

## Initial Misleading Hypotheses

初期切り分けで優先候補になりやすいのは次の方向です。

- WhatsApp session の失効
- QR pairing の異常
- gateway service の起動不全
- listener 初期化と send path の間にある lifecycle timing 問題

これらは妥当な仮説ですが、「monitor では listener 存在が確認できるのに send では欠落している」という症状全体は説明しきれません。

## Root Cause

OpenClaw の WhatsApp 経路には、現在有効な web listener を保持する共有状態があります。通常は monitor path が listener を登録し、send path が同じ registry を参照して主動送信を行います。

実際の失敗点は bundling 後にありました。monitor と send はどちらも `active-listener` を import していましたが、build 産物では別 chunk に入り、同じ module-scoped runtime store を共有しなくなっていました。

つまり本質は registry の上書きではありません。状態が二つに分裂していました。

- chunk A には本物の listener がいる
- chunk B の registry は空のまま
- それぞれの側は局所的には正しい信号を出す
- 合成すると state consistency failure として現れる

このため、log と send error は individually には成立していても、全体として一つの整合した runtime state を表していませんでした。

## Why the Existing Patch Was Not Sufficient

最初の応急処置は Homebrew で入れたシステム側に当てました。

診断の妥当性を確認するには十分でしたが、恒久的な保守点としては不十分です。

- 直しているのはインストール済みの成果物
- 更新が入れば修正が消えるかもしれない
- 次に再発したとき、また同じ追跡をやり直す

そのため、最終的な修復は `~/openclaw` の source tree に戻し、source・test・runtime を同じ保守面に揃える必要がありました。

## Final Remediation

修復方針は抽象性より runtime 一貫性を優先しました。active listener registry を module-local singleton から `globalThis` 上の共有 store に移し、`Symbol.for(...)` で安定した key を与えています。

識別子の中心は次の形です。

```ts
const STORE_KEY = Symbol.for("openclaw.whatsapp.active-web-listener-store");
```

意図は明確です。monitor と send が同じ JavaScript runtime に存在する限り、参照先は常に同じ listener store でなければなりません。

## Regression Test Strategy

コード修正だけでは bundling 変更に対して脆弱です。そのため、今回の failure mode を直接再現・監視する regression coverage も追加しました。

確認しているのは単純な set/get ではありません。

- module reload 境界をまたいでも store が残るか
- 共有 listener が同じ backing state を見続けるか
- send path が別の registry を取得しないか

## Verification Caveats

検証手順にも注意点があります。`openclaw message send` は主系統の WhatsApp push が回復したことを証明する最適な方法ではありません。

理由は、そのコマンドが CLI process 自身の lazy-loaded send path を通るためです。

つまり確認できるのは「CLI process は送れる」ことであり、「常駐 gateway service が listener を共有できている」ことではありません。

repo-based main system の修復を検証するには、gateway の `send` RPC を直接叩く方が適切です。最終 smoke test もこの経路で実施しました。

## Generalized Engineering Lessons

この事例の教訓は再利用可能です。システムが同時に「monitor は見えている」「service は生きている」「能動操作だけ失敗する」「共有オブジェクトだけ欠落する」という症状を示した場合、最初に確認すべきなのは runtime state が process・bundle・lazy-load 境界をまたいで一貫しているかどうかです。

表面的には transport failure に見えやすい問題でも、実際に先に壊れているのは state-sharing model であることがあります。state ownership と runtime boundary を優先して点検する方が、診断は速く安定します。
