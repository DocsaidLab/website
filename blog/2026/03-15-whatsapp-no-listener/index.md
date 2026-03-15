---
slug: whatsapp-online-but-no-listener
title: WhatsApp 明明在線，為什麼 OpenClaw 一發送就說沒有 listener？
authors: Z. Yuan
tags: [openclaw, whatsapp, debugging, javascript, bundling]
image: /img/2026/0315-whatsapp-no-listener.svg
description: Gateway 明明顯示 WhatsApp 已在監聽，主動發送卻回 No active WhatsApp Web listener。問題不是連線斷了，而是 bundle chunk 把共享狀態拆成兩份。
---

它看起來在線。

log 也說它在聽。

你按下發送。

它回你一句：

```text
No active WhatsApp Web listener
```

這種錯法很適合讓人懷疑人生，因為每一邊都像在講真話。

monitor 端說自己已經掛上 listener。

send 端說自己這裡沒有 listener。

兩邊都沒說謊。

只是它們活在不同宇宙。

> **問題不是 WhatsApp 沒連上，而是同一份 runtime state 被 bundler 切成了兩份。**

這次把問題、修法、還有為什麼最後要搬回 `~/openclaw` 一次寫清楚。

<!-- truncate -->

## 先看症狀：看起來通了，實際上送不出去

當時看到的現象很矛盾：

- gateway log 會印出正在監聽 WhatsApp inbound
- dashboard 也能正常打開
- 但一走主動發送，系統就回 `No active WhatsApp Web listener`

如果只看表面，你很容易往錯方向走：

- 懷疑 WhatsApp session 斷了
- 懷疑 QR pairing 壞了
- 懷疑 gateway service 沒起來
- 懷疑是某一段重啟 timing 不對

這些方向都合理。

只是這次都不是主因。

## 真正的根因：listener 被拆成兩份，各自以為自己是唯一真相

OpenClaw 的 WhatsApp 路徑裡，有一個「目前活著的 web listener」共享狀態。

理論上：

- monitor 路徑把 listener 註冊進去
- send 路徑把它讀出來，拿來做主動發送

問題出在 bundle 之後。

monitor 和 send 被打進不同 chunk。

兩邊都 import 了 `active-listener`，但在輸出結果裡，它們拿到的是兩份彼此不共享的模組狀態。

結果就變成：

- A chunk 裡真的有 listener
- B chunk 裡的 registry 還是空的
- log 與 send 錯誤訊息都各自合理
- 合在一起就很欠揍

這不是「誰蓋掉誰」那種普通 bug。

比較像一間辦公室複印了兩份白板，每個人都認真在自己的白板上更新進度，然後彼此很困惑為什麼對方永遠看不到。

## 為什麼先前在 Homebrew 那套修了，還是不夠安心

前一輪的止血是在 Homebrew 安裝的系統副本裡處理。

它可以暫時驗證方向是對的。

但那個位置有一個很現實的問題：

- 你改的是安裝產物
- 套件一更新，修補就可能被覆蓋
- 下次再壞，還要重新追一次

所以這次真正該做的，不只是「讓它能送」。

而是把主系統搬回 `~/openclaw` 這份 repo，自行 build、自行跑 service，讓修法落在真正可維護的來源碼上。

這樣之後：

- 修補不會被 Homebrew 更新順手抹掉
- regression test 可以一起留在 repo
- gateway 真的跑的是你眼前這份程式

## 修法很直接：把共享狀態從模組層搬到 `globalThis`

既然問題是 chunk 之間沒有共用模組狀態，修法就不要再賭 bundler 會替你共享。

這次的做法是把 active listener registry 掛到 `globalThis`，再用 `Symbol.for(...)` 拿一個穩定 key。

意思很簡單：

- 不管 monitor 從哪個 chunk 進來
- 不管 send 從哪個 chunk 進來
- 只要還在同一個 JavaScript runtime
- 它們就會碰到同一份 store

核心方向像這樣：

```ts
const STORE_KEY = Symbol.for("openclaw.whatsapp.active-web-listener-store");
```

然後把原本模組內的單例，改成從 `globalThis` 上初始化或讀取。

這種修法沒有很浪漫。

但它有效，而且有效得很老實。

## 只修程式不夠，還要補一個會咬人的測試

這類問題最麻煩的地方，在於它很容易「現在好了」，但下次 bundling 或 lazy load 稍微一調整，又悄悄回來。

所以這次不只修 `active-listener.ts`，也補了 regression test。

測試重點不是單純驗證 set/get。

而是驗證：

- 模組重新載入之後
- 共享 listener 仍然指向同一份 store
- send 路徑不會因為 module boundary 重新長出一個平行宇宙

簡單講，就是把這次真正踩到的坑，釘成一顆之後會叫的地雷。

## 驗證也有陷阱：不要拿 `message send` 當主系統證明

這次還順手釐清了一個很容易誤判的點。

`openclaw message send` 不是驗證主系統 WhatsApp 主動推送是否恢復的最好方法。

原因不是它不能送。

而是它走的是 CLI 自己那個 process 的 lazy-loaded send path。

你測到的，有可能只是：

- 這次 CLI process 自己可以送
- 但真正常駐的 gateway service 還沒恢復共享 listener

如果你要驗證的是「主系統 service 已經修好」，比較準的做法是直接打 gateway 的 `send` RPC。

這次就是用那條路重新做 smoke test，確認主系統的送訊息路徑已經真的通了。

## 這次整理出來的結論，其實不只適用於 WhatsApp

這次 bug 很像傳輸層壞掉。

其實不是。

它更接近一種 runtime 邊界問題：

- process boundary
- lazy loading boundary
- bundle chunk boundary
- global state boundary

這類 bug 的特點是：

- log 常常不是假的
- 只是每條 log 只代表它自己看到的世界
- 你把它們拼起來，才發現世界其實裂成兩塊

所以如果你以後再看到某個系統表現出這種症狀：

- 「明明在線」
- 「明明有 monitor」
- 「明明 service 活著」
- 「但某條主動操作就是說找不到共享物件」

先不要急著怪 transport。

先看看是不是狀態根本沒有跨邊界共享。

很多時候，鬼不在網路上。

鬼在你自己的 runtime 裡。
