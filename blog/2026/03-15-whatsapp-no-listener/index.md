---
slug: whatsapp-online-but-no-listener
title: "OpenClaw × WhatsApp：一次由 bundler 引發的 runtime state 分裂"
authors: Z. Yuan
tags: [openclaw, whatsapp, debugging, javascript, bundling]
image: /img/2026/0315-whatsapp-no-listener.svg
description: 分析 OpenClaw WhatsApp 路徑中的狀態一致性問題。
---

難得有空，我決定嘗試把 OpenClaw 串接到 WhatsApp。

基本流程其實很順利：
當使用者從 WhatsApp 傳訊息進來時，AI Agent 可以正常回覆。

但當我嘗試 **從 OpenClaw 主動推送訊息到 WhatsApp** 時，系統卻始終回覆：

```
No active WhatsApp Web listener
```

奇怪的是，系統的其他訊號卻顯示一切正常。

<!-- truncate -->

## 問題現象

整體症狀呈現出一種不一致的狀態：

- gateway log 顯示 WhatsApp inbound listener 已啟動
- dashboard 可以正常打開
- inbound message 可以觸發 agent 回覆
- 但只要走 **主動 send path**，就會得到：

```
No active WhatsApp Web listener
```

換句話說：

- **monitor path** 可以觀察到 listener
- **send path** 卻認為 listener 不存在

這表示問題不在 WhatsApp 連線，也不在 gateway service 本身，而是 **listener 狀態在系統內部出現了不一致的觀察結果**。

---

## 初步排查（但其實都不是）

一開始最合理的懷疑方向其實是這些：

- WhatsApp session 失效
- QR pairing 流程問題
- gateway service 沒有正確啟動
- listener lifecycle timing race condition

這些方向都很合理，但它們都無法解釋一個關鍵訊號：

> monitor 明確記錄 listener 已存在，但 send path 卻仍然回報不存在。

這意味著 **listener 狀態不是消失，而是「被不同模組看成不同版本」**。

---

## Root Cause：bundler 造成 runtime state 分裂

OpenClaw 的 WhatsApp integration 內部有一份共享狀態：

```
active web listener registry
```

設計上：

- monitor path 會 **註冊 listener**
- send path 會 **讀取 listener**

兩者理論上應該共享同一份 module state。

但在 bundling 之後，事情變得不同。

在 build 產物中：

- monitor code 與 send code 落在 **不同 bundle chunk**
- module-scoped store 因此被 **各自初始化**

結果就是：

```
monitor chunk -> store A
send chunk    -> store B
```

兩邊 individually 都是合理的：

- monitor 確實寫入 listener
- send 確實找不到 listener

但它們操作的是 **兩份完全不同的 runtime state**。

這也是為什麼：

- log 看起來正確
- error message 也正確
- 但整體行為卻完全不一致

---

## 為什麼第一版 patch 不夠

最初的修補其實是在 **Homebrew 安裝的 OpenClaw 副本**裡完成的。

這對於驗證 diagnosis 有幫助，但不是一個可維護的解法。

原因很簡單：

- 你修改的是 **安裝產物**
- 套件更新就會被覆蓋
- 下次壞掉又要重新 patch

因此最終修復必須回到：

```
~/openclaw
```

在 **source tree 層級完成修改並重新 build**，讓 runtime、source 與測試維持同一個維護面。

---

## 最終修復策略

修復的目標很單純：

> 確保 monitor 與 send 永遠共享同一份 listener store。

做法是把 module-scoped state 移到 **global runtime store**：

```javascript
const STORE_KEY = Symbol.for("openclaw.whatsapp.active-web-listener-store");
```

然後將 listener registry 掛在：

```
globalThis[STORE_KEY]
```

這樣做的好處是：

- 不同 bundle chunk 仍會命中同一個 Symbol key
- module reload 不會重新初始化 state
- 只要還在同一個 JavaScript runtime 內，就一定共享同一份 store

換句話說：

```
module state   -> unreliable
global runtime -> stable
```

---

## Regression Test

修復 runtime state 的 bug，如果沒有測試，很容易在未來的 build 調整中重新出現。

因此這次同步加入 regression test，覆蓋以下情境：

- module reload
- lazy load boundary
- bundle chunk boundary

測試確保：

- listener registry 始終指向同一份 store
- send path 不會重新取得新的 registry

---

## 驗證時的一個陷阱

驗證這類問題時，很容易出現一個誤判來源。

`openclaw message send` 其實不是最佳 smoke test。

原因是：

- CLI command 會啟動 **自己的 process**
- send path 是 **lazy-loaded**

因此它只能證明：

```
CLI process 可以找到 listener
```

但不一定代表：

```
常駐 gateway service 的 listener 已恢復
```

比較準確的驗證方式是直接呼叫：

```
gateway send RPC
```

這次的最終 smoke test 也是使用這條路徑。

---

## 工程上的通用教訓

當系統同時出現以下訊號時：

- monitor 可見
- service 存活
- 主動操作失敗
- 共享物件缺失

問題往往不在 transport layer，而是在 **runtime state ownership**。

特別需要檢查以下邊界：

- process boundary
- lazy-loading boundary
- bundle chunk boundary
- global state boundary

這類問題在表面上很像 network 或 session failure，但實際上通常是 **state 在不同 runtime context 中被複製或重新初始化**。

只要排查順序能優先落在 runtime state 與 module boundary，定位速度通常會快非常多。
