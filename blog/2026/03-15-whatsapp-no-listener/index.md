---
slug: whatsapp-online-but-no-listener
title: 從 `No active WhatsApp Web listener` 到 Runtime State Split 的定位與修復
authors: Z. Yuan
tags: [openclaw, whatsapp, debugging, javascript, bundling]
image: /img/2026/0315-whatsapp-no-listener.svg
description: 分析 OpenClaw WhatsApp 路徑中 monitor 已成功註冊 listener，但 send 仍回報 `No active WhatsApp Web listener` 的狀態一致性問題，根因是 bundling 後產生分裂的 module-scoped runtime state。
---

## Summary

本文分析一個發生在 OpenClaw WhatsApp 路徑中的狀態一致性問題：monitor 端已成功掛載 listener，但主動發送路徑仍回報 `No active WhatsApp Web listener`。根因不是 session 失效，也不是 gateway 未啟動，而是 bundling 後產生了分裂的 module-scoped runtime state，導致 monitor 與 send 讀寫的並非同一份 listener registry。

最終修復包含兩個部分：將共享狀態從模組層移到 `globalThis`，並以 `Symbol.for(...)` 提供穩定 key；同時補上跨模組重載情境的 regression test，避免後續 bundling 或 lazy-load 調整再次導致狀態分裂。

- monitor 端的監聽流程已經成功註冊 listener
- 主動發送路徑仍回報 listener 不存在
- 問題核心不在 transport layer，而在 runtime state 的共享邊界
- 最終維護點需要回到 repo source tree，而不是停留在 Homebrew 安裝產物

<!-- truncate -->

## Observed Symptoms

實際症狀具有明顯的不一致性：

- gateway log 會印出正在監聽 WhatsApp inbound
- dashboard 也能正常打開
- 但一走主動發送，系統就回 `No active WhatsApp Web listener`

這組訊號表示 monitor path 與 send path 對 listener 狀態的觀察並不一致。從表面症狀看，系統像是局部可用，但在真正需要共享 listener 的主動發送操作上失敗。

## Initial Misleading Hypotheses

初步排查時，最容易被優先懷疑的方向包括：

- WhatsApp session 狀態失效
- QR pairing 流程異常
- gateway service 未正確啟動
- listener 初始化與 send path 之間存在 lifecycle timing 問題

這些方向都合理，但不足以解釋「monitor 已成功記錄 listener 存在，而 send 仍回報 listener 缺失」這組訊號。

## Root Cause

OpenClaw 的 WhatsApp 路徑內存在一份「目前活躍中的 web listener」共享狀態。理論上，monitor path 會註冊 listener，而 send path 會讀取同一份 registry 以完成主動發送。

實際問題發生在 bundling 之後。monitor 與 send 雖然都引用 `active-listener`，但 build 產物中它們位於不同 chunk，最終並未共享同一份 module-scoped runtime store。

這代表問題本質不是 registry 被覆寫，而是產生了兩份彼此隔離的狀態：

- monitor chunk 寫入的是 store A
- send chunk 讀取的是 store B
- 兩端 individually 都能輸出合理訊號
- 組合後則呈現出 listener state 不一致問題

這也是為什麼 log 與錯誤訊息各自成立，但無法共同描述真實執行狀態。

## Why the Existing Patch Was Not Sufficient

前一輪修補是在 Homebrew 安裝的系統副本內完成。這對驗證 diagnosis 有幫助，但不適合作為最終維護點，原因很直接：

- 你改的是安裝產物
- 套件一更新，修補就可能被覆蓋
- 下次再壞，還要重新追一次

因此，最終修復需要回到 `~/openclaw` 的 source tree，自行 build 並讓 service 直接執行 repo 版本，才能讓 source、測試與 runtime 行為維持同一個維護面。

## Final Remediation

修復策略不追求抽象優雅，而是優先保證 runtime 一致性。做法是將 active listener registry 從模組內單例改成 `globalThis` 上的共享 store，並使用 `Symbol.for(...)` 確保不同 chunk 會命中同一個 key。

核心識別方式如下：

```ts
const STORE_KEY = Symbol.for("openclaw.whatsapp.active-web-listener-store");
```

這個調整的目的很單純：只要 monitor 與 send 還在同一個 JavaScript runtime 內，就必須讀寫同一份 listener store，而不是各自持有 chunk-local state。

## Regression Test Strategy

修復之後，單靠人工驗證並不足以覆蓋後續 bundling 變化。因此這次同步補上 regression test，確認狀態共享不會在模組重載或 lazy-load 邊界重新失效。

測試關注點包括：

- 模組重新載入之後
- 共享 listener 仍然指向同一份 store
- send path 不會因為 module boundary 重新取得另一份 registry

## Verification Caveats

驗證策略本身也存在一個容易誤判的點：`openclaw message send` 並不是確認主系統 WhatsApp 推送已恢復的最佳證據。

原因在於它走的是 CLI process 自身的 lazy-loaded send path。這能證明 CLI process 可用，但未必能證明常駐中的 gateway service 已恢復共享 listener。

若驗證目標是「repo-based main system 是否已修復」，較準確的方式是直接打 gateway 的 `send` RPC。這次最終 smoke test 也是沿用這條路徑。

## Generalized Engineering Lessons

這次事件的核心教訓是：當系統同時出現「monitor 可見、service 存活、主動操作失敗、共享物件缺失」這組症狀時，應優先檢查 runtime state 是否跨越了 process、bundle 或 lazy-load 邊界而失去一致性。

- process boundary
- lazy loading boundary
- bundle chunk boundary
- global state boundary

這類問題在表面上容易被誤判為 transport failure，但真正失效的往往是狀態共享模型本身。只要排查順序能優先落在 state ownership 與 runtime boundary，定位速度通常會明顯改善。
