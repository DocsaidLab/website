---
slug: sqlite-lease-heartbeat-recovery
title: SQLite 實戰（三）：救救你的 worker
authors: Z. Yuan
tags: [sqlite, job-queue, reliability, heartbeat]
image: /img/2025/1008.jpg
description: 讓卡住的任務自動回收，別讓佇列變成墓園。
---

你一定遇過這種狀況：

- 佇列看起來卡住了
- UI 顯示一堆 job 還在 RUNNING

但那台 worker 早就被你重開三次了，job 沒死，worker 死了。

你在本來會很愉快的週末，唱起了悼念 worker 的輓歌。

<!-- truncate -->

## 為什麼 worker 會死？

因為你在一個不可控的世界。

在理想世界，你的 worker 會宣告一個 job，然後跑完它，最後把狀態改成 `SUCCEEDED` 或 `FAILED`。

世界和平。

但在真實世界，worker 常見死法大概是這幾類：

1. **程式崩潰**：例外沒抓住、第三方 library segmentation fault、模型推論故障。
2. **被系統殺掉**：OOM killer、容器重啟、節點被回收、資源配額被打爆。
3. **卡死但沒死**：外部 API 無限等待、I/O hang、死鎖、某個 while True 正在執行。
4. **部署/升級中斷**：rolling update、手動 kill、進程被替換。
5. **網路與依賴壞掉**：S3、MQ、內部服務超時。

你會發現：**很多時候 worker 不會「優雅地失敗」**。

它會突然消失，或者永遠卡住，最終留下的只有一筆孤單的 `RUNNING`。

而「孤單的 `RUNNING`」就是 queue 系統的腐敗源頭。

## 這件事情為什麼重要？

如果你的 queue 系統無法回答以下三個問題，那它就不具備「可維運性」：

1. **這個 job 現在是誰拿走的？**
2. **他最後一次報平安是什麼時候？**
3. **如果他永遠不回來，我要怎麼收拾？**

你會注意到，這三個問題都不是「效能」問題，而是「可信度」問題。

因為我們人力有限，實務上：

- 你不可能每次卡住都去查 log、查機器、查 pod
- 你不可能每次都手動把 job 改回 `QUEUED`
- 你更不可能接受「某些 job 就這樣永遠 RUNNING，反正偶爾清一下」

所以我們需要一套很務實的架構：**lease（租約）+ heartbeat（心跳）**。

目的就是把「永久擁有」改成「暫時借走」。

## lease + heartbeat

這套理念很簡單：

- worker claim job 不是「永久擁有」，而是「暫時借走一段時間」。
- 借走之後要定期報平安、續租。
- 如果不續租，就視為失聯，系統可以回收

換句話說，你把「可用性」這件事，從「相信 worker」改成「相信資料」。

而資料庫，至少比某台隨時會死掉的 worker 值得信任。

## 最小模型

這裡通常不需要很華麗的 schema，只需要能判斷：

- 這筆 job 是否仍被某個 worker 持有？
- 這個持有是否仍有效？
- 過期後要怎麼處理？

常見最小欄位如下：

- `claimed_at`：被拿走的時間
- `heartbeat_at`：最後心跳
- `retry_count`：已重試次數

但在實務上，我們會強烈建議你多補兩個，因為這會讓系統更穩定：

- `lease_expires_at`：租約到期時間（更直觀、查詢更乾淨）
- `lease_token`：本次 claim 的憑證（防殭屍 worker）

:::info
**為什麼要 `lease_token`？**

我們要防的是這個場景：

- worker A claim job → 跑一半當機
- sweeper 回收 → worker B 重新 claim
- worker A 復活後還在更新 heartbeat / finish
- B 被 A 搞死，系統開始出現「一筆 job 兩個人寫」的靈異現象

`lease_token` 的概念就是：**每次 claim 都換一把鑰匙**。
後續更新必須帶著這把鑰匙才算數。
:::

## 範例 schema

時間建議用 **UTC epoch seconds**（`INTEGER`），因爲字串時間一定會在未來的某一刻坑到你。

```sql title="jobs schema（示意）"
CREATE TABLE IF NOT EXISTS jobs (
  id               INTEGER PRIMARY KEY,
  status           TEXT NOT NULL
                   CHECK (status IN ('QUEUED','CLAIMED','RUNNING','SUCCEEDED','FAILED')),

  owner_id         TEXT,     -- 哪個 worker 拿走（hostname/uuid）
  lease_token      TEXT,     -- 本次 claim 的憑證（防殭屍）

  claimed_at       INTEGER,  -- unix epoch seconds (UTC)
  heartbeat_at     INTEGER,  -- unix epoch seconds (UTC)
  lease_expires_at INTEGER,  -- unix epoch seconds (UTC)

  retry_count      INTEGER NOT NULL DEFAULT 0,
  max_retry        INTEGER NOT NULL DEFAULT 5,

  finished_at      INTEGER,
  error_code       TEXT,
  error_detail     TEXT
);

-- sweeper 常用查詢路徑：狀態 + 到期時間
CREATE INDEX IF NOT EXISTS idx_jobs_lease
ON jobs(status, lease_expires_at);

-- queue 撿 job 路徑：QUEUED + id（或 priority）
CREATE INDEX IF NOT EXISTS idx_jobs_queue
ON jobs(status, id);
```

## 心跳更新

心跳的本質是：「我還活著，而且我還持有這個 job」。

所以 heartbeart 更新必須驗兩件事：

- 我是誰（`owner_id`）
- 我拿的是不是這次的鑰匙（`lease_token`）

```sql title="heartbeat（示意）"
UPDATE jobs
SET heartbeat_at = :now,
    lease_expires_at = :now + :lease_seconds
WHERE id = :job_id
  AND status IN ('CLAIMED', 'RUNNING')
  AND owner_id = :owner_id
  AND lease_token = :lease_token;
```

這是一筆很小的寫入。它不是要記錄大量資訊，它是在維護「系統的信任」。

## 回收過期 job

當 worker 失聯時，資料庫只會看到一件事：

- `lease_expires_at < now`

接下來你要做的，就是把這些 job 從 `RUNNING/CLAIMED` 拉回現實。

通常分兩類：

1. 還能救 → 放回 `QUEUED`，讓別人重跑
2. 救不了 → `FAILED`，讓它有個結果

## 常見問題

1. **時間格式亂寫**

   如果你把時間存成 `2025/10/8 9:3:1` 這種格式，再用字串比較大小，結果會很精彩。

   **對策**：用 UTC epoch seconds（`INTEGER`）。
   SQLite 直接拿現在時間：

   - `unixepoch('now')`（秒）
   - 或 `strftime('%s','now')`

2. **sweeper 太勤勞，反而讓 DB 開始吵架**

   你如果 sweeper 每 1 秒掃一次，每次更新一大堆 row，worker 同時在 heartbeat，你就會開始看到：

   - lock contention
   - busy_timeout 被打滿

   本來是想要救援，結果事情變得更糟？

   **對策**：

   - sweeper 間隔 10 ～ 30 秒
   - 每次批次處理（例如最多處理 100 筆）
   - 索引放對：`(status, lease_expires_at)`

3. **lease/heartbeat 解決「失聯」，但不解決「活著但永遠跑不完」**

   如果 worker 進入無限迴圈，但仍在更新 heartbeat，lease 不會救你。

   這時你需要的是 **job-level timeout**（最大執行時間），例如 `max_runtime_seconds`。
   sweeper 除了看 `lease_expires_at`，也要看「已跑多久」。

## 小結

要讓 polling queue 可維運，你至少要有兩個概念：

1. **lease（租約）**：worker claim job 不是永久擁有，而是暫時借走；到期就回收。
2. **heartbeat（心跳）**：worker 活著就續租；不續租就視為失聯。

再配上三個務實的配件：

- `max_retry / retry_count`：避免無限重試把機器跑到冒煙
- `error_code / error_detail`：讓失敗可以被理解、被分析
- `lease_token`：防殭屍 worker，避免兩個人同時寫同一個結果

做完這些之後，原本的 queue 才會從「看起來能跑」變成「出了事也能自己回來」。

把 worker 救回來，也就是把你從 debug 地獄中救回來。

怎麼算都不虧。

## 參考資料

- [SQLite: Date And Time Functions](https://www.sqlite.org/lang_datefunc.html)
- [The “Leases” Pattern](https://martinfowler.com/articles/patterns-of-distributed-systems/lease.html)
