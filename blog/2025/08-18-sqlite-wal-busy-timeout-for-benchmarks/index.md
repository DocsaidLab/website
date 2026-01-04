---
slug: sqlite-wal-busy-timeout-for-workers
title: SQLite 實作（一）：資料庫又鎖死啦！
authors: Z. Yuan
tags: [sqlite, wal, busy-timeout, concurrency, job-queue]
image: /img/2025/0818.jpg
description: 用 WAL 與 busy_timeout，把多 worker 場景下的鎖衝突降到可接受的程度。
---

你以為用 SQLite 做一個內部系統會很省事。

直到你把 worker 多開兩隻，畫面開始不時跳出：

```
database is locked
```

更麻煩的是：
你其實沒有在做什麼「高併發操作」，只是很普通的背景工作流程：

- 撿一個 job
- 更新 heartbeat
- 寫回結果

每一筆寫入都很小，也很單純。

這樣也會鎖？

SQLite 到底行不行？

<!-- truncate -->

:::info
如果你對 SQLite 還不熟，可以建議先看看前面的文章：

- [**SQLite 初探（一）：為什麼又是你？**](/blog/sqlite-intro-embedded-database)
- [**SQLite 初探（二）：CLI、索引與 PRAGMA**](/blog/sqlite-basics-cli-transactions-pragmas)
  :::

## 為什麼這麼容易「鎖」？

SQLite 的併發模型其實不複雜，但很現實：

- **同一時間只有一個 writer**
- 讀通常很快
- 只要有寫入，就進入「鎖的世界」

在多 worker 的背景工作系統裡，問題往往不是「寫很多」，而是：

> **很多人同時想寫一點點。**

例如：

- claim job（改狀態）
- heartbeat（更新時間戳）
- 寫入結果或錯誤碼

每一筆都很小，但當 worker 數量一多，這些小寫入會在同一時間競爭同一把寫鎖。

最後，如你所見：又鎖住了。

## 先分清楚兩種錯誤

是 `BUSY` 還是 `LOCKED`？

實務上你看到的 `database is locked`，背後通常對應兩種狀況：

### `SQLITE_BUSY`

- 有其他 connection 正在寫
- 你現在拿不到鎖
- **可以等**

這正是 `busy_timeout` 能幫上忙的地方。

### `SQLITE_LOCKED`

- 同一條 connection 裡，有 statement 或 transaction 還沒結束
- 或是你在錯誤的時機重入使用連線
- **等也沒用**

如果你已經設了 `busy_timeout`，但錯誤還是「秒爆」，通常代表你遇到的是後者。

這時候該修的是使用方式，而不是再把 timeout 調更大。

## Write-Ahead Logging

WAL（Write-Ahead Logging）幾乎是多 worker SQLite 的基本配備。

它帶來的改變很明確：

- **reader 不太會擋 writer**
- 讀多、偶爾寫的場景，順很多

原因也很單純：
writer 把變更寫進 `-wal` 檔，reader 讀自己的 snapshot，彼此干擾大幅降低。

但 WAL **沒有**改變這件事：

> **同一時間，writer 還是只能有一個。**

所以常見現象會是：

- 單 worker：完全沒問題
- 多 worker：偶發鎖衝突

這算是併發壓力開始出現的自然結果，雖然問題還沒有完全解決，但至少處理了一部分。

不論如何，你可以用這個指令確認 WAL 是否真的生效：

```sql
PRAGMA journal_mode;
```

回傳 `wal` 才算數。

---

## busy_timeout

有時候，你其實可以接受「等一下」，不是「立刻失敗」。

SQLite 遇到鎖衝突時，預設行為非常直接：

> 拿不到鎖 → 直接回錯

`busy_timeout` 的作用，就是把這個策略改成：

> 拿不到鎖 → 等一小段時間重試 → 真的不行才失敗

在多 worker 的系統裡，這個差異非常關鍵。

這裡用 Python 寫個簡單的例子：

```python
import sqlite3


def open_db(path: str, *, busy_ms: int = 5000) -> sqlite3.Connection:
    conn = sqlite3.connect(path)

    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute(f"PRAGMA busy_timeout = {busy_ms};")

    return conn
```

在 Python 的 `sqlite3` 模組中，你也可以直接用：

```python
sqlite3.connect(path, timeout=5.0)
```

本質上做的是同一件事，擇一即可。

## 真正的關鍵

雖然剛剛解決了一部分的死鎖問題，但仍然會不定期發生。

真正排除問題的關鍵在於：

> **交易一定要短！**

WAL 和 `busy_timeout` 能改善衝突，但**救不了壞習慣**。

最常見的反模式大概是這樣：

```
BEGIN;
-- 撿 job
-- 跑一段很久的處理（I/O / 計算 / 外部呼叫）
-- 寫結果
COMMIT;
```

這等於對其他 worker 宣告：

> 我會拿著寫鎖很久，請大家排隊。

這是不良習慣，請不要這樣做。

正確的思路是把流程拆開：

1. **claim（占位）**：只做狀態切換，立刻提交
2. **結果寫入**：需要落盤時，再開一個短 transaction

claim 的典型寫法像是這樣：

```sql
BEGIN IMMEDIATE;

SELECT id
FROM jobs
WHERE queue = :queue
  AND status = 'QUEUED'
ORDER BY created_at ASC
LIMIT 1;

UPDATE jobs
SET status = 'RUNNING'
WHERE id = :id
  AND status = 'QUEUED';

COMMIT;
```

- `BEGIN IMMEDIATE`：先確認「我能不能寫」，不能就早點退
- `AND status = 'QUEUED'`：最簡單、但非常有效的 CAS
- `UPDATE` 影響筆數 ≠ 1 → claim 失敗，重試即可

這裡的重點在於：**縮短拿鎖的時間**。

## 幾個常見問題

1. **以為開 WAL 就萬事 OK**

   不會。

   WAL 只解決「讀寫互卡」，不解決「寫寫競爭」。

   **對策**：
   WAL + `busy_timeout` + 短 transaction，三件事缺一不可。

   ***

2. **claim 條件沒索引**

   如果你在 `BEGIN IMMEDIATE` 之後才開始掃全表，其他 worker 只能乾等。

   **對策**：
   把 claim 會用到的條件做成索引：

   ```sql
   CREATE INDEX IF NOT EXISTS idx_jobs_queue_status_created_at
   ON jobs(queue, status, created_at);
   ```

   ***

3. **多執行緒共用同一條連線**

   這通常只會省到錯誤，不會省到效能。

   **對策**：
   一個 thread / process 一條 connection，或明確序列化連線使用。

   ***

4. **把 SQLite 放在不可靠的共享檔案系統**

   某些 NFS 或網路磁碟，會讓鎖行為變得不可預期。

   **對策**：
   DB 放本機磁碟；如果不行，就該重新評估技術選擇。

   ***

其他可以嘗試的方向像是：

- 把 queue 狀態與結果寫入拆開（分表或分庫）
- 批次寫入結果，降低 transaction 次數
- 當你真的需要多 writer 時，改用 client/server DB

SQLite 很好用，但它不會替你承擔所有併發壓力。

## 小結

WAL 和 `busy_timeout` 不是讓 SQLite 變成「可以亂寫」的魔法。

它們只是讓你的系統在現實條件下，**更有餘裕面對衝突**。

只要記住三件事：

- WAL 解決讀寫互卡，不解決寫寫併發
- busy_timeout 能讓衝突變成等待，但前提是交易夠短
- 把會拿寫鎖的 SQL 壓縮到最小，並在連線初始化時設好 PRAGMA

那 `database is locked` 通常就會從「滿版紅字」，減少成「偶爾重試一下」。

## 參考資料

- [SQLite 官方文件：Write-Ahead Logging](https://www.sqlite.org/wal.html)
- [PRAGMA busy_timeout 的行為說明](https://www.sqlite.org/pragma.html#pragma_busy_timeout)
- [SQLite 的鎖模型與併發說明](https://www.sqlite.org/lockingv3.html)
- [Transaction 的實際語意與限制](https://www.sqlite.org/lang_transaction.html)
