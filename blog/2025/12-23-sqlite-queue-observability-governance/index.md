---
slug: sqlite-queue-observability-governance
title: SQLite 實戰（五）：我是誰？我在哪？
authors: Z. Yuan
tags: [sqlite, observability, data-governance, job-queue]
image: /img/2025/1223.jpg
description: 從錯誤碼、留存策略到 audit log，讓佇列系統可維運。
---

你一定有過這種對話：

- 你：系統現在跑得怎樣？
- 系統：我在跑。
- 你：跑哪裡？
- 系統：我在跑。

好的，謝謝，完全沒有幫助。

<!-- truncate -->

## Queue

用 SQLite 做 queue 當然可以，但你至少要能回答五個工程問題：

1. **現在佇列裡有多少 job？各狀態多少？**
2. **最老的 job 卡了多久？卡在誰身上（哪個 worker）？**
3. **重試次數分佈如何？是不是某一類一直失敗？**
4. **失敗的原因是什麼？能不能聚合（Top-N）？**
5. **資料要留多久？怎麼刪才不會刪出孤兒資料、也不會把 DB 刪爆？**

這五個問題背後，其實就是兩個關鍵字：

- **Observability（可觀測性）**：你能不能「用資料」看懂系統目前狀態，而不是靠祈禱。
- **Governance（資料治理）**：你能不能「可控地」保留與清理資料，且清理後仍然一致、可追溯。

下面我們用一套偏實務的設計，把這兩件事落地。

## 不要把 Queue 當成一張表

很多人做 queue 的直覺是：

- 一張 `jobs` 表
- 狀態欄位 `status`
- 反正需要什麼就塞欄位

跑起來當然可以，但它會很快變成「你想問的問題，schema 回答不了」。

### 你真正需要的，是三種資料層次

把 queue 拆成三個層次，你會突然變得很好維運：

1. **Job 本體（Current State）**
   你要能快速知道「現在這個 job 目前是什麼狀態」。

2. **嘗試紀錄（Attempts / Retries）**
   同一個 job 可能跑很多次。你需要回答：「第幾次失敗？每次失敗原因一樣嗎？耗時如何？」

3. **事件紀錄（Audit Log / Event Log）**
   狀態是結果，事件是過程。真正好 debug 的系統，靠的是事件流而不是「盯著某一列」。

## 你需要哪些「可觀測」欄位？

「可觀測」不是叫你把 schema 寫成百科全書，而是要能回答剛才那五個問題。

### 建議的 jobs 表

- `status`：狀態機（Queued / Claimed / Running / Succeeded / Failed / Cancelled …）
- `created_at / started_at / finished_at`：時間軸（讓你能算等待時間、處理時間）
- `priority`（可選）：不要小看它，後面排程與維運會謝謝你
- `worker_id`（或 `claimed_by`）：誰拿走的（用於定位問題 worker）
- `heartbeat_at`：心跳時間（判斷 worker 是否還活著）
- `retry_count`：目前已重試幾次（快速統計用）
- `error_code`：可聚合的錯誤分類（Top-N 靠它）
- `error_detail`：短摘要（用來快速定位，不是用來存整份日誌）

### 建議的 schema

這裡只是一個建議，我們試著把維運問題切乾淨。

1. **jobs：當前狀態（跑系統要快）**

   ```sql
   CREATE TABLE jobs (
     id            INTEGER PRIMARY KEY,
     type          TEXT NOT NULL,                 -- job 類型（用於分群統計）
     status        TEXT NOT NULL,                 -- 狀態機
     priority      INTEGER NOT NULL DEFAULT 0,

     created_at    INTEGER NOT NULL,              -- Unix epoch seconds
     started_at    INTEGER,
     finished_at   INTEGER,

     claimed_by    TEXT,                          -- worker_id
     claim_token   TEXT,                          -- 防止誤更新（可選）
     heartbeat_at  INTEGER,

     retry_count   INTEGER NOT NULL DEFAULT 0,
     max_retries   INTEGER NOT NULL DEFAULT 3,

     error_code    TEXT,                          -- 可聚合分類
     error_detail  TEXT,                          -- 短摘要（建議限制長度）

     payload_ref   TEXT,                          -- 大 payload 另外存（檔案/物件儲存）
     result_ref    TEXT                           -- 大結果另外存
   );
   ```

2. **job_attempts：每次嘗試（你要知道「第幾次」）**

   ```sql
   CREATE TABLE job_attempts (
     id           INTEGER PRIMARY KEY,
     job_id       INTEGER NOT NULL,
     attempt      INTEGER NOT NULL,               -- 1,2,3...
     started_at   INTEGER NOT NULL,
     finished_at  INTEGER,
     status       TEXT NOT NULL,                  -- RUNNING/SUCCEEDED/FAILED
     error_code   TEXT,
     error_detail TEXT,
     worker_id    TEXT,

     FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE,
     UNIQUE(job_id, attempt)
   );
   ```

3. **job_events：事件流（debug 靠它）**

   ```sql
   CREATE TABLE job_events (
     id          INTEGER PRIMARY KEY,
     job_id      INTEGER NOT NULL,
     ts          INTEGER NOT NULL,
     event       TEXT NOT NULL,                   -- CLAIMED/STARTED/HEARTBEAT/FAILED/RETRY_SCHEDULED...
     actor       TEXT,                            -- worker_id / system
     detail      TEXT,                            -- 短 JSON（別塞爆）

     FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
   );
   ```

   > 為什麼要 event 表？
   > 因為「狀態」只告訴你現在是 FAILED，但「事件」會告訴你：
   >
   > - 何時被誰 claim
   > - 跑了多久
   > - 心跳最後一次在哪
   > - 哪次 retry 開始失敗
   >
   > 我們 debug 不靠猜，而是靠歷史。

## 索引

可觀測性不是寫幾條 SQL 就完成了：

- **至少，你要確保它們跑得動，對吧？**。

### 常用索引（建議）

```sql
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_type_status ON jobs(type, status);

-- 你最常用來找卡死、找最老的 running
CREATE INDEX idx_jobs_running_age
ON jobs(status, heartbeat_at, created_at);

-- 失敗聚合 Top-N
CREATE INDEX idx_jobs_failed_code
ON jobs(status, error_code);

-- attempts / events 的查詢主鍵
CREATE INDEX idx_attempts_job ON job_attempts(job_id, attempt);
CREATE INDEX idx_events_job_ts ON job_events(job_id, ts);
```

設計原則很簡單：**我們想在 Dashboard 查什麼，你就替它做索引。**

## 最常用的觀測查詢

1. **各狀態計數：我現在到底多忙？**

   ```sql
   SELECT status, COUNT(*) AS cnt
   FROM jobs
   GROUP BY status
   ORDER BY cnt DESC;
   ```

   你會很快發現：

   - `QUEUED` 爆量 → 上游送太快、下游處理太慢
   - `RUNNING` 很多但吞吐沒上升 → worker 卡住或資源不足
   - `FAILED` 緩慢爬升 → 某類 job 出 bug 或外部依賴壞了

2. **找出卡最久的進行中 job？**

   ```sql
   SELECT id, type, claimed_by, created_at, heartbeat_at
   FROM jobs
   WHERE status IN ('CLAIMED', 'RUNNING')
   ORDER BY COALESCE(heartbeat_at, created_at) ASC
   LIMIT 20;
   ```

   這條的重點在 `COALESCE`：

   - 有心跳就用心跳當「最近活著的證據」
   - 沒心跳就退回用 created_at（通常表示還沒真的跑起來或設計漏了）

3. **重試分佈：是不是某一類一直在重跑？**

   ```sql
   SELECT retry_count, COUNT(*) AS cnt
   FROM jobs
   WHERE status IN ('QUEUED', 'CLAIMED', 'RUNNING', 'FAILED')
   GROUP BY retry_count
   ORDER BY retry_count DESC;
   ```

   如果你看到 `retry_count = 3` 一坨，通常意味著：

   - 外部服務掛掉（timeout）
   - 資料格式不合（invalid input）
   - 系統性 bug（internal）

   下一步就該接錯誤碼聚合。

4. **失敗原因 Top-N：必須要能「聚合」**

   ```sql
   SELECT error_code, COUNT(*) AS cnt
   FROM jobs
   WHERE status = 'FAILED'
   GROUP BY error_code
   ORDER BY cnt DESC
   LIMIT 20;
   ```

   這條能跑起來的前提，是你有「像樣的 error_code」。

## 錯誤碼設計

error_code 要為「統計」而生，最常見的反模式是：

- `error_detail` 存整段 stack trace
- `error_code` 不存或亂存

最後你只能「逐筆看日誌」，完全無法聚合

### 建議的錯誤碼層級

你可以用 `CATEGORY:SUBCATEGORY` 或 `CATEGORY/SUBCATEGORY`：

- `TIMEOUT:UPSTREAM_API`
- `INVALID_INPUT:SCHEMA_MISMATCH`
- `RESOURCE:OUT_OF_MEMORY`
- `INTERNAL:ASSERTION_FAILED`
- `DEPENDENCY:DB_LOCKED`

### error_detail 的原則

- **短摘要**即可（例如 200~500 字）
- 完整 stack trace、完整 JSON、完整 report 放到檔案系統或 object storage，DB 只存 `path/URL`

我們看過太多「垃圾場型 log storage」，真心建議你不要步上這個後塵。

## Foreign key 與孤兒資料

> 你不解決它？之後它就會解決你。

SQLite 的外鍵支援是「有，但不預設啟用」。

也就是你不開，它就當沒看到。

這裡你必須做到兩件事：

1. **每條連線都開外鍵**

```sql
PRAGMA foreign_keys = ON;
```

這不是一次性設定。你用連線池或每次開新連線，都要做。

2. **用 ON DELETE CASCADE 保一致性**

你刪 `jobs`，就要讓 `attempts/events` 自動跟著刪，不然你一定會留下孤兒資料。

## Upsert

別再用 INSERT OR REPLACE 當「更新」！

這個坑值得用粗體寫三遍：

- **`REPLACE` 不是 update**
- **`REPLACE` 不是 update**
- **`REPLACE` 不是 update**

它比較像：

> 先 DELETE 再 INSERT

如果你有外鍵、有 audit log、有 attempt/event，`REPLACE` 會直接讓你往生。

然後你在往生之後，還找不到問題在哪裡。

正確姿勢是使用 `ON CONFLICT DO UPDATE`，狀態更新要可預期、可追溯、不要偷做刪除，像是這樣：

```sql
INSERT INTO jobs(id, status, heartbeat_at)
VALUES (?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
  status = excluded.status,
  heartbeat_at = excluded.heartbeat_at;
```

## Audit log（事件表）

如果你只靠 `jobs` 那一列資料，debug 就像你只看到了「現在」，但是看不到「為什麼變成現在」。

事件表的設計哲學是：**狀態是快照；事件是時間線。**

你至少應該記這些資訊：

- `ENQUEUED`：入列
- `CLAIMED`：被 worker 拿走
- `STARTED`：正式開始（可與 claimed 分開）
- `HEARTBEAT`：心跳（可降低頻率或只記「最後一次」）
- `FAILED`：失敗（附 error_code）
- `RETRY_SCHEDULED`：安排重試（附 delay、attempt）
- `SUCCEEDED`：成功
- `CANCELLED`：人工取消
- `RECOVERED`：回收/重派（如果你有 lease 機制）

## 留存策略

資料要刪，但要「有規則地刪」

Queue 系統最終都會面對一個問題：

> **你要留多久？**

如果你說「都留」，SQLite 會用容量跟你討論人生。
如果你說「都刪」，你下一次事故就沒有證據。

比較務實的策略是：

- `SUCCEEDED`：留 30 天（通常夠追問題）
- `FAILED`：留 90 天（失敗要多留，因為會反覆回來）
- `CANCELLED`：看需求（常見 30 天）

刪除流程則是建議：分批、可中斷、可重跑。

不要一次刪 100 萬筆，尤其你還要提供服務的情況下，可以用排程每分鐘跑一次，慢慢清，像是這樣：

```sql
-- 刪成功 job（30 天前）
DELETE FROM jobs
WHERE status = 'SUCCEEDED'
  AND finished_at < strftime('%s', 'now') - 30*86400
LIMIT 5000;
```

### 需要 VACUUM 嗎？

- SQLite 的檔案空間不一定會立刻縮回去（刪除只是在內部標記可重用）
- `VACUUM` 會重寫整個 DB，I/O 大、時間長

建議做法：

- **日常靠分批刪除**
- **真的需要縮檔（例如容量壓力）才在低峰期 VACUUM**

如果你是 WAL 模式，也記得理解 WAL 檔案與 checkpoint 的行為。

## 常見問題

1. **`error_detail` 無上限，最後 DB 變垃圾場**

   **對策**：

   - `error_code` 做分類
   - `error_detail` 只存短摘要
   - 完整報告放外部（DB 只存引用）

2. 忘了開 `foreign key`，刪主表留下孤兒資料

   **對策**：每條連線都 `PRAGMA foreign_keys = ON`，並用 `ON DELETE CASCADE`

3. 用 `INSERT OR REPLACE` 當 upsert

   **對策**：用 `ON CONFLICT DO UPDATE`，不要用 REPLACE 偷偷做 DELETE

4. 你寫了觀測查詢，但沒索引

   **對策**：把 dashboard/維運最常用的查詢，反推索引（不要跟 SQLite 硬拼全表掃描）

## 小結

當你把 queue 做到能回答那五個問題，你會發現系統突然變得「像個系統」：

- 你能描述現況（可觀測）
- 你能追溯過程（audit）
- 你能控制資料生命週期（治理）
- 你能把事故從「猜」變成「查」

做完這些，SQLite 被我們拿來做 queue 也是水到渠成的事。

## 參考資料

- [**SQLite: Foreign Key Support**](https://www.sqlite.org/foreignkeys.html)
- [**SQLite: UPSERT**](https://www.sqlite.org/lang_UPSERT.html)
- [**SQLite: VACUUM**](https://www.sqlite.org/lang_vacuum.html)
