---
slug: sqlite-basics-cli-transactions-pragmas
title: SQLite 初探（二）：CLI、索引與 PRAGMA
authors: Z. Yuan
tags: [sqlite, sql, cli, transactions, pragmas, index]
image: /img/2025/0811.jpg
description: 讓 SQLite 聽話：從看 schema 到 query plan。
---

你在專案裡看到一個 `.db` 檔。

它不像是一般的檔案，一時半刻還打不開，打開了還看不懂？

於是你超級好奇：這檔案裡到底塞了什麼？

> **什麼？你說沒有，誰會好奇這種東西？**

<!-- truncate -->

## 1. sqlite3 CLI 幾個常用指令

總之，我們先認識一些基本指令吧。

先進入 CLI：

```bash
sqlite3 app.db
```

這幾個指令基本上可以救你一半的人生：

```text
.tables          -- 看有哪些表
.schema          -- 看整份 schema
.schema jobs     -- 看單一表 schema
.headers on      -- 查詢結果顯示欄名
.mode column     -- 用欄位對齊的方式顯示
.timer on        -- 顯示每個 SQL 花多久
.quit            -- 離開
```

如果你想快速驗證某段 SQL，也可以：

```text
.read init.sql
```

把一份 SQL 檔直接執行掉。

## 2. 交易（Transaction）

SQLite 的讀取併發通常沒問題，但它對「寫入交易」非常保守：

- 同一時間只能有一個 writer（能持有寫入權）

所以在做「狀態更新」「扣款轉帳」「claim job」這種事情時，請先把交易概念放到腦袋最前面。

最小的交易長這樣：

```sql
BEGIN; -- 預設是 DEFERRED
-- 你的多筆更新
COMMIT;
```

BEGIN（DEFERRED）代表：一開始先不搶寫入權，等到第一個需要寫入的語句才嘗試拿鎖。

如果你不希望做到一半才發現「寫不了」，可以用這種寫法（示意）：

```sql
BEGIN IMMEDIATE;
-- 需要原子性的更新
COMMIT;
```

:::tip
`IMMEDIATE` 的直覺就是：先把「我要寫」這件事講清楚，拿得到鎖再開始做。
:::

### 範例：用 CAS 方式 claim job

假設你要把一筆任務從 `QUEUED` 變成 `RUNNING`，最安全的做法是把「舊狀態比對」放進同一條 SQL：

```sql
UPDATE jobs
SET status = 'RUNNING'
WHERE id = :id
  AND status = 'QUEUED';
```

然後在應用層檢查影響筆數：

- 是 1：水啦！你搶到了！
- 是 0：噢不，它被別人搶走了（這時候別硬跑）

## 3. PRAGMA

PRAGMA 是 SQLite 用來調整行為的「設定指令」，而且很多是只影響當下這條連線。

SQLite 有一堆 PRAGMA，先記住幾個你最常需要的：

```sql
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 30000;
```

這些設定的意思是：

- `foreign_keys`：資料完整性的最低防線（但你要自己開）
- `journal_mode`：併發讀寫的體感差很多（WAL 常是第一步）
- `synchronous`：更安全 vs 更快（通常是取捨）
- `busy_timeout`：遇到鎖先等一下，不要立刻爆炸

如果你的程式會頻繁建立新連線，請把 PRAGMA 當成「連線初始化的一部分」。

:::warning
有些 PRAGMA 是「連線層級」，你換一條連線就要再設一次；不要以為設過一次就永遠有效。
:::

## 4. EXPLAIN QUERY PLAN

> 怎麼這麼慢？你是不是又偷偷跑全表掃描？

很多慢查詢看起來都很無辜：

- 明明有 `LIMIT 100`
- 明明條件也不複雜

但如果沒有合適的索引，LIMIT 幫不了你太多。

因為資料庫在排序前，根本不知道哪幾筆才是你要的那 100 筆，於是 SQLite 只好：

1. 掃很多資料
2. 排序
3. 再從裡面拿前 100 筆

你可以用 `EXPLAIN QUERY PLAN` 問它到底怎麼跑（示意）：

```sql
EXPLAIN QUERY PLAN
SELECT id
FROM jobs
WHERE status = 'QUEUED'
ORDER BY created_at ASC
LIMIT 1;
```

如果你看到的是類似 `SCAN TABLE jobs`，那代表它真的在掃表。

## 常見問題

很多人看到 `INSERT OR REPLACE`，直覺是：

> **「喔，資料在就更新，不在就插入。」**

但 `REPLACE` 的語意更接近：

> **「我先把舊資料刪掉，再插一筆新的。」**

這在你有外鍵、或你想保留某些欄位（例如 created_at）時，特別容易踩雷。

**對策**：

- 優先用 `ON CONFLICT DO UPDATE`（SQLite 支援 UPSERT）
- 或者明確拆成 `INSERT` / `UPDATE`（應用層控制）

## 小結

SQLite 從一開始就很誠實地告訴你：

> **資料庫引擎就在你的程式裡，後果你自己承擔。**

這篇文章介紹的東西，看起來零碎：

- CLI 指令
- 交易模式
- PRAGMA
- 索引與 query plan

但它們其實都在回答同一個問題：

> **當資料庫不再替你管理一切時，你至少要自己顧好哪些事？**

如果你記住這幾個操作，SQLite 通常不會有問題：

- 寫入一定要有交易意識
- PRAGMA 要當成連線初始化的一部分
- 沒有索引，就不要怪查詢慢
- 用 `EXPLAIN QUERY PLAN` 問清楚它在幹嘛

之後，當你發現自己開始需要處理多個 writer 長時間同時寫入，或是有複雜的權限與角色，中央化的備援、replica、觀測等。那通常不是 SQLite 做錯事，而是你已經走到它設計邊界之外了。

在那之前，把這些基本功顧好，它會是一個非常可靠、低摩擦、低維運成本的工具。

只有一個 `.db` 檔？

足夠了。

## 參考資料

- [**SQLite: Command Line Shell**](https://www.sqlite.org/cli.html)
- [**SQLite: PRAGMA**](https://www.sqlite.org/pragma.html)
- [**SQLite: UPSERT**](https://www.sqlite.org/lang_UPSERT.html)
- [**SQLite: EXPLAIN QUERY PLAN**](https://www.sqlite.org/eqp.html)
