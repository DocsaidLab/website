---
slug: sqlite-indexing-for-leaderboards
title: SQLite 實戰（四）：查詢好慢？
authors: Z. Yuan
tags: [sqlite, index, performance, leaderboard, query-plan]
image: /img/2025/1110.jpg
description: 用正確的索引，讓 `ORDER BY ... LIMIT` 真的只跑前 100 名。
---

你做了一個評測平台。

為了讓使用者有點成就感，還大發慈悲地做了個排行榜。

然後你打開 DevTools，看了一眼 API 回傳時間：三秒、五秒、十秒。

恩？

你不信邪，又看了一眼 SQL。

```sql
LIMIT 100
```

明明就只要前 100 名，為什麼能慢成這樣？

<!-- truncate -->

## 問題其實不在 `LIMIT`

這可能是個常見錯誤。

`LIMIT 100` **並不代表 SQLite 只處理 100 筆資料**。

如果它在找「前 100 名」之前，必須先把「全部符合條件的資料」都掃出來、排序完......

那你只是告訴它：

> 「最後只給我 100 筆就好，但前面你該做的事情，一樣都不能少。」

查詢慢，通常是因為你限制了 SQLite 只能用最笨的方式幫你找答案。

## 一個典型的查詢

多數評測平台的 schema，大致會長這樣：

- `jobs`：一次提交 / 一次 run 的基本資訊
  （狀態、版本、時間、queue）
- `job_scores`：實際的評測指標
  （可能還分 train / public / private split）

示意 SQL：

```sql title="leaderboard（示意）"
SELECT
  j.id,
  j.model_name,
  s.split,
  s.score1,
  s.score2
FROM jobs j
JOIN job_scores s ON s.job_id = j.id
WHERE j.status = 'SUCCEEDED'
  AND j.queue = :queue
  AND s.split = :split
ORDER BY
  s.score1 DESC,
  s.score2 DESC,
  j.created_at ASC
LIMIT :limit;
```

最後那個 `j.created_at ASC` 很關鍵。

這是一個**非常合理的設計**：
當分數相同時，用時間當 tie-breaker，讓排名穩定、不亂跳。

但對 SQLite 來說，這代表一件事：

> **你現在要求的是「跨表、多條件、複合排序」的 Top-N 查詢。**

如果索引下錯，它就只能硬幹。

## SQLite 在沒有索引時，會怎麼做？

簡化來看，流程通常是：

1. 找出所有 `status = 'SUCCEEDED'` 且 `queue = ?` 的 job
2. join 上所有符合 `split = ?` 的 score
3. 把結果全部拉出來
4. 依照 `score1 → score2 → created_at` 排序
5. 排完之後，丟掉 99% 的資料，只留下前 100 筆

你看到的慢，不是錯覺。

SQLite 真的有把資料「全掃一遍、喘一口氣、再幫你排好」。

## 索引怎麼下，才真的有用？

索引不是「有下就會快」。

排行榜這種 Top-N 查詢，要快，只有一個正確順序，而且不能顛倒：

1. **先讓 `WHERE` 快速縮小候選集合**
2. **再讓 `ORDER BY ... LIMIT` 能沿著索引順序吐出前 N 筆**

只要 SQLite 在排序階段還得「另外做一次排序」，那麼 `LIMIT 100` 幾乎救不了你，因為它已經先把該排的都排完了。

舉個例子，先看 `job_scores`（評測指標表）。

假設我們在查詢裡的排序是：

```sql
ORDER BY s.score1 DESC, s.score2 DESC, ...
```

所以索引也要用同樣的排序邏輯鋪路：

```sql title="index（示意）"
CREATE INDEX IF NOT EXISTS idx_scores_rank
ON job_scores (
  split,
  score1 DESC,
  score2 DESC
);
```

這個索引要達成的效果是：

- **先用 `split = ?` 篩出我們要的那一群資料**
- 篩完後，資料在索引裡已經是照 `score1 → score2` 排好的
- SQLite 可以「沿著索引往下讀」，很快就拿到前 100 名的候選

這裡比較像是：你不是叫 SQLite「把所有人叫來排隊」，而是直接帶它去「本來就排好隊的那條走廊」。

接著是 `jobs`（run 的基本資訊表），我們的查詢條件與 tie-breaker 在這裡：

- 篩選：`queue`、`status`
- 最後排序用：`created_at ASC`（分數相同時穩定排名）

所以索引要對應這三件事：

```sql title="index（示意）"
CREATE INDEX IF NOT EXISTS idx_jobs_filter
ON jobs (
  queue,
  status,
  created_at
);
```

這裡的 `created_at` **不是為了加速篩選**，而是為了加速「最後那一段」。

因為 `ORDER BY` 最後一段是：

```sql
... , j.created_at ASC
```

如果很多筆資料 `score1/score2` 都一樣（同分很多很常見），SQLite 必須：

1. 先把同分那一大坨候選人找出來
2. join 到 `jobs` 拿到每筆的 `created_at`
3. **再用 `created_at` 把這坨人重新排順**
4. 才能穩定選出前 100

如果 `created_at` 沒有對應的索引支援，SQLite 通常就會選擇：

- 建一個臨時排序結構（temp B-tree）
- 把同分候選塞進去
- 排完再吐結果

這就是我們在 query plan 裡看到的： `USE TEMP B-TREE FOR ORDER BY`

如果你想知道自己的指令是否有額外的排序，可以直接用 `EXPLAIN QUERY PLAN` 問 SQLite：

```sql
EXPLAIN QUERY PLAN
SELECT ...;
```

一個你會想看到的計畫通常長這樣：

- `USING INDEX ...`（有用到你下的索引）
- 沒有 `SCAN TABLE`（不是全表掃描）
- 沒有 `USE TEMP B-TREE FOR ORDER BY`（沒有額外排序）

只要看到 `TEMP B-TREE`，你就可以幾乎直接判定：

- 這個查詢還在做額外排序；Top-N 的 LIMIT 只是最後才生效，要改！

## 常見問題

1. **把數字存成 TEXT，再用 CAST 排序**

   ```sql
   ORDER BY CAST(score1 AS REAL) DESC
   ```

   這一行，會直接讓索引失效。

   SQLite 沒辦法用「轉型後的結果」來走索引。

   **對策很單純**：
   數值就用 `INTEGER / REAL` 存，不要留給排序時補救。

2. **排序欄位一路加，索引一路肥**

   `score1 → score2 → score3 → score4 → …`

   你最後會得到一個寫入慢、佔空間但查詢卻不一定真的變快的索引怪獸。

   **對策**：確認哪些指標，真的影響排名？盡量減少排序數量。

如果你的排行榜真的開始被頻繁點擊，那可以考慮幾個方式：

- **快取 Top-N 結果**（排行榜是標準讀多寫少）
- **針對熱門 split 建 partial index**
- **離線算排名，查詢結果表**

SQLite 很適合做「查詢引擎」，但也不代表你一定要每次都即時計算，透過一些小技巧，可以讓產品更穩定。

## 小結

查詢慢，最常見的原因不是 SQLite 不行。

而是你要求它處理的是：

- 跨表
- 多條件
- 複合排序
- 還要穩定排名的 Top-N 查詢

卻沒有給它能對應這些需求的索引。

於是它只能照最保守、最通用的方式做事：把資料全掃出來、排完，再丟掉大部分結果。

真正該調整的，往往不是查詢語法本身，而是你有沒有先想清楚：

- 哪些條件用來「縮小範圍」
- 哪些欄位真的「影響名次」
- 哪一段排序，不能讓 SQLite 再補一次

SQLite 一直都很穩定。

只是它需要你用更精確的方式，告訴它你真正想要的是什麼。

## 參考資料

- [SQLite 官方文件：Indexes](https://www.sqlite.org/lang_createindex.html)
- [SQLite 官方文件：Query Planner](https://www.sqlite.org/queryplanner.html)
- [SQLite 官方文件：EXPLAIN QUERY PLAN](https://www.sqlite.org/eqp.html)
