---
slug: sqlite-indexing-for-leaderboards
title: "SQLite in Practice (4): Why Is My Query So Slow?"
authors: Z. Yuan
tags: [sqlite, index, performance, leaderboard, query-plan]
image: /en/img/2025/1110.jpg
description: Use the right indexes so `ORDER BY ... LIMIT` can stop after the top 100.
---

You built a benchmarking platform.

To give users a bit of a sense of achievement, you even (in a rare act of mercy) added a leaderboard.

Then you open DevTools and check the API latency: three seconds, five seconds, ten seconds.

Huh?

You don’t buy it, so you look at the SQL again.

```sql
LIMIT 100
```

We only need the top 100. How can it be this slow?

<!-- truncate -->

## The Problem Isn’t `LIMIT`

This is a common misunderstanding.

`LIMIT 100` **does not mean SQLite only processes 100 rows**.

If, before it can find the “top 100”, it has to scan all matching rows and sort them first...

Then all you’ve told it is:

> “Just give me 100 rows at the end — but everything you need to do before that still has to happen.”

Queries get slow when you leave SQLite no choice but to do it the dumbest, most brute-force way.

## A Typical Query

Most benchmarking platforms have a schema that looks roughly like this:

- `jobs`: basic info for a submission / a run
  (status, version, timestamp, queue)
- `job_scores`: the actual evaluation metrics
  (often split into train / public / private)

Illustrative SQL:

```sql title="leaderboard (illustration)"
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

That last `j.created_at ASC` is key.

It’s a **very reasonable design**:
when scores tie, use time as a tie-breaker so the ranking stays stable and doesn’t jump around.

But to SQLite, it means one thing:

> **You’re asking for a Top-N query with joins, filters, and a composite sort.**

If the indexes are wrong, it has to brute-force it.

## What SQLite Does Without Indexes

Simplified, the flow is usually:

1. Find all jobs with `status = 'SUCCEEDED'` and `queue = ?`
2. Join all scores that match `split = ?`
3. Pull the entire result set
4. Sort by `score1 → score2 → created_at`
5. After sorting, throw away 99% and keep only the top 100

The slowness isn’t your imagination.

SQLite really does “scan everything, catch its breath, and then sort it for you.”

## Indexes That Actually Help

Indexes aren’t “fast just because they exist.”

For Top-N leaderboard queries, there’s only one correct order — and you can’t swap it:

1. **Make `WHERE` quickly shrink the candidate set**
2. **Make `ORDER BY ... LIMIT` able to stream the first N rows along index order**

If SQLite still needs to do an extra sort, `LIMIT 100` won’t save you — because it already sorted everything it needed to.

Let’s start with `job_scores` (the metrics table).

Suppose the query orders by:

```sql
ORDER BY s.score1 DESC, s.score2 DESC, ...
```

Then your index needs to “pave the road” with the same ordering:

```sql title="index (illustration)"
CREATE INDEX IF NOT EXISTS idx_scores_rank
ON job_scores (
  split,
  score1 DESC,
  score2 DESC
);
```

What this index gives you:

- **Use `split = ?` to filter down to the group you actually want**
- After filtering, the rows are already ordered by `score1 → score2` inside the index
- SQLite can “walk the index” and quickly grab candidates for the top 100

It’s less like asking SQLite to “gather everyone and make them line up,” and more like taking it straight to the hallway where they’re already lined up.

Next is `jobs` (the run metadata table). This is where your filters and tie-breaker live:

- Filters: `queue`, `status`
- Final tie-breaker: `created_at ASC` (stable ordering for equal scores)

So your index needs to match those three things:

```sql title="index (illustration)"
CREATE INDEX IF NOT EXISTS idx_jobs_filter
ON jobs (
  queue,
  status,
  created_at
);
```

Here, `created_at` **isn’t for filtering** — it’s for speeding up “the last mile.”

Because the last part of `ORDER BY` is:

```sql
... , j.created_at ASC
```

If many rows share the same `score1/score2` (ties are common), SQLite has to:

1. First gather that big blob of tied candidates
2. Join into `jobs` to read each row’s `created_at`
3. **Re-order that blob by `created_at`**
4. Only then can it stably pick the top 100

If there’s no supporting index for `created_at`, SQLite will often choose to:

- build a temporary sort structure (a temp B-tree)
- push those tied candidates into it
- sort and then emit results

That’s what you see in the query plan: `USE TEMP B-TREE FOR ORDER BY`.

If you want to know whether your query is doing an extra sort, ask SQLite directly with `EXPLAIN QUERY PLAN`:

```sql
EXPLAIN QUERY PLAN
SELECT ...;
```

A plan you want to see usually looks like:

- `USING INDEX ...` (it’s using your indexes)
- no `SCAN TABLE` (it’s not doing a full table scan)
- no `USE TEMP B-TREE FOR ORDER BY` (no extra sort)

If you see `TEMP B-TREE`, you can almost immediately conclude:

- the query is still doing an extra sort; the Top-N LIMIT only kicks in at the very end — fix it.

## Common Pitfalls

1. **Storing numbers as TEXT, then sorting with `CAST`**

   ```sql
   ORDER BY CAST(score1 AS REAL) DESC
   ```

   This line makes your index unusable.

   SQLite can’t use an index on “the casted result.”

   **Fix**:
   store numbers as `INTEGER / REAL` — don’t rely on `CAST` at sort time to save you.

2. **Keep adding sort columns, and the index keeps getting fatter**

   `score1 → score2 → score3 → score4 → …`

   Eventually you end up with an index monster that slows writes, eats space, and still doesn’t necessarily make the query faster.

   **Fix**: confirm which metrics actually affect ranking, and keep the sort key as small as you can.

If your leaderboard starts getting clicked a lot, consider:

- **Caching Top-N results** (leaderboards are read-heavy, write-light)
- **Partial indexes for popular splits**
- **Offline ranking + a result table**

SQLite is great as a “query engine,” but that doesn’t mean you must compute rankings live every time. A few small tricks can make the product much more stable.

## Wrap-up

When a query is slow, the most common reason isn’t that SQLite is bad.

It’s that you’re asking it to handle:

- joins
- multiple filters
- a composite sort
- a stable Top-N ranking

...without giving it indexes that actually match those requirements.

So it falls back to the most conservative, general approach: scan everything, sort everything, then throw most of it away.

What you usually need to adjust isn’t the query syntax itself, but whether you’ve thought through:

- which conditions actually “shrink the search space”
- which columns truly “change the ranking”
- which part of the ordering must not force SQLite to do another sort pass

SQLite has always been stable.

It just needs you to tell it — more precisely — what you really want.

## References

- [SQLite Docs: Indexes](https://www.sqlite.org/lang_createindex.html)
- [SQLite Docs: Query Planner](https://www.sqlite.org/queryplanner.html)
- [SQLite Docs: EXPLAIN QUERY PLAN](https://www.sqlite.org/eqp.html)
