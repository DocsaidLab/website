---
slug: sqlite-basics-cli-transactions-pragmas
title: "A First Look at SQLite (2): The CLI, Indexes, and PRAGMA"
authors: Z. Yuan
tags: [sqlite, sql, cli, transactions, pragmas, index]
image: /en/img/2025/0811.jpg
description: "Taming SQLite: from schema inspection to query plans."
---

You spot a `.db` file in a project.

It’s not like a normal file—you can’t just open it and read it. And even if you do open it, it still doesn’t make sense at a glance.

So you get really curious: what’s actually inside this thing?

> **What? You’re saying you’re not curious—who would be curious about this?**

<!-- truncate -->

## 1. A few handy sqlite3 CLI commands

Anyway, let’s start with some basic commands.

Enter the CLI:

```bash
sqlite3 app.db
```

These commands will save half your life:

```text
.tables          -- list tables
.schema          -- show full schema
.schema jobs     -- show schema for a table
.headers on      -- show column headers
.mode column     -- align output in columns
.timer on        -- show execution time per SQL
.quit            -- exit
```

If you want to quickly validate a chunk of SQL, you can also run:

```text
.read init.sql
```

That executes an entire SQL file directly.

## 2. Transactions

SQLite is usually fine with concurrent reads, but it’s very conservative about **write transactions**:

- At any given time, there can only be one writer (holding the write lock)

So for things like “status updates”, “debit/transfer”, or “claiming jobs”, keep transactions at the very top of your mind.

The smallest transaction looks like this:

```sql
BEGIN; -- default is DEFERRED
-- your multiple updates
COMMIT;
```

`BEGIN` (DEFERRED) means it won’t try to grab the write lock at the start; it only attempts to lock when the first statement that needs to write runs.

If you don’t want to discover “I can’t write” halfway through, you can do this instead (example):

```sql
BEGIN IMMEDIATE;
-- updates that must be atomic
COMMIT;
```

:::tip
Intuitively, `IMMEDIATE` means: make it explicit that “I’m going to write”, and only start once you actually have the lock.
:::

### Example: claiming a job with CAS

Suppose you want to move a job from `QUEUED` to `RUNNING`. The safest approach is to put the “old state check” into the same SQL statement:

```sql
UPDATE jobs
SET status = 'RUNNING'
WHERE id = :id
  AND status = 'QUEUED';
```

Then check the affected row count in your application:

- 1: nice—you got it
- 0: nope, someone else got there first (don’t force it)

## 3. PRAGMA

`PRAGMA` is SQLite’s “configuration command” for adjusting behavior, and many of them only apply to the current connection.

SQLite has a lot of PRAGMAs; here are a few you’ll use the most:

```sql
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 30000;
```

What they mean:

- `foreign_keys`: the bare minimum line of defense for data integrity (but you have to enable it)
- `journal_mode`: makes a huge difference for read/write concurrency (WAL is often the first step)
- `synchronous`: safer vs faster (usually a trade-off)
- `busy_timeout`: when you hit a lock, wait a bit instead of exploding immediately

If your program frequently opens new connections, treat PRAGMAs as part of “connection initialization”.

:::warning
Some PRAGMAs are connection-level: if you open a new connection, you must set them again. Don’t assume “setting it once” lasts forever.
:::

## 4. EXPLAIN QUERY PLAN

> Why is this so slow? Did you secretly run a full table scan again?

A lot of slow queries look completely innocent:

- you clearly have `LIMIT 100`
- the conditions don’t look complicated either

But without the right index, `LIMIT` doesn’t help as much as you’d think.

Because before sorting, the database doesn’t know which 100 rows you want—so SQLite has to:

1. scan a lot of data
2. sort it
3. take the first 100 rows from the sorted result

You can use `EXPLAIN QUERY PLAN` to ask how it’s actually executed (example):

```sql
EXPLAIN QUERY PLAN
SELECT id
FROM jobs
WHERE status = 'QUEUED'
ORDER BY created_at ASC
LIMIT 1;
```

If you see something like `SCAN TABLE jobs`, it means it really is scanning the table.

## Common questions

When people see `INSERT OR REPLACE`, the first instinct is:

> **“Oh, it updates if the row exists, otherwise it inserts.”**

But `REPLACE` is closer to:

> **“Delete the old row first, then insert a new one.”**

This is especially easy to trip over if you have foreign keys, or you want to preserve some columns (like `created_at`).

**What to do instead**:

- Prefer `ON CONFLICT DO UPDATE` (SQLite supports UPSERT)
- Or explicitly split it into `INSERT` / `UPDATE` (control it in the application)

## Summary

From day one, SQLite has been very honest with you:

> **The database engine is inside your application, and you’re responsible for the consequences.**

What this post covered may feel a bit scattered:

- CLI commands
- transaction modes
- PRAGMAs
- indexes and query plans

But they all answer the same question:

> **When the database no longer manages everything for you, what do you at least need to manage yourself?**

If you remember these, SQLite is usually fine:

- Always think in transactions when writing
- Treat PRAGMAs as part of connection initialization
- If you don’t have an index, don’t blame the query for being slow
- Use `EXPLAIN QUERY PLAN` to see what it’s doing

Later, when you realize you need multiple writers writing for long periods of time, or complex permissions/roles, centralized backups, replicas, observability, and so on—that’s usually not SQLite “doing something wrong”; it’s you reaching beyond its design boundary.

Until then, keep these fundamentals solid, and SQLite will be a very reliable, low-friction, low-ops-cost tool.

Just one `.db` file?

That’s enough.

## References

- [SQLite: Command Line Shell](https://www.sqlite.org/cli.html)
- [SQLite: PRAGMA](https://www.sqlite.org/pragma.html)
- [SQLite: UPSERT](https://www.sqlite.org/lang_UPSERT.html)
- [SQLite: EXPLAIN QUERY PLAN](https://www.sqlite.org/eqp.html)
