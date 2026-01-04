---
slug: sqlite-wal-busy-timeout-for-workers
title: "SQLite in Practice (1): The Database Is Locked Again!"
authors: Z. Yuan
tags: [sqlite, wal, busy-timeout, concurrency, job-queue]
image: /en/img/2025/0818.jpg
description: Use WAL and busy_timeout to bring lock contention in multi-worker setups down to an acceptable level.
---

You think building an internal system with SQLite is going to be easy.

Until you spin up a couple more workers, and the screen starts flashing this from time to time:

```
database is locked
```

Even worse:
you’re not doing anything “high-concurrency”. It’s just a normal background workflow:

- pick a job
- update the heartbeat
- write back the result

Each write is tiny and straightforward.

And it still locks?

Is SQLite actually usable or not?

<!-- truncate -->

:::info
If you’re not familiar with SQLite yet, it’s worth reading the earlier posts first:

- [**A First Look at SQLite (1): Why Is It Always You?**](/en/blog/sqlite-intro-embedded-database)
- [**A First Look at SQLite (2): CLI, Indexes, and PRAGMA**](/en/blog/sqlite-basics-cli-transactions-pragmas)
:::

## Why does it lock so easily?

SQLite’s concurrency model isn’t complicated—but it’s very real:

- **Only one writer at a time**
- reads are usually fast
- once you write, you enter the “world of locks”

In multi-worker background systems, the issue is usually not “we write a lot”, but:

> **a lot of people trying to write a little, at the same time.**

For example:

- claim a job (state transition)
- heartbeat (update a timestamp)
- write results or error codes

Each is small, but once you have enough workers, those small writes end up competing for the same write lock at the same moment.

And then, as you’ve seen: locked again.

## First, distinguish two errors

Is it `BUSY` or `LOCKED`?

In practice, the `database is locked` you see usually corresponds to one of two situations:

### `SQLITE_BUSY`

- another connection is currently writing
- you can’t get the lock right now
- **you can wait**

That’s exactly where `busy_timeout` helps.

### `SQLITE_LOCKED`

- within the same connection, a statement or transaction hasn’t finished yet
- or you’re re-entering a connection at the wrong time
- **waiting won’t help**

If you’ve already set `busy_timeout` but it still fails “instantly”, you’re usually hitting the latter.

At that point, you should fix how the connection is used—not just crank the timeout higher.

## Write-Ahead Logging

WAL (Write-Ahead Logging) is basically standard equipment for multi-worker SQLite.

The change is very straightforward:

- **readers are less likely to block writers**
- read-heavy, occasional-write workloads become much smoother

The reason is simple:
the writer records changes into the `-wal` file, and readers read their own snapshot, so interference drops dramatically.

But WAL does **not** change this:

> **at any given moment, there can still be only one writer.**

So a common pattern looks like:

- single worker: no problems at all
- multiple workers: occasional lock contention

That’s what it looks like when concurrency pressure starts to show up: not fully solved, but at least improved.

Either way, you can confirm WAL is really active with:

```sql
PRAGMA journal_mode;
```

It only counts if it returns `wal`.

---

## busy_timeout

Sometimes, “wait a bit” is acceptable; “fail immediately” is not.

When SQLite hits lock contention, the default behavior is very direct:

> can’t get the lock → return an error

`busy_timeout` changes the strategy to:

> can’t get the lock → wait and retry for a short while → fail only if it still can’t

In multi-worker systems, that difference matters a lot.

Here’s a simple Python example:

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

In Python’s `sqlite3` module, you can also do:

```python
sqlite3.connect(path, timeout=5.0)
```

It’s the same thing under the hood—pick one.

## The real key

Even after improving things, it may still happen from time to time.

The real key to eliminating the problem is:

> **keep transactions short.**

WAL and `busy_timeout` reduce contention, but they **can’t save you from bad habits**.

The most common anti-pattern looks like this:

```
BEGIN;
-- claim a job
-- run a long processing step (I/O / compute / external call)
-- write the result
COMMIT;
```

Which is basically telling every other worker:

> I’m holding the write lock for a while. Please queue up.

Don’t do this.

The correct way is to split the flow:

1. **claim (reserve it)**: only flip the state, commit immediately
2. **write results**: when you actually need to persist, open another short transaction

A typical claim implementation looks like:

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

- `BEGIN IMMEDIATE`: first confirm “can I write?”; if not, back off early
- `AND status = 'QUEUED'`: the simplest—but very effective—CAS
- `UPDATE` affected rows ≠ 1 → claim failed; retry

The point is: **minimize the time you hold the lock**.

## Common pitfalls

1. **Assuming WAL solves everything**

   It doesn’t.

   WAL solves “reader/writer blocking”, not “writer/writer contention”.

   **Fix**:
   WAL + `busy_timeout` + short transactions. You need all three.

   ***

2. **No index for the claim condition**

   If you start scanning the whole table only after `BEGIN IMMEDIATE`, other workers can only wait.

   **Fix**:
   add an index for the claim predicates:

   ```sql
   CREATE INDEX IF NOT EXISTS idx_jobs_queue_status_created_at
   ON jobs(queue, status, created_at);
   ```

   ***

3. **Sharing one connection across multiple threads**

   This usually earns you errors, not performance.

   **Fix**:
   one connection per thread/process, or explicitly serialize access to the connection.

   ***

4. **Putting SQLite on an unreliable shared filesystem**

   Some NFS setups or network drives make locking behavior unpredictable.

   **Fix**:
   keep the DB on local disk. If you can’t, rethink the tech choice.

   ***

Other directions worth trying:

- split queue state and result writes (separate tables or separate DBs)
- batch result writes to reduce the number of transactions
- if you truly need multiple writers, switch to a client/server DB

SQLite is great, but it won’t carry all your concurrency pressure for you.

## Summary

WAL and `busy_timeout` aren’t magic spells that make SQLite safe to “write however you want”.

They simply give your system more breathing room under real-world contention.

Just remember three things:

- WAL reduces reader/writer blocking; it doesn’t solve writer/writer concurrency
- `busy_timeout` turns contention into waiting, but only if transactions are short
- keep lock-taking SQL minimal, and set your PRAGMAs when initializing the connection

Then `database is locked` usually goes from “wall of red” to “an occasional retry”.

## References

- [SQLite docs: Write-Ahead Logging](https://www.sqlite.org/wal.html)
- [PRAGMA busy_timeout behavior](https://www.sqlite.org/pragma.html#pragma_busy_timeout)
- [SQLite locking model and concurrency](https://www.sqlite.org/lockingv3.html)
- [Transactions: semantics and limitations](https://www.sqlite.org/lang_transaction.html)
