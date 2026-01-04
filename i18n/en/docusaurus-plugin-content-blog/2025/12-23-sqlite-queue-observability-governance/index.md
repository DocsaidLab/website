---
slug: sqlite-queue-observability-governance
title: "SQLite in Practice (5): Who Am I? Where Am I?"
authors: Z. Yuan
tags: [sqlite, observability, data-governance, job-queue]
image: /en/img/2025/1223.jpg
description: From error codes and retention policies to audit logs—make your queue operable.
---

You’ve probably had a conversation like this:

- You: How’s the system doing right now?
- System: I’m running.
- You: Running where?
- System: I’m running.

Cool. Thanks. That helps exactly zero.

<!-- truncate -->

## Queue

Sure, you can build a queue with SQLite—but at minimum you should be able to answer five engineering questions:

1. **How many jobs are in the queue right now? How many in each state?**
2. **How long has the oldest job been stuck? And who is it stuck on (which worker)?**
3. **What does the retry distribution look like? Is one type failing over and over?**
4. **What are the failure reasons? Can you aggregate them (Top-N)?**
5. **How long should you retain data? How do you delete it without creating orphaned rows or blowing up the DB?**

Behind these five questions are really just two keywords:

- **Observability**: can you understand the system’s current state **with data**, not with prayer?
- **Governance**: can you retain and clean up data **in a controlled way**, and still stay consistent and auditable afterward?

Below is a practical design that makes both of these real.

## Don’t Treat a Queue as One Table

Most people’s first instinct for a queue is:

- one `jobs` table
- a `status` column
- add whatever columns you need later

Sure, it runs—but it quickly turns into “the schema can’t answer the questions you want to ask.”

### What You Actually Need: Three Layers of Data

Split the queue into three layers and operations suddenly get much easier:

1. **Job entity (Current State)**
   You need to quickly answer: “What state is this job in right now?”

2. **Attempt records (Attempts / Retries)**
   A job may run many times. You need to answer: “Which attempt failed? Did it fail for the same reason each time? How long did each attempt take?”

3. **Event records (Audit Log / Event Log)**
   State is an outcome; events are the process. Systems that are actually debuggable rely on event streams, not on “staring at one row.”

## Which “Observable” Fields Do You Need?

“Observable” doesn’t mean turning your schema into an encyclopedia—it means being able to answer those five questions.

### Recommended fields for `jobs`

- `status`: the state machine (Queued / Claimed / Running / Succeeded / Failed / Cancelled …)
- `created_at / started_at / finished_at`: timeline (so you can compute wait time and processing time)
- `priority` (optional): don’t underestimate it—your scheduler and on-call self will thank you
- `worker_id` (or `claimed_by`): who took it (for finding the problematic worker)
- `heartbeat_at`: heartbeat timestamp (to tell whether the worker is still alive)
- `retry_count`: retries so far (for fast aggregation)
- `error_code`: an aggregatable error category (Top-N depends on it)
- `error_detail`: a short summary (for quick diagnosis—not for storing the whole log)

### Suggested schema

This is just a suggestion—we’re trying to separate operational concerns cleanly.

1. **jobs: current state (fast for serving)**

   ```sql
   CREATE TABLE jobs (
     id            INTEGER PRIMARY KEY,
     type          TEXT NOT NULL,                 -- job type (for grouped stats)
     status        TEXT NOT NULL,                 -- state machine
     priority      INTEGER NOT NULL DEFAULT 0,

     created_at    INTEGER NOT NULL,              -- Unix epoch seconds
     started_at    INTEGER,
     finished_at   INTEGER,

     claimed_by    TEXT,                          -- worker_id
     claim_token   TEXT,                          -- prevent accidental updates (optional)
     heartbeat_at  INTEGER,

     retry_count   INTEGER NOT NULL DEFAULT 0,
     max_retries   INTEGER NOT NULL DEFAULT 3,

     error_code    TEXT,                          -- aggregatable category
     error_detail  TEXT,                          -- short summary (consider a length limit)

     payload_ref   TEXT,                          -- store large payload elsewhere (file/object storage)
     result_ref    TEXT                           -- store large results elsewhere
   );
   ```

2. **job_attempts: each attempt (“which try was it?”)**

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

3. **job_events: event stream (debugging depends on it)**

   ```sql
   CREATE TABLE job_events (
     id          INTEGER PRIMARY KEY,
     job_id      INTEGER NOT NULL,
     ts          INTEGER NOT NULL,
     event       TEXT NOT NULL,                   -- CLAIMED/STARTED/HEARTBEAT/FAILED/RETRY_SCHEDULED...
     actor       TEXT,                            -- worker_id / system
     detail      TEXT,                            -- short JSON (don’t bloat it)

     FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
   );
   ```

   > Why have an event table?
   > Because “state” only tells you it’s FAILED, but “events” tell you:
   >
   > - when and by whom it was claimed
   > - how long it ran
   > - when the last heartbeat was
   > - which retry started failing
   >
   > Debugging shouldn’t be guessing. It should be reading history.

## Indexes

Observability isn’t finished just because you wrote a few queries:

- **At the very least, you need to make sure they can run fast, right?**

### Common indexes (recommended)

```sql
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_type_status ON jobs(type, status);

-- commonly used to find stuck jobs and the oldest running ones
CREATE INDEX idx_jobs_running_age
ON jobs(status, heartbeat_at, created_at);

-- failure Top-N aggregation
CREATE INDEX idx_jobs_failed_code
ON jobs(status, error_code);

-- primary query keys for attempts / events
CREATE INDEX idx_attempts_job ON job_attempts(job_id, attempt);
CREATE INDEX idx_events_job_ts ON job_events(job_id, ts);
```

The design principle is simple: **index whatever your dashboard and ops queries ask for most often.**

## Most Common Observability Queries

1. **Count by state: how busy are we right now?**

   ```sql
   SELECT status, COUNT(*) AS cnt
   FROM jobs
   GROUP BY status
   ORDER BY cnt DESC;
   ```

   You’ll quickly find patterns like:

   - `QUEUED` exploding → upstream sends too fast, downstream processes too slowly
   - lots of `RUNNING` but throughput doesn’t rise → workers are stuck or resources are constrained
   - `FAILED` slowly creeping up → a job type has a bug, or an external dependency is unhealthy

2. **Find the oldest in-progress jobs (who’s stuck the longest?)**

   ```sql
   SELECT id, type, claimed_by, created_at, heartbeat_at
   FROM jobs
   WHERE status IN ('CLAIMED', 'RUNNING')
   ORDER BY COALESCE(heartbeat_at, created_at) ASC
   LIMIT 20;
   ```

   The key is `COALESCE`:

   - if there’s a heartbeat, use it as the “latest proof of life”
   - if there isn’t, fall back to `created_at` (usually means it never truly started, or your design missed something)

3. **Retry distribution: is one class of jobs rerunning forever?**

   ```sql
   SELECT retry_count, COUNT(*) AS cnt
   FROM jobs
   WHERE status IN ('QUEUED', 'CLAIMED', 'RUNNING', 'FAILED')
   GROUP BY retry_count
   ORDER BY retry_count DESC;
   ```

   If you see a big pile at `retry_count = 3`, it usually means:

   - an external service is down (timeouts)
   - input format is wrong (invalid input)
   - a systemic bug (internal)

   The next step is error-code aggregation.

4. **Failure reasons Top-N: you must be able to aggregate**

   ```sql
   SELECT error_code, COUNT(*) AS cnt
   FROM jobs
   WHERE status = 'FAILED'
   GROUP BY error_code
   ORDER BY cnt DESC
   LIMIT 20;
   ```

   This only works if you have a decent `error_code`.

## Designing Error Codes

`error_code` should exist for **statistics**. The most common anti-pattern is:

- dumping an entire stack trace into `error_detail`
- not storing `error_code` at all (or storing it randomly)

In the end, you can’t aggregate anything—you can only read logs row by row.

### Suggested error-code structure

Use `CATEGORY:SUBCATEGORY` (or `CATEGORY/SUBCATEGORY`), for example:

- `TIMEOUT:UPSTREAM_API`
- `INVALID_INPUT:SCHEMA_MISMATCH`
- `RESOURCE:OUT_OF_MEMORY`
- `INTERNAL:ASSERTION_FAILED`
- `DEPENDENCY:DB_LOCKED`

### Guidelines for `error_detail`

- keep it a **short summary** (e.g. 200–500 characters)
- store full stack traces / full JSON / full reports in the filesystem or object storage; the DB should only keep a `path/URL`

We’ve seen too many “dumpster-fire log storages.” Don’t follow that path.

## Foreign Keys and Orphaned Rows

> If you don’t deal with it, it will deal with you later.

SQLite supports foreign keys—but they’re **not enabled by default**.

If you don’t turn them on, SQLite will pretend they don’t exist.

You need to do two things:

1. **Enable foreign keys on every connection**

```sql
PRAGMA foreign_keys = ON;
```

This is not a one-time setting. If you use a connection pool or open new connections, you must do this each time.

2. **Use `ON DELETE CASCADE` to keep consistency**

If you delete from `jobs`, `attempts/events` should be deleted automatically—otherwise you will create orphaned rows.

## Upsert

Stop using `INSERT OR REPLACE` as “update”!

This is worth writing in bold three times:

- **`REPLACE` is not update**
- **`REPLACE` is not update**
- **`REPLACE` is not update**

It’s closer to:

> delete first, then insert

If you have foreign keys, audit logs, attempts/events… `REPLACE` will kill you.

And after it kills you, you still won’t know where the bug is.

The right way is `ON CONFLICT DO UPDATE`. State updates should be predictable and traceable, and should never secretly delete rows:

```sql
INSERT INTO jobs(id, status, heartbeat_at)
VALUES (?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
  status = excluded.status,
  heartbeat_at = excluded.heartbeat_at;
```

## Audit Log (Event Table)

If you only look at the `jobs` row, debugging is like seeing “now” but not “how it became now.”

The event table philosophy is: **state is a snapshot; events are a timeline.**

At minimum, you should record:

- `ENQUEUED`: enqueued
- `CLAIMED`: claimed by a worker
- `STARTED`: actually started (can be separate from claimed)
- `HEARTBEAT`: heartbeat (lower frequency, or only keep “the latest”)
- `FAILED`: failed (with `error_code`)
- `RETRY_SCHEDULED`: retry scheduled (with delay, attempt)
- `SUCCEEDED`: succeeded
- `CANCELLED`: manually cancelled
- `RECOVERED`: reclaimed / reassigned (if you have a lease mechanism)

## Retention Policy

You need to delete data—but you need to delete it **with rules**.

Every queue system eventually faces this question:

> **How long do you keep it?**

If your answer is “keep everything,” SQLite will use disk space to have a philosophical conversation with you.
If your answer is “delete everything,” your next incident will leave you with no evidence.

A practical policy looks like:

- `SUCCEEDED`: keep 30 days (usually enough for investigations)
- `FAILED`: keep 90 days (failures deserve longer retention—they tend to come back)
- `CANCELLED`: depends (commonly 30 days)

For deletion, make it batched, interruptible, and rerunnable.

Don’t delete a million rows at once—especially while serving traffic. Run a scheduled cleanup every minute and delete slowly, like this:

```sql
-- delete succeeded jobs (older than 30 days)
DELETE FROM jobs
WHERE status = 'SUCCEEDED'
  AND finished_at < strftime('%s', 'now') - 30*86400
LIMIT 5000;
```

### Do you need `VACUUM`?

- SQLite file size doesn’t necessarily shrink immediately (deletes just mark pages as reusable)
- `VACUUM` rewrites the entire DB: heavy I/O and long runtime

Suggested approach:

- **use batched deletes for day-to-day cleanup**
- **only `VACUUM` during off-peak when you truly need to shrink the file**

If you use WAL mode, make sure you understand WAL files and checkpoint behavior too.

## Common Issues

1. **Unlimited `error_detail` turns the DB into a dumpster**

   **Fix**:

   - categorize with `error_code`
   - keep `error_detail` as a short summary
   - store full reports externally (DB keeps references)

2. Forgot to enable `foreign key`, leaving orphans after deleting the parent table

   **Fix**: `PRAGMA foreign_keys = ON` on every connection, and use `ON DELETE CASCADE`

3. Using `INSERT OR REPLACE` as upsert

   **Fix**: use `ON CONFLICT DO UPDATE`; don’t let `REPLACE` secretly do a DELETE

4. You wrote observability queries, but added no indexes

   **Fix**: derive indexes from the dashboard/ops queries you run most often (don’t force SQLite into full table scans)

## Summary

Once your queue can answer those five questions, it suddenly starts to feel like a “real system”:

- you can describe the current state (observability)
- you can trace the process (audit)
- you can control the data lifecycle (governance)
- you can turn incidents from “guess” into “query”

After that, using SQLite as a queue is the natural next step.

## References

- [**SQLite: Foreign Key Support**](https://www.sqlite.org/foreignkeys.html)
- [**SQLite: UPSERT**](https://www.sqlite.org/lang_UPSERT.html)
- [**SQLite: VACUUM**](https://www.sqlite.org/lang_vacuum.html)
