---
slug: sqlite-lease-heartbeat-recovery
title: "SQLite in Practice (3): Save Your Workers"
authors: Z. Yuan
tags: [sqlite, job-queue, reliability, heartbeat]
image: /en/img/2025/1008.jpg
description: Automatically reclaim stuck tasks—don’t let your queue turn into a graveyard.
---

You’ve definitely seen this:

- The queue looks stuck
- The UI shows a pile of jobs still in `RUNNING`

But that worker has already been restarted three times. The jobs didn’t die — the worker did.

On what should have been a perfectly pleasant weekend, you end up singing a dirge for your worker.

<!-- truncate -->

## Why do workers die?

Because you live in an uncontrollable world.

In the ideal world, a worker claims a job, runs it, and then sets its status to `SUCCEEDED` or `FAILED`.

World peace.

But in the real world, the usual ways a worker dies look like this:

1. **Process crash**: unhandled exceptions, a third-party library segfault, model inference failures.
2. **Killed by the system**: OOM killer, container restarts, nodes reclaimed, quotas blown.
3. **Hung but not dead**: an external API waiting forever, I/O hangs, deadlocks, some `while True` spinning.
4. **Interrupted by deploy/upgrade**: rolling updates, manual kills, process replacement.
5. **Network/dependencies break**: S3, MQ, internal services timing out.

You’ll notice: **workers often don’t “fail gracefully.”**

They disappear abruptly, or hang forever. And what’s left behind is a single lonely `RUNNING`.

That lonely `RUNNING` is how a queue system starts to rot.

## Why does this matter?

If your queue system can’t answer these three questions, it’s not maintainable:

1. **Who owns this job right now?**
2. **When was the last heartbeat?**
3. **If they never come back, how do I clean this up?**

Notice that none of these are performance questions — they’re trust questions.

Because with limited human time, in practice:

- you can’t dig through logs/machines/pods every time things get stuck
- you can’t manually flip jobs back to `QUEUED` every time
- and you certainly can’t accept “some jobs just stay RUNNING forever; we’ll clean them up sometimes”

So you need a pragmatic design: **lease + heartbeat**.

The whole point is to turn “permanent ownership” into “borrowing for a while.”

## lease + heartbeat

The idea is simple:

- Claiming a job isn’t “owning it forever,” it’s “borrowing it for a fixed period.”
- After you borrow it, you renew it by reporting in periodically.
- If you stop renewing, you’re considered missing, and the system can reclaim the job.

In other words, you shift availability from “trust the worker” to “trust the data.”

And the database is, at minimum, more trustworthy than a worker that can die at any moment.

## Minimal model

You usually don’t need a fancy schema. You just need to be able to tell:

- Is this job still held by some worker?
- Is that hold still valid?
- What do we do once it expires?

A common minimal set of fields:

- `claimed_at`: when it was claimed
- `heartbeat_at`: last heartbeat
- `retry_count`: how many times it has been retried

But in practice, I strongly recommend adding two more — they make the system much more stable:

- `lease_expires_at`: when the lease expires (more explicit; cleaner queries)
- `lease_token`: the credential for this claim (prevents zombie workers)

:::info
**Why `lease_token`?**

This is the scenario you want to prevent:

- worker A claims a job → crashes halfway through
- the sweeper reclaims it → worker B claims it again
- worker A comes back and keeps updating heartbeat / marking it finished
- A kills B, and you enter the paranormal world of “one job, two writers”

The concept of `lease_token` is: **every claim gets a new key**.
All subsequent updates must present that key to count.
:::

## Example schema

Use **UTC epoch seconds** (`INTEGER`) for time. String timestamps will absolutely bite you someday.

```sql title="jobs schema (example)"
CREATE TABLE IF NOT EXISTS jobs (
  id               INTEGER PRIMARY KEY,
  status           TEXT NOT NULL
                   CHECK (status IN ('QUEUED','CLAIMED','RUNNING','SUCCEEDED','FAILED')),

  owner_id         TEXT,     -- which worker claimed it (hostname/uuid)
  lease_token      TEXT,     -- credential for this claim (prevents zombies)

  claimed_at       INTEGER,  -- unix epoch seconds (UTC)
  heartbeat_at     INTEGER,  -- unix epoch seconds (UTC)
  lease_expires_at INTEGER,  -- unix epoch seconds (UTC)

  retry_count      INTEGER NOT NULL DEFAULT 0,
  max_retry        INTEGER NOT NULL DEFAULT 5,

  finished_at      INTEGER,
  error_code       TEXT,
  error_detail     TEXT
);

-- common sweeper query path: status + expiry time
CREATE INDEX IF NOT EXISTS idx_jobs_lease
ON jobs(status, lease_expires_at);

-- queue pick path: QUEUED + id (or priority)
CREATE INDEX IF NOT EXISTS idx_jobs_queue
ON jobs(status, id);
```

## Heartbeat updates

A heartbeat essentially says: “I’m alive, and I still hold this job.”

So every heartbeat update must validate two things:

- who I am (`owner_id`)
- whether I’m holding the key for this lease (`lease_token`)

```sql title="heartbeat (example)"
UPDATE jobs
SET heartbeat_at = :now,
    lease_expires_at = :now + :lease_seconds
WHERE id = :job_id
  AND status IN ('CLAIMED', 'RUNNING')
  AND owner_id = :owner_id
  AND lease_token = :lease_token;
```

This is a very small write. It’s not about recording lots of information — it’s about maintaining trust in the system.

## Reclaim expired jobs

When a worker goes missing, the database only sees one thing:

- `lease_expires_at < now`

From there, your job is to pull these jobs back from `RUNNING/CLAIMED` into reality.

Usually there are two categories:

1. Still recoverable → put it back to `QUEUED` so someone else can rerun it
2. Not recoverable → mark it `FAILED` so it has an outcome

## Common pitfalls

1. **Random time formats**

   If you store timestamps like `2025/10/8 9:3:1` and compare them as strings, the results will be… entertaining.

   **Fix**: use UTC epoch seconds (`INTEGER`).
   To get “now” in SQLite:

   - `unixepoch('now')` (seconds)
   - or `strftime('%s','now')`

2. **An overzealous sweeper that makes the DB fight**

   If your sweeper runs every second and updates a ton of rows, while workers are heartbeating at the same time, you’ll start seeing:

   - lock contention
   - `busy_timeout` getting maxed out

   You wanted rescue, and things got worse.

   **Fix**:

   - run the sweeper every 10–30 seconds
   - batch each run (e.g. at most 100 jobs)
   - index properly: `(status, lease_expires_at)`

3. **lease/heartbeat handles “missing,” not “alive but never finishing”**

   If a worker gets stuck in an infinite loop but keeps updating heartbeat, lease won’t save you.

   What you need is a **job-level timeout** (max runtime), e.g. `max_runtime_seconds`.
   Besides `lease_expires_at`, your sweeper should also check how long the job has been running.

## Summary

To make a polling queue maintainable, you need at least two concepts:

1. **lease**: claiming a job isn’t permanent ownership; it’s a temporary borrow. When it expires, it gets reclaimed.
2. **heartbeat**: if a worker is alive, it renews. If it stops renewing, it’s considered missing.

Then add three pragmatic pieces:

- `max_retry / retry_count`: avoid infinite retries that burn your machines
- `error_code / error_detail`: make failures understandable and debuggable
- `lease_token`: prevent zombie workers; avoid two writers producing one result

Once you do this, your queue goes from “seems to run” to “recovers itself when things go wrong.”

Saving your workers is really saving yourself from debug hell.

It’s hard to lose.

## References

- [SQLite: Date And Time Functions](https://www.sqlite.org/lang_datefunc.html)
- [The “Leases” Pattern](https://martinfowler.com/articles/patterns-of-distributed-systems/lease.html)
