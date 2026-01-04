---
slug: sqlite-intro-embedded-database
title: "A First Look at SQLite (1): Why Is It Always You?"
authors: Z. Yuan
tags: [sqlite, database, storage]
image: /en/img/2025/0804.jpg
description: A lightweight database that runs without a service.
---

Early in a system design, you often face a trade-off like this:

You only need structured data storage, but bringing in a full database service implies connection management, deployment, monitoring, backups, failover, and long-term operational cost. When the math simply doesn’t work out, a file tends to appear in the repo:

```
something.db
```

And that usually means: SQLite got picked again.

<!-- truncate -->

## What is SQLite?

SQLite is an **embedded database engine**.

If we translate that into something that affects engineering decisions, it means:

- It’s **not a service** (unlike MySQL / Postgres, which need a daemon)
- It’s a **library** that your program links against
- Data doesn’t go through a socket or TCP; your program reads/writes the file directly
- In most cases, **one file is a complete database**
- You can even use `:memory:` to get a DB that exists only for the lifetime of the process

So SQLite’s essence is pretty simple:

> **It ships the database engine inside your application.**

Once that positioning is clear, its strengths and limitations are effectively decided at the same time.

## Why do you keep running into SQLite?

Because it perfectly solves a very specific “I want both” problem:

> **I want structured data, and I want it to be easy.**

Running and maintaining a full database system is a hassle; if you can avoid it, you will.

With SQLite, you don’t need to:

- open a port
- keep a daemon running
- plan for backup / replicas / failover
- worry that “if the DB isn’t up, the whole program won’t run”

That’s why SQLite keeps showing up in places like:

- **local development and tests**: instant startup; throw the data away when done
- **desktop and mobile apps**: built-in, reliable, no dependency on external services
- **internal tools, prototypes, small admin backends**
- **edge or single-node deployments**: no network, or can’t rely on a remote DB
- **read-heavy, write-light data layers**: caches, indexes, job states, metadata

If you’ve ever thought:

> “This isn’t worth spinning up Postgres for.”

SQLite is probably right there next to you.

## Core concepts in SQLite

You don’t need to read the official docs cover-to-cover, but you should at least know what these terms control.

### 1. Connection

In SQLite, a “connection” is not a network connection; it’s an **access context to the database file**.

One practical rule:

> **Don’t share a single connection across multiple threads or processes.**

A safer approach is to open one per worker/thread.

### 2. Transaction

You might think you’re just running a couple lines of SQL, but in SQLite you’re deciding:

- whether these operations succeed together
- whether to rollback everything on failure
- when locks are acquired and when they’re released

**Without transactions, SQLite will suffer in both performance and consistency.**

### 3. Journal / WAL

This is the key to how SQLite behaves under concurrency.

- The default is rollback journal (conservative and simple)
- WAL (Write-Ahead Logging) can make it so:
  - multiple readers can read at the same time
  - writers are less likely to stall the entire DB

Any time you see `database is locked`, you’ll almost certainly come back to this part.

### 4. Type affinity

SQLite is **not a strongly-typed database**.

It won’t hard-stop you (like Postgres would) from putting a string into an integer column.

It basically says:

> “I suggest this type, but I’m not responsible.”

You get a lot of freedom, and all the responsibility.

### 5. Constraints

`PRIMARY KEY`, `UNIQUE`, `CHECK`, `FOREIGN KEY`
These aren’t decoration—they’re the **last line of defense in the data layer**.

SQLite won’t “fill in the gaps” for you; if you don’t define it, it simply isn’t there.

:::tip
SQLite lets you build something that “runs” quickly, but **you must design for data correctness yourself**.
:::

## Try it out

The examples below assume you’re on macOS / Linux, or you already have SQLite and Python set up.

If the `sqlite3` command is missing, it means the SQLite CLI isn’t installed (but Python can still use SQLite directly).

On Ubuntu, you can install it like this:

```bash
sudo apt update
sudo apt install sqlite3
```

On macOS:

```bash
brew install sqlite
```

### 1. Create a database with the CLI

```bash
sqlite3 demo.db
```

Create table:

```sql
CREATE TABLE IF NOT EXISTS notes (
  id INTEGER PRIMARY KEY,
  title TEXT NOT NULL,
  body TEXT,
  created_at TEXT NOT NULL
);
```

Insert data:

```sql
INSERT INTO notes (title, body, created_at)
VALUES ('hello', 'sqlite is a file', '2025-08-04T12:00:00Z');
```

Query:

```sql
SELECT id, title, created_at
FROM notes
ORDER BY created_at DESC
LIMIT 5;
```

### 2. Write with Python

```python title="python sqlite3 (example)"
import sqlite3

conn = sqlite3.connect("demo.db")

# Open a dedicated connection per worker
conn.execute("PRAGMA foreign_keys = ON;")

conn.execute(
    "INSERT INTO notes(title, body, created_at) VALUES (?, ?, ?)",
    ("hello", "sqlite is a file", "2025-08-04T12:00:00Z"),
)
conn.commit()
```

**One key rule**: always use `?` parameter binding; never build SQL strings by hand.

## Common questions

- **Why do foreign keys “seem to do nothing”?**

  This is one of the most common beginner misconceptions in SQLite.

  You clearly:

  - wrote `FOREIGN KEY` in your schema
  - set `ON DELETE CASCADE`
  - but when you delete from the parent table, the child table doesn’t react at all

  There’s only one reason:

  > **SQLite foreign key constraints are off by default—and it’s per connection.**

- **The correct way**

  If you use foreign keys, **the first thing after opening a connection should be:**

  ```sql
  PRAGMA foreign_keys = ON;
  ```

## When should you *not* use SQLite?

SQLite is great, but it isn’t a universal solution.

You should consider moving to a client/server database if you need:

- **high write concurrency** (multiple writers hammering writes at the same time)
- **sharing the same data across machines**
- **complex permissions, auditing, HA, replication**
- you’ve started implementing “database features” yourself

At that point, don’t make life harder—pick a proper database.

## References

- [SQLite Official Site](https://www.sqlite.org/index.html)
- [SQLite: In-Memory Databases](https://www.sqlite.org/inmemorydb.html)
- [SQLite: Foreign Key Support](https://www.sqlite.org/foreignkeys.html)
