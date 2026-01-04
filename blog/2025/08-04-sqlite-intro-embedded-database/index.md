---
slug: sqlite-intro-embedded-database
title: SQLite 初探（一）：為什麼又是你？
authors: Z. Yuan
tags: [sqlite, database, storage]
image: /img/2025/0804.jpg
description: 一個不用開服務的輕巧資料庫。
---

在系統設計的早期，常常會遇到這樣的取捨：

需求只是結構化資料儲存，但引入一個完整的資料庫服務，卻意味著連線管理、部署、監控、備援，以及長期維運成本。當這筆帳怎麼算都不划算時，專案裡通常會出現一個檔案：

```
something.db
```

這往往代表：SQLite 又被選上了。

<!-- truncate -->

## SQLite 是什麼？

SQLite 是一個**嵌入式（embedded）資料庫引擎**。

如果把這句話翻譯成「工程師能判斷影響」的版本，那應該是這樣：

- 它**不是一個服務**（不像 MySQL / Postgres 那樣需要常駐）
- 它是一個 **library**，會被你的程式直接 link 進來
- 資料**不經過 socket 或 TCP**，而是由程式直接讀寫檔案
- 多數情境下，**一個檔案就是一個完整的資料庫**
- 你甚至可以用 `:memory:`，得到一個只存在於 process 生命週期內的 DB

所以 SQLite 的本質其實非常單純：

> **它把資料庫引擎直接塞進你的程式裡。**

這個定位一旦確立，它的優點與限制其實也就同時被決定了。

## 為什麼你會一直遇到 SQLite？

因為它精準地解決了一個「既要...又要...」的問題：

> **我既要結構化資料，又要省時省事。**

維護一整套資料庫系統是一件麻煩事，能省則省。

用了 SQLite 你就不需要：

- 開 port
- 顧 daemon
- 規劃 backup / replica / failover
- 擔心「資料庫沒起來，整個程式就不能跑」

這也是為什麼 SQLite 會反覆出現在這些地方：

- **本機開發與測試**：啟動快，資料用完就丟
- **桌面 / 行動裝置 App**：內建、可靠、不依賴外部服務
- **內部工具、原型系統、小型後台**
- **邊緣或單機部署**：沒網路，或不能依賴遠端 DB
- **讀多寫少的資料層**：快取、索引、任務狀態、metadata

如果你曾經冒出過這個念頭：

> 「這個東西，其實不值得為它開一個 Postgres。」

那 SQLite 幾乎一定就在你旁邊。

## SQLite 的基本概念

你不需要把官方文件從頭讀到尾，但下面幾個名詞，你至少要知道它們在控制什麼行為。

### 1. Connection

SQLite 的「連線」不是網路連線，而是**對資料庫檔案的存取上下文**。

實務原則只有一句話：

> **不要在多個 thread / process 之間共用同一條 connection。**

每個 worker、每個 thread，各自開一條是比較安全的用法。

### 2. Transaction

你以為你只是在跑兩行 SQL，但對 SQLite 來說，那是在決定：

- 這些操作要不要一起成功
- 中途失敗時要不要全部回滾
- 什麼時候鎖會被拿走、什麼時候釋放

**沒有 transaction 的 SQLite，效能跟一致性都會出問題。**

### 3. Journal / WAL

這是 SQLite 能不能撐住併發的關鍵。

- 預設是 rollback journal（保守、簡單）
- WAL（Write-Ahead Logging）可以讓：
  - 多個 reader 同時讀
  - writer 不那麼容易把整個 DB 卡死

你之後只要遇到 `database is locked`，幾乎一定會回來看這一塊。

### 4. Type affinity

SQLite **不是強型別資料庫**。

它不會像 Postgres 一樣，硬性阻止你把字串塞進 integer 欄位。

它只會說：

> 「我建議你是這個型別，但我不會替你負責。」

自由度很高，責任完全在你身上。

### 5. Constraint

`PRIMARY KEY`、`UNIQUE`、`CHECK`、`FOREIGN KEY`
這些不是裝飾，是**資料層最後一道防線**。

SQLite 不會替你補，沒寫就真的沒有。

:::tip
SQLite 讓你很快寫出「可以跑的系統」，但**資料正確性要靠你自己設計**。
:::

## 跑跑看

以下示範假設你使用的是 macOS / Linux，或已有 SQLite 與 Python 環境；

若 sqlite3 指令不存在，代表系統尚未安裝 SQLite CLI（但 Python 仍可直接使用）。

如果你是使用 ubuntu 可以這樣裝：

```bash
sudo apt update
sudo apt install sqlite3
```

或是 MacOC，就換成這樣：

```bash
brew install sqlite
```

### 1. 用 CLI 建立資料庫

```bash
sqlite3 demo.db
```

建表：

```sql
CREATE TABLE IF NOT EXISTS notes (
  id INTEGER PRIMARY KEY,
  title TEXT NOT NULL,
  body TEXT,
  created_at TEXT NOT NULL
);
```

插入資料：

```sql
INSERT INTO notes (title, body, created_at)
VALUES ('hello', 'sqlite is a file', '2025-08-04T12:00:00Z');
```

查詢：

```sql
SELECT id, title, created_at
FROM notes
ORDER BY created_at DESC
LIMIT 5;
```

### 2. 用 Python 寫入

```python title="python sqlite3（示意）"
import sqlite3

conn = sqlite3.connect("demo.db")

# 每條連線都要自己開
conn.execute("PRAGMA foreign_keys = ON;")

conn.execute(
    "INSERT INTO notes(title, body, created_at) VALUES (?, ?, ?)",
    ("hello", "sqlite is a file", "2025-08-04T12:00:00Z"),
)
conn.commit()
```

**唯一重點**：永遠用 `?` 做參數綁定，不要自己拼 SQL 字串。

## 常見問題

- **為什麼外鍵「看起來沒作用」？**

  這是 SQLite 新手最常遇到的錯覺之一。

  你明明：

  - 在 schema 裡寫了 `FOREIGN KEY`
  - 設了 `ON DELETE CASCADE`
  - 但刪主表資料時，子表卻完全沒反應

  原因只有一個：

  > **SQLite 的 foreign key constraint 預設是關的，而且是「每條 connection 各自關」。**

- **正確作法**

  只要你有用外鍵，**每次建立連線後第一件事就是：**

  ```sql
  PRAGMA foreign_keys = ON;
  ```

## 什麼時候「不該」用 SQLite？

SQLite 很好用，但它不是萬用。

你應該考慮換成 client/server DB 的情境包括：

- **高寫入併發**（多個 writer 同時狂寫）
- **跨機器共享同一份資料**
- **需要複雜權限、審計、HA、replication**
- 你已經開始自己補「資料庫該有的功能」

這時候就不要自找麻煩，你該換一個資料庫了。

## 參考資料

- [SQLite 官方網站](https://www.sqlite.org/index.html)
- [SQLite: In-Memory Databases](https://www.sqlite.org/inmemorydb.html)
- [SQLite: Foreign Key Support](https://www.sqlite.org/foreignkeys.html)
