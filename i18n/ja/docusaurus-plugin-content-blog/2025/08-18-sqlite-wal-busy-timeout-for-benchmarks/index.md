---
slug: sqlite-wal-busy-timeout-for-workers
title: SQLite 実装（1）：またデータベースがロックした！
authors: Z. Yuan
tags: [sqlite, wal, busy-timeout, concurrency, job-queue]
image: /ja/img/2025/0818.jpg
description: WAL と busy_timeout を使って、複数 worker 環境のロック競合を許容できる範囲まで下げる。
---

SQLite で社内システムを作れば楽勝だと思っていた。

ところが worker を 2 つ増やしただけで、画面に時々こんな文字が出始める：

```
database is locked
```

さらに厄介なのは：
「高い並行性が必要な操作」をしているわけではなく、ただの普通のバックグラウンド処理だということ。

- job を拾う
- heartbeat を更新する
- 結果を書き戻す

どの書き込みも小さくて単純。

それでもロックする？

SQLite って本当に使えるの？

<!-- truncate -->

:::info
SQLite にまだ慣れていないなら、先に次の記事を読むのがおすすめ：

- [**SQLite 入門（1）：なぜまた君なの？**](/ja/blog/sqlite-intro-embedded-database)
- [**SQLite 入門（2）：CLI、インデックス、PRAGMA**](/ja/blog/sqlite-basics-cli-transactions-pragmas)
:::

## なぜこんなに「ロック」しやすいのか？

SQLite の並行モデルは複雑ではありませんが、現実はシビアです：

- **同時に writer は 1 つだけ**
- 読み取りはだいたい速い
- 書き込みが入った瞬間に「ロックの世界」に入る

複数 worker のバックグラウンドシステムで問題になるのは、「大量に書く」ことではなく：

> **みんなが同時に、少しだけ書きたい。**

例えば：

- job の claim（状態変更）
- heartbeat（タイムスタンプ更新）
- 結果やエラーコードの書き込み

どれも小さいですが、worker が増えると、この小さな書き込みが同じタイミングで同じ write lock を奪い合います。

そして、見ての通り：またロック。

## まず 2 種類のエラーを分ける

`BUSY` か `LOCKED` か？

実務で見る `database is locked` の裏側は、だいたい 2 パターンです：

### `SQLITE_BUSY`

- 別の connection が書き込み中
- 今はロックが取れない
- **待てば取れる可能性がある**

ここで `busy_timeout` が効きます。

### `SQLITE_LOCKED`

- 同一 connection 内で statement / transaction がまだ終わっていない
- あるいは、良くないタイミングで connection を再入している
- **待っても無駄**

`busy_timeout` を設定しているのに「秒で落ちる」なら、だいたいこちらです。

この場合、直すべきは timeout ではなく、connection の使い方です。

## Write-Ahead Logging

WAL（Write-Ahead Logging）は、複数 worker の SQLite ではほぼ標準装備です。

変わる点は明確：

- **reader が writer を止めにくくなる**
- 読み取りが多く、たまに書くケースがかなりスムーズになる

理由はシンプルです：
writer は変更を `-wal` ファイルに書き込み、reader は自分の snapshot を読むため、干渉が大きく減ります。

ただし WAL でも **変わらない** ことがあります：

> **同時に writer は結局 1 つだけ。**

だからよくある症状は：

- worker 1 つ：問題なし
- worker 複数：たまにロック競合が起きる

並行圧力が出始めた自然な結果で、完全に解決ではないですが、少なくとも改善はします。

WAL が有効かどうかは、次で確認できます：

```sql
PRAGMA journal_mode;
```

`wal` が返ってきて初めて有効です。

---

## busy_timeout

「少し待つ」は許容できても、「即エラー」は困ることがあります。

SQLite がロック競合に当たったときのデフォルト挙動はとても直球です：

> ロックが取れない → すぐエラー

`busy_timeout` はこれを次に変えます：

> ロックが取れない → 少し待ってリトライ → それでも無理ならエラー

複数 worker のシステムでは、この差が重要です。

Python の簡単な例：

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

Python の `sqlite3` モジュールでは、次のように書くこともできます：

```python
sqlite3.connect(path, timeout=5.0)
```

本質的には同じなので、どちらか一方で OK です。

## 本当のポイント

ここまでやっても、まだ不定期に起きることがあります。

根本的に潰すポイントは：

> **トランザクションを短くする！**

WAL と `busy_timeout` は衝突を緩和しますが、**悪い習慣までは救えません**。

よくあるアンチパターンはこんな感じ：

```
BEGIN;
-- job を拾う
-- 長い処理（I/O / 計算 / 外部呼び出し）
-- 結果を書く
COMMIT;
```

これは他の worker にこう言っているのと同じです：

> しばらく write lock を握るので、みんな並んでね。

良くないので、やめましょう。

正しい考え方は、流れを分けることです：

1. **claim（占有）**：状態変更だけして即コミット
2. **結果の書き込み**：必要になったときに、短い transaction をもう一回

claim の典型的な書き方はこんな感じ：

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

- `BEGIN IMMEDIATE`：最初に「書けるか」を確認し、ダメなら早めに引く
- `AND status = 'QUEUED'`：最もシンプルで効果的な CAS
- `UPDATE` の更新行数 ≠ 1 → claim 失敗なのでリトライ

ここで重要なのは：**ロックを握る時間を短くする**こと。

## よくある落とし穴

1. **WAL を有効にすれば全部 OK と思う**

   そうはなりません。

   WAL が解決するのは「読書きが互いにブロックする」問題であって、「書き込み同士の競合」ではありません。

   **対策**：
   WAL + `busy_timeout` + 短い transaction。3 つ全部必要。

   ***

2. **claim 条件にインデックスがない**

   `BEGIN IMMEDIATE` の後で全表スキャンを始めると、他の worker はただ待つしかありません。

   **対策**：
   claim に使う条件へインデックスを貼る：

   ```sql
   CREATE INDEX IF NOT EXISTS idx_jobs_queue_status_created_at
   ON jobs(queue, status, created_at);
   ```

   ***

3. **複数スレッドで同じ connection を共有する**

   だいたい得られるのはエラーだけで、性能は上がりません。

   **対策**：
   thread / process ごとに connection を 1 本、または明示的にアクセスを直列化する。

   ***

4. **信頼できない共有ファイルシステムに SQLite を置く**

   一部の NFS やネットワークドライブでは、ロック挙動が予測不能になります。

   **対策**：
   DB をローカルディスクに置く。できないなら技術選定を見直す。

   ***

他に試せる方向：

- queue の状態と結果書き込みを分離（別 table / 別 DB）
- 結果書き込みをバッチ化して transaction 回数を減らす
- 本当に複数 writer が必要なら、client/server DB を使う

SQLite は便利ですが、すべての並行圧力を肩代わりしてくれるわけではありません。

## まとめ

WAL と `busy_timeout` は、SQLite を「雑に書いても大丈夫」にする魔法ではありません。

現実的な条件下で、**衝突に対する余裕**を少し増やすだけです。

覚えておくことは 3 つ：

- WAL は読書きのブロッキングを減らすが、書き込み同士の並行は解決しない
- `busy_timeout` は衝突を「待ち」に変えるが、前提として transaction が短い必要がある
- write lock を取る SQL を最小化し、connection 初期化で PRAGMA を設定する

そうすれば `database is locked` は、「真っ赤な画面」から「たまにリトライするだけ」になるはずです。

## 参考資料

- [SQLite 公式ドキュメント：Write-Ahead Logging](https://www.sqlite.org/wal.html)
- [PRAGMA busy_timeout の挙動](https://www.sqlite.org/pragma.html#pragma_busy_timeout)
- [SQLite のロックモデルと並行性](https://www.sqlite.org/lockingv3.html)
- [Transaction の意味と制約](https://www.sqlite.org/lang_transaction.html)
