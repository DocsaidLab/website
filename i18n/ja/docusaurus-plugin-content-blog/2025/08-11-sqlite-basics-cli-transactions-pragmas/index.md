---
slug: sqlite-basics-cli-transactions-pragmas
title: SQLite 入門（2）：CLI、インデックス、PRAGMA
authors: Z. Yuan
tags: [sqlite, sql, cli, transactions, pragmas, index]
image: /ja/img/2025/0811.jpg
description: SQLite を言うことを聞かせる。schema から query plan まで。
---

プロジェクトで `.db` ファイルを見かけたとします。

普通のファイルみたいにすぐ開けないし、開けたところでぱっと見では読めない。

となると、めちゃくちゃ気になるわけです。「この中に何が入ってるの？」

> **え、興味ない？こんなの誰が気になるんだろうね？**

<!-- truncate -->

## 1. sqlite3 CLI のよく使うコマンド

まずは基本コマンドからいきましょう。

CLI に入る：

```bash
sqlite3 app.db
```

この辺は人生の半分を救ってくれます：

```text
.tables          -- テーブル一覧
.schema          -- 全体の schema
.schema jobs     -- 特定テーブルの schema
.headers on      -- カラム名を表示
.mode column     -- カラム揃え表示
.timer on        -- 各 SQL の実行時間を表示
.quit            -- 終了
```

SQL をサクッと検証したいなら、こういうのもあります：

```text
.read init.sql
```

SQL ファイルをそのまま実行します。

## 2. トランザクション（Transaction）

SQLite の読み取りはだいたい並行でも問題になりませんが、「書き込みトランザクション」にはかなり慎重です：

- 同時に writer は 1 つだけ（書き込みロックを保持できるのは 1 つ）

なので「状態更新」「引き落とし/送金」「job の claim」みたいなことをするなら、まずトランザクションを頭の一番上に置いてください。

最小のトランザクションはこうです：

```sql
BEGIN; -- デフォルトは DEFERRED
-- 複数の更新
COMMIT;
```

`BEGIN`（DEFERRED）は、最初は書き込みロックを取りに行かず、最初に書き込みが必要なステートメントでロックを取りに行きます。

途中までやってから「書けない！」を踏みたくないなら、こういう書き方もできます（例）：

```sql
BEGIN IMMEDIATE;
-- 原子性が必要な更新
COMMIT;
```

:::tip
`IMMEDIATE` の感覚は、「書くつもり」を先に宣言して、ロックが取れてから始める、です。
:::

### 例：CAS で job を claim する

例えば、`QUEUED` の job を `RUNNING` に変えたいとします。いちばん安全なのは、「旧状態の照合」を同じ SQL に入れることです：

```sql
UPDATE jobs
SET status = 'RUNNING'
WHERE id = :id
  AND status = 'QUEUED';
```

あとはアプリ側で、影響行数を確認します：

- 1：よし、取れた！
- 0：だめだ、誰かに取られた（ここで無理に走らない）

## 3. PRAGMA

`PRAGMA` は SQLite の挙動を調整するための「設定コマンド」です。多くは、その接続にだけ効きます。

PRAGMA は山ほどありますが、まずはよく使うものを覚えましょう：

```sql
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 30000;
```

ざっくり言うと：

- `foreign_keys`：データ整合性の最低防衛線（でも自分で ON にする必要がある）
- `journal_mode`：並行読み書きの体感がかなり変わる（まず WAL を試すのが定番）
- `synchronous`：安全 vs 速さ（だいたいトレードオフ）
- `busy_timeout`：ロックに当たったら少し待って、即死しない

接続を頻繁に作り直すなら、PRAGMA は「接続初期化の一部」として扱うのが良いです。

:::warning
PRAGMA の中には「接続単位」のものがあります。接続を作り直したら、もう一度設定しないといけません。1 回設定したら永遠に効くと思わないこと。
:::

## 4. EXPLAIN QUERY PLAN

> なんでこんなに遅いの？またこっそり全表スキャンしてない？

遅いクエリって、見た目は無罪なことが多いです：

- `LIMIT 100` も付いてる
- 条件もそんなに複雑に見えない

でも適切なインデックスが無いと、`LIMIT` は思ったほど効きません。

なぜなら、ソートする前の段階では「欲しい 100 件」がどれか分からないので、SQLite は結局こうするしかないからです：

1. たくさん読む
2. ソートする
3. その中から先頭 100 件を取る

`EXPLAIN QUERY PLAN` で実行計画を聞けます（例）：

```sql
EXPLAIN QUERY PLAN
SELECT id
FROM jobs
WHERE status = 'QUEUED'
ORDER BY created_at ASC
LIMIT 1;
```

もし `SCAN TABLE jobs` みたいなのが出てきたら、それは本当に全表スキャンです。

## よくある質問

`INSERT OR REPLACE` を見ると、ついこう思いがちです：

> **「あ、存在すれば更新、なければ挿入ね。」**

でも `REPLACE` はむしろこうです：

> **「古い行を削除してから、新しい行を挿入する。」**

外部キーがある場合や、特定のカラム（例えば `created_at`）を保持したい場合に、特に踏みやすい罠です。

**対策**：

- `ON CONFLICT DO UPDATE` を優先する（SQLite は UPSERT をサポート）
- もしくは `INSERT` / `UPDATE` に明確に分ける（アプリ側で制御）

## まとめ

SQLite は最初から正直です：

> **データベースエンジンはアプリの中にあり、その責任はあなたが負う。**

この投稿で扱った内容は、ぱっと見だとバラバラです：

- CLI コマンド
- トランザクションのモード
- PRAGMA
- インデックスと query plan

でも全部、同じ問いに答えています：

> **データベースが「全部面倒見てくれる」存在でなくなったとき、最低限なにを自分で守るべきか？**

これだけ押さえておけば、SQLite はだいたい大丈夫です：

- 書き込みは常にトランザクションを意識する
- PRAGMA は接続初期化の一部にする
- インデックスが無いなら、遅さを嘆かない
- `EXPLAIN QUERY PLAN` で何をしているか確認する

そのうち、複数の writer が長時間同時に書き続ける必要が出たり、権限/ロールが複雑になったり、中央集権的なバックアップ、replica、観測などが必要になったりするでしょう。そうなったとき、それは SQLite が悪いわけではなく、設計上の境界を超え始めているサインです。

それまでは、この基本をしっかり押さえておけば、SQLite はとても信頼できる、低摩擦で低運用コストなツールになります。

`.db` ファイルが 1 つだけ？

それで十分。

## 参考資料

- [SQLite: Command Line Shell](https://www.sqlite.org/cli.html)
- [SQLite: PRAGMA](https://www.sqlite.org/pragma.html)
- [SQLite: UPSERT](https://www.sqlite.org/lang_UPSERT.html)
- [SQLite: EXPLAIN QUERY PLAN](https://www.sqlite.org/eqp.html)
