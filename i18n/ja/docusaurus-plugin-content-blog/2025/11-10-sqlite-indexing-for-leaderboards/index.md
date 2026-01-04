---
slug: sqlite-indexing-for-leaderboards
title: SQLite 実戦（4）：クエリが遅い？
authors: Z. Yuan
tags: [sqlite, index, performance, leaderboard, query-plan]
image: /ja/img/2025/1110.jpg
description: 正しいインデックスで、`ORDER BY ... LIMIT` が本当に上位 100 件で止まるようにする。
---

あなたは評価プラットフォームを作った。

ユーザーに少し達成感を与えるために、慈悲深く leaderboard まで用意した。

そして DevTools を開き、API の応答時間を見た。3 秒、5 秒、10 秒。

ん？

信じられなくて、SQL ももう一度見直す。

```sql
LIMIT 100
```

上位 100 件だけのはずなのに、なぜここまで遅い？

<!-- truncate -->

## 問題は `LIMIT` ではない

これはよくある誤解です。

`LIMIT 100` は、**SQLite が 100 件しか処理しない**という意味ではありません。

もし「上位 100」を探す前に、「条件に合う行」を全部集めてソートしないといけないなら……

あなたが言っているのは結局こうです：

> 「最後に 100 件だけ返してくれればいい。でもその前に必要なことは、全部やってね。」

クエリが遅いのはたいてい、SQLite に「いちばん愚直なやり方」しか残していないからです。

## 典型的なクエリ

多くの評価プラットフォームの schema は、だいたいこんな感じになります：

- `jobs`：1 回の submit / 1 回の run の基本情報
  （status、version、timestamp、queue）
- `job_scores`：実際の評価指標
  （train / public / private split に分かれることもある）

SQL の例：

```sql title="leaderboard（例）"
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

最後の `j.created_at ASC` が重要です。

これは **とても筋の良い設計**です：
同点なら時間を tie-breaker にして、ランキングが安定して跳ねないようにする。

でも SQLite から見ると、こういう要求になります：

> **いま欲しいのは「join あり・条件複数・複合ソート」の Top-N クエリです。**

インデックスがズレていると、力技でやるしかありません。

## インデックスが無いとき、SQLite はどう動く？

単純化すると、流れはだいたいこうです：

1. `status = 'SUCCEEDED'` かつ `queue = ?` の job を全部探す
2. `split = ?` の score を join する
3. 結果を全部引っ張り出す
4. `score1 → score2 → created_at` でソートする
5. ソートしたあと 99% を捨てて、上位 100 件だけ残す

遅いのは気のせいではありません。

SQLite は本当に「全部読み、ひと息ついて、並べ替える」ことをやっています。

## 本当に効くインデックスの貼り方

インデックスは「貼れば速い」わけではありません。

leaderboard のような Top-N クエリを速くするには、正しい順序が 1 つだけあります。逆にすると壊れます：

1. **まず `WHERE` で候補集合を一気に絞る**
2. **次に `ORDER BY ... LIMIT` がインデックス順に先頭 N 件を吐けるようにする**

ソートの段階で SQLite が「もう一回ソート」しているなら、`LIMIT 100` はほとんど救ってくれません。必要なものはもう全部ソートし終わっているからです。

例として `job_scores`（指標テーブル）から見ます。

クエリの並び順がこうなら：

```sql
ORDER BY s.score1 DESC, s.score2 DESC, ...
```

インデックスも同じ並びを先に敷きます：

```sql title="index（例）"
CREATE INDEX IF NOT EXISTS idx_scores_rank
ON job_scores (
  split,
  score1 DESC,
  score2 DESC
);
```

このインデックスが狙う効果：

- **`split = ?` で欲しいグループに絞る**
- 絞ったあと、行はインデックス内ですでに `score1 → score2` の順になっている
- SQLite はインデックスを順に辿るだけで、上位 100 の候補をすぐ取れる

これは SQLite に「全員集合して整列して」と言うのではなく、「最初から並んでいる廊下」に連れて行く感じです。

次は `jobs`（run のメタ情報）。ここに条件と tie-breaker があります：

- 絞り込み：`queue`、`status`
- 最後の tie-breaker：`created_at ASC`（同点のとき順位を安定させる）

なので、この 3 つに対応するインデックスを貼ります：

```sql title="index（例）"
CREATE INDEX IF NOT EXISTS idx_jobs_filter
ON jobs (
  queue,
  status,
  created_at
);
```

ここでの `created_at` は、**絞り込みのためではなく**「最後のひと押し」のためです。

というのも、`ORDER BY` の最後はこうだから：

```sql
... , j.created_at ASC
```

もし `score1/score2` が同じ行が大量にあるなら（同点はよくある）、SQLite は：

1. まず同点の巨大な候補の塊を集める
2. `jobs` に join して `created_at` を読む
3. **その塊を `created_at` で並べ替える**
4. そこで初めて、安定して上位 100 件を選べる

`created_at` を支えるインデックスが無いと、SQLite はだいたいこうします：

- 一時的なソート構造（temp B-tree）を作る
- 同点候補を突っ込む
- ソートしてから結果を返す

query plan で見えるのがこれ：`USE TEMP B-TREE FOR ORDER BY`

自分のクエリが追加ソートしているか知りたいなら、`EXPLAIN QUERY PLAN` で SQLite に聞けます：

```sql
EXPLAIN QUERY PLAN
SELECT ...;
```

見たい計画はだいたいこうです：

- `USING INDEX ...`（貼ったインデックスを使っている）
- `SCAN TABLE` が無い（全表スキャンしていない）
- `USE TEMP B-TREE FOR ORDER BY` が無い（追加ソートしていない）

もし `TEMP B-TREE` が見えたら、ほぼこう断定できます：

- まだ追加ソートしている。Top-N の LIMIT は最後にしか効いていない。直そう。

## よくある落とし穴

1. **数字を TEXT で持って、CAST でソートする**

   ```sql
   ORDER BY CAST(score1 AS REAL) DESC
   ```

   この 1 行でインデックスが死にます。

   SQLite は「変換後の結果」を使ってインデックスを辿れません。

   **対策は単純**：
   数値は最初から `INTEGER / REAL` で持ちましょう。ソート時の CAST に頼らない。

2. **ソート列を足し続けると、インデックスが太り続ける**

   `score1 → score2 → score3 → score4 → …`

   最後に残るのは、書き込みが遅くて容量を食うのに、クエリが必ずしも速くならない“インデックス怪獣”です。

   **対策**：どの指標が本当に順位を変えるか確認し、ソートキーは最小限に。

leaderboard が頻繁にクリックされるようになったら、こんな選択肢もあります：

- **Top-N 結果をキャッシュする**（leaderboard は典型的に read-heavy / write-light）
- **人気 split だけ partial index を作る**
- **オフラインでランキングを計算し、結果テーブルを引く**

SQLite は「クエリエンジン」として非常に向いています。でも毎回リアルタイムに計算する必要はありません。ちょっとした工夫で、プロダクトはかなり安定します。

## まとめ

クエリが遅いとき、よくある理由は SQLite がダメだからではありません。

あなたが SQLite に要求しているのは：

- join（テーブル結合）
- 条件が複数
- 複合ソート
- しかも順位が安定した Top-N クエリ

なのに、それに対応したインデックスを与えていないからです。

だから SQLite は一番保守的で一般的なやり方を選びます：全部読む、全部並べ替える、そのあと大半を捨てる。

本当に調整すべきなのは、SQL の書き方そのものよりも、事前に考えたかどうかです：

- どの条件が「範囲を絞る」のか
- どの列が本当に「順位を変える」のか
- どのソートが SQLite に「追加の並べ替え」をさせてはいけないのか

SQLite はずっと堅実です。

必要なのは、あなたが「本当に欲しいもの」をもう少し精密に伝えることだけ。

## 参考資料

- [SQLite: Indexes](https://www.sqlite.org/lang_createindex.html)
- [SQLite: Query Planner](https://www.sqlite.org/queryplanner.html)
- [SQLite: EXPLAIN QUERY PLAN](https://www.sqlite.org/eqp.html)
