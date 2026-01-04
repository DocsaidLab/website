---
slug: sqlite-queue-observability-governance
title: SQLite 実戦（5）：私は誰？ここはどこ？
authors: Z. Yuan
tags: [sqlite, observability, data-governance, job-queue]
image: /ja/img/2025/1223.jpg
description: エラーコード、保持戦略、監査ログまで。キューを運用できる形にする。
---

きっとこんな会話をしたことがある：

- あなた：システム、いまどう？
- システム：動いてる。
- あなた：どこで？
- システム：動いてる。

なるほど。ありがとう。まったく役に立たない。

<!-- truncate -->

## Queue

SQLite で queue を作るのはもちろんできる。でも最低限、次の 5 つのエンジニアリングな問いに答えられないといけない：

1. **いまキューの中に job は何件ある？状態ごとに何件？**
2. **いちばん古い job はどれだけ詰まってる？誰（どの worker）に詰まってる？**
3. **リトライ回数の分布は？特定のタイプだけ延々と失敗してない？**
4. **失敗理由は何？集計（Top-N）できる？**
5. **データはどれくらい保持する？孤児データを作らず、DB を壊さずにどう消す？**

この 5 つの問いの裏にあるのは、実は 2 つのキーワードだ：

- **Observability（可観測性）**：祈るのではなく、「データで」いまの状態を理解できるか？
- **Governance（データガバナンス）**：データの保持と削除を「コントロール可能に」行い、削除後も整合性と追跡可能性を保てるか？

ここでは、実務寄りの設計でこの 2 つを地に足のついた形にする。

## Queue を 1 枚のテーブルで済ませない

queue の直感的な作り方はこうなりがち：

- `jobs` テーブル 1 枚
- `status` カラム
- 必要になったらカラムを足す

動きはする。でもすぐに「聞きたい質問に schema が答えられない」状態になる。

### 本当に必要なのは 3 つのデータ階層

queue を 3 つの層に分けると、運用が急に楽になる：

1. **Job 本体（Current State）**
   いまこの job が「どの状態か」を素早く知れること。

2. **試行記録（Attempts / Retries）**
   同じ job が何度も走ることがある。「何回目が失敗した？毎回同じ理由？所要時間は？」に答えたい。

3. **イベント記録（Audit Log / Event Log）**
   状態は「結果」、イベントは「過程」。デバッグしやすいシステムは「1 行を見張る」より、イベントの流れで追う。

## どんな「可観測」カラムが要る？

「可観測」とは schema を百科事典にすることではない。さっきの 5 つの問いに答えるための設計だ。

### `jobs` にあるとよいカラム

- `status`：状態機械（Queued / Claimed / Running / Succeeded / Failed / Cancelled …）
- `created_at / started_at / finished_at`：時間軸（待ち時間・処理時間を計算できる）
- `priority`（任意）：侮るな。あとでスケジューリングと運用が感謝する
- `worker_id`（または `claimed_by`）：誰が取ったか（問題 worker の特定）
- `heartbeat_at`：ハートビート時刻（worker が生きているか判断）
- `retry_count`：これまでのリトライ回数（集計を速くする）
- `error_code`：集計可能なエラー分類（Top-N の要）
- `error_detail`：短い要約（全ログを入れる場所ではない）

### 提案スキーマ

あくまで提案だが、運用上の問いをきれいに分離することを狙っている。

1. **jobs：現在状態（提供側は速く）**

   ```sql
   CREATE TABLE jobs (
     id            INTEGER PRIMARY KEY,
     type          TEXT NOT NULL,                 -- job 種別（集計のグルーピング用）
     status        TEXT NOT NULL,                 -- 状態機械
     priority      INTEGER NOT NULL DEFAULT 0,

     created_at    INTEGER NOT NULL,              -- Unix epoch seconds
     started_at    INTEGER,
     finished_at   INTEGER,

     claimed_by    TEXT,                          -- worker_id
     claim_token   TEXT,                          -- 誤更新防止（任意）
     heartbeat_at  INTEGER,

     retry_count   INTEGER NOT NULL DEFAULT 0,
     max_retries   INTEGER NOT NULL DEFAULT 3,

     error_code    TEXT,                          -- 集計可能な分類
     error_detail  TEXT,                          -- 短い要約（長さ制限推奨）

     payload_ref   TEXT,                          -- 大きい payload は別に保存（ファイル/オブジェクトストレージ）
     result_ref    TEXT                           -- 大きい結果は別に保存
   );
   ```

2. **job_attempts：各試行（「何回目？」が必要）**

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

3. **job_events：イベント流（デバッグはこれ）**

   ```sql
   CREATE TABLE job_events (
     id          INTEGER PRIMARY KEY,
     job_id      INTEGER NOT NULL,
     ts          INTEGER NOT NULL,
     event       TEXT NOT NULL,                   -- CLAIMED/STARTED/HEARTBEAT/FAILED/RETRY_SCHEDULED...
     actor       TEXT,                            -- worker_id / system
     detail      TEXT,                            -- 短い JSON（詰め込みすぎない）

     FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
   );
   ```

   > なぜ event テーブルが必要？
   > 「状態」はいま FAILED だとしか言わない。でも「イベント」ならこう教えてくれる：
   >
   > - いつ、誰に claim されたか
   > - どれくらい走ったか
   > - 最後のハートビートはいつか
   > - どの retry から失敗し始めたか
   >
   > デバッグは当てずっぽうじゃない。履歴を見る。

## インデックス

可観測性は、SQL を何本か書いて終わりじゃない：

- **最低でも、ちゃんと速く走るようにしないとね？**

### よく使うインデックス（提案）

```sql
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_type_status ON jobs(type, status);

-- stuck を探す／最古の running を探すのに一番使う
CREATE INDEX idx_jobs_running_age
ON jobs(status, heartbeat_at, created_at);

-- 失敗 Top-N 集計
CREATE INDEX idx_jobs_failed_code
ON jobs(status, error_code);

-- attempts / events の検索キー
CREATE INDEX idx_attempts_job ON job_attempts(job_id, attempt);
CREATE INDEX idx_events_job_ts ON job_events(job_id, ts);
```

設計原則はシンプルだ：**ダッシュボードでよく投げるクエリから逆算してインデックスを貼る。**

## よく使う観測クエリ

1. **状態ごとの件数：いまどれくらい忙しい？**

   ```sql
   SELECT status, COUNT(*) AS cnt
   FROM jobs
   GROUP BY status
   ORDER BY cnt DESC;
   ```

   すぐにこういうサインが見えてくる：

   - `QUEUED` が爆増 → 上流が送りすぎ、下流が捌けてない
   - `RUNNING` は多いのにスループットが上がらない → worker が詰まってる or リソース不足
   - `FAILED` がじわじわ増える → 特定 job のバグ or 外部依存が壊れてる

2. **いちばん長く詰まっている進行中 job を探す**

   ```sql
   SELECT id, type, claimed_by, created_at, heartbeat_at
   FROM jobs
   WHERE status IN ('CLAIMED', 'RUNNING')
   ORDER BY COALESCE(heartbeat_at, created_at) ASC
   LIMIT 20;
   ```

   ここでのポイントは `COALESCE`：

   - ハートビートがあれば、それを「直近の生存証拠」として使う
   - 無ければ `created_at` にフォールバック（そもそも動いていないか、設計の抜けがある）

3. **リトライ分布：特定の種類だけ延々と回ってない？**

   ```sql
   SELECT retry_count, COUNT(*) AS cnt
   FROM jobs
   WHERE status IN ('QUEUED', 'CLAIMED', 'RUNNING', 'FAILED')
   GROUP BY retry_count
   ORDER BY retry_count DESC;
   ```

   `retry_count = 3` に塊ができていたら、だいたいこういう原因だ：

   - 外部サービスが落ちてる（timeout）
   - 入力が不正（invalid input）
   - 仕組みレベルのバグ（internal）

   次はエラーコードの集計に進む。

4. **失敗理由 Top-N：集計できないと話にならない**

   ```sql
   SELECT error_code, COUNT(*) AS cnt
   FROM jobs
   WHERE status = 'FAILED'
   GROUP BY error_code
   ORDER BY cnt DESC
   LIMIT 20;
   ```

   これが回る前提は「まともな `error_code` があること」だ。

## エラーコード設計

`error_code` は「統計」のために存在する。よくあるアンチパターンはこれ：

- `error_detail` に stack trace 全部を突っ込む
- `error_code` を入れない（または適当に入れる）

最後は「1 件ずつログを見る」しかなくなり、集計できない。

### 推奨するエラーコード階層

`CATEGORY:SUBCATEGORY`（または `CATEGORY/SUBCATEGORY`）にする：

- `TIMEOUT:UPSTREAM_API`
- `INVALID_INPUT:SCHEMA_MISMATCH`
- `RESOURCE:OUT_OF_MEMORY`
- `INTERNAL:ASSERTION_FAILED`
- `DEPENDENCY:DB_LOCKED`

### `error_detail` の原則

- **短い要約**で十分（例：200〜500 文字）
- stack trace 全文／JSON 全文／レポート全文はファイルシステムや object storage に置く。DB は `path/URL` だけを持つ

「ゴミ箱型 log storage」は何度も見てきた。本当におすすめしない。

## 外部キーと孤児データ

> 放っておくと、あとであなたを殺しに来る。

SQLite の外部キーは「ある」。でも **デフォルトで有効ではない**。

有効にしなければ、無かったことになる。

ここでやるべきことは 2 つ：

1. **接続ごとに外部キーを有効化する**

```sql
PRAGMA foreign_keys = ON;
```

これは 1 回の設定ではない。接続プールを使う、接続を張り直す——そのたびに必要だ。

2. **`ON DELETE CASCADE` で整合性を保つ**

`jobs` を消すなら `attempts/events` も自動で消えるようにする。そうしないと必ず孤児データが残る。

## Upsert

「更新」として `INSERT OR REPLACE` を使うのはやめよう！

これは太字で 3 回書く価値がある：

- **`REPLACE` は update ではない**
- **`REPLACE` は update ではない**
- **`REPLACE` は update ではない**

挙動としては、むしろこうだ：

> 先に DELETE してから INSERT

外部キーがある、audit log がある、attempt/event がある……そんな状態で `REPLACE` を使うと即死する。

しかも死んだあとで、原因が分からない。

正しいやり方は `ON CONFLICT DO UPDATE`。状態更新は予測可能で追跡可能であるべきで、こっそり削除してはいけない：

```sql
INSERT INTO jobs(id, status, heartbeat_at)
VALUES (?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
  status = excluded.status,
  heartbeat_at = excluded.heartbeat_at;
```

## Audit log（イベントテーブル）

`jobs` の 1 行だけを見ていると、見えるのは「いま」だけで「なぜそうなったか」が見えない。

イベントテーブルの哲学はこうだ：**状態はスナップショット、イベントはタイムライン。**

最低でも、次は記録したい：

- `ENQUEUED`：入列
- `CLAIMED`：worker に取られた
- `STARTED`：処理開始（claimed と分けてもよい）
- `HEARTBEAT`：ハートビート（頻度を下げる／「最後だけ」でもよい）
- `FAILED`：失敗（`error_code` を付ける）
- `RETRY_SCHEDULED`：リトライ予定（delay、attempt を付ける）
- `SUCCEEDED`：成功
- `CANCELLED`：手動キャンセル
- `RECOVERED`：回収／再割り当て（lease があるなら）

## 保持戦略

データは消す。でも「ルールに従って」消す。

queue システムが最後に必ずぶつかる問いがある：

> **どれくらい残す？**

「全部残す」なら、SQLite は容量で人生相談を始める。
「全部消す」なら、次の事故で証拠が残らない。

現実的には、こんな戦略になる：

- `SUCCEEDED`：30 日（だいたい追跡には十分）
- `FAILED`：90 日（失敗は長めに。何度も戻ってくる）
- `CANCELLED`：要件次第（よくあるのは 30 日）

削除は、分割して、中断できて、再実行できるようにするのがおすすめだ。

100 万件を一気に消すな。サービス提供中ならなおさらだ。1 分ごとに掃除ジョブを走らせて、少しずつ消す：

```sql
-- 成功 job を削除（30 日より前）
DELETE FROM jobs
WHERE status = 'SUCCEEDED'
  AND finished_at < strftime('%s', 'now') - 30*86400
LIMIT 5000;
```

### `VACUUM` は必要？

- SQLite は削除してもすぐにファイルサイズが縮まらない（内部で再利用可能とマークするだけ）
- `VACUUM` は DB 全体を書き直す。I/O が重く、時間もかかる

おすすめは：

- **普段は分割削除で回す**
- **本当に縮めたい（容量が逼迫）ときだけ、低負荷時間に `VACUUM`**

WAL モードなら、WAL ファイルと checkpoint の挙動も理解しておく。

## よくある問題

1. **`error_detail` 無制限で、DB がゴミ箱になる**

   **対策**：

   - `error_code` で分類する
   - `error_detail` は短い要約だけ
   - 全文レポートは外に置く（DB は参照だけ）

2. `foreign key` を有効にし忘れて、親を消したあとに孤児データが残る

   **対策**：接続ごとに `PRAGMA foreign_keys = ON`、そして `ON DELETE CASCADE`

3. `INSERT OR REPLACE` を upsert として使う

   **対策**：`ON CONFLICT DO UPDATE` を使い、`REPLACE` に DELETE をさせない

4. 観測クエリを書いたのに、インデックスを貼っていない

   **対策**：ダッシュボード／運用でよく使うクエリから逆算してインデックスを貼る（全表走査で SQLite と殴り合わない）

## まとめ

キューがあの 5 つの問いに答えられるようになると、システムは急に「システムらしく」なる：

- 現況を説明できる（可観測）
- 過程を追跡できる（audit）
- データのライフサイクルを制御できる（ガバナンス）
- 事故対応を「勘」から「クエリ」に変えられる

ここまでやれば、SQLite を queue に使うのも水到渠成だ。

## 参考資料

- [**SQLite: Foreign Key Support**](https://www.sqlite.org/foreignkeys.html)
- [**SQLite: UPSERT**](https://www.sqlite.org/lang_UPSERT.html)
- [**SQLite: VACUUM**](https://www.sqlite.org/lang_vacuum.html)
