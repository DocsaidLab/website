---
slug: sqlite-lease-heartbeat-recovery
title: SQLite 実戦（3）：worker を救え
authors: Z. Yuan
tags: [sqlite, job-queue, reliability, heartbeat]
image: /ja/img/2025/1008.jpg
description: 詰まったタスクを自動回収し、job queue を墓場にしない。
---

きっとこんな状況に遭遇したことがある：

- queue が詰まっているように見える
- UI には RUNNING の job が山ほど残っている

でもその worker は、もう 3 回再起動した。job は死なないのに、worker が死ぬ。

本来なら愉快な週末のはずが、worker を悼む挽歌を歌う羽目になる。

<!-- truncate -->

## なぜ worker は死ぬのか？

なぜなら、あなたは制御できない世界にいるからです。

理想の世界では、worker は job を claim し、最後まで走り切って、状態を `SUCCEEDED` か `FAILED` にします。

世界は平和。

でも現実の世界で worker が死ぬ理由は、大体こんな感じです：

1. **プロセスが落ちる**：例外を拾いきれない、サードパーティ library の segfault、モデル推論の失敗。
2. **システムに殺される**：OOM killer、コンテナ再起動、ノード回収、リソースクォータ超過。
3. **死なないけど固まる**：外部 API が無限待ち、I/O hang、デッドロック、どこかの `while True` が回り続ける。
4. **デプロイ/アップグレードで中断**：rolling update、手動 kill、プロセスの置き換え。
5. **ネットワークと依存が壊れる**：S3、MQ、内部サービスの timeout。

気づくはずです：**worker は「優雅に失敗」してくれないことが多い。**

突然消えるか、永遠に固まる。そして残るのは、孤独な `RUNNING` だけ。

この「孤独な `RUNNING`」こそ、queue システムの腐敗源です。

## なぜ重要なのか？

queue システムが次の 3 つに答えられないなら、それは「運用できる」とは言えません：

1. **この job を今持っているのは誰？**
2. **最後の heartbeat はいつ？**
3. **もし永遠に戻らないなら、どう片づける？**

この 3 つはどれも「性能」の話ではなく、「信頼性」の話です。

なぜなら、人間の時間は有限なので、実務では：

- 毎回詰まるたびに log / マシン / pod を調べるのは無理
- 毎回 job を手動で `QUEUED` に戻すのも無理
- 「一部の job は永遠に RUNNING のままだけど、たまに掃除する」なんて受け入れられない

だから必要なのは、実務的な設計：**lease（租約）+ heartbeat（心拍）**です。

目的は「永久に所有」を「しばらく借りる」に変えること。

## lease + heartbeat

考え方はシンプルです：

- worker が job を claim するのは「永久所有」ではなく、「一定期間だけ借りる」こと。
- 借りたら定期的に生存報告し、lease を更新する。
- 更新しなければ音信不通として扱い、システムが回収できる。

つまり「可用性」を「worker を信じる」から「データを信じる」に移します。

そして DB は、少なくともいつ死ぬかわからない worker よりは信じられます。

## 最小モデル

ここで必要なのは派手な schema ではなく、次が判断できることです：

- この job をまだどこかの worker が持っているか？
- その保有（lease）はまだ有効か？
- 期限切れになったらどうするか？

よくある最小フィールドはこのあたり：

- `claimed_at`：取られた時刻
- `heartbeat_at`：最後の heartbeat
- `retry_count`：リトライ回数

ただし実務では、次の 2 つを足すのを強くおすすめします。システムがかなり安定します：

- `lease_expires_at`：租約の期限（直感的で、クエリもきれいになる）
- `lease_token`：今回 claim の証明（ゾンビ worker 対策）

:::info
**なぜ `lease_token`？**

防ぎたいのはこのシナリオです：

- worker A が job を claim → 途中で落ちる
- sweeper が回収 → worker B が再 claim
- worker A が復活して heartbeat を更新し続けたり、完了を書き込んだりする
- A に B がやられて、「1 件の job を 2 人が書く」という心霊現象が起きる

`lease_token` の考え方は：**claim のたびに鍵を替える**こと。
以降の更新は、その鍵を持っている場合だけ有効になります。
:::

## schema の例

時間は **UTC epoch seconds**（`INTEGER`）がおすすめです。文字列の timestamp は、いつか必ずあなたを刺します。

```sql title="jobs schema（例）"
CREATE TABLE IF NOT EXISTS jobs (
  id               INTEGER PRIMARY KEY,
  status           TEXT NOT NULL
                   CHECK (status IN ('QUEUED','CLAIMED','RUNNING','SUCCEEDED','FAILED')),

  owner_id         TEXT,     -- どの worker が取ったか（hostname/uuid）
  lease_token      TEXT,     -- 今回 claim の証明（ゾンビ対策）

  claimed_at       INTEGER,  -- unix epoch seconds (UTC)
  heartbeat_at     INTEGER,  -- unix epoch seconds (UTC)
  lease_expires_at INTEGER,  -- unix epoch seconds (UTC)

  retry_count      INTEGER NOT NULL DEFAULT 0,
  max_retry        INTEGER NOT NULL DEFAULT 5,

  finished_at      INTEGER,
  error_code       TEXT,
  error_detail     TEXT
);

-- sweeper がよく使う検索パス：status + 期限
CREATE INDEX IF NOT EXISTS idx_jobs_lease
ON jobs(status, lease_expires_at);

-- queue が job を拾うパス：QUEUED + id（または priority）
CREATE INDEX IF NOT EXISTS idx_jobs_queue
ON jobs(status, id);
```

## heartbeat の更新

heartbeat の本質は：「まだ生きていて、まだこの job を持っている」という宣言です。

だから heartbeat 更新では、次の 2 つを検証する必要があります：

- 自分は誰か（`owner_id`）
- 今回の鍵を持っているか（`lease_token`）

```sql title="heartbeat（例）"
UPDATE jobs
SET heartbeat_at = :now,
    lease_expires_at = :now + :lease_seconds
WHERE id = :job_id
  AND status IN ('CLAIMED', 'RUNNING')
  AND owner_id = :owner_id
  AND lease_token = :lease_token;
```

これは小さな write です。大量の情報を記録するのではなく、「システムの信頼」を保つための更新です。

## 期限切れ job の回収

worker が音信不通になると、DB が見えるのはこれだけ：

- `lease_expires_at < now`

あとは `RUNNING/CLAIMED` の job を現実に引き戻すだけです。

通常は 2 種類に分かれます：

1. まだ救える → `QUEUED` に戻して、誰かにやり直させる
2. 救えない → `FAILED` にして、結果を残す

## よくある落とし穴

1. **時間フォーマットがバラバラ**

   時刻を `2025/10/8 9:3:1` みたいな形式で保存して、文字列比較したら、結果はとても楽しいことになります。

   **対策**：UTC epoch seconds（`INTEGER`）を使う。
   SQLite で「今」を取るなら：

   - `unixepoch('now')`（秒）
   - または `strftime('%s','now')`

2. **sweeper が働きすぎて、DB が喧嘩し始める**

   sweeper を 1 秒ごとに回して毎回大量の row を更新し、同時に worker が heartbeat を更新すると、こうなります：

   - lock contention
   - `busy_timeout` が叩き切られる

   救いたかったのに、むしろ悪化してしまう？

   **対策**：

   - sweeper は 10 ～ 30 秒間隔
   - 1 回あたりバッチ処理（例：最大 100 件）
   - インデックスを正しく：`(status, lease_expires_at)`

3. **lease/heartbeat は「音信不通」を解決するが、「生きているのに終わらない」は解決しない**

   worker が無限ループに入っていても heartbeat を更新し続けるなら、lease では救えません。

   必要なのは **job-level timeout**（最大実行時間）です。例えば `max_runtime_seconds`。
   sweeper は `lease_expires_at` だけでなく、「どれくらい走っているか」も見るべきです。

## まとめ

polling queue を運用可能にするには、最低でも次の 2 つが必要です：

1. **lease（租約）**：worker が job を claim するのは永久所有ではなく、しばらく借りるだけ。期限が切れたら回収する。
2. **heartbeat（心拍）**：生きているなら lease を更新し、更新しなければ音信不通として扱う。

さらに、現実的なパーツを 3 つ足します：

- `max_retry / retry_count`：無限リトライでマシンを燃やさない
- `error_code / error_detail`：失敗を理解・分析できるようにする
- `lease_token`：ゾンビ worker 対策。2 人が同じ結果を書かないようにする

これで queue は「動いているように見える」から「事故っても自力で戻ってくる」に変わります。

worker を救うことは、あなたを debug 地獄から救うことでもある。

どう考えても損はない。

## 参考資料

- [SQLite: Date And Time Functions](https://www.sqlite.org/lang_datefunc.html)
- [The “Leases” Pattern](https://martinfowler.com/articles/patterns-of-distributed-systems/lease.html)
