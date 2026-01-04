---
slug: sqlite-intro-embedded-database
title: SQLite 入門（1）：なぜまた君なの？
authors: Z. Yuan
tags: [sqlite, database, storage]
image: /ja/img/2025/0804.jpg
description: サービスを起動せずに使える軽量データベース。
---

システム設計の初期には、こんなトレードオフに遭遇することがよくあります：

必要なのは構造化データの保存だけ。でも、フル機能のデータベースサービスを導入すると、接続管理、デプロイ、監視、冗長化、そして長期運用コストまで抱えることになります。どう計算しても割に合わないとき、プロジェクトにはたいていこんなファイルが現れます：

```
something.db
```

たいていそれは、「また SQLite が選ばれた」というサインです。

<!-- truncate -->

## SQLite とは？

SQLite は **組み込み（embedded）データベースエンジン**です。

これを「エンジニアが影響を判断できる」言い方にすると、こんな感じです：

- **サービスではない**（MySQL / Postgres のように常駐プロセスが要らない）
- **ライブラリ**としてアプリにリンクされる
- socket や TCP を経由せず、アプリがファイルを直接読み書きする
- 多くのケースで、**1 つのファイルがそのまま 1 つのデータベース**になる
- `:memory:` を使えば、プロセスの寿命の間だけ存在する DB にもできる

つまり SQLite の本質はとてもシンプルです：

> **データベースエンジンをアプリの中に入れてしまう。**

この立ち位置が決まると、強みと制約も同時に決まります。

## なぜ SQLite に何度も出会うのか？

SQLite は、ある特定の「どっちも欲しい」をピンポイントで解決するからです：

> **構造化データは欲しい。でも、手間は省きたい。**

データベース一式の運用は面倒です。省けるなら省きたい。

SQLite を使えば、次のことをしなくて済みます：

- port を開ける
- daemon を常駐させる
- backup / replica / failover を設計する
- 「DB が起動していないとアプリが動かない」を心配する

そのため SQLite は、こんな場所に何度も登場します：

- **ローカル開発・テスト**：起動が速く、用が済んだら捨てられる
- **デスクトップ / モバイルアプリ**：内蔵できて信頼でき、外部サービスに依存しない
- **社内ツール、プロトタイプ、小規模な管理画面**
- **エッジ / 単一ノード運用**：ネットワークが無い、またはリモート DB に依存できない
- **読み取りが多く書き込みが少ない層**：キャッシュ、インデックス、タスク状態、メタデータ

もし一度でもこう思ったことがあるなら：

> 「これのためだけに Postgres を立ち上げるのは割に合わない。」

SQLite はほぼ確実に近くにいます。

## SQLite の基本概念

公式ドキュメントを最初から最後まで読む必要はありませんが、次の用語が何を制御しているかは押さえておくと良いです。

### 1. Connection

SQLite の「接続」はネットワーク接続ではなく、**データベースファイルへのアクセスコンテキスト**です。

実務上の原則は一つだけ：

> **複数のスレッド / プロセスで同じ connection を共有しない。**

各 worker / スレッドごとに接続を開く方が安全です。

### 2. Transaction

2 行の SQL を実行しているだけに見えても、SQLite にとっては次を決めています：

- これらの操作をまとめて成功させるか
- 途中で失敗したときに全てロールバックするか
- ロックをいつ取得し、いつ解放するか

**トランザクション無しの SQLite は、性能も一貫性も崩れます。**

### 3. Journal / WAL

SQLite が並行アクセスにどこまで耐えられるかの鍵です。

- デフォルトは rollback journal（保守的でシンプル）
- WAL（Write-Ahead Logging）にすると：
  - 複数の reader が同時に読める
  - writer が DB 全体を止めにくくなる

`database is locked` に遭遇したら、ほぼ間違いなくここに戻ってきます。

### 4. Type affinity

SQLite は**強い型付けのデータベースではありません**。

Postgres のように、integer カラムに文字列を入れようとしても、強制的に止めてはくれません。

代わりにこう言います：

> 「この型を推奨するけど、責任は取らない。」

自由度は高いですが、責任はあなたにあります。

### 5. Constraint

`PRIMARY KEY`、`UNIQUE`、`CHECK`、`FOREIGN KEY`
これは飾りではなく、**データ層の最後の防衛線**です。

SQLite は勝手に補ってくれません。書かなければ、本当に無いです。

:::tip
SQLite は「動くシステム」をすぐに作れますが、**データの正しさは設計で担保する必要があります**。
:::

## 試してみる

以下の例は macOS / Linux、または SQLite と Python 環境があることを前提にしています。

もし `sqlite3` コマンドが無い場合は、SQLite CLI が未インストールという意味です（ただし Python からはそのまま使えます）。

Ubuntu ならこう入れます：

```bash
sudo apt update
sudo apt install sqlite3
```

macOS ならこちら：

```bash
brew install sqlite
```

### 1. CLI でデータベースを作る

```bash
sqlite3 demo.db
```

テーブル作成：

```sql
CREATE TABLE IF NOT EXISTS notes (
  id INTEGER PRIMARY KEY,
  title TEXT NOT NULL,
  body TEXT,
  created_at TEXT NOT NULL
);
```

データ挿入：

```sql
INSERT INTO notes (title, body, created_at)
VALUES ('hello', 'sqlite is a file', '2025-08-04T12:00:00Z');
```

クエリ：

```sql
SELECT id, title, created_at
FROM notes
ORDER BY created_at DESC
LIMIT 5;
```

### 2. Python で書き込む

```python title="python sqlite3（例）"
import sqlite3

conn = sqlite3.connect("demo.db")

# 接続は worker ごとに個別に開く
conn.execute("PRAGMA foreign_keys = ON;")

conn.execute(
    "INSERT INTO notes(title, body, created_at) VALUES (?, ?, ?)",
    ("hello", "sqlite is a file", "2025-08-04T12:00:00Z"),
)
conn.commit()
```

**唯一のポイント**：必ず `?` を使ってパラメータをバインドし、SQL を文字列結合で組み立てないこと。

## よくある質問

- **外部キーが「効いてないように見える」のはなぜ？**

  SQLite 初心者が一番ハマりやすい錯覚のひとつです。

  たしかに：

  - schema に `FOREIGN KEY` を書いた
  - `ON DELETE CASCADE` も付けた
  - でも親テーブルの行を消しても、子テーブルが何も反応しない

  原因は 1 つだけ：

  > **SQLite の外部キー制約はデフォルトで OFF で、しかも接続ごとに別です。**

- **正しいやり方**

  外部キーを使うなら、**接続を開いたら最初にこれを実行します：**

  ```sql
  PRAGMA foreign_keys = ON;
  ```

## いつ SQLite を「使わないべき」か？

SQLite は便利ですが、万能ではありません。

次のような場合は client/server 型の DB を検討してください：

- **書き込みの並行性が高い**（複数 writer が同時に激しく書き込む）
- **複数マシンで同じデータを共有する**
- **複雑な権限、監査、HA、レプリケーションが必要**
- 自分で「DB が持つべき機能」を作り始めてしまった

そうなったら無理をせず、別のデータベースを選びましょう。

## 参考資料

- [SQLite 公式サイト](https://www.sqlite.org/index.html)
- [SQLite: In-Memory Databases](https://www.sqlite.org/inmemorydb.html)
- [SQLite: Foreign Key Support](https://www.sqlite.org/foreignkeys.html)
