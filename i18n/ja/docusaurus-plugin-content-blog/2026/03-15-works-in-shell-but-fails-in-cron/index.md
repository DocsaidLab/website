---
slug: works-in-shell-but-fails-in-cron
title: ターミナルでは動くのに、なぜ cron / systemd に入れると壊れるのか
authors: Z. Yuan
tags: [linux, cron, systemd, debugging, shell]
image: /img/2026/0315-shell-vs-cron.svg
description: 対話シェルでは動くコマンドが、cron や systemd に入れた瞬間に崩れる。たいてい原因は魔法ではなく、環境、作業ディレクトリ、shell、権限です。
---

ターミナルで実行した。

動いた。

同じコマンドを cron に入れた。

壊れた。

さらに systemd に移した。

もっと静かに壊れた。

この手の問題は、見た目だけならかなり不条理です。

だから人はすぐ宗教に寄りがちです。

でも機械は別に神秘的ではありません。

言っていることはかなり単純です。

> **同じコマンドだと思っているだけで、実際には「似た文字列」が別の実行環境で動いているだけです。**

この記事では、よくある落とし穴を 4 つに絞って整理します。

1. `PATH` が違う
2. 作業ディレクトリが違う
3. shell が違う
4. 権限と環境変数が違う

最後に、私ならこう見る、という切り分け順も置いておきます。

<!-- truncate -->

## 先に結論：変わったのはコマンドではなく文脈

よくある言い方はこうです。

- 「本当に同じ 1 行です」
- 「ターミナルでは動きます」
- 「スケジューラだと失敗します」

どれも事実でしょう。

でも 1 行足りません。

- **実行するユーザー、ディレクトリ、環境、起動方法がもう同じではない。**

対話シェルには、目に見えない追い風がたくさんあります。

- ログイン時に読み込まれる shell 設定
- 今いる作業ディレクトリ
- 自分のユーザー権限
- export 済みの環境変数
- alias、function、`pyenv`、`nvm`、`conda` みたいな便利な魔法

cron と systemd の態度はだいたいこうです。

- それは知らない
- 必要なら明示して
- 勝手に補完はしない

## 症状 1: `python` がない、`node` がない、`mytool` がない

いちばん多くて、いちばん地味です。

ターミナルではこう動く。

```bash
python script.py
```

でも cron ではこうなる。

```text
python: command not found
```

ほぼ `PATH` の問題です。

### なぜ対話シェルでは動くのか

シェルが裏でいろいろ準備しているからです。

- `~/.zshrc` を読む
- `~/.bashrc` を読む
- `pyenv` を初期化する
- `nvm` を初期化する
- `~/.local/bin` を `PATH` に足す

cron はそのログイン儀式を再現してくれません。

最小限の `PATH` しか持っていないことも多いです。

```text
/usr/bin:/bin
```

そうなると：

- `python` が想定したものではない
- `node` が存在しない
- 自分で入れた CLI が消えたように見える

### 対策：`PATH` に期待しすぎない

絶対パスで書きます。

```bash
/usr/bin/python3 /opt/jobs/report.py
```

対話シェルで実体を確認するには：

```bash
command -v python3
command -v node
command -v mytool
```

出てきたパスを、そのままジョブに書くのが安全です。

### cron の例

```cron
PATH=/usr/bin:/bin:/usr/local/bin
*/10 * * * * /usr/bin/python3 /opt/jobs/report.py >> /var/log/report.log 2>&1
```

少なくとも運任せではなくなります。

## 症状 2: ファイルはあるのに、プログラムは「ない」と言う

これも定番です。

```python
with open("config/settings.json") as f:
    ...
```

プロジェクトルートで手動実行すると：

```bash
python app.py
```

問題なし。

cron に入れると：

```text
FileNotFoundError: config/settings.json
```

これは**作業ディレクトリ**です。

### 対話シェルの錯覚

手動実行するときは、たいてい自分がすでにプロジェクトルートにいます。

だから相対パスの：

```text
config/settings.json
```

は、実質こう解釈されます。

```text
/your/project/config/settings.json
```

でも cron はその開始位置を保証しません。

systemd も `WorkingDirectory` を書かなければ保証しません。

### 対策 A: スクリプトの場所基準で解決する

Python:

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
config_path = BASE_DIR / "config" / "settings.json"
```

今いる場所に期待するより、こちらの方がずっと安定します。

### 対策 B: systemd で `WorkingDirectory` を指定する

```ini
[Service]
WorkingDirectory=/opt/myapp
ExecStart=/usr/bin/python3 /opt/myapp/app.py
```

### 対策 C: cron 側で先に `cd`

```cron
*/5 * * * * cd /opt/myapp && /usr/bin/python3 app.py >> /var/log/myapp.log 2>&1
```

動きます。

ただ、排程設定に脆さを隠すので、私はコード側で絶対パスに寄せる方を好みます。

## 症状 3: bash のつもりで書いたが、実際には bash ではない

これもかなり多いです。

例えば：

```bash
source venv/bin/activate
for file in *.json; do
  echo "$file"
done
```

ターミナルでは平和。

cron では突然：

```text
source: not found
```

あるいは shell が機嫌を損ねたようなエラーが出ます。

### 理由

cron の既定 shell はよく `sh` です。

```text
/bin/sh
```

`bash` ではありません。もちろん `zsh` でもありません。

すると：

- `source` は使えないことがある
- `[[ ... ]]` がこける
- 配列や process substitution が壊れる
- 細かい挙動が静かに変わる

### 対策

bash が必要なら、明示します。

```cron
SHELL=/bin/bash
PATH=/usr/bin:/bin:/usr/local/bin
*/10 * * * * /bin/bash /opt/jobs/run.sh >> /var/log/run.log 2>&1
```

さらに安全なのは：

1. ロジックをスクリプトに寄せる
2. 先頭に shebang を書く

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /opt/jobs
./venv/bin/python task.py
```

これで shell の種類が事故ではなくなります。

## 症状 4: 権限が変わると、見える世界も変わる

手動実行では自分のユーザーかもしれません。

自動実行では、たとえば：

- `root`
- `www-data`
- service account
- 制限の強い systemd user

このとき、よく出るのは：

```text
Permission denied
```

もっと嫌なのは、静かな失敗です。

- 書き込みが通らない
- 認証情報を読めない
- socket に触れない
- home 配下の設定ファイルが見えない

### systemd はここをかなり露骨にする

たとえば：

```ini
[Service]
User=myapp
ExecStart=/usr/bin/python3 /opt/myapp/app.py
```

なら：

- ログイン中の自分では動かない
- `HOME` が違うかもしれない
- 個人の shell 設定は前提にできない
- home 配下の秘密ファイルは読めないかもしれない

これは本来よいことです。安全だから。

ただし debug 時は容赦がありません。

## 直す前に、現場を出力する

失敗するとすぐ設定をいじりたくなります。

でも私は先にもっと地味なことをします。

**その実行環境を自分で吐かせる。**

たとえば `debug-env.sh`：

```bash
#!/usr/bin/env bash
set -x

echo "whoami=$(whoami)"
echo "pwd=$(pwd)"
echo "shell=$SHELL"
echo "home=$HOME"
echo "path=$PATH"
env | sort
command -v python3 || true
command -v node || true
ls -la
```

まずこれを cron や systemd から実行します。

すると、だいたいすぐ分かります。

- `pwd=/`
- `HOME=/`
- `PATH=/usr/bin:/bin`
- `python3` が思っていた場所にない

裏切ったのは機械ではありません。

聞くべきことを、まだ聞いていなかっただけです。

## cron より systemd の方が debug はやりやすいことが多い

cron の最大の特徴は静かすぎることです。

静かすぎて、こっちが悪い気になってきます。

systemd service なら、まずここを見ます。

```bash
systemctl status myapp.service
journalctl -u myapp.service -n 100 --no-pager
```

そして unit file に、前提を明示します。

```ini
[Unit]
Description=My scheduled job

[Service]
Type=oneshot
User=myapp
WorkingDirectory=/opt/myapp
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
ExecStart=/usr/bin/python3 /opt/myapp/task.py
```

環境ファイルが必要なら：

```ini
EnvironmentFile=/etc/myapp.env
```

systemd の思想はかなりまともです。

- 魔法を持ち込まない
- 実行条件を設定として残す

最初は面倒に見えます。

慣れると、対話シェルの方が雑に見えてきます。

## 踏み抜きにくい構成

ジョブが重要なら、私は長い inline command をそのまま scheduler に書きません。

代わりに：

1. 専用スクリプトを書く
2. 絶対パスを使う
3. shell とエラーモードを先頭で宣言する
4. 作業ディレクトリを明示する
5. stdout / stderr を log に流す

例：

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /opt/report-job
export PATH=/usr/local/bin:/usr/bin:/bin

/opt/report-job/.venv/bin/python generate_report.py \
  >> /var/log/report-job.log 2>&1
```

cron 側は 1 行だけで済みます。

```cron
0 * * * * /opt/report-job/run.sh
```

この方が壊れる場所が少なく、調べるのも楽です。

## 私ならこの順で見る

いままさに「ターミナルでは動くのに、自動実行だと死ぬ」に詰まっているなら、この順です。

1. **log を見る**: stderr がどこへ行ったか
2. **環境を出す**: `whoami`、`pwd`、`env`、`PATH`
3. **絶対パスを確認する**: binary、スクリプト、設定ファイル、出力先
4. **shell を確認する**: `sh` か `bash` か
5. **権限を確認する**: 誰が実行し、何を読めて何を書けるか
6. **最小再現に縮める**: まず一番小さい形で再現する

たいてい最初の 3 つで捕まります。

こういう不具合の多くは高度な問題ではなく、前提が固定されていないだけだからです。

## 終わりに

「ターミナルでは動いた」という言葉が証明することは、実はそんなに多くありません。

せいぜい次を証明するだけです。

- そのユーザーで
- その shell で
- そのディレクトリで
- その環境変数で
- その瞬間に

動いた。

それは「自動実行でも動く」の 1 段手前にすぎません。

だから次に cron や systemd が、あなたの無実そうなコマンドを冷たく処刑したら、まず幽霊を疑わないでください。

文脈を疑ってください。

cron と systemd が得意なのは、暗黙の前提を一つずつ剥がすことです。

冷たい。

でも、かなり正直です。
