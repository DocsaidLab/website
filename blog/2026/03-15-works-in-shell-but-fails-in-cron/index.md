---
slug: works-in-shell-but-fails-in-cron
title: 在 terminal 明明正常，為什麼進了 cron / systemd 就突然壞掉？
authors: Z. Yuan
tags: [linux, cron, systemd, debugging, shell]
image: /img/2026/0315-shell-vs-cron.svg
description: 同一條指令，互動式 shell 正常，到了 cron 或 systemd 卻翻車。問題通常不是鬼，而是環境、工作目錄、權限與 shell 差異。
---

你在 terminal 裡跑過了。

成功。

你把同一條指令塞進 cron。

炸掉。

你再把它搬去 systemd。

炸得更安靜。

這種問題很適合拿來培養宗教信仰，因為表面上看起來完全不合理。

但電腦其實很誠實。

它只是默默告訴你：

> **你以為是「同一條指令」，實際上只是「長得很像的兩個執行環境」。**

這篇要拆的不是玄學，而是四個常見落點：

1. `PATH` 不一樣
2. 工作目錄不一樣
3. shell 不一樣
4. 權限與環境變數不一樣

最後再給一套我自己比較信的排查順序。

<!-- truncate -->

## 先講結論：不是指令變了，是上下文變了

很多人會說：

- 「我明明貼的是同一行」
- 「本機 shell 可以跑」
- 「放進排程就失敗」

這三句通常都是真的。

但還少了第四句：

- **「執行它的人、位置、環境、與啟動方式都變了。」**

在互動式 shell 裡，你有很多隱形加成：

- 登入時自動載入的 shell 設定
- 你目前所在的工作目錄
- 你的使用者權限
- 已經 export 好的環境變數
- shell alias / function / pyenv / nvm / conda 之類的魔法

cron 與 systemd 對這些東西的態度通常是：

- 沒興趣
- 不幫你猜
- 猜錯也不會道歉

## 症狀一：`python` 找不到、`node` 找不到、`mytool` 找不到

最常見，也最無聊。

例如你在 terminal 裡可以跑：

```bash
python script.py
```

但 cron 裡卻得到：

```text
python: command not found
```

這幾乎就是 `PATH` 問題。

### 為什麼互動式 shell 正常？

因為你登入時，shell 可能偷偷幫你做了很多事，例如：

- 載入 `~/.zshrc`
- 載入 `~/.bashrc`
- 初始化 `pyenv`
- 初始化 `nvm`
- 把 `~/.local/bin` 加進 `PATH`

但 cron 不會很熱心地替你重演人生。

它常常只給你一條很短的 `PATH`，像這樣：

```text
/usr/bin:/bin
```

那麼：

- `python` 可能不是你想的那個 python
- `node` 可能根本不存在
- 你自己安裝的 CLI 幾乎等於蒸發

### 解法：不要賭 `PATH`

直接寫絕對路徑：

```bash
/usr/bin/python3 /opt/jobs/report.py
```

如果你想知道某個指令在互動式 shell 的實際位置：

```bash
command -v python3
command -v node
command -v mytool
```

把結果明確寫進排程。

### cron 範例

```cron
PATH=/usr/bin:/bin:/usr/local/bin
*/10 * * * * /usr/bin/python3 /opt/jobs/report.py >> /var/log/report.log 2>&1
```

這樣至少你不是把成敗交給運氣。

## 症狀二：檔案明明存在，程式卻說找不到

另一個經典問題：

```python
with open("config/settings.json") as f:
    ...
```

你在專案根目錄手動跑：

```bash
python app.py
```

沒事。

到了 cron：

```text
FileNotFoundError: config/settings.json
```

恭喜，你撞到**工作目錄**了。

### 互動式 shell 的幻覺

你手動執行時，通常剛好站在專案根目錄。

所以相對路徑：

```text
config/settings.json
```

會被解析成：

```text
/your/project/config/settings.json
```

但 cron 啟動時，工作目錄不一定是你的專案。

systemd 也一樣，如果你沒設定 `WorkingDirectory`，它不會替你體貼地猜。

### 解法 A：程式內改成基於腳本位置

Python：

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
config_path = BASE_DIR / "config" / "settings.json"
```

這種寫法通常比「祈禱目前目錄正確」可靠很多。

### 解法 B：systemd 明確指定工作目錄

```ini
[Service]
WorkingDirectory=/opt/myapp
ExecStart=/usr/bin/python3 /opt/myapp/app.py
```

### 解法 C：cron 先 `cd`

```cron
*/5 * * * * cd /opt/myapp && /usr/bin/python3 app.py >> /var/log/myapp.log 2>&1
```

這可以用，但比起在程式裡處理絕對路徑，我通常沒那麼喜歡。

因為它把脆弱性藏在排程設定裡。

## 症狀三：用了 bash 語法，但實際上不是 bash

這也很常見。

你寫了一段：

```bash
source venv/bin/activate
for file in *.json; do
  echo "$file"
done
```

在 terminal 裡好好的。

進 cron 後開始出現：

```text
source: not found
```

或者其他看起來像 shell 在鬧脾氣的錯。

### 原因：cron 預設 shell 不一定是你平常那個

很多 cron 環境預設是：

```text
/bin/sh
```

不是 `bash`，更不是 `zsh`。

而：

- `source` 是 bash / zsh 常見語法
- `[[ ... ]]` 不是 POSIX `sh` 保證支援
- 陣列、某些展開語法、process substitution，也都可能直接翻車

### 解法

如果你真的需要 bash：

```cron
SHELL=/bin/bash
PATH=/usr/bin:/bin:/usr/local/bin
*/10 * * * * /bin/bash /opt/jobs/run.sh >> /var/log/run.log 2>&1
```

更穩一點的做法是：

1. 把邏輯收進腳本
2. 腳本第一行寫清楚 shebang

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /opt/jobs
./venv/bin/python task.py
```

讓 shell 類型明確，事情會少很多。

## 症狀四：權限不一樣，所以世界也不一樣

你手動執行時，也許是你自己。

但排程執行時，可能是：

- `root`
- `www-data`
- 某個 service account
- 某個權限很節制的 systemd user

這時候你會看到的錯誤通常像：

```text
Permission denied
```

或是更陰險一點：

- 寫檔 silently 失敗
- 讀不到憑證
- 連不到只有你自己能用的 socket
- 找不到 home 目錄底下的設定檔

### systemd 特別容易出現這個問題

假設你有：

```ini
[Service]
User=myapp
ExecStart=/usr/bin/python3 /opt/myapp/app.py
```

那就表示：

- 它不是用你的登入身分跑
- 它的 `HOME` 可能不同
- 它看不到你個人 shell 裡那些設定
- 它未必能讀你家目錄裡的秘密檔案

這其實是好事，因為比較安全。

只是 debug 時會比較誠實。

## 先別修，先印出現場

很多人一失敗就開始改設定。

我通常先做更無聊、但更有效的事：

**把排程環境自己印出來。**

例如做一個 `debug-env.sh`：

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

先讓 cron 或 systemd 跑這個。

你通常很快就會看到真相，例如：

- `pwd=/`
- `HOME=/`
- `PATH=/usr/bin:/bin`
- `python3` 路徑跟你以為的不一樣

電腦沒有背叛你。

只是你之前沒問。

## systemd 的排查方式通常比 cron 友善

cron 最大的問題是，它很安靜。

安靜到你會開始懷疑自己。

如果是 systemd service，我通常會先看：

```bash
systemctl status myapp.service
journalctl -u myapp.service -n 100 --no-pager
```

然後在 service 檔裡把幾個關鍵欄位寫清楚：

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

如果還需要環境檔：

```ini
EnvironmentFile=/etc/myapp.env
```

systemd 的哲學其實很合理：

- 不替你偷載一堆魔法
- 把執行條件寫成可檢查的設定

剛開始會覺得它麻煩。

習慣之後，你會開始覺得互動式 shell 太隨便。

## 一個比較不容易踩雷的做法

如果這件事很重要，我通常不讓排程直接塞一大串 inline 指令。

我會：

1. 寫一個獨立腳本
2. 用絕對路徑
3. 在腳本開頭明確設定 shell 與錯誤模式
4. 明確設定工作目錄
5. 把 stdout / stderr 收進 log

例如：

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /opt/report-job
export PATH=/usr/local/bin:/usr/bin:/bin

/opt/report-job/.venv/bin/python generate_report.py \
  >> /var/log/report-job.log 2>&1
```

然後 cron 只做一件事：

```cron
0 * * * * /opt/report-job/run.sh
```

這樣問題會比較集中，排查成本低很多。

## 我的排查順序

如果你現在就卡在「terminal 可以，排程不行」，我會照這個順序看：

1. **看 log**：有沒有 stderr，被丟去哪裡
2. **印環境**：`whoami`、`pwd`、`env`、`PATH`
3. **確認絕對路徑**：binary、腳本、設定檔、輸出目錄
4. **確認 shell 類型**：`sh` 還是 `bash`
5. **確認權限**：執行者是誰，能不能讀寫需要的檔案
6. **縮小腳本**：先只跑最小可重現版本

通常前 3 步就會抓到。

因為大多數這類 bug，不是高深，而是基本面沒有寫死。

## 結語

「我在 terminal 跑過了」這句話，本身證明的事情其實很少。

它最多只能證明：

- 在那個使用者
- 那個 shell
- 那個目錄
- 那組環境變數
- 那個當下

它是成功的。

離「在排程環境也會成功」還差一整層上下文。

所以，下次遇到這種問題時，不要先懷疑電腦中邪。

先懷疑你把太多假設留在互動式 shell 裡。

因為 cron 和 systemd 最擅長的事，就是把這些假設一條一條拆掉。

非常冷酷。

但老實說，這也算是一種服務。
