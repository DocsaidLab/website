---
sidebar_position: 9
---

# 排程任務

我們期待可以在每天早上看到最新的郵件摘要，因此我們需要一個自動化的排程任務來完成這個目標。

## 使用 `crontab`

為了讓這個過程完全自動化，我利用了 Linux 的 `crontab` 功能來設置定時任務。這樣可以確保每天固定時間自動執行程式，抓取新郵件，生成摘要，並更新 GitHub 存儲庫。

具體的 `crontab` 設定如下：

```bash
crontab -e
```

接著添加以下內容：

```bash
# Edit this file to introduce tasks to be run by cron.
#
# Each task to run has to be defined through a single line
# indicating with different fields when the task will be run
# and what command to run for the task
#
# To define the time you can provide concrete values for
# minute (m), hour (h), day of month (dom), month (mon),
# and day of week (dow) or use '*' in these fields (for 'any').
#
# Notice that tasks will be started based on the cron's system
# daemon's notion of time and timezones.
#
# Output of the crontab jobs (including errors) is sent through
# email to the user the crontab file belongs to (unless redirected).
#
# For example, you can run a backup of all your user accounts
# at 5 a.m every week with:
# 0 5 * * 1 tar -zcf /var/backups/home.tgz /home/
#
# For more information see the manual pages of crontab(5) and cron(8)
#
# m h  dom mon dow   command

# Define your environment variables
OPENAI_API_KEY="your_openai_api_key"

# 每天早上 6 點自動執行更新程式
0 6 * * * /path/to/your/script/update_targets_infos.sh

# 每小時更新 GmailAPI Token
*/50 * * * * /path/to/your/script/refresh_token.sh
```

在設置定時任務之前，不要忘記給程式文件賦予執行權限：

```bash
chmod +x /path/to/your/script/update_targets_infos.sh
chmod +x /path/to/your/script/refresh_token.sh
```

此外，由於 crontab 的環境特殊性，你必須確保執行的 python 環境和相關套件都是正確的。

因此在程式中，我通常會使用絕對路徑來執行 python 程式，請記得要修改程式中的路徑。

```bash
# `update_targets_infos.sh` and `refresh_token.sh`

# ...以上省略

# 執行 Python 程式，要把這邊改成你自己的 python 路徑
$HOME/your/python main.py --project_name $project_name --time_length 1 2>&1

# ...以下省略
```

:::tip
crontab 不會讀取你的 `.bashrc` 或 `.bash_profile` 等文件，所以你需要在程式中指定所有的環境變數。

這也是為什麼我們會在 `crontab` 的執行程式中設置 `OPENAI_API_KEY` 環境變數的原因。
:::

## 測試 `crontab`

那麼完成設定後，該如何測試基於 crontab 環境的自動化任務呢？

一個可行的方法是：啟動一個新的終端，剔除所有的環境變數，然後執行程式。

```bash
env -i HOME=$HOME OPENAI_API_KEY=your_openai_api_key /bin/bash --noprofile --norc

# 接著執行程式
/path/to/your/script/update_targets_infos.sh
```

從這個終端執行程式，你就可以看到程式在 crontab 環境下的執行狀況。