---
slug: works-in-shell-but-fails-in-cron
title: It Works in Your Terminal. Why Does cron / systemd Kill It?
authors: Z. Yuan
tags: [linux, cron, systemd, debugging, shell]
image: /img/2026/0315-shell-vs-cron.svg
description: The command works fine in your interactive shell, then falls apart under cron or systemd. Usually the culprit is not magic. It is environment, cwd, shell, or permissions.
---

You ran it in your terminal.

It worked.

You copied the same command into cron.

It broke.

Then you moved it into systemd.

It broke more quietly.

This kind of bug is excellent for growing superstition, because on the surface it looks irrational.

The machine is not being mystical, though.

It is telling you something simple:

> **You think this is the same command. In practice, it is the same text inside a different execution context.**

This post focuses on the four usual failure points:

1. different `PATH`
2. different working directory
3. different shell
4. different permissions and environment variables

Then I will give you a debugging order that wastes less life.

<!-- truncate -->

## The short version: the command did not change, the context did

People usually say things like:

- "It is literally the same line"
- "It works in my shell"
- "It fails in the scheduler"

Those can all be true.

The missing sentence is this:

- **The user, directory, environment, and startup path are no longer the same.**

Interactive shells come with a lot of hidden help:

- login-time shell initialization
- your current working directory
- your user permissions
- exported environment variables
- aliases, functions, `pyenv`, `nvm`, `conda`, and other decorative chaos

cron and systemd tend to respond with:

- not my problem
- write it down explicitly
- and no, they will not apologize for guessing differently

## Symptom 1: `python` not found, `node` not found, `mytool` not found

This is the most common failure, and also the most boring.

You run this in a terminal:

```bash
python script.py
```

Under cron you get:

```text
python: command not found
```

That is almost always a `PATH` problem.

### Why does the interactive shell work?

Because your shell probably did a lot of setup behind the curtain:

- loaded `~/.zshrc`
- loaded `~/.bashrc`
- initialized `pyenv`
- initialized `nvm`
- added `~/.local/bin` to `PATH`

cron is not interested in reenacting your login ceremony.

A minimal cron `PATH` often looks more like this:

```text
/usr/bin:/bin
```

That means:

- `python` may not be the one you expect
- `node` may not exist at all
- user-installed CLIs may have vanished from history

### Fix: stop gambling on `PATH`

Use absolute paths:

```bash
/usr/bin/python3 /opt/jobs/report.py
```

To find the actual path from your interactive shell:

```bash
command -v python3
command -v node
command -v mytool
```

Then put that exact path in the job.

### cron example

```cron
PATH=/usr/bin:/bin:/usr/local/bin
*/10 * * * * /usr/bin/python3 /opt/jobs/report.py >> /var/log/report.log 2>&1
```

At least then success is no longer a coin toss.

## Symptom 2: the file exists, but the program says it does not

Another classic:

```python
with open("config/settings.json") as f:
    ...
```

Run manually from the project root:

```bash
python app.py
```

Everything is fine.

Run from cron:

```text
FileNotFoundError: config/settings.json
```

Congratulations. You found the **working directory** problem.

### The interactive-shell illusion

When you run commands manually, you are usually already standing in the project root.

So the relative path:

```text
config/settings.json
```

gets resolved as:

```text
/your/project/config/settings.json
```

cron does not promise that starting directory.

systemd does not either unless you set `WorkingDirectory`.

### Fix A: resolve paths from the script location

Python:

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
config_path = BASE_DIR / "config" / "settings.json"
```

That is usually much more reliable than praying the current directory is correct.

### Fix B: set `WorkingDirectory` in systemd

```ini
[Service]
WorkingDirectory=/opt/myapp
ExecStart=/usr/bin/python3 /opt/myapp/app.py
```

### Fix C: `cd` first in cron

```cron
*/5 * * * * cd /opt/myapp && /usr/bin/python3 app.py >> /var/log/myapp.log 2>&1
```

It works. I just trust it less than code that resolves absolute paths on its own.

## Symptom 3: you used bash syntax, but the job is not running in bash

Also common.

You wrote this:

```bash
source venv/bin/activate
for file in *.json; do
  echo "$file"
done
```

It behaves in your terminal.

Under cron, suddenly:

```text
source: not found
```

or some other shell-shaped insult.

### Why?

Because cron often defaults to:

```text
/bin/sh
```

not `bash`, and definitely not your carefully domesticated `zsh`.

So now:

- `source` may not exist
- `[[ ... ]]` may fail
- arrays and process substitution may explode
- shell behavior changes in small, annoying ways

### Fix

If you need bash, say so explicitly:

```cron
SHELL=/bin/bash
PATH=/usr/bin:/bin:/usr/local/bin
*/10 * * * * /bin/bash /opt/jobs/run.sh >> /var/log/run.log 2>&1
```

Better yet:

1. put the logic in a script
2. make the shebang explicit

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /opt/jobs
./venv/bin/python task.py
```

Now the shell type is not an accident.

## Symptom 4: permissions changed, so reality changed too

When you run the command manually, it may run as you.

Under automation, it may run as:

- `root`
- `www-data`
- a service account
- a restricted systemd user

Then you start seeing errors like:

```text
Permission denied
```

Or worse, quieter failures:

- writes fail silently
- credentials cannot be read
- sockets are inaccessible
- config files under your home directory no longer exist from that process perspective

### systemd makes this obvious

Suppose you have:

```ini
[Service]
User=myapp
ExecStart=/usr/bin/python3 /opt/myapp/app.py
```

That means:

- it is not running as your login user
- `HOME` may be different
- your personal shell setup is irrelevant
- files in your home directory may be off-limits

This is actually good. It is safer.

It is just less flattering during debugging.

## Before fixing anything, print the crime scene

A lot of people start editing configs immediately.

I usually start with something much duller and much more effective:

**dump the runtime environment itself.**

For example, create `debug-env.sh`:

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

Run that from cron or systemd first.

You usually get the answer very quickly:

- `pwd=/`
- `HOME=/`
- `PATH=/usr/bin:/bin`
- `python3` is not where you thought

The machine did not betray you.

You just had not asked the right question yet.

## systemd is usually easier to debug than cron

cron's biggest feature is silence.

Too much silence.

With a systemd service, I usually start here:

```bash
systemctl status myapp.service
journalctl -u myapp.service -n 100 --no-pager
```

And I make the important execution assumptions explicit in the unit file:

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

If an env file is needed:

```ini
EnvironmentFile=/etc/myapp.env
```

systemd's attitude is actually sensible:

- do not smuggle in magic
- write the runtime contract down where it can be inspected

At first this feels annoying.

Later it feels civilized.

## A safer pattern

If the job matters, I usually do not put a long inline command directly in the scheduler.

I do this instead:

1. write a dedicated script
2. use absolute paths
3. declare the shell and error mode at the top
4. set the working directory explicitly
5. send stdout and stderr to logs

For example:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /opt/report-job
export PATH=/usr/local/bin:/usr/bin:/bin

/opt/report-job/.venv/bin/python generate_report.py \
  >> /var/log/report-job.log 2>&1
```

Then cron only does one thing:

```cron
0 * * * * /opt/report-job/run.sh
```

That keeps the failure surface smaller and the debugging cheaper.

## My debugging order

If you are currently stuck in the classic "works in terminal, fails in automation" trap, this is the order I would use:

1. **check logs**: where did stderr go
2. **dump the environment**: `whoami`, `pwd`, `env`, `PATH`
3. **verify absolute paths**: binaries, scripts, config files, output dirs
4. **confirm the shell**: `sh` or `bash`
5. **confirm permissions**: who runs it, and what files can it read or write
6. **shrink the script**: reproduce with the smallest possible version

Most of the time, the first three steps are enough.

Because most of these bugs are not advanced.

They are just assumptions left undocumented.

## Closing

The sentence "I already tested it in my terminal" proves less than people think.

At best, it proves this:

- with that user
- in that shell
- from that directory
- with that environment
- at that moment

it worked.

That is still one full layer away from proving it will work under automation.

So the next time cron or systemd humiliates your perfectly innocent command, do not start with ghosts.

Start with context.

Because cron and systemd are very good at one thing:

removing assumptions one by one.

Cold, yes.

But honestly, useful.
