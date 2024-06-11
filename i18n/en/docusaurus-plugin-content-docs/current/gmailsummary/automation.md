---
sidebar_position: 9
---

# Scheduled

We expect to see the latest email summaries every morning, so we need an automated scheduling task to achieve this goal.

## Using `crontab`

To fully automate this process, we utilize the `crontab` feature of Linux to set up scheduled tasks.

This ensures that the program runs automatically at a fixed time every day, fetching new emails, generating summaries, and updating the GitHub repository.

The specific `crontab` configuration is as follows:

```bash
crontab -e
```

Then add the following content:

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

# Automatically execute the update script every morning at 6
0 6 * * * /path/to/your/script/update_targets_infos.sh
```

Before setting up scheduled tasks, don't forget to grant execution permissions to the script files:

```bash
chmod +x /path/to/your/script/update_targets_infos.sh
```

Additionally, due to the specific environment of `crontab`, you must ensure that the Python environment and related packages being executed are correct.

Therefore, in the program, we typically use absolute paths to execute Python scripts. Remember to modify the paths in the program.

```bash
# `update_targets_infos.sh`

# ...omitting above

# Execute Python program, replace this with your own python path
$HOME/your/python main.py --project_name $project_name --time_length 1 2>&1

# ...omitting below
```

:::tip
Crontab does not read your `.bashrc` or `.bash_profile` files, so you need to specify all environment variables in the program.

This is also why we set the `OPENAI_API_KEY` environment variable in the `crontab` execution program.
:::

## Testing `crontab`

So, after completing the setup, how can you test the automated tasks based on the `crontab` environment?

One feasible method is: start a new terminal, remove all environment variables, and then execute the program.

```bash
env -i HOME=$HOME OPENAI_API_KEY=your_openai_api_key /bin/bash --noprofile --norc

# Then execute the program
/path/to/your/script/update_targets_infos.sh
```

By running the program from this terminal, you can see how the program behaves in the `crontab` environment.
