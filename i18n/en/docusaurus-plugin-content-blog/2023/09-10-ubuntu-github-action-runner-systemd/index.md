---
slug: ubuntu-github-runner-systemd
title: Automating GitHub Runner with Systemd
authors: Z. Yuan
tags: [github-action, runner]
image: /en/img/2023/0910.webp
description: Running automatically with Ubuntu Systemd.
---

In collaborative development on GitHub, we often use private hosts to manage CI/CD workflows. GitHub provides documentation for setting up a self-hosted runner, and by following these steps, you can get the runner up and running quickly.

<!-- truncate -->

<div align="center">
<figure style={{"width": "80%"}}>
![github_set_runner](./img/github_set_runner.jpg)
</figure>
<figcaption>Documentation Screenshot</figcaption>
</div>

## The Issue

However, after setup, if the host machine is restarted for any reason, the runner does not automatically start. Often, this issue goes unnoticed until someone realizes that CI/CD jobs have stopped running, sometimes several days later.

This situation can happen repeatedly and become quite a nuisance. So, to prevent this, we need to configure the runner to start automatically on boot!

## Setup Process

To automatically run a task on system startup, we’ll use `systemd`.

1. **Create a New `systemd` Service File:**

   ```bash
   sudo vim /etc/systemd/system/actions-runner.service
   ```

2. **Paste the Following Content into the File:**

   ```bash {7-9}
   [Unit]
   Description=GitHub Action Runner
   After=network.target

   [Service]
   Type=simple
   User=your-username
   WorkingDirectory=/home/your-username/actions-runner
   ExecStart=/home/your-username/actions-runner/run.sh
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```

   Pay close attention to the following fields:

   - `User`, `ExecStart`, and `WorkingDirectory` should be updated with your specific username.

3. **Reload `systemd` to Apply the New Service Configuration:**

   ```bash
   sudo systemctl daemon-reload
   ```

4. **Enable the Service to Start on Boot:**

   ```bash
   sudo systemctl enable actions-runner.service
   ```

5. **Start the Service Manually or Reboot to Test:**

   ```bash
   sudo systemctl start actions-runner.service
   ```

With this setup, the `actions-runner` will automatically run in the background whenever your machine boots.

To stop the service, you can use the following command:

```bash
sudo systemctl stop actions-runner.service
```

:::warning
Ensure that `run.sh` has executable permissions.
:::

## Checking Service Status

With `systemd` managing the service, you can check the logs to monitor the runner’s status.

Use the following command to view logs:

```bash
sudo journalctl -u actions-runner.service -f
```

Explanation:

- `-u actions-runner.service`: Displays logs for the `actions-runner` service only.
- `-f`: Follows the log output, allowing you to monitor new entries in real-time.

If you want to check the service’s current status, use:

```bash
sudo systemctl status actions-runner.service
```

This command displays the current status of the `actions-runner` service, including whether it’s running and recent log output:

<div align="center">
<figure style={{"width": "80%"}}>
![action-service](./img/action-service.jpg)
</figure>
</div>

## Reconfiguring the Runner

If the original runner configuration is missing, this might occur when switching the repository’s visibility between Public and Private, or if the runner has been inactive for a long time. In such cases, you’ll need to reconfigure the runner.

To do this, go to the `actions-runner` directory, delete the `.runner` file, and re-run the configuration script:

```bash
./config.sh --url ... (use the new token configuration)
```

After completing the setup, restart the service to ensure everything is running smoothly.
