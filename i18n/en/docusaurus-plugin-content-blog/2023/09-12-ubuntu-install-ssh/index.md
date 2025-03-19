---
slug: ubuntu-install-ssh
title: Set Up SSH Server on Ubuntu
authors: Z. Yuan
tags: [ubuntu, ssh]
image: /en/img/2023/0912.webp
description: Server setup and passwordless login tutorial.
---

SSH is a network protocol that allows users to securely access and manage remote servers.

This time, we’ll document the detailed steps for passwordless login.

<!-- truncate -->

## Install OpenSSH Server

Open the terminal.

Enter the following commands to install the OpenSSH server:

```bash
sudo apt update
sudo apt install openssh-server
```

## Check SSH Server Status

Use the following command to check the SSH server’s status:

```bash
sudo systemctl status ssh
```

If you see “Active: active (running),” then the SSH server has started successfully.

## SSH Passwordless Login Setup:

### Generate SSH Key Pair on the Client

Open the terminal.

Enter the following command to generate the key pair:

```bash
ssh-keygen
```

Follow the prompts. The default settings are usually sufficient. When asked for a password, simply press Enter to create a key pair without a password.

### Copy the Public Key to the Server

Use the `ssh-copy-id` command to copy the public key to the server. Replace [username] and [server-ip] with your server details.

```bash
ssh-copy-id [username]@[server-ip]
```

For example:

```bash
ssh-copy-id john@192.168.0.100
```

If the server uses a different SSH port (e.g., 2222), use the `-p` parameter:

```bash
ssh-copy-id -p 2222 john@192.168.0.100
```

This command will prompt you for the server's password.

Once verified successfully, your public key will be added to the server's `~/.ssh/authorized_keys` file.

### Test Passwordless Login

Try SSH into the server:

```bash
ssh [username]@[server-ip]
```

If everything is configured correctly, you should be able to log into the server without a password.

## Disable Password Authentication

With the SSH key, you may want to disable password authentication for increased security.

This can be configured in the server’s `/etc/ssh/sshd_config`:

```bash
sudo vim /etc/ssh/sshd_config
```

Find the `PasswordAuthentication` option in the file and set it to `no`.

After completing these steps, congratulations! You can now happily use SSH without a password.