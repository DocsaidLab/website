---
slug: ubuntu-install-ssh
title: Setting Up SSH Server on Ubuntu
authors: Z. Yuan
tags: [ubuntu, ssh]
image: /en/img/2023/0912.webp
description: Tutorial on configuring ssh server.
---

SSH is a network protocol that allows users to securely access and manage remote servers. This time, we will set up passwordless login.

<!-- truncate -->

## Installing the OpenSSH Server

Open the terminal.

Enter the following commands to install the OpenSSH server:

```bash
sudo apt update
sudo apt install openssh-server
```

## Check SSH Server Status

Use the following command to check the status of the SSH server:

```bash
sudo systemctl status ssh
```

If you see “Active: active (running),” then the SSH server has started successfully.

## SSH Passwordless Login Setup:

### Generate SSH Key Pair on the Client Side

Open the terminal.

Enter the following command to generate an SSH key pair:

```bash
ssh-keygen
```

Follow the prompts. The default settings are usually sufficient. When prompted for a passphrase, you can simply press Enter to create a key pair without a password.

### Copy the Public Key to the Server

Use the `ssh-copy-id` command to copy the public key to the server. Replace `[username]` and `[server-ip]` with your server details.

```bash
ssh-copy-id [username]@[server-ip]
```

For example:

```bash
ssh-copy-id john@192.168.0.100
```

If your server uses a non-default SSH port (e.g., 2222), use the `-p` parameter:

```bash
ssh-copy-id -p 2222 john@192.168.0.100
```

This command will prompt you to enter the server password.

Once authentication is successful, your public key will be added to the server’s `~/.ssh/authorized_keys` file.

### Test Passwordless Login

Try SSH-ing into the server:

```bash
ssh [username]@[server-ip]
```

If everything is set up correctly, you should be able to log in to the server without a password.

## Disable Password Authentication

With SSH keys configured, you may consider disabling password authentication for added security.

This can be set in the server’s `/etc/ssh/sshd_config` file:

```bash
sudo vim /etc/ssh/sshd_config
```

Find the `PasswordAuthentication` option in the file and set it to `no`.

After completing these steps, you’re ready to enjoy using SSH!
