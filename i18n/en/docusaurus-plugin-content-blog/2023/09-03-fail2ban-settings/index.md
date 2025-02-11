---
slug: fail2ban-settings
title: "Fail2ban: Protecting SSH Service"
authors: Z. Yuan
tags: [ubuntu, fail2ban]
image: /en/img/2023/0903.webp
description: Keeping the malicious out.
---

As soon as you successfully open an external SSH channel, you'll notice a barrage of malicious connections attempting to log into your host.

<!-- truncate -->

<div align="center">
<figure style={{"width": "40%"}}>
![attack from ssh](./img/ban_1.jpg)
</figure>
<figcaption>Malicious Attack Illustration</figcaption>
</div>

---

Common practice is to use Fail2ban to protect our host. It's software designed to protect servers from brute force attacks.

It automatically adjusts firewall rules to block attackers' IP addresses when suspicious behavior, such as repeated login failures, occurs.

## 1. Installation of Fail2ban

On most Linux distributions, you can install Fail2ban using package management tools.

On Ubuntu, you can use apt to install:

```bash
sudo apt install fail2ban
```

## 2. Configuration

The configuration file is located at `/etc/fail2ban/jail.conf`.

It's recommended not to modify this file directly but to make a copy named `jail.local` and edit it:

```bash
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
```

Edit `jail.local`:

```bash
sudo vim /etc/fail2ban/jail.local
```

**Important configuration parameters:**

- **ignoreip:** Ignored IP addresses or networks, e.g., 127.0.0.1/8
- **bantime:** Duration of the ban in seconds (default is 600 seconds)
- **findtime:** How many failures within this time frame (default is 600 seconds)
- **maxretry:** Maximum number of failed attempts allowed within findtime.

## 3. Start and Monitor

Start Fail2ban:

```bash
sudo service fail2ban start
```

Check Fail2ban's status:

```bash
sudo fail2ban-client status
```

## 4. Adding Custom Rules

If you want to set specific rules for particular services (e.g., SSH or Apache), you can add or modify corresponding sections in `jail.local`, for example, SSH settings:

```bash
[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
```

## 5. Testing

After making configuration changes, restart Fail2ban to apply the changes:

```bash
sudo service fail2ban restart
```

Then, from another machine or using a different IP, attempt multiple failed logins to see if it gets blocked.

## 6. Review

Ensure to periodically check log files and update rules for the best protection.

```bash
sudo fail2ban-client status sshd
```

## Conclusion

The entire process is somewhat meticulous but not overly complex.

Hopefully, this guide helps you smoothly complete the setup.
