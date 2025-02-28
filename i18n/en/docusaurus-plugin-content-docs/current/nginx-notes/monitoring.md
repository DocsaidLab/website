---
sidebar_position: 5
---

# Nginx Monitoring

Once we expose the API to the public, we immediately start receiving malicious requests from all directions.

If you have looked at the log records, you should not have trouble noticing that various IP addresses are repeatedly attempting to log into your server with different usernames and passwords. This is a common brute force attack, where attackers use automated tools to continuously try logging in until they find the correct password.

Although we can make the password complexity sky-high, constantly worrying about this is not a solution.

So, we use the tool Fail2ban to block those IP addresses.

## What is Fail2ban?

Fail2ban is an open-source intrusion prevention tool, mainly used to monitor server logs and block malicious IPs based on predefined rules. It effectively defends against brute force attacks, such as malicious behaviors targeting SSH, HTTP, and other network services.

The operation of Fail2ban is based on two key concepts: **filters** and **jails**:

- **Filters**: Used to define suspicious behavior patterns that should be looked for in the logs.
- **Jails**: Responsible for combining filters with blocking mechanisms. When abnormal behavior is detected, jails automatically perform the corresponding action.

In the context of Nginx, Fail2ban monitors Nginx's access log and error log to identify suspicious requests, such as:

- **Frequent 404 errors** (potential scanning attacks)
- **Continuous login failures** (brute force attacks)
- **High-frequency requests within a short time** (DDoS or malicious bots)

Once abnormal behavior is detected, Fail2ban, based on the configuration, automatically blocks the source IP via iptables or nftables, preventing further attacks on the Nginx server.

In an Ubuntu environment, the Nginx logs are stored by default at:

- **Access log**: `/var/log/nginx/access.log`
- **Error log**: `/var/log/nginx/error.log`

Fail2ban can be configured to monitor these logs and match the log content with predefined or custom filter rule templates. When log entries match malicious behavior patterns, Fail2ban adds the related IPs to the blocklist.

:::info
Rule templates are usually located in `/etc/fail2ban/filter.d/`
:::

:::tip
In addition to protecting Nginx, Fail2ban can also be applied to SSH, FTP, mail servers, and other services, providing extensive server security protection. This guide mainly focuses on defense configurations related to Nginx.
:::

## Basic Fail2ban Configuration

On Ubuntu, Fail2ban can be installed directly via the APT package manager, as it is included in the official software repository. Follow these steps to install and start Fail2ban:

1. **Update the system package repository**:
   ```bash
   sudo apt update
   ```
2. **Install Fail2ban**:

   ```bash
   sudo apt install -y fail2ban
   ```

   This will automatically download and install Fail2ban and its dependencies. After installation, Fail2ban will start automatically and be set to run on boot.

3. **Check the status of the Fail2ban service**:

   ```bash
   sudo systemctl status fail2ban
   ```

   If the installation is successful, it should show `active (running)`, indicating that Fail2ban is running correctly.

4. **Check the Fail2ban version and status**:

   - Display the currently installed Fail2ban version:

     ```bash
     fail2ban-client --version
     ```

   - Check the currently enabled jails:

     ```bash
     sudo fail2ban-client status
     ```

     By default, Fail2ban may only have SSH protection enabled. Additional configuration is required for Nginx-related defenses.

---

The main configuration file for Fail2ban is `/etc/fail2ban/jail.conf`, but it is recommended by the official documentation: **"Do not modify this file directly"** to avoid overwriting your settings during software updates.

Therefore, we create a local configuration file `jail.local` to override the default values.

1. **Copy `jail.conf` to `jail.local`**:

   ```bash
   sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
   ```

2. **Edit `jail.local`**:
   ```bash
   sudo vim /etc/fail2ban/jail.local
   ```

Find the `[DEFAULT]` section and set key parameters, such as trusted IPs, ban time, monitoring time, and maximum retry attempts:

```ini
[DEFAULT]
ignoreip = 127.0.0.1/8 ::1    # Set trusted IP addresses that will not be blocked
bantime  = 1h                 # Default ban time, can be set in seconds or time units (e.g., 1h, 10m)
findtime = 10m                # Within this period...
maxretry = 5                  # If failed more than maxretry times, the IP will be banned
```

The meanings of these settings are as follows:

- **ignoreip**: Specifies which IPs will not be blocked, such as the local IP (`127.0.0.1`) or other management IPs.
- **bantime**: The duration of the ban, by default 1 hour (`1h`). During this time, the IP will be blocked from accessing the server.
- **findtime**: The observation time range, for example, `10m` means that failed attempts within 10 minutes will be counted.
- **maxretry**: The maximum number of retries, for example, `5` means that the same IP will be banned after triggering the rule 5 times within `findtime`.

These parameters are global settings and apply to all jails. We will later configure a separate blocking strategy for Nginx to ensure more effective interception of malicious attacks.

Once the settings are complete, restart Fail2ban to apply the new configuration:

```bash
sudo systemctl restart fail2ban
```

At this point, the basic installation and global configuration of Fail2ban have been completed.

Next, we will further configure Fail2ban to monitor Nginx logs to prevent malicious requests.

## Monitoring Common Attack Types

Here, we will configure Fail2ban to monitor and automatically block several common attack types against Nginx. Each type of attack usually has clear characteristics in the Nginx logs, so we first define "filter rules" to detect them and apply the corresponding blocking strategies.

### Preventing Malicious Crawlers

Many malicious crawlers or attack tools use specific User-Agent strings, such as:

- `Java`
- `Python-urllib`
- `Curl`
- `sqlmap`
- `Wget`
- `360Spider`
- `MJ12bot`

These User-Agents mostly come from automated scanning tools and do not belong to legitimate users. Therefore, we can block such User-Agents in real-time.

First, create a new filter file in the `/etc/fail2ban/filter.d/` directory:

```bash
sudo vim /etc/fail2ban/filter.d/nginx-badbots.conf
```

Add the following content:

```ini
[Definition]
failregex = <HOST> -.* "(GET|POST).*HTTP.*" .* "(?:Java|Python-urllib|Curl|sqlmap|Wget|360Spider|MJ12bot)"
ignoreregex =

# failregex is the regular expression used to match malicious requests
# - <HOST> represents the IP address, and the placeholder is parsed by Fail2Ban
# - "(GET|POST).*HTTP.*" restricts to HTTP GET or POST requests
# - .* matches the request content
# - "(?:Java|Python-urllib|Curl|sqlmap|Wget|360Spider|MJ12bot)"
#   - These are common crawlers, scanners, or attack tools:
#     - Java: Typically from Java-based crawlers
#     - Python-urllib: Python's built-in URL request library
#     - Curl: Command-line HTTP request tool
#     - sqlmap: Automated SQL injection tool
#     - Wget: Download tool, also used for crawling websites
#     - 360Spider: 360 Search engine crawler
#     - MJ12bot: Majestic SEO crawler

# ignoreregex is used to exclude certain requests with regular expressions
# - Leave it empty to not exclude any requests
```

Then configure the Jail by adding the following to the `jail.local` file:

```ini
[nginx-badbots]
enabled  = true
port     = http,https
filter   = nginx-badbots
logpath  = /var/log/nginx/access.log
maxretry = 1
bantime  = 86400
```

This will immediately block any requests with these User-Agents for 24 hours.

### Preventing 404 Scan Attacks

This type of attack involves brute-forcing nonexistent pages, typically generating a large number of requests within a short period.

Attackers may use scripts to repeatedly access non-existent pages on the website in an attempt to discover vulnerabilities or sensitive files, causing a large number of HTTP 404 errors.

Create a new filter file:

```bash
sudo vim /etc/fail2ban/filter.d/nginx-404.conf
```

Add the following content:

```ini
[Definition]
failregex = <HOST> -.* "(GET|POST).*HTTP.*" 404
ignoreregex =

# failregex is the regular expression used to match specific log patterns
# - <HOST> represents the IP address (Fail2Ban will automatically replace it with the actual source IP)
# - "(GET|POST).*HTTP.*":
#   - Restricts to GET or POST requests
#   - .* matches the URL and HTTP protocol version (e.g., HTTP/1.1)
# - "404" specifies matching HTTP 404 status code (indicating the requested resource was not found)

# This rule is used to block IPs that frequently trigger 404 errors, such as crawlers or attackers scanning non-existent pages
# Suitable for web servers (such as Nginx, Apache) log analysis

# ignoreregex is used to exclude certain requests from matching
# - Leave it empty to not exclude any requests
```

Then configure the Jail by adding the following to the `jail.local` file:

```ini
[nginx-404]
enabled  = true
port     = http,https
filter   = nginx-404
logpath  = /var/log/nginx/access.log
findtime = 10m
maxretry = 5
bantime  = 86400
```

This will block any IPs that trigger 5 HTTP 404 errors within 10 minutes for 24 hours.

### Preventing DDoS Attacks

When an IP sends a large number of requests in a short period, it could be a DDoS attack or malicious crawling.

Create a new filter file:

```bash
sudo vim /etc/fail2ban/filter.d/nginx-limitreq.conf
```

Add the following content:

```ini
[Definition]
failregex = limiting requests, excess: .* by zone .* client: <HOST>
ignoreregex =

# failregex is used to match rate limiting events in Nginx logs
# - "limiting requests, excess: .* by zone .* client: <HOST>"
#   - "limiting requests, excess:" indicates that the request exceeded the configured limit (e.g., Nginx's limit_req_zone limit)
#   - .* allows matching any data (such as the number of requests or details of the excess)
#   - "by zone .*" represents the rate limiting zone configured in Nginx
#   - "client: <HOST>" is where Fail2Ban will replace with the actual client IP address
#
# ignoreregex is used to exclude specific matching patterns
# - Leave it empty to exclude no requests
```

This `failregex` is primarily used to match Nginx rate-limiting (limit_req_zone) log entries, such as:

```log
2024/02/15 12:34:56 [error] 1234#5678: *90123 limiting requests, excess: 20.000 by zone "api_limit" client: 192.168.1.100, server: example.com, request: "GET /api/v1/data HTTP/1.1"
```

The corresponding Nginx configuration might look like:

```nginx
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
server {
    location /api/ {
        limit_req zone=api_limit burst=20 nodelay;
    }
}
```

With this setup, when an IP sends too many requests in a short period, Nginx will log the message `"limiting requests, excess: ... client: <IP>"`, and Fail2Ban will use the `failregex` to match and block that IP.

Next, configure the Jail by adding the following to `jail.local`:

```ini
[nginx-limitreq]
enabled  = true
port     = http,https
filter   = nginx-limitreq
logpath  = /var/log/nginx/error.log
findtime = 10m
maxretry = 10
bantime  = 86400
```

This will block any IPs that trigger rate limiting more than 10 times within 10 minutes for 24 hours.

### Preventing Brute Force Login Attacks

If your site provides a login feature, attackers might try brute-forcing the username and password.

Create a new filter file:

```bash
sudo vim /etc/fail2ban/filter.d/nginx-login.conf
```

Add the following content:

```ini
[Definition]
failregex = <HOST> -.* "(POST|GET) /(admin|wp-login.php) HTTP.*"
ignoreregex =

# failregex is used to match malicious requests targeting common admin login pages
# - <HOST> represents the IP address (Fail2Ban will automatically replace it with the actual source IP)
# - "(POST|GET) /(admin|wp-login.php) HTTP.*"
#   - "(POST|GET)" indicates the HTTP method (attackers might try to log in with either GET or POST)
#   - "/admin" is a common path for site admin dashboards
#   - "/wp-login.php" is the WordPress login page
#   - "HTTP.*" is used to match different HTTP versions, such as HTTP/1.1 or HTTP/2
#
# ignoreregex is used to exclude specific requests
# - Leave it empty to exclude no requests
```

This rule will match login attempts to WordPress or typical website admin login pages.

Next, configure the Jail by adding the following to `jail.local`:

```ini
[nginx-login]
enabled  = true
port     = http,https
filter   = nginx-login
logpath  = /var/log/nginx/access.log
findtime = 5m
maxretry = 5
bantime  = 86400
```

This will block any IPs that fail to log in more than 5 times within 5 minutes for 24 hours.

### Attempting to Access Sensitive Data

Malicious attackers may attempt to access sensitive paths such as `/etc/passwd`, `/.git`, or `/.env`.

Create a new filter file:

```bash
sudo vim /etc/fail2ban/filter.d/nginx-sensitive.conf
```

Add the following content:

```ini
[Definition]
failregex = <HOST> -.* "(GET|POST) /(etc/passwd|\.git|\.env) HTTP.*"
ignoreregex =

# failregex is used to match malicious requests attempting to access sensitive files
# - <HOST> represents the IP address (Fail2Ban will automatically replace it with the actual source IP)
# - "(GET|POST) /(etc/passwd|\.git|\.env) HTTP.*"
#   - "(GET|POST)" indicates the HTTP method, attackers may try GET or POST to probe files
#   - "/etc/passwd" is a password file on Linux systems, attackers may attempt to read it
#   - "/.git" is a directory that may contain source code and sensitive information, attackers may try to download the `.git` folder
#   - "/.env" is an environment variables file that may contain **database passwords, API keys, secrets**
#   - "HTTP.*" matches different HTTP versions (e.g., HTTP/1.1, HTTP/2)
#
# ignoreregex is used to exclude specific requests
# - Leave it empty to exclude no requests
```

This rule is designed to prevent directory traversal attacks, protect Git directories, block attempts to probe `.env` files, and guard against web scanners.

Next, configure the Jail by adding the following to `jail.local`:

```ini
[nginx-sensitive]
enabled  = true
port     = http,https
filter   = nginx-sensitive
logpath  = /var/log/nginx/access.log
maxretry = 1
bantime  = 86400
```

This will immediately block any IP attempting to access these sensitive files for 24 hours.

### Attacks Targeting APIs

:::tip
Remember to adjust the filter based on your specific API endpoints.
:::

For brute force attacks targeting API endpoints like `/api/login`, `/api/register`, it is important to limit request frequency.

Create a new filter file:

```bash
sudo vim /etc/fail2ban/filter.d/nginx-api.conf
```

Add the following content:

```ini
[Definition]
failregex = <HOST> -.* "(POST) /api/(login|register) HTTP.*"
ignoreregex =

# failregex is used to match malicious requests targeting API login and registration
# - <HOST> represents the IP address (Fail2Ban will automatically replace it with the actual source IP)
# - "(POST) /api/(login|register) HTTP.*"
#   - "(POST)" restricts to HTTP POST requests (to prevent brute force login or abuse of registration)
#   - "/api/login" is the login API, which may become a target for brute force attacks
#   - "/api/register" is the registration API, which attackers may try to abuse for mass account creation (e.g., bot registrations)
#   - "HTTP.*" matches HTTP versions (e.g., HTTP/1.1, HTTP/2)
#
# ignoreregex is used to exclude specific requests
# - Leave it empty to exclude no requests
```

This rule is designed to prevent brute force login attacks, automated registration abuse, and enhance API security.

Next, configure the Jail by adding the following to `jail.local`:

```ini
[nginx-api]
enabled  = true
port     = http,https
filter   = nginx-api
logpath  = /var/log/nginx/access.log
findtime = 1m
maxretry = 10
bantime  = 86400
```

This will block any IP that exceeds 10 login attempts within 1 minute for 24 hours.

### Apply New Rules

Once the configuration is complete, restart Fail2ban:

```bash
sudo systemctl restart fail2ban
```

And check the status of the jails:

```bash
sudo fail2ban-client status
```

If you want to check the status of a specific jail, use:

```bash
sudo fail2ban-client status nginx-404 # Replace with your jail name
```

At this point, we have configured Fail2ban to defend against common attack types targeting Nginx, ensuring the security of the website.

## Testing Defense Effectiveness

After completing the Fail2ban Nginx defense configuration, we will perform a simple test.

### Simulate Attacks Using `curl`

You can use `curl` on the local machine or another computer to send malicious requests to trigger Fail2ban's blocking mechanism.

Note: Do not send requests from `localhost` (127.0.0.1) during testing, as `127.0.0.1` may be excluded by the `ignoreip` setting, and Fail2ban will not block it. You can use another computer or server in a different network environment to perform the test, or temporarily clear the `ignoreip` setting.

- **1. Test 404 Scan Interception**

  Simulate an attacker randomly scanning for non-existent pages:

  ```bash
  for i in $(seq 1 6); do
      curl -I http://<your-server-IP-or-domain>/nonexistentpage_$i ;
  done
  ```

  This will consecutively request 6 non-existent pages, generating HTTP 404 errors. If your rule is set to block after 5 404 errors, Fail2ban should block the IP on the 5th or 6th attempt.

  After the block, running the `curl` command again should result in a failed connection (e.g., timeout or connection refused).

---

- **2. Test Sensitive URL Interception**

  Attackers often try to access system-sensitive files, such as `/etc/passwd`, to check if the server has vulnerabilities:

  ```bash
  curl -I http://<your-server>/etc/passwd
  ```

  If Fail2ban's `nginx-sensitive` filter is configured correctly, this request should be detected and the IP should be immediately blocked (if `maxretry = 1`).

---

- **3. Test User-Agent Interception**

  Simulate a malicious crawler tool, such as `sqlmap`:

  ```bash
  curl -A "sqlmap/1.5.2#stable" http://<your-server>/
  ```

  If your rule includes the `sqlmap` User-Agent, Fail2ban should immediately block the IP.

---

- **4. Test Brute Force Login Attack**

  Send multiple requests to WordPress or other login pages to simulate an attacker attempting brute-force login:

  ```bash
  for i in $(seq 1 6); do
      curl -X POST -d "username=admin&password=wrongpassword" http://<your-server>/wp-login.php ;
  done
  ```

  If your `nginx-login` filter is configured to block after 5 failed attempts, Fail2ban should block the IP on the 5th or 6th attempt.

### Verify Status and Logs

You can check the status of the jail using the following command. For example, to check the `nginx-sensitive` jail:

```bash
sudo fail2ban-client status nginx-sensitive
```

Expected output:

```
Status for the jail: nginx-sensitive
|- Filter
|  |- Currently failed: 0
|  |- Total failed: 9
|  `- File list: /var/log/nginx/access.log /var/log/nginx/error.log
`- Actions
   |- Currently banned: 1
   |- Total banned: 2
   `- Banned IP list: 203.0.113.45
```

- **Currently banned**: The number of currently banned IPs.
- **Total failed**: The total number of malicious requests detected.
- **Banned IP list**: Shows the current banned IPs (e.g., `203.0.113.45`).

---

Next, you can check Fail2ban's log to confirm the block records:

```bash
sudo tail -n 20 /var/log/fail2ban.log
```

Expected output example:

```
fail2ban.actions [INFO] Ban 203.0.113.45 on nginx-sensitive
```

This indicates that the IP `203.0.113.45` was banned for triggering the `nginx-sensitive` rule.

### Unblocking an IP

During testing or regular management, you might need to manually block or unblock a specific IP.

You can manually block an IP:

```bash
sudo fail2ban-client set nginx-sensitive banip 203.0.113.45
```

This will immediately block `203.0.113.45`, and it will be added to the `nginx-sensitive` block list.

If you find that an IP has been mistakenly blocked, you can unblock it:

```bash
sudo fail2ban-client set nginx-sensitive unbanip 203.0.113.45
```

After running this command, you can verify whether the IP has been removed from the block list by using:

```bash
sudo fail2ban-client status nginx-sensitive
```

---

Finally, you can check the firewall rules (`iptables` / `nftables`) to ensure they are correctly set.

Fail2ban primarily uses `iptables` (or `nftables`) to block IPs, so you can directly check the firewall rules:

```bash
sudo iptables -L -n --line-numbers
```

If the block rule is active, you should see something like:

```
Chain f2b-nginx-sensitive (1 references)
num  target     prot opt source               destination
1    REJECT     all  --  203.0.113.45          0.0.0.0/0   reject-with icmp-port-unreachable
```

This indicates that `203.0.113.45` is blocked, and all traffic from this IP is being rejected.

## Conclusion

With this, we have completed the basic Fail2ban setup for Nginx, setting up defenses against common attacks.

Be sure to adjust the parameters according to your actual needs, such as splitting different types of attacks into separate jails, setting different `maxretry`/`findtime`, and carefully configuring the `ignoreip` whitelist to avoid mistakenly blocking critical IPs.

After deployment, it is important to continue monitoring Fail2ban logs to ensure it is functioning properly and effectively defending against malicious attacks.
