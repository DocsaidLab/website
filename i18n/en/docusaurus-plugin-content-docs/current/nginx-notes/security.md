---
sidebar_position: 4
---

# Security Hardening

Due to its flexibility and efficiency, Nginx is widely used in modern network environments. However, without proper security configurations, your well-set-up website could become a target for attackers.

Key security risks include:

- **Unhardened TLS configurations**: Could lead to man-in-the-middle attacks and weak encryption vulnerabilities.
- **Lack of HSTS mechanism**: Could lead to users accidentally accessing the site via HTTP, reducing the security of HTTPS.
- **Improperly configured HTTP security headers**: Could make the website vulnerable to XSS, clickjacking, and other attacks.
- **Lack of traffic limiting and DDoS protection**: Could lead to server resource misuse, even causing service disruption.

To ensure the security of the Nginx server, we must implement a series of hardening measures in the configuration.

## TLS Security Hardening

:::tip
The configurations for this section are within the `http` block, as TLS is a security extension based on the HTTP protocol. Therefore, these settings should be written in the main configuration file `/etc/nginx/nginx.conf`.
:::

TLS, or Transport Layer Security, is a cryptographic protocol used to ensure the security of network communications.

In modern network environments, HTTPS encryption has become the standard for websites, and TLS is the core technology behind HTTPS. While HTTPS encrypts the traffic between the website and visitors, many unoptimized TLS configurations still have vulnerabilities, such as:

- **Using outdated TLS protocols (e.g., TLS 1.0/1.1)** → Susceptible to attacks like BEAST and POODLE.
- **Allowing weak encryption algorithms (e.g., RC4, 3DES)** → Vulnerable to cracking.
- **Lack of OCSP Stapling** → Leads to performance degradation during certificate verification.
- **Enabling Session Tickets** → May allow attackers to reuse old encryption keys, leading to replay attacks.

To ensure website security, we must properly configure Nginx and strengthen the TLS settings to prevent potential security risks such as man-in-the-middle (MITM) attacks, certificate forgery, and weak encryption attacks.

Here are several optimizations to consider:

1. **Disable older TLS versions, allow only TLS 1.2 / 1.3**

   ```nginx
   ssl_protocols TLSv1.2 TLSv1.3;
   ```

   - **TLS 1.0 / 1.1 are no longer secure** and are no longer supported by many browsers (like Chrome and Firefox).
   - **TLS 1.2 is still secure and widely supported**.
   - **TLS 1.3 offers better performance and security**, with support for "zero-latency handshakes," making it ideal for modern applications.

---

2. **Use secure encryption algorithms**

   ```nginx
   ssl_ciphers 'ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384';
   ssl_prefer_server_ciphers on;
   ```

   These cipher suites **exclude weak encryption algorithms like RC4, 3DES, MD5**, and ensure the server prefers its own secure cipher combinations.

---

3. **Enable OCSP Stapling to improve certificate validation speed**

   ```nginx
   ssl_stapling on;
   ssl_stapling_verify on;
   resolver 8.8.8.8 1.1.1.1 valid=300s;
   resolver_timeout 5s;
   ```

   OCSP Stapling allows the server to cache the certificate status in advance, reducing the delay of querying the CA server every time a connection is made and improving performance.

---

4. **Disable Session Tickets to prevent key reuse**

   ```nginx
   ssl_session_tickets off;
   ```

   Session Tickets may lead to the reuse of keys, allowing attackers to steal tickets and reuse encrypted sessions.

---

5. **Set secure DH parameters**

   ```nginx
   ssl_dhparam /etc/nginx/ssl/dhparam.pem;
   ```

   Diffie-Hellman (DH) key exchange provides additional security, preventing certain types of attacks.

   If no DH parameter file is available, you can generate it with OpenSSL:

   ```bash
   sudo mkdir -p /etc/nginx/ssl
   sudo openssl dhparam -out /etc/nginx/ssl/dhparam.pem 2048
   ```

---

6. **Enable session caching to improve TLS connection performance**

   ```nginx
   ssl_session_cache shared:SSL:10m;
   ssl_session_timeout 1h;
   ```

   - **Allow TLS session reuse to enhance performance**.
   - **Reduce CPU load and improve user experience**.

## Advanced HSTS Configuration

:::tip
This configuration should be placed in the `server` block, as HSTS is a security mechanism specific to individual websites. Therefore, the settings below should be written in the website configuration files, such as `/etc/nginx/sites-available/example.com`.
:::

HSTS, or HTTP Strict Transport Security, is a security mechanism that forces browsers to only access websites over HTTPS.

While HTTPS has become the standard for website configurations, HTTPS alone cannot prevent downgrade attacks (Downgrade Attack) or man-in-the-middle attacks (MITM Attack). This is where HSTS becomes a key mechanism—it ensures that users are automatically redirected to HTTPS every time they connect, preventing the website from being exposed to insecure HTTP connections.

HSTS is defined by the **RFC 6797** standard and its core functions are:

- **Force the website to be accessed only via HTTPS**
- **Prevent downgrade attacks (Downgrade Attack)**
- **Block SSL stripping attacks (SSL Stripping Attack)**
- **Improve website trustworthiness and SEO ranking**

For example: if a website supports HTTPS but doesn't have HSTS enabled, attackers can perform a man-in-the-middle attack by tampering with the initial HTTP connection, forcing users to downgrade to HTTP, and intercepting all data traffic. At this point, all sensitive data such as login credentials and credit card information can be stolen, exposing users' privacy to risk.

Once HSTS is enabled:

1. The browser will remember that the website only allows HTTPS connections after the first successful connection.
2. For all future connections, the browser will directly switch to HTTPS, even if the user types `http://example.com`, it will automatically be redirected to `https://example.com`.
3. Even if an attacker intercepts the initial HTTP connection, they won't be able to downgrade the website to HTTP.

In Nginx, HSTS must only be applied to HTTPS servers, so it should be placed within the `server` block.

```nginx
server {
    listen 443 ssl http2;
    server_name example.com www.example.com;

    # Enable HSTS and apply to all subdomains
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
}
```

- The meanings of each parameter are as follows:

  - `max-age=31536000`: The duration of the HSTS setting (in seconds), here set to 1 year.
  - `includeSubDomains`: Enforces HSTS for all subdomains, preventing subdomains from being exploited by attackers.
  - `preload`: Allows the website to be added to the browser's HSTS preload list.
  - `always`: Ensures Nginx adds the HSTS header to all responses, including error pages.

---

When configuring, HSTS must only be enabled for HTTPS to ensure that HTTP connections are redirected with a 301 redirect to HTTPS, without including the HSTS header. Therefore, the following configuration should be added to the HTTP server:

```nginx
server {
    listen 80;
    server_name example.com www.example.com;

    # Redirect to HTTPS but do not include the HSTS header
    return 301 https://$host$request_uri;
}
```

This setup prevents attackers from exploiting the initial HTTP connection to bypass HSTS and ensures the browser records it correctly.

## Other Security Headers

:::tip
Since this is still related to headers, the following configurations should also be written in website configuration files such as `/etc/nginx/sites-available/example.com`.
:::

When we talk about website security, most people first think of HTTPS encryption and firewalls, but these only address security at the transport layer.

In reality, many web attacks (such as XSS, clickjacking, MIME sniffing) are executed through browser vulnerabilities, and correctly configuring HTTP security headers is one of the best ways to prevent such attacks.

HTTP security headers can effectively defend against:

- **XSS (Cross-Site Scripting)**
- **Clickjacking**
- **MIME Sniffing**
- **Cookie theft and hijacking**

Even if your website has HTTPS enabled, these headers need to be correctly configured to fully enhance security.

---

Within Nginx's `server` block, we can add these headers using the `add_header` directive.

```nginx
server {
    listen 443 ssl http2;
    server_name example.com www.example.com;

    # Enhance security
    add_header X-Frame-Options "DENY" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Permissions-Policy "geolocation=(), microphone=()" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' https://trusted-cdn.com;" always;

    root /var/www/html;
    index index.html index.htm;
}
```

Let's go over the usage of these security headers:

1. **X-Frame-Options: Prevent Clickjacking Attacks**

   ```nginx
   add_header X-Frame-Options "DENY" always;
   ```

   Prevents the website from being embedded in an `iframe`, blocking attackers from tricking users into clicking malicious buttons through hidden frames.

   The options can be `DENY` (completely disallow embedding) or `SAMEORIGIN` (allow embedding only from the same origin). It is recommended to use `DENY` unless your website needs to be embedded in an iframe.

---

2. **X-XSS-Protection: Prevent XSS Attacks**

   ```nginx
   add_header X-XSS-Protection "1; mode=block" always;
   ```

   Enables the browser's built-in XSS filter to prevent attackers from injecting malicious scripts into the site. `1; mode=block` means the browser will block the page from loading if an XSS attack is detected.

   While CSP (Content Security Policy) is a stronger XSS defense mechanism, `X-XSS-Protection` serves as an additional safeguard.

---

3. **X-Content-Type-Options: Prevent MIME Type Sniffing Attacks**

   ```nginx
   add_header X-Content-Type-Options "nosniff" always;
   ```

   Prevents browsers from guessing the MIME type for unknown content, stopping attackers from uploading malicious files (like changing `.js` to `.jpg`). Ensures all downloaded content is parsed according to the MIME type set by the server, reducing vulnerability risks.

   This is one of the mandatory security headers recommended by OWASP.

---

4. **Referrer-Policy: Protect User Privacy**

   ```nginx
   add_header Referrer-Policy "strict-origin-when-cross-origin" always;
   ```

   Controls how the `Referer` header is sent, preventing external websites from obtaining complete URL information. The setting `strict-origin-when-cross-origin` means full `Referer` is sent for same-origin requests, while only the origin (`origin`) is sent for cross-origin requests, and the `Referer` is not sent when downgrading from HTTPS to HTTP.

   This reduces privacy leakage risk while maintaining website analytics data.

---

5. **Permissions-Policy (Feature Policy): Restrict Browser Features**

   ```nginx
   add_header Permissions-Policy "geolocation=(), microphone=()" always;
   ```

   Restricts browser features available to the website, preventing abuse of user permissions (such as access to camera, microphone, or geolocation). The above configuration disables access to geolocation and microphone.

   This is critical for modern privacy protection, especially for websites subject to GDPR or privacy regulations.

---

6. **Content-Security-Policy (CSP): The Strongest Defense Against XSS**

   ```nginx
   add_header Content-Security-Policy "default-src 'self'; script-src 'self' https://trusted-cdn.com;" always;
   ```

   CSP is the strongest defense against XSS, limiting the sources from which resources can be loaded on the site. `default-src 'self'` means only resources from the same origin are allowed by default. `script-src 'self' https://trusted-cdn.com;` means only JavaScript from the same site and the specified trusted CDN are allowed.

   CSP effectively prevents XSS attacks and enhances the overall security of the website.

## Main Configuration File Setup

In the previous chapters, we reviewed the main configuration file but skipped modifying it for now.

Now, let's take a closer look. Generally speaking, global configurations such as TLS settings, HTTP security headers, and request rate limits should be placed in `nginx.conf`, while site-specific configurations such as HSTS and redirects to HTTPS should go into site configuration files like `sites-available/default`.

Based on the discussion earlier, we can make some adjustments to the default `nginx.conf` to suit our needs.

```nginx title="/etc/nginx/nginx.conf"
user www-data;
worker_processes auto;
pid /run/nginx.pid;

# Error log settings
error_log /var/log/nginx/error.log warn;
include /etc/nginx/modules-enabled/*.conf;

events {
    worker_connections 1024;  # Increase the maximum number of concurrent connections (default is 768)
    multi_accept on;          # Allow worker processes to accept multiple new connections simultaneously
}

http {
    ##
    # Basic settings
    ##
    sendfile on;
    tcp_nopush on;
    types_hash_max_size 2048;
    server_tokens off;  # Hide Nginx version info to prevent attack information leakage

    ##
    # SSL settings
    ##
    ssl_protocols TLSv1.2 TLSv1.3;  # Disable TLS 1.0/1.1 for increased security
    ssl_ciphers 'ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384';
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;
    ssl_session_tickets off;
    ssl_stapling on;
    ssl_stapling_verify on;
    resolver 8.8.8.8 1.1.1.1 valid=300s;
    resolver_timeout 5s;
    ssl_dhparam /etc/nginx/ssl/dhparam.pem;  # Enhance DH parameters (manual generation required)

    ##
    # Log settings
    ##
    log_format main '$remote_addr - $remote_user [$time_local] '
                    '"$request" $status $body_bytes_sent '
                    '"$http_referer" "$http_user_agent"';
    access_log /var/log/nginx/access.log main;

    ##
    # Gzip compression
    ##
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_buffers 16 8k;
    gzip_http_version 1.1;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    ##
    # HTTP security headers (apply site-wide, but can be overridden by individual servers)
    ##
    add_header X-Frame-Options "DENY" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Permissions-Policy "geolocation=(), microphone=()" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' https://trusted-cdn.com;" always;

    ##
    # Virtual host configurations
    ##
    include /etc/nginx/conf.d/*.conf;
    include /etc/nginx/sites-enabled/*;

    ##
    # Global DDoS defense variables (specific settings within `server` block)
    ##
    map $http_user_agent $bad_bot {
        default 0;
        "~*(nikto|curl|wget|python)" 1;  # Block malicious crawlers and scanning tools
    }

    limit_req_zone $binary_remote_addr zone=general:10m rate=5r/s;  # Limit requests per IP to 5 requests per second
    limit_conn_zone $binary_remote_addr zone=connlimit:10m;         # Limit simultaneous connections per IP
}
```

## Site Configuration File

In the site configuration files, we can further configure security settings specific to individual websites.

Below is a simple example; in practice, you may need to configure more settings depending on the characteristics of the website.

```nginx title="/etc/nginx/sites-available/example.com"
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name example.com www.example.com;

    # Force all HTTP traffic to be redirected to HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;

    server_name example.com www.example.com;

    root /var/www/html;
    index index.html index.htm;

    ##
    # SSL Certificate (Let's Encrypt)
    ##
    ssl_certificate /etc/letsencrypt/live/example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;

    ##
    # HSTS (Only includeSubDomains if all subdomains are HTTPS enabled)
    ##
    add_header Strict-Transport-Security "max-age=31536000; preload" always;

    ##
    # Site-specific DDoS defense
    ##
    limit_req zone=general burst=10 nodelay;
    limit_conn connlimit 20;

    ##
    # Main route handling
    ##
    location / {
        try_files $uri $uri/ =404;
    }

    ##
    # PHP parsing (Only enable on sites that require PHP to avoid security issues)
    ##
    location ~ \.php$ {
        try_files $uri =404;
        include snippets/fastcgi-php.conf;
        fastcgi_pass unix:/run/php/php7.4-fpm.sock;
        fastcgi_param SCRIPT_FILENAME $document_root$fastcgi_script_name;
        include fastcgi_params;
    }

    ##
    # Deny access to hidden files and sensitive files (preserve Let's Encrypt)
    ##
    location ~ /\.(?!well-known).* {
        deny all;
    }

    location ^~ /.well-known/acme-challenge/ {
        allow all;
    }
}
```

---

Another example, based on the API deployment scenario we discussed earlier, assumes our API endpoint is:

- `https://temp_api.example.com/test`

We can configure Nginx to secure this API endpoint:

```nginx title="/etc/nginx/sites-available/temp_api.example.com"
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name temp_api.example.com;

    # Redirect all HTTP requests to HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name temp_api.example.com;

    ##
    # SSL certificate paths depend on your certificate location
    ##
    ssl_certificate /etc/letsencrypt/live/temp_api.example.com/fullchain.pem; # Managed by Certbot
    ssl_certificate_key /etc/letsencrypt/live/temp_api.example.com/privkey.pem; # Managed by Certbot

    ##
    # HTTP Security Headers
    ##
    add_header Strict-Transport-Security "max-age=63072000; includeSubdomains; preload" always;

    ##
    # Limit request body size (Prevent DDoS)
    ##
    client_max_body_size 10M;

    ##
    # Site-specific request limits
    ##
    limit_req zone=general burst=10 nodelay;
    limit_conn connlimit 20;

    ##
    # Proxy requests to FastAPI app
    ##
    location /test {
      proxy_pass http://127.0.0.1:8000;
      proxy_http_version 1.1;
      proxy_set_header Host $host;
      proxy_set_header X-Real-IP $remote_addr;
      proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
      proxy_set_header X-Forwarded-Proto $scheme;
      proxy_cache_bypass $http_cache_control;
      proxy_no_cache $http_cache_control;

      # Let Nginx handle FastAPI's error responses
      proxy_intercept_errors on;

      # Restrict to only allow GET, POST, HEAD
      limit_except GET POST HEAD {
          deny all;
      }
    }

    ##
    # Block malicious User-Agents
    ##
    if ($bad_bot) {
        return 403;  # Block common scanners' User-Agent
    }
}
```

## Conclusion

As the saying goes, "The higher the road, the taller the devil." In the online environment, security is always an ongoing issue.

While these security measures cannot guarantee absolute safety, they can significantly reduce the risk of attacks on your website:

> **Attackers tend to choose softer targets. When they see that your website has security measures in place, they may not waste time attacking it and instead look for easier targets. (If your website is of exceptionally high value, that may be a different case.)**

For regular users, enabling these security measures not only protects the website but also improves SEO rankings and user trust, making it a worthwhile investment.

Although the entire process can be tedious and may leave you feeling frustrated and exhausted, don't give up. The internet environment is full of malicious actors, and a small lapse in attention can lead to significant losses.

Take a break for now, and in the next chapter, we will learn how to use Fail2Ban for Nginx security monitoring.
