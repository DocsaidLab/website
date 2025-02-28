---
sidebar_position: 3
---

# Nginx HTTPS Settings

In today's internet environment, website security has become a crucial concern, and SSL/TLS certificates are essential technologies to protect data transmission between websites and users.

Let's Encrypt is a free, automated, and open Certificate Authority (CA) that provides encryption protection for millions of websites worldwide, making HTTPS more accessible.

Since it’s free, we definitely have to support it, right?

So, we’ll configure HTTPS on the Nginx server using Let's Encrypt.

The goals of this section include:

- **Obtain an SSL certificate**: Use Let's Encrypt’s free certificate to encrypt website traffic.
- **Enforce HTTPS traffic**: Automatically redirect all HTTP requests to HTTPS (301 permanent redirect).
- **Nginx Reverse Proxy**: Ensure the front-end SSL works correctly with Nginx as a reverse proxy.
- **Automatic Renewal**: Set up an automatic certificate renewal mechanism to prevent certificate expiration after 90 days.

:::tip
In the previous chapter, we briefly introduced the pros and cons of Let's Encrypt.

If you’re not familiar with it, you can first check out our supplementary document: [**Let's Encrypt**](./supplementary/about-lets-encrypt.md)
:::

## Installing Let's Encrypt

We use Let's Encrypt's ACME protocol to obtain SSL certificates.

First, we need to install Certbot, the recommended ACME client from Let's Encrypt. Certbot automatically interacts with Let's Encrypt to request and renew certificates.

For Debian/Ubuntu, you can install it directly from the package repository:

```bash
sudo apt update
sudo apt install -y certbot python3-certbot-nginx
```

This command will install Certbot along with the Nginx plugin, allowing Certbot to automatically modify Nginx configuration files to deploy the certificate.

Before proceeding, ensure the following preparations are complete:

- **You have a domain name** and its DNS records are pointing to your server's IP address.
- **You have opened ports 80 and 443 in your firewall**, allowing HTTP/HTTPS traffic for verification and subsequent access.

## Requesting a Let's Encrypt Certificate

With Certbot, we can request a certificate from Let's Encrypt.

Let's Encrypt uses an automated ACME validation mechanism, requiring you to prove control over the domain. A common validation method is HTTP validation (Certbot will place a temporary file on your site for Let's Encrypt to validate) or DNS validation (more complex, used for wildcard certificate requests).

Here, we'll use simple HTTP validation and let the Nginx plugin configure it automatically:

```bash
# Stop Nginx (may be required in webroot mode, but not needed with --nginx plugin)
# sudo systemctl stop nginx

# Run Certbot with the Nginx plugin to request a certificate
sudo certbot --nginx -d your-domain.com -d www.your-domain.com
```

Replace `your-domain.com` and `www.your-domain.com` with your actual domain name.

Once executed, Certbot will communicate with the Let's Encrypt server:

- **Domain Validation**: Certbot might temporarily modify the Nginx configuration or start a temporary service to respond to Let's Encrypt’s validation request. For example, Let's Encrypt will attempt to access a validation file via `http://your-domain.com/.well-known/acme-challenge/`. If the validation is successful, it confirms you control the domain.
- **Obtain Certificate**: Once validated, Let's Encrypt will issue an SSL certificate (including the full certificate chain `fullchain.pem` and the private key file `privkey.pem`). Certbot will store these files in the default path (usually `/etc/letsencrypt/live/your-domain.com/`).

During the process, Certbot might ask:

- Whether you agree to the terms of service and provide an email address (for renewal notifications).
- Whether to automatically redirect HTTP traffic to HTTPS. **It’s recommended to choose redirect** so Certbot can automatically set up the 301 redirect rule for you to ensure users access the site via HTTPS.

:::tip
If you're not using the `--nginx` plugin, you can also use the `certbot certonly --webroot` mode to manually obtain the certificate and then edit the Nginx configuration yourself. Using `--nginx` saves you from having to manually configure the settings.
:::

## Configuring Nginx to Enable HTTPS

Once the certificate has been issued, we need to enable HTTPS (SSL) in Nginx and load the certificate we just obtained.

Here’s an example configuration (assuming the default installation path for Certbot):

```nginx
# In Nginx's configuration file (e.g., /etc/nginx/sites-available/your-domain.conf):

# HTTP service on port 80, redirect all requests to HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name your-domain.com www.your-domain.com;
    # Redirect HTTP requests permanently to HTTPS
    return 301 https://$host$request_uri;
}

# HTTPS service on port 443
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name your-domain.com www.your-domain.com;

    # SSL certificate file paths (issued by Certbot)
    ssl_certificate      /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key  /etc/letsencrypt/live/your-domain.com/privkey.pem;

    # (Reverse proxy configuration example)
    location / {
        proxy_pass http://127.0.0.1:8000;  # Forward requests to the backend application (e.g., FastAPI running on port 8000)
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_intercept_errors on;
    }
}
```

Explanation of each part:

- **HTTP service on port 80 (force redirect to HTTPS)**

  ```nginx
  server {
      listen 80;
      listen [::]:80;
      server_name your-domain.com www.your-domain.com;
      # Redirect HTTP requests permanently to HTTPS
      return 301 https://$host$request_uri;
  }
  ```

  **Purpose of this block:**

  1. **Listen on HTTP (port 80)**

     - `listen 80;` : Nginx listens for IPv4 connections on port 80 (standard HTTP connection).
     - `listen [::]:80;` : Allows IPv6 connections.

  2. **Set `server_name`**

     - `server_name your-domain.com www.your-domain.com;`
     - This tells Nginx that this configuration applies to `your-domain.com` and `www.your-domain.com`.
     - Ensures that both the `www` and non-`www` versions of the domain are correctly handled.

  3. **Force HTTP to HTTPS redirection**

     - `return 301 https://$host$request_uri;`
     - This will redirect all HTTP requests to HTTPS using a 301 permanent redirect.
     - `$host` represents the requested host (`your-domain.com` or `www.your-domain.com`).
     - `$request_uri` represents the full URI of the request (e.g., `/about`).

---

- **HTTPS service on port 443**

  ```nginx
  server {
      listen 443 ssl http2;
      listen [::]:443 ssl http2;
      server_name your-domain.com www.your-domain.com;
  ```

  **Purpose of this block:**

  1. **Listen on HTTPS (port 443)**

     - `listen 443 ssl http2;`
     - `ssl` : Enable SSL encryption.
     - `http2` : Enable HTTP/2 for improved performance (reduces connection latency).
     - `listen [::]:443 ssl http2;` : Allows IPv6 connections using HTTPS.

  2. **Set `server_name`**

     - Same as the HTTP configuration, it applies to `your-domain.com` and `www.your-domain.com`.

---

- **SSL certificate configuration**

  ```nginx
      # SSL certificate file paths (issued by Certbot)
      ssl_certificate      /etc/letsencrypt/live/your-domain.com/fullchain.pem;
      ssl_certificate_key  /etc/letsencrypt/live/your-domain.com/privkey.pem;
  ```

  **Purpose of this block:**

  - These are the certificate paths automatically generated by **Let’s Encrypt**:
    - `fullchain.pem` : The full certificate (including the intermediate CA certificates).
    - `privkey.pem` : The private key used for SSL encryption.

---

- **Reverse proxy to FastAPI**

  ```nginx
      location / {
          proxy_pass http://127.0.0.1:8000;  # Forward requests to the backend application (FastAPI)
          proxy_http_version 1.1;
          proxy_set_header Host $host;
          proxy_set_header X-Real-IP $remote_addr;
          proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
          proxy_set_header X-Forwarded-Proto $scheme;

          # Let Nginx handle errors returned by FastAPI
          proxy_intercept_errors on;
      }
  ```

  **Purpose of this block:**

  - `proxy_pass http://127.0.0.1:8000;`

    - Assumes that FastAPI is running on `127.0.0.1:8000` (command: `uvicorn --host 127.0.0.1 --port 8000`).
    - Forwards API requests to the FastAPI application running on the local server.

  - **Proxy headers**

    - `proxy_set_header Host $host;` : Passes the original `Host` header.
    - `proxy_set_header X-Real-IP $remote_addr;` : Passes the real client IP to the backend.
    - `proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;` : Records the entire proxy chain.
    - `proxy_set_header X-Forwarded-Proto $scheme;` : Ensures the backend knows whether the original request was HTTP or HTTPS.

  - **Error handling**

    - `proxy_intercept_errors on;`
      - Allows Nginx to intercept errors returned by the backend (e.g., 502, 503).
      - Custom error pages can be specified with `error_page`.

---

After completing the configuration, reload Nginx to apply the changes:

```bash
sudo nginx -t  # Test Nginx configuration syntax for correctness
sudo systemctl reload nginx  # Apply configuration changes and reload the service
```

## Nginx Reverse Proxy and Backend Services

In the architecture described above, Nginx serves as both the HTTPS endpoint and reverse proxy.

Nginx handles the TLS handshake and encryption/decryption, meaning the SSL termination occurs at the Nginx layer. The backend application only needs to handle HTTP requests from Nginx, without needing to support HTTPS. Internal communication within the server remains lightweight, without the additional burden of encryption.

This setup simplifies the backend application's configuration and reduces the maintenance overhead related to HTTPS.

Additionally, when using Nginx as a reverse proxy, the backend application by default cannot directly access the client’s real IP address. Instead, it will only see Nginx's IP.

To resolve this, we need to pass important HTTP headers to the backend application via `proxy_set_header` to ensure the backend correctly identifies client information.

Key headers include:

- **`X-Forwarded-For`** → Passes the original client’s IP address.
- **`Host`** → Retains the original host name of the request.
- **`X-Forwarded-Proto`** → Indicates the protocol used in the original request (`http` or `https`).

Nginx configuration:

```nginx
proxy_set_header Host $host;
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
proxy_set_header X-Forwarded-Proto $scheme;
```

Assuming our backend is a Python-based FastAPI application, by default FastAPI will not parse the proxy-passed headers. Therefore, we need to enable the `proxy-headers` mode to correctly identify headers like `X-Forwarded-For`:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --proxy-headers
```

Alternatively, inside FastAPI, we can use `StarletteMiddleware.ProxyHeaders`:

```python
from fastapi import FastAPI
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.proxy_headers import ProxyHeadersMiddleware

app = FastAPI()

# Enable proxy header parsing
app.add_middleware(ProxyHeadersMiddleware)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
```

With this configuration, the backend application can correctly obtain the client’s real IP, request protocol, and other information, preventing misidentification.

Finally, to ensure security, FastAPI should only listen on the local interface to prevent direct external access to the HTTP service, reducing the risk of unauthorized access.

Best practice is:

- **FastAPI should listen on the local interface**

  ```bash
  uvicorn app:app --host 127.0.0.1 --port 8000
  ```

- **Nginx `proxy_pass` should point to the local port**
  ```nginx
  location / {
      proxy_pass http://127.0.0.1:8000;
  }
  ```

This way, external users cannot directly access the backend FastAPI application; they can only access it through Nginx as a proxy, enhancing security. Internal traffic remains unencrypted over HTTP, minimizing the additional burden of HTTPS and improving performance.

## Configuring Automatic Certificate Renewal

Let's Encrypt certificates are only valid for 90 days. This short validity period is designed to enhance security and encourage website administrators to implement automated renewal mechanisms. Manual renewal might lead to certificate expiration, disrupting HTTPS connections on the site. Therefore, it’s recommended to use Certbot's automatic renewal feature to ensure the certificate remains valid at all times.

### Certbot's Automatic Renewal Mechanism

Certbot uses different methods for automatic renewal depending on the system's initialization system (init system):

For systemd-based systems (most modern Linux distributions), Certbot uses a systemd timer to manage renewals, rather than a cron job.

You can check if the systemd timer is enabled:

```bash
systemctl list-timers | grep certbot
```

If enabled, you should see something like this:

```
certbot.timer               2025-02-13 02:00:00 UTC   12h left
```

You can also manually check the status of the systemd timer:

```bash
systemctl status certbot.timer
```

systemd runs the renewal check twice a day via the `certbot.timer`. The actual renewal only occurs if the certificate is within **30 days of expiry**.

---

For non-systemd systems, Certbot uses a cron job, typically located at `/etc/cron.d/certbot`.

The cron job might look like this:

```
0 */12 * * * root test -x /usr/bin/certbot -a \! -d /run/systemd/system && perl -e 'sleep int(rand(43200))' && certbot -q renew --no-random-sleep-on-renew
```

This means it runs every 12 hours. If the system is using systemd, the cron job will not be executed.

To avoid multiple renewals occurring simultaneously on the server, the cron job introduces a random delay of up to 12 hours.

### Verifying Automatic Renewal

Whether using **systemd timer** or **cron**, you can manually test the renewal mechanism:

```bash
sudo certbot renew --dry-run
```

If there are no errors in the output, it means the automatic renewal mechanism is functioning correctly.

You can also check the installed certificates:

```bash
sudo certbot certificates
```

This will list the expiration dates and file paths of all certificates, ensuring they are not expired.

### Ensuring the New Certificate Takes Effect After Renewal

After Let's Encrypt renews a certificate, Nginx/Apache will still use the old certificate until it's reloaded.

To ensure the new certificate is loaded, you can use the following command:

```bash
sudo systemctl reload nginx  # or systemctl reload apache2
```

You can also add a `--post-hook "systemctl reload nginx"` to the `certbot renew` command, so that Certbot automatically reloads the server after a successful renewal.

### Verifying Successful Renewal of the Certificate

To check the current certificate expiration date:

```bash
sudo certbot certificates
```

This will list all certificates managed by Certbot, including the expiration date and storage location. Make sure that after renewal, the expiration date has been updated.

## Testing and Verifying HTTPS Configuration

After completing the HTTPS setup, perform the following tests to ensure everything is working correctly and meets security standards:

### Browser Test

Go to `https://your-domain.com` in your browser, ensuring that the secure lock icon 🔒 appears. Click on the certificate information to verify that the issuer is **Let's Encrypt** and that the certificate is valid and matches the domain.

Try accessing `http://your-domain.com`; it should automatically redirect to HTTPS.

### Checking HTTP to HTTPS Redirection

Use `curl` to test:

```bash
curl -I http://your-domain.com
```

Ensure that it returns a `301 Moved Permanently` response, and the `Location` header should point to `https://your-domain.com/...`.

### Ensuring the Website Functions Correctly

Check that the backend API/website works as expected, such as:

- Does the FastAPI API respond correctly?
- Are internal HTTPS links on the website working?
- Have you taken `X-Forwarded-Proto` into account for handling HTTPS redirects?

### Checking Nginx Logs

Review error and access logs to ensure there are no abnormalities:

```bash
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

If you encounter TLS connection errors, it may be due to outdated devices not supporting **TLS 1.2**. Adjust settings accordingly if necessary.

## Conclusion

We've accomplished a lot.

In this chapter, we successfully enabled HTTPS for Nginx, ensuring that all traffic is encrypted using a certificate issued by Let's Encrypt. We also forced HTTP traffic to be redirected to HTTPS, providing a more secure connection environment.

In addition to the basic HTTPS deployment, we used Nginx as a reverse proxy for FastAPI, ensuring that the front-end Nginx securely forwards requests to the backend service while preserving original request information such as client IP and protocol (HTTP/HTTPS). This ensures the integrity and traceability of the backend application.

Finally, we implemented an automatic certificate renewal mechanism, ensuring that Let's Encrypt certificates are updated automatically and take effect after renewal, keeping the website's HTTPS connection active at all times.

So, are the website's security standards complete?

Not quite yet. In the next chapter, we will explore advanced security configurations, such as HSTS, CSP headers, and how to prevent common website attacks.
