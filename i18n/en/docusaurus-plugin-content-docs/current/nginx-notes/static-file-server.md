---
sidebar_position: 6
---

# Nginx Serving Static Resources

This chapter uses a static website built with Docusaurus as an example to explain how to serve a website using Nginx.

Assume you have already prepared a domain name and pointed it to your server, for example:

- `your.domain`

:::tip
Make sure to replace `your.domain` in all the commands and configuration examples below with the actual domain name you're using.
:::

## Build the Static Website

Before building, make sure the `url` in your `docusaurus.config.js` file is correctly set to your domain:

```javascript
module.exports = {
  url: "https://your.domain",
  // ...other configurations
};
```

Once verified, run the following command to generate the static files:

```bash
DOCUSAURUS_IGNORE_SSG_WARNINGS=true yarn build
```

:::tip
If you do not use the `DOCUSAURUS_IGNORE_SSG_WARNINGS` environment variable, you might see many strange warning messages, but they won't affect the build result.
:::

This command will generate static HTML, CSS, and JS files in the `build/` folder.

Next, upload the built files to the specified directory on your server and set the file permissions:

```bash
sudo mkdir -p /var/www/your.domain
sudo rsync -av build/ /var/www/your.domain/
sudo chown -R www-data:www-data /var/www/your.domain
```

## Obtain SSL Certificate

It is recommended to use Let's Encrypt to issue an SSL certificate to ensure the website is served securely via HTTPS:

```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your.domain
```

## Configure Nginx

Create a dedicated Nginx configuration file:

```bash
sudo vim /etc/nginx/sites-available/your.domain
```

Example configuration:

```nginx
server {
    listen 80;
    server_name your.domain;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your.domain;

    # SSL certificate (requires Certbot issuance)
    ssl_certificate /etc/letsencrypt/live/your.domain/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your.domain/privkey.pem;

    # Set the static file serving directory
    root /var/www/your.domain;
    index index.html;

    # Set MIME types
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

After completing the configuration, enable the file:

```bash
sudo ln -s /etc/nginx/sites-available/your.domain /etc/nginx/sites-enabled/
```

Test and reload the Nginx configuration:

```bash
sudo nginx -t
sudo systemctl reload nginx
```

## Advanced Configuration

In production environments, it is recommended to add more security and performance settings. Here is a complete example of advanced configuration:

```nginx
server {
    listen 80;
    listen [::]:80;
    server_name your.domain;

    # 🔒 Automatically redirect HTTP to HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name your.domain;

    ssl_certificate /etc/letsencrypt/live/your.domain/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your.domain/privkey.pem;

    # 🔒 Security headers
    add_header Strict-Transport-Security "max-age=63072000; includeSubdomains; preload" always;

    # 🔧 Limit file upload size to prevent DDoS
    client_max_body_size 10M;

    # 📁 build directory
    root /var/www/your.domain;
    index index.html;

    # 🗃️ Static resource caching
    location ~* \.(jpg|jpeg|png|gif|ico|svg|woff2?|ttf|css|js)$ {
        expires 7d;
        add_header Cache-Control "public, must-revalidate";
    }

    # 🔧 Main routing rules
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Set MIME types
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # 🔍 Custom log file location
    access_log /var/log/nginx/your.domain.access.log main;
    error_log /var/log/nginx/your.domain.error.log warn;
}
```

## Conclusion

With the steps above and advanced configurations, you can use Nginx to serve static resources securely and efficiently. Once the configuration is complete, you can access your website at `https://your.domain`.

Finally, remember to regularly check the SSL certificate expiration, update the Nginx version, and maintain security rules. These measures effectively protect against potential security risks and ensure the stable operation of the website over time.
