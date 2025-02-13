# nginx.conf

## Nginx Main Configuration File

All Nginx configurations start here, with the file path being **`/etc/nginx/nginx.conf`**.

The structure is as follows:

- **Global Level**: Defines global settings for the service, such as the user under which it runs and the number of processes.
- **Events Level**: Responsible for connection management, such as the maximum number of concurrent connections.
- **HTTP Level**: Defines HTTP-related settings, including log formats, Gzip compression, and virtual host configurations.
- **Server Level**: Defines domain names, listening ports, and SSL configurations for specific websites.
- **Location Level**: Matches specific URL paths and specifies how requests are processed.

However, this structure might seem a bit abstract, so let's take a closer look at the contents of the file:

```nginx title="/etc/nginx/nginx.conf"
user www-data;
worker_processes auto;
pid /run/nginx.pid;
include /etc/nginx/modules-enabled/*.conf;

events {
	worker_connections 768;
	# multi_accept on;
}

http {

	##
	# Basic Settings
	##

	sendfile on;
	tcp_nopush on;
	types_hash_max_size 2048;
	# server_tokens off;

	# server_names_hash_bucket_size 64;
	# server_name_in_redirect off;

	include /etc/nginx/mime.types;
	default_type application/octet-stream;

	##
	# SSL Settings
	##

	ssl_protocols TLSv1 TLSv1.1 TLSv1.2 TLSv1.3; # Dropping SSLv3, ref: POODLE
	ssl_prefer_server_ciphers on;

	##
	# Logging Settings
	##

	access_log /var/log/nginx/access.log;
	error_log /var/log/nginx/error.log;

	##
	# Gzip Settings
	##

	gzip on;

	# gzip_vary on;
	# gzip_proxied any;
	# gzip_comp_level 6;
	# gzip_buffers 16 8k;
	# gzip_http_version 1.1;
	# gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

	##
	# Virtual Host Configs
	##

	include /etc/nginx/conf.d/*.conf;
	include /etc/nginx/sites-enabled/*;
}


#mail {
#	# See sample authentication script at:
#	# http://wiki.nginx.org/ImapAuthenticateWithApachePhpScript
#
#	# auth_http localhost/auth.php;
#	# pop3_capabilities "TOP" "USER";
#	# imap_capabilities "IMAP4rev1" "UIDPLUS";
#
#	server {
#		listen     localhost:110;
#		protocol   pop3;
#		proxy      on;
#	}
#
#	server {
#		listen     localhost:143;
#		protocol   imap;
#		proxy      on;
#	}
#}
```

## Main Configuration File: Global Settings

This section defines the basic operational parameters for the entire Nginx server.

```nginx
user www-data;
worker_processes auto;
pid /run/nginx.pid;
include /etc/nginx/modules-enabled/*.conf;
```

- **`user www-data;`**:
  - Defines the system user under which Nginx runs, typically `www-data`.
- **`worker_processes auto;`**:
  - Defines the number of worker processes. `auto` means it will automatically adjust.
  - Typically, it is based on the number of CPU cores to ensure optimal performance.
- **`pid /run/nginx.pid;`**:
  - Specifies the file location where Nginx stores its process ID (PID), allowing system administrators to check and control Nginx processes.
- **`include /etc/nginx/modules-enabled/*.conf;`**:
  - Loads additional Nginx modules, allowing the activation of various functionalities from the `modules-enabled/` directory (e.g., HTTP/2, stream modules, etc.).

## Main Configuration File: Event Handling Settings

This section determines how Nginx handles connections and requests.

```nginx
events {
    worker_connections 768;
    # multi_accept on;
}
```

- **`worker_connections 768;`**:
  - The maximum number of connections that each `worker_process` can handle simultaneously. If `worker_processes` is set to 4, the total maximum concurrent connections would be `4 × 768 = 3072`.
- **`# multi_accept on;`** (commented out):
  - If enabled, Nginx will accept multiple connections at once when a new connection arrives, rather than processing them one by one. This can enhance performance in high-traffic scenarios.

## Main Configuration File: HTTP Service Settings

This section contains multiple HTTP-related settings that apply to all HTTP services on the site.

```nginx
http {...}
```

Here, we'll break down the sections, and interested readers can explore them further.

- **HTTP Settings**

  ```nginx
  sendfile on;
  tcp_nopush on;
  types_hash_max_size 2048;
  # server_tokens off;
  ```

  These settings affect how Nginx handles HTTP connections.

  - **`sendfile on;`**:
    - Enables the `sendfile()` system call to accelerate the transfer of static files, improving performance.
  - **`tcp_nopush on;`**:
    - Allows Nginx to send the entire HTTP response in a single burst, optimizing network performance.
  - **`types_hash_max_size 2048;`**:
    - Sets the maximum size of the MIME types hash table, affecting the efficiency of parsing the `mime.types` configuration.
  - **`# server_tokens off;`** (commented out):
    - When enabled, Nginx will not display version information in error pages or HTTP response headers, improving security.

- **MIME Type Settings**

  ```nginx
  include /etc/nginx/mime.types;
  default_type application/octet-stream;
  ```

  - **`include /etc/nginx/mime.types;`**:
    - Loads the `mime.types` file, setting the `Content-Type` response header for different file types.
  - **`default_type application/octet-stream;`**:
    - Sets the default `Content-Type` when the file type cannot be identified, responding with `application/octet-stream`.

- **SSL Settings**

  ```nginx
  ssl_protocols TLSv1 TLSv1.1 TLSv1.2 TLSv1.3; # Dropping SSLv3, ref: POODLE
  ssl_prefer_server_ciphers on;
  ```

  - **`ssl_protocols TLSv1 TLSv1.1 TLSv1.2 TLSv1.3;`**:
    - Defines supported SSL/TLS versions and disables outdated SSLv3 to prevent the POODLE vulnerability.
  - **`ssl_prefer_server_ciphers on;`**:
    - Ensures the server prefers its own cipher suite, enhancing security.

- **Log Settings**

  ```nginx
  access_log /var/log/nginx/access.log;
  error_log /var/log/nginx/error.log;
  ```

  - **`access_log /var/log/nginx/access.log;`**:
    - Logs all HTTP access requests, useful for monitoring website traffic.
  - **`error_log /var/log/nginx/error.log;`**:
    - Logs Nginx error messages, helping with debugging and diagnosing issues.

- **Gzip Compression Settings**

  ```nginx
  gzip on;
  # gzip_vary on;
  # gzip_proxied any;
  # gzip_comp_level 6;
  # gzip_buffers 16 8k;
  # gzip_http_version 1.1;
  # gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
  ```

  - **`gzip on;`**:
    - Enables Gzip compression to improve webpage transmission efficiency and reduce bandwidth consumption.
  - **(commented sections)**:
    - `gzip_comp_level 6;` adjusts the compression ratio, higher values yield better compression but increase CPU usage.
    - **`gzip_types ...`**:
      - Defines which MIME types will be compressed using Gzip (e.g., CSS, JS, JSON).

- **Virtual Host Settings**

  ```nginx
  include /etc/nginx/conf.d/*.conf;
  include /etc/nginx/sites-enabled/*;
  ```

  - **`include /etc/nginx/conf.d/*.conf;`**:
    - Loads all `.conf` files in the `conf.d` directory, typically used for global settings.
  - **`include /etc/nginx/sites-enabled/*;`**:
    - Loads site-specific configuration files from the `sites-enabled` directory. Each site's configuration file is usually located in `sites-available`, and symbolic links are used to enable them.

## Main Configuration File: Mail Proxy

```nginx
#mail {
#   server {
#       listen     localhost:110;
#       protocol   pop3;
#       proxy      on;
#   }
#   server {
#       listen     localhost:143;
#       protocol   imap;
#       proxy      on;
#   }
#}
```

This section defines Nginx's mail proxy functionality (POP3/IMAP), which is commented out by default and not enabled.

---

The default content of this file is as shown. Typically, we don't modify this file at first, as it contains global Nginx settings. Instead, we create separate site configuration files in the `sites-available` and `sites-enabled` directories to manage multiple sites.
