# sites-available

## Nginx Site Configuration Files

By default, Nginx uses `/etc/nginx/sites-available/default` as the default site configuration.

We can create new configuration files as needed (e.g., `temp_api.conf`) to specifically configure new API reverse proxies and static resource services.

Now, let's take a look at the contents of this file:

```nginx title="/etc/nginx/sites-available/default.conf"
##
# You should look at the following URL's in order to grasp a solid understanding
# of Nginx configuration files in order to fully unleash the power of Nginx.
# https://www.nginx.com/resources/wiki/start/
# https://www.nginx.com/resources/wiki/start/topics/tutorials/config_pitfalls/
# https://wiki.debian.org/Nginx/DirectoryStructure
#
# In most cases, administrators will remove this file from sites-enabled/ and
# leave it as reference inside of sites-available where it will continue to be
# updated by the nginx packaging team.
#
# This file will automatically load configuration files provided by other
# applications, such as Drupal or Wordpress. These applications will be made
# available underneath a path with that package name, such as /drupal8.
#
# Please see /usr/share/doc/nginx-doc/examples/ for more detailed examples.
##

# Default server configuration
#
server {
	listen 80 default_server;
	listen [::]:80 default_server;

	# SSL configuration
	#
	# listen 443 ssl default_server;
	# listen [::]:443 ssl default_server;
	#
	# Note: You should disable gzip for SSL traffic.
	# See: https://bugs.debian.org/773332
	#
	# Read up on ssl_ciphers to ensure a secure configuration.
	# See: https://bugs.debian.org/765782
	#
	# Self signed certs generated by the ssl-cert package
	# Don't use them in a production server!
	#
	# include snippets/snakeoil.conf;

	root /var/www/html;

	# Add index.php to the list if you are using PHP
	index index.html index.htm index.nginx-debian.html;

	server_name _;

	location / {
		# First attempt to serve request as file, then
		# as directory, then fall back to displaying a 404.
		try_files $uri $uri/ =404;
	}

	# pass PHP scripts to FastCGI server
	#
	#location ~ \.php$ {
	#	include snippets/fastcgi-php.conf;
	#
	#	# With php-fpm (or other unix sockets):
	#	fastcgi_pass unix:/run/php/php7.4-fpm.sock;
	#	# With php-cgi (or other tcp sockets):
	#	fastcgi_pass 127.0.0.1:9000;
	#}

	# deny access to .htaccess files, if Apache's document root
	# concurs with nginx's one
	#
	#location ~ /\.ht {
	#	deny all;
	#}
}


# Virtual Host configuration for example.com
#
# You can move that to a different file under sites-available/ and symlink that
# to sites-enabled/ to enable it.
#
#server {
#	listen 80;
#	listen [::]:80;
#
#	server_name example.com;
#
#	root /var/www/example.com;
#	index index.html;
#
#	location / {
#		try_files $uri $uri/ =404;
#	}
#}
```

## Configuration File Overview

```nginx
##
# You should look at the following URL's in order to grasp a solid understanding
# of Nginx configuration files in order to fully unleash the power of Nginx.
# https://www.nginx.com/resources/wiki/start/
# https://www.nginx.com/resources/wiki/start/topics/tutorials/config_pitfalls/
# https://wiki.debian.org/Nginx/DirectoryStructure
#
# In most cases, administrators will remove this file from sites-enabled/ and
# leave it as reference inside of sites-available where it will continue to be
# updated by the nginx packaging team.
#
# This file will automatically load configuration files provided by other
# applications, such as Drupal or Wordpress.
#
# Please see /usr/share/doc/nginx-doc/examples/ for more detailed examples.
##
```

This section contains the author's "comment," providing some learning resources for Nginx configuration and suggesting that administrators keep this file in the `sites-available/` directory as a reference. The actual running configuration should be loaded from the `sites-enabled/` directory.

Since this advice comes from the developers, there is a reason behind it, so let's leave it as is and continue reading.

## Default Server Configuration

```nginx
server {
    listen 80 default_server;
    listen [::]:80 default_server;
```

- **`listen 80 default_server;`**
  - The server listens on **port 80 (HTTP)** and is set as the "default server". This means that when there is no match for `server_name`, the request will be handled by this block.
- **`listen [::]:80 default_server;`**

  - This allows **IPv6** requests to access the server on port 80 as well.

- **Default Web Root and Homepage Settings**

  ```nginx
  	root /var/www/html;

  	# Add index.php to the list if you are using PHP
  	index index.html index.htm index.nginx-debian.html;
  ```

  - **`root /var/www/html;`**
  - Sets the web root directory. All static files (HTML, CSS, JS) for requests will be served from this directory.
  - **`index index.html index.htm index.nginx-debian.html;`**
  - Sets the default homepage files. If `index.html` exists, it will be served first; if not, it will look for `index.htm`. If neither exists, it will handle the request according to the `location` rules.

- **Server Name Configuration**

  ```nginx
  	server_name _;
  ```

  - **`server_name _;`**
  - `_` means that "any unmatched requests" will be handled by this server block.
  - It can be changed to a specific domain, such as `server_name example.com;`, to make this configuration apply only to `example.com`.

- **Handling Static Resources and 404 Page**

  ```nginx
  	location / {
  		# First attempt to serve request as file, then
  		# as directory, then fall back to displaying a 404.
  		try_files $uri $uri/ =404;
  	}
  ```

  - **`location / {}`**
  - Defines how requests to `/` (root) are handled.
  - **`try_files $uri $uri/ =404;`**
  - Attempts to:
    1. Serve `$uri` (the requested file, such as `/index.html`).
    2. If the request is for a directory, it checks if `index.html` exists in that directory (`$uri/`).
    3. If neither is found, it returns **404 Not Found**.

## FastCGI PHP Handling

:::tip
Default as Commented Out.
:::

```nginx
    # pass PHP scripts to FastCGI server
    #
    #location ~ \.php$ {
    #    include snippets/fastcgi-php.conf;
    #
    #    # With php-fpm (or other unix sockets):
    #    fastcgi_pass unix:/run/php/php7.4-fpm.sock;
    #    # With php-cgi (or other tcp sockets):
    #    fastcgi_pass 127.0.0.1:9000;
    #}
```

This section is for **FastCGI configuration**, allowing Nginx to handle PHP requests and pass them to the PHP-FPM server.

- **Enable PHP-FPM processing**:
  - Uncomment the `location ~ \.php$` block.
  - Ensure PHP-FPM is running at `/run/php/php7.4-fpm.sock` or `127.0.0.1:9000`.

## Restrict Access to `.htaccess`

:::tip
Default as Commented Out.
:::

```nginx
    # deny access to .htaccess files, if Apache's document root
    # concurs with nginx's one
    #
    #location ~ /\.ht {
    #    deny all;
    #}
```

`.htaccess` is a configuration file used by Apache, and Nginx **does not support `.htaccess`**.

However, if Nginx and Apache coexist, this rule can be used to prevent `.htaccess` from being exposed to external access.

## Example Virtual Host Configuration

:::tip
Default as Commented Out.
:::

```nginx
#server {
#    listen 80;
#    listen [::]:80;
#
#    server_name example.com;
#
#    root /var/www/example.com;
#    index index.html;
#
#    location / {
#        try_files $uri $uri/ =404;
#    }
#}
```

This is an additional example of a virtual host configuration for the `example.com` website.

You can move this code to `sites-available/example.com`, then create a symbolic link to `sites-enabled/` to enable it.

## How to Enable/Disable This Configuration

Once the configuration file in `sites-available` is complete, you need to link it to `sites-enabled` for it to take effect.

For this `default` configuration file, on Ubuntu/Debian systems, use the following command to enable it:

```bash
sudo ln -s /etc/nginx/sites-available/default /etc/nginx/sites-enabled/
```

To disable it, first remove the symlink in `sites-enabled`:

```bash
sudo rm /etc/nginx/sites-enabled/default
```

Then reload Nginx:

```bash
sudo systemctl reload nginx
```

---

This `default` configuration file is typically used as the **default Nginx website**, suitable for testing if Nginx is running properly.

If you want to customize your website, it's recommended to "create a new configuration file" and enable the appropriate `server_name` settings.
