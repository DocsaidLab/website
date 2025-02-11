---
slug: setting-up-pypiserver-on-ubuntu-with-docker
title: Setting Up PyPiServer on Ubuntu
authors: Z. Yuan
tags: [docker, pypiserver]
image: /en/img/2023/0913.webp
description: Setting up PyPiServer on Ubuntu using Docker.
---

Today, we’ll document the process of setting up a PyPi Server using Docker on Ubuntu.

We assume you have Docker installed on Ubuntu and are familiar with basic Docker operations.

<!-- truncate -->

## Pull the Image

```bash
docker pull pypiserver/pypiserver:latest
```

## Create a Directory

Let’s quickly create a directory in the home directory to store Python packages.

```bash
mkdir ~/packages
```

You can use a different name if you prefer, but remember to adjust it in the configuration files.

## Set up htpasswd

:::tip
If you don’t want to set a password, you can skip this step.
:::

htpasswd is a file format for storing usernames and passwords, which pypiserver uses for user authentication. This is a simple and effective way to enhance pypiserver’s security.

First, install `apache2-utils`:

```bash
sudo apt install apache2-utils
```

Then, use the following command to create a new `.htpasswd` file:

```bash
htpasswd -c ~/.htpasswd [username]
```

You will be prompted to enter a password for `username`. After entering the password, the `.htpasswd` file will be created in your home directory.

Once the file is created, you can run `pypiserver` with the `docker run` command and enable authentication with the `.htpasswd` file.

## Run as a Background Service

To run the Docker container as a background service, we’ll use Docker Compose with Systemd.

### Install Docker Compose

If you haven’t installed Docker Compose, start by doing so:

- [**Official Docker Compose Installation Guide**](https://docs.docker.com/compose/install/)

Docker Compose has recently undergone major updates, changing many commands. Notably, `docker-compose` has been replaced by `docker compose`.

Install the latest version of Docker Compose:

```bash
sudo apt update
sudo apt install docker-compose-plugin
```

Verify the installation:

```bash
docker compose version
```

### Create a Configuration File

Find a location to create the `docker-compose.yml` file and add the following content:

```yaml
version: "3.3"
services:
  pypiserver:
    image: pypiserver/pypiserver:latest
    volumes:
      - /home/[username]/auth:/data/auth
      - /home/[username]/packages:/data/packages
    command: run -P /data/auth/.htpasswd -a update,download,list /data/packages
    ports:
      - "8080:8080"
```

- Replace `[username]` with your actual username.
- You can modify the external port mapping if needed, for example: `"18080:8080"`.

:::tip
You can refer to `pypiserver`’s example file here: [**docker-compose.yml**](https://github.com/pypiserver/pypiserver/blob/master/docker-compose.yml)
:::

If you want to skip setting a password, modify the `command` line as follows:

```yaml
command: run -a . -P . /data/packages --server wsgiref
```

### Set up a Systemd Service

Create a configuration file:

```bash
sudo vim /etc/systemd/system/pypiserver.service
```

Add the following content:

```bash
[Unit]
Description=PypiServer Docker Compose
Requires=docker.service
After=docker.service

[Service]
WorkingDirectory=/path/to/your/docker-compose/directory
ExecStart=/usr/bin/docker compose up --remove-orphans
ExecStop=/usr/bin/docker compose down
Restart=always

[Install]
WantedBy=multi-user.target
```

- Replace `/path/to/your/docker-compose/directory` with the actual path to the `docker-compose.yml` file (only the path, not the filename).
- Ensure the Docker path is correct by using `which docker` to confirm.
- This setup uses the new `docker compose` command instead of `docker-compose`.

### Start the Service

Let systemd know about the new service configuration:

```bash
sudo systemctl daemon-reload
```

Start the service:

```bash
sudo systemctl enable pypiserver.service
sudo systemctl start pypiserver.service
```

## Check the Status

To check the current status of the service, use:

```bash
sudo systemctl status pypiserver.service
```

This will display the current status of the `pypiserver` service, including whether it’s running and the latest log output.

## Start Using the Server

You can now use `pip` to install and upload packages.

### Uploading Packages

Let’s say you have a package named `example_package-0.1-py3-none-any.whl`.

Use `twine` to upload the package:

```bash
pip install twine
twine upload --repository-url http://localhost:8080/ example_package-0.1-py3-none-any.whl
```

- Make sure that `localhost:8080` is the address and port of your pypiserver.

### Downloading and Installing Packages

Use `pip` to install packages by specifying the address and port of `pypiserver`:

```bash
pip install --index-url http://localhost:8080/ example_package
```

### Using Basic Authentication

If you set up basic authentication for your pypiserver, you’ll need to provide credentials when uploading or downloading:

- To upload a package:

  ```bash
  twine upload \
    --repository-url http://localhost:8080/ \
    --username [username] \
    --password [password] \
    example_package-0.1-py3-none-any.whl
  ```

- To install a package:

  ```bash
  pip install \
    --index-url http://[username]:[password]@localhost:8080/ \
    example_package
  ```

## Configuring `pip.conf`

To avoid specifying `--index-url` each time you use `pip install`, we can add the relevant configuration to `pip.conf`.

### Configuration File

The `pip.conf` file can be located in several places, with the following order of precedence:

- Level 1: Site-level configuration:

  - `/home/[username]/.pyenv/versions/3.8.18/envs/main/pip.conf`

- Level 2: User-level configuration:

  - `/home/[username]/.pip/pip.conf`
  - `/home/[username]/.config/pip/pip.conf`

- Level 3: Global configuration:

  - `/etc/pip.conf`
  - `/etc/xdg/pip/pip.conf`

Identify the file that corresponds to your Python environment and add the following content:

```bash
[global]
index-url = http://[server_ip]:8080/
trusted-host = [server_ip]
```

Again, replace `[server_ip]:8080` with the correct address and port of your `pypiserver`.

After setting this up, the server address configured in `pip.conf` will automatically be used as the package source when you use `pip install [package_name]`.

## Conclusion

You’ve now successfully set up your own PyPI server, and you know how to upload and download packages.

We hope this guide solves your problem.
