---
slug: setting-up-pypiserver-on-ubuntu-with-docker
title: Setting Up PyPiServer on Ubuntu
authors: Zephyr
tags: [docker, pypiserver]
image: /en/img/2023/0913.webp
description: Setting up PyPiServer on Ubuntu using Docker.
---

<figure>
![title](/img/2023/0913.webp)
<figcaption>Cover Image: Automatically generated by GPT-4 after reading the article</figcaption>
</figure>

---

As the Python community continues to evolve, many developers and teams opt to establish their own private Python package index servers to store and manage their Python packages. This not only provides better version control but also ensures the security of software packages.

<!-- truncate -->

In this article, we'll use Docker to set up a PyPi Server and run it on Ubuntu.

We assume you've already installed Docker on Ubuntu and are familiar with basic Docker operations.

## 1. Pull the pypiserver Docker Image

```bash
docker pull pypiserver/pypiserver:latest
```

## 2. Create a Directory to Store Python Packages

Without further ado, let's create a directory to store Python packages in the home directory.

```bash
mkdir ~/packages
```

## 3. Set Up htpasswd

htpasswd is a file format used to store usernames and passwords (often used for basic HTTP authentication).

pypiserver uses this file to authenticate users attempting to upload or download packages. It's a simple yet effective way to enhance the security of pypiserver.

To create a `.htpasswd` file, you need the `apache2-utils` package:

```bash
sudo apt install apache2-utils
```

Then, use the following command to create a new `.htpasswd` file:

```bash
htpasswd -c ~/.htpasswd [username]
```

It will prompt you to enter a password for `username`. After entering the password, the `.htpasswd` file will be created in your home directory.

Now, you can use the `docker run` command mentioned above to run `pypiserver` and authenticate using the `.htpasswd` file you just created.

## 4. Mount pypiserver as a Background Service

To run the Docker container as a background service, we can use Docker Compose and Systemd.

### 4.1 Install Docker Compose

If you haven't installed Docker Compose yet, install it first by referring to the [official Docker Compose installation documentation](https://docs.docker.com/compose/install/).

It's worth noting that Docker Compose has undergone significant updates recently, with many changes in usage compared to before. The most obvious change is the shift from using `docker-compose` commands to `docker compose` commands.

Following the official documentation, here's how you can install the latest version of Docker Compose:

```bash
sudo apt update
sudo apt install docker-compose-plugin
```

Check if Docker Compose is installed correctly:

```bash
docker compose version
```

### 4.2 Create Files

Create a `docker-compose.yml` file somewhere and fill it with the following content:

You can also refer to the template provided by `pypiserver`: [**docker-compose.yml**](https://github.com/pypiserver/pypiserver/blob/master/docker-compose.yml)

```yaml
version: "3.3"
services:
  pypiserver:
    image: pypiserver/pypiserver:latest
    volumes:
      - /home/[your_username]/auth:/data/auth
      - /home/[your_username]/packages:/data/packages
    command: run -P /data/auth/.htpasswd -a update,download,list /data/packages
    ports:
      - "8080:8080"
```

- Replace `[your_username]` with your actual username in the above configuration.
- You can modify the external port mapping here, for example, change it to: `"18080:8080"`.

### 4.3 Create Systemd Service

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

- Make sure to replace `/path/to/your/docker-compose/directory` with the actual path to the `docker-compose.yml` file, specifying only the directory, not the filename.
- Make sure your Docker path is correct, you can use `which docker` to confirm.
- We're using the new `docker compose` commands instead of `docker-compose`.

### 4.4 Start the pypiserver Service

Tell systemd to reload the new service configuration:

```bash
sudo systemctl daemon-reload
```

Start the service:

```bash
sudo systemctl start pypiserver.service
sudo systemctl enable pypiserver.service
```

Now, `pypiserver` will run as a `systemd` service and automatically start each time the host boots up.

## 5. Check Status

If you want to check the current status of the service, you can use:

```bash
sudo systemctl status pypiserver.service
```

This will display the current status of the `pypiserver` service, including whether it's running and recent log outputs.

![pypiserver status](./img/pypiserver.jpg)

## 6. Using pypiserver

Now, you can use `pip` to install and upload packages.

### 6.1 Upload Packages

First, you need a packaged Python software package (usually in .whl or .tar.gz format). Suppose you already have a package named `example_package-0.1-py3-none-any.whl`.

To upload the software package to your `pypiserver`, use `twine`:

```bash
pip install twine
twine upload --repository-url http://localhost:8080/ example_package-0.1-py3-none-any.whl
```

- Ensure that `localhost:8080` is the address and port of your pypiserver service.

### 6.2 Install Packages

To install packages, use `pip` and specify the address and port of your `pypiserver` service:

```bash
pip install --index-url http://localhost:8080/ example_package
```

### 6.3 Use Basic Authentication

If your pypiserver is configured with basic authentication (which may be done for security reasons), you need to provide authentication information when uploading or downloading:

- Uploading packages:

  ```bash
  twine upload --repository-url http://localhost:8080/ --username [username] --password [password] example_package-0.1-py3-none-any.whl
  ```

- Installing packages:

  ```bash
  pip install --index-url http://[username]:[password]@localhost:8080/ example_package
  ```

### 6.4 Configure pip.conf for Long-Term Use

If you frequently install packages from this server, you may not want to specify `--index-url` every time you use `pip install`. In this case, you can set default package index sources in `pip.conf`.

First, find or create a `pip.conf` file. Here are the files that might exist on your machine in order of precedence:

- Priority 1: Site-level configuration file:

  - `/home/[your_username]/.pyenv/versions/3.8.18/envs/main/pip.conf`

- Priority 2: User-level configuration files:

  - `/home/[your_username]/.pip/pip.conf`
  - `/home/[your_username]/.config/pip/pip.conf`

- Priority 3: Global-level configuration files:

  - `/etc/pip.conf`
  - `/etc/xdg/pip/pip.conf`

So make sure to clarify which Python environment is using which file, and once you've found or created the file, add the following content:

```bash
[global]
index-url = http://[your_server_IP]:8080/
trusted-host = [your_server_IP]
```

Again, ensure to replace `[your_server_IP]:8080` with the correct address and port of your `pypiserver`.

From now on, when you use `pip install [package_name]`, `pip` will automatically use the server address configured in `pip.conf` as the package source.

## 7. Conclusion

Congratulations! You've successfully set up your own private PyPI server and learned how to upload and download packages.

With `pypiserver`, you can efficiently manage your Python packages and ensure they're in a secure environment. I hope this article proves to be practically helpful for you.
