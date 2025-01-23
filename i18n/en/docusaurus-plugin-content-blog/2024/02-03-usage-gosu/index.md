---
slug: gosu-usage
title: "User Switching Tool in Containers: gosu"
authors: Zephyr
tags: [docker, gosu]
image: /en/img/2024/0203.webp
description: A tool worth learning how to use.
---

Docker technology has been widely adopted for deployment and management purposes.

We typically package various applications and their dependencies together to ensure consistent operation across different environments.

<!-- truncate -->

## Common Issues

However, with frequent use, you can't escape encountering a few common problems.

### TTY Conversion

A common scenario is when you output a file within a container, exit the container, and then realize that the file permissions are all set to root. You then have to use `chown` to change the file permissions repeatedly, which can be quite cumbersome.

---

Or, when running an application within a Docker container using sudo that requires interaction with the terminal (TTY), these applications might fail to properly detect the terminal because sudo might not handle terminal ownership and control properly when creating a new session. As a result, these applications requiring terminal interaction may not run correctly or encounter input/output errors when attempting to use them.

### Signal Forwarding

Suppose you have a container running a web server like Apache or Nginx. Typically, you might use command-line tools to manage this container, including starting and stopping it. Inside the container, if you use sudo to start the web server, sudo will create a new process to run the web server.

The problem arises when you want to stop or restart the container. The container management system sends signals (like SIGTERM) to the container to notify processes inside it to prepare for shutdown. However, if the web server was started via sudo, this signal might only be sent to the sudo process, not the actual web server process. This means the web server might not receive the stop signal, preventing proper cleanup and safe shutdown.

:::tip
The design intention of sudo is to enhance security by allowing regular users to execute commands as other users (usually the root user). In this process, sudo starts a new session to execute the command. While this behavior typically poses no issues in traditional operating system environments, it can lead to signal delivery problems in lightweight virtualized environments like containers, as the new session created by sudo might be incompatible with how the container management system sends signals.
:::

## What is gosu?

- [**gosu GitHub repository**](https://github.com/tianon/gosu)

gosu is a tool specifically designed for containers, aiming to simplify and secure command execution within containers. When you need to run a command as a different user (e.g., switching from an administrator to a regular user) within a container, gosu comes in handy. Its core mechanism directly borrows from how `Docker/libcontainer` starts applications within containers (in fact, it directly uses code from the `libcontainer` library for handling `/etc/passwd`).

If you're not interested in its inner workings, think of gosu as a helper. When you tell it to "run this command as this user," it does just that, then exits, leaving no trace behind.

### Practical Use Cases

Using gosu in the ENTRYPOINT script of a Docker container is a typical use case, especially when we need to downgrade from the root user to a non-privileged user to perform certain operations. This practice is crucial for safeguarding the security of the container runtime environment, as it effectively reduces potential security risks.

Installing gosu is straightforward, usually requiring just a few lines in the Dockerfile for installation and configuration. The following example demonstrates how to install gosu in a Dockerfile and dynamically create users and groups using an entrypoint script, then use gosu to execute commands with the specified user identity.

```Dockerfile title="Dockerfile"
# Based on an existing base image
FROM some_base_image:latest

WORKDIR /app

# Install gosu
RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

# Prepare entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["default_command"]
```

The example content of the `entrypoint.sh` script is as follows. It dynamically creates a user and group based on the environment variables USER_ID and GROUP_ID, then executes a command using gosu:

```bash title="entrypoint.sh"
#!/bin/bash
# Check if USER_ID and GROUP_ID environment variables are set
if [ ! -z "$USER_ID" ] && [ ! -z "$GROUP_ID" ]; then
    # Create group and user
    groupadd -g "$GROUP_ID" usergroup
    useradd -u "$USER_ID" -g usergroup -m user
    # Execute command using gosu
    exec gosu user "$@"
else
    exec "$@"
fi
```

For a real-world example, refer to: [**Example training docker**](https://github.com/DocsaidLab/Otter/blob/main/docker/Dockerfile)

### Security Considerations

While gosu's primary purpose is to switch from the `root` user to a non-privileged user during container startup, its developers also emphasize potential security risks associated with using gosu in certain contexts.

This is because any tool that allows user switching, if misused, could open doors to security vulnerabilities. Therefore, development and operations teams need to have a thorough understanding of gosu's usage scenarios and ensure it is only used in secure environments.
