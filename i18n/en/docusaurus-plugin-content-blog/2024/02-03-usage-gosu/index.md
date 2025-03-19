---
slug: gosu-usage
title: "User Switching Tool in Containers: gosu"
authors: Z. Yuan
tags: [docker, gosu]
image: /en/img/2024/0203.webp
description: A tool that is so useful, you definitely need to learn how to use it.
---

Docker technology has been widely used in deployment and management.

We often package various applications and their dependencies together to ensure consistent operation across different environments.

<!-- truncate -->

## Common Problems

However, if you use it frequently, you're bound to run into a few common issues.

### TTY Conversion

A common situation is when you output a file inside the container.

After exiting the container, you may notice that the file permissions are set to root.

At this point, you need to use `chown` to change the file's permissions.

Over and over again, isn't that annoying?

---

Another case is when using `sudo` to start an interactive application inside a Docker container. These applications may not correctly detect the terminal because `sudo` may not handle the terminal's ownership and control properly when creating a new session.

As a result, applications that need terminal interaction may not work properly or encounter input/output errors when trying to use them.

### Signal Forwarding

Suppose you have a container running a web server, like Apache or Nginx.

Typically, you might use command-line tools to manage the container, including starting and stopping it. Inside the container, if you start the web server using `sudo`, then `sudo` will create a new process to run the web server.

The problem arises when you want to stop or restart the container. The container management system will send a signal (like SIGTERM) to notify the processes inside the container to stop. However, if the web server is started via `sudo`, this signal may only be sent to the `sudo` process, not the actual web server process. This means the web server may not receive the stop signal and thus cannot perform proper cleanup and shutdown.

:::tip
The design of `sudo` is to improve security, allowing regular users to execute commands as other users (typically root). In this process, `sudo` creates a new session to execute the command.

This behavior typically doesn’t cause issues in traditional operating system environments, but in lightweight virtualization environments like containers, it may lead to signal forwarding problems, as the new session created by `sudo` may not be compatible with the way the container management system sends signals.
:::

## What is gosu?

- [**gosu GitHub repository**](https://github.com/tianon/gosu)

gosu is a tool specifically designed for containers. Its purpose is to make it easier and safer to execute commands inside containers.

When you need to run a program as a different user (e.g., switching from the admin user to a regular user), gosu comes in handy. Its core functionality is directly inspired by the way `Docker/libcontainer` launches applications inside containers (in fact, it directly uses the `/etc/passwd` handling code from the `libcontainer` codebase).

If you're not interested in the technical details, simply put, gosu is like an assistant: when you tell it "please run this command as this user," it does it for you and then exits, leaving no trace behind.

### Practical Use Case

The most common use of gosu is in Docker entrypoint scripts, where it downgrades the container from the root user to a regular user to avoid permission issues.

Here’s an example:

First, add a few lines to your Dockerfile to install gosu:

```Dockerfile title="Dockerfile"
# Base the image on an existing base image
FROM some_base_image:latest

WORKDIR /app

# Install gosu
RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

# Prepare the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["default_command"]
```

Next, create an entrypoint script `entrypoint.sh`, which dynamically creates users based on environment variables and then uses gosu to switch users to run the command:

```bash title="entrypoint.sh"
#!/bin/bash
# Check if USER_ID and GROUP_ID environment variables are set
if [ ! -z "$USER_ID" ] && [ ! -z "$GROUP_ID" ]; then
    # Create user group and user
    groupadd -g "$GROUP_ID" usergroup
    useradd -u "$USER_ID" -g usergroup -m user
    # Use gosu to execute the command
    exec gosu user "$@"
else
    exec "$@"
fi
```

For a real example, refer to: [**Example training docker**](https://github.com/DocsaidLab/Otter/blob/main/docker/Dockerfile)

## Security Considerations

While gosu is very convenient in container environments, developers have pointed out potential security risks. Any tool that allows user identity switching should be used cautiously.

It’s like the key to your house: while it's very useful, if misused, it might leave your security door wide open. Therefore, when using gosu, teams should ensure they fully understand the use case to avoid abuse in unsafe scenarios.

For related discussions, refer to: [**Keeping the TTY across a privilege boundary might be insecure #37**](https://github.com/tianon/gosu/issues/37)

:::info
I know you’re too lazy to read, so here’s a brief excerpt of the key points:

**A developer raised concerns that keeping TTY across a privilege boundary might pose a security risk.**

When a program switches from high to low privilege, if a new virtual terminal is not created, file descriptors (such as standard input/output) that were not closed in the parent process might be used by the new process. Using the TIOCSTI ioctl call, an attacker can inject input characters into the TTY buffer, simulating keyboard input and executing unauthorized commands. This type of vulnerability is allowed by design.

For example, the following code will inject "id\n" character by character into standard input:

```c
#include <unistd.h>
#include <sys/ioctl.h>
#include <stdio.h>

int main()
{
    for (char *cmd = "id\n"; *cmd; cmd++) {
        if (ioctl(STDIN_FILENO, TIOCSTI, cmd)) {
            fprintf(stderr, "ioctl failed\n");
            return 1;
        }
    }
    return 0;
}
```

This code is considered malicious because it deliberately uses the TIOCSTI ioctl call to simulate keyboard input and inject commands into the terminal without user interaction. This allows for malicious injection or privilege escalation attacks, making it a security risk.

Since Docker assigns a new TTY and replaces the parent shell, the injected commands cannot affect the host or the original terminal, so the risk is lower. In unexpected interactive environments, if you run gosu directly in the terminal without replacing the original TTY, an attacker’s program could successfully inject commands, creating a security vulnerability.

However, this vulnerability primarily affects unintended usage scenarios. If used as designed, such as in Docker containers as an entrypoint with Docker assigning a new TTY, the risk is significantly reduced.

Therefore, as long as it's used in the intended environment, there's no need to overly worry about the risk.
:::