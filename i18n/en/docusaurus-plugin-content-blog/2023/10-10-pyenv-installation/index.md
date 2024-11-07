---
slug: pyenv-installation
title: Managing Python Versions with pyenv
authors: Zephyr
tags: [pyenv, virtualenv]
image: /en/img/2023/1010.webp
description: Documenting the installation and usage of pyenv.
---

In earlier years, Conda was predominantly used for managing Python environments. Nowadays, pyenv is commonly employed.

This article aims to document the installation and usage of pyenv.

<!-- truncate -->

## Prerequisites

Before installing `pyenv`, ensure that `Git` is installed on your system.

:::info
The pyenv package provides a [**Common build problems guide**](https://github.com/pyenv/pyenv/wiki/Common-build-problems) to address installation issues.

If you encounter any problems during installation, refer to this page.
:::

## Installing `pyenv`

1. **Execute Installation Command**:

   You can quickly install `pyenv` by running the following command:

   ```bash
   curl https://pyenv.run | bash
   ```

   This command fetches and executes the installation script from the `pyenv-installer` repository on GitHub.

2. **Configure Your Shell Environment**:

   After installation, follow the [**setup guide**](https://github.com/pyenv/pyenv#set-up-your-shell-environment-for-pyenv) to configure your shell environment to ensure that `pyenv` works correctly.

   If you are using `bash`, add the following code to your `.bashrc` file:

   ```bash
   export PATH="$HOME/.pyenv/bin:$PATH"
   eval "$(pyenv init --path)"
   eval "$(pyenv virtualenv-init -)"
   ```

   For `zsh` users, add the above code to your `.zshrc` file.

3. **Restart Your Shell**:

   After completing the above steps, reload the new configuration.

   ```bash
   exec $SHELL
   ```

## Using `pyenv`

Once installed and configured, you can start using `pyenv` to manage multiple Python versions:

- **Install a New Python Version**:

  ```bash
  pyenv install 3.10.14
  ```

- **Switch the Global Python Version**:

  ```bash
  pyenv global 3.10.14
  ```

- **Use a Specific Version in a Directory**:
  ```bash
  pyenv local 3.8.5
  ```

## Virtual Environments

Virtual environments are crucial in Python development.

They allow us to use different Python versions and dependencies in different projects.

At the very least, when you accidentally mess up your Python environment, you can simply delete the virtual environment and start over.

:::tip
It's highly recommended to use virtual environments when developing Python projects.
:::

### Installation

`pyenv` also provides a `pyenv-virtualenv` plugin, making it easier to manage Python virtual environments.

Previously, this feature required separate installation, but it's now integrated into `pyenv`, and we can directly use:

```bash
pyenv virtualenv 3.10.14 your-env-name
```

Here, `3.10.14` is the Python version you want to use, which you've already installed in the previous step, and `your-env-name` is the name of the virtual environment.

### Usage

To activate the virtual environment, run:

```bash
pyenv activate your-env-name
```

### Removal

Finally, when you no longer need the virtual environment, you can run the following command to delete it:

```bash
pyenv virtualenv-delete your-env-name
```

## Updating `pyenv`

To update `pyenv` to the latest version, simply run:

```bash
pyenv update
```

## Uninstalling `pyenv`

If you decide to no longer use `pyenv`, follow these steps to uninstall:

1. **Remove the `pyenv` Installation Directory**:

   ```bash
   rm -fr ~/.pyenv
   ```

2. **Clean Your `.bashrc`**:

   Remove or comment out the relevant `pyenv` configuration lines, then restart your shell:

   ```bash
   exec $SHELL
   ```
