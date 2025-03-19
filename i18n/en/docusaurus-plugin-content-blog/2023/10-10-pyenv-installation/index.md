---
slug: pyenv-installation
title: Managing Python Versions with pyenv
authors: Z. Yuan
tags: [pyenv, virtualenv]
image: /en/img/2023/1010.webp
description: A record of installing and using pyenv.
---

In the past, when using Python, I mostly relied on Conda for management. Nowadays, pyenv has become the go-to tool.

In this article, I briefly document how to install and use pyenv and provide necessary supplemental information for different operating systems.

<!-- truncate -->

## Prerequisites

Before installing `pyenv`, you need to have `Git` installed on your system.

:::info
The pyenv package provides a [**troubleshooting guide**](https://github.com/pyenv/pyenv/wiki/Common-build-problems).

If you encounter issues during the installation, you can refer to this page.
:::

## Common Issues and Solutions

Here are a few important cases and their solutions:

- **Missing dependencies**
  Please first install all necessary packages and build tools according to the [**official pyenv dependency guide**](https://github.com/pyenv/pyenv/wiki#suggested-build-environment).

- **zlib extension compilation failure**

  The common error message is:

  - `ERROR: The Python zlib extension was not compiled. Missing the zlib?`

  Solution:
  - On Ubuntu/Debian systems, install `zlib1g` and `zlib1g-dev`:
    ```bash
    sudo apt install zlib1g zlib1g-dev
    ```
  - On macOS, if you installed zlib with Homebrew, set the environment variable:
    ```bash
    CPPFLAGS="-I$(brew --prefix zlib)/include" pyenv install -v <python-version>
    ```

- **OpenSSL extension compilation failure**

  If you see:

  - `ERROR: The Python ssl extension was not compiled. Missing the OpenSSL lib?`

  Solution:
  - Make sure the OpenSSL development packages are installed (e.g., on Ubuntu use `sudo apt install libssl-dev`, on Fedora use `sudo dnf install openssl-devel`).
  - If OpenSSL is installed in a non-standard location, set the following:
    ```bash
    CPPFLAGS="-I<openssl-install-path>/include" \
    LDFLAGS="-L<openssl-install-path>/lib" \
    pyenv install -v <python-version>
    ```

- **System resources are insufficient**

  If you encounter the "resource temporarily unavailable" error, try reducing the make parallelism:

  ```bash
  MAKE_OPTS='-j 1' pyenv install <python-version>
  ```

- **python-build definition not found**

  If you encounter the `python-build: definition not found` error, update the python-build definitions:

  ```bash
  cd ~/.pyenv/plugins/python-build && git pull
  ```

- **macOS architecture-related errors**

  If you see errors like `ld: symbol(s) not found for architecture x86_64` or `ld: symbol(s) not found for architecture arm64`, ensure that the Homebrew packages match the correct architecture and check if additional environment variables (such as CPPFLAGS, LDFLAGS, and CONFIGURE_OPTS) need to be set.

For more detailed information, refer to [**Common build problems**](https://github.com/pyenv/pyenv/wiki/Common-build-problems).

## Cross-platform Considerations

- **Linux/macOS:**
  - The installation method is generally the same, and you can directly use the commands in the next section.
  - Install the necessary compilation dependencies based on your operating system (e.g., on Ubuntu, you may need to install `build-essential`, `libssl-dev`, `zlib1g-dev`, etc.).

- **Windows users:**
  - pyenv is natively designed for Unix-like environments, so it’s recommended to use the [**pyenv-win**](https://github.com/pyenv-win/pyenv-win) version.
  - Alternatively, you can use WSL, Git Bash, or similar tools on Windows to get a Unix-like environment.

- **Other Shell Users:**
  - If you use a shell other than bash or zsh (such as fish), refer to the corresponding shell configuration files for adjustments.

## Installing `pyenv`

1. **Run the installation command:**

   You can quickly install `pyenv` with the following command:

   ```bash
   curl https://pyenv.run | bash
   ```

   This command will fetch and execute the installation script from the `pyenv-installer` repository on GitHub.

2. **Configure your shell environment:**

   After installation, follow the [**setup guide**](https://github.com/pyenv/pyenv#set-up-your-shell-environment-for-pyenv) to configure your shell environment to ensure `pyenv` works properly.

   For bash, add the following lines to your `.bashrc` file:

   ```bash
   export PATH="$HOME/.pyenv/bin:$PATH"
   eval "$(pyenv init --path)"
   eval "$(pyenv virtualenv-init -)"
   ```

   For zsh, add the same lines to your `.zshrc` file; for other shells, refer to their respective configuration files.

3. **Restart your shell:**

   After completing the above steps, reload the configuration:

   ```bash
   exec $SHELL
   ```

## Using `pyenv`

Once installed and configured, you can use `pyenv` to manage multiple Python versions:

- **Install a new Python version:**

  ```bash
  pyenv install 3.10.14
  ```

- **Switch the global Python version:**

  ```bash
  pyenv global 3.10.14
  ```

- **Use a specific version in a particular directory:**

  ```bash
  pyenv local 3.8.5
  ```

## Virtual Environments

Virtual environments are crucial in Python development as they help you use independent Python versions and dependencies in different projects, avoiding conflicts.

:::tip
I personally recommend using virtual environments in every Python project, even if the environment is accidentally damaged, it can be easily deleted and recreated.
:::

### Installation

`pyenv` provides the `pyenv-virtualenv` plugin, making virtual environment management more convenient.

This functionality is now integrated into `pyenv` and can be used directly:

```bash
pyenv virtualenv 3.10.14 your-env-name
```

Here, `3.10.14` is the Python version you want to use (make sure it's installed), and `your-env-name` is the name of the virtual environment.

### Usage

Activate the virtual environment:

```bash
pyenv activate your-env-name
```

### Deleting

When you no longer need the virtual environment, you can delete it with:

```bash
pyenv virtualenv-delete your-env-name
```

## Updating `pyenv`

If you need to update `pyenv` to the latest version, you can follow these methods:

- **Using the update plugin:** If you’ve installed the [**pyenv-update**](https://github.com/pyenv/pyenv-update) plugin, you can directly execute:

  ```bash
  pyenv update
  ```

- **Manual update:**
  Go to the `~/.pyenv` directory and update using Git:

  ```bash
  cd ~/.pyenv
  git pull
  ```

## Removing `pyenv`

If you decide to stop using `pyenv`, follow these steps to remove it:

1. **Remove the `pyenv` installation directory:**

   ```bash
   rm -fr ~/.pyenv
   ```

2. **Clean up shell configurations:**

   Remove or comment out the lines related to `pyenv` in `.bashrc`, `.zshrc` (or other shell configuration files), then restart the shell:

   ```bash
   exec $SHELL
   ```

## Conclusion

These are the commonly used commands. I hope you have a great Python environment!