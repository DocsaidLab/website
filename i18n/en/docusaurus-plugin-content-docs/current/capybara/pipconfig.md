---
sidebar_position: 4
---

# PIP Configuration

This section takes a deeper look into the pip configuration mechanism to help you avoid package conflicts and permission issues across multiple Python environments.

## Usage

On Linux/macOS systems, you can use the following commands to manage local and global configurations:

```bash
python -m pip config [<file-option>] list
python -m pip config [<file-option>] [--editor <editor-path>] edit
```

Here, `<file-option>` can be one of the following options:

- `--global`: Specifies the global configuration file for the operating system.
- `--user`: Specifies the configuration file for the user level.
- `--site`: Specifies the configuration file for the current virtual environment.

The `--editor` parameter allows you to specify the path to an external editor. If this parameter is not provided, the default text editor will be used based on the `VISUAL` or `EDITOR` environment variable.

For example, to edit the global configuration file using the Vim editor, you can run:

```bash
python -m pip config --global --editor vim edit
```

:::tip
If you are using Windows, the configuration file may be located at `%APPDATA%\pip\pip.ini` or `%HOMEPATH%\.pip\pip.ini`, among other paths. You can refer to the official documentation or use `pip config list` to further confirm the actual location.
:::

## Priority

The priority of configuration files is crucial. Below is a list of configuration files that may exist on your machine, sorted by priority:

1. **Site-level files**:
   - `/home/user/.pyenv/versions/3.x.x/envs/envs_name/pip.conf`
2. **User-level files**:
   - `/home/user/.config/pip/pip.conf`
   - `/home/user/.pip/pip.conf`
3. **Global-level files**:
   - `/etc/pip.conf`
   - `/etc/xdg/pip/pip.conf`

In a Python environment, pip will search for and apply configuration files in the order listed above.

Ensuring you are modifying the correct configuration file can help prevent difficult-to-trace errors.

## Example Configuration File

Here is an example configuration file:

```ini
[global]
index-url = https://pypi.org/simple
trusted-host = pypi.org
               pypi.python.org
               files.pythonhosted.org
extra-index-url = https://pypi.anaconda.org/simple
```

In this configuration file, the meanings of the parameters are as follows:

- `index-url`: Sets the default source that pip uses when installing packages.
- `trusted-host`: Lists sources that do not require HTTPS for secure verification, to prevent SSL errors.
- `extra-index-url`: Provides additional source addresses for searching and installing dependencies. Unlike `index-url`, pip will look for `extra-index-url` when the required package is not found in the source specified by `index-url`.

:::warning
Please note that when using multiple sources, all sources should be trusted because the most suitable version of a package will be selected from these sources during installation. Untrusted sources may pose security risks.
:::

:::tip
If you have a private package server or need to specify a username and password for authentication, you can place your credentials in your `pip.conf` for automation. However, ensure the file permissions are secure.
:::
