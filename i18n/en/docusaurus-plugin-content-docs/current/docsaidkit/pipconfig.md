---
sidebar_position: 4
---

# PIP configs

## Usage

On Linux/macOS systems, you can use the following commands to manage local and global configurations:

```bash
python -m pip config [<file-option>] list
python -m pip config [<file-option>] [--editor <editor-path>] edit
```

Where `<file-option>` can be one of the following options:

- `--global`: Specifies to operate on the system-wide configuration file.
- `--user`: Specifies to operate on the user-level configuration file.
- `--site`: Specifies to operate on the configuration file within the current virtual environment.

The `--editor` parameter allows you to specify the path to an external editor. If this parameter is not provided, the default text editor set in the `VISUAL` or `EDITOR` environment variables will be used.

For example, if you want to use Vim to edit the global configuration file, you can use the following command:

```bash
python -m pip config --global --editor vim edit
```

## Priority

The order of precedence for configuration files is crucial. Below is a list of configuration files that may exist on your machine, ordered by priority:

1. **Site-level file**:
    - `/home/user/.pyenv/versions/3.x.x/envs/env_name/pip.conf`
2. **User-level file**:
    - `/home/user/.config/pip/pip.conf`
    - `/home/user/.pip/pip.conf`
3. **Global-level file**:
    - `/etc/pip.conf`
    - `/etc/xdg/pip/pip.conf`

In the Python environment, pip will search for and apply configuration files in the order listed above. Ensuring that you are modifying the correct configuration file can help avoid hard-to-trace errors.

## Example Configuration File

Here is an example of a configuration file:

```ini
[global]
index-url = https://pypi.org/simple
trusted-host = pypi.org
               pypi.python.org
               files.pythonhosted.org
extra-index-url = https://pypi.anaconda.org/simple
```

In this configuration file, the parameters mean the following:

- `index-url`: Sets the default source that pip uses when installing packages.
- `trusted-host`: Lists sources that do not require SSL verification, to prevent SSL errors.
- `extra-index-url`: Provides additional source addresses for searching and installing dependencies. Unlike `index-url`, when the needed package is not found in the source specified by `index-url`, pip will look at `extra-index-url`.

:::warning
Please be aware that when using multiple sources, all sources should be trusted, as the installation process will select the most suitable version from these sources. Untrusted sources could pose security risks.
:::