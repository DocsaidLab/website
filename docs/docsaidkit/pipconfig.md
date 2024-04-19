---
sidebar_position: 4
---

# PIP 參數配置

## 使用方式

在 Linux/macOS 系統上，你可以使用以下指令來管理本地和全局配置：

```bash
python -m pip config [<file-option>] list
python -m pip config [<file-option>] [--editor <editor-path>] edit
```

其中，`<file-option>` 可以是以下選項：

- `--global`：指定操作系統全局配置文件。
- `--user`：指定操作用戶級別配置文件。
- `--site`：指定操作當前虛擬環境內的配置文件。

`--editor` 參數允許你指定一個外部編輯器的路徑。如果不提供此參數，則會依照 `VISUAL` 或 `EDITOR` 環境變數使用預設的文本編輯器。

例如：若想要使用 Vim 編輯器修改全局配置文件，可以使用以下指令：

```bash
python -m pip config --global --editor vim edit
```

## 優先級

配置文件的優先級順序非常重要。以下是可能存在於你的機器上的配置文件列表，按優先級排序：

1. **站點級文件**：
    - `/home/user/.pyenv/versions/3.x.x/envs/envs_name/pip.conf`
2. **用戶級文件**：
    - `/home/user/.config/pip/pip.conf`
    - `/home/user/.pip/pip.conf`
3. **全局級文件**：
    - `/etc/pip.conf`
    - `/etc/xdg/pip/pip.conf`

在 python 環境中，pip 會按照上述順序來尋找並應用配置文件。確認你正在修改的是正確的配置文件，可以幫助避免產生難以追蹤的錯誤。

## 配置文件範例

以下是一個配置文件的範例：

```ini
[global]
index-url = https://pypi.org/simple
trusted-host = pypi.org
               pypi.python.org
               files.pythonhosted.org
extra-index-url = https://pypi.anaconda.org/simple
```

這個配置文件中，各參數的意義如下：

- `index-url`：設定 pip 在安裝套件時使用的默認源。
- `trusted-host`：列出無需使用 HTTPS 進行安全驗證的來源，以防出現 SSL 錯誤。
- `extra-index-url`：提供額外的來源地址，用於搜索和安裝依賴套件。與 `index-url` 不同，當需要的套件在 `index-url` 指定的源中找不到時，pip 會尋找 `extra-index-url`。

:::warning
請注意，當使用多個源時，所有的源都應該是可信的，因為安裝過程中將會從這些源中選擇最適合的版本。未經信任的源可能會帶來安全風險。
:::