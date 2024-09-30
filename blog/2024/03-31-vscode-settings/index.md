---
slug: vscode-settings
title: 常用的 VScode 參數設定
authors: Zephyr
tags: [vscode, settings]
image: /img/2024/0331.webp
description: 記錄常用的 VScode 參數設定，以便日後使用。
---

<figure>
![title](/img/2024/0331.webp)
<figcaption>封面圖片：由 GPT-4 閱讀本文之後自動生成</figcaption>
</figure>

---

不久前，因為不明原因導致我的 VScode 設定檔消失。費了我一小時又重新設定了一次。

剛好也讓我有機會重新檢視一下自己的設定檔，並且將一些常用的設定檔記錄下來。

<!-- truncate -->

## 設定檔

```json
{
  "editor.fontFamily": "Fira Code, MesloLGS NF",
  "editor.fontLigatures": true,
  "files.associations": {
    "Dockerfile_base": "dockerfile"
  },
  "markdown.preview.fontSize": 15,
  "debug.console.fontSize": 14,
  "explorer.confirmDragAndDrop": true,
  "editor.minimap.enabled": true,
  "editor.minimap.maxColumn": 80,
  "editor.smoothScrolling": true,
  "editor.rulers": [80, 120],
  "workbench.colorCustomizations": {
    "editorRuler.foreground": "#ff4081",
    "minimap.background": "#00000050",
    "editor.background": "#1e1e1e",
    "editor.foreground": "#d4d4d4"
  },
  "terminal.integrated.fontFamily": "Fira Code, MesloLGS NF",
  "files.trimTrailingWhitespace": true,
  "files.trimFinalNewlines": true,
  "diffEditor.ignoreTrimWhitespace": true,
  "python.terminal.activateEnvironment": true,
  "git.ignoreLegacyWarning": true,
  "git.autofetch": true,
  "editor.largeFileOptimizations": false,
  "editor.mouseWheelZoom": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true,
    "source.fixAll": true
  },
  "editor.formatOnSave": true,
  "workbench.editorAssociations": {
    "*.ipynb": "jupyter-notebook"
  },
  "debug.onTaskErrors": "abort",
  "explorer.confirmDelete": true,
  "terminal.integrated.copyOnSelection": true,
  "terminal.integrated.cursorBlinking": true,
  "terminal.integrated.cursorStyle": "line",
  "remote.downloadExtensionsLocally": true,
  "terminal.integrated.scrollback": 10000,
  "editor.cursorStyle": "line",
  "editor.insertSpaces": true,
  "editor.lineNumbers": "on",
  "editor.wordWrap": "on",
  "workbench.editor.wrapTabs": false,
  "files.watcherExclude": {
    "**/.git/objects/**": true,
    "**/.git/subtree-cache/**": true,
    "**/node_modules/*/**": true
  },
  "notebook.cellToolbarLocation": {
    "default": "right",
    "jupyter-notebook": "left"
  },
  "github.copilot.editor.enableAutoCompletions": true,
  "github.copilot.enable": {
    "*": true,
    "plaintext": false,
    "markdown": true,
    "scminput": false
  },
  "workbench.colorTheme": "Monokai Pro",
  "editor.multiCursorModifier": "ctrlCmd",
  "editor.wordWrapColumn": 120,
  "files.autoSave": "onFocusChange"
}
```

## 參數說明

- `editor.fontFamily`：設定字體，這裡使用了 Fira Code 和 MesloLGS NF 字體。

- `editor.fontLigatures`：設定是否啟用字體連字。

- `files.associations`：設定檔案關聯，這裡將 Dockerfile_base 關聯到 dockerfile。

- `markdown.preview.fontSize`：設定 markdown 預覽的字體大小。

- `debug.console.fontSize`：設定 debug console 的字體大小。

- `explorer.confirmDragAndDrop`：設定是否確認拖放。

- `editor.minimap.enabled`：設定是否啟用縮略圖。

- `editor.minimap.maxColumn`：設定縮略圖的最大列數。

- `editor.smoothScrolling`：設定是否啟用平滑滾動。

- `editor.rulers`：設定縮排對齊的列數。

- `workbench.colorCustomizations`：設定顏色自定義。

- `terminal.integrated.fontFamily`：設定終端字體。

- `files.trimTrailingWhitespace`：設定是否刪除行尾空格。

- `files.trimFinalNewlines`：設定是否刪除最後一行的空行。

- `diffEditor.ignoreTrimWhitespace`：設定是否忽略空格。

- `python.terminal.activateEnvironment`：設定是否啟用 Python 環境。

- `git.ignoreLegacyWarning`：設定是否忽略 Git 警告。

- `git.autofetch`：設定是否自動更新。

- `editor.largeFileOptimizations`：設定是否優化大文件。

- `editor.mouseWheelZoom`：設定是否滑鼠滾輪縮放。

- `editor.codeActionsOnSave`：設定保存時的代碼操作。

- `editor.formatOnSave`：設定保存時是否格式化。

- `workbench.editorAssociations`：設定編輯器關聯。

- `debug.onTaskErrors`：設定任務錯誤時的操作。

- `explorer.confirmDelete`：設定是否確認刪除。

- `terminal.integrated.copyOnSelection`：設定是否選中即複製。

- `terminal.integrated.cursorBlinking`：設定終端游標閃爍。

- `terminal.integrated.cursorStyle`：設定終端游標樣式。

- `remote.downloadExtensionsLocally`：設定是否本地下載擴展。

- `terminal.integrated.scrollback`：設定終端滾動緩衝區大小。

- `editor.cursorStyle`：設定游標樣式。

- `editor.insertSpaces`：設定是否插入空格。

- `editor.lineNumbers`：設定是否顯示行號。

- `editor.wordWrap`：設定是否自動換行。

- `workbench.editor.wrapTabs`：設定是否換行標籤。

- `files.watcherExclude`：設定文件監視排除。

- `notebook.cellToolbarLocation`：設定筆記本工具欄位置。

- `github.copilot.editor.enableAutoCompletions`：設定是否啟用自動完成。

- `github.copilot.enable`：設定是否啟用 GitHub Copilot。

- `workbench.colorTheme`：設定顏色主題。

- `editor.multiCursorModifier`：設定多游標修改器。

- `editor.wordWrapColumn`：設定自動換行的列數。

- `files.autoSave`：設定自動保存。

## 結語

以上是我常用的 VScode 參數設定。
