---
slug: vscode-settings
title: Common VSCode Configuration Settings
authors: Z. Yuan
tags: [vscode, settings]
image: /en/img/2024/0331.webp
description: VSCode settings for future reference.
---

A while ago, due to unknown reasons, the VSCode configuration file disappeared, and we had to reconfigure it, which took us some time.

It also provided an opportunity to review our own configuration files and record some commonly used configuration files.

<!-- truncate -->

## Configuration Settings

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

## Parameter Descriptions

- `editor.fontFamily`: Sets the font family; here, Fira Code and MesloLGS NF fonts are used.
- `editor.fontLigatures`: Sets whether ligatures in the font are enabled.
- `files.associations`: Sets file associations; here, Dockerfile_base is associated with dockerfile.
- `markdown.preview.fontSize`: Sets the font size for markdown preview.
- `debug.console.fontSize`: Sets the font size for the debug console.
- `explorer.confirmDragAndDrop`: Sets whether drag and drop confirmation is enabled.
- `editor.minimap.enabled`: Sets whether the minimap is enabled.
- `editor.minimap.maxColumn`: Sets the maximum number of columns in the minimap.
- `editor.smoothScrolling`: Sets whether smooth scrolling is enabled.
- `editor.rulers`: Sets the column numbers for indentation guides.
- `workbench.colorCustomizations`: Sets color customizations.
- `terminal.integrated.fontFamily`: Sets the terminal font family.
- `files.trimTrailingWhitespace`: Sets whether trailing whitespace is trimmed.
- `files.trimFinalNewlines`: Sets whether final newlines are trimmed.
- `diffEditor.ignoreTrimWhitespace`: Sets whether whitespace in diffs is ignored.
- `python.terminal.activateEnvironment`: Sets whether the Python environment is activated.
- `git.ignoreLegacyWarning`: Sets whether Git warnings are ignored.
- `git.autofetch`: Sets whether autofetch is enabled.
- `editor.largeFileOptimizations`: Sets whether optimizations for large files are enabled.
- `editor.mouseWheelZoom`: Sets whether mouse wheel zoom is enabled.
- `editor.codeActionsOnSave`: Sets code actions on save.
- `editor.formatOnSave`: Sets whether formatting is applied on save.
- `workbench.editorAssociations`: Sets editor associations.
- `debug.onTaskErrors`: Sets action on task errors.
- `explorer.confirmDelete`: Sets whether delete confirmation is enabled.
- `terminal.integrated.copyOnSelection`: Sets whether copying is done on selection.
- `terminal.integrated.cursorBlinking`: Sets terminal cursor blinking.
- `terminal.integrated.cursorStyle`: Sets terminal cursor style.
- `remote.downloadExtensionsLocally`: Sets whether extensions are downloaded locally.
- `terminal.integrated.scrollback`: Sets terminal scrollback buffer size.
- `editor.cursorStyle`: Sets cursor style.
- `editor.insertSpaces`: Sets whether spaces are inserted.
- `editor.lineNumbers`: Sets whether line numbers are displayed.
- `editor.wordWrap`: Sets whether word wrap is enabled.
- `workbench.editor.wrapTabs`: Sets whether tab wrapping is enabled.
- `files.watcherExclude`: Sets file watcher exclusion patterns.
- `notebook.cellToolbarLocation`: Sets notebook cell toolbar location.
- `github.copilot.editor.enableAutoCompletions`: Sets whether auto completions are enabled in GitHub Copilot.
- `github.copilot.enable`: Sets whether GitHub Copilot is enabled.
- `workbench.colorTheme`: Sets the color theme.
- `editor.multiCursorModifier`: Sets the multi-cursor modifier.
- `editor.wordWrapColumn`: Sets the column for word wrapping.
- `files.autoSave`: Sets auto-save behavior.
