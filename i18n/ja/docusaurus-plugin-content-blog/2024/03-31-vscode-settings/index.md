---
slug: vscode-settings
title: よく使う VScode 設定
authors: Zephyr
tags: [vscode, settings]
image: /ja/img/2024/0331.webp
description: よく使う VScode の設定を記録し、今後の参考にします。
---

少し前、不明な理由で VScode の設定ファイルが消失し、再設定を余儀なくされました。その際、再設定にかなりの時間を費やすことになりました。

ちょうどよい機会でもあったので、自分たちの設定ファイルを見直し、よく使う設定を記録しておくことにしました。

<!-- truncate -->

## 設定ファイル

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

## パラメータの説明

- `editor.fontFamily`：フォントを設定します。ここでは Fira Code と MesloLGS NF フォントを使用しています。
- `editor.fontLigatures`：フォントのリガチャ（合字）を有効にするかどうかを設定します。
- `files.associations`：ファイルの関連付けを設定します。ここでは Dockerfile_base を dockerfile に関連付けています。
- `markdown.preview.fontSize`：Markdown プレビューのフォントサイズを設定します。
- `debug.console.fontSize`：デバッグコンソールのフォントサイズを設定します。
- `explorer.confirmDragAndDrop`：ドラッグ＆ドロップの確認を有効にするかどうかを設定します。
- `editor.minimap.enabled`：ミニマップを有効にするかどうかを設定します。
- `editor.minimap.maxColumn`：ミニマップの最大列数を設定します。
- `editor.smoothScrolling`：スムーズスクロールを有効にするかどうかを設定します。
- `editor.rulers`：インデントのガイドラインとなる列番号を設定します。
- `workbench.colorCustomizations`：カスタム配色を設定します。
- `terminal.integrated.fontFamily`：ターミナルフォントを設定します。
- `files.trimTrailingWhitespace`：行末の空白を削除するかどうかを設定します。
- `files.trimFinalNewlines`：最終行の空行を削除するかどうかを設定します。
- `diffEditor.ignoreTrimWhitespace`：空白の差分を無視するかどうかを設定します。
- `python.terminal.activateEnvironment`：Python 環境を有効化するかどうかを設定します。
- `git.ignoreLegacyWarning`：Git 警告を無視するかどうかを設定します。
- `git.autofetch`：自動フェッチを有効にするかどうかを設定します。
- `editor.largeFileOptimizations`：大きなファイルの最適化を有効にするかどうかを設定します。
- `editor.mouseWheelZoom`：マウスホイールでのズームを有効にするかどうかを設定します。
- `editor.codeActionsOnSave`：保存時のコード操作を設定します。
- `editor.formatOnSave`：保存時にフォーマットを有効にするかどうかを設定します。
- `workbench.editorAssociations`：エディターの関連付けを設定します。
- `debug.onTaskErrors`：タスクエラー時の動作を設定します。
- `explorer.confirmDelete`：削除の確認を有効にするかどうかを設定します。
- `terminal.integrated.copyOnSelection`：選択したテキストを自動的にコピーするかどうかを設定します。
- `terminal.integrated.cursorBlinking`：ターミナルカーソルの点滅を設定します。
- `terminal.integrated.cursorStyle`：ターミナルカーソルのスタイルを設定します。
- `remote.downloadExtensionsLocally`：拡張機能をローカルでダウンロードするかどうかを設定します。
- `terminal.integrated.scrollback`：ターミナルのスクロールバックバッファサイズを設定します。
- `editor.cursorStyle`：カーソルのスタイルを設定します。
- `editor.insertSpaces`：スペースの挿入を有効にするかどうかを設定します。
- `editor.lineNumbers`：行番号を表示するかどうかを設定します。
- `editor.wordWrap`：自動改行を有効にするかどうかを設定します。
- `workbench.editor.wrapTabs`：タブの折り返しを有効にするかどうかを設定します。
- `files.watcherExclude`：ファイル監視から除外するパターンを設定します。
- `notebook.cellToolbarLocation`：ノートブックツールバーの位置を設定します。
- `github.copilot.editor.enableAutoCompletions`：自動補完を有効にするかどうかを設定します。
- `github.copilot.enable`：GitHub Copilot を有効にするかどうかを設定します。
- `workbench.colorTheme`：配色テーマを設定します。
- `editor.multiCursorModifier`：マルチカーソルの修飾キーを設定します。
- `editor.wordWrapColumn`：自動改行の列数を設定します。
- `files.autoSave`：自動保存を有効にするかどうかを設定します。
