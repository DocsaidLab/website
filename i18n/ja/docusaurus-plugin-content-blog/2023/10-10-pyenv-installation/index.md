---
slug: pyenv-installation
title: pyenv を使用した Python バージョン管理
authors: Z. Yuan
tags: [pyenv, virtualenv]
image: /ja/img/2023/1010.webp
description: pyenv のインストールおよび使用方法を記録しました。
---

以前は Python を使用する際に主に Conda を利用して管理していましたが、最近では pyenv をよく使用しています。

この記事では、pyenv のインストール方法と使い方を記録します。

<!-- truncate -->

## 前提条件

`pyenv` をインストールする前に、システムに `Git` がインストールされている必要があります。

:::info
`pyenv` のドキュメントには、[**インストール時の問題に関するガイド**](https://github.com/pyenv/pyenv/wiki/Common-build-problems)が用意されています。

インストール時に問題が発生した場合は、このページを参照してください。
:::

## `pyenv` のインストール

1. **インストールコマンドを実行**：

   以下のコマンドを使用して `pyenv` を素早くインストールできます：

   ```bash
   curl https://pyenv.run | bash
   ```

   このコマンドは、GitHub 上の `pyenv-installer` リポジトリからインストールスクリプトを取得して実行します。

2. **Shell 環境を設定**：

   インストール後、[**設定ガイド**](https://github.com/pyenv/pyenv#set-up-your-shell-environment-for-pyenv) に従って Shell 環境を設定し、`pyenv` が正しく動作するようにします。

   `bash` を使用している場合、以下のコードを `.bashrc` ファイルに追加してください：

   ```bash
   export PATH="$HOME/.pyenv/bin:$PATH"
   eval "$(pyenv init --path)"
   eval "$(pyenv virtualenv-init -)"
   ```

   `zsh` を使用している場合、上記のコードを `.zshrc` ファイルに追加してください。

3. **Shell の再起動**：

   上記の手順を完了した後、新しい設定を再読み込みします：

   ```bash
   exec $SHELL
   ```

## `pyenv` の使用

インストールと設定が完了したら、`pyenv` を使用して複数の Python バージョンを管理できます：

- **新しい Python バージョンをインストール**：

  ```bash
  pyenv install 3.10.14
  ```

- **グローバルな Python バージョンを切り替え**：

  ```bash
  pyenv global 3.10.14
  ```

- **特定のディレクトリで特定のバージョンを使用**：

  ```bash
  pyenv local 3.8.5
  ```

## 仮想環境

Python 開発において仮想環境は非常に重要な概念です。

仮想環境を使用することで、異なるプロジェクトごとに異なる Python バージョンや依存関係を使用できます。

万が一 Python 環境を壊してしまった場合でも、仮想環境を削除して再構築するだけで済みます。

:::tip
Python プロジェクトを開発する際には、仮想環境を使用することを強く推奨します。
:::

### インストール

`pyenv` には `pyenv-virtualenv` プラグインが組み込まれており、これを使用して仮想環境を簡単に管理できます。

以下のコマンドで仮想環境を作成します：

```bash
pyenv virtualenv 3.10.14 your-env-name
```

ここで、`3.10.14` は使用する Python バージョン（事前にインストール済み）で、`your-env-name` は仮想環境の名前です。

### 使用

仮想環境を有効化するには、以下を実行します：

```bash
pyenv activate your-env-name
```

### 削除

仮想環境が不要になった場合、以下のコマンドで削除できます：

```bash
pyenv virtualenv-delete your-env-name
```

## `pyenv` の更新

`pyenv` を最新バージョンに更新するには、以下を実行します：

```bash
pyenv update
```

## `pyenv` の削除

`pyenv` を使用しないことを決めた場合、以下の手順で削除できます：

1. **`pyenv` のインストールディレクトリを削除**：

   ```bash
   rm -fr ~/.pyenv
   ```

2. **`.bashrc` をクリーンアップ**：

   `pyenv` に関連する設定行を削除またはコメントアウトし、Shell を再起動します：

   ```bash
   exec $SHELL
   ```
