---
slug: pyenv-installation
title: pyenvでPythonバージョンを管理する
authors: Z. Yuan
tags: [pyenv, virtualenv]
image: /ja/img/2023/1010.webp
description: pyenvのインストールと使用方法を記録しました。
---

昔、Pythonを使用していた頃は主にCondaを使って管理していましたが、現在ではpyenvがよく使われるツールとなっています。

この記事では、pyenvのインストールと使用方法を簡単に記録し、異なるオペレーティングシステムに対して必要な補足説明も行います。

<!-- truncate -->

## 前提条件

`pyenv`をインストールする前に、`Git`がシステムにインストールされている必要があります。

:::info
pyenvのパッケージには、[**インストールの問題ガイド**](https://github.com/pyenv/pyenv/wiki/Common-build-problems)も提供されています。

インストール中に問題が発生した場合は、このページを参照してください。
:::

## よくある問題と解決策

以下は重要なケースとその解決策です：

- **依存パッケージが不足している**
  必要なパッケージやビルドツールを、[**pyenv公式の依存環境**](https://github.com/pyenv/pyenv/wiki#suggested-build-environment)に従ってインストールしてください。

- **zlib拡張のコンパイルに失敗**
  エラーメッセージ：
  - `ERROR: The Python zlib extension was not compiled. Missing the zlib?`

  解決策：
  - Ubuntu/Debianシステムでは、`zlib1g`と`zlib1g-dev`をインストールします：
    ```bash
    sudo apt install zlib1g zlib1g-dev
    ```
  - macOSでは、Homebrewでzlibをインストールしている場合、環境変数を設定します：
    ```bash
    CPPFLAGS="-I$(brew --prefix zlib)/include" pyenv install -v <pythonバージョン>
    ```

- **OpenSSL拡張のコンパイルに失敗**
  エラーメッセージ：
  - `ERROR: The Python ssl extension was not compiled. Missing the OpenSSL lib?`

  解決策：
  - OpenSSL開発パッケージがインストールされていることを確認します（例えば、Ubuntuでは`sudo apt install libssl-dev`、Fedoraでは`sudo dnf install openssl-devel`）。
  - OpenSSLが標準パスにインストールされていない場合、次のように設定します：
    ```bash
    CPPFLAGS="-I<opensslインストールパス>/include" \
    LDFLAGS="-L<opensslインストールパス>/lib" \
    pyenv install -v <pythonバージョン>
    ```

- **システムリソース不足**
  `resource temporarily unavailable`エラーが発生した場合、`make`の並列数を減らすことを試みます：
  ```bash
  MAKE_OPTS='-j 1' pyenv install <pythonバージョン>
  ```

- **python-build定義が見つからない**
  `python-build: definition not found`エラーが発生した場合、python-build定義を更新します：
  ```bash
  cd ~/.pyenv/plugins/python-build && git pull
  ```

- **macOSアーキテクチャ関連のエラー**
  `ld: symbol(s) not found for architecture x86_64`や`ld: symbol(s) not found for architecture arm64`のようなエラーが発生した場合、Homebrewパッケージが正しいアーキテクチャに対応しているか確認し、追加の環境変数（CPPFLAGS、LDFLAGS、CONFIGURE_OPTS）を設定する必要があるかもしれません。

詳細については[**Common build problems**](https://github.com/pyenv/pyenv/wiki/Common-build-problems)を参照してください。

## クロスプラットフォームに関する注意事項

- **Linux/macOS：**
  - インストール方法は基本的に同じで、次の指示をそのまま使用できます。
  - オペレーティングシステムに必要なコンパイル依存ライブラリ（例えば、Ubuntuでは`build-essential`、`libssl-dev`、`zlib1g-dev`など）をインストールする必要があります。

- **Windowsユーザー：**
  - pyenvは元々Unix系環境向けに設計されているため、[**pyenv-win**](https://github.com/pyenv-win/pyenv-win)バージョンの使用を推奨します。
  - あるいは、WindowsでWSL（Windows Subsystem for Linux）やGit Bashなどを使用し、Unix系の操作環境を得ることも可能です。

- **その他のシェルユーザー：**
  - bashやzsh以外のシェル（例えば、fish）を使用している場合、そのシェルの設定ファイルを参照して設定を調整してください。

## `pyenv`のインストール

1. **インストールコマンドを実行：**

   以下のコマンドを実行して、`pyenv`を簡単にインストールできます：

   ```bash
   curl https://pyenv.run | bash
   ```

   このコマンドは、GitHub上の`pyenv-installer`リポジトリからインストールスクリプトを取得して実行します。

2. **シェル環境の設定：**

   インストール後、[**設定ガイド**](https://github.com/pyenv/pyenv#set-up-your-shell-environment-for-pyenv)に従ってシェル環境を設定し、`pyenv`が正しく動作するようにします。

   `bash`を使用している場合は、次のコードを`.bashrc`ファイルに追加します：

   ```bash
   export PATH="$HOME/.pyenv/bin:$PATH"
   eval "$(pyenv init --path)"
   eval "$(pyenv virtualenv-init -)"
   ```

   `zsh`を使用している場合は、同様のコードを`.zshrc`ファイルに追加します。他のシェルについては、対応する設定ファイルを参照してください。

3. **シェルの再起動：**

   上記の手順を完了した後、設定をリロードします：

   ```bash
   exec $SHELL
   ```

## `pyenv`の使用

インストールと設定が完了したら、`pyenv`を使って複数のPythonバージョンを管理できます：

- **新しいPythonバージョンをインストール：**

  ```bash
  pyenv install 3.10.14
  ```

- **グローバルPythonバージョンを切り替え：**

  ```bash
  pyenv global 3.10.14
  ```

- **特定のディレクトリで特定バージョンを使用：**

  ```bash
  pyenv local 3.8.5
  ```

## 仮想環境

Python開発において仮想環境は非常に重要で、異なるプロジェクトで独立したPythonバージョンと依存関係を使用し、環境の衝突を避けるのに役立ちます。

:::tip
私は個人的に、各Pythonプロジェクトで仮想環境を使用することをお勧めします。万が一環境が壊れても、簡単に削除して再作成できます。
:::

### インストール

`pyenv`は、仮想環境を管理するための`pyenv-virtualenv`プラグインも提供しています。この機能は`pyenv`に統合されており、直接使用できます：

```bash
pyenv virtualenv 3.10.14 your-env-name
```

ここで、`3.10.14`は使用するPythonバージョン（インストールされていることを確認）、`your-env-name`は仮想環境の名前です。

### 使用

仮想環境を起動するには：

```bash
pyenv activate your-env-name
```

### 削除

仮想環境が不要になった場合、以下のコマンドで削除できます：

```bash
pyenv virtualenv-delete your-env-name
```

## `pyenv`の更新

`pyenv`を最新バージョンに更新するには、以下の方法を参照してください：

- **更新プラグインを使用：** `pyenv-update`プラグインがインストールされている場合、以下のコマンドで直接更新できます：

  ```bash
  pyenv update
  ```

- **手動更新：**
  `~/.pyenv`ディレクトリに移動して、Gitコマンドで更新します：

  ```bash
  cd ~/.pyenv
  git pull
  ```

## `pyenv`の削除

`pyenv`を使用しなくなった場合、以下の手順で削除できます：

1. **`pyenv`インストールディレクトリの削除：**

   ```bash
   rm -fr ~/.pyenv
   ```

2. **シェル設定のクリーニング：**

   `.bashrc`や`.zshrc`（または他のシェル設定ファイル）にある`pyenv`関連の設定行を削除またはコメントアウトし、その後シェルを再起動します：

   ```bash
   exec $SHELL
   ```

## まとめ

よく使うコマンドはこんな感じです。あなたのPython環境が素晴らしいものになることを願っています！