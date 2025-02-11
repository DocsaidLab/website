---
slug: fixed-pyenv-install-error
title: pyenvビルドエラーを修復
authors: Z. Yuan
image: /ja/img/2024/1125.webp
tags: [pyenv, python]
description: ビルドエラーの解消
---

pyenv のインストール自体は問題ありません。

しかし、Python のバージョンをビルドする際にエラーが発生しました。

<!-- truncate -->

## 問題の説明

以下のコマンドを実行：

```shell
pyenv install 3.10.15
```

しばらくすると、システムが一連のエラーメッセージを出力：

```shell
Downloading Python-3.10.15.tar.xz...
-> https://www.python.org/ftp/python/3.10.15/Python-3.10.15.tar.xz
Installing Python-3.10.15...

BUILD FAILED (Ubuntu 22.04 using python-build 20180424)

Inspect or clean up the working tree at /tmp/python-build.20241125102533.16978
Results logged to /tmp/python-build.20241125102533.16978.log

Last 10 log lines:
        ./signal/../sysdeps/unix/sysv/linux/x86_64/libc_sigaction.c:0
0x7be2d6829d8f __libc_start_call_main
        ../sysdeps/nptl/libc_start_call_main.h:58
0x7be2d6829e3f __libc_start_main_impl
        ../csu/libc-start.c:392
Please submit a full bug report,
with preprocessed source if appropriate.
Please include the complete backtrace with any bug report.
See <file:///usr/share/doc/gcc-11/README.Bugs> for instructions.
make: *** [Makefile:1856: Python/getargs.o] Error 1
```

え？開発者の日常って感じですね！

## 問題の解決

以下の方法を試してみます：

### 1. 必要な依存パッケージをインストール

まず、システムパッケージを更新：

```shell
sudo apt update && sudo apt upgrade -y
```

次に、関連パッケージがインストールされているか確認：

```shell
sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

システムが、これらのパッケージはすでに最新であると教えてくれました。

つまり、この部分に問題はありません。

### 2. gcc バージョンを確認

gcc のバージョンを更新：

```shell
sudo apt install --reinstall gcc
```

再度試してみましたが、同じエラーが発生。

これも原因ではないようです。

### 3. ログファイルを確認

エラーメッセージからログファイルの場所を確認し、内容を確認：

```shell
less /tmp/python-build.20241125102533.16978.log
```

ログには大量のメッセージがありましたが、最後の部分を直接確認します：

```shell
gcc -c -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall    -std=c99 -Wextra -Wno-unused-result -Wno-unused-parameter -Wno-missing-field-initializers -Werror=implicit-function-declaration -fvisibility=hidden  -I./Include/internal  -I. -I./Include -I/home/user/.pyenv/versions/3.10.15/include -I/home/user/.pyenv/versions/3.10.15/include -fPIC -DPy_BUILD_CORE -o Programs/_testembed.o ./Programs/_testembed.c
sed -e "s,@EXENAME@,/home/user/.pyenv/versions/3.10.15/bin/python3.10," < ./Misc/python-config.in >python-config.py
LC_ALL=C sed -e 's,\$(\([A-Za-z0-9_]*\)),\$\{\1\},g' < Misc/python-config.sh >python-config
/tmp/ccTVJtRi.s: Assembler messages:
/tmp/ccTVJtRi.s: Internal error in emit_inc_line_addr at ../../gas/dwarf2dbg.c:1643.
Please report this bug.
make: *** [Makefile:1856: Objects/typeobject.o] Error 1
make: *** Waiting for unfinished jobs....
```

エラーを発見しました！ここにはこう書かれています：

- **"Internal error in emit_inc_line_addr at ../../gas/dwarf2dbg.c"**

これは、binutils のアセンブラコンポーネントに問題があることを示しており、通常、パッケージ内のバグや非互換性が原因です。

### 4. binutils を再インストール

問題が特定できたので、binutils を更新します：

```shell
sudo apt update
sudo apt install --reinstall binutils
```

更新後、再度以下のコマンドを実行：

```shell
pyenv install 3.10.15
```

今回は成功しました！これで問題解決です。
