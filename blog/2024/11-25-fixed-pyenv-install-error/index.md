---
slug: fixed-pyenv-install-error
title: 修復 pyenv 建置錯誤
authors: Z. Yuan
image: /img/2024/1125.webp
tags: [pyenv, python]
description: 排除建置錯誤
---

安裝 pyenv 本身是沒問題的。

但是在建置 python 版本時出現錯誤。

<!-- truncate -->

## 問題描述

使用指令：

```shell
pyenv install 3.10.15
```

沒多久，系統噴出一串錯誤訊息：

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

嗯？果然是開發者的日常啊！

## 解決問題

這裡嘗試幾個解決方法：

### 1. 安裝相依套件

先更新一下系統套件：

```shell
sudo apt update && sudo apt upgrade -y
```

然後確認一下相關套件是否安裝：

```shell
sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

系統告訴我們，上面這些套件都已經是最新的了。

所以，不是這裡的問題。

### 2. 檢查 gcc 版本

更新一下 gcc 的版本：

```shell
sudo apt install --reinstall gcc
```

然後再試一次，得到同樣的錯誤。

看來也不是這個問題。

### 3. 檢查一下 log 檔

從剛才的錯誤訊息中找到 log 檔的位置，然後進去看一下：

```shell
less /tmp/python-build.20241125102533.16978.log
```

這裡面有一大堆的訊息，我們直接查看最後一段：

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

發現錯誤了！這裡寫的是：

- **"Internal error in emit_inc_line_addr at ../../gas/dwarf2dbg.c"**

指的是 binutils 的組譯器組件出現問題，通常表示套件中存在錯誤或不相容性。

### 4. 重裝 binutils

很好，知道錯誤了，那就更新一下 binutils：

```shell
sudo apt update
sudo apt install --reinstall binutils
```

更新後，再次執行：

```shell
pyenv install 3.10.15
```

這次就成功了，結束這個問題。
