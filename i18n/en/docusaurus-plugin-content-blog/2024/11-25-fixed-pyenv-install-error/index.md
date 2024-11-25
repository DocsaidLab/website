---
slug: fixed-pyenv-install-error
title: Fixing pyenv Build Errors
authors: Zephyr
image: /en/img/2024/1125.webp
tags: [pyenv, python]
description: Troubleshooting build errors
---

Installing pyenv itself is fine.

However, an error occurs when building the Python version.

<!-- truncate -->

## Problem Description

Using the command:

```shell
pyenv install 3.10.15
```

Shortly after, the system throws a series of error messages:

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

Huh? Just another day in the life of a developer!

## Fixing the Problem

Here are a few potential fixes:

### 1. Install Dependencies

First, update the system packages:

```shell
sudo apt update && sudo apt upgrade -y
```

Then, make sure the necessary packages are installed:

```shell
sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

The system tells us that all the above packages are already up-to-date.

So, it’s not the problem here.

### 2. Check GCC Version

Update the GCC version:

```shell
sudo apt install --reinstall gcc
```

Then try again, but the same error persists.

Looks like this isn’t the issue either.

### 3. Check the Log File

From the error messages, find the location of the log file and take a look:

```shell
less /tmp/python-build.20241125102533.16978.log
```

Inside, there’s a ton of information. Let’s go straight to the last section:

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

The error here says:

- **"Internal error in emit_inc_line_addr at ../../gas/dwarf2dbg.c"**

This points to an issue with the assembler component of binutils, typically indicating a bug or incompatibility in the package.

### 4. Reinstall Binutils

Great, now we know the problem. Let’s update binutils:

```shell
sudo apt update
sudo apt install --reinstall binutils
```

After updating, try running:

```shell
pyenv install 3.10.15
```

This time, it works, and the problem is resolved.
