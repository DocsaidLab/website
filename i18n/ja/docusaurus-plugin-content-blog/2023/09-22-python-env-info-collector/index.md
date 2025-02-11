---
slug: python-env-info-collector
title: モデルトレーニング環境の問題記録とトラブルシューティング
authors: Z. Yuan
tags: [python, training-log]
image: /ja/img/2023/0922.webp
description: 自作の記録ツールを共有します。
---

モデルのトレーニングが失敗した場合、その原因を突き止めたいと思うのは自然なことです。その際、トレーニング環境の情報を確認する必要があります。例えば、Python バージョン、PyTorch バージョン、CUDA バージョン、GPU 情報、CPU 情報、RAM 情報、ディスク情報、IP アドレスなどです。

ここでは、自作の Python ツールを共有します。このツールを使えば、これらの情報を簡単に確認できます。全てを網羅しているわけではありませんが、基本的なトラブルシューティングには十分です。

通常、トレーニングを開始する段階で、環境情報をトレーニングログに記録します。

<!-- truncate -->

## インストール

まずは必要なパッケージをインストールします：

```bash
pip install psutil requests
```

:::tip
完全なコードは GitHub 上に公開しています。本記事の最後にもコードを掲載しています。

- [**system_info.py**](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/system_info.py)
  :::

## `get_package_versions` の使用

`capybara` をインストール済みでプロジェクトに含まれている場合、以下のコードでテストできます：

```python
from capybara import get_package_versions

get_package_versions()
```

実行結果：

```json
{
  "PyTorch Version": "2.1.1+cu121",
  "PyTorch Lightning Version": "2.1.2",
  "TensorFlow Error": "No module named 'tensorflow'",
  "Keras Error": "No module named 'keras'",
  "NumPy Version": "1.24.4",
  "Pandas Version": "2.0.3",
  "Scikit-learn Version": "1.3.2",
  "OpenCV Version": "4.8.1"
}
```

- PyTorch Version: PyTorch のバージョン
- PyTorch Lightning Version: PyTorch Lightning のバージョン
- TensorFlow Error: TensorFlow のバージョン
- Keras Error: Keras のバージョン
- NumPy Version: NumPy のバージョン
- Pandas Version: Pandas のバージョン
- Scikit-learn Version: Scikit-learn のバージョン
- OpenCV Version: OpenCV のバージョン

## `get_gpu_cuda_versions` の使用

テストコード：

```python
from capybara import get_gpu_cuda_versions

get_gpu_cuda_versions()
```

実行結果：

```json
{
  "CUDA Version": "12.1",
  "NVIDIA Driver Version": "535.129.03"
}
```

- CUDA Version: CUDA のバージョン
- NVIDIA Driver Version: NVIDIA ドライバのバージョン

## `get_system_info` の使用

テストコード：

```python
from capybara import get_system_info

get_system_info()
```

実行結果：

```json
{
  "OS Version": "Linux-6.2.0-37-generic-x86_64-with-glibc2.34",
  "CPU Model": "13th Gen Intel(R) Core(TM) i9-13900K",
  "Physical CPU Cores": 24,
  "Logical CPU Cores (incl. hyper-threading)": 32,
  "Total RAM (GB)": 125.56,
  "Available RAM (GB)": 110.9,
  "Disk Total (GB)": 1832.21,
  "Disk Used (GB)": 188.94,
  "Disk Free (GB)": 1550.12,
  "GPU Info": "NVIDIA GeForce RTX 4090",
  "IPV4 Address": ["192.168.---.---"],
  "IPV4 Address (External)": "---.---.---.---",
  "MAC Address": ["--.--.--.--.--.--"]
}
```

- OS Version: OS のバージョン
- CPU Model: CPU モデル
- Physical CPU Cores: 物理 CPU コア数
- Logical CPU Cores (incl. hyper-threading): 論理 CPU コア数（ハイパースレッディングを含む）
- Total RAM (GB): 総 RAM 容量 (GB)
- Available RAM (GB): 利用可能な RAM 容量 (GB)
- Disk Total (GB): ディスク総容量 (GB)
- Disk Used (GB): 使用中のディスク容量 (GB)
- Disk Free (GB): 空きディスク容量 (GB)
- GPU Info: GPU 情報
- IPV4 Address: 内部 IPV4 アドレス
- IPV4 Address (External): 外部 IPV4 アドレス
- MAC Address: MAC アドレス

## 注意事項と代替案

このツールは Ubuntu で作成されたため、他の OS では予期せぬ挙動が発生する可能性があります。

以下の点に注意してください：

- 一部の関数は OS に依存しており、すべてのプラットフォームで動作するとは限りません。例えば、`get_cpu_info` は Windows では完全な CPU モデルを表示しません。この場合、他のツールを使うか手動で情報を取得することを検討してください。
- Windows 環境で `nvidia-smi` を直接使用して GPU 情報を取得できない場合、NVIDIA ドライバと関連ツールがインストールされていることを確認し、コマンドプロンプトで実行してください。
- 外部 IP アドレスは `https://httpbin.org/ip` から取得しているため、ネットワーク接続がアクティブである必要があります。

## コード

```python showLineNumbers
import platform
import socket
import subprocess

import psutil
import requests


def get_package_versions():
    """
    Get versions of commonly used packages in deep learning and data science.

    Returns:
        dict: Dictionary containing versions of installed packages.
    """
    versions_info = {}

    # PyTorch
    try:
        import torch
        versions_info["PyTorch Version"] = torch.__version__
    except Exception as e:
        versions_info["PyTorch Error"] = str(e)

    # PyTorch Lightning
    try:
        import pytorch_lightning as pl
        versions_info["PyTorch Lightning Version"] = pl.__version__
    except Exception as e:
        versions_info["PyTorch Lightning Error"] = str(e)

    # TensorFlow
    try:
        import tensorflow as tf
        versions_info["TensorFlow Version"] = tf.__version__
    except Exception as e:
        versions_info["TensorFlow Error"] = str(e)

    # Keras
    try:
        import keras
        versions_info["Keras Version"] = keras.__version__
    except Exception as e:
        versions_info["Keras Error"] = str(e)

    # NumPy
    try:
        import numpy as np
        versions_info["NumPy Version"] = np.__version__
    except Exception as e:
        versions_info["NumPy Error"] = str(e)

    # Pandas
    try:
        import pandas as pd
        versions_info["Pandas Version"] = pd.__version__
    except Exception as e:
        versions_info["Pandas Error"] = str(e)

    # Scikit-learn
    try:
        import sklearn
        versions_info["Scikit-learn Version"] = sklearn.__version__
    except Exception as e:
        versions_info["Scikit-learn Error"] = str(e)

    # OpenCV
    try:
        import cv2
        versions_info["OpenCV Version"] = cv2.__version__
    except Exception as e:
        versions_info["OpenCV Error"] = str(e)

    # ... and so on for any other packages you"re interested in

    return versions_info


def get_gpu_cuda_versions():
    """
    Get GPU and CUDA versions using popular Python libraries.

    Returns:
        dict: Dictionary containing CUDA and GPU driver versions.
    """

    cuda_version = None

    # Attempt to retrieve CUDA version using PyTorch
    try:
        import torch
        cuda_version = torch.version.cuda
    except ImportError:
        pass

    # If not retrieved via PyTorch, try using TensorFlow
    if not cuda_version:
        try:
            import tensorflow as tf
            cuda_version = tf.version.COMPILER_VERSION
        except ImportError:
            pass

    # If still not retrieved, try using CuPy
    if not cuda_version:
        try:
            import cupy
            cuda_version = cupy.cuda.runtime.runtimeGetVersion()
        except ImportError:
            cuda_version = "Error: None of PyTorch, TensorFlow, or CuPy are installed."

    # Try to get Nvidia driver version using nvidia-smi command
    try:
        smi_output = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits"
        ]).decode("utf-8").strip()
        nvidia_driver_version = smi_output.split("\n")[0]
    except Exception as e:
        nvidia_driver_version = f"Error getting NVIDIA driver version: {e}"

    return {
        "CUDA Version": cuda_version,
        "NVIDIA Driver Version": nvidia_driver_version
    }


def get_cpu_info():
    """
    Retrieve the CPU model name based on the platform.

    Returns:
        str: CPU model name or "N/A" if not found.
    """
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        # For macOS
        command = "sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command, shell=True).strip().decode()
    elif platform.system() == "Linux":
        # For Linux
        command = "cat /proc/cpuinfo | grep "model name" | uniq"
        return subprocess.check_output(command, shell=True).strip().decode().split(":")[1].strip()
    else:
        return "N/A"


def get_external_ip():
    try:
        response = requests.get("https://httpbin.org/ip")
        return response.json()["origin"]
    except Exception as e:
        return f"Error obtaining IP: {e}"


def get_system_info():
    """
    Fetch system information like OS version, CPU info, RAM, Disk usage, etc.

    Returns:
        dict: Dictionary containing system information.
    """
    info = {
        "OS Version": platform.platform(),
        "CPU Model": get_cpu_info(),
        "Physical CPU Cores": psutil.cpu_count(logical=False),
        "Logical CPU Cores (incl. hyper-threading)": psutil.cpu_count(logical=True),
        "Total RAM (GB)": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "Available RAM (GB)": round(psutil.virtual_memory().available / (1024 ** 3), 2),
        "Disk Total (GB)": round(psutil.disk_usage("/").total / (1024 ** 3), 2),
        "Disk Used (GB)": round(psutil.disk_usage("/").used / (1024 ** 3), 2),
        "Disk Free (GB)": round(psutil.disk_usage("/").free / (1024 ** 3), 2)
    }

    # Try to fetch GPU information using nvidia-smi command
    try:
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"]
        ).decode("utf-8").strip()
        info["GPU Info"] = gpu_info
    except Exception:
        info["GPU Info"] = "N/A or Error"

    # Get network information
    addrs = psutil.net_if_addrs()
    info["IPV4 Address"] = [
        addr.address for addr in addrs.get("enp5s0", []) if addr.family == socket.AF_INET
    ]

    info["IPV4 Address (External)"] = get_external_ip()

    # Determine platform and choose correct address family for MAC
    if hasattr(socket, "AF_LINK"):
        AF_LINK = socket.AF_LINK
    elif hasattr(psutil, "AF_LINK"):
        AF_LINK = psutil.AF_LINK
    else:
        raise Exception(
            "Cannot determine the correct AF_LINK value for this platform.")

    info["MAC Address"] = [
        addr.address for addr in addrs.get("enp5s0", []) if addr.family == AF_LINK
    ]

    return info
```
