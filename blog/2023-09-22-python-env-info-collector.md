---
slug: python-env-info-collector
title: 打造 Python 環境偵測工具：快速記錄與排查模型訓練環境問題
authors: Zephyr
tags: [python, training, environment, info, log]
---

當模型 train 壞了，我們總是會想知道是什麼原因導致的，這時候我們需要檢查訓練主機的環境資訊，例如：Python 版本、PyTorch 版本、CUDA 版本、GPU 資訊、CPU 資訊、RAM 資訊、磁碟資訊、IP 地址等等。

我們在本文中分享一個自己手刻的 Python 小工具，可以快速查看這些資訊，雖然說不是包山包海，但基本的問題排查應該足夠用了。

一般來說，我們會在訓練啟動的環節，將環境資訊紀錄到訓練主機的日誌裡面。

<!--truncate-->

## 安裝

我們先安裝必要套件：

```bash
pip install psutil requests
```

接著您可以在本文的最後取得完整的程式碼，並且將它放在您的專案裡面。

## 使用 `get_package_versions`

我們假設您有安裝 docsaidkit，並且已經在專案裡面，則可以透過以下指令測試：

```python
from docsaidkit import get_package_versions

get_package_versions()
```

執行後得到結果：

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

- PyTorch Version: PyTorch 版本
- PyTorch Lightning Version: PyTorch Lightning 版本
- TensorFlow Error: TensorFlow 版本
- Keras Error: Keras 版本
- NumPy Version: NumPy 版本
- Pandas Version: Pandas 版本
- Scikit-learn Version: Scikit-learn 版本
- OpenCV Version: OpenCV 版本

## 使用 `get_gpu_cuda_versions`

測試程式：

```python
from docsaidkit import get_gpu_cuda_versions

get_gpu_cuda_versions()
```

執行後得到結果：

```json
{
    "CUDA Version": "12.1",
    "NVIDIA Driver Version": "535.129.03"
}
```

- CUDA Version: CUDA 版本
- NVIDIA Driver Version: NVIDIA 驅動版本

## 使用 `get_system_info`

測試程式：

```python
from docsaidkit import get_system_info

get_system_info()
```

執行後得到結果：

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

- OS Version: 操作系統版本
- CPU Model: CPU 型號
- Physical CPU Cores: 物理 CPU 核心數
- Logical CPU Cores (incl. hyper-threading): 邏輯 CPU 核心數 (包括超執行緒)
- Total RAM (GB): 總 RAM 容量 (GB)
- Available RAM (GB): 可用 RAM 容量 (GB)
- Disk Total (GB): 磁碟總容量 (GB)
- Disk Used (GB): 已使用的磁碟容量 (GB)
- Disk Free (GB): 空閒磁碟容量 (GB)
- GPU Info: GPU 資訊
- IPV4 Address: 內部 IPV4 地址
- IPV4 Address (External): 外部 IPV4 地址
- MAC Address: MAC 地址

## 注意事項與替代方案

由於我們是在 Ubuntu 上撰寫本函數，因此在其他作業系統上可能會有劇情之外的發展。

以下幾個可能需要注意的要點：

- 因操作系統的限制，某些函數可能無法在所有平台上運行。例如：`get_cpu_info` 在 Windows 上不會顯示完整的 CPU 型號。在這種情況下，您可以考慮使用其他工具或手動獲取此資訊。
- 如果您在 Windows 環境中，無法直接使用 `nvidia-smi` 來獲取 GPU 資訊，請確保已安裝 NVIDIA 驅動和相關的工具，並在命令提示符窗口中執行它。
- 外部 IP 地址是從 `https://httpbin.org/ip` 獲取的，所以必須確保網路連線是活躍的。



## 程式碼

我們期待這份程式碼可以幫助您快速檢視訓練環境，您可以將此資訊輸出後與模型的訓練日誌一起保存，以確保訓練的可追溯性和可重現性。

- [system_info.py](https://github.com/DocsaidLab/DocsaidKit/blob/main/docsaidkit/utils/system_info.py)

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
